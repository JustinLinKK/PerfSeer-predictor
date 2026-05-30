"""Training harness for the PerfSeer SeerNet single-metric predictor.

Protocol (paper section 4.1):
  * split 2:1:1 train/val/test, random with a fixed seed
  * batch size 128, Adam optimizer, lr 1e-3
  * ReduceLROnPlateau: factor 0.5, patience 5 epochs, min_lr 1e-6
  * up to 500 epochs, MSE loss in the standardized log space
  * early stopping on validation loss (default patience ~30 epochs)
  * best-validation checkpoint saved; CPU device

Run examples:
  # train a single metric (index 2 = train_time)
  python -m perfseer.train --metric 2 --data-root dataset --out runs

  # train all six single-metric models in a loop
  python -m perfseer.train --metric all --data-root dataset --out runs

This module depends ONLY on the documented data.py / model.py contract:
  data.py  -> PerfSeerDataset, split_dataset, compute_norm_stats,
              NODE_DIM, EDGE_DIM, GLOBAL_DIM, NUM_TARGETS
  model.py -> SeerNet
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyG DataLoader handles graph batching (collates x/edge_index/edge_attr/u/y
# and produces the `batch` vector for node->graph scatter ops).
from torch_geometric.loader import DataLoader

from perfseer.data import (
    EDGE_DIM,
    GLOBAL_DIM,
    NODE_DIM,
    NUM_TARGETS,
    PerfSeerDataset,
    compute_norm_stats,
    split_dataset,
)
from perfseer.model import SeerNet

# Human-readable names for the 6 target metrics (order fixed by the contract).
METRIC_NAMES: List[str] = [
    "train_util",  # 0: train.average_sm_util
    "train_mem",   # 1: train.peak_memory_usuage
    "train_time",  # 2: train.time
    "infer_util",  # 3: infer.average_sm_util
    "infer_mem",   # 4: infer.peak_memory_usuage
    "infer_time",  # 5: infer.time
]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs for deterministic CPU training."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only: no torch.cuda seeding needed.


# --------------------------------------------------------------------------- #
# Target-normalisation extraction
# --------------------------------------------------------------------------- #
def extract_target_stats(norm_stats: Dict, metric_idx: int) -> Dict[str, float]:
    """Pull the log-space mean/std of one target out of the norm_stats dict.

    The data.py contract stores per-target standardisation statistics so that
    predictions (made in standardized log1p space) can be inverted back to the
    original metric space at evaluation time. We persist exactly the slice this
    model needs inside its checkpoint.

    Expected layout (any of these key conventions is accepted):
      norm_stats['target']['mean'] / ['std']  -> arrays of length NUM_TARGETS
      norm_stats['y_mean'] / norm_stats['y_std'] -> arrays of length NUM_TARGETS
    Values are interpreted as the mean/std of log1p(y) over the train split.
    """
    mean_arr = None
    std_arr = None

    if "target" in norm_stats and isinstance(norm_stats["target"], dict):
        mean_arr = norm_stats["target"].get("mean")
        std_arr = norm_stats["target"].get("std")
    if mean_arr is None and "y_mean" in norm_stats:
        mean_arr = norm_stats["y_mean"]
    if std_arr is None and "y_std" in norm_stats:
        std_arr = norm_stats["y_std"]

    if mean_arr is None or std_arr is None:
        raise KeyError(
            "norm_stats must expose target log-space mean/std under "
            "norm_stats['target']['mean'/'std'] or norm_stats['y_mean'/'y_std']."
        )

    mean_arr = np.asarray(mean_arr, dtype=np.float64).reshape(-1)
    std_arr = np.asarray(std_arr, dtype=np.float64).reshape(-1)
    return {
        "log_mean": float(mean_arr[metric_idx]),
        "log_std": float(std_arr[metric_idx]),
    }


# --------------------------------------------------------------------------- #
# One full training run for a single metric
# --------------------------------------------------------------------------- #
def train_one_metric(
    metric_idx: int,
    train_ds,
    val_ds,
    norm_stats: Dict,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    """Train a single-output SeerNet for ``metric_idx``; return checkpoint path.

    The dataset yields ``data.y`` of shape [1, NUM_TARGETS] already in
    standardized log space. We select column ``metric_idx`` as the target and
    optimise MSE in that space.
    """
    metric_name = METRIC_NAMES[metric_idx]
    print(f"\n=== Training metric [{metric_idx}] {metric_name} ===", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = SeerNet(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        num_outputs=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    target_stats = extract_target_stats(norm_stats, metric_idx)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"seernet_metric{metric_idx}_{metric_name}.pt")

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    def select_target(batch) -> torch.Tensor:
        """Reshape batched y to [B, NUM_TARGETS] and slice the target column."""
        y = batch.y.view(-1, NUM_TARGETS)
        return y[:, metric_idx : metric_idx + 1]  # keep [B, 1]

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)  # [B, 1] in std-log space
            target = select_target(batch)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            bs = target.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs
        train_loss = train_loss_sum / max(train_n, 1)

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = select_target(batch)
                loss = criterion(pred, target)
                bs = target.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
        val_loss = val_loss_sum / max(val_n, 1)

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": cur_lr}
        )
        print(
            f"[{metric_name}] epoch {epoch:3d} | train {train_loss:.6f} "
            f"| val {val_loss:.6f} | lr {cur_lr:.2e}",
            flush=True,
        )

        # ---- checkpoint best ----
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "metric_idx": metric_idx,
                    "metric_name": metric_name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "target_stats": target_stats,  # log_mean / log_std for inversion
                    "model_config": {
                        "node_dim": NODE_DIM,
                        "edge_dim": EDGE_DIM,
                        "global_dim": GLOBAL_DIM,
                        "hidden": args.hidden,
                        "num_blocks": args.num_blocks,
                        "num_outputs": 1,
                    },
                    "num_targets": NUM_TARGETS,
                    "seed": args.seed,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        # ---- early stopping ----
        if epochs_no_improve >= args.patience:
            print(
                f"[{metric_name}] early stop at epoch {epoch} "
                f"(no val improvement for {args.patience} epochs; "
                f"best epoch {best_epoch}, best val {best_val:.6f})",
                flush=True,
            )
            break

    # persist the validation curve next to the checkpoint
    curve_path = os.path.join(args.out, f"seernet_metric{metric_idx}_{metric_name}.curve.json")
    with open(curve_path, "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val": best_val, "history": history}, f, indent=2)

    print(
        f"[{metric_name}] done. best val {best_val:.6f} @ epoch {best_epoch} "
        f"-> {ckpt_path}",
        flush=True,
    )
    return ckpt_path


# --------------------------------------------------------------------------- #
# Dataset construction (shared by all metrics in a single run)
# --------------------------------------------------------------------------- #
def build_datasets(args: argparse.Namespace):
    """Split files 2:1:1, compute train-only norm stats, build PyG datasets.

    Returns (train_ds, val_ds, test_ds, norm_stats).
    """
    train_files, val_files, test_files = split_dataset(args.data_root, seed=args.seed)
    # Optional truncation for smoke tests / data-dependency study (paper Table 6).
    limit = getattr(args, "limit", 0)
    if limit and limit > 0:
        train_files = train_files[:limit]
        val_files = val_files[: max(1, limit // 2)]
        test_files = test_files[: max(1, limit // 2)]
    print(
        f"split: {len(train_files)} train / {len(val_files)} val / "
        f"{len(test_files)} test",
        flush=True,
    )

    # Normalisation statistics are computed on the TRAIN split only.
    norm_stats = compute_norm_stats(train_files)

    # PerfSeerDataset processes/caches graphs under dataset/processed/.
    common = dict(root=args.data_root, norm_stats=norm_stats)
    train_ds = PerfSeerDataset(file_list=train_files, split="train", **common)
    val_ds = PerfSeerDataset(file_list=val_files, split="val", **common)
    test_ds = PerfSeerDataset(file_list=test_files, split="test", **common)
    return train_ds, val_ds, test_ds, norm_stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PerfSeer SeerNet (single-metric).")
    p.add_argument(
        "--metric",
        default="all",
        help="target metric index 0..5, or 'all' to loop over the 6 single-metric models",
    )
    p.add_argument("--epochs", type=int, default=500, help="max epochs (paper: up to 500)")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-root", default="dataset", dest="data_root")
    p.add_argument("--out", default="runs", help="output dir for checkpoints / curves")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    p.add_argument("--hidden", type=int, default=256, help="SeerBlock/MLP hidden width")
    p.add_argument("--num-blocks", type=int, default=1, dest="num_blocks")
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    p.add_argument("--limit", type=int, default=0, help="cap #train files (0=all); for smoke tests / Table 6")
    p.add_argument("--threads", type=int, default=0, help="torch intra-op threads per process (0=torch default)")
    return p.parse_args(argv)


def resolve_metrics(metric_arg: str) -> List[int]:
    """Map the --metric argument to a list of target indices to train."""
    if str(metric_arg).lower() == "all":
        return list(range(NUM_TARGETS))
    idx = int(metric_arg)
    if not (0 <= idx < NUM_TARGETS):
        raise ValueError(f"--metric must be in [0, {NUM_TARGETS - 1}] or 'all', got {metric_arg}")
    return [idx]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    # Cap intra-op threads per process so several metric models can run in
    # parallel (each `--metric i`) without oversubscribing the 32 CPU cores.
    if getattr(args, "threads", 0) and args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {device} ({dev_name}) | torch threads: {torch.get_num_threads()}", flush=True)

    metric_indices = resolve_metrics(args.metric)

    # Build datasets ONCE; reused across every per-metric model so the same
    # split / normalisation is applied consistently.
    train_ds, val_ds, _test_ds, norm_stats = build_datasets(args)

    ckpts: List[str] = []
    t0 = time.time()
    for midx in metric_indices:
        ckpts.append(train_one_metric(midx, train_ds, val_ds, norm_stats, args, device))
    print(f"\nall done in {time.time() - t0:.1f}s. checkpoints:", flush=True)
    for c in ckpts:
        print(f"  {c}", flush=True)


if __name__ == "__main__":
    main()
