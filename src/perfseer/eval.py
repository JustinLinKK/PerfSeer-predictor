"""Evaluation harness for PerfSeer SeerNet checkpoints.

Loads one or more single-metric checkpoints produced by train.py, runs them on
the TEST split, inverts predictions/targets from standardized-log space back to
the ORIGINAL metric space, and prints a per-metric table of
MAPE / RMSPE / 5%Acc / 10%Acc plus the 6-metric mean MAPE.

Run examples:
  # evaluate every checkpoint in a run directory
  python -m perfseer.eval --ckpt-dir runs --data-root dataset

  # evaluate explicit checkpoint files
  python -m perfseer.eval --ckpt runs/seernet_metric2_train_time.pt --data-root dataset

Inversion (must match the data.py target normalisation):
  the network predicts  s = (log1p(y) - log_mean) / log_std  in std-log space.
  original value        y = expm1(s * log_std + log_mean).
The per-target log_mean / log_std are stored inside each checkpoint
(``target_stats``) by train.py, so evaluation never needs to recompute stats.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
from perfseer.metrics import all_metrics
from perfseer.model import SeerNet
from perfseer.train import METRIC_NAMES


# --------------------------------------------------------------------------- #
# Inversion: standardized-log space -> original metric space
# --------------------------------------------------------------------------- #
def invert(std_log: np.ndarray, log_mean: float, log_std: float) -> np.ndarray:
    """Undo z-score-in-log1p-space: y = expm1(s * log_std + log_mean)."""
    log_y = std_log * log_std + log_mean
    return np.expm1(log_y)


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str, device: torch.device) -> Tuple[SeerNet, int, Dict[str, float]]:
    """Rebuild a SeerNet from a checkpoint; return (model, metric_idx, target_stats)."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["model_config"]
    model = SeerNet(
        node_dim=cfg.get("node_dim", NODE_DIM),
        edge_dim=cfg.get("edge_dim", EDGE_DIM),
        global_dim=cfg.get("global_dim", GLOBAL_DIM),
        hidden=cfg.get("hidden", 256),
        num_blocks=cfg.get("num_blocks", 1),
        num_outputs=cfg.get("num_outputs", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, int(ckpt["metric_idx"]), ckpt["target_stats"]


def discover_checkpoints(args: argparse.Namespace) -> List[str]:
    """Resolve the list of checkpoint files from --ckpt / --ckpt-dir."""
    if args.ckpt:
        return list(args.ckpt)
    if args.ckpt_dir:
        found = sorted(glob.glob(os.path.join(args.ckpt_dir, "seernet_metric*.pt")))
        if not found:
            raise FileNotFoundError(f"no seernet_metric*.pt found under {args.ckpt_dir}")
        return found
    raise ValueError("provide --ckpt <file...> or --ckpt-dir <dir>")


# --------------------------------------------------------------------------- #
# Per-checkpoint evaluation on the test split
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(
    model: SeerNet,
    metric_idx: int,
    target_stats: Dict[str, float],
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run one model over the test set; return metrics in original space."""
    preds_std: List[np.ndarray] = []
    targets_std: List[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)  # [B, 1] std-log space
            y = batch.y.view(-1, NUM_TARGETS)[:, metric_idx : metric_idx + 1]
            preds_std.append(pred.cpu().numpy().reshape(-1))
            targets_std.append(y.cpu().numpy().reshape(-1))

    pred_std = np.concatenate(preds_std) if preds_std else np.zeros(0)
    targ_std = np.concatenate(targets_std) if targets_std else np.zeros(0)

    log_mean = target_stats["log_mean"]
    log_std = target_stats["log_std"]
    y_pred = invert(pred_std, log_mean, log_std)
    y_true = invert(targ_std, log_mean, log_std)
    return all_metrics(y_true, y_pred)


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_table(rows: Dict[int, Dict[str, float]]) -> None:
    """Print a per-metric results table and the 6-metric mean MAPE."""
    header = f"{'metric':<12} {'MAPE%':>9} {'RMSPE%':>9} {'5%Acc':>8} {'10%Acc':>8}"
    print("\n" + header)
    print("-" * len(header))
    mapes: List[float] = []
    for idx in sorted(rows):
        m = rows[idx]
        name = METRIC_NAMES[idx]
        print(
            f"{name:<12} {m['MAPE']:>9.3f} {m['RMSPE']:>9.3f} "
            f"{m['5Acc'] * 100:>7.2f}% {m['10Acc'] * 100:>7.2f}%"
        )
        if not np.isnan(m["MAPE"]):
            mapes.append(m["MAPE"])
    print("-" * len(header))
    if mapes:
        print(f"{'MEAN MAPE':<12} {np.mean(mapes):>9.3f}   (over {len(mapes)} metrics)")
    print(f"(paper target: mean MAPE 5.14% across 6 metrics)\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PerfSeer SeerNet checkpoints.")
    p.add_argument("--ckpt", nargs="*", help="explicit checkpoint file(s)")
    p.add_argument("--ckpt-dir", dest="ckpt_dir", help="directory of seernet_metric*.pt files")
    p.add_argument("--data-root", default="dataset", dest="data_root")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--seed", type=int, default=42, help="MUST match the training seed for an identical split")
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    p.add_argument("--limit", type=int, default=0, help="cap #train files (0=all); MUST match the train --limit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = discover_checkpoints(args)

    # Rebuild the SAME split + normalisation used at training time (seed must match).
    train_files, _val_files, test_files = split_dataset(args.data_root, seed=args.seed)
    # Mirror train.py truncation so norm_stats + test split match a --limit run.
    if getattr(args, "limit", 0) and args.limit > 0:
        train_files = train_files[: args.limit]
        test_files = test_files[: max(1, args.limit // 2)]
    norm_stats = compute_norm_stats(train_files)
    test_ds = PerfSeerDataset(
        file_list=test_files, split="test", root=args.data_root, norm_stats=norm_stats
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"test graphs: {len(test_ds)}", flush=True)

    rows: Dict[int, Dict[str, float]] = {}
    for path in ckpt_paths:
        model, metric_idx, target_stats = load_checkpoint(path, device)
        print(f"eval {os.path.basename(path)} -> metric [{metric_idx}] {METRIC_NAMES[metric_idx]}", flush=True)
        rows[metric_idx] = evaluate_checkpoint(model, metric_idx, target_stats, test_loader, device)

    print_table(rows)


if __name__ == "__main__":
    main()
