"""Train the current separate-head PerfSeer architecture on 12 A10 labels."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import types
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler

from perfseer_v3 import features as current_features
from perfseer_v3.features import (
    NormalizationBlock,
    NormalizationStatsV3,
    apply_normalization,
)

from .model import SeerNetConfig, SeerTrunk, make_mlp


# The prepared cache predates the rename from perfseer_v31.features_core.
legacy_package = types.ModuleType("perfseer_v31")
legacy_package.features_core = current_features
sys.modules.setdefault("perfseer_v31", legacy_package)
sys.modules.setdefault("perfseer_v31.features_core", current_features)


TARGET_NAMES = (
    "train_step_wall_ms",
    "train_step_gpu_ms",
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_avg_vram_mib",
    "train_peak_vram_mib",
    "train_peak_torch_allocated_mib",
    "infer_step_wall_ms",
    "infer_step_gpu_ms",
    "infer_avg_sm_util_percent",
    "infer_avg_vram_mib",
    "infer_peak_vram_mib",
)

METRIC_HEAD_WIDTHS = (3, 1, 3, 2, 1, 2)


class SixMetricTwelveLabelSeerNet(nn.Module):
    """Latest PerfSeer trunk with six semantic metric heads."""

    def __init__(self, cfg: SeerNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = SeerTrunk(cfg)
        head_hidden = cfg.head_hidden or cfg.hidden
        self.heads = nn.ModuleList(
            make_mlp(cfg.hidden, width, head_hidden, 2, cfg.activation, cfg.dropout)
            for width in METRIC_HEAD_WIDTHS
        )

    def forward(self, data) -> torch.Tensor:
        embedding = self.trunk(data)
        return torch.cat([head(embedding) for head in self.heads], dim=-1)


class NodeBudgetBatchSampler(Sampler[list[int]]):
    """Shuffle samples while bounding the total graph nodes per batch."""

    def __init__(self, node_counts: list[int], max_items: int, max_nodes: int,
                 rank: int, world_size: int) -> None:
        self.node_counts = node_counts
        self.max_items = max_items
        self.max_nodes = max_nodes
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self, indices: list[int]):
        batch: list[int] = []
        node_total = 0
        for index in indices:
            nodes = self.node_counts[index]
            if batch and (len(batch) >= self.max_items or node_total + nodes > self.max_nodes):
                yield batch
                batch, node_total = [], 0
            batch.append(index)
            node_total += nodes
        if batch:
            yield batch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(42 + self.epoch)
        indices = torch.randperm(len(self.node_counts), generator=generator).tolist()
        yield from self._batches(indices[self.rank::self.world_size])

    def __len__(self) -> int:
        return sum(1 for _ in self._batches(list(range(self.rank, len(self.node_counts), self.world_size))))


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def load_targets(label_root: Path) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    paths = sorted((label_root / "extracted").glob("**/labels/**/*.jsonl*"))
    paths += sorted((label_root / "raw_source" / "a10_bs").glob("*.jsonl*"))
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok" or tuple(row.get("target_names", ())) != TARGET_NAMES:
                    continue
                point = str(row["profile_point_id"])
                target = torch.tensor([float(row["targets"][name]) for name in TARGET_NAMES])
                previous = values.get(point)
                if previous is not None and not torch.equal(previous, target):
                    raise ValueError(f"conflicting duplicate target: {point}")
                values[point] = target
    if not values:
        raise ValueError("no 12-label A10 targets found")
    return values


def normalization_from_json(value: dict) -> NormalizationStatsV3:
    def block(name: str) -> NormalizationBlock:
        payload = value[name]
        return NormalizationBlock(
            tuple(payload["mean"]), tuple(payload["std"]),
            tuple(payload["clip_low"]), tuple(payload["clip_high"]),
        )

    return NormalizationStatsV3(
        feature_schema_sha256=value["feature_schema_sha256"],
        operator_registry_sha256=value["operator_registry_sha256"],
        layout_sha256=value["layout_sha256"],
        split_name=value["split_name"],
        split_fingerprint=value["split_fingerprint"],
        quantiles=tuple(value["quantiles"]),
        node=block("node"), edge=block("edge"), global_features=block("global_features"),
        normalization_version=value["normalization_version"],
    )


class PreparedTwelveLabelDataset(Dataset):
    def __init__(self, rows: list[dict], targets: dict[str, torch.Tensor], feature_cache: Path,
                 normalization: NormalizationStatsV3, target_mean: torch.Tensor, target_std: torch.Tensor):
        super().__init__()
        self.rows = rows
        self.targets = targets
        self.feature_cache = feature_cache
        self.normalization = normalization
        self.target_mean = target_mean
        self.target_std = target_std
        self.cache: OrderedDict[str, object] = OrderedDict()
        self.node_count_cache: dict[str, int] = {}

    def len(self) -> int:
        return len(self.rows)

    def _feature_path(self, sha256: str) -> Path:
        path = self.feature_cache / f"{sha256}.pt"
        if path.exists():
            return path
        matches = sorted(self.feature_cache.parent.glob(f"rank-*/{sha256}.pt"))
        if not matches:
            raise FileNotFoundError(path)
        return matches[0]

    def node_count(self, sha256: str) -> int:
        count = self.node_count_cache.get(sha256)
        if count is None:
            count = int(torch.load(self._feature_path(sha256), map_location="cpu", weights_only=False).x_cont.size(0))
            self.node_count_cache[sha256] = count
        return count

    def _feature(self, sha256: str):
        cached = self.cache.get(sha256)
        if cached is None:
            cached = apply_normalization(
                torch.load(self._feature_path(sha256), map_location="cpu", weights_only=False),
                self.normalization,
            )
            self.cache[sha256] = cached
            if len(self.cache) > 4096:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sha256)
        return cached

    def get(self, index: int) -> Data:
        row = self.rows[index]
        features = self._feature(row["input_sha256"])
        raw = self.targets[row["sample_id"]]
        target = (torch.log1p(raw) - self.target_mean) / self.target_std
        return Data(
            x=features.x_cont,
            edge_index=features.edge_index,
            edge_attr=features.edge_cont,
            u=features.u_cont,
            y=target.view(1, -1),
            y_raw=raw.view(1, -1),
        )


def context() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             target_mean: torch.Tensor, target_std: torch.Tensor) -> dict:
    model.eval()
    count = 0
    relative_sum = torch.zeros(len(TARGET_NAMES), dtype=torch.float64)
    within5 = torch.zeros(len(TARGET_NAMES), dtype=torch.float64)
    within10 = torch.zeros(len(TARGET_NAMES), dtype=torch.float64)
    mean = target_mean.to(device)
    std = target_std.to(device)
    for batch in loader:
        batch = batch.to(device)
        prediction = torch.expm1(model(batch) * std + mean).clamp_min(0)
        truth = batch.y_raw.view(-1, len(TARGET_NAMES))
        denominator = truth.abs().clamp_min(1e-6)
        denominator[:, 3].clamp_(min=1.0)
        denominator[:, 9].clamp_(min=1.0)
        error = (prediction - truth).abs() / denominator
        relative_sum += error.sum(0).double().cpu()
        within5 += (error <= 0.05).sum(0).double().cpu()
        within10 += (error <= 0.10).sum(0).double().cpu()
        count += truth.size(0)
    return {
        "rows": count,
        "mean_relative_error": dict(zip(TARGET_NAMES, (relative_sum / count).tolist(), strict=True)),
        "within_5pct_accuracy": dict(zip(TARGET_NAMES, (within5 / count).tolist(), strict=True)),
        "within_10pct_accuracy": dict(zip(TARGET_NAMES, (within10 / count).tolist(), strict=True)),
        "gate_passed": bool((within5 / count >= 0.95).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--local-batch", type=int, default=12)
    parser.add_argument("--max-batch-nodes", type=int, default=32_000)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit each split for a fast architecture smoke test.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the final test split when its graph cache is unavailable.")
    args = parser.parse_args()

    rank, world_size, device = context()
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    manifest = read_json(args.prepared / "dataset_manifest.json")
    splits = {name: read_json(args.prepared / meta["path"]) for name, meta in manifest["split_files"].items()}
    if args.limit:
        splits = {name: rows[:args.limit] for name, rows in splits.items()}
    targets = load_targets(args.labels)
    if any(row["sample_id"] not in targets for rows in splits.values() for row in rows):
        raise ValueError("prepared samples missing raw labels")
    train_raw = torch.stack([targets[row["sample_id"]] for row in splits["train"]]).float()
    target_log = torch.log1p(train_raw)
    target_mean, target_std = target_log.mean(0), target_log.std(0).clamp_min(1e-6)
    normalization = normalization_from_json(read_json(args.normalization))
    datasets = {
        name: PreparedTwelveLabelDataset(rows, targets, args.feature_cache, normalization, target_mean, target_std)
        for name, rows in splits.items()
    }
    train_sampler = NodeBudgetBatchSampler(
        [datasets["train"].node_count(row["input_sha256"]) for row in splits["train"]],
        args.local_batch,
        args.max_batch_nodes,
        rank,
        world_size,
    )
    train_loader = DataLoader(datasets["train"], batch_sampler=train_sampler, num_workers=0)
    if rank == 0:
        val_loader = DataLoader(datasets["validation"], batch_size=args.local_batch, shuffle=False, num_workers=0)
        test_loader = DataLoader(datasets["test"], batch_size=args.local_batch, shuffle=False, num_workers=0)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "manifest.json", {
            "architecture": "latest_perfseer_optimized_six_metric_head_extension",
            "metric_heads": "separate", "head_count": len(METRIC_HEAD_WIDTHS),
            "head_widths": METRIC_HEAD_WIDTHS,
            "max_batch_nodes": args.max_batch_nodes,
            "target_names": TARGET_NAMES, "rows": {name: len(rows) for name, rows in splits.items()},
            "source_targets": len(targets), "a10_bs_included": True,
        })
    barrier(world_size)
    config = SeerNetConfig(node_dim=40, edge_dim=14, global_dim=110, hidden=1024, num_blocks=8,
                            head_hidden=1024, num_outputs=len(METRIC_HEAD_WIDTHS), metric_heads="separate",
                            activation="relu", dropout=0.05, encoder_norm="layernorm", block_norm="prenorm",
                            residual="gated", residual_gate_mode="vector_per_stream", global_agg="synmm_plus",
                            attention_pool=True, mlp_z_num_linear_layers=4)
    model = SixMetricTwelveLabelSeerNet(config).to(device)
    model = DistributedDataParallel(model, device_ids=[device.index]) if world_size > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-6)
    best = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = torch.zeros(1, device=device)
        count = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            prediction = model(batch)
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch.y.view(-1, len(TARGET_NAMES)))
            (loss / args.gradient_accumulation).backward()
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * batch.num_graphs
            count += batch.num_graphs
        if world_size > 1:
            dist.all_reduce(loss_sum)
            dist.all_reduce(count)
        barrier(world_size)
        stop = torch.zeros(1, dtype=torch.int64, device=device)
        if rank == 0:
            validation = evaluate(model.module if world_size > 1 else model, val_loader, device, target_mean, target_std)
            score = float(np.mean(list(validation["mean_relative_error"].values())))
            scheduler.step(score)
            record = {
                "epoch": epoch, "train_loss": float((loss_sum / count).item()),
                "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
            }
            atomic_json(args.output / f"epoch-{epoch:04d}.json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"model_state_dict": (model.module if world_size > 1 else model).state_dict(),
                            "model_config": asdict(config), "target_names": TARGET_NAMES,
                            "target_mean": target_mean, "target_std": target_std,
                            "epoch": epoch, "validation": validation}, args.output / "best.pt")
            else:
                stale += 1
            if validation["gate_passed"] or stale >= args.patience:
                stop.fill_(1)
        if world_size > 1:
            dist.broadcast(stop, src=0)
        if stop.item():
            break
    if rank == 0:
        payload = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        (model.module if world_size > 1 else model).load_state_dict(payload["model_state_dict"])
        report = {"best_epoch": best_epoch, "validation": payload["validation"]}
        if args.skip_test:
            report["test"] = {"skipped": "feature cache unavailable for held-out graph inputs"}
        else:
            report["test"] = evaluate(
                model.module if world_size > 1 else model,
                test_loader,
                device,
                target_mean,
                target_std,
            )
        atomic_json(args.output / "final.json", report)
        print(json.dumps({"final": report}, sort_keys=True), flush=True)
    barrier(world_size)


if __name__ == "__main__":
    main()
