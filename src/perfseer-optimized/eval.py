"""Evaluation CLI for optimized PerfSeer checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from .bench import configure_cpu_threads, summarize_latencies
from .calibration import LinearCalibrator
from .data import (
    FeatureConfig,
    LABEL_DOMAIN_VOCAB,
    NUM_TARGETS,
    PRECISION_CONFIG_VOCAB,
    RESOURCE_REGIME_VOCAB,
    TARGET_NAMES,
    PerfSeerOptimizedDataset,
    feature_layout,
    graph_signature_bucket,
    invert_targets,
    list_precision_pairs,
    split_dataset,
    split_hash,
    validate_precision_hardware_pairs,
)
from .metrics import all_metrics
from .model import SeerNet, SeerNetConfig, SeerNetMulti, count_parameters
from .train import append_jsonl, json_default


def safe_torch_load(path: str, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def stats_from_metadata(ckpt: dict[str, Any]) -> dict[str, np.ndarray]:
    meta = ckpt.get("metadata", {})
    stats = meta.get("norm_stats")
    if not stats:
        raise KeyError("checkpoint metadata does not include full norm_stats")
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def discover_checkpoints(args: argparse.Namespace) -> list[str]:
    if args.ckpt:
        return list(args.ckpt)
    if args.ckpt_dir:
        multi = sorted(glob.glob(os.path.join(args.ckpt_dir, "seernet_multi.pt")))
        singles = sorted(glob.glob(os.path.join(args.ckpt_dir, "seernet_metric*.pt")))
        found = multi or singles
        if not found:
            raise FileNotFoundError(f"no optimized checkpoint files found under {args.ckpt_dir}")
        return found
    raise ValueError("provide --ckpt <file...> or --ckpt-dir <dir>")


def load_model(path: str, device: torch.device):
    ckpt = safe_torch_load(path, device)
    cfg = SeerNetConfig.from_dict(ckpt["model_config"])
    is_multi = ckpt.get("model_name") == "seernet_multi" or "metric_idx" not in ckpt
    model = SeerNetMulti(cfg).to(device) if is_multi else SeerNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, is_multi


def build_test_dataset(args: argparse.Namespace, ckpt: dict[str, Any], norm_stats: dict[str, np.ndarray]):
    meta = ckpt.get("metadata", {})
    cfg = meta.get("config", {})
    data_root = args.data_root or cfg.get("data", {}).get("root", "dataset")
    feature_cfg = FeatureConfig.from_dict(meta.get("feature_config") or cfg.get("features"))
    split_meta = meta.get("split", {})
    seed = int(args.seed if args.seed is not None else split_meta.get("seed", cfg.get("seed", 42)))
    split_unit = str(split_meta.get("split_unit", cfg.get("data", {}).get("split_unit", "pair")) or "pair")
    test_pair_ids = split_meta.get("test_pair_ids")
    test_stems = split_meta.get("test_stems")
    reconstruction_source = "computed_split"
    if test_pair_ids:
        by_pair = {(Path(gp).stem, Path(lp).stem): (gp, lp) for gp, lp in list_precision_pairs(data_root)}
        test_files = [
            by_pair[(item["graph_stem"], item["label_stem"])]
            for item in test_pair_ids
            if (item["graph_stem"], item["label_stem"]) in by_pair
        ]
        reconstruction_source = "checkpoint_test_pair_ids"
    elif test_stems:
        by_stem = {Path(gp).stem: (gp, lp) for gp, lp in list_precision_pairs(data_root)}
        test_files = [by_stem[stem] for stem in test_stems if stem in by_stem]
        reconstruction_source = "checkpoint_test_stems"
    else:
        _train, _val, test_files = split_dataset(data_root, seed=seed, split_unit=split_unit)
        limit = int(args.limit if args.limit is not None else cfg.get("data", {}).get("limit", 0) or 0)
        if limit > 0:
            test_files = test_files[: max(1, limit // 2)]
    if args.limit is not None and args.limit > 0 and not test_stems:
        test_files = test_files[: max(1, args.limit // 2)]
    setattr(
        args,
        "_eval_split_metadata",
        {
            "seed": seed,
            "split_unit": split_unit,
            "source": reconstruction_source,
            "checkpoint_split_unit": split_meta.get("split_unit"),
            "checkpoint_test_hash": split_meta.get("test_hash"),
            "test_hash": split_hash(test_files),
            "test_count": len(test_files),
            "test_pair_ids_reconstructed": bool(test_pair_ids),
            "limit_applied": bool(args.limit is not None and args.limit > 0),
        },
    )
    supported = meta.get("supported_precision_hardware") or split_meta.get("supported_precision_hardware")
    validate_precision_hardware_pairs(test_files, feature_cfg, supported, context="evaluation")
    ds = PerfSeerOptimizedDataset(file_list=test_files, split="test", root=data_root, norm_stats=norm_stats, feature_config=feature_cfg)
    return ds, feature_cfg, data_root


def split_result_fields(args: argparse.Namespace) -> dict[str, Any]:
    eval_split = getattr(args, "_eval_split_metadata", {}) or {}
    return {
        "split_unit": eval_split.get("split_unit"),
        "test_hash": eval_split.get("test_hash"),
        "evaluation_split": eval_split,
    }


def apply_calibration(pred_std: np.ndarray, ckpt: dict[str, Any]) -> np.ndarray:
    cal = LinearCalibrator.from_dict(ckpt.get("calibration"))
    return cal.apply(pred_std) if cal is not None else pred_std


def feature_config_from_checkpoint(ckpt: dict[str, Any]) -> FeatureConfig:
    meta = ckpt.get("metadata", {})
    cfg = meta.get("config", {})
    return FeatureConfig.from_dict(meta.get("feature_config") or cfg.get("features"))


def invert_selected(
    pred_std: np.ndarray,
    stats: dict[str, np.ndarray],
    metric_idx: int,
    feature_cfg: FeatureConfig | None = None,
    base_raw: np.ndarray | None = None,
) -> np.ndarray:
    base = None if base_raw is None else np.asarray(base_raw)[:, metric_idx : metric_idx + 1]
    return invert_targets(pred_std.reshape(-1, 1), {"y_mean": stats["y_mean"][metric_idx : metric_idx + 1], "y_std": stats["y_std"][metric_idx : metric_idx + 1]}, feature_cfg, base).reshape(-1)


def invert_all(
    pred_std: np.ndarray,
    stats: dict[str, np.ndarray],
    feature_cfg: FeatureConfig | None = None,
    base_raw: np.ndarray | None = None,
) -> np.ndarray:
    return invert_targets(pred_std, stats, feature_cfg, base_raw)


def batch_base_raw(batch) -> np.ndarray | None:
    value = getattr(batch, "y_base_raw", None)
    if value is None:
        return None
    return value.view(-1, NUM_TARGETS).detach().cpu().numpy().astype(np.float64)


def batch_eval_raw(batch) -> np.ndarray | None:
    value = getattr(batch, "y_eval_raw", None)
    if value is None:
        return None
    return value.view(-1, NUM_TARGETS).detach().cpu().numpy().astype(np.float64)


def batch_counts(batch) -> tuple[np.ndarray, np.ndarray]:
    graph_count = int(getattr(batch, "num_graphs", 1))
    node_batch = getattr(batch, "batch", None)
    if node_batch is None:
        node_counts = np.asarray([batch.x.size(0)], dtype=np.int64)
    else:
        node_counts = torch.bincount(node_batch, minlength=graph_count).cpu().numpy()
    if batch.edge_index.numel() == 0:
        edge_counts = np.zeros(graph_count, dtype=np.int64)
    else:
        src_graph = node_batch[batch.edge_index[0]] if node_batch is not None else torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=batch.edge_index.device)
        edge_counts = torch.bincount(src_graph, minlength=graph_count).cpu().numpy()
    return node_counts, edge_counts


def batch_precision_ids(batch) -> np.ndarray:
    ids = getattr(batch, "precision_config_idx", None)
    graph_count = int(getattr(batch, "num_graphs", 1))
    if ids is None:
        return np.full(graph_count, -1, dtype=np.int64)
    return ids.view(-1).detach().cpu().numpy().astype(np.int64)


def batch_label_domain_ids(batch) -> np.ndarray:
    ids = getattr(batch, "label_domain_idx", None)
    graph_count = int(getattr(batch, "num_graphs", 1))
    if ids is None:
        return np.full(graph_count, -1, dtype=np.int64)
    return ids.view(-1).detach().cpu().numpy().astype(np.int64)


def batch_resource_regime_ids(batch) -> np.ndarray:
    ids = getattr(batch, "resource_regime_idx", None)
    graph_count = int(getattr(batch, "num_graphs", 1))
    if ids is None:
        return np.full(graph_count, -1, dtype=np.int64)
    return ids.view(-1).detach().cpu().numpy().astype(np.int64)


def batch_size_values(batch) -> np.ndarray:
    values = getattr(batch, "batch_size_raw", None)
    graph_count = int(getattr(batch, "num_graphs", 1))
    if values is None:
        return np.zeros(graph_count, dtype=np.float64)
    return values.view(-1).detach().cpu().numpy().astype(np.float64)


def batch_graph_family_names(batch) -> np.ndarray:
    graph_count = int(getattr(batch, "num_graphs", 1))
    value = getattr(batch, "graph_family_name", None)
    if value is None:
        return np.full(graph_count, "unknown", dtype=object)
    if isinstance(value, str):
        return np.full(graph_count, value, dtype=object)
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
        if len(names) == graph_count:
            return np.asarray(names, dtype=object)
        if len(names) == 1:
            return np.full(graph_count, names[0], dtype=object)
    return np.full(graph_count, "unknown", dtype=object)


def batch_hardware_ids(batch) -> np.ndarray:
    graph_count = int(getattr(batch, "num_graphs", 1))
    value = getattr(batch, "hardware_id_name", None)
    if value is None:
        return np.full(graph_count, "unknown", dtype=object)
    if isinstance(value, str):
        return np.full(graph_count, value, dtype=object)
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
        if len(names) == graph_count:
            return np.asarray(names, dtype=object)
        if len(names) == 1:
            return np.full(graph_count, names[0], dtype=object)
    return np.full(graph_count, "unknown", dtype=object)


def evaluate_multi(model, ckpt, loader, device, stats):
    feature_cfg = feature_config_from_checkpoint(ckpt)
    preds_std: list[np.ndarray] = []
    targets_std: list[np.ndarray] = []
    targets_eval_raw: list[np.ndarray] = []
    base_raw: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    precision_ids: list[np.ndarray] = []
    label_domain_ids: list[np.ndarray] = []
    batch_sizes: list[np.ndarray] = []
    resource_ids: list[np.ndarray] = []
    hardware_ids: list[np.ndarray] = []
    graph_families: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds_std.append(pred.cpu().numpy())
            targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
            eval_raw = batch_eval_raw(batch)
            base = batch_base_raw(batch)
            if eval_raw is not None:
                targets_eval_raw.append(eval_raw)
            if base is not None:
                base_raw.append(base)
            nc, ec = batch_counts(batch)
            node_counts.append(nc)
            edge_counts.append(ec)
            precision_ids.append(batch_precision_ids(batch))
            label_domain_ids.append(batch_label_domain_ids(batch))
            batch_sizes.append(batch_size_values(batch))
            resource_ids.append(batch_resource_regime_ids(batch))
            hardware_ids.append(batch_hardware_ids(batch))
            graph_families.append(batch_graph_family_names(batch))
    pred_std = apply_calibration(np.concatenate(preds_std, axis=0), ckpt)
    true_std = np.concatenate(targets_std, axis=0)
    base_all = np.concatenate(base_raw, axis=0) if base_raw else None
    y_pred = invert_all(pred_std, stats, feature_cfg, base_all)
    y_true = np.concatenate(targets_eval_raw, axis=0) if targets_eval_raw else invert_all(true_std, stats, feature_cfg, base_all)
    return (
        y_true,
        y_pred,
        np.concatenate(node_counts),
        np.concatenate(edge_counts),
        np.concatenate(precision_ids),
        np.concatenate(label_domain_ids),
        np.concatenate(batch_sizes),
        np.concatenate(resource_ids),
        np.concatenate(hardware_ids),
        np.concatenate(graph_families),
    )


def evaluate_singles(models, ckpts, loader, device, stats):
    feature_cfg = feature_config_from_checkpoint(ckpts[0])
    metric_to_pred: dict[int, list[np.ndarray]] = {int(c["metric_idx"]): [] for c in ckpts}
    targets_std: list[np.ndarray] = []
    targets_eval_raw: list[np.ndarray] = []
    base_raw: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    precision_ids: list[np.ndarray] = []
    label_domain_ids: list[np.ndarray] = []
    batch_sizes: list[np.ndarray] = []
    resource_ids: list[np.ndarray] = []
    hardware_ids: list[np.ndarray] = []
    graph_families: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
            eval_raw = batch_eval_raw(batch)
            base = batch_base_raw(batch)
            if eval_raw is not None:
                targets_eval_raw.append(eval_raw)
            if base is not None:
                base_raw.append(base)
            nc, ec = batch_counts(batch)
            node_counts.append(nc)
            edge_counts.append(ec)
            precision_ids.append(batch_precision_ids(batch))
            label_domain_ids.append(batch_label_domain_ids(batch))
            batch_sizes.append(batch_size_values(batch))
            resource_ids.append(batch_resource_regime_ids(batch))
            hardware_ids.append(batch_hardware_ids(batch))
            graph_families.append(batch_graph_family_names(batch))
            for model, ckpt in zip(models, ckpts):
                pred = model(batch).cpu().numpy()
                pred = apply_calibration(pred, ckpt)
                metric_to_pred[int(ckpt["metric_idx"])].append(pred.reshape(-1))
    true_std = np.concatenate(targets_std, axis=0)
    base_all = np.concatenate(base_raw, axis=0) if base_raw else None
    y_true_all = np.concatenate(targets_eval_raw, axis=0) if targets_eval_raw else invert_all(true_std, stats, feature_cfg, base_all)
    y_pred_all = np.full_like(y_true_all, np.nan)
    for metric_idx, chunks in metric_to_pred.items():
        pred_std = np.concatenate(chunks)
        y_pred_all[:, metric_idx] = invert_selected(pred_std, stats, metric_idx, feature_cfg, base_all)
    return (
        y_true_all,
        y_pred_all,
        np.concatenate(node_counts),
        np.concatenate(edge_counts),
        np.concatenate(precision_ids),
        np.concatenate(label_domain_ids),
        np.concatenate(batch_sizes),
        np.concatenate(resource_ids),
        np.concatenate(hardware_ids),
        np.concatenate(graph_families),
    )


def rows_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for idx, name in enumerate(TARGET_NAMES):
        if np.all(np.isnan(y_pred[:, idx])):
            continue
        rows[idx] = all_metrics(y_true[:, idx], y_pred[:, idx])
        rows[idx]["name"] = name
    return rows


def rows_by_precision(y_true: np.ndarray, y_pred: np.ndarray, precision_ids: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for idx in sorted(set(int(value) for value in precision_ids if int(value) >= 0)):
        if idx >= len(PRECISION_CONFIG_VOCAB):
            name = f"unknown_{idx}"
        else:
            name = PRECISION_CONFIG_VOCAB[idx]
        mask = precision_ids == idx
        rows = rows_from_predictions(y_true[mask], y_pred[mask])
        out[name] = {TARGET_NAMES[metric_idx]: rows[metric_idx] for metric_idx in rows}
    return out


def precision_config_counts(precision_ids: np.ndarray) -> dict[str, int]:
    return counts_by_index_slice(precision_ids, PRECISION_CONFIG_VOCAB, "precision")


def rows_by_label_domain(y_true: np.ndarray, y_pred: np.ndarray, label_domain_ids: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    return rows_by_index_slice(y_true, y_pred, label_domain_ids, LABEL_DOMAIN_VOCAB, "label_domain")


def counts_by_index_slice(ids: np.ndarray, vocab: Sequence[str], unknown_prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx in sorted(set(int(value) for value in ids if int(value) >= 0)):
        name = vocab[idx] if idx < len(vocab) else f"{unknown_prefix}_{idx}"
        out[name] = int(np.sum(ids == idx))
    return out


def label_domain_counts(label_domain_ids: np.ndarray) -> dict[str, int]:
    return counts_by_index_slice(label_domain_ids, LABEL_DOMAIN_VOCAB, "label_domain")


def string_value_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        name = str(value)
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def rows_by_index_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ids: np.ndarray,
    vocab: Sequence[str],
    unknown_prefix: str,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for idx in sorted(set(int(value) for value in ids if int(value) >= 0)):
        name = vocab[idx] if idx < len(vocab) else f"{unknown_prefix}_{idx}"
        mask = ids == idx
        rows = rows_from_predictions(y_true[mask], y_pred[mask])
        out[name] = {TARGET_NAMES[metric_idx]: rows[metric_idx] for metric_idx in rows}
    return out


def batch_bucket(value: float) -> str:
    if value <= 0:
        return "unknown"
    for threshold in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if value <= threshold:
            return f"bs_le_{threshold}"
    return "bs_gt_256"


def rows_by_batch_size(y_true: np.ndarray, y_pred: np.ndarray, batch_sizes: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    buckets = np.asarray([batch_bucket(float(value)) for value in batch_sizes], dtype=object)
    for bucket in sorted(set(str(value) for value in buckets)):
        if bucket == "unknown":
            continue
        mask = buckets == bucket
        rows = rows_from_predictions(y_true[mask], y_pred[mask])
        out[bucket] = {TARGET_NAMES[metric_idx]: rows[metric_idx] for metric_idx in rows}
    return out


def rows_by_graph_signature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_counts: np.ndarray,
    edge_counts: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    buckets = np.asarray(
        [graph_signature_bucket(float(nodes), float(edges)) for nodes, edges in zip(node_counts, edge_counts)],
        dtype=object,
    )
    for bucket in sorted(set(str(value) for value in buckets)):
        if bucket == "unknown":
            continue
        mask = buckets == bucket
        rows = rows_from_predictions(y_true[mask], y_pred[mask])
        out[bucket] = {TARGET_NAMES[metric_idx]: rows[metric_idx] for metric_idx in rows}
    return out


def rows_by_graph_family(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    graph_families: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    families = np.asarray([str(value) for value in graph_families], dtype=object)
    for family in sorted(set(str(value) for value in families)):
        if family == "unknown":
            continue
        mask = families == family
        rows = rows_from_predictions(y_true[mask], y_pred[mask])
        out[family] = {TARGET_NAMES[metric_idx]: rows[metric_idx] for metric_idx in rows}
    return out


def print_slice_summary(title: str, grouped_rows: dict[str, dict[str, dict[str, float]]]) -> None:
    for name, metrics in grouped_rows.items():
        if not metrics:
            continue
        mapes = [row["MAPE"] for row in metrics.values() if not np.isnan(row["MAPE"])]
        if mapes:
            print(f"{title} {name}: mean MAPE {np.mean(mapes):.3f}% over {len(metrics)} metrics", flush=True)


def print_table(rows: dict[int, dict[str, float]]) -> None:
    header = f"{'metric':<12} {'MAPE%':>9} {'RMSPE%':>9} {'5%Acc':>8} {'10%Acc':>8}"
    print("\n" + header)
    print("-" * len(header))
    mapes: list[float] = []
    for idx in sorted(rows):
        row = rows[idx]
        print(f"{TARGET_NAMES[idx]:<12} {row['MAPE']:>9.3f} {row['RMSPE']:>9.3f} {row['5Acc'] * 100:>7.2f}% {row['10Acc'] * 100:>7.2f}%")
        if not np.isnan(row["MAPE"]):
            mapes.append(row["MAPE"])
    print("-" * len(header))
    if mapes:
        print(f"{'MEAN MAPE':<12} {np.mean(mapes):>9.3f}   (over {len(mapes)} metrics)")


def export_predictions(path: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = []
        for name in TARGET_NAMES:
            header.extend([f"{name}_true", f"{name}_pred"])
        writer.writerow(header)
        for i in range(y_true.shape[0]):
            row: list[float] = []
            for j in range(NUM_TARGETS):
                row.extend([float(y_true[i, j]), float(y_pred[i, j])])
            writer.writerow(row)


def bucket_summary(values: np.ndarray, errors: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    finite = np.isfinite(values) & np.isfinite(errors)
    if not np.any(finite):
        return []
    values = values[finite]
    errors = errors[finite]
    qs = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if qs.size < 2:
        qs = np.asarray([np.min(values), np.max(values) + 1e-9])
    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (values >= lo) & (values <= hi if hi == qs[-1] else values < hi)
        if np.any(mask):
            out.append({"lo": float(lo), "hi": float(hi), "count": int(np.sum(mask)), "mape": float(np.mean(errors[mask]) * 100.0)})
    return out


def write_error_analysis(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, node_counts: np.ndarray, edge_counts: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for metric_idx, name in enumerate(TARGET_NAMES):
        if np.all(np.isnan(y_pred[:, metric_idx])):
            continue
        denom = np.maximum(np.abs(y_true[:, metric_idx]), 1e-8)
        rel = np.abs(y_true[:, metric_idx] - y_pred[:, metric_idx]) / denom
        for bucket_name, values in [("node_count", node_counts), ("edge_count", edge_counts), ("target_magnitude", y_true[:, metric_idx])]:
            for b in bucket_summary(np.asarray(values, dtype=np.float64), rel):
                rows.append({"metric": name, "bucket": bucket_name, **b})
    with open(os.path.join(out_dir, "error_by_bucket.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "bucket", "lo", "hi", "count", "mape"])
        writer.writeheader()
        writer.writerows(rows)

    worst = []
    for i in range(y_true.shape[0]):
        for metric_idx, name in enumerate(TARGET_NAMES):
            if np.isnan(y_pred[i, metric_idx]):
                continue
            rel = abs(y_true[i, metric_idx] - y_pred[i, metric_idx]) / max(abs(y_true[i, metric_idx]), 1e-8)
            worst.append({"sample_idx": i, "metric": name, "relative_error": rel, "true": y_true[i, metric_idx], "pred": y_pred[i, metric_idx]})
    worst = sorted(worst, key=lambda r: r["relative_error"], reverse=True)[:100]
    with open(os.path.join(out_dir, "worst_100_predictions.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_idx", "metric", "relative_error", "true", "pred"])
        writer.writeheader()
        writer.writerows(worst)

    try:
        import matplotlib.pyplot as plt

        for metric_idx, name in enumerate(TARGET_NAMES):
            if np.all(np.isnan(y_pred[:, metric_idx])):
                continue
            plt.figure(figsize=(5, 5))
            plt.scatter(y_true[:, metric_idx], y_pred[:, metric_idx], s=4, alpha=0.35)
            lo = float(np.nanmin([np.nanmin(y_true[:, metric_idx]), np.nanmin(y_pred[:, metric_idx])]))
            hi = float(np.nanmax([np.nanmax(y_true[:, metric_idx]), np.nanmax(y_pred[:, metric_idx])]))
            plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
            plt.xlabel("true")
            plt.ylabel("pred")
            plt.title(name)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"prediction_scatter_{name}.png"), dpi=160)
            plt.close()
    except Exception as exc:
        print(f"plot generation skipped: {exc}", flush=True)


def benchmark_models(models: list[torch.nn.Module], loader, device, num_graphs: int, warmup: int = 20) -> dict[str, float]:
    latencies: list[float] = []
    seen = 0
    for model in models:
        model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for _ in range(warmup if seen == 0 else 0):
                for model in models:
                    _ = model(batch)
            t0 = time.perf_counter()
            for model in models:
                _ = model(batch)
            elapsed = (time.perf_counter() - t0) * 1000.0
            n = int(getattr(batch, "num_graphs", 1))
            latencies.extend([elapsed / max(n, 1)] * n)
            seen += n
            if seen >= num_graphs:
                break
    return summarize_latencies(latencies[:num_graphs])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate optimized PerfSeer checkpoints.")
    p.add_argument("--eval-profile", help="optional YAML profile under configs/eval_profiles")
    p.add_argument("--ckpt", nargs="*", help="explicit checkpoint file(s)")
    p.add_argument("--ckpt-dir", dest="ckpt_dir", help="checkpoint directory")
    p.add_argument("--data-root", dest="data_root")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    p.add_argument("--device", default="auto")
    p.add_argument("--bench-cpu", action="store_true")
    p.add_argument("--num-bench-graphs", type=int, default=1000)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--cpu-interop-threads", type=int, default=0)
    p.add_argument("--export-predictions", help="CSV path")
    p.add_argument("--error-analysis", action="store_true")
    p.add_argument("--results-path", default="runs/results.jsonl")
    return p.parse_args(argv)


def apply_eval_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not args.eval_profile:
        return args
    with open(args.eval_profile, "r") as fh:
        profile = yaml.safe_load(fh) or {}
    if "batch_size" in profile:
        args.batch_size = int(profile["batch_size"])
    if "device" in profile:
        args.device = str(profile["device"])
    if "bench_cpu" in profile:
        args.bench_cpu = bool(profile["bench_cpu"])
    if "num_bench_graphs" in profile:
        args.num_bench_graphs = int(profile["num_bench_graphs"])
    if "cpu_threads" in profile:
        args.cpu_threads = int(profile["cpu_threads"] or 0)
    if "cpu_interop_threads" in profile:
        args.cpu_interop_threads = int(profile["cpu_interop_threads"] or 0)
    args._eval_profile_data = profile
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = apply_eval_profile(parse_args(argv))
    if args.bench_cpu:
        configure_cpu_threads(args.cpu_threads, args.cpu_interop_threads)
    if args.bench_cpu and args.device == "auto":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    paths = discover_checkpoints(args)
    loaded = [load_model(path, device) for path in paths]
    models = [item[0] for item in loaded]
    ckpts = [item[1] for item in loaded]
    first = ckpts[0]
    stats = stats_from_metadata(first)
    ds, feature_cfg, data_root = build_test_dataset(args, first, stats)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"test graphs: {len(ds)} | feature dims: {feature_layout(feature_cfg)}", flush=True)

    if loaded[0][2]:
        y_true, y_pred, node_counts, edge_counts, precision_ids, label_domain_ids, batch_sizes, resource_ids, hardware_ids, graph_families = evaluate_multi(models[0], first, loader, device, stats)
    else:
        y_true, y_pred, node_counts, edge_counts, precision_ids, label_domain_ids, batch_sizes, resource_ids, hardware_ids, graph_families = evaluate_singles(models, ckpts, loader, device, stats)
    rows = rows_from_predictions(y_true, y_pred)
    precision_rows = rows_by_precision(y_true, y_pred, precision_ids)
    precision_count_rows = precision_config_counts(precision_ids)
    label_domain_rows = rows_by_label_domain(y_true, y_pred, label_domain_ids)
    label_domain_count_rows = label_domain_counts(label_domain_ids)
    hardware_id_count_rows = string_value_counts(hardware_ids)
    batch_rows = rows_by_batch_size(y_true, y_pred, batch_sizes)
    resource_rows = rows_by_index_slice(y_true, y_pred, resource_ids, RESOURCE_REGIME_VOCAB, "resource")
    graph_signature_rows = rows_by_graph_signature(y_true, y_pred, node_counts, edge_counts)
    graph_family_rows = rows_by_graph_family(y_true, y_pred, graph_families)
    print_table(rows)
    print_slice_summary("precision", precision_rows)
    print_slice_summary("label_domain", label_domain_rows)
    print_slice_summary("batch", batch_rows)
    print_slice_summary("resource", resource_rows)
    print_slice_summary("graph_signature", graph_signature_rows)
    print_slice_summary("graph_family", graph_family_rows)

    bench = None
    if args.bench_cpu:
        bench_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        bench = benchmark_models(models, bench_loader, torch.device("cpu"), args.num_bench_graphs)
        print(f"CPU forward latency per graph: mean={bench['mean_ms']:.3f} ms p50={bench['p50_ms']:.3f} ms p95={bench['p95_ms']:.3f} ms", flush=True)

    out_dir = args.ckpt_dir or os.path.dirname(paths[0])
    if args.export_predictions:
        export_predictions(args.export_predictions, y_true, y_pred)
    if args.error_analysis:
        write_error_analysis(out_dir, y_true, y_pred, node_counts, edge_counts)

    mapes = [rows[idx]["MAPE"] for idx in rows if not np.isnan(rows[idx]["MAPE"])]
    run_id = first.get("metadata", {}).get("run_id", Path(out_dir).name)
    result_row = {
        "event": "eval_complete",
        "track": getattr(args, "_eval_profile_data", {}).get("track", "accuracy"),
        "run_id": run_id,
        "ckpt_paths": paths,
        "runtime_backend": getattr(args, "_eval_profile_data", {}).get("runtime_backend", "pytorch"),
        "runtime_backend_actual": "pytorch",
        "device": str(device),
        "cpu_threads": args.cpu_threads,
        "cpu_interop_threads": args.cpu_interop_threads,
        "batch_size": args.batch_size,
        "num_bench_graphs": args.num_bench_graphs if args.bench_cpu else 0,
        "data_root": data_root,
        "num_test_graphs": len(ds),
        **split_result_fields(args),
        "params": int(sum(count_parameters(model) for model in models)),
        "model_params": int(sum(count_parameters(model) for model in models)),
        "mean_mape": float(np.mean(mapes)) if mapes else float("nan"),
        "metrics": {TARGET_NAMES[idx]: rows[idx] for idx in rows},
        "metrics_by_precision": precision_rows,
        "precision_config_counts": precision_count_rows,
        "metrics_by_label_domain": label_domain_rows,
        "label_domain_counts": label_domain_count_rows,
        "hardware_id_counts": hardware_id_count_rows,
        "metrics_by_batch_size": batch_rows,
        "metrics_by_resource_regime": resource_rows,
        "metrics_by_graph_signature": graph_signature_rows,
        "metrics_by_graph_family": graph_family_rows,
        "cpu_forward": bench,
        "latency_forward_ms_mean": bench["mean_ms"] if bench else float("nan"),
        "latency_forward_ms_p50": bench["p50_ms"] if bench else float("nan"),
        "latency_forward_ms_p95": bench["p95_ms"] if bench else float("nan"),
        "graphs_per_sec": (1000.0 / bench["mean_ms"]) if bench and bench["mean_ms"] > 0 else float("nan"),
        "eval_profile": getattr(args, "_eval_profile_data", None),
    }
    append_jsonl(args.results_path, result_row)


if __name__ == "__main__":
    main()
