"""Evaluation CLI for optimized PerfSeer checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from perfseer.data import list_pairs

from .bench import configure_cpu_threads, summarize_latencies
from .calibration import LinearCalibrator
from .data import (
    FeatureConfig,
    NUM_TARGETS,
    TARGET_NAMES,
    PerfSeerOptimizedDataset,
    feature_layout,
    split_dataset,
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
    test_stems = split_meta.get("test_stems")
    if test_stems:
        by_stem = {Path(gp).stem: (gp, lp) for gp, lp in list_pairs(data_root)}
        test_files = [by_stem[stem] for stem in test_stems if stem in by_stem]
    else:
        seed = int(args.seed if args.seed is not None else split_meta.get("seed", cfg.get("seed", 42)))
        _train, _val, test_files = split_dataset(data_root, seed=seed)
        limit = int(args.limit if args.limit is not None else cfg.get("data", {}).get("limit", 0) or 0)
        if limit > 0:
            test_files = test_files[: max(1, limit // 2)]
    if args.limit is not None and args.limit > 0 and not test_stems:
        test_files = test_files[: max(1, args.limit // 2)]
    ds = PerfSeerOptimizedDataset(file_list=test_files, split="test", root=data_root, norm_stats=norm_stats, feature_config=feature_cfg)
    return ds, feature_cfg, data_root


def apply_calibration(pred_std: np.ndarray, ckpt: dict[str, Any]) -> np.ndarray:
    cal = LinearCalibrator.from_dict(ckpt.get("calibration"))
    return cal.apply(pred_std) if cal is not None else pred_std


def invert_selected(pred_std: np.ndarray, stats: dict[str, np.ndarray], metric_idx: int) -> np.ndarray:
    return np.expm1(pred_std.reshape(-1) * float(stats["y_std"][metric_idx]) + float(stats["y_mean"][metric_idx]))


def invert_all(pred_std: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return np.expm1(pred_std * stats["y_std"].reshape(1, -1) + stats["y_mean"].reshape(1, -1))


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


def evaluate_multi(model, ckpt, loader, device, stats):
    preds_std: list[np.ndarray] = []
    targets_std: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds_std.append(pred.cpu().numpy())
            targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
            nc, ec = batch_counts(batch)
            node_counts.append(nc)
            edge_counts.append(ec)
    pred_std = apply_calibration(np.concatenate(preds_std, axis=0), ckpt)
    true_std = np.concatenate(targets_std, axis=0)
    y_pred = invert_all(pred_std, stats)
    y_true = invert_all(true_std, stats)
    return y_true, y_pred, np.concatenate(node_counts), np.concatenate(edge_counts)


def evaluate_singles(models, ckpts, loader, device, stats):
    metric_to_pred: dict[int, list[np.ndarray]] = {int(c["metric_idx"]): [] for c in ckpts}
    targets_std: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
            nc, ec = batch_counts(batch)
            node_counts.append(nc)
            edge_counts.append(ec)
            for model, ckpt in zip(models, ckpts):
                pred = model(batch).cpu().numpy()
                pred = apply_calibration(pred, ckpt)
                metric_to_pred[int(ckpt["metric_idx"])].append(pred.reshape(-1))
    true_std = np.concatenate(targets_std, axis=0)
    y_true_all = invert_all(true_std, stats)
    y_pred_all = np.full_like(y_true_all, np.nan)
    for metric_idx, chunks in metric_to_pred.items():
        pred_std = np.concatenate(chunks)
        y_pred_all[:, metric_idx] = invert_selected(pred_std, stats, metric_idx)
    return y_true_all, y_pred_all, np.concatenate(node_counts), np.concatenate(edge_counts)


def rows_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for idx, name in enumerate(TARGET_NAMES):
        if np.all(np.isnan(y_pred[:, idx])):
            continue
        rows[idx] = all_metrics(y_true[:, idx], y_pred[:, idx])
        rows[idx]["name"] = name
    return rows


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


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
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
        y_true, y_pred, node_counts, edge_counts = evaluate_multi(models[0], first, loader, device, stats)
    else:
        y_true, y_pred, node_counts, edge_counts = evaluate_singles(models, ckpts, loader, device, stats)
    rows = rows_from_predictions(y_true, y_pred)
    print_table(rows)

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
        "run_id": run_id,
        "ckpt_paths": paths,
        "data_root": data_root,
        "num_test_graphs": len(ds),
        "params": int(sum(count_parameters(model) for model in models)),
        "mean_mape": float(np.mean(mapes)) if mapes else float("nan"),
        "metrics": {TARGET_NAMES[idx]: rows[idx] for idx in rows},
        "cpu_forward": bench,
    }
    append_jsonl(args.results_path, result_row)


if __name__ == "__main__":
    main()
