"""Deployment/runtime evaluation for optimized PerfSeer checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .bench import configure_cpu_threads
from .deploy import EvalProfile, benchmark_runtime, current_rss_mb, prepare_runtime
from .eval import (
    apply_calibration,
    build_test_dataset,
    discover_checkpoints,
    export_predictions,
    invert_all,
    invert_selected,
    load_model,
    print_table,
    rows_from_predictions,
    stats_from_metadata,
    write_error_analysis,
)
from .data import NUM_TARGETS, TARGET_NAMES
from .model import count_parameters
from .train import append_jsonl


def collect_predictions(runtime, ckpts, loader, is_multi: bool, stats):
    targets_std: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    if is_multi:
        preds_std: list[np.ndarray] = []
    else:
        metric_to_pred: dict[int, list[np.ndarray]] = {int(c["metric_idx"]): [] for c in ckpts}

    for batch in loader:
        targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
        node_batch = getattr(batch, "batch", None)
        graph_count = int(getattr(batch, "num_graphs", 1))
        if node_batch is None:
            node_counts.append(np.asarray([batch.x.size(0)], dtype=np.int64))
            edge_counts.append(np.asarray([batch.edge_index.size(1)], dtype=np.int64))
        else:
            node_counts.append(torch.bincount(node_batch, minlength=graph_count).cpu().numpy())
            if batch.edge_index.numel() == 0:
                edge_counts.append(np.zeros(graph_count, dtype=np.int64))
            else:
                src_graph = node_batch[batch.edge_index[0]]
                edge_counts.append(torch.bincount(src_graph, minlength=graph_count).cpu().numpy())

        if is_multi:
            pred = runtime.predict_one_model(0, batch)
            preds_std.append(apply_calibration(pred, ckpts[0]))
        else:
            for idx, ckpt in enumerate(ckpts):
                pred = runtime.predict_one_model(idx, batch)
                pred = apply_calibration(pred, ckpt)
                metric_to_pred[int(ckpt["metric_idx"])].append(pred.reshape(-1))

    true_std = np.concatenate(targets_std, axis=0)
    y_true = invert_all(true_std, stats)
    if is_multi:
        y_pred = invert_all(np.concatenate(preds_std, axis=0), stats)
    else:
        y_pred = np.full_like(y_true, np.nan)
        for metric_idx, chunks in metric_to_pred.items():
            y_pred[:, metric_idx] = invert_selected(np.concatenate(chunks), stats, metric_idx)
    return y_true, y_pred, np.concatenate(node_counts), np.concatenate(edge_counts)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PerfSeer deployment/runtime profiles.")
    p.add_argument("--eval-profile", required=True, help="YAML profile under configs/eval_profiles")
    p.add_argument("--ckpt", nargs="*", help="explicit checkpoint file(s)")
    p.add_argument("--ckpt-dir", dest="ckpt_dir", help="checkpoint directory")
    p.add_argument("--data-root", dest="data_root")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    p.add_argument("--num-bench-graphs", type=int)
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--cpu-interop-threads", type=int)
    p.add_argument("--export-predictions", help="CSV path")
    p.add_argument("--error-analysis", action="store_true")
    p.add_argument("--results-path", default="runs/results.jsonl")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    profile = EvalProfile.from_file(args.eval_profile)
    if args.batch_size is not None:
        profile.batch_size = args.batch_size
    if args.num_bench_graphs is not None:
        profile.num_bench_graphs = args.num_bench_graphs
    if args.cpu_threads is not None:
        profile.cpu_threads = args.cpu_threads
    if args.cpu_interop_threads is not None:
        profile.cpu_interop_threads = args.cpu_interop_threads

    if profile.device == "cpu" or profile.bench_cpu:
        configure_cpu_threads(profile.cpu_threads, profile.cpu_interop_threads)

    load_device = torch.device("cpu")
    paths = discover_checkpoints(args)
    loaded = [load_model(path, load_device) for path in paths]
    models = [item[0] for item in loaded]
    ckpts = [item[1] for item in loaded]
    is_multi = bool(loaded[0][2])
    first = ckpts[0]
    stats = stats_from_metadata(first)
    ds, _feature_cfg, data_root = build_test_dataset(args, first, stats)
    loader = DataLoader(ds, batch_size=profile.batch_size, shuffle=False, num_workers=args.num_workers)
    example_batch = next(iter(loader))

    out_dir = args.ckpt_dir or os.path.dirname(paths[0])
    artifact_dir = profile.artifact_dir or os.path.join(out_dir, "deploy_artifacts", profile.name)
    runtime = prepare_runtime(models, profile, example_batch, artifact_dir)

    y_true, y_pred, node_counts, edge_counts = collect_predictions(runtime, ckpts, loader, is_multi, stats)
    rows = rows_from_predictions(y_true, y_pred)
    print_table(rows)

    bench_loader = DataLoader(ds, batch_size=profile.batch_size, shuffle=False, num_workers=0)
    bench = benchmark_runtime(runtime, bench_loader, profile.num_bench_graphs, profile.warmup)
    print(
        f"{profile.name}: backend={runtime.backend} "
        f"mean={bench['mean_ms']:.3f} ms p50={bench['p50_ms']:.3f} ms "
        f"p95={bench['p95_ms']:.3f} ms throughput={bench['graphs_per_sec']:.2f} graphs/s",
        flush=True,
    )
    for status in runtime.statuses:
        if status.get("status") != "ok":
            print(f"runtime fallback: {status}", flush=True)

    if args.export_predictions:
        export_predictions(args.export_predictions, y_true, y_pred)
    if args.error_analysis:
        write_error_analysis(out_dir, y_true, y_pred, node_counts, edge_counts)

    mapes = [rows[idx]["MAPE"] for idx in rows if not np.isnan(rows[idx]["MAPE"])]
    run_id = first.get("metadata", {}).get("run_id", Path(out_dir).name)
    append_jsonl(
        args.results_path,
        {
            "event": "eval_deploy_complete",
            "track": profile.track,
            "run_id": run_id,
            "ckpt_paths": paths,
            "runtime_backend": profile.runtime_backend,
            "runtime_backend_actual": runtime.backend,
            "runtime_statuses": runtime.statuses,
            "device": profile.device,
            "cpu_threads": profile.cpu_threads,
            "cpu_interop_threads": profile.cpu_interop_threads,
            "batch_size": profile.batch_size,
            "num_bench_graphs": profile.num_bench_graphs,
            "num_test_graphs": len(ds),
            "data_root": data_root,
            "model_params": int(sum(count_parameters(model) for model in models)),
            "artifact_size_mb": runtime.artifact_size_mb,
            "rss_mb": current_rss_mb(),
            "latency_forward_ms_mean": bench["mean_ms"],
            "latency_forward_ms_p50": bench["p50_ms"],
            "latency_forward_ms_p95": bench["p95_ms"],
            "graphs_per_sec": bench["graphs_per_sec"],
            "mean_mape": float(np.mean(mapes)) if mapes else float("nan"),
            "metrics": {TARGET_NAMES[idx]: rows[idx] for idx in rows},
            "eval_profile": profile.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
