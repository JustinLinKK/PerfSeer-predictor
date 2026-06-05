"""Deployment/runtime evaluation for optimized PerfSeer checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .bench import configure_cpu_threads
from .deploy import EvalProfile, PreparedRuntime, benchmark_runtime, current_rss_mb, prepare_runtime
from .eval import (
    apply_calibration,
    batch_hardware_ids,
    batch_label_domain_ids,
    batch_precision_ids,
    batch_resource_regime_ids,
    batch_size_values,
    batch_graph_family_names,
    batch_base_raw,
    batch_eval_raw,
    build_test_dataset,
    discover_checkpoints,
    export_predictions,
    feature_config_from_checkpoint,
    invert_all,
    invert_selected,
    label_domain_counts,
    load_model,
    print_table,
    print_slice_summary,
    precision_config_counts,
    rows_by_batch_size,
    rows_by_graph_family,
    rows_by_graph_signature,
    rows_by_index_slice,
    rows_by_label_domain,
    rows_by_precision,
    rows_from_predictions,
    split_result_fields,
    stats_from_metadata,
    string_value_counts,
    write_error_analysis,
)
from .data import NUM_TARGETS, RESOURCE_REGIME_VOCAB, TARGET_NAMES, FeatureConfig, feature_layout, precision_hardware_config
from .model import count_parameters
from .train import append_jsonl, json_default


def collect_predictions(runtime, ckpts, loader, is_multi: bool, stats):
    feature_cfg = feature_config_from_checkpoint(ckpts[0])
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
    if is_multi:
        preds_std: list[np.ndarray] = []
    else:
        metric_to_pred: dict[int, list[np.ndarray]] = {int(c["metric_idx"]): [] for c in ckpts}

    for batch in loader:
        targets_std.append(batch.y.view(-1, NUM_TARGETS).cpu().numpy())
        eval_raw = batch_eval_raw(batch)
        base = batch_base_raw(batch)
        if eval_raw is not None:
            targets_eval_raw.append(eval_raw)
        if base is not None:
            base_raw.append(base)
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
        precision_ids.append(batch_precision_ids(batch))
        label_domain_ids.append(batch_label_domain_ids(batch))
        batch_sizes.append(batch_size_values(batch))
        resource_ids.append(batch_resource_regime_ids(batch))
        hardware_ids.append(batch_hardware_ids(batch))
        graph_families.append(batch_graph_family_names(batch))

        if is_multi:
            pred = runtime.predict_one_model(0, batch)
            preds_std.append(apply_calibration(pred, ckpts[0]))
        else:
            for idx, ckpt in enumerate(ckpts):
                pred = runtime.predict_one_model(idx, batch)
                pred = apply_calibration(pred, ckpt)
                metric_to_pred[int(ckpt["metric_idx"])].append(pred.reshape(-1))

    true_std = np.concatenate(targets_std, axis=0)
    base_all = np.concatenate(base_raw, axis=0) if base_raw else None
    y_true = np.concatenate(targets_eval_raw, axis=0) if targets_eval_raw else invert_all(true_std, stats, feature_cfg, base_all)
    if is_multi:
        y_pred = invert_all(np.concatenate(preds_std, axis=0), stats, feature_cfg, base_all)
    else:
        y_pred = np.full_like(y_true, np.nan)
        for metric_idx, chunks in metric_to_pred.items():
            y_pred[:, metric_idx] = invert_selected(np.concatenate(chunks), stats, metric_idx, feature_cfg, base_all)
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


def deployment_metadata(
    *,
    ckpt: dict,
    ckpt_paths: list[str],
    runtime: PreparedRuntime,
    profile: EvalProfile,
    feature_cfg: FeatureConfig,
    data_root: str,
    split_fields: dict,
) -> dict:
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    layout = feature_layout(feature_cfg)
    return {
        "schema_version": 1,
        "run_id": meta.get("run_id"),
        "checkpoint_paths": list(ckpt_paths),
        "runtime_backend": profile.runtime_backend,
        "runtime_backend_actual": runtime.backend,
        "runtime_statuses": runtime.statuses,
        "runtime_fallback": runtime.fallback,
        "artifact_paths": list(runtime.artifact_paths),
        "eval_profile": profile.to_dict(),
        "data_root": data_root,
        "split_unit": split_fields.get("split_unit"),
        "test_hash": split_fields.get("test_hash"),
        "evaluation_split": split_fields.get("evaluation_split"),
        "feature_config": feature_cfg.to_dict(),
        "precision_hardware_config": meta.get("precision_hardware_config") or precision_hardware_config(feature_cfg),
        "feature_layout": meta.get(
            "feature_layout",
            {
                "node_dim": layout.node_dim,
                "edge_dim": layout.edge_dim,
                "global_dim": layout.global_dim,
                "node_names": list(layout.node_names),
                "edge_names": list(layout.edge_names),
                "global_names": list(layout.global_names),
            },
        ),
        "supported_precision_hardware": meta.get("supported_precision_hardware", {}),
        "required_inputs": {
            "x": {"dim": layout.node_dim},
            "edge_attr": {"dim": layout.edge_dim},
            "u": {"dim": layout.global_dim},
            "edge_index": {"dtype": "int64", "shape": ["2", "num_edges"]},
            "batch": {"dtype": "int64", "shape": ["num_nodes"]},
        },
    }


def write_deployment_metadata(
    artifact_dir: str,
    *,
    ckpt: dict,
    ckpt_paths: list[str],
    runtime: PreparedRuntime,
    profile: EvalProfile,
    feature_cfg: FeatureConfig,
    data_root: str,
    split_fields: dict,
) -> str:
    metadata = deployment_metadata(
        ckpt=ckpt,
        ckpt_paths=ckpt_paths,
        runtime=runtime,
        profile=profile,
        feature_cfg=feature_cfg,
        data_root=data_root,
        split_fields=split_fields,
    )
    path = Path(artifact_dir) / "deployment_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n")
    metadata_path = str(path)
    if metadata_path not in runtime.artifact_paths:
        runtime.artifact_paths.append(metadata_path)
    return metadata_path


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
    ds, feature_cfg, data_root = build_test_dataset(args, first, stats)
    loader = DataLoader(ds, batch_size=profile.batch_size, shuffle=False, num_workers=args.num_workers)
    example_batch = next(iter(loader))

    out_dir = args.ckpt_dir or os.path.dirname(paths[0])
    artifact_dir = profile.artifact_dir or os.path.join(out_dir, "deploy_artifacts", profile.name)
    runtime = prepare_runtime(models, profile, example_batch, artifact_dir)
    split_fields = split_result_fields(args)
    deployment_metadata_path = write_deployment_metadata(
        artifact_dir,
        ckpt=first,
        ckpt_paths=paths,
        runtime=runtime,
        profile=profile,
        feature_cfg=feature_cfg,
        data_root=data_root,
        split_fields=split_fields,
    )

    y_true, y_pred, node_counts, edge_counts, precision_ids, label_domain_ids, batch_sizes, resource_ids, hardware_ids, graph_families = collect_predictions(runtime, ckpts, loader, is_multi, stats)
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
            **split_fields,
            "model_params": int(sum(count_parameters(model) for model in models)),
            "artifact_size_mb": runtime.artifact_size_mb,
            "deployment_metadata": deployment_metadata_path,
            "rss_mb": current_rss_mb(),
            "latency_forward_ms_mean": bench["mean_ms"],
            "latency_forward_ms_p50": bench["p50_ms"],
            "latency_forward_ms_p95": bench["p95_ms"],
            "graphs_per_sec": bench["graphs_per_sec"],
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
            "eval_profile": profile.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
