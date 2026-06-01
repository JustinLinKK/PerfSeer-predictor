#!/usr/bin/env python
"""Create analysis plots from the optimized PerfSeer result ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET_NAMES = [
    "train_util",
    "train_mem",
    "train_time",
    "infer_util",
    "infer_mem",
    "infer_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PerfSeer experiment results.")
    p.add_argument("--results", default="runs/results.jsonl")
    p.add_argument("--out-dir", default="runs/plots")
    p.add_argument("--title", default="PerfSeer optimized experiment")
    return p.parse_args()


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") not in {"eval_complete", "eval_deploy_complete"}:
            continue
        backend = row.get("runtime_backend_actual") or row.get("runtime_backend") or "pytorch"
        key = (
            row.get("event"),
            row.get("track", "accuracy"),
            row.get("run_id", ""),
            backend,
            row.get("device", ""),
            row.get("batch_size", ""),
        )
        row["_line_no"] = line_no
        latest[key] = row
    return list(latest.values())


def backend(row: dict[str, Any]) -> str:
    return str(row.get("runtime_backend_actual") or row.get("runtime_backend") or "pytorch")


def metric_mape(row: dict[str, Any], name: str) -> float:
    metrics = row.get("metrics") or {}
    return as_float((metrics.get(name) or {}).get("MAPE"))


def row_label(row: dict[str, Any]) -> str:
    run_id = str(row.get("run_id", ""))
    short = run_id.replace("accuracy_", "acc_").replace("deploy_", "dep_")
    return f"{short}\n{backend(row)}"


def finite_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [row for row in rows if math.isfinite(as_float(row.get(field)))]


def write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "track",
        "run_id",
        "event",
        "backend",
        "device",
        "mean_mape",
        "train_time_mape",
        "infer_time_mape",
        "latency_p50_ms",
        "latency_p95_ms",
        "graphs_per_sec",
        "model_params",
        "artifact_size_mb",
        "rss_mb",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.get("track", ""), as_float(r.get("mean_mape")))):
            params = as_float(row.get("model_params", row.get("params", 0)))
            if not math.isfinite(params):
                params = 0
            writer.writerow(
                {
                    "track": row.get("track", "accuracy"),
                    "run_id": row.get("run_id", ""),
                    "event": row.get("event", ""),
                    "backend": backend(row),
                    "device": row.get("device", ""),
                    "mean_mape": as_float(row.get("mean_mape")),
                    "train_time_mape": metric_mape(row, "train_time"),
                    "infer_time_mape": metric_mape(row, "infer_time"),
                    "latency_p50_ms": as_float(row.get("latency_forward_ms_p50")),
                    "latency_p95_ms": as_float(row.get("latency_forward_ms_p95")),
                    "graphs_per_sec": as_float(row.get("graphs_per_sec")),
                    "model_params": int(params),
                    "artifact_size_mb": as_float(row.get("artifact_size_mb")),
                    "rss_mb": as_float(row.get("rss_mb")),
                }
            )


def save_bar(
    rows: list[dict[str, Any]],
    values: list[float],
    labels: list[str],
    ylabel: str,
    title: str,
    out_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 5.5))
    x = np.arange(len(rows))
    ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_accuracy(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    acc_rows = [
        row
        for row in rows
        if row.get("event") == "eval_complete" and row.get("track", "accuracy") == "accuracy"
    ]
    acc_rows = sorted(finite_rows(acc_rows, "mean_mape"), key=lambda r: as_float(r.get("mean_mape")))
    if not acc_rows:
        return
    save_bar(
        acc_rows,
        [as_float(row.get("mean_mape")) for row in acc_rows],
        [str(row.get("run_id", "")) for row in acc_rows],
        "Mean MAPE (%)",
        f"{title}: GPU accuracy ranking",
        out_dir / "accuracy_mean_mape.png",
        "#4C78A8",
    )


def plot_cpu_latency(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    cpu_rows = finite_rows(rows, "latency_forward_ms_p95")
    cpu_rows = sorted(cpu_rows, key=lambda r: as_float(r.get("latency_forward_ms_p95")))
    if not cpu_rows:
        return
    save_bar(
        cpu_rows,
        [as_float(row.get("latency_forward_ms_p95")) for row in cpu_rows],
        [row_label(row) for row in cpu_rows],
        "CPU p95 forward latency per graph (ms)",
        f"{title}: CPU latency ranking",
        out_dir / "cpu_latency_p95.png",
        "#F58518",
    )


def plot_pareto(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    cpu_rows = [
        row
        for row in rows
        if math.isfinite(as_float(row.get("latency_forward_ms_p95")))
        and math.isfinite(as_float(row.get("mean_mape")))
    ]
    if not cpu_rows:
        return
    colors = {
        "pytorch": "#4C78A8",
        "pytorch_dynamic_int8": "#72B7B2",
        "torchscript": "#54A24B",
        "onnxruntime": "#E45756",
        "onnxruntime_int8": "#B279A2",
        "openvino": "#F58518",
        "openvino_int8": "#9D755D",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in cpu_rows:
        b = backend(row)
        ax.scatter(
            as_float(row.get("latency_forward_ms_p95")),
            as_float(row.get("mean_mape")),
            s=80,
            color=colors.get(b, "#777777"),
            label=b,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.annotate(row_label(row), (as_float(row.get("latency_forward_ms_p95")), as_float(row.get("mean_mape"))), fontsize=7, xytext=(4, 4), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")
    ax.set_xlabel("CPU p95 forward latency per graph (ms)")
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title(f"{title}: accuracy/latency tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_mean_mape_vs_cpu_p95.png", dpi=180)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    candidates = [row for row in rows if math.isfinite(as_float(row.get("mean_mape")))]
    candidates = sorted(candidates, key=lambda r: (r.get("track", ""), as_float(r.get("mean_mape"))))[:40]
    if not candidates:
        return
    matrix = np.asarray([[metric_mape(row, name) for name in TARGET_NAMES] for row in candidates], dtype=float)
    if not np.isfinite(matrix).any():
        return
    fig, ax = plt.subplots(figsize=(8.5, max(5.5, 0.35 * len(candidates))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(TARGET_NAMES)))
    ax.set_xticklabels(TARGET_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(candidates)))
    ax.set_yticklabels([row_label(row) for row in candidates], fontsize=7)
    ax.set_title(f"{title}: per-metric MAPE heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("MAPE (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_metric_mape_heatmap.png", dpi=180)
    plt.close(fig)


def plot_size_latency(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    candidates = [
        row
        for row in rows
        if math.isfinite(as_float(row.get("artifact_size_mb")))
        and as_float(row.get("artifact_size_mb")) > 0
        and math.isfinite(as_float(row.get("latency_forward_ms_p95")))
    ]
    if not candidates:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in candidates:
        ax.scatter(as_float(row.get("artifact_size_mb")), as_float(row.get("latency_forward_ms_p95")), s=80, alpha=0.85)
        ax.annotate(row_label(row), (as_float(row.get("artifact_size_mb")), as_float(row.get("latency_forward_ms_p95"))), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Artifact size (MB)")
    ax.set_ylabel("CPU p95 forward latency per graph (ms)")
    ax.set_title(f"{title}: size/latency tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "artifact_size_vs_latency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results.exists():
        raise SystemExit(f"missing results file: {results}")
    rows = load_rows(results)
    if not rows:
        raise SystemExit(f"no evaluation rows found in {results}")
    write_summary_csv(rows, out_dir / "summary.csv")
    plot_accuracy(rows, out_dir, args.title)
    plot_cpu_latency(rows, out_dir, args.title)
    plot_pareto(rows, out_dir, args.title)
    plot_heatmap(rows, out_dir, args.title)
    plot_size_latency(rows, out_dir, args.title)
    print(f"wrote plots and summary CSV to {out_dir}")


if __name__ == "__main__":
    main()
