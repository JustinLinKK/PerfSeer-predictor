#!/usr/bin/env python
"""Create analysis plots from the optimized PerfSeer result ledger."""

from __future__ import annotations

import argparse
import csv
import html
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


def pareto_flags(points: list[dict[str, Any]]) -> list[bool]:
    flags = []
    for idx, point in enumerate(points):
        dominated = False
        for other_idx, other in enumerate(points):
            if idx == other_idx:
                continue
            smaller_or_equal = other["size_mb"] <= point["size_mb"]
            faster_or_equal = other["graphs_per_sec"] >= point["graphs_per_sec"]
            more_accurate_or_equal = other["mean_mape"] <= point["mean_mape"]
            strictly_better = (
                other["size_mb"] < point["size_mb"]
                or other["graphs_per_sec"] > point["graphs_per_sec"]
                or other["mean_mape"] < point["mean_mape"]
            )
            if smaller_or_equal and faster_or_equal and more_accurate_or_equal and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def plot_interactive_tradeoff(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    candidates = [
        row
        for row in rows
        if row.get("event") == "eval_deploy_complete"
        and math.isfinite(as_float(row.get("artifact_size_mb")))
        and as_float(row.get("artifact_size_mb")) > 0
        and math.isfinite(as_float(row.get("graphs_per_sec")))
        and as_float(row.get("graphs_per_sec")) > 0
        and math.isfinite(as_float(row.get("mean_mape")))
    ]
    if not candidates:
        return

    points: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda r: (str(r.get("run_id", "")), backend(r))):
        run_id = str(row.get("run_id", ""))
        b = backend(row)
        points.append(
            {
                "run_id": run_id,
                "backend": b,
                "label": f"{run_id} / {b}",
                "track": str(row.get("track", "cpu_deploy")),
                "size_mb": as_float(row.get("artifact_size_mb")),
                "graphs_per_sec": as_float(row.get("graphs_per_sec")),
                "p50_ms": as_float(row.get("latency_forward_ms_p50")),
                "p95_ms": as_float(row.get("latency_forward_ms_p95")),
                "mean_mape": as_float(row.get("mean_mape")),
                "model_params": int(as_float(row.get("model_params", row.get("params", 0))) or 0),
            }
        )

    for point, is_pareto in zip(points, pareto_flags(points)):
        point["pareto"] = is_pareto

    payload = json.dumps(points, ensure_ascii=False, allow_nan=False)
    safe_title = html.escape(title)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title} CPU Trade-Off 3D</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f5;
      --ink: #172027;
      --muted: #62717a;
      --line: #cfd7d9;
      --panel: rgba(255, 255, 255, 0.88);
      --accent: #2f6f73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 14px 18px 10px;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 248, 245, 0.94);
    }}
    h1 {{
      flex: 1;
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button, select {{
      min-height: 34px;
      border: 1px solid #b7c1c5;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    button {{
      width: 38px;
      display: inline-grid;
      place-items: center;
      cursor: pointer;
    }}
    select {{ padding: 0 28px 0 10px; }}
    main {{
      position: relative;
      min-height: 0;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
      touch-action: none;
    }}
    canvas.dragging {{ cursor: grabbing; }}
    .legend {{
      position: absolute;
      left: 16px;
      top: 16px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      max-width: min(920px, calc(100vw - 32px));
      padding: 8px;
      border: 1px solid rgba(186, 197, 200, 0.9);
      border-radius: 8px;
      background: var(--panel);
      backdrop-filter: blur(6px);
    }}
    .legend label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      padding: 0 7px;
      border: 1px solid #d7dedf;
      border-radius: 6px;
      background: #fff;
      font-size: 12px;
      white-space: nowrap;
      cursor: pointer;
    }}
    .legend input {{ margin: 0; }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      border: 1px solid rgba(0, 0, 0, 0.18);
    }}
    .readout {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      width: min(440px, calc(100vw - 32px));
      padding: 12px 14px;
      border: 1px solid rgba(186, 197, 200, 0.9);
      border-radius: 8px;
      background: var(--panel);
      backdrop-filter: blur(6px);
      font-size: 12px;
      color: var(--muted);
    }}
    .readout strong {{
      color: var(--ink);
      font-weight: 700;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      max-width: 360px;
      padding: 10px 11px;
      border: 1px solid #b7c1c5;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 14px 34px rgba(18, 30, 36, 0.16);
      color: var(--ink);
      font-size: 12px;
      line-height: 1.45;
      opacity: 0;
      transform: translate(10px, 10px);
      transition: opacity 120ms ease;
      white-space: normal;
    }}
    .tooltip.visible {{ opacity: 1; }}
    .tooltip .name {{ font-weight: 700; margin-bottom: 4px; }}
    .tooltip .pareto {{ color: var(--accent); font-weight: 700; }}
    @media (max-width: 720px) {{
      header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      h1 {{ font-size: 16px; }}
      .toolbar {{ justify-content: flex-start; }}
      .legend {{
        top: 12px;
        left: 12px;
        right: 12px;
        max-height: 116px;
        overflow: auto;
      }}
      .readout {{
        left: 12px;
        right: 12px;
        bottom: 12px;
        width: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>{safe_title}: CPU Model Size, Speed, and MAPE</h1>
      <div class="toolbar">
        <select id="viewMode" aria-label="Color mode">
          <option value="backend">Backend</option>
          <option value="family">Model Family</option>
          <option value="pareto">Pareto</option>
        </select>
        <button id="resetView" title="Reset view" aria-label="Reset view">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M7 7h5a6 6 0 1 1-5.2 9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M7 3v4h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        <button id="togglePareto" title="Highlight Pareto frontier" aria-label="Highlight Pareto frontier">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M12 3 21 12 12 21 3 12Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    </header>
    <main>
      <canvas id="tradeoff" aria-label="Interactive 3D CPU deployment trade-off plot"></canvas>
      <div id="legend" class="legend"></div>
      <div class="readout">
        <strong>Axes:</strong> model artifact size MB, CPU throughput graphs/s, CPU mean MAPE %. Lower size and MAPE are better; higher throughput is better.
      </div>
      <div id="tooltip" class="tooltip"></div>
    </main>
  </div>
  <script>
    const rawPoints = {payload};
    const palette = {{
      pytorch: "#2f6f73",
      pytorch_dynamic_int8: "#b65f24",
      torchscript: "#4f7f2a",
      onnxruntime: "#8f4e8b",
      onnxruntime_int8: "#4869a9",
      openvino: "#c2872a",
      openvino_int8: "#6f5b4b",
      accuracy: "#4869a9",
      deploy: "#2f6f73",
      pareto: "#138a64",
      dominated: "#8d989b"
    }};

    const canvas = document.getElementById("tradeoff");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const legend = document.getElementById("legend");
    const viewMode = document.getElementById("viewMode");
    const resetView = document.getElementById("resetView");
    const togglePareto = document.getElementById("togglePareto");

    let yaw = -0.72;
    let pitch = -0.55;
    let zoom = 1;
    let highlightPareto = true;
    let activeBackends = new Set(rawPoints.map(p => p.backend));
    let projected = [];
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let hoverPoint = null;

    const log10 = value => Math.log(value) / Math.LN10;
    const extents = {{
      size: extent(rawPoints.map(p => log10(p.size_mb))),
      speed: extent(rawPoints.map(p => log10(p.graphs_per_sec))),
      mape: extent(rawPoints.map(p => p.mean_mape))
    }};

    function extent(values) {{
      const finite = values.filter(Number.isFinite);
      let min = Math.min(...finite);
      let max = Math.max(...finite);
      if (min === max) {{
        min -= 1;
        max += 1;
      }}
      return [min, max];
    }}

    function normalize(value, [min, max], invert = false) {{
      const t = (value - min) / (max - min);
      const n = (invert ? 1 - t : t) * 2 - 1;
      return Math.max(-1, Math.min(1, n));
    }}

    function family(point) {{
      if (point.run_id.includes("distill")) return "distilled";
      if (point.run_id.includes("shared_multitask")) return "multitask";
      return "accuracy";
    }}

    function colorFor(point) {{
      if (viewMode.value === "pareto") return point.pareto ? palette.pareto : palette.dominated;
      if (viewMode.value === "family") {{
        const f = family(point);
        return f === "distilled" ? "#b65f24" : f === "multitask" ? "#2f6f73" : "#4869a9";
      }}
      return palette[point.backend] || "#6f5b4b";
    }}

    function world(point) {{
      return {{
        x: normalize(log10(point.size_mb), extents.size),
        y: normalize(log10(point.graphs_per_sec), extents.speed),
        z: normalize(point.mean_mape, extents.mape)
      }};
    }}

    function rotate(p) {{
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * p.x - sy * p.y;
      const y1 = sy * p.x + cy * p.y;
      const z1 = p.z;
      return {{
        x: x1,
        y: cp * y1 - sp * z1,
        z: sp * y1 + cp * z1
      }};
    }}

    function project(p) {{
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width, rect.height) * 0.31 * zoom;
      const r = rotate(p);
      return {{
        sx: rect.width * 0.52 + r.x * scale,
        sy: rect.height * 0.53 - r.z * scale,
        depth: r.y,
        scale
      }};
    }}

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function drawLine(a, b, color = "#8c999d", width = 1) {{
      const pa = project(a);
      const pb = project(b);
      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.lineTo(pb.sx, pb.sy);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    }}

    function drawLabel(text, p, offsetX, offsetY) {{
      const pp = project(p);
      ctx.fillStyle = "#3f4d55";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(text, pp.sx + offsetX, pp.sy + offsetY);
    }}

    function drawAxes() {{
      const corners = [
        {{x:-1,y:-1,z:-1}}, {{x:1,y:-1,z:-1}}, {{x:-1,y:1,z:-1}}, {{x:1,y:1,z:-1}},
        {{x:-1,y:-1,z:1}}, {{x:1,y:-1,z:1}}, {{x:-1,y:1,z:1}}, {{x:1,y:1,z:1}}
      ];
      const edges = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      for (const [a, b] of edges) drawLine(corners[a], corners[b], "#d0d8da", 1);
      drawLine({{x:-1,y:-1,z:-1}}, {{x:1,y:-1,z:-1}}, "#54646b", 2);
      drawLine({{x:-1,y:-1,z:-1}}, {{x:-1,y:1,z:-1}}, "#54646b", 2);
      drawLine({{x:-1,y:-1,z:-1}}, {{x:-1,y:-1,z:1}}, "#54646b", 2);
      drawLabel("Size MB", {{x:1,y:-1,z:-1}}, 8, 4);
      drawLabel("CPU speed", {{x:-1,y:1,z:-1}}, 8, 4);
      drawLabel("MAPE", {{x:-1,y:-1,z:1}}, 8, 4);
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#f7f8f5";
      ctx.fillRect(0, 0, rect.width, rect.height);
      drawAxes();

      const visible = rawPoints.filter(p => activeBackends.has(p.backend));
      projected = visible.map(point => {{
        const pos = project(world(point));
        return {{...point, ...pos}};
      }}).sort((a, b) => a.depth - b.depth);

      for (const point of projected) {{
        const radius = point.pareto && highlightPareto ? 8 : 6;
        ctx.beginPath();
        ctx.arc(point.sx, point.sy, radius + (point.pareto && highlightPareto ? 4 : 0), 0, Math.PI * 2);
        ctx.fillStyle = point.pareto && highlightPareto ? "rgba(19, 138, 100, 0.16)" : "rgba(255, 255, 255, 0.28)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(point.sx, point.sy, radius, 0, Math.PI * 2);
        ctx.fillStyle = colorFor(point);
        ctx.fill();
        ctx.lineWidth = point === hoverPoint ? 3 : 1.2;
        ctx.strokeStyle = point === hoverPoint ? "#172027" : "rgba(255, 255, 255, 0.92)";
        ctx.stroke();
      }}
    }}

    function updateLegend() {{
      const backends = Array.from(new Set(rawPoints.map(p => p.backend))).sort();
      legend.innerHTML = "";
      for (const backend of backends) {{
        const item = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = activeBackends.has(backend);
        input.addEventListener("change", () => {{
          if (input.checked) activeBackends.add(backend);
          else activeBackends.delete(backend);
          draw();
        }});
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = palette[backend] || "#6f5b4b";
        const text = document.createElement("span");
        text.textContent = backend;
        item.append(input, swatch, text);
        legend.appendChild(item);
      }}
    }}

    function nearestPoint(x, y) {{
      let best = null;
      let bestDistance = Infinity;
      for (const point of projected) {{
        const dx = point.sx - x;
        const dy = point.sy - y;
        const distance = Math.hypot(dx, dy);
        if (distance < bestDistance) {{
          best = point;
          bestDistance = distance;
        }}
      }}
      return bestDistance <= 18 ? best : null;
    }}

    function moveTooltip(event) {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      hoverPoint = nearestPoint(x, y);
      if (!hoverPoint) {{
        tooltip.classList.remove("visible");
        draw();
        return;
      }}
      tooltip.innerHTML = `
        <div class="name">${{hoverPoint.label}}</div>
        <div>Size: <strong>${{hoverPoint.size_mb.toFixed(2)}} MB</strong></div>
        <div>CPU speed: <strong>${{hoverPoint.graphs_per_sec.toFixed(1)}} graphs/s</strong></div>
        <div>CPU p50/p95: <strong>${{hoverPoint.p50_ms.toFixed(3)}} / ${{hoverPoint.p95_ms.toFixed(3)}} ms</strong></div>
        <div>CPU mean MAPE: <strong>${{hoverPoint.mean_mape.toFixed(3)}}%</strong></div>
        <div>Parameters: <strong>${{hoverPoint.model_params.toLocaleString()}}</strong></div>
        ${{hoverPoint.pareto ? '<div class="pareto">Pareto frontier</div>' : ''}}
      `;
      const left = Math.min(rect.width - 380, Math.max(8, x + 14));
      const top = Math.min(rect.height - 190, Math.max(8, y + 14));
      tooltip.style.left = `${{left}}px`;
      tooltip.style.top = `${{top}}px`;
      tooltip.classList.add("visible");
      draw();
    }}

    canvas.addEventListener("pointerdown", event => {{
      dragging = true;
      canvas.classList.add("dragging");
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointermove", event => {{
      if (dragging) {{
        yaw += (event.clientX - lastX) * 0.008;
        pitch += (event.clientY - lastY) * 0.008;
        pitch = Math.max(-1.25, Math.min(1.25, pitch));
        lastX = event.clientX;
        lastY = event.clientY;
        tooltip.classList.remove("visible");
        draw();
      }} else {{
        moveTooltip(event);
      }}
    }});
    canvas.addEventListener("pointerup", event => {{
      dragging = false;
      canvas.classList.remove("dragging");
      canvas.releasePointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointerleave", () => {{
      dragging = false;
      hoverPoint = null;
      tooltip.classList.remove("visible");
      canvas.classList.remove("dragging");
      draw();
    }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      zoom *= event.deltaY < 0 ? 1.08 : 0.92;
      zoom = Math.max(0.65, Math.min(2.4, zoom));
      draw();
    }}, {{passive: false}});
    viewMode.addEventListener("change", draw);
    resetView.addEventListener("click", () => {{
      yaw = -0.72;
      pitch = -0.55;
      zoom = 1;
      draw();
    }});
    togglePareto.addEventListener("click", () => {{
      highlightPareto = !highlightPareto;
      draw();
    }});
    window.addEventListener("resize", resize);
    updateLegend();
    resize();
  </script>
</body>
</html>
"""
    (out_dir / "tradeoff_3d_cpu_size_speed_mape.html").write_text(html_doc)


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
    plot_interactive_tradeoff(rows, out_dir, args.title)
    print(f"wrote plots and summary CSV to {out_dir}")


if __name__ == "__main__":
    main()
