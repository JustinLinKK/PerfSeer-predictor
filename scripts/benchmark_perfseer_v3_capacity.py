#!/usr/bin/env python3
"""Emit reproducible v3 capacity counts and optional CPU student benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.capacity import (
    DEFAULT_CAPACITY_STUDY_PATH,
    exact_parameter_count,
    load_capacity_study,
)
from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.features import batch_graph_features, build_graph_features
from perfseer_v3.model import SeerNetV3, graph_batch_tensors
from perfseer_v3.op_registry import OperationRegistry


DEFAULT_OUTPUT = ROOT / "reports" / "perfseer_v3_capacity_study.json"


class _CapacityProbe(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(16, 16) for _ in range(depth))
        self.norm = nn.LayerNorm(16)
        self.output = nn.Linear(16, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + F.gelu(layer(x))
        return self.output(self.norm(x).mean(dim=1))


def _probe_batch(depth: int) -> Any:
    torch.manual_seed(1000 + depth)
    model = _CapacityProbe(depth).train()
    args = (torch.randn(2, 8, 16),)
    target = torch.randn(2, 6)
    capture = capture_training_graph(
        model,
        args,
        target=target,
        loss_fn=F.mse_loss,
        optimizer_name="adamw",
    )
    if not capture.success or capture.graph is None:
        raise RuntimeError(f"capacity probe capture failed: {capture.failures}")
    features = build_graph_features(capture.graph)
    return batch_graph_features([features])


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _benchmark(
    model: SeerNetV3,
    batches: dict[str, Any],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], int]:
    scripted = torch.jit.script(model.eval())
    with tempfile.TemporaryDirectory(prefix="perfseer-v3-capacity-") as temporary:
        artifact_path = Path(temporary) / "student.torchscript.pt"
        scripted.save(str(artifact_path))
        artifact_bytes = artifact_path.stat().st_size
    bucket_results: dict[str, Any] = {}
    with torch.inference_mode():
        for bucket, batch in batches.items():
            tensors = graph_batch_tensors(batch)
            for _ in range(warmup):
                scripted(*tensors)
            durations = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                output = scripted(*tensors)
                duration = (time.perf_counter_ns() - start) / 1_000_000.0
                if output.prediction.shape != (1, 6):
                    raise RuntimeError("capacity probe changed the six-output contract")
                durations.append(duration)
            bucket_results[bucket] = {
                "iterations": iterations,
                "p50_cpu_latency_ms": statistics.median(durations),
                "p95_cpu_latency_ms": _percentile(durations, 0.95),
                "minimum_cpu_latency_ms": min(durations),
                "maximum_cpu_latency_ms": max(durations),
            }
    return bucket_results, artifact_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CAPACITY_STUDY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-students", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 2 or args.cpu_threads < 1:
        parser.error("warmup must be nonnegative, iterations >= 2, and cpu-threads >= 1")
    torch.set_num_threads(args.cpu_threads)
    registry = OperationRegistry.load()
    batches = {"small": _probe_batch(1)}
    if args.benchmark_students:
        batches.update({"median": _probe_batch(4), "large": _probe_batch(12)})
    layout = batches["small"].layout
    study = load_capacity_study(args.config)
    rows = []
    for candidate in study.candidates:
        config = candidate.model_config(registry, layout)
        parameter_count = exact_parameter_count(config)
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "role": candidate.role,
            "purpose": candidate.purpose,
            "hidden": config.hidden,
            "num_blocks": config.num_blocks,
            "pooling_mode": config.pooling_mode,
            "node_identity_fusion": config.node_identity_fusion,
            "trainable_parameter_count": parameter_count,
            "fp32_parameter_payload_bytes": parameter_count * 4,
            "validation_metrics": None,
            "calibration_metrics": None,
            "peak_predictor_training_memory_bytes": None,
            "predictor_training_time_seconds": None,
            "cpu_latency_by_graph_bucket": None,
            "torchscript_artifact_bytes": None,
        }
        if args.benchmark_students and candidate.role == "student":
            torch.manual_seed(77)
            model = SeerNetV3(config).cpu().eval()
            latency, artifact_bytes = _benchmark(
                model,
                batches,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            row["cpu_latency_by_graph_bucket"] = latency
            row["torchscript_artifact_bytes"] = artifact_bytes
        rows.append(row)
    t0 = next(row for row in rows if row["candidate_id"] == "T0")
    t2 = next(row for row in rows if row["candidate_id"] == "T2")
    if t2["trainable_parameter_count"] > 3 * t0["trainable_parameter_count"]:
        raise RuntimeError("T2 exceeds the three-times-T0 parameter guardrail")
    payload: dict[str, Any] = {
        "report_version": "perfseer_v3_capacity_report_v1",
        "capacity_study_sha256": study.sha256,
        "feature_schema_sha256": layout.feature_schema_sha256,
        "operator_registry_sha256": registry.sha256,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cpu_threads": args.cpu_threads,
        "benchmark_students": args.benchmark_students,
        "student_benchmark_protocol": {
            "execution": "torchscript_cpu_inference",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "graph_buckets": {"small": 1, "median": 4, "large": 12},
        },
        "candidates": rows,
        "deployment_gates": study.deployment_gates,
        "selection_policy": study.selection,
        "production_measurement_status": {
            "available": False,
            "missing": [
                "grouped production scheduler-label corpus",
                "family-held-out validation predictions",
                "predictor-training peak memory and duration",
                "matched v2 graph-bucket latency runner",
                "production calibration and OOM evaluation",
            ],
        },
    }
    payload["report_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"candidates={len(rows)} benchmarked_students="
        f"{sum(row['cpu_latency_by_graph_bucket'] is not None for row in rows)} "
        f"sha256={payload['report_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
