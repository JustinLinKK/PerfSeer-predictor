"""Profile the exact bootstrap operations on an allocated CUDA device."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json, sha256_bytes
from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.coverage import smallest_time_vocabulary
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.profiling import ProfileOptions, ProfileWorkload, profile_workload
from perfseer_v3.workloads import (
    SHAPE_REGIMES,
    build_workload,
    default_microbenchmarks,
    validate_declared_operations,
)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _nvml_sampler(device_index: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        def sample() -> dict[str, float]:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "sm_util_percent": float(utilization.gpu),
                "memory_controller_util_percent": float(utilization.memory),
                "device_memory_used_bytes": float(memory.used),
                "sm_clock_mhz": float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                ),
                "power_usage_watts": float(
                    pynvml.nvmlDeviceGetPowerUsage(handle)
                )
                / 1000.0,
            }

        return sample, pynvml.nvmlShutdown
    except Exception:
        return None, lambda: None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_microbenchmark_gpu_time.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_microbenchmark_profiles.jsonl",
    )
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument(
        "--shape-regimes",
        default=",".join(SHAPE_REGIMES),
        help="comma-separated subset of tiny,small,medium,large,boundary",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("microbenchmark GPU-time profiling requires CUDA")
    regimes = tuple(part.strip() for part in args.shape_regimes.split(",") if part.strip())
    invalid = set(regimes) - set(SHAPE_REGIMES)
    if invalid:
        raise ValueError(f"unknown shape regimes: {sorted(invalid)}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    device = torch.device("cuda")
    device_index = torch.cuda.current_device()
    sampler, shutdown_nvml = _nvml_sampler(device_index)
    descriptors = tuple(
        row
        for row in default_microbenchmarks(
            dtypes=(args.dtype,),
            phases=("forward",),
            batch_sizes=(args.batch_size,),
        )
        if row.shape_regime in regimes
    )
    registry = OperationRegistry.load()
    rows = []
    cumulative: dict[str, float] = {}
    raw_lines = []
    started_at = datetime.now(UTC).isoformat()
    try:
        for index, descriptor in enumerate(descriptors):
            torch.manual_seed(index)
            instance = build_workload(descriptor)
            model = instance.model.to(device).eval()
            profile_args = _to_device(instance.args, device)
            profile_kwargs = _to_device(instance.kwargs, device)
            capture = capture_export(
                model,
                profile_args,
                profile_kwargs,
                registry=registry,
                options=CaptureOptions(precision=args.dtype),
            )
            if not capture.success or capture.graph is None:
                raise RuntimeError(
                    f"capture failed for {descriptor.workload_id}: {capture.failures}"
                )
            validate_declared_operations(capture.graph, descriptor)
            record = profile_workload(
                ProfileWorkload(
                    model,
                    profile_args,
                    profile_kwargs,
                    capture,
                    precision=args.dtype,
                ),
                options=ProfileOptions(
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    nvml_sample_interval_s=0.02,
                ),
                nvml_sampler=sampler,
                metadata={
                    "workload_id": descriptor.workload_id,
                    "shape_regime": descriptor.shape_regime,
                    "declared_operation": descriptor.declared_operations[0],
                },
            )
            if record.status != "ok":
                raise RuntimeError(
                    f"profile failed for {descriptor.workload_id}: "
                    f"{record.failure_stage} {record.error_message}"
                )
            operation = descriptor.declared_operations[0]
            mean_ms = statistics.fmean(record.measured_step_ms)
            cumulative[operation] = cumulative.get(operation, 0.0) + mean_ms
            rows.append(
                {
                    "workload_id": descriptor.workload_id,
                    "operation": operation,
                    "shape_regime": descriptor.shape_regime,
                    "dtype": descriptor.dtype,
                    "batch_size": descriptor.batch_size,
                    "mean_step_ms": mean_ms,
                    "measured_step_ms": list(record.measured_step_ms),
                    "profile_record_sha256": record.record_sha256,
                    "graph_sha256": capture.graph.graph_sha256,
                }
            )
            raw_lines.append(json.dumps(record.to_dict(), sort_keys=True))
    finally:
        shutdown_nvml()

    report: dict[str, Any] = {
        "report_version": "perfseer_v3_microbenchmark_gpu_time_v1",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "scope": "bootstrap_exact_microbenchmarks",
        "training_approval": False,
        "training_approval_reason": (
            "Microbenchmarks provide measured coverage evidence but do not replace "
            "the required composite and real-corpus profiler-time distribution."
        ),
        "hardware": {
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": ".".join(
                str(value)
                for value in torch.cuda.get_device_capability(device_index)
            ),
            "total_memory_bytes": torch.cuda.get_device_properties(device_index).total_memory,
        },
        "software": {
            "pytorch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "operator_registry_sha256": registry.sha256,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "shape_regimes": list(regimes),
        "profile_rows": rows,
        "profiler_time_by_operation": dict(sorted(cumulative.items())),
        "recommended_exact_vocabulary_95pct_gpu_time": list(
            smallest_time_vocabulary(cumulative)
        ),
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report).encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.raw_output.write_text(
        "\n".join(raw_lines) + ("\n" if raw_lines else ""),
        encoding="utf-8",
    )
    print(
        f"profiles={len(rows)} operations={len(cumulative)} "
        f"report_sha256={report['report_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
