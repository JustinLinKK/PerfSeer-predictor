"""Measure exact dispatcher operation time for the local supported v3 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json, sha256_bytes
from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.coverage import smallest_time_vocabulary
from perfseer_v3.coverage_corpus import (
    CoverageCase,
    p0_cases,
    representative_source_cases,
)
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.profiling import (
    ProfileOptions,
    ProfileWorkload,
    profile_workload,
)
from perfseer_v3.workloads import (
    WorkloadDescriptor,
    build_workload,
    default_composites,
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


def _requires_grad(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.detach().requires_grad_(True)
    if isinstance(value, tuple):
        return tuple(_requires_grad(item) for item in value)
    if isinstance(value, list):
        return [_requires_grad(item) for item in value]
    if isinstance(value, dict):
        return {key: _requires_grad(item) for key, item in value.items()}
    return value


def _clear_grad(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        value.grad = None
    elif isinstance(value, (tuple, list)):
        for item in value:
            _clear_grad(item)
    elif isinstance(value, dict):
        for item in value.values():
            _clear_grad(item)


def _floating_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.is_floating_point() else []
    if isinstance(value, (tuple, list)):
        return [
            tensor
            for item in value
            for tensor in _floating_tensors(item)
        ]
    if isinstance(value, dict):
        return [
            tensor
            for key in sorted(value, key=str)
            for tensor in _floating_tensors(value[key])
        ]
    return []


def _generic_loss(output: Any, _target: torch.Tensor) -> torch.Tensor:
    tensors = _floating_tensors(output)
    if not tensors:
        raise ValueError("gradient-only workload produced no floating tensor")
    return sum(tensor.float().square().mean() for tensor in tensors)


def _has_cuda_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, (tuple, list)):
        return any(_has_cuda_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_has_cuda_tensor(item) for item in value.values())
    return False


class _CudaDispatchTimer(TorchDispatchMode):
    """Record CUDA event pairs around exact ``OpOverload`` dispatches."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        if _has_cuda_tensor(args) or _has_cuda_tensor(kwargs) or _has_cuda_tensor(result):
            try:
                name = str(func.name())
            except (AttributeError, TypeError):
                name = str(func)
            self.events.append((name, start, end))
        return result

    def operation_time_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals: Counter[str] = Counter()
        for operation, start, end in self.events:
            elapsed = float(start.elapsed_time(end))
            if elapsed > 0:
                totals[operation] += elapsed
        return dict(sorted(totals.items()))


def _nvml_sampler(device_index: int) -> tuple[
    Callable[[], dict[str, float]] | None,
    Callable[[], None],
]:
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
                    pynvml.nvmlDeviceGetClockInfo(
                        handle,
                        pynvml.NVML_CLOCK_SM,
                    )
                ),
                "power_usage_watts": (
                    float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                ),
            }

        return sample, pynvml.nvmlShutdown
    except Exception:
        return None, lambda: None


def _case_entries() -> tuple[tuple[str, str, CoverageCase | WorkloadDescriptor], ...]:
    coverage = (
        ("p0", case.case_id, case)
        for case in p0_cases()
    )
    real = (
        ("real", case.case_id, case)
        for case in representative_source_cases()
    )
    composites = (
        ("composite", descriptor.workload_id, descriptor)
        for descriptor in default_composites()
    )
    return tuple((*coverage, *real, *composites))


def _build_entry(
    entry: CoverageCase | WorkloadDescriptor,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    tuple[Any, ...],
    dict[str, Any],
    bool,
    tuple[str, ...],
]:
    if isinstance(entry, CoverageCase):
        model, args, kwargs = entry.build()
        training = entry.training
        declared_operations: tuple[str, ...] = ()
    else:
        instance = build_workload(entry)
        model, args, kwargs = instance.model, instance.args, instance.kwargs
        training = entry.phase == "training"
        declared_operations = entry.declared_operations
    model = model.to(device)
    model.train(training)
    moved_args = _to_device(args, device)
    moved_kwargs = _to_device(kwargs, device)
    if training:
        moved_args = _requires_grad(moved_args)
        moved_kwargs = _requires_grad(moved_kwargs)
    return model, tuple(moved_args), dict(moved_kwargs), training, declared_operations


def _execute_timed(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    training: bool,
) -> None:
    if not training:
        with torch.no_grad():
            model(*args, **kwargs)
        return
    model.zero_grad(set_to_none=True)
    _clear_grad(args)
    _clear_grad(kwargs)
    loss = _generic_loss(model(*args, **kwargs), torch.zeros((), device="cuda"))
    loss.backward()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_supported_corpus_gpu_time.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_supported_corpus_profiles.jsonl",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--operator-repeats", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="profile only the first N entries; zero profiles the full corpus",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("supported-corpus GPU-time profiling requires CUDA")
    if args.operator_repeats < 1 or args.limit < 0:
        raise ValueError("operator repeats must be positive and limit cannot be negative")

    entries = _case_entries()
    if args.limit:
        entries = entries[: args.limit]
    device = torch.device("cuda")
    device_index = torch.cuda.current_device()
    registry = OperationRegistry.load()
    sampler, shutdown_nvml = _nvml_sampler(device_index)
    cumulative: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    environment: dict[str, Any] = {}
    started_at = datetime.now(UTC).isoformat()
    try:
        for index, (layer, entry_id, entry) in enumerate(entries):
            torch.manual_seed(index)
            torch.cuda.manual_seed_all(index)
            model, model_args, model_kwargs, training, declared = _build_entry(
                entry,
                device,
            )
            if training:
                target = torch.zeros((), device=device)
                capture = capture_training_graph(
                    model,
                    model_args,
                    kwargs=model_kwargs,
                    target=target,
                    loss_fn=_generic_loss,
                    optimizer_name="none",
                    registry=registry,
                )
                workload = ProfileWorkload(
                    model,
                    model_args,
                    model_kwargs,
                    capture,
                    mode="training",
                    loss_fn=_generic_loss,
                    target=target,
                )
            else:
                capture = capture_export(
                    model,
                    model_args,
                    model_kwargs,
                    registry=registry,
                    options=CaptureOptions(precision="float32"),
                )
                workload = ProfileWorkload(
                    model,
                    model_args,
                    model_kwargs,
                    capture,
                )
            if not capture.success or capture.graph is None:
                raise RuntimeError(
                    f"capture failed for {entry_id}: {capture.failures}"
                )
            if declared:
                validate_declared_operations(capture.graph, entry)
            record = profile_workload(
                workload,
                options=ProfileOptions(
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    nvml_sample_interval_s=0.02,
                ),
                nvml_sampler=sampler,
                metadata={
                    "entry_id": entry_id,
                    "data_layer": layer,
                    "operator_timer": "torch_dispatch_cuda_event_pairs",
                },
            )
            if record.status != "ok":
                raise RuntimeError(
                    f"profile failed for {entry_id}: "
                    f"{record.failure_stage} {record.error_message}"
                )
            environment = environment or record.environment
            timer = _CudaDispatchTimer()
            with timer:
                for _ in range(args.operator_repeats):
                    _execute_timed(
                        model,
                        model_args,
                        model_kwargs,
                        training=training,
                    )
            operation_times = timer.operation_time_ms()
            cumulative.update(operation_times)
            graph_operations = sorted(
                {node.raw_target for node in capture.graph.nodes}
            )
            rows.append(
                {
                    "entry_id": entry_id,
                    "data_layer": layer,
                    "training": training,
                    "capture_quality": capture.graph.coverage.capture_quality,
                    "backward_capture_quality": (
                        capture.graph.coverage.backward_capture_quality
                    ),
                    "graph_sha256": capture.graph.graph_sha256,
                    "profile_record_sha256": record.record_sha256,
                    "measured_step_ms": list(record.measured_step_ms),
                    "operator_time_ms": operation_times,
                    "captured_operations": graph_operations,
                    "timed_operations": len(operation_times),
                }
            )
            raw_lines.append(json.dumps(record.to_dict(), sort_keys=True))
    finally:
        shutdown_nvml()

    report: dict[str, Any] = {
        "report_version": "perfseer_v3_supported_corpus_gpu_time_v1",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "scope": "local_p0_representative_real_and_composite_corpus",
        "operator_registry_sha256": registry.sha256,
        "method": {
            "operator_identity": "TorchDispatchMode OpOverload.name()",
            "operator_time": "per-dispatch CUDA event elapsed time",
            "units": "milliseconds",
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "operator_repeats": args.operator_repeats,
        },
        "training_approval": False,
        "training_approval_reason": (
            "This local diagnostic corpus has no production scheduler labels, "
            "limited real-model breadth, and no authenticated Nautilus target-"
            "hardware run; it can propose but not approve a production vocabulary."
        ),
        "environment": environment,
        "profiled_entries": len(rows),
        "data_layer_counts": dict(
            sorted(Counter(row["data_layer"] for row in rows).items())
        ),
        "profile_rows": rows,
        "profiler_time_by_operation": dict(sorted(cumulative.items())),
        "recommended_exact_vocabulary_95pct_gpu_time": list(
            smallest_time_vocabulary(cumulative)
        ),
    }
    report["report_sha256"] = sha256_bytes(
        canonical_json(report).encode("utf-8")
    )
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
        f"vocabulary95={len(report['recommended_exact_vocabulary_95pct_gpu_time'])} "
        f"report_sha256={report['report_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
