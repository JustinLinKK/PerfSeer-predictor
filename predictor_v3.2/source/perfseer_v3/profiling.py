"""Source-first profiling with graph/callable identity enforcement."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from .baseline import canonical_json
from .capture_export import CaptureResult, _model_fingerprint
from .inputs import input_signature
from .tensor_metadata import clone_inputs, compare_output_pytrees


class WorkloadIdentityError(ValueError):
    """Raised before profiling if graph and executable identity differ."""


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(f"{type(value).__name__}:{value!r}".encode("utf-8"))


def input_value_fingerprint(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, args)
    _hash_value(digest, dict(kwargs))
    return digest.hexdigest()


@dataclass(frozen=True)
class ProfileOptions:
    warmup_steps: int = 3
    measured_steps: int = 10
    gradient_accumulation_steps: int = 1
    execution_mode: str = "eager"
    compile_backend: str | None = None
    compile_warmup_excluded: bool = True
    correctness_rtol: float = 1e-4
    correctness_atol: float = 1e-5
    trace_path: str | None = None
    nvml_sample_interval_s: float = 0.05

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.measured_steps < 1:
            raise ValueError("warmup_steps must be >= 0 and measured_steps must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.execution_mode not in {"eager", "compile"}:
            raise ValueError("execution_mode must be 'eager' or 'compile'")
        if (
            self.execution_mode == "compile"
            and self.compile_warmup_excluded
            and self.warmup_steps < 1
        ):
            raise ValueError("compiled profiling needs a warmup step when compile time is excluded")
        if self.nvml_sample_interval_s <= 0:
            raise ValueError("nvml_sample_interval_s must be positive")


@dataclass
class ProfileWorkload:
    model: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    capture: Any
    mode: str = "inference"
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None
    target: Any | None = None
    optimizer: torch.optim.Optimizer | None = None
    precision: str = "float32"

    def validate_contract(self) -> None:
        if self.mode not in {"inference", "training"}:
            raise WorkloadIdentityError(f"unsupported profile mode {self.mode!r}")
        if not self.capture.success or self.capture.graph is None:
            raise WorkloadIdentityError("profiling requires a successful capture result")
        graph = self.capture.graph
        if self.capture.model_object_id != id(self.model):
            raise WorkloadIdentityError(
                "profile model is not the exact model instance used for graph capture"
            )
        descriptor = f"{self.model.__class__.__module__}.{self.model.__class__.__qualname__}"
        if self.capture.callable_qualname != descriptor:
            raise WorkloadIdentityError("captured callable descriptor does not match profile callable")
        if _model_fingerprint(self.model) != graph.model_fingerprint:
            raise WorkloadIdentityError("model parameter fingerprint changed after graph capture")
        if canonical_json(input_signature(self.args, self.kwargs)) != canonical_json(
            graph.input_signature
        ):
            raise WorkloadIdentityError("profile input signature differs from graph capture")
        if self.precision != graph.precision:
            raise WorkloadIdentityError(
                f"precision mismatch: graph={graph.precision!r}, profile={self.precision!r}"
            )
        if self.model.training != graph.training_mode:
            raise WorkloadIdentityError(
                "profile callable train/eval mode differs from graph capture"
            )
        if self.mode == "training":
            if not graph.training_mode:
                raise WorkloadIdentityError("training profiling requires a training-mode graph")
            if self.loss_fn is None or self.target is None:
                raise WorkloadIdentityError(
                    "training profiling requires loss_fn and target"
                )
            expected_name = str(graph.optimizer_config.get("name", "")).lower()
            actual_name = (
                self.optimizer.__class__.__name__.lower()
                if self.optimizer is not None
                else "none"
            )
            if expected_name not in {"", actual_name}:
                raise WorkloadIdentityError(
                    f"optimizer mismatch: graph={expected_name!r}, callable={actual_name!r}"
                )
        elif graph.training_mode:
            raise WorkloadIdentityError("inference profiling cannot use a training-mode graph")


@dataclass(frozen=True)
class ProfileSample:
    timestamp_unix_s: float
    boundary: str
    step: int
    duration_ms: float | None
    nvml: dict[str, float]
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int


@dataclass(frozen=True)
class WorkloadIdentity:
    callable_qualname: str
    model_object_id: int
    source_fingerprint: str
    model_fingerprint: str
    input_value_fingerprint: str
    graph_sha256: str
    graph_ir_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    operator_registry_version: str
    operator_registry_sha256: str
    optimizer_config: dict[str, Any]
    precision: str
    mode: str
    execution_mode: str
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ProfileRecord:
    record_version: str
    status: str
    failure_stage: str | None
    error_type: str | None
    error_message: str | None
    started_at: str
    completed_at: str
    identity: WorkloadIdentity
    environment: dict[str, Any]
    correctness_validated: bool
    warmup_steps: int
    measured_steps_completed: int
    raw_samples: tuple[ProfileSample, ...]
    measured_step_ms: tuple[float, ...]
    epoch_duration_ms: float | None
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    trace_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ProfileRecord":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["identity"] = WorkloadIdentity(**raw["identity"])
        raw["raw_samples"] = tuple(ProfileSample(**sample) for sample in raw["raw_samples"])
        raw["measured_step_ms"] = tuple(raw["measured_step_ms"])
        return cls(**raw)


NvmlSampler = Callable[[], Mapping[str, float]]


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    optional_versions = {}
    for distribution in ("transformer-engine", "torchvision", "torchaudio"):
        try:
            optional_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            optional_versions[distribution] = None
    result["optional_versions"] = optional_versions
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "index": device,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
        }
        try:
            result["driver_version"] = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            result["driver_version"] = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            process_reserved = torch.cuda.memory_reserved(device)
            result["memory_precheck"] = {
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "device_used_bytes": int(total_bytes - free_bytes),
                "process_reserved_bytes": int(process_reserved),
                "estimated_other_process_bytes": max(
                    0,
                    int(total_bytes - free_bytes - process_reserved),
                ),
            }
        except RuntimeError:
            result["memory_precheck"] = None
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                query = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=clocks.current.sm,clocks.max.sm,power.limit",
                        "--format=csv,noheader,nounits",
                        f"--id={device}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                current_clock, maximum_clock, power_limit = (
                    value.strip()
                    for value in query.stdout.strip().split(",", maxsplit=2)
                )
                result["gpu_operating_limits"] = {
                    "sm_clock_mhz": float(current_clock),
                    "maximum_sm_clock_mhz": float(maximum_clock),
                    "power_limit_watts": float(power_limit),
                }
            except (OSError, ValueError, subprocess.SubprocessError):
                result["gpu_operating_limits"] = None
    return result


def _cuda_memory() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample(
    boundary: str,
    step: int,
    duration_ms: float | None,
    sampler: NvmlSampler | None,
) -> ProfileSample:
    allocated, reserved = _cuda_memory()
    nvml = {str(key): float(value) for key, value in (sampler() if sampler else {}).items()}
    return ProfileSample(
        timestamp_unix_s=time.time(),
        boundary=boundary,
        step=step,
        duration_ms=duration_ms,
        nvml=nvml,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
    )


def _correctness_check(workload: ProfileWorkload, options: ProfileOptions) -> None:
    assert workload.capture.exported_program is not None
    eager_args = clone_inputs(workload.args)
    eager_kwargs = clone_inputs(workload.kwargs)
    export_args = clone_inputs(workload.args)
    export_kwargs = clone_inputs(workload.kwargs)
    was_training = workload.model.training
    saved_state = {
        name: value.detach().clone()
        for name, value in workload.model.state_dict().items()
    }
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    def restore_rng() -> None:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    workload.model.train(workload.capture.graph.training_mode)
    try:
        restore_rng()
        with torch.no_grad():
            eager = workload.model(*eager_args, **eager_kwargs)
        restore_rng()
        with torch.no_grad():
            exported = workload.capture.exported_program.module()(*export_args, **export_kwargs)
    finally:
        workload.model.load_state_dict(saved_state)
        workload.model.train(was_training)
        restore_rng()
    compare_output_pytrees(
        eager,
        exported,
        rtol=options.correctness_rtol,
        atol=options.correctness_atol,
    )


def _execute(
    workload: ProfileWorkload,
    options: ProfileOptions,
    model_callable: Callable[..., Any],
) -> Any:
    if workload.mode == "inference":
        return model_callable(*workload.args, **workload.kwargs)
    assert workload.loss_fn is not None
    workload.model.zero_grad(set_to_none=True)
    if workload.optimizer is not None:
        workload.optimizer.zero_grad(set_to_none=True)

    def clear_input_grad(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value.grad = None
        elif isinstance(value, Mapping):
            for item in value.values():
                clear_input_grad(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                clear_input_grad(item)

    clear_input_grad(workload.args)
    clear_input_grad(workload.kwargs)
    loss = None
    for _ in range(options.gradient_accumulation_steps):
        output = model_callable(*workload.args, **workload.kwargs)
        loss = workload.loss_fn(output, workload.target)
        (loss / options.gradient_accumulation_steps).backward()
    if workload.optimizer is not None:
        workload.optimizer.step()
    return loss


def profile_workload(
    workload: ProfileWorkload,
    *,
    options: ProfileOptions | None = None,
    nvml_sampler: NvmlSampler | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProfileRecord:
    options = options or ProfileOptions()
    workload.validate_contract()
    graph = workload.capture.graph
    assert graph is not None
    _correctness_check(workload, options)
    model_callable: Callable[..., Any] = workload.model
    if options.execution_mode == "compile":
        compile_kwargs = {}
        if options.compile_backend is not None:
            compile_kwargs["backend"] = options.compile_backend
        model_callable = torch.compile(workload.model, **compile_kwargs)
    identity = WorkloadIdentity(
        callable_qualname=workload.capture.callable_qualname or "",
        model_object_id=id(workload.model),
        source_fingerprint=graph.source_fingerprint,
        model_fingerprint=graph.model_fingerprint,
        input_value_fingerprint=input_value_fingerprint(workload.args, workload.kwargs),
        graph_sha256=graph.graph_sha256,
        graph_ir_version=graph.graph_ir_version,
        feature_schema_version=graph.feature_schema_version,
        feature_schema_sha256=graph.feature_schema_sha256,
        operator_registry_version=graph.operator_registry_version,
        operator_registry_sha256=graph.operator_registry_sha256,
        optimizer_config=graph.optimizer_config,
        precision=workload.precision,
        mode=workload.mode,
        execution_mode=options.execution_mode,
        gradient_accumulation_steps=options.gradient_accumulation_steps,
    )
    started_at = datetime.now(UTC).isoformat()
    samples: list[ProfileSample] = [_sample("profile_start", -1, None, nvml_sampler)]
    durations: list[float] = []
    status = "ok"
    failure_stage = error_type = error_message = None
    measured_completed = 0
    trace_path = options.trace_path
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def run_steps() -> None:
        nonlocal measured_completed
        for step in range(options.warmup_steps):
            samples.append(_sample("warmup_start", step, None, nvml_sampler))
            _execute(workload, options, model_callable)
            _synchronize()
            samples.append(_sample("warmup_end", step, None, nvml_sampler))
        for step in range(options.measured_steps):
            samples.append(_sample("measured_start", step, None, nvml_sampler))
            _synchronize()
            start = time.perf_counter()
            _execute(workload, options, model_callable)
            _synchronize()
            duration_ms = (time.perf_counter() - start) * 1000.0
            durations.append(duration_ms)
            measured_completed += 1
            samples.append(_sample("measured_end", step, duration_ms, nvml_sampler))

    sampler_stop = threading.Event()
    sampler_thread: threading.Thread | None = None
    if nvml_sampler is not None:
        def poll_nvml() -> None:
            while not sampler_stop.wait(options.nvml_sample_interval_s):
                samples.append(_sample("nvml_poll", -1, None, nvml_sampler))

        sampler_thread = threading.Thread(
            target=poll_nvml,
            name="perfseer-v3-nvml-sampler",
            daemon=True,
        )
        sampler_thread.start()
    try:
        if trace_path:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                run_steps()
            trace = Path(trace_path)
            trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace))
        else:
            run_steps()
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        status = "oom"
        failure_stage = "warmup" if measured_completed == 0 and not durations else "measurement"
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=max(1.0, options.nvml_sample_interval_s * 4))
    samples.append(_sample("profile_end", measured_completed, None, nvml_sampler))
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    completed_at = datetime.now(UTC).isoformat()
    return ProfileRecord(
        record_version="perfseer_profile_v3",
        status=status,
        failure_stage=failure_stage,
        error_type=error_type,
        error_message=error_message,
        started_at=started_at,
        completed_at=completed_at,
        identity=identity,
        environment=_environment(),
        correctness_validated=True,
        warmup_steps=options.warmup_steps,
        measured_steps_completed=measured_completed,
        raw_samples=tuple(samples),
        measured_step_ms=tuple(durations),
        epoch_duration_ms=sum(durations) if durations else None,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        trace_path=trace_path,
        metadata={
            **dict(metadata or {}),
            "compile_backend": options.compile_backend,
            "compile_warmup_excluded": options.compile_warmup_excluded,
            "nvml_sample_interval_s": options.nvml_sample_interval_s,
        },
    )


__all__ = [
    "NvmlSampler",
    "ProfileOptions",
    "ProfileRecord",
    "ProfileSample",
    "ProfileWorkload",
    "WorkloadIdentity",
    "WorkloadIdentityError",
    "input_value_fingerprint",
    "profile_workload",
]
