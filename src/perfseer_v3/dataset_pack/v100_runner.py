"""Isolated five-epoch training engine for one V100 label attempt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

import torch

from .adapters import adapter_for_task
from .contracts import EPOCH_MEASUREMENT_VERSION, EpochMeasurement, TelemetrySample
from .fingerprints import canonical_sha256
from ..hardware import HardwareProfileV3, assert_physical_hardware_identity
from .local_runtime import (
    _all_finite,
    _build_model,
    _build_optimizer,
    _build_scheduler,
    _forward_loss,
    _gradients_finite,
    _move,
    _precision_context,
    bind_candidate_batch_shape,
    configure_precision_backends,
    five_epoch_optimizer_steps,
)
from .real_data import VerifiedPreparedDataset
from .sampler import TargetCandidate
from .labeler_profile import PROFILE
from .task_registry import TaskRegistryEntry


MIB = 1024**2
if PROFILE.name == "native_a10_speech_v2":
    V100_RUN_RESULT_VERSION = "perfseer_v3_nrp_a10_speech_v2_five_epoch_result_v2"
elif PROFILE.is_native_a10:
    V100_RUN_RESULT_VERSION = "perfseer_v3_nrp_a10_five_epoch_result_v1"
else:
    V100_RUN_RESULT_VERSION = "perfseer_v3_v100_five_epoch_result_v1"


class V100RunError(RuntimeError):
    """Raised when a single configuration cannot produce an accepted run payload."""


@dataclass(frozen=True)
class TelemetryReading:
    sm_util_percent: float
    memory_controller_util_percent: float
    device_used_vram_mib: float
    throttle_reason_bits: int
    compute_process_ids: tuple[int, ...]


class TelemetryBackend(Protocol):
    @property
    def gpu_uuid(self) -> str: ...

    @property
    def hardware_fingerprint(self) -> str: ...

    def read(self) -> TelemetryReading: ...


class NvmlTelemetryBackend:
    """NVML adapter bound to the one physical GPU exposed to the child."""

    def __init__(
        self,
        physical_gpu_index: int,
        *,
        expected_hardware_profile: HardwareProfileV3 | None = None,
    ) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(self._uuid, bytes):
                self._uuid = self._uuid.decode("utf-8")
            properties = torch.cuda.get_device_properties(0)
            if expected_hardware_profile is None:
                normalized_name = "".join(
                    character for character in str(name).upper() if character.isalnum()
                )
                if PROFILE.is_native_a10:
                    qualified = (
                        normalized_name == "NVIDIAA10"
                        and 22 * 1024**3 <= memory.total <= 26 * 1024**3
                        and (properties.major, properties.minor) == (8, 6)
                    )
                    expected_description = "NVIDIA A10 24GB with compute capability 8.6"
                else:
                    qualified = (
                        "TESLAV100SXM2" in normalized_name
                        and 30 * 1024**3 <= memory.total <= 34 * 1024**3
                        and (properties.major, properties.minor) == (7, 0)
                    )
                    expected_description = "Tesla V100 SXM2 32GB with compute capability 7.0"
                if not qualified:
                    raise V100RunError(f"label worker is not bound to {expected_description}")
                provenance = {
                    "target_hardware_id": PROFILE.target_hardware_id,
                    "name": str(name),
                    "uuid": self._uuid,
                    "total_memory_bytes": int(memory.total),
                    "compute_capability": [properties.major, properties.minor],
                }
                self._hardware_profile_sha256 = None
            else:
                expected_hardware_profile.validate(require_complete_signature=True)
                try:
                    assert_physical_hardware_identity(
                        expected_hardware_profile.hardware_id,
                        name,
                    )
                except ValueError as error:
                    raise V100RunError(str(error)) from error
                expected = expected_hardware_profile.canonical_payload
                expected_memory = expected["static"].get("memory_bytes")
                expected_capability = expected["static"].get("compute_capability")
                observed_capability = properties.major + properties.minor / 10.0
                if expected_memory is not None and abs(memory.total - expected_memory) / expected_memory > 0.02:
                    raise V100RunError("visible GPU memory differs from the frozen target profile")
                if expected_capability is not None and abs(observed_capability - expected_capability) > 1e-6:
                    raise V100RunError("visible GPU compute capability differs from the frozen target profile")
                self._hardware_profile_sha256 = expected_hardware_profile.sha256
                provenance = {
                    "target_hardware_id": expected_hardware_profile.hardware_id,
                    "hardware_profile_sha256": expected_hardware_profile.sha256,
                    "name": str(name),
                    "uuid": self._uuid,
                    "total_memory_bytes": int(memory.total),
                    "compute_capability": [properties.major, properties.minor],
                }
            self._hardware_fingerprint = canonical_sha256(provenance)
        except V100RunError:
            raise
        except Exception as error:
            raise V100RunError("NVML V100 qualification failed") from error

    @property
    def gpu_uuid(self) -> str:
        return self._uuid

    @property
    def hardware_fingerprint(self) -> str:
        return self._hardware_fingerprint

    @property
    def hardware_profile_sha256(self) -> str | None:
        return self._hardware_profile_sha256

    def read(self) -> TelemetryReading:
        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            throttle = self._pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle)
            processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception as error:
            raise V100RunError("NVML telemetry read failed") from error
        harmful_throttle_mask = 0
        for name in (
            "nvmlClocksThrottleReasonSwPowerCap",
            "nvmlClocksThrottleReasonHwSlowdown",
            "nvmlClocksThrottleReasonSwThermalSlowdown",
            "nvmlClocksThrottleReasonHwThermalSlowdown",
            "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
        ):
            harmful_throttle_mask |= int(getattr(self._pynvml, name, 0))
        return TelemetryReading(
            sm_util_percent=float(utilization.gpu),
            memory_controller_util_percent=float(utilization.memory),
            device_used_vram_mib=float(memory.used / MIB),
            throttle_reason_bits=int(throttle) & harmful_throttle_mask,
            compute_process_ids=tuple(sorted(int(row.pid) for row in processes)),
        )


@dataclass(frozen=True)
class _StampedReading:
    epoch: int
    timestamp: float
    reading: TelemetryReading


class AsyncTelemetryCollector:
    def __init__(
        self,
        backend: TelemetryBackend,
        *,
        interval_seconds: float = 0.1,
        allowed_process_ids: Sequence[int] = (),
    ) -> None:
        if not 0.02 <= interval_seconds <= 1.0:
            raise V100RunError("telemetry interval must be between 20 ms and 1 s")
        self.backend = backend
        self.interval_seconds = interval_seconds
        self.allowed = frozenset((*allowed_process_ids, os.getpid()))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._epoch: int | None = None
        self._origin = 0.0
        self._rows: list[_StampedReading] = []
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise V100RunError("telemetry collector already started")
        self._origin = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="v100-nvml", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._capture()
            except BaseException as error:  # retained and raised by the training thread
                self._error = error
                self._stop.set()

    def _capture(self) -> None:
        with self._lock:
            epoch = self._epoch
        if epoch is None:
            return
        reading = self.backend.read()
        stamped = _StampedReading(epoch, time.monotonic(), reading)
        with self._lock:
            self._rows.append(stamped)

    def begin_epoch(self, epoch: int) -> None:
        with self._lock:
            self._epoch = epoch
        self._capture()

    def end_epoch(self, epoch: int, boundary: float) -> tuple[tuple[TelemetrySample, ...], bool]:
        self._capture()
        with self._lock:
            self._epoch = None
            rows = [row for row in self._rows if row.epoch == epoch]
        if self._error is not None:
            raise V100RunError("asynchronous telemetry failed") from self._error
        if not rows:
            raise V100RunError("measured epoch has no telemetry")
        samples = []
        for index, row in enumerate(rows):
            next_timestamp = rows[index + 1].timestamp if index + 1 < len(rows) else boundary
            duration = max(1e-9, next_timestamp - row.timestamp)
            reading = row.reading
            samples.append(
                TelemetrySample(
                    timestamp_offset_s=max(0.0, row.timestamp - self._origin),
                    duration_s=duration,
                    sm_util_percent=reading.sm_util_percent,
                    memory_controller_util_percent=reading.memory_controller_util_percent,
                    device_used_vram_mib=reading.device_used_vram_mib,
                    throttle_reason_bits=reading.throttle_reason_bits,
                )
            )
        foreign = any(
            not set(row.reading.compute_process_ids) <= self.allowed for row in rows
        )
        return tuple(samples), foreign

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self.interval_seconds))
            if self._thread.is_alive():
                raise V100RunError("telemetry thread did not terminate")
        if self._error is not None:
            raise V100RunError("asynchronous telemetry failed") from self._error


@dataclass(frozen=True)
class FiveEpochRunResult:
    version: str
    gpu_uuid: str
    hardware_sha256: str
    requested_backend_id: str
    observed_backend_id: str
    compile_completed_before_epoch_1: bool
    completed_epochs: tuple[int, ...]
    finite_loss_epochs: tuple[int, ...]
    finite_gradient_epochs: tuple[int, ...]
    epoch_measurements: tuple[EpochMeasurement, ...]

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FiveEpochRunResult":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise V100RunError("serialized five-epoch result schema differs")
        measurements = []
        for raw in value["epoch_measurements"]:
            if not isinstance(raw, Mapping):
                raise V100RunError("serialized epoch measurement is invalid")
            samples = tuple(TelemetrySample(**row) for row in raw["telemetry_samples"])
            measurements.append(EpochMeasurement(**{**raw, "telemetry_samples": samples}))
        result = cls(
            **{
                **dict(value),
                "completed_epochs": tuple(value["completed_epochs"]),
                "finite_loss_epochs": tuple(value["finite_loss_epochs"]),
                "finite_gradient_epochs": tuple(value["finite_gradient_epochs"]),
                "epoch_measurements": tuple(measurements),
            }
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.version != V100_RUN_RESULT_VERSION:
            raise V100RunError("five-epoch result version mismatch")
        if self.completed_epochs != (1, 2, 3, 4, 5):
            raise V100RunError("five-epoch result is partial")
        if self.finite_loss_epochs != self.completed_epochs or self.finite_gradient_epochs != self.completed_epochs:
            raise V100RunError("five-epoch result contains non-finite training state")
        if tuple(row.epoch for row in self.epoch_measurements) != (3, 4, 5):
            raise V100RunError("five-epoch result must retain epochs 3-5 only")
        for row in self.epoch_measurements:
            row.validate(accepted=True)
        if not self.gpu_uuid or len(self.hardware_sha256) != 64:
            raise V100RunError("five-epoch hardware identity is invalid")
        if not self.requested_backend_id or self.requested_backend_id != self.observed_backend_id:
            raise V100RunError("five-epoch backend identity differs")


def _compile_training_path(
    candidate: TargetCandidate,
    model: Any,
    adapter: Any,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Any:
    def forward_loss(value: Mapping[str, Any]) -> tuple[Any, torch.Tensor]:
        return _forward_loss(
            model, adapter, value, bool(candidate.activation_checkpointing["enabled"])
        )

    if candidate.execution["mode"] == "eager":
        return forward_loss
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    compiled = torch.compile(forward_loss, backend="inductor", fullgraph=False, dynamic=False)
    with _precision_context(candidate, device):
        _, loss = compiled(batch)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state)
    return compiled


def _optimizer_step(
    candidate: TargetCandidate,
    model: Any,
    adapter: Any,
    batches: Sequence[Mapping[str, Any]],
    forward_loss: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
) -> tuple[bool, bool, torch.Tensor]:
    if not batches:
        raise V100RunError("optimizer step has no microbatches")
    optimizer.zero_grad(set_to_none=True)
    outputs_finite = True
    losses_finite = True
    last_loss: torch.Tensor | None = None
    device = next(model.parameters()).device
    if candidate.optimizer["name"] == "lbfgs":
        holder: dict[str, Any] = {}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for batch in batches:
                with _precision_context(candidate, device):
                    output, loss = forward_loss(batch)
                losses.append(loss)
                holder["output"] = output
            combined = torch.stack(losses).mean()
            combined.backward()
            holder["loss"] = combined
            return combined

        last_loss = optimizer.step(closure)
        outputs_finite = _all_finite(holder["output"])
        losses_finite = bool(torch.isfinite(holder["loss"]))
        last_loss = holder["loss"]
    else:
        for batch in batches:
            with _precision_context(candidate, device):
                output, loss = forward_loss(batch)
            outputs_finite &= _all_finite(output)
            losses_finite &= bool(torch.isfinite(loss))
            scaled_loss = loss / len(batches)
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            last_loss = loss
        if scaler is None:
            if not _gradients_finite(model):
                return losses_finite and outputs_finite, False, last_loss.detach()
            optimizer.step()
        else:
            for item in optimizer.optimizers:
                scaler.unscale_(item)
            if not _gradients_finite(model):
                return losses_finite and outputs_finite, False, last_loss.detach()
            for item in optimizer.optimizers:
                scaler.step(item)
            scaler.update()
    gradients_finite = _gradients_finite(model)
    if last_loss is None:
        raise V100RunError("optimizer step produced no loss")
    scheduler.step(last_loss, unit="optimizer_step")
    return losses_finite and outputs_finite, gradients_finite, last_loss.detach()


def run_five_epoch_training(
    candidate: TargetCandidate,
    task_entry: TaskRegistryEntry,
    *,
    public_directory: str | Path,
    prepared_directory: str | Path,
    archive_sha256: str,
    telemetry_backend: TelemetryBackend,
    telemetry_interval_seconds: float = 0.1,
    device: str | torch.device = "cuda",
    allow_non_v100_test_device: bool = False,
) -> FiveEpochRunResult:
    """Execute exactly one accepted-path attempt; callers isolate it in a child."""

    candidate.validate()
    task_entry.validate()
    if candidate.task_id != task_entry.task_id:
        raise V100RunError("candidate/task mismatch")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        if not allow_non_v100_test_device:
            raise V100RunError("V100 label training requires one visible CUDA device")
    else:
        if not torch.cuda.is_available():
            raise V100RunError("V100 label training requires one visible CUDA device")
        if torch.cuda.device_count() != 1:
            raise V100RunError("fresh label child must see exactly one physical GPU")
    torch.manual_seed(int(candidate.seed_policy["seed"]))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(candidate.seed_policy["seed"]))
        configure_precision_backends(candidate)
    adapter = adapter_for_task(candidate.task_id)
    dataset = VerifiedPreparedDataset(
        task_entry,
        Path(public_directory),
        Path(prepared_directory),
        archive_sha256,
    )
    if len(dataset) != task_entry.expected_train_examples:
        raise V100RunError("prepared dataset traversal count changed")
    model = _build_model(candidate, adapter, resolved_device)
    first = dataset.build_batch(range(candidate.microbatch_size))
    first = bind_candidate_batch_shape(candidate, adapter, _move(first, resolved_device))
    forward_loss = _compile_training_path(candidate, model, adapter, first, resolved_device)
    total_optimizer_steps = five_epoch_optimizer_steps(
        candidate,
        expected_train_examples=len(dataset),
    )
    optimizer = _build_optimizer(candidate, model)
    scheduler = _build_scheduler(
        candidate, optimizer, total_optimizer_steps=total_optimizer_steps
    )
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1_000)
        if resolved_device.type == "cuda" and candidate.precision_policy["gradient_scaler"]
        else None
    )
    collector = AsyncTelemetryCollector(
        telemetry_backend,
        interval_seconds=telemetry_interval_seconds,
        allowed_process_ids=(os.getpid(),),
    )
    completed: list[int] = []
    finite_losses: list[int] = []
    finite_gradients: list[int] = []
    measurements: list[EpochMeasurement] = []
    try:
        for epoch in range(1, 6):
            measured = epoch >= 3
            if epoch == 3:
                if resolved_device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                collector.start()
            if measured:
                collector.begin_epoch(epoch)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
            batches = examples = optimizer_steps = 0
            epoch_loss_finite = True
            epoch_gradients_finite = True
            buffered: list[Mapping[str, Any]] = []
            last_loss_for_epoch: torch.Tensor | None = None
            for start in range(0, len(dataset), candidate.microbatch_size):
                stop = min(len(dataset), start + candidate.microbatch_size)
                batch = dataset.build_batch(range(start, stop))
                batch = bind_candidate_batch_shape(
                    candidate, adapter, _move(batch, resolved_device)
                )
                buffered.append(batch)
                batches += 1
                examples += stop - start
                if (
                    len(buffered) == candidate.gradient_accumulation_steps
                    or stop == len(dataset)
                ):
                    loss_ok, gradient_ok, step_loss = _optimizer_step(
                        candidate,
                        model,
                        adapter,
                        buffered,
                        forward_loss,
                        optimizer,
                        scheduler,
                        scaler,
                    )
                    epoch_loss_finite &= loss_ok
                    epoch_gradients_finite &= gradient_ok
                    last_loss_for_epoch = step_loss
                    optimizer_steps += 1
                    buffered.clear()
            if last_loss_for_epoch is None:
                raise V100RunError("epoch produced no optimizer steps")
            scheduler.step(last_loss_for_epoch, unit="epoch")
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            ended = time.monotonic()
            completed.append(epoch)
            if epoch_loss_finite:
                finite_losses.append(epoch)
            if epoch_gradients_finite:
                finite_gradients.append(epoch)
            if measured:
                samples, foreign = collector.end_epoch(epoch, ended)
                measurement = EpochMeasurement(
                    version=EPOCH_MEASUREMENT_VERSION,
                    epoch=epoch,
                    epoch_completed=True,
                    epoch_ms=(ended - started) * 1_000.0,
                    examples_seen=examples,
                    batches_seen=batches,
                    microsteps=batches,
                    optimizer_steps=optimizer_steps,
                    loss_finite=epoch_loss_finite,
                    gradients_finite=epoch_gradients_finite,
                    telemetry_complete=True,
                    telemetry_samples=samples,
                    peak_torch_reserved_mib=(
                        torch.cuda.max_memory_reserved() / MIB
                        if resolved_device.type == "cuda"
                        else 0.0
                    ),
                    requested_backend_id=str(candidate.execution["backend_id"]),
                    observed_backend_id=str(candidate.execution["backend_id"]),
                    foreign_process_detected=foreign,
                )
                measurement.validate(accepted=True)
                measurements.append(measurement)
    finally:
        if collector._thread is not None:
            collector.stop()
    result = FiveEpochRunResult(
        version=V100_RUN_RESULT_VERSION,
        gpu_uuid=telemetry_backend.gpu_uuid,
        hardware_sha256=telemetry_backend.hardware_fingerprint,
        requested_backend_id=str(candidate.execution["backend_id"]),
        observed_backend_id=str(candidate.execution["backend_id"]),
        compile_completed_before_epoch_1=candidate.execution["mode"] == "compiled",
        completed_epochs=tuple(completed),
        finite_loss_epochs=tuple(finite_losses),
        finite_gradient_epochs=tuple(finite_gradients),
        epoch_measurements=tuple(measurements),
    )
    result.validate()
    return result


__all__ = [
    "V100_RUN_RESULT_VERSION",
    "V100RunError",
    "AsyncTelemetryCollector",
    "FiveEpochRunResult",
    "NvmlTelemetryBackend",
    "TelemetryBackend",
    "TelemetryReading",
    "run_five_epoch_training",
]
