"""Typed contracts for the simplified V100 18K planning and label records."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from .fingerprints import canonical_sha256, canonical_value


WORKLOAD_CONFIG_VERSION = "perfseer_v3_v100_workload_config_v2"
LABEL_RUN_RECORD_VERSION = "perfseer_v3_v100_label_run_v2"
EPOCH_MEASUREMENT_VERSION = "perfseer_v3_v100_epoch_measurement_v1"
TASK_REGISTRY_VERSION = "perfseer_v3_v100_task_registry_v1"
MODEL_REGISTRY_VERSION = "perfseer_v3_v100_model_registry_v1"
TARGET_NAMES = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_p95_sm_util_percent",
    "train_peak_vram_used_mib",
    "train_peak_torch_reserved_mib",
    "train_peak_memory_controller_util_percent",
)
AUXILIARY_TARGET_NAMES = (
    "operator_or_block_time_us",
    "forward_time_us",
    "backward_time_us",
    "optimizer_time_us",
    "bytes_read",
    "bytes_written",
    "workspace_peak_mib",
    "saved_for_backward_mib",
    "peak_live_tensor_mib",
    "profiler_gpu_time_fraction",
)


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CorpusLayer(_StringEnum):
    """Corpus identity retained only so operation/composite local QA can coexist."""

    OPERATION = "operation"
    COMPOSITE = "composite"
    END_TO_END = "end_to_end"
    OOM_AUXILIARY = "oom_auxiliary"


class AttemptStatus(_StringEnum):
    ACCEPTED = "accepted"
    FAILED = "failed"
    OOM = "oom"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    CONTAMINATED = "contaminated"
    QUARANTINED = "quarantined"


class FailureStage(_StringEnum):
    NONE = "none"
    MATERIALIZATION = "materialization"
    CAPTURE = "capture"
    COMPILE = "compile"
    FORWARD = "forward"
    LOSS = "loss"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    ALLOCATOR = "allocator"
    TIMEOUT = "timeout"
    TELEMETRY = "telemetry"
    CLEANUP = "cleanup"
    CONTAMINATION = "contamination"
    STABILITY = "stability"


def _require_text(value: str, *, context: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")


def _require_sha256(value: str, *, context: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")


def _require_finite_nonnegative(value: float, *, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{context} must be finite and nonnegative")


def _require_percent(value: float, *, context: str) -> None:
    _require_finite_nonnegative(value, context=context)
    if float(value) > 100:
        raise ValueError(f"{context} must be between 0 and 100")


@dataclass(frozen=True)
class FingerprintBundle:
    source_sha256: str
    graph_sha256: str
    environment_sha256: str
    hardware_sha256: str
    support_contract_sha256: str
    dataset_sha256: str | None = None
    synthetic_input_spec_sha256: str | None = None

    def validate(self, corpus_layer: CorpusLayer) -> None:
        for name in (
            "source_sha256",
            "graph_sha256",
            "environment_sha256",
            "hardware_sha256",
            "support_contract_sha256",
        ):
            _require_sha256(getattr(self, name), context=name)
        if corpus_layer == CorpusLayer.END_TO_END:
            if self.dataset_sha256 is None:
                raise ValueError("end-to-end records require dataset_sha256")
            if self.synthetic_input_spec_sha256 is not None:
                raise ValueError("end-to-end records must not use a synthetic input fingerprint")
        elif corpus_layer in {CorpusLayer.OPERATION, CorpusLayer.COMPOSITE}:
            if self.synthetic_input_spec_sha256 is None:
                raise ValueError("operation/composite records require synthetic_input_spec_sha256")
        for name in ("dataset_sha256", "synthetic_input_spec_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, context=name)


@dataclass(frozen=True)
class WorkloadConfiguration:
    version: str
    source_lineage: str
    source_sha256: str
    generator_version: str
    mutation_specification: Mapping[str, Any]
    modality: str
    architecture_family: str
    architecture_parameters: Mapping[str, Any]
    task_id: str
    dataset_revision: str
    input_signature: Mapping[str, Any]
    training_step_id: str
    microbatch_size: int
    gradient_accumulation_steps: int
    precision_policy: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    activation_checkpointing: Mapping[str, Any]
    execution: Mapping[str, Any]
    seed_policy: Mapping[str, Any]
    dependency_revision: str
    container_digest: str
    cuda_revision: str
    driver_revision: str
    pytorch_revision: str
    kernel_library_revisions: Mapping[str, Any]
    target_hardware_id: str
    observation_protocol_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(asdict(self))

    @property
    def configuration_id(self) -> str:
        self.validate()
        return canonical_sha256(self.to_dict())

    def validate(self) -> None:
        if self.version != WORKLOAD_CONFIG_VERSION:
            raise ValueError(f"unsupported workload config version {self.version!r}")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise ValueError("workload configuration must target NRP V100, never generic V100")
        for name in (
            "source_lineage",
            "generator_version",
            "modality",
            "architecture_family",
            "task_id",
            "dataset_revision",
            "training_step_id",
            "dependency_revision",
            "container_digest",
            "cuda_revision",
            "driver_revision",
            "pytorch_revision",
        ):
            _require_text(getattr(self, name), context=name)
        _require_sha256(self.source_sha256, context="source_sha256")
        _require_sha256(self.observation_protocol_sha256, context="observation_protocol_sha256")
        if type(self.microbatch_size) is not int or self.microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        if type(self.gradient_accumulation_steps) is not int or self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        canonical_value(self.to_dict())


@dataclass(frozen=True)
class TargetVector:
    values: tuple[float | None, ...]
    validity: tuple[bool, ...]

    def validate(self, *, require_all_valid: bool = False) -> None:
        if len(self.values) != len(TARGET_NAMES) or len(self.validity) != len(TARGET_NAMES):
            raise ValueError("target values and validity must follow the six-target contract")
        if any(type(valid) is not bool for valid in self.validity):
            raise ValueError("target validity entries must be booleans")
        for name, value, valid in zip(TARGET_NAMES, self.values, self.validity):
            if valid:
                if value is None:
                    raise ValueError(f"valid target {name} cannot be null")
                _require_finite_nonnegative(value, context=name)
            elif value is not None:
                raise ValueError(f"invalid target {name} must be null, never fabricated")
        if require_all_valid and not all(self.validity):
            raise ValueError("accepted end-to-end records require all six valid targets")


@dataclass(frozen=True)
class AuxiliaryTargetVector:
    """Historical/local-QA target vector; never emitted by the 18K NRP workflow."""

    values: tuple[float | None, ...]
    validity: tuple[bool, ...]
    observed_backend_id: str
    measurement_quality: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.values) != len(AUXILIARY_TARGET_NAMES) or len(self.validity) != len(AUXILIARY_TARGET_NAMES):
            raise ValueError("auxiliary values and validity do not match the auxiliary target contract")
        if any(type(valid) is not bool for valid in self.validity):
            raise ValueError("auxiliary validity entries must be booleans")
        for name, value, valid in zip(AUXILIARY_TARGET_NAMES, self.values, self.validity):
            if valid:
                if value is None:
                    raise ValueError(f"valid auxiliary target {name} cannot be null")
                _require_finite_nonnegative(value, context=name)
            elif value is not None:
                raise ValueError(f"invalid auxiliary target {name} must be null")
        _require_text(self.observed_backend_id, context="observed_backend_id")
        canonical_value(self.measurement_quality)


@dataclass(frozen=True)
class GpuCleanupEvidence:
    pre_sample_device_used_vram_mib: float
    post_cleanup_device_used_vram_mib: float
    release_tolerance_mib: float
    cleanup_seconds: float
    stable_dwell_seconds: float
    child_process_tree_exited: bool
    owned_gpu_processes_remaining: int
    passed: bool

    def validate(self) -> None:
        for name in (
            "pre_sample_device_used_vram_mib",
            "post_cleanup_device_used_vram_mib",
            "release_tolerance_mib",
            "cleanup_seconds",
            "stable_dwell_seconds",
        ):
            _require_finite_nonnegative(getattr(self, name), context=name)
        if self.release_tolerance_mib <= 0:
            raise ValueError("release_tolerance_mib must be positive")
        if type(self.owned_gpu_processes_remaining) is not int or self.owned_gpu_processes_remaining < 0:
            raise ValueError("owned_gpu_processes_remaining must be a nonnegative integer")
        if type(self.child_process_tree_exited) is not bool or type(self.passed) is not bool:
            raise ValueError("cleanup status fields must be booleans")
        memory_released = self.post_cleanup_device_used_vram_mib <= (
            self.pre_sample_device_used_vram_mib + self.release_tolerance_mib
        )
        derived_pass = (
            self.child_process_tree_exited
            and self.owned_gpu_processes_remaining == 0
            and memory_released
            and self.stable_dwell_seconds > 0
        )
        if self.passed != derived_pass:
            raise ValueError("cleanup passed flag disagrees with process/memory/dwell evidence")


@dataclass(frozen=True)
class TelemetrySample:
    """Only target-producing NVML fields retained for an accepted run."""

    timestamp_offset_s: float
    duration_s: float
    sm_util_percent: float
    memory_controller_util_percent: float
    device_used_vram_mib: float
    throttle_reason_bits: int = 0

    def validate(self) -> None:
        _require_finite_nonnegative(self.timestamp_offset_s, context="timestamp_offset_s")
        _require_finite_nonnegative(self.duration_s, context="duration_s")
        if self.duration_s <= 0:
            raise ValueError("telemetry duration_s must be positive")
        _require_percent(self.sm_util_percent, context="sm_util_percent")
        _require_percent(
            self.memory_controller_util_percent,
            context="memory_controller_util_percent",
        )
        _require_finite_nonnegative(self.device_used_vram_mib, context="device_used_vram_mib")
        if type(self.throttle_reason_bits) is not int or self.throttle_reason_bits < 0:
            raise ValueError("throttle_reason_bits must be a nonnegative integer")


@dataclass(frozen=True)
class EpochMeasurement:
    version: str
    epoch: int
    epoch_completed: bool
    epoch_ms: float | None
    examples_seen: int
    batches_seen: int
    microsteps: int
    optimizer_steps: int
    loss_finite: bool
    gradients_finite: bool
    telemetry_complete: bool
    telemetry_samples: tuple[TelemetrySample, ...]
    peak_torch_reserved_mib: float | None
    requested_backend_id: str
    observed_backend_id: str
    foreign_process_detected: bool

    def validate(self, *, accepted: bool = False) -> None:
        if self.version != EPOCH_MEASUREMENT_VERSION:
            raise ValueError(f"unsupported epoch measurement version {self.version!r}")
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("epoch must be a positive integer")
        for name in ("epoch_completed", "loss_finite", "gradients_finite", "telemetry_complete", "foreign_process_detected"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("examples_seen", "batches_seen", "microsteps", "optimizer_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _require_text(self.requested_backend_id, context="requested_backend_id")
        _require_text(self.observed_backend_id, context="observed_backend_id")
        for sample in self.telemetry_samples:
            sample.validate()
        if self.epoch_completed:
            if self.epoch_ms is None:
                raise ValueError("a complete measured epoch requires direct epoch_ms")
            _require_finite_nonnegative(self.epoch_ms, context="epoch_ms")
            if self.epoch_ms <= 0:
                raise ValueError("complete epoch_ms must be positive")
        elif self.epoch_ms is not None:
            raise ValueError("a partial epoch must not retain an epoch-time label")
        if self.peak_torch_reserved_mib is not None:
            _require_finite_nonnegative(
                self.peak_torch_reserved_mib,
                context="peak_torch_reserved_mib",
            )
        if accepted:
            if not self.epoch_completed:
                raise ValueError("accepted runs require complete measured epochs")
            if min(self.examples_seen, self.batches_seen, self.microsteps, self.optimizer_steps) < 1:
                raise ValueError("accepted measured epochs require positive traversal counts")
            if not self.loss_finite or not self.gradients_finite:
                raise ValueError("accepted measured epochs require finite loss and gradients")
            if not self.telemetry_complete or not self.telemetry_samples:
                raise ValueError("accepted measured epochs require complete telemetry")
            if self.peak_torch_reserved_mib is None:
                raise ValueError("accepted measured epochs require reserved-memory peak")
            if self.foreign_process_detected:
                raise ValueError("accepted measured epochs cannot contain a foreign GPU process")
            if self.requested_backend_id != self.observed_backend_id:
                raise ValueError("accepted measured epochs require matching backend identity")
            if any(sample.throttle_reason_bits != 0 for sample in self.telemetry_samples):
                raise ValueError("accepted measured epochs cannot contain throttle reasons")


def _nearest_rank_p95(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("P95 requires at least one telemetry sample")
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


@dataclass(frozen=True)
class LabelRunRecord:
    version: str
    run_id: str
    configuration_id: str
    status: AttemptStatus
    failure_stage: FailureStage
    fingerprints: FingerprintBundle
    gpu_uuid: str
    target_hardware_id: str
    capture_workload_sha256: str
    profile_workload_sha256: str
    coverage_cell_ids: tuple[str, ...]
    task_id: str
    execution_mode: str
    compile_completed_before_epoch_1: bool
    expected_train_examples: int
    microbatch_size: int
    gradient_accumulation_steps: int
    total_epochs: int
    warmup_epochs: tuple[int, ...]
    measured_epochs: tuple[int, ...]
    completed_epochs: tuple[int, ...]
    finite_loss_epochs: tuple[int, ...]
    finite_gradient_epochs: tuple[int, ...]
    epoch_measurements: tuple[EpochMeasurement, ...]
    cleanup: GpuCleanupEvidence
    targets: TargetVector | None = None

    def aggregate_targets(self) -> TargetVector:
        if tuple(row.epoch for row in self.epoch_measurements) != (3, 4, 5):
            raise ValueError("target aggregation requires measured epochs 3, 4, and 5")
        for row in self.epoch_measurements:
            row.validate(accepted=True)
        epoch_times = tuple(float(row.epoch_ms) for row in self.epoch_measurements if row.epoch_ms is not None)
        samples = tuple(
            sample for row in self.epoch_measurements for sample in row.telemetry_samples
        )
        total_duration = sum(sample.duration_s for sample in samples)
        if total_duration <= 0:
            raise ValueError("target aggregation requires positive telemetry duration")
        values = (
            sum(epoch_times) / 3.0,
            sum(sample.sm_util_percent * sample.duration_s for sample in samples) / total_duration,
            _nearest_rank_p95(tuple(sample.sm_util_percent for sample in samples)),
            max(sample.device_used_vram_mib for sample in samples),
            max(float(row.peak_torch_reserved_mib) for row in self.epoch_measurements if row.peak_torch_reserved_mib is not None),
            max(sample.memory_controller_util_percent for sample in samples),
        )
        return TargetVector(values=values, validity=(True,) * len(TARGET_NAMES))

    @property
    def epoch_time_relative_spread(self) -> float | None:
        if tuple(row.epoch for row in self.epoch_measurements) != (3, 4, 5):
            return None
        if any(not row.epoch_completed or row.epoch_ms is None for row in self.epoch_measurements):
            return None
        values = tuple(float(row.epoch_ms) for row in self.epoch_measurements if row.epoch_ms is not None)
        mean = sum(values) / len(values)
        return (max(values) - min(values)) / mean if mean > 0 else math.inf

    def validate(self) -> None:
        if self.version != LABEL_RUN_RECORD_VERSION:
            raise ValueError(f"unsupported label run version {self.version!r}")
        _require_text(self.run_id, context="run_id")
        _require_sha256(self.configuration_id, context="configuration_id")
        if not isinstance(self.status, AttemptStatus):
            raise ValueError("status must be an AttemptStatus enum")
        if not isinstance(self.failure_stage, FailureStage):
            raise ValueError("failure_stage must be a FailureStage enum")
        self.fingerprints.validate(CorpusLayer.END_TO_END)
        _require_text(self.gpu_uuid, context="gpu_uuid")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise ValueError("label records must be measured on the frozen NRP V100 target")
        _require_sha256(self.capture_workload_sha256, context="capture_workload_sha256")
        _require_sha256(self.profile_workload_sha256, context="profile_workload_sha256")
        if self.capture_workload_sha256 != self.profile_workload_sha256:
            raise ValueError("capture/profile workload fingerprints must match")
        if not self.coverage_cell_ids or any(type(value) is not str or not value for value in self.coverage_cell_ids):
            raise ValueError("at least one coverage cell ID is required")
        _require_text(self.task_id, context="task_id")
        _require_text(self.execution_mode, context="execution_mode")
        if type(self.compile_completed_before_epoch_1) is not bool:
            raise ValueError("compile_completed_before_epoch_1 must be boolean")
        for name in (
            "expected_train_examples",
            "microbatch_size",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.expected_train_examples < 8:
            raise ValueError("expected_train_examples must supply at least eight examples")
        expected_batches = math.ceil(self.expected_train_examples / self.microbatch_size)
        expected_optimizer_steps = math.ceil(
            expected_batches / self.gradient_accumulation_steps
        )
        if self.total_epochs != 5 or self.warmup_epochs != (1, 2) or self.measured_epochs != (3, 4, 5):
            raise ValueError("label runs require five epochs with [1,2] warmup and [3,4,5] measured")
        for row in self.epoch_measurements:
            row.validate(accepted=self.status == AttemptStatus.ACCEPTED)
        self.cleanup.validate()
        if self.status == AttemptStatus.ACCEPTED:
            if self.failure_stage != FailureStage.NONE:
                raise ValueError("accepted records cannot have a failure stage")
            if self.completed_epochs != (1, 2, 3, 4, 5):
                raise ValueError("accepted records require all five complete epochs")
            if self.finite_loss_epochs != (1, 2, 3, 4, 5) or self.finite_gradient_epochs != (1, 2, 3, 4, 5):
                raise ValueError("accepted records require finite loss and gradients in all five epochs")
            if tuple(row.epoch for row in self.epoch_measurements) != self.measured_epochs:
                raise ValueError("accepted records require exactly three epoch measurements")
            for row in self.epoch_measurements:
                if (
                    row.examples_seen != self.expected_train_examples
                    or row.batches_seen != expected_batches
                    or row.microsteps != expected_batches
                    or row.optimizer_steps != expected_optimizer_steps
                ):
                    raise ValueError(
                        "accepted epoch traversal differs from its prepared-count, batch, or accumulation contract"
                    )
            if self.execution_mode == "compiled" and not self.compile_completed_before_epoch_1:
                raise ValueError("compiled runs must finish compilation before epoch 1")
            if not self.cleanup.passed:
                raise ValueError("accepted records require a passing cleanup gate")
            spread = self.epoch_time_relative_spread
            if spread is None or spread > 0.10:
                raise ValueError("accepted records exceed the 10% epoch stability gate")
            derived = self.aggregate_targets()
            if self.targets is None:
                raise ValueError("accepted records require all six aggregated targets")
            self.targets.validate(require_all_valid=True)
            if self.targets != derived:
                raise ValueError("stored targets do not equal the epoch 3-5 aggregate")
        else:
            if self.failure_stage == FailureStage.NONE:
                raise ValueError("rejected attempts require a failure stage")
            if self.targets is not None:
                raise ValueError("rejected attempts must not retain supervised targets")

    def validate_against_configuration(
        self,
        configuration: Any,
        task_entry: "TaskRegistryEntry",
    ) -> None:
        """Bind traversal values to the frozen manifest and task registry."""

        self.validate()
        configuration.validate()
        task_entry.validate()
        configuration_id = getattr(
            configuration,
            "candidate_id",
            getattr(configuration, "configuration_id", None),
        )
        execution = getattr(configuration, "execution", {})
        execution_mode = execution.get("mode") if isinstance(execution, Mapping) else None
        batch_plan = getattr(configuration, "batch_plan", None)
        planned_count = (
            getattr(batch_plan, "expected_train_examples", None)
            if batch_plan is not None
            else task_entry.expected_train_examples
        )
        if (
            self.configuration_id != configuration_id
            or self.task_id != getattr(configuration, "task_id", None)
            or self.task_id != task_entry.task_id
            or self.execution_mode != execution_mode
            or self.expected_train_examples != task_entry.expected_train_examples
            or self.expected_train_examples != planned_count
            or self.microbatch_size != getattr(configuration, "microbatch_size", None)
            or self.gradient_accumulation_steps
            != getattr(configuration, "gradient_accumulation_steps", None)
        ):
            raise ValueError("label run traversal envelope is not bound to its configuration/task")

    def bound_record_sha256(
        self,
        configuration: Any,
        task_entry: "TaskRegistryEntry",
    ) -> str:
        """Hash only after the record is proven against its frozen inputs."""

        self.validate_against_configuration(configuration, task_entry)
        return canonical_sha256(asdict(self))


def label_run_record_from_dict(value: Mapping[str, Any]) -> LabelRunRecord:
    """Strict loader used by resumable task completion/finalization gates."""

    if not isinstance(value, Mapping) or set(value) != set(LabelRunRecord.__dataclass_fields__):
        raise ValueError("serialized label-run schema differs")
    measurements = []
    for raw in value["epoch_measurements"]:
        if not isinstance(raw, Mapping) or set(raw) != set(EpochMeasurement.__dataclass_fields__):
            raise ValueError("serialized epoch-measurement schema differs")
        samples = tuple(TelemetrySample(**sample) for sample in raw["telemetry_samples"])
        measurements.append(EpochMeasurement(**{**dict(raw), "telemetry_samples": samples}))
    targets_raw = value["targets"]
    targets = None
    if targets_raw is not None:
        targets = TargetVector(
            values=tuple(targets_raw["values"]),
            validity=tuple(targets_raw["validity"]),
        )
    result = LabelRunRecord(
        **{
            **dict(value),
            "status": AttemptStatus(value["status"]),
            "failure_stage": FailureStage(value["failure_stage"]),
            "fingerprints": FingerprintBundle(**value["fingerprints"]),
            "coverage_cell_ids": tuple(value["coverage_cell_ids"]),
            "warmup_epochs": tuple(value["warmup_epochs"]),
            "measured_epochs": tuple(value["measured_epochs"]),
            "completed_epochs": tuple(value["completed_epochs"]),
            "finite_loss_epochs": tuple(value["finite_loss_epochs"]),
            "finite_gradient_epochs": tuple(value["finite_gradient_epochs"]),
            "epoch_measurements": tuple(measurements),
            "cleanup": GpuCleanupEvidence(**value["cleanup"]),
            "targets": targets,
        }
    )
    result.validate()
    return result


@dataclass(frozen=True)
class TaskRegistryEntry:
    registry_version: str
    task_id: str
    kaggle_kind: str
    kaggle_slug: str
    modality: str
    license_or_rules_url: str
    manual_acceptance_required: bool
    expected_archive_files: tuple[str, ...]
    required_file_patterns: tuple[str, ...]
    optional_file_patterns: tuple[str, ...]
    compressed_size_hint_bytes: int
    maximum_extracted_bytes: int
    expected_train_examples: int
    dataset_revision: str
    source_checksums: Mapping[str, str]
    split_recipe: Mapping[str, Any]
    target_schema: Mapping[str, Any]
    validation_rules: Mapping[str, Any]
    prepared_view_recipe: Mapping[str, Any]
    cache_policy: Mapping[str, Any]
    eviction_policy: Mapping[str, Any]

    def validate(self) -> None:
        if self.registry_version != TASK_REGISTRY_VERSION:
            raise ValueError("task registry version mismatch")
        if self.kaggle_kind not in {"competition", "dataset"}:
            raise ValueError("kaggle_kind must be competition or dataset")
        for name in ("task_id", "kaggle_slug", "modality", "license_or_rules_url", "dataset_revision"):
            _require_text(getattr(self, name), context=name)
        if type(self.manual_acceptance_required) is not bool:
            raise ValueError("manual_acceptance_required must be boolean")
        for name in (
            "compressed_size_hint_bytes",
            "maximum_extracted_bytes",
            "expected_train_examples",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_extracted_bytes < self.compressed_size_hint_bytes:
            raise ValueError("maximum extracted bytes cannot be smaller than compressed size hint")
        if self.expected_train_examples < 8:
            raise ValueError("prepared training view must supply at least eight examples")
        if not self.expected_archive_files or not self.required_file_patterns:
            raise ValueError("task registry requires archive and file validation patterns")
        for name, digest in self.source_checksums.items():
            _require_text(name, context="source checksum path")
            _require_sha256(digest, context=f"source checksum {name}")
        canonical_value(asdict(self))


@dataclass(frozen=True)
class ModelFamilyDefinition:
    registry_version: str
    family_id: str
    modality: str
    factory_id: str
    adapter_ids: tuple[str, ...]
    source_lineage: str
    source_sha256: str
    faithful_operation_assertions: tuple[str, ...]
    adjustable_architecture_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.registry_version != MODEL_REGISTRY_VERSION:
            raise ValueError("model registry version mismatch")
        for name in ("family_id", "modality", "factory_id", "source_lineage"):
            _require_text(getattr(self, name), context=name)
        _require_sha256(self.source_sha256, context="source_sha256")
        if not self.adapter_ids or not self.faithful_operation_assertions:
            raise ValueError("model family requires adapters and faithful operation assertions")


__all__ = [
    "AUXILIARY_TARGET_NAMES",
    "AttemptStatus",
    "AuxiliaryTargetVector",
    "CorpusLayer",
    "EPOCH_MEASUREMENT_VERSION",
    "EpochMeasurement",
    "FailureStage",
    "FingerprintBundle",
    "GpuCleanupEvidence",
    "LABEL_RUN_RECORD_VERSION",
    "LabelRunRecord",
    "label_run_record_from_dict",
    "MODEL_REGISTRY_VERSION",
    "ModelFamilyDefinition",
    "TARGET_NAMES",
    "TASK_REGISTRY_VERSION",
    "TaskRegistryEntry",
    "TargetVector",
    "TelemetrySample",
    "WORKLOAD_CONFIG_VERSION",
    "WorkloadConfiguration",
]
