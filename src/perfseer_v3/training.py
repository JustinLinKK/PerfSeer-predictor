"""Staged v3 encoder pretraining, teacher training, distillation, and calibration."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .baseline import canonical_json
from .features import (
    GraphBatchV3,
    GraphFeaturesV3,
    NormalizationStatsV3,
    batch_graph_features,
)
from .model import (
    OOM_FAILURE_STAGES,
    SeerNetV3,
    SeerNetV3Config,
    graph_batch_tensors,
)
from .op_registry import OperationRegistry
from .version import (
    FEATURE_SCHEMA_VERSION,
    GRAPH_IR_VERSION,
    LABEL_SCHEMA_VERSION,
    OP_REGISTRY_VERSION,
    STUDENT_MODEL_RELEASE,
    TEACHER_MODEL_RELEASE,
)


TARGET_NAMES = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_p95_sm_util_percent",
    "train_peak_vram_used_mib",
    "train_peak_torch_reserved_mib",
    "train_peak_memory_controller_util_percent",
)


class TrainingGateError(RuntimeError):
    """Raised before training when corpus/schema safety gates are not satisfied."""


@dataclass(frozen=True)
class TrainingConfigV3:
    run: dict[str, Any]
    features: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    gates: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfigV3":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("training config root must be a mapping")
        config = cls(
            run=dict(raw.get("run", {})),
            features=dict(raw.get("features", {})),
            model=dict(raw.get("model", {})),
            training=dict(raw.get("training", {})),
            gates=dict(raw.get("gates", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "graph_ir_version": GRAPH_IR_VERSION,
            "op_registry_version": OP_REGISTRY_VERSION,
            "capture_backend": "torch_export",
            "categorical_encoder": "hierarchical_embedding",
            "unknown_policy": "generic_hash",
            "graph_view": "coarsened_v3",
            "gpu_specific_model_pair": True,
            "per_operation_dtype": True,
            "optimizer_identity": "exact_family_hash",
            "scheduler_identity": "exact_family_hash",
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.features.get(key) != value
        ]
        if mismatches:
            raise ValueError("invalid v3 feature config: " + ", ".join(mismatches))
        if not self.features.get("include_training_graph"):
            raise ValueError("v3 training config must include the training graph")
        if not self.features.get("include_tensor_liveness"):
            raise ValueError("v3 training config must include tensor liveness")
        release = self.run.get("model_release")
        if release not in {TEACHER_MODEL_RELEASE, STUDENT_MODEL_RELEASE}:
            raise ValueError(f"invalid v3 model release {release!r}")
        if self.run.get("label_schema_version") != LABEL_SCHEMA_VERSION:
            raise ValueError("label schema version mismatch")
        if str(self.training.get("initialization", "")).startswith("v2"):
            raise ValueError("v2 checkpoints cannot initialize changed v3 feature semantics")
        if self.model.get("node_identity_fusion", "additive") not in {
            "additive",
            "concatenation",
        }:
            raise ValueError("invalid node identity fusion")
        if self.model.get("pooling_mode", "existing") not in {
            "existing",
            "phase_aware",
        }:
            raise ValueError("invalid pooling mode")
        if int(self.model.get("num_outputs", len(TARGET_NAMES))) != len(TARGET_NAMES):
            raise ValueError("v3 configs must preserve the six scheduler outputs")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    def model_config(
        self,
        registry: OperationRegistry,
        layout: Any,
        **overrides: Any,
    ) -> SeerNetV3Config:
        values = dict(self.model)
        values.pop("capacity_candidate", None)
        values.update(overrides)
        return SeerNetV3Config.from_registry(registry, layout, **values)


@dataclass(frozen=True)
class DatasetGateReport:
    strict_capture_rate: float
    complete_encoding_rate: float
    unknown_gpu_time_fraction: float | None
    source_group_isolated: bool
    measured_gpu_time: bool
    dataset_fingerprint: str
    split_fingerprint: str


def assert_training_ready(
    config: TrainingConfigV3,
    report: DatasetGateReport,
    registry: OperationRegistry,
) -> None:
    failures: list[str] = []
    if config.training.get("require_training_approved_registry", True) and not registry.training_approved:
        failures.append("operator registry is not training-approved from measured GPU time")
    if report.strict_capture_rate < float(config.gates["minimum_strict_capture_rate"]):
        failures.append("strict capture rate is below the configured gate")
    if report.complete_encoding_rate < float(config.gates["minimum_complete_encoding_rate"]):
        failures.append("complete encoding rate is below the configured gate")
    if not report.measured_gpu_time or report.unknown_gpu_time_fraction is None:
        failures.append("measured GPU-time operation coverage is missing")
    elif report.unknown_gpu_time_fraction > float(
        config.gates["maximum_unknown_gpu_time_fraction"]
    ):
        failures.append("unknown GPU-time fraction exceeds the configured gate")
    if config.gates.get("require_source_group_isolation", True) and not report.source_group_isolated:
        failures.append("source-family split leakage was detected")
    if failures:
        raise TrainingGateError("; ".join(failures))


@dataclass(frozen=True)
class TrainingSampleV3:
    features: GraphFeaturesV3
    target: torch.Tensor
    oom: float = 0.0
    oom_stage: int = 0
    peak_live_bytes: float | None = None
    domain_weight: float = 1.0

    def validate(self) -> None:
        self.features.validate()
        if self.target.shape != (len(TARGET_NAMES),):
            raise ValueError("training target must follow the six-target v3 contract")
        if not torch.isfinite(self.target).all():
            raise ValueError("training target must be finite")
        if self.oom not in {0.0, 1.0}:
            raise ValueError("OOM target must be binary")
        if not 0 <= self.oom_stage < len(OOM_FAILURE_STAGES):
            raise ValueError("OOM failure stage is out of range")
        if self.oom == 0.0 and self.oom_stage != 0:
            raise ValueError("non-OOM samples must use the none failure stage")
        if self.oom == 1.0 and self.oom_stage == 0:
            raise ValueError("OOM samples must identify a failure stage")
        if self.peak_live_bytes is not None and self.peak_live_bytes < 0:
            raise ValueError("peak live bytes must be nonnegative")
        if self.domain_weight <= 0:
            raise ValueError("domain weight must be positive")


class EncoderPretrainer(nn.Module):
    def __init__(self, model: SeerNetV3) -> None:
        super().__init__()
        self.encoder = model.node_encoder
        hidden = model.config.hidden
        self.family_head = nn.Linear(hidden, model.config.num_families)
        self.exact_head = nn.Linear(hidden, model.config.num_exact_ops)
        self.cost_head = nn.Linear(hidden, 2)

    def forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            batch.x_cont,
            batch.op_exact_id,
            batch.op_family_id,
            batch.op_hash_id,
            batch.op_overload_hash_id,
            batch.phase_id,
            batch.input_dtype_id,
            batch.dtype_id,
            batch.accumulation_dtype_id,
            batch.backend_id,
            batch.feature_quality_id,
            batch.layout_id,
            batch.rank_id,
            batch.node_flags,
        )
        return self.family_head(hidden), self.exact_head(hidden), self.cost_head(hidden)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _model_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _batch_to_device(batch: GraphBatchV3, device: torch.device) -> GraphBatchV3:
    """Move every tensor in the frozen graph batch while preserving its layout."""

    return GraphBatchV3(
        **{
            name: (
                value.to(device, non_blocking=device.type == "cuda")
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in vars(batch).items()
        }
    )


def _autocast_context(device: torch.device, dtype: torch.dtype | None) -> Any:
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    if scaler is None:
        loss.backward()
        optimizer.step()
        return
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def encoder_pretrain_step(
    module: EncoderPretrainer,
    samples: Sequence[TrainingSampleV3],
    optimizer: torch.optim.Optimizer,
    *,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    device = _model_device(module)
    batch = _batch_to_device(
        batch_graph_features([sample.features for sample in samples]),
        device,
    )
    if batch.x_cont.size(0) == 0:
        return 0.0
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, autocast_dtype):
        family_logits, exact_logits, costs = module(batch)
        cost_indices = (7, 9)  # FLOPs and bytes-read in the named v3 layout.
        cost_target = torch.log1p(batch.x_cont[:, cost_indices].clamp_min(0.0))
        loss = (
            F.cross_entropy(family_logits, batch.op_family_id)
            + F.cross_entropy(exact_logits, batch.op_exact_id)
            + F.smooth_l1_loss(costs, cost_target)
        )
    _optimizer_step(loss, optimizer, scaler)
    return float(loss.detach())


def _supervised_loss(
    output: Any,
    targets: torch.Tensor,
    oom: torch.Tensor,
    oom_stage: torch.Tensor,
    peak_live_bytes: torch.Tensor,
    domain_weights: torch.Tensor,
) -> torch.Tensor:
    squared = (output.prediction - targets).square()
    heteroscedastic = 0.5 * (
        torch.exp(-output.log_variance) * squared + output.log_variance
    )
    regression = (heteroscedastic.mean(dim=1) * domain_weights).mean()
    oom_loss = F.binary_cross_entropy_with_logits(output.oom_logit.flatten(), oom)
    oom_stage_loss = F.cross_entropy(output.oom_stage_logits, oom_stage)
    peak_mask = torch.isfinite(peak_live_bytes)
    peak_loss = output.prediction.new_zeros(())
    if bool(peak_mask.any()):
        peak_target = torch.log1p(peak_live_bytes[peak_mask])
        peak_loss = F.smooth_l1_loss(
            output.peak_live_bytes_log1p.flatten()[peak_mask],
            peak_target,
        )
    confidence_target = 1.0 - oom
    confidence_loss = F.mse_loss(output.confidence.flatten(), confidence_target)
    return (
        regression
        + 0.1 * oom_loss
        + 0.05 * oom_stage_loss
        + 0.05 * peak_loss
        + 0.05 * confidence_loss
    )


def teacher_train_step(
    model: SeerNetV3,
    samples: Sequence[TrainingSampleV3],
    optimizer: torch.optim.Optimizer,
    *,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    for sample in samples:
        sample.validate()
    device = _model_device(model)
    batch = _batch_to_device(
        batch_graph_features([sample.features for sample in samples]),
        device,
    )
    targets = torch.stack([sample.target for sample in samples]).to(device)
    oom = torch.tensor(
        [sample.oom for sample in samples], dtype=torch.float32, device=device
    )
    oom_stage = torch.tensor(
        [sample.oom_stage for sample in samples], dtype=torch.long, device=device
    )
    peak_live_bytes = torch.tensor(
        [
            float("nan") if sample.peak_live_bytes is None else sample.peak_live_bytes
            for sample in samples
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        [sample.domain_weight for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, autocast_dtype):
        output = model(*graph_batch_tensors(batch))
        loss = _supervised_loss(
            output,
            targets,
            oom,
            oom_stage,
            peak_live_bytes,
            weights,
        )
    _optimizer_step(loss, optimizer, scaler)
    return float(loss.detach())


def student_distill_step(
    student: SeerNetV3,
    teacher: SeerNetV3,
    samples: Sequence[TrainingSampleV3],
    optimizer: torch.optim.Optimizer,
    *,
    hard_label_weight: float,
    representation_weight: float = 0.05,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    if not 0.0 <= hard_label_weight <= 1.0:
        raise ValueError("hard_label_weight must be in [0, 1]")
    if representation_weight < 0:
        raise ValueError("representation_weight must be nonnegative")
    for sample in samples:
        sample.validate()
    device = _model_device(student)
    teacher_device = _model_device(teacher)
    if teacher_device != device:
        raise ValueError("teacher and student must be on the same device")
    batch = _batch_to_device(
        batch_graph_features([sample.features for sample in samples]),
        device,
    )
    targets = torch.stack([sample.target for sample in samples]).to(device)
    oom = torch.tensor(
        [sample.oom for sample in samples], dtype=torch.float32, device=device
    )
    oom_stage = torch.tensor(
        [sample.oom_stage for sample in samples], dtype=torch.long, device=device
    )
    peak_live_bytes = torch.tensor(
        [
            float("nan") if sample.peak_live_bytes is None else sample.peak_live_bytes
            for sample in samples
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        [sample.domain_weight for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    teacher.eval()

    def relational_similarity(embedding: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embedding, dim=-1)
        return normalized @ normalized.transpose(0, 1)

    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        teacher_output = teacher(*graph_batch_tensors(batch))
    with _autocast_context(device, autocast_dtype):
        output = student(*graph_batch_tensors(batch))
        hard = _supervised_loss(
            output,
            targets,
            oom,
            oom_stage,
            peak_live_bytes,
            weights,
        )
        soft = (
            F.smooth_l1_loss(output.prediction, teacher_output.prediction)
            + 0.05 * F.smooth_l1_loss(output.oom_logit, teacher_output.oom_logit)
            + 0.05
            * F.smooth_l1_loss(
                output.oom_stage_logits,
                teacher_output.oom_stage_logits,
            )
            + 0.05
            * F.smooth_l1_loss(
                output.peak_live_bytes_log1p,
                teacher_output.peak_live_bytes_log1p,
            )
            + 0.05 * F.smooth_l1_loss(output.confidence, teacher_output.confidence)
        )
        uncertainty = F.smooth_l1_loss(
            output.log_variance,
            teacher_output.log_variance,
        )
        graph_relation = F.smooth_l1_loss(
            relational_similarity(output.graph_embedding),
            relational_similarity(teacher_output.graph_embedding),
        )
        student_phase = output.phase_embedding.flatten(0, 1)
        teacher_phase = teacher_output.phase_embedding.flatten(0, 1)
        phase_relation = F.smooth_l1_loss(
            relational_similarity(student_phase),
            relational_similarity(teacher_phase),
        )
        representation = graph_relation + phase_relation
        loss = (
            hard_label_weight * hard
            + (1.0 - hard_label_weight) * soft
            + 0.01 * uncertainty
            + representation_weight * representation
        )
    _optimizer_step(loss, optimizer, scaler)
    return float(loss.detach())


@dataclass(frozen=True)
class LinearCalibrationV3:
    slope: tuple[float, ...]
    intercept: tuple[float, ...]

    def apply(self, prediction: torch.Tensor) -> torch.Tensor:
        slope = prediction.new_tensor(self.slope)
        intercept = prediction.new_tensor(self.intercept)
        return prediction * slope + intercept


@dataclass(frozen=True)
class BinaryTemperatureCalibrationV3:
    temperature: float

    def apply_probability(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            raise ValueError("calibration temperature must be positive")
        return torch.sigmoid(logits / self.temperature)


@dataclass(frozen=True)
class UncertaintyCalibrationV3:
    log_variance_offset: tuple[float, ...]

    def apply_log_variance(self, log_variance: torch.Tensor) -> torch.Tensor:
        if log_variance.size(-1) != len(self.log_variance_offset):
            raise ValueError("uncertainty calibration does not match target width")
        return log_variance + log_variance.new_tensor(self.log_variance_offset)


def fit_linear_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> LinearCalibrationV3:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("calibration prediction/target must have matching [N, T] shapes")
    slopes: list[float] = []
    intercepts: list[float] = []
    for index in range(prediction.size(1)):
        x = prediction[:, index].double()
        y = target[:, index].double()
        x_mean, y_mean = x.mean(), y.mean()
        variance = ((x - x_mean) ** 2).sum()
        slope = ((x - x_mean) * (y - y_mean)).sum() / variance if variance > 1e-12 else x.new_tensor(1.0)
        intercept = y_mean - slope * x_mean
        slopes.append(float(slope))
        intercepts.append(float(intercept))
    return LinearCalibrationV3(tuple(slopes), tuple(intercepts))


def fit_binary_temperature_calibration(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> BinaryTemperatureCalibrationV3:
    logits = logits.detach().float().flatten().cpu()
    target = target.detach().float().flatten().cpu()
    if logits.shape != target.shape or logits.numel() == 0:
        raise ValueError("OOM calibration requires matching nonempty logits and targets")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("OOM calibration inputs must be finite")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("OOM calibration targets must be binary")
    candidates = torch.logspace(
        float(np.log10(0.05)),
        float(np.log10(20.0)),
        201,
    )
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(logits / value, target) for value in candidates]
    )
    temperature = float(candidates[int(losses.argmin())])
    return BinaryTemperatureCalibrationV3(temperature=temperature)


def fit_uncertainty_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
) -> UncertaintyCalibrationV3:
    if (
        prediction.shape != target.shape
        or prediction.shape != log_variance.shape
        or prediction.ndim != 2
        or prediction.size(0) == 0
    ):
        raise ValueError("uncertainty calibration requires matching nonempty [N, T] tensors")
    if not all(torch.isfinite(value).all() for value in (prediction, target, log_variance)):
        raise ValueError("uncertainty calibration inputs must be finite")
    standardized_squared_error = (
        (prediction.double() - target.double()).square()
        * torch.exp(-log_variance.double().clamp(-20.0, 20.0))
    )
    scale = standardized_squared_error.mean(dim=0).clamp(1e-6, 1e6)
    return UncertaintyCalibrationV3(
        tuple(float(value) for value in torch.log(scale))
    )


def checkpoint_metadata(
    *,
    config: TrainingConfigV3,
    model_config: SeerNetV3Config,
    sample: GraphFeaturesV3,
    registry: OperationRegistry,
    dataset_gate: DatasetGateReport,
    normalization: NormalizationStatsV3 | None = None,
    coarsening_sha256: str,
) -> dict[str, Any]:
    return {
        "model_release": config.run["model_release"],
        "graph_ir_version": GRAPH_IR_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": sample.layout.feature_schema_sha256,
        "operator_registry_version": OP_REGISTRY_VERSION,
        "operator_registry_sha256": registry.sha256,
        "ordered_feature_layout": asdict(sample.layout),
        "normalization_sha256": normalization.sha256 if normalization else None,
        "coarsening_sha256": coarsening_sha256,
        "target_names": list(TARGET_NAMES),
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_gate.dataset_fingerprint,
        "split_fingerprint": dataset_gate.split_fingerprint,
        "config_sha256": config.sha256,
        "model_config": model_config.to_dict(),
        "trainable_parameter_count": trainable_parameter_count(model_config),
        "output_contract_version": "perfseer_v3_outputs_v2",
        "optional_output_names": [
            "log_variance",
            "oom_probability",
            "oom_failure_stage",
            "confidence",
            "peak_live_bytes_log1p",
        ],
        "initialization": config.training["initialization"],
        "v2_checkpoint_loaded": False,
    }


def trainable_parameter_count(config: SeerNetV3Config) -> int:
    """Count exact configured parameters without allocating their storage."""

    with torch.device("meta"):
        model = SeerNetV3(config)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@dataclass(frozen=True)
class TinyTrainingResult:
    pretrain_loss: float
    teacher_loss: float
    student_loss: float
    teacher: SeerNetV3
    student: SeerNetV3


def run_tiny_training_smoke(
    samples: Sequence[TrainingSampleV3],
    *,
    seed: int = 42,
) -> TinyTrainingResult:
    if not samples:
        raise ValueError("tiny training smoke requires samples")
    _seed_everything(seed)
    registry = OperationRegistry.load()
    layout = samples[0].features.layout
    teacher_config = SeerNetV3Config.from_registry(
        registry,
        layout,
        hidden=24,
        num_blocks=2,
        exact_embedding_dim=12,
        family_embedding_dim=8,
        hash_embedding_dim=8,
        phase_embedding_dim=4,
        dtype_embedding_dim=4,
        dropout=0.0,
    )
    student_config = SeerNetV3Config.from_registry(
        registry,
        layout,
        hidden=16,
        num_blocks=1,
        exact_embedding_dim=12,
        family_embedding_dim=8,
        hash_embedding_dim=8,
        phase_embedding_dim=4,
        dtype_embedding_dim=4,
        dropout=0.0,
    )
    teacher = SeerNetV3(teacher_config)
    student = SeerNetV3(student_config)
    pretrainer = EncoderPretrainer(teacher)
    pretrain_optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)
    pretrain_loss = encoder_pretrain_step(pretrainer, samples, pretrain_optimizer)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    teacher_loss = teacher_train_step(teacher, samples, teacher_optimizer)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    student_loss = student_distill_step(
        student,
        teacher,
        samples,
        student_optimizer,
        hard_label_weight=0.6,
    )
    losses = (pretrain_loss, teacher_loss, student_loss)
    if not all(np.isfinite(value) and value >= 0 for value in losses):
        raise RuntimeError(f"tiny v3 training produced invalid losses: {losses}")
    return TinyTrainingResult(pretrain_loss, teacher_loss, student_loss, teacher, student)


__all__ = [
    "BinaryTemperatureCalibrationV3",
    "DatasetGateReport",
    "EncoderPretrainer",
    "LinearCalibrationV3",
    "TARGET_NAMES",
    "TinyTrainingResult",
    "TrainingConfigV3",
    "TrainingGateError",
    "TrainingSampleV3",
    "UncertaintyCalibrationV3",
    "assert_training_ready",
    "checkpoint_metadata",
    "encoder_pretrain_step",
    "fit_linear_calibration",
    "fit_binary_temperature_calibration",
    "fit_uncertainty_calibration",
    "run_tiny_training_smoke",
    "student_distill_step",
    "teacher_train_step",
    "trainable_parameter_count",
]
