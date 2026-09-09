"""Staged v3 encoder pretraining, teacher training, distillation, and calibration."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
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
from .hardware_transfer import (
    PairedResidualTransformV3,
    adapter_identity_regularization_loss,
    adapter_regularization_loss,
)
from .model import (
    OOM_FAILURE_STAGES,
    SeerNetV3,
    SeerNetV3Config,
    _scatter_max as _scatter_max_for_pretraining,
    _scatter_mean as _scatter_mean_for_pretraining,
    _scatter_sum as _scatter_sum_for_pretraining,
    graph_batch_tensors,
)
from .op_registry import OperationRegistry
from .version import (
    FEATURE_SCHEMA_VERSION,
    GRAPH_IR_VERSION,
    LABEL_SCHEMA_VERSION,
    OP_REGISTRY_VERSION,
    OUTPUT_CONTRACT_VERSION,
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

PRETRAINING_OBJECTIVES = (
    "masked_operation_family",
    "masked_exact_operation",
    "masked_flop_byte_workspace_reconstruction",
    "masked_tensor_shape_reconstruction",
    "masked_input_dtype_reconstruction",
    "masked_output_dtype_reconstruction",
    "masked_layout_reconstruction",
    "masked_rank_reconstruction",
    "masked_phase_prediction",
    "safe_configuration_variant_contrastive",
    "graph_compute_memory_regime_classification",
)
WORKLOAD_REGIMES = ("launch_bound", "memory_bound", "compute_bound", "capacity_bound")


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
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("training config root must be a mapping")
        if "extends" in raw:
            parent_path = (config_path.parent / str(raw["extends"])).resolve()
            parent_raw = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
            if not isinstance(parent_raw, Mapping) or "extends" in parent_raw:
                raise ValueError("training config extends requires one concrete parent")
            merged: dict[str, Any] = {
                name: dict(parent_raw.get(name, {}))
                for name in ("run", "features", "model", "training", "gates")
            }
            for name in merged:
                override = raw.get(name, {})
                if not isinstance(override, Mapping):
                    raise ValueError(f"training config section {name!r} must be a mapping")
                merged[name].update(override)
            raw = merged
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
        for name in (
            "validation_interval_epochs",
            "early_stopping_patience_checks",
        ):
            if int(self.training.get(name, 0)) < 1:
                raise ValueError(f"training config {name} must be positive")
        if float(self.training.get("early_stopping_min_delta", -1.0)) < 0:
            raise ValueError("training config early_stopping_min_delta must be nonnegative")

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
    source_group: str = ""
    graph_signature: str = ""
    workload_regime: int = 0
    regression_available: bool = True

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
        if bool(self.source_group) != bool(self.graph_signature):
            raise ValueError("source group and graph signature must be provided together")
        if not 0 <= self.workload_regime < len(WORKLOAD_REGIMES):
            raise ValueError("workload regime is out of range")
        if type(self.regression_available) is not bool:
            raise ValueError("regression availability must be boolean")


@dataclass(frozen=True)
class PairedTransferSampleV3:
    """One exact A10/target configuration pair used for adapter training."""

    features: GraphFeaturesV3
    base_target: torch.Tensor
    target: torch.Tensor
    oom: float = 0.0
    oom_stage: int = 0
    peak_live_bytes: float | None = None
    domain_weight: float = 1.0
    regression_available: bool = True
    evidence_kind: str = "paired_label"

    def validate(self) -> None:
        TrainingSampleV3(
            features=self.features,
            target=self.target,
            oom=self.oom,
            oom_stage=self.oom_stage,
            peak_live_bytes=self.peak_live_bytes,
            domain_weight=self.domain_weight,
            regression_available=self.regression_available,
        ).validate()
        if self.base_target.shape != (len(TARGET_NAMES),):
            raise ValueError("paired A10 target must follow the six-target order")
        if not torch.isfinite(self.base_target).all():
            raise ValueError("paired A10 target must be finite")
        if self.evidence_kind not in {
            "paired_label",
            "oom_attempt",
            "memory_probe",
        }:
            raise ValueError("paired transfer evidence kind is invalid")


class EncoderPretrainer(nn.Module):
    """Self-supervise the complete hardware-independent workload backbone."""

    def __init__(self, model: SeerNetV3) -> None:
        super().__init__()
        # Register only the reusable workload path. Hardware encoders, adapters,
        # metric heads, and target calibration must never receive Stage A
        # gradients.
        self.node_encoder = model.node_encoder
        self.edge_encoder = model.edge_encoder
        self.global_encoder = model.global_encoder
        self.blocks = model.blocks
        self.phase_encoder = model.phase_encoder
        self.phase_fusion = model.phase_fusion
        self.phase_fusion_norm = model.phase_fusion_norm
        self.use_phase_aware_pooling = model.use_phase_aware_pooling
        self.num_phases = model.num_phases
        self.hidden = model.hidden
        hidden = model.config.hidden
        self.family_head = nn.Linear(hidden, model.config.num_families)
        self.exact_head = nn.Linear(hidden, model.config.num_exact_ops)
        self.cost_head = nn.Linear(hidden, 3)
        self.shape_head = nn.Linear(hidden, 10)
        self.input_dtype_head = nn.Linear(hidden, model.config.num_dtypes)
        self.dtype_head = nn.Linear(hidden, model.config.num_dtypes)
        self.layout_head = nn.Linear(hidden, model.config.num_layouts)
        self.rank_head = nn.Linear(hidden, model.config.num_ranks)
        self.phase_head = nn.Linear(hidden, model.config.num_phases)
        self.regime_head = nn.Linear(hidden, len(WORKLOAD_REGIMES))

    @property
    def objective_names(self) -> tuple[str, ...]:
        return PRETRAINING_OBJECTIVES

    def forward(
        self,
        batch: GraphBatchV3,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        nodes = self.node_encoder(
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
        edges = self.edge_encoder(
            batch.edge_cont,
            batch.edge_flags,
            batch.edge_role_id,
            batch.edge_source_slot_id,
            batch.edge_destination_slot_id,
            batch.edge_dtype_id,
            batch.edge_layout_id,
            batch.edge_rank_id,
            batch.edge_alias_id,
            batch.edge_dynamic_quality_id,
            batch.edge_phase_transition_id,
        )
        globals_ = self.global_encoder(
            batch.u_cont,
            batch.quality,
            batch.precision_id,
            batch.optimizer_id,
            batch.optimizer_family_id,
            batch.optimizer_hash_id,
            batch.scheduler_id,
            batch.scheduler_family_id,
            batch.scheduler_hash_id,
            batch.capture_mode_id,
            batch.capture_backend_id,
            batch.dynamic_shape_id,
            batch.training_mode_id,
        )
        for block in self.blocks:
            nodes, edges, globals_ = block(
                nodes,
                edges,
                globals_,
                batch.edge_index,
                batch.batch,
            )
        graph_count = globals_.size(0)
        phase_index = batch.batch * self.num_phases + batch.phase_id
        phase_count = graph_count * self.num_phases
        phase_mean = _scatter_mean_for_pretraining(nodes, phase_index, phase_count)
        phase_max = _scatter_max_for_pretraining(nodes, phase_index, phase_count)
        phase_presence = _scatter_sum_for_pretraining(
            nodes.new_ones((nodes.size(0), 1)),
            phase_index,
            phase_count,
        ).gt(0).to(nodes.dtype)
        phase_hidden = self.phase_encoder(torch.cat([phase_mean, phase_max], dim=-1))
        phase_hidden = phase_hidden * phase_presence
        phase_hidden = phase_hidden.view(graph_count, self.num_phases, self.hidden)
        if self.use_phase_aware_pooling:
            fused = self.phase_fusion(
                torch.cat([globals_, phase_hidden.flatten(1)], dim=-1)
            )
            globals_ = self.phase_fusion_norm(globals_ + fused)
        return (
            self.family_head(nodes),
            self.exact_head(nodes),
            self.cost_head(nodes),
            self.shape_head(nodes),
            self.input_dtype_head(nodes),
            self.dtype_head(nodes),
            self.layout_head(nodes),
            self.rank_head(nodes),
            self.phase_head(nodes),
            self.regime_head(globals_),
            globals_,
        )


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


def _masked_pretraining_batch(
    batch: GraphBatchV3,
    *,
    mask_probability: float = 0.15,
) -> tuple[GraphBatchV3, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Mask workload semantics while retaining graph topology and global context."""

    node_count = batch.x_cont.size(0)
    mask = torch.rand(node_count, device=batch.x_cont.device) < mask_probability
    for graph_index in range(batch.u_cont.size(0)):
        graph_nodes = torch.nonzero(batch.batch == graph_index, as_tuple=False).flatten()
        if graph_nodes.numel() and not bool(mask[graph_nodes].any()):
            mask[graph_nodes[0]] = True
    reconstruction_fields = (
        "flops",
        "bytes_read",
        "estimated_workspace_bytes",
        "input_numel",
        "output_numel",
        "input_rank_max",
        "output_rank_max",
        "input_dimension_min",
        "input_dimension_max",
        "input_dimension_mean",
        "output_dimension_min",
        "output_dimension_max",
        "output_dimension_mean",
    )
    indices = tuple(batch.layout.node_continuous_fields.index(name) for name in reconstruction_fields)
    cost_indices = indices[:3]
    shape_indices = indices[3:]
    masked_cont = batch.x_cont.clone()
    if bool(mask.any()):
        for index in indices:
            masked_cont[mask, index] = 0.0

    def mask_ids(values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        result[mask] = 0
        return result

    return (
        replace(
            batch,
            x_cont=masked_cont,
            op_exact_id=mask_ids(batch.op_exact_id),
            op_family_id=mask_ids(batch.op_family_id),
            op_hash_id=mask_ids(batch.op_hash_id),
            op_overload_hash_id=mask_ids(batch.op_overload_hash_id),
            phase_id=mask_ids(batch.phase_id),
            input_dtype_id=mask_ids(batch.input_dtype_id),
            dtype_id=mask_ids(batch.dtype_id),
            accumulation_dtype_id=mask_ids(batch.accumulation_dtype_id),
            layout_id=mask_ids(batch.layout_id),
            rank_id=mask_ids(batch.rank_id),
        ),
        mask,
        cost_indices,
        shape_indices,
    )


def _safe_variant_contrastive_loss(
    embeddings: torch.Tensor,
    samples: Sequence[TrainingSampleV3],
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE over different graph signatures from the same safe source group."""

    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    normalized = F.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.transpose(0, 1) / temperature
    eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
    positive = torch.zeros_like(eye)
    for left, left_sample in enumerate(samples):
        if not left_sample.source_group:
            continue
        for right, right_sample in enumerate(samples):
            positive[left, right] = bool(
                left != right
                and left_sample.source_group == right_sample.source_group
                and left_sample.graph_signature != right_sample.graph_signature
            )
    usable = positive.any(dim=1)
    if not bool(usable.any()):
        return embeddings.sum() * 0.0
    denominator = torch.logsumexp(logits.masked_fill(eye, -float("inf")), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -float("inf")), dim=1)
    return (denominator[usable] - numerator[usable]).mean()


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
    masked_batch, mask, cost_indices, shape_indices = _masked_pretraining_batch(batch)
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, autocast_dtype):
        (
            family_logits,
            exact_logits,
            costs,
            shapes,
            input_dtype_logits,
            dtype_logits,
            layout_logits,
            rank_logits,
            phase_logits,
            regime_logits,
            graph_embeddings,
        ) = module(masked_batch)
        cost_target = batch.x_cont[:, cost_indices]
        shape_target = batch.x_cont[:, shape_indices]
        regime_target = torch.tensor(
            [sample.workload_regime for sample in samples],
            dtype=torch.long,
            device=device,
        )
        loss = (
            F.cross_entropy(family_logits[mask], batch.op_family_id[mask])
            + F.cross_entropy(exact_logits[mask], batch.op_exact_id[mask])
            + F.smooth_l1_loss(costs[mask], cost_target[mask])
            + F.smooth_l1_loss(shapes[mask], shape_target[mask])
            + 0.25 * F.cross_entropy(
                input_dtype_logits[mask], batch.input_dtype_id[mask]
            )
            + 0.25 * F.cross_entropy(dtype_logits[mask], batch.dtype_id[mask])
            + 0.25 * F.cross_entropy(layout_logits[mask], batch.layout_id[mask])
            + 0.25 * F.cross_entropy(rank_logits[mask], batch.rank_id[mask])
            + 0.25 * F.cross_entropy(phase_logits[mask], batch.phase_id[mask])
            + 0.25 * F.cross_entropy(regime_logits, regime_target)
            + 0.1 * _safe_variant_contrastive_loss(graph_embeddings, samples)
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
    regression_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared = (output.prediction - targets).square()
    heteroscedastic = 0.5 * (
        torch.exp(-output.log_variance) * squared + output.log_variance
    )
    positive_indexes = (0, 3, 4)
    utilization_indexes = (1, 2, 5)
    positive_prediction = torch.log1p(
        output.prediction[:, positive_indexes].clamp_min(0)
    )
    positive_target = torch.log1p(targets[:, positive_indexes].clamp_min(0))
    utilization_prediction = torch.logit(
        (output.prediction[:, utilization_indexes] / 100.0).clamp(0.005, 0.995)
    )
    utilization_target = torch.logit(
        (targets[:, utilization_indexes] / 100.0).clamp(0.005, 0.995)
    )
    transformed = 0.5 * (
        F.smooth_l1_loss(
            positive_prediction,
            positive_target,
            reduction="none",
        ).mean(dim=1)
        + F.smooth_l1_loss(
            utilization_prediction,
            utilization_target,
            reduction="none",
        ).mean(dim=1)
    )
    # Keep heteroscedastic uncertainty in deployable output units while also
    # fitting the output-appropriate log/logit geometry required by the base
    # and target objectives.
    per_sample_regression = (
        heteroscedastic.mean(dim=1) + 0.25 * transformed
    ) * domain_weights
    if regression_mask is None:
        regression = per_sample_regression.mean()
    else:
        if regression_mask.shape != per_sample_regression.shape:
            raise ValueError("regression mask must contain one value per sample")
        regression = (
            per_sample_regression * regression_mask.to(per_sample_regression.dtype)
        ).sum() / regression_mask.to(per_sample_regression.dtype).sum().clamp_min(1.0)
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


def _paired_batch_tensors(
    samples: Sequence[PairedTransferSampleV3],
    device: torch.device,
) -> tuple[
    GraphBatchV3,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    for sample in samples:
        sample.validate()
    batch = _batch_to_device(
        batch_graph_features([sample.features for sample in samples]),
        device,
    )
    base_targets = torch.stack([sample.base_target for sample in samples]).to(device)
    targets = torch.stack([sample.target for sample in samples]).to(device)
    oom = torch.tensor([sample.oom for sample in samples], dtype=torch.float32, device=device)
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
    regression_mask = torch.tensor(
        [sample.regression_available for sample in samples],
        dtype=torch.bool,
        device=device,
    )
    return (
        batch,
        base_targets,
        targets,
        oom,
        oom_stage,
        peak_live_bytes,
        weights,
        regression_mask,
    )


def target_teacher_adapter_step(
    model: SeerNetV3,
    samples: Sequence[PairedTransferSampleV3],
    optimizer: torch.optim.Optimizer,
    *,
    residual_transform: PairedResidualTransformV3 | None = None,
    adapter_regularization_weight: float = 1e-4,
    l2_sp_weight: float = 1e-4,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train only the target teacher adapter against paired residual labels."""

    if adapter_regularization_weight < 0 or l2_sp_weight < 0:
        raise ValueError("adapter regularization weights must be nonnegative")
    transform = residual_transform or PairedResidualTransformV3()
    device = _model_device(model)
    (
        batch,
        base,
        targets,
        oom,
        oom_stage,
        peak_live,
        weights,
        regression_mask,
    ) = _paired_batch_tensors(samples, device)
    residual_targets = transform.encode(base, targets)
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, autocast_dtype):
        output = model(*graph_batch_tensors(batch))
        supervised = _supervised_loss(
            output,
            targets,
            oom,
            oom_stage,
            peak_live,
            weights,
            regression_mask,
        )
        residual_loss = (
            F.smooth_l1_loss(
                output.paired_residual[regression_mask],
                residual_targets[regression_mask],
            )
            if bool(regression_mask.any())
            else output.prediction.new_zeros(())
        )
        identity_regularization = adapter_identity_regularization_loss(model)
        l2_sp = (
            adapter_regularization_loss(model, reference=reference_parameters)
            if reference_parameters is not None
            else output.prediction.new_zeros(())
        )
        loss = (
            supervised
            + residual_loss
            + adapter_regularization_weight * identity_regularization
            + l2_sp_weight * l2_sp
        )
    _optimizer_step(loss, optimizer, scaler)
    return float(loss.detach())


def target_student_adapter_step(
    student: SeerNetV3,
    teacher: SeerNetV3,
    samples: Sequence[PairedTransferSampleV3],
    optimizer: torch.optim.Optimizer,
    *,
    hard_label_weight: float = 0.6,
    representation_weight: float = 0.05,
    residual_transform: PairedResidualTransformV3 | None = None,
    adapter_regularization_weight: float = 1e-4,
    l2_sp_weight: float = 1e-4,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Distill a target student only from the adapted teacher for that GPU."""

    if not 0 <= hard_label_weight <= 1:
        raise ValueError("hard_label_weight must be in [0, 1]")
    if (
        representation_weight < 0
        or adapter_regularization_weight < 0
        or l2_sp_weight < 0
    ):
        raise ValueError("student transfer weights must be nonnegative")
    device = _model_device(student)
    if _model_device(teacher) != device:
        raise ValueError("target teacher and student must use the same device")
    transform = residual_transform or PairedResidualTransformV3()
    (
        batch,
        base,
        targets,
        oom,
        oom_stage,
        peak_live,
        weights,
        regression_mask,
    ) = _paired_batch_tensors(samples, device)
    residual_targets = transform.encode(base, targets)
    optimizer.zero_grad(set_to_none=True)
    teacher.eval()
    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        teacher_output = teacher(*graph_batch_tensors(batch))
    with _autocast_context(device, autocast_dtype):
        output = student(*graph_batch_tensors(batch))
        hard = _supervised_loss(
            output,
            targets,
            oom,
            oom_stage,
            peak_live,
            weights,
            regression_mask,
        )
        if bool(regression_mask.any()):
            hard = hard + F.smooth_l1_loss(
                output.paired_residual[regression_mask],
                residual_targets[regression_mask],
            )
            metric_soft = (
                F.smooth_l1_loss(
                    output.prediction[regression_mask],
                    teacher_output.prediction[regression_mask],
                )
                + F.smooth_l1_loss(
                    output.paired_residual[regression_mask],
                    teacher_output.paired_residual[regression_mask],
                )
                + 0.05
                * F.smooth_l1_loss(
                    output.log_variance[regression_mask],
                    teacher_output.log_variance[regression_mask],
                )
            )
        else:
            metric_soft = output.prediction.new_zeros(())
        soft = (
            metric_soft
            + 0.05 * F.smooth_l1_loss(output.oom_logit, teacher_output.oom_logit)
            + 0.05 * F.smooth_l1_loss(
                output.oom_stage_logits, teacher_output.oom_stage_logits
            )
        )
        graph_relation = F.smooth_l1_loss(
            F.normalize(output.graph_embedding, dim=-1)
            @ F.normalize(output.graph_embedding, dim=-1).transpose(0, 1),
            F.normalize(teacher_output.graph_embedding, dim=-1)
            @ F.normalize(teacher_output.graph_embedding, dim=-1).transpose(0, 1),
        )
        student_phase = output.phase_embedding.flatten(0, 1)
        teacher_phase = teacher_output.phase_embedding.flatten(0, 1)
        phase_relation = F.smooth_l1_loss(
            F.normalize(student_phase, dim=-1)
            @ F.normalize(student_phase, dim=-1).transpose(0, 1),
            F.normalize(teacher_phase, dim=-1)
            @ F.normalize(teacher_phase, dim=-1).transpose(0, 1),
        )
        identity_regularization = adapter_identity_regularization_loss(student)
        l2_sp = (
            adapter_regularization_loss(student, reference=reference_parameters)
            if reference_parameters is not None
            else output.prediction.new_zeros(())
        )
        loss = (
            hard_label_weight * hard
            + (1.0 - hard_label_weight) * soft
            + representation_weight * (graph_relation + phase_relation)
            + adapter_regularization_weight * identity_regularization
            + l2_sp_weight * l2_sp
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
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "optional_output_names": [
            "log_variance",
            "oom_probability",
            "oom_failure_stage",
            "confidence",
            "peak_live_bytes_log1p",
            "base_prediction",
            "paired_residual",
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
    "PairedTransferSampleV3",
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
    "target_student_adapter_step",
    "target_teacher_adapter_step",
    "teacher_train_step",
    "trainable_parameter_count",
]
