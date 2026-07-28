"""Hierarchical, phase-aware SeerNet encoder for PerfSeer v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureLayoutV3, GraphBatchV3
from .graph_ir_v3 import PHASES, TENSOR_ROLES
from .op_registry import OperationRegistry
from .schema import (
    CAPTURE_BACKENDS,
    CAPTURE_MODES,
    DTYPES,
    DYNAMIC_SHAPE_POLICIES,
    EDGE_ALIAS_CLASSES,
    EDGE_DYNAMIC_QUALITIES,
    FEATURE_QUALITIES,
    HARDWARE_HASH_BUCKETS,
    LAYOUTS,
    OPERATOR_BACKENDS,
    OPTIMIZERS,
    OPTIMIZER_FAMILIES,
    OPTIMIZER_HASH_BUCKETS,
    PHASE_TRANSITIONS,
    SCHEDULERS,
    SCHEDULER_FAMILIES,
    SCHEDULER_HASH_BUCKETS,
    SLOT_BUCKETS,
)

OOM_FAILURE_STAGES = (
    "none",
    "capture",
    "forward",
    "loss",
    "backward",
    "optimizer",
    "allocator",
)


class SeerOutputV3(NamedTuple):
    # The first four fields preserve the original v3 deployment tuple order.
    prediction: torch.Tensor
    log_variance: torch.Tensor
    oom_logit: torch.Tensor
    confidence: torch.Tensor
    oom_stage_logits: torch.Tensor
    peak_live_bytes_log1p: torch.Tensor
    graph_embedding: torch.Tensor
    phase_embedding: torch.Tensor


@dataclass(frozen=True)
class SeerNetV3Config:
    node_continuous_dim: int
    edge_continuous_dim: int
    global_continuous_dim: int
    node_flag_dim: int
    edge_flag_dim: int
    quality_dim: int
    num_exact_ops: int
    num_families: int
    num_hash_buckets: int
    num_phases: int
    num_dtypes: int
    num_layouts: int
    num_ranks: int
    num_operator_backends: int
    num_feature_qualities: int
    num_edge_roles: int
    num_edge_alias_classes: int
    num_edge_dynamic_qualities: int
    num_phase_transitions: int
    num_capture_modes: int
    num_capture_backends: int
    num_optimizers: int
    num_optimizer_families: int
    num_optimizer_hash_buckets: int
    num_schedulers: int
    num_scheduler_families: int
    num_scheduler_hash_buckets: int
    num_dynamic_shape_policies: int
    num_hardware_buckets: int
    num_slot_buckets: int
    hidden: int = 192
    num_blocks: int = 2
    num_outputs: int = 6
    exact_embedding_dim: int = 32
    family_embedding_dim: int = 16
    hash_embedding_dim: int = 12
    overload_hash_embedding_dim: int = 8
    phase_embedding_dim: int = 4
    input_dtype_embedding_dim: int = 8
    dtype_embedding_dim: int = 8
    accumulation_dtype_embedding_dim: int = 8
    backend_embedding_dim: int = 8
    feature_quality_embedding_dim: int = 4
    layout_embedding_dim: int = 4
    rank_embedding_dim: int = 4
    edge_role_embedding_dim: int = 8
    edge_slot_embedding_dim: int = 4
    edge_dtype_embedding_dim: int = 4
    edge_layout_embedding_dim: int = 4
    edge_rank_embedding_dim: int = 4
    edge_alias_embedding_dim: int = 4
    edge_dynamic_quality_embedding_dim: int = 4
    phase_transition_embedding_dim: int = 4
    hardware_embedding_dim: int = 8
    global_precision_embedding_dim: int = 4
    optimizer_embedding_dim: int = 4
    optimizer_family_embedding_dim: int = 4
    optimizer_hash_embedding_dim: int = 4
    scheduler_embedding_dim: int = 4
    scheduler_family_embedding_dim: int = 4
    scheduler_hash_embedding_dim: int = 4
    capture_mode_embedding_dim: int = 4
    capture_backend_embedding_dim: int = 4
    dynamic_shape_embedding_dim: int = 4
    training_mode_embedding_dim: int = 2
    dropout: float = 0.05
    node_identity_fusion: str = "additive"
    pooling_mode: str = "existing"
    predict_uncertainty: bool = True
    predict_oom: bool = True
    predict_oom_stage: bool = True
    predict_peak_live_bytes: bool = True
    num_oom_stages: int = len(OOM_FAILURE_STAGES)

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden and num_blocks must be positive")
        if self.num_outputs != 6:
            raise ValueError("PerfSeer v3 must preserve the six scheduler outputs")
        if self.node_identity_fusion not in {"additive", "concatenation"}:
            raise ValueError("node_identity_fusion must be additive or concatenation")
        if self.pooling_mode not in {"existing", "phase_aware"}:
            raise ValueError("pooling_mode must be existing or phase_aware")
        if self.num_oom_stages < 2:
            raise ValueError("num_oom_stages must include no-OOM and at least one failure stage")

    @classmethod
    def from_registry(
        cls,
        registry: OperationRegistry,
        layout: FeatureLayoutV3,
        **overrides: Any,
    ) -> "SeerNetV3Config":
        maximum_exact = max((rule.exact_id for rule in registry.rules), default=0)
        values = {
            "node_continuous_dim": len(layout.node_continuous_fields),
            "edge_continuous_dim": len(layout.edge_continuous_fields),
            "global_continuous_dim": len(layout.global_continuous_fields),
            "node_flag_dim": len(layout.node_flag_fields),
            "edge_flag_dim": len(layout.edge_flag_fields),
            "quality_dim": len(layout.quality_fields),
            "num_exact_ops": maximum_exact + 1,
            "num_families": len(registry.families),
            "num_hash_buckets": registry.hash_buckets,
            "num_phases": len(PHASES),
            "num_dtypes": len(DTYPES),
            "num_layouts": len(LAYOUTS),
            "num_ranks": 17,
            "num_operator_backends": len(OPERATOR_BACKENDS),
            "num_feature_qualities": len(FEATURE_QUALITIES),
            "num_edge_roles": len(TENSOR_ROLES),
            "num_edge_alias_classes": len(EDGE_ALIAS_CLASSES),
            "num_edge_dynamic_qualities": len(EDGE_DYNAMIC_QUALITIES),
            "num_phase_transitions": len(PHASE_TRANSITIONS),
            "num_capture_modes": len(CAPTURE_MODES),
            "num_capture_backends": len(CAPTURE_BACKENDS),
            "num_optimizers": len(OPTIMIZERS),
            "num_optimizer_families": len(OPTIMIZER_FAMILIES),
            "num_optimizer_hash_buckets": OPTIMIZER_HASH_BUCKETS,
            "num_schedulers": len(SCHEDULERS),
            "num_scheduler_families": len(SCHEDULER_FAMILIES),
            "num_scheduler_hash_buckets": SCHEDULER_HASH_BUCKETS,
            "num_dynamic_shape_policies": len(DYNAMIC_SHAPE_POLICIES),
            "num_hardware_buckets": HARDWARE_HASH_BUCKETS,
            "num_slot_buckets": SLOT_BUCKETS,
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scatter_sum(source: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = source.new_zeros((size, source.size(-1)))
    if source.size(0) == 0:
        return result
    expanded = index.view(-1, 1).expand(-1, source.size(-1))
    return result.scatter_add(0, expanded, source)


def _scatter_mean(source: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = _scatter_sum(source, index, size)
    count = source.new_zeros((size, 1))
    if source.size(0) > 0:
        count.scatter_add_(0, index.view(-1, 1), source.new_ones((source.size(0), 1)))
    return total / count.clamp_min(1.0)


def _scatter_max(source: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if source.size(0) == 0:
        return source.new_zeros((size, source.size(-1)))
    result = torch.full(
        (size, source.size(-1)),
        -float("inf"),
        dtype=source.dtype,
        device=source.device,
    )
    expanded = index.view(-1, 1).expand(-1, source.size(-1))
    result = torch.scatter_reduce(
        result,
        0,
        expanded,
        source,
        reduce="amax",
        include_self=True,
    )
    return torch.where(torch.isinf(result), torch.zeros_like(result), result)


def _mlp(input_dim: int, output_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )


class HierarchicalNodeEncoder(nn.Module):
    def __init__(self, config: SeerNetV3Config) -> None:
        super().__init__()
        self.additive_identity = config.node_identity_fusion == "additive"
        self.exact = nn.Embedding(config.num_exact_ops, config.exact_embedding_dim)
        self.family = nn.Embedding(config.num_families, config.family_embedding_dim)
        self.hash = nn.Embedding(config.num_hash_buckets, config.hash_embedding_dim)
        self.overload_hash = nn.Embedding(
            config.num_hash_buckets,
            config.overload_hash_embedding_dim,
        )
        self.phase = nn.Embedding(config.num_phases, config.phase_embedding_dim)
        self.input_dtype = nn.Embedding(
            config.num_dtypes,
            config.input_dtype_embedding_dim,
        )
        self.dtype = nn.Embedding(config.num_dtypes, config.dtype_embedding_dim)
        self.accumulation_dtype = nn.Embedding(
            config.num_dtypes,
            config.accumulation_dtype_embedding_dim,
        )
        self.backend = nn.Embedding(
            config.num_operator_backends,
            config.backend_embedding_dim,
        )
        self.feature_quality = nn.Embedding(
            config.num_feature_qualities,
            config.feature_quality_embedding_dim,
        )
        self.layout = nn.Embedding(config.num_layouts, config.layout_embedding_dim)
        self.rank = nn.Embedding(config.num_ranks, config.rank_embedding_dim)
        auxiliary_dim = (
            config.family_embedding_dim
            + config.hash_embedding_dim
            + config.overload_hash_embedding_dim
            + config.phase_embedding_dim
            + config.backend_embedding_dim
            + config.feature_quality_embedding_dim
        )
        identity_projection_input = (
            auxiliary_dim
            if self.additive_identity
            else config.exact_embedding_dim + auxiliary_dim
        )
        self.identity_projection = nn.Linear(
            identity_projection_input,
            config.exact_embedding_dim,
        )
        input_dim = (
            config.exact_embedding_dim
            + config.input_dtype_embedding_dim
            + config.dtype_embedding_dim
            + config.accumulation_dtype_embedding_dim
            + config.layout_embedding_dim
            + config.rank_embedding_dim
            + config.node_flag_dim
            + config.node_continuous_dim
        )
        self.output = _mlp(input_dim, config.hidden, config.hidden, config.dropout)

    def forward(
        self,
        x_cont: torch.Tensor,
        op_exact_id: torch.Tensor,
        op_family_id: torch.Tensor,
        op_hash_id: torch.Tensor,
        op_overload_hash_id: torch.Tensor,
        phase_id: torch.Tensor,
        input_dtype_id: torch.Tensor,
        dtype_id: torch.Tensor,
        accumulation_dtype_id: torch.Tensor,
        backend_id: torch.Tensor,
        feature_quality_id: torch.Tensor,
        layout_id: torch.Tensor,
        rank_id: torch.Tensor,
        node_flags: torch.Tensor,
    ) -> torch.Tensor:
        auxiliary = torch.cat(
            [
                self.family(op_family_id),
                self.hash(op_hash_id),
                self.overload_hash(op_overload_hash_id),
                self.phase(phase_id),
                self.backend(backend_id),
                self.feature_quality(feature_quality_id),
            ],
            dim=-1,
        )
        exact = self.exact(op_exact_id)
        if self.additive_identity:
            identity = exact + self.identity_projection(auxiliary)
        else:
            identity = self.identity_projection(torch.cat([exact, auxiliary], dim=-1))
        return self.output(
            torch.cat(
                [
                    identity,
                    self.input_dtype(input_dtype_id),
                    self.dtype(dtype_id),
                    self.accumulation_dtype(accumulation_dtype_id),
                    self.layout(layout_id),
                    self.rank(rank_id),
                    node_flags,
                    x_cont,
                ],
                dim=-1,
            )
        )


class HierarchicalEdgeEncoder(nn.Module):
    def __init__(self, config: SeerNetV3Config) -> None:
        super().__init__()
        self.role = nn.Embedding(config.num_edge_roles, config.edge_role_embedding_dim)
        self.source_slot = nn.Embedding(
            config.num_slot_buckets,
            config.edge_slot_embedding_dim,
        )
        self.destination_slot = nn.Embedding(
            config.num_slot_buckets,
            config.edge_slot_embedding_dim,
        )
        self.dtype = nn.Embedding(config.num_dtypes, config.edge_dtype_embedding_dim)
        self.layout = nn.Embedding(config.num_layouts, config.edge_layout_embedding_dim)
        self.rank = nn.Embedding(config.num_ranks, config.edge_rank_embedding_dim)
        self.alias = nn.Embedding(
            config.num_edge_alias_classes,
            config.edge_alias_embedding_dim,
        )
        self.dynamic_quality = nn.Embedding(
            config.num_edge_dynamic_qualities,
            config.edge_dynamic_quality_embedding_dim,
        )
        self.phase_transition = nn.Embedding(
            config.num_phase_transitions,
            config.phase_transition_embedding_dim,
        )
        input_dim = (
            config.edge_continuous_dim
            + config.edge_flag_dim
            + config.edge_role_embedding_dim
            + 2 * config.edge_slot_embedding_dim
            + config.edge_dtype_embedding_dim
            + config.edge_layout_embedding_dim
            + config.edge_rank_embedding_dim
            + config.edge_alias_embedding_dim
            + config.edge_dynamic_quality_embedding_dim
            + config.phase_transition_embedding_dim
        )
        self.output = _mlp(input_dim, config.hidden, config.hidden, config.dropout)

    def forward(
        self,
        edge_cont: torch.Tensor,
        edge_flags: torch.Tensor,
        edge_role_id: torch.Tensor,
        edge_source_slot_id: torch.Tensor,
        edge_destination_slot_id: torch.Tensor,
        edge_dtype_id: torch.Tensor,
        edge_layout_id: torch.Tensor,
        edge_rank_id: torch.Tensor,
        edge_alias_id: torch.Tensor,
        edge_dynamic_quality_id: torch.Tensor,
        edge_phase_transition_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            torch.cat(
                [
                    edge_cont,
                    edge_flags,
                    self.role(edge_role_id),
                    self.source_slot(edge_source_slot_id),
                    self.destination_slot(edge_destination_slot_id),
                    self.dtype(edge_dtype_id),
                    self.layout(edge_layout_id),
                    self.rank(edge_rank_id),
                    self.alias(edge_alias_id),
                    self.dynamic_quality(edge_dynamic_quality_id),
                    self.phase_transition(edge_phase_transition_id),
                ],
                dim=-1,
            )
        )


class HierarchicalGlobalEncoder(nn.Module):
    def __init__(self, config: SeerNetV3Config) -> None:
        super().__init__()
        self.hardware = nn.Embedding(
            config.num_hardware_buckets,
            config.hardware_embedding_dim,
        )
        self.precision = nn.Embedding(
            config.num_dtypes,
            config.global_precision_embedding_dim,
        )
        self.optimizer = nn.Embedding(config.num_optimizers, config.optimizer_embedding_dim)
        self.optimizer_family = nn.Embedding(
            config.num_optimizer_families,
            config.optimizer_family_embedding_dim,
        )
        self.optimizer_hash = nn.Embedding(
            config.num_optimizer_hash_buckets,
            config.optimizer_hash_embedding_dim,
        )
        self.scheduler = nn.Embedding(
            config.num_schedulers,
            config.scheduler_embedding_dim,
        )
        self.scheduler_family = nn.Embedding(
            config.num_scheduler_families,
            config.scheduler_family_embedding_dim,
        )
        self.scheduler_hash = nn.Embedding(
            config.num_scheduler_hash_buckets,
            config.scheduler_hash_embedding_dim,
        )
        self.capture_mode = nn.Embedding(
            config.num_capture_modes,
            config.capture_mode_embedding_dim,
        )
        self.capture_backend = nn.Embedding(
            config.num_capture_backends,
            config.capture_backend_embedding_dim,
        )
        self.dynamic_shape = nn.Embedding(
            config.num_dynamic_shape_policies,
            config.dynamic_shape_embedding_dim,
        )
        self.training_mode = nn.Embedding(2, config.training_mode_embedding_dim)
        input_dim = (
            config.global_continuous_dim
            + config.quality_dim
            + config.hardware_embedding_dim
            + config.global_precision_embedding_dim
            + config.optimizer_embedding_dim
            + config.optimizer_family_embedding_dim
            + config.optimizer_hash_embedding_dim
            + config.scheduler_embedding_dim
            + config.scheduler_family_embedding_dim
            + config.scheduler_hash_embedding_dim
            + config.capture_mode_embedding_dim
            + config.capture_backend_embedding_dim
            + config.dynamic_shape_embedding_dim
            + config.training_mode_embedding_dim
        )
        self.output = _mlp(input_dim, config.hidden, config.hidden, config.dropout)

    def forward(
        self,
        u_cont: torch.Tensor,
        quality: torch.Tensor,
        hardware_id: torch.Tensor,
        precision_id: torch.Tensor,
        optimizer_id: torch.Tensor,
        optimizer_family_id: torch.Tensor,
        optimizer_hash_id: torch.Tensor,
        scheduler_id: torch.Tensor,
        scheduler_family_id: torch.Tensor,
        scheduler_hash_id: torch.Tensor,
        capture_mode_id: torch.Tensor,
        capture_backend_id: torch.Tensor,
        dynamic_shape_id: torch.Tensor,
        training_mode_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            torch.cat(
                [
                    u_cont,
                    quality,
                    self.hardware(hardware_id),
                    self.precision(precision_id),
                    self.optimizer(optimizer_id),
                    self.optimizer_family(optimizer_family_id),
                    self.optimizer_hash(optimizer_hash_id),
                    self.scheduler(scheduler_id),
                    self.scheduler_family(scheduler_family_id),
                    self.scheduler_hash(scheduler_hash_id),
                    self.capture_mode(capture_mode_id),
                    self.capture_backend(capture_backend_id),
                    self.dynamic_shape(dynamic_shape_id),
                    self.training_mode(training_mode_id),
                ],
                dim=-1,
            )
        )


class SeerBlockV3(nn.Module):
    """The v2-style gated residual message-passing control trunk."""

    def __init__(self, config: SeerNetV3Config) -> None:
        super().__init__()
        hidden = config.hidden
        self.edge_update = _mlp(4 * hidden, hidden, hidden, config.dropout)
        self.node_update = _mlp(3 * hidden, hidden, hidden, config.dropout)
        self.global_update = _mlp(4 * hidden, hidden, hidden, config.dropout)
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.global_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        globals_: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_count = globals_.size(0)
        if edge_index.size(1) > 0:
            source, destination = edge_index[0], edge_index[1]
            edge_graph = batch[source]
            edge_delta = self.edge_update(
                torch.cat(
                    [
                        edges,
                        nodes[source],
                        nodes[destination],
                        globals_[edge_graph],
                    ],
                    dim=-1,
                )
            )
            edges = self.edge_norm(edges + edge_delta)
            aggregated_edges = _scatter_mean(edges, destination, nodes.size(0))
        else:
            aggregated_edges = nodes.new_zeros(nodes.shape)
        node_globals = globals_[batch] if nodes.size(0) else nodes.new_zeros(nodes.shape)
        node_delta = self.node_update(
            torch.cat([nodes, aggregated_edges, node_globals], dim=-1)
        )
        nodes = self.node_norm(nodes + node_delta)
        node_mean = _scatter_mean(nodes, batch, graph_count)
        node_max = _scatter_max(nodes, batch, graph_count)
        node_sum = _scatter_sum(nodes, batch, graph_count)
        global_delta = self.global_update(
            torch.cat([globals_, node_mean, node_max, node_sum], dim=-1)
        )
        globals_ = self.global_norm(globals_ + global_delta)
        return nodes, edges, globals_


class SeerNetV3(nn.Module):
    def __init__(self, config: SeerNetV3Config) -> None:
        super().__init__()
        self.config = config
        self.use_phase_aware_pooling = config.pooling_mode == "phase_aware"
        self.num_phases = config.num_phases
        self.hidden = config.hidden
        self.num_oom_stages = config.num_oom_stages
        self.node_encoder = HierarchicalNodeEncoder(config)
        self.edge_encoder = HierarchicalEdgeEncoder(config)
        self.global_encoder = HierarchicalGlobalEncoder(config)
        self.blocks = nn.ModuleList(SeerBlockV3(config) for _ in range(config.num_blocks))
        self.phase_encoder = _mlp(
            2 * config.hidden,
            config.hidden,
            config.hidden,
            config.dropout,
        )
        self.phase_fusion = _mlp(
            (config.num_phases + 1) * config.hidden,
            config.hidden,
            config.hidden,
            config.dropout,
        )
        self.phase_fusion_norm = nn.LayerNorm(config.hidden)
        self.prediction_head = _mlp(config.hidden, config.num_outputs, config.hidden, config.dropout)
        self.uncertainty_head = (
            _mlp(config.hidden, config.num_outputs, config.hidden, config.dropout)
            if config.predict_uncertainty
            else None
        )
        self.oom_head = (
            _mlp(config.hidden, 1, config.hidden, config.dropout)
            if config.predict_oom
            else None
        )
        self.oom_stage_head = (
            _mlp(config.hidden, config.num_oom_stages, config.hidden, config.dropout)
            if config.predict_oom_stage
            else None
        )
        self.peak_live_head = (
            _mlp(config.hidden, 1, config.hidden, config.dropout)
            if config.predict_peak_live_bytes
            else None
        )
        self.confidence_head = _mlp(config.hidden, 1, config.hidden, config.dropout)

    def forward(
        self,
        x_cont: torch.Tensor,
        op_exact_id: torch.Tensor,
        op_family_id: torch.Tensor,
        op_hash_id: torch.Tensor,
        op_overload_hash_id: torch.Tensor,
        phase_id: torch.Tensor,
        input_dtype_id: torch.Tensor,
        dtype_id: torch.Tensor,
        accumulation_dtype_id: torch.Tensor,
        backend_id: torch.Tensor,
        feature_quality_id: torch.Tensor,
        layout_id: torch.Tensor,
        rank_id: torch.Tensor,
        node_flags: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cont: torch.Tensor,
        edge_role_id: torch.Tensor,
        edge_source_slot_id: torch.Tensor,
        edge_destination_slot_id: torch.Tensor,
        edge_dtype_id: torch.Tensor,
        edge_layout_id: torch.Tensor,
        edge_rank_id: torch.Tensor,
        edge_alias_id: torch.Tensor,
        edge_dynamic_quality_id: torch.Tensor,
        edge_phase_transition_id: torch.Tensor,
        edge_flags: torch.Tensor,
        u_cont: torch.Tensor,
        hardware_id: torch.Tensor,
        precision_id: torch.Tensor,
        optimizer_id: torch.Tensor,
        optimizer_family_id: torch.Tensor,
        optimizer_hash_id: torch.Tensor,
        scheduler_id: torch.Tensor,
        scheduler_family_id: torch.Tensor,
        scheduler_hash_id: torch.Tensor,
        capture_mode_id: torch.Tensor,
        capture_backend_id: torch.Tensor,
        dynamic_shape_id: torch.Tensor,
        training_mode_id: torch.Tensor,
        quality: torch.Tensor,
        batch: torch.Tensor,
    ) -> SeerOutputV3:
        nodes = self.node_encoder(
            x_cont,
            op_exact_id,
            op_family_id,
            op_hash_id,
            op_overload_hash_id,
            phase_id,
            input_dtype_id,
            dtype_id,
            accumulation_dtype_id,
            backend_id,
            feature_quality_id,
            layout_id,
            rank_id,
            node_flags,
        )
        edges = self.edge_encoder(
            edge_cont,
            edge_flags,
            edge_role_id,
            edge_source_slot_id,
            edge_destination_slot_id,
            edge_dtype_id,
            edge_layout_id,
            edge_rank_id,
            edge_alias_id,
            edge_dynamic_quality_id,
            edge_phase_transition_id,
        )
        globals_ = self.global_encoder(
            u_cont,
            quality,
            hardware_id,
            precision_id,
            optimizer_id,
            optimizer_family_id,
            optimizer_hash_id,
            scheduler_id,
            scheduler_family_id,
            scheduler_hash_id,
            capture_mode_id,
            capture_backend_id,
            dynamic_shape_id,
            training_mode_id,
        )
        for block in self.blocks:
            nodes, edges, globals_ = block(nodes, edges, globals_, edge_index, batch)

        graph_count = globals_.size(0)
        phase_index = batch * self.num_phases + phase_id
        phase_count = graph_count * self.num_phases
        phase_mean = _scatter_mean(nodes, phase_index, phase_count)
        phase_max = _scatter_max(nodes, phase_index, phase_count)
        phase_presence = _scatter_sum(
            nodes.new_ones((nodes.size(0), 1)),
            phase_index,
            phase_count,
        ).gt(0).to(nodes.dtype)
        phase_hidden = self.phase_encoder(torch.cat([phase_mean, phase_max], dim=-1))
        phase_hidden = phase_hidden * phase_presence
        phase_hidden = phase_hidden.view(
            graph_count,
            self.num_phases,
            self.hidden,
        )
        if self.use_phase_aware_pooling:
            fused = self.phase_fusion(
                torch.cat([globals_, phase_hidden.flatten(1)], dim=-1)
            )
            globals_ = self.phase_fusion_norm(globals_ + fused)

        prediction = self.prediction_head(globals_)
        if self.uncertainty_head is None:
            log_variance = torch.zeros_like(prediction)
        else:
            log_variance = self.uncertainty_head(globals_).clamp(-10.0, 10.0)
        if self.oom_head is None:
            oom_logit = globals_.new_zeros((globals_.size(0), 1))
        else:
            oom_logit = self.oom_head(globals_)
        if self.oom_stage_head is None:
            oom_stage_logits = globals_.new_zeros(
                (globals_.size(0), self.num_oom_stages)
            )
        else:
            oom_stage_logits = self.oom_stage_head(globals_)
        if self.peak_live_head is None:
            peak_live_bytes_log1p = globals_.new_zeros((globals_.size(0), 1))
        else:
            peak_live_bytes_log1p = F.softplus(self.peak_live_head(globals_))
        learned_confidence = torch.sigmoid(self.confidence_head(globals_))
        unknown_fraction = quality[:, 0:1].clamp(0.0, 1.0)
        custom_fraction = quality[:, 3:4].clamp(0.0, 1.0)
        capture_factor = 0.5 + 0.5 * quality[:, 4:5].clamp(0.0, 1.0)
        confidence = (
            learned_confidence
            * (1.0 - unknown_fraction)
            * (1.0 - 0.5 * custom_fraction)
            * capture_factor
        )
        return SeerOutputV3(
            prediction,
            log_variance,
            oom_logit,
            confidence,
            oom_stage_logits,
            peak_live_bytes_log1p,
            globals_,
            phase_hidden,
        )

    @torch.jit.ignore
    def forward_batch(self, batch: GraphBatchV3) -> SeerOutputV3:
        return self(*graph_batch_tensors(batch))


def graph_batch_tensors(batch: GraphBatchV3) -> tuple[torch.Tensor, ...]:
    return (
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
        batch.edge_index,
        batch.edge_cont,
        batch.edge_role_id,
        batch.edge_source_slot_id,
        batch.edge_destination_slot_id,
        batch.edge_dtype_id,
        batch.edge_layout_id,
        batch.edge_rank_id,
        batch.edge_alias_id,
        batch.edge_dynamic_quality_id,
        batch.edge_phase_transition_id,
        batch.edge_flags,
        batch.u_cont,
        batch.hardware_id,
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
        batch.quality,
        batch.batch,
    )


__all__ = [
    "HierarchicalEdgeEncoder",
    "HierarchicalGlobalEncoder",
    "HierarchicalNodeEncoder",
    "OOM_FAILURE_STAGES",
    "SeerBlockV3",
    "SeerNetV3",
    "SeerNetV3Config",
    "SeerOutputV3",
    "graph_batch_tensors",
]
