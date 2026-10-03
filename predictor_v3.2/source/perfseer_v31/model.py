"""Three-output A10 predictor using the established v3 graph backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from perfseer_v3.features import FeatureLayoutV3, GraphBatchV3
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.model import (
    HierarchicalNodeEncoder, HierarchicalEdgeEncoder, WorkloadGlobalEncoderV3,
    SeerBlockV3, _mlp, _scatter_mean, _scatter_max, _scatter_sum, graph_batch_tensors,
    PHASES, DTYPES, LAYOUTS, OPERATOR_BACKENDS, FEATURE_QUALITIES, TENSOR_ROLES,
    EDGE_ALIAS_CLASSES, EDGE_DYNAMIC_QUALITIES, PHASE_TRANSITIONS, CAPTURE_MODES,
    CAPTURE_BACKENDS, OPTIMIZERS, OPTIMIZER_FAMILIES, OPTIMIZER_HASH_BUCKETS,
    SCHEDULERS, SCHEDULER_FAMILIES, SCHEDULER_HASH_BUCKETS, DYNAMIC_SHAPE_POLICIES,
    HARDWARE_HASH_BUCKETS, SLOT_BUCKETS,
)

from .version import TARGET_NAMES


class SeerOutputV31(NamedTuple):
    prediction: torch.Tensor
    graph_embedding: torch.Tensor
    phase_embedding: torch.Tensor


@dataclass(frozen=True)
class SeerNetV31Config:
    node_continuous_dim: int
    edge_continuous_dim: int
    global_continuous_dim: int
    hardware_continuous_dim: int
    hardware_missing_mask_dim: int
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
    num_outputs: int = 3
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
    checkpoint_blocks: bool = False

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden and num_blocks must be positive")
        if self.num_outputs != 3:
            raise ValueError("PerfSeer v3.1 requires exactly three outputs")
        if self.node_identity_fusion not in {"additive", "concatenation"}:
            raise ValueError("unsupported node identity fusion")
        if self.pooling_mode not in {"existing", "phase_aware"}:
            raise ValueError("unsupported pooling mode")

    @classmethod
    def from_registry(
        cls,
        registry: OperationRegistry,
        layout: FeatureLayoutV3,
        **overrides: Any,
    ) -> "SeerNetV31Config":
        maximum_exact = max((rule.exact_id for rule in registry.rules), default=0)
        values = {
            "node_continuous_dim": len(layout.node_continuous_fields),
            "edge_continuous_dim": len(layout.edge_continuous_fields),
            "global_continuous_dim": len(layout.global_continuous_fields),
            "hardware_continuous_dim": len(layout.hardware_continuous_fields),
            "hardware_missing_mask_dim": len(layout.hardware_missing_mask_fields),
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


class SeerNetV31(nn.Module):
    def __init__(self, config: SeerNetV31Config, target_scales: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        scales = torch.as_tensor(target_scales, dtype=torch.float32).detach().clone()
        if scales.shape != (3,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("target scales must contain three finite positive values")
        self.register_buffer("target_scales", scales)
        self.use_phase_aware_pooling = config.pooling_mode == "phase_aware"
        self.num_phases = config.num_phases
        self.hidden = config.hidden
        self.node_encoder = HierarchicalNodeEncoder(config)
        self.edge_encoder = HierarchicalEdgeEncoder(config)
        self.global_encoder = WorkloadGlobalEncoderV3(config)
        self.blocks = nn.ModuleList(SeerBlockV3(config) for _ in range(config.num_blocks))
        self.phase_encoder = _mlp(2 * config.hidden, config.hidden, config.hidden, config.dropout)
        self.phase_fusion = _mlp(
            (config.num_phases + 1) * config.hidden, config.hidden, config.hidden, config.dropout
        )
        self.phase_fusion_norm = nn.LayerNorm(config.hidden)
        self.prediction_head = _mlp(config.hidden, 3, config.hidden, config.dropout)

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
        hardware_cont: torch.Tensor,
        hardware_missing_mask: torch.Tensor,
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
    ) -> SeerOutputV31:
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
            if self.training and self.config.checkpoint_blocks:
                nodes, edges, globals_ = checkpoint(block, nodes, edges, globals_, edge_index, batch, use_reentrant=False)
            else:
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

        raw = self.prediction_head(globals_).float()
        prediction = torch.stack((
            (F.softplus(raw[:, 0]) + 1e-6) * self.target_scales[0],
            100.0 * torch.sigmoid(raw[:, 1]),
            (F.softplus(raw[:, 2]) + 1e-6) * self.target_scales[2],
        ), dim=-1)
        return SeerOutputV31(prediction, globals_, phase_hidden)

    def predict_batch(self, batch: GraphBatchV3) -> SeerOutputV31:
        return self(*graph_batch_tensors(batch))
