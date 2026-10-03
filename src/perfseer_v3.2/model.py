"""Twelve physical outputs from shared training/inference graph backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from perfseer_v3.features import GraphBatchV3
from perfseer_v31.model import SeerNetV31Config
from perfseer_v3.model import (
    HierarchicalNodeEncoder, HierarchicalEdgeEncoder, WorkloadGlobalEncoderV3,
    SeerBlockV3, _mlp, _scatter_mean, _scatter_max, _scatter_sum, graph_batch_tensors,
)

from .version import TARGET_NAMES, HEAD_GROUPS, MODES, SM_INDICES, POSITIVE_INDICES


class SeerOutputV32(NamedTuple):
    prediction: torch.Tensor
    graph_embedding: torch.Tensor
    phase_embedding: torch.Tensor
    phase_presence: torch.Tensor


@dataclass(frozen=True)
class SeerNetV32Config(SeerNetV31Config):
    num_outputs: int = 12
    pooling_mode: str = "phase_aware"

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden and num_blocks must be positive")
        if self.num_outputs != len(TARGET_NAMES):
            raise ValueError("PerfSeer v3.2 requires exactly twelve outputs")
        if self.node_identity_fusion not in {"additive", "concatenation"}:
            raise ValueError("unsupported node identity fusion")
        if self.pooling_mode not in {"existing", "phase_aware"}:
            raise ValueError("unsupported pooling mode")


class SeerNetV32(nn.Module):
    def __init__(self, config: SeerNetV32Config, target_scales: torch.Tensor, target_medians=None) -> None:
        super().__init__()
        self.config = config
        scales = torch.as_tensor(target_scales, dtype=torch.float32).detach().clone()
        if scales.shape != (len(TARGET_NAMES),) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("target scales must contain twelve finite positive values")
        self.register_buffer("target_scales", scales)
        self.use_phase_aware_pooling = config.pooling_mode == "phase_aware"
        self.num_phases = config.num_phases
        self.hidden = config.hidden
        self.node_encoder = HierarchicalNodeEncoder(config)
        self.edge_encoder = HierarchicalEdgeEncoder(config)
        self.global_encoder = WorkloadGlobalEncoderV3(config)
        self.blocks = nn.ModuleList(SeerBlockV3(config) for _ in range(config.num_blocks))
        self.phase_encoder = nn.ModuleList(_mlp(2 * config.hidden, config.hidden, config.hidden, config.dropout) for _ in MODES)
        self.phase_fusion = nn.ModuleList(_mlp((config.num_phases + 1) * config.hidden, config.hidden, config.hidden, config.dropout) for _ in MODES)
        self.phase_fusion_norm = nn.ModuleList(nn.LayerNorm(config.hidden) for _ in MODES)
        self.prediction_heads = nn.ModuleList(_mlp(config.hidden, len(indices), config.hidden, config.dropout) for indices in HEAD_GROUPS)
        medians = scales.clone() if target_medians is None else torch.as_tensor(target_medians, dtype=torch.float32)
        if medians.shape != scales.shape or not torch.isfinite(medians).all() or (medians[list(POSITIVE_INDICES)] <= 0).any() or ((medians[list(SM_INDICES)] < 0) | (medians[list(SM_INDICES)] > 100)).any():
            raise ValueError("invalid training target medians")
        with torch.no_grad():
            for head, indices in zip(self.prediction_heads, HEAD_GROUPS, strict=True):
                # Start each physical output at its training median.
                head[-1].weight.zero_()
                for local, index in enumerate(indices):
                    if index in SM_INDICES:
                        value = torch.logit((medians[index] / 100).clamp(1e-6, 1 - 1e-6))
                    else:
                        value = torch.log(torch.expm1(medians[index] / scales[index]))
                    head[-1].bias[local].copy_(value)

    def _encode(
        self,
        mode: int,
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
    ):
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
        phase_hidden = self.phase_encoder[mode](torch.cat([phase_mean, phase_max], dim=-1))
        phase_hidden = phase_hidden * phase_presence
        phase_hidden = phase_hidden.view(
            graph_count,
            self.num_phases,
            self.hidden,
        )
        if self.use_phase_aware_pooling:
            fused = self.phase_fusion[mode](
                torch.cat([globals_, phase_hidden.flatten(1)], dim=-1)
            )
            globals_ = self.phase_fusion_norm[mode](globals_ + fused)

        return globals_, phase_hidden, phase_presence.view(graph_count, self.num_phases).bool()

    def forward(self, training: GraphBatchV3, inference: GraphBatchV3) -> SeerOutputV32:
        if training.u_cont.size(0) != inference.u_cont.size(0):
            raise ValueError("paired graph batch sizes differ")
        encoded = [self._encode(index, *graph_batch_tensors(batch)) for index, batch in enumerate((training, inference))]
        with torch.autocast(training.x_cont.device.type, enabled=False):
            raw = torch.cat([head(encoded[index // 3][0].float()) for index, head in enumerate(self.prediction_heads)], dim=-1)
            values = []
            for index in range(len(TARGET_NAMES)):
                values.append(100 * torch.sigmoid(raw[:, index]) if index in SM_INDICES else
                              F.softplus(raw[:, index]).clamp_min(1e-6) * self.target_scales[index])
            prediction = torch.stack(values, dim=-1)
        return SeerOutputV32(prediction, torch.stack([value[0] for value in encoded], dim=1),
                             torch.stack([value[1] for value in encoded], dim=1),
                             torch.stack([value[2] for value in encoded], dim=1))

    def predict_batch(self, batch) -> SeerOutputV32:
        return self(batch.training, batch.inference)
