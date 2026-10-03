"""Seven physical training outputs from one shared graph backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from perfseer_v3.features import GraphBatchV3
from perfseer_v3.model import (
    HierarchicalNodeEncoder, HierarchicalEdgeEncoder, WorkloadGlobalEncoderV3,
    SeerBlockV3, _mlp, graph_batch_tensors,
)
from perfseer_v31.model import SeerNetV31Config
from perfseer_v32.model import SeerNetV32

from .version import TARGET_NAMES, HEAD_GROUPS, MODES, SM_INDICES, POSITIVE_INDICES


class SeerOutputV4(NamedTuple):
    prediction: torch.Tensor
    graph_embedding: torch.Tensor
    phase_embedding: torch.Tensor
    phase_presence: torch.Tensor


@dataclass(frozen=True)
class SeerNetV4Config(SeerNetV31Config):
    num_outputs: int = 7
    pooling_mode: str = "phase_aware"

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden and num_blocks must be positive")
        if self.num_outputs != len(TARGET_NAMES):
            raise ValueError("PerfSeer v4 requires exactly seven training outputs")
        if self.node_identity_fusion not in {"additive", "concatenation"}:
            raise ValueError("unsupported node identity fusion")
        if self.pooling_mode not in {"existing", "phase_aware"}:
            raise ValueError("unsupported pooling mode")


class SeerNetV4(nn.Module):
    model_variant = "v4.0"
    _encode = SeerNetV32._encode

    def __init__(self, config: SeerNetV4Config, target_scales: torch.Tensor, target_medians=None) -> None:
        super().__init__()
        self.config = config
        scales = torch.as_tensor(target_scales, dtype=torch.float32).detach().clone()
        if scales.shape != (len(TARGET_NAMES),) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("target scales must contain seven finite positive values")
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
                head[-1].weight.zero_()
                for local, index in enumerate(indices):
                    if index in SM_INDICES:
                        value = torch.logit((medians[index] / 100).clamp(1e-6, 1 - 1e-6))
                    else:
                        value = torch.log(torch.expm1(medians[index] / scales[index]))
                    head[-1].bias[local].copy_(value)

    def forward(self, training: GraphBatchV3) -> SeerOutputV4:
        graph, phases, presence = self._encode(0, *graph_batch_tensors(training))
        with torch.autocast(training.x_cont.device.type, enabled=False):
            raw = torch.cat([head(graph.float()) for head in self.prediction_heads], dim=-1)
            values = []
            for index in range(len(TARGET_NAMES)):
                values.append(100 * torch.sigmoid(raw[:, index]) if index in SM_INDICES else
                              F.softplus(raw[:, index]).clamp_min(1e-6) * self.target_scales[index])
            prediction = torch.stack(values, dim=-1)
        return SeerOutputV4(prediction, graph[:, None], phases[:, None], presence[:, None])

    def predict_batch(self, batch) -> SeerOutputV4:
        return self(batch if isinstance(batch, GraphBatchV3) else batch.training)
