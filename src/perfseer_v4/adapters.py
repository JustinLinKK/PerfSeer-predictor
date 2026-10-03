"""Identity-initialized hardware FiLM and low-rank training-head adaptation."""

import torch
from torch import nn
from torch.nn import functional as F

from perfseer_v3.features import GraphBatchV3
from perfseer_v3.model import graph_batch_tensors

from .model import SeerNetV4, SeerOutputV4
from .version import HEAD_GROUPS, TARGET_NAMES, SM_INDICES


class TrainingHeadAdapter(nn.Module):
    def __init__(self, hidden, hardware_dim, rank=8, hardware_hidden=32):
        super().__init__()
        if min(hidden, hardware_dim, rank, hardware_hidden) < 1:
            raise ValueError("adapter dimensions must be positive")
        self.rank = rank
        self.film = nn.Sequential(nn.Linear(hardware_dim, hardware_hidden), nn.GELU(),
                                  nn.Linear(hardware_hidden, 2 * hidden))
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, graph, hardware):
        gamma, beta = self.film(hardware).chunk(2, dim=-1)
        return (1 + gamma) * graph + beta + self.up(self.down(graph)) / self.rank


class HardwareAdaptedV4(nn.Module):
    model_variant = "v4.2"

    def __init__(self, base, rank=8, hardware_hidden=32, *, require_trained_source=True):
        super().__init__()
        if not isinstance(base, SeerNetV4):
            raise ValueError("hardware adaptation requires a v4.0 graph source")
        if require_trained_source and any(not torch.count_nonzero(head[-1].weight).item() for head in base.prediction_heads):
            raise ValueError("hardware adaptation requires trained source prediction heads")
        self.base = base.eval().requires_grad_(False)
        self.config = base.config
        self.rank, self.hardware_hidden = rank, hardware_hidden
        self.regularization_weight = 1e-3
        hardware_dim = self.config.hardware_continuous_dim + self.config.hardware_missing_mask_dim
        self.adapters = nn.ModuleList(TrainingHeadAdapter(self.config.hidden, hardware_dim, rank, hardware_hidden)
                                      for _ in HEAD_GROUPS)
        self.adapters.to(device=next(base.parameters()).device, dtype=next(base.parameters()).dtype)
        parameters = dict(self.named_parameters())
        self._adaptation_names = tuple(name for name, parameter in parameters.items() if parameter.requires_grad)
        initial = torch.cat([parameters[name].detach().flatten() for name in self._adaptation_names])
        self.register_buffer("initial_parameters", initial.clone())

    @property
    def target_scales(self):
        return self.base.target_scales

    @property
    def adapter_config(self):
        return {"rank": self.rank, "hardware_hidden": self.hardware_hidden}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def displacement_penalty(self):
        parameters = dict(self.named_parameters())
        current = torch.cat([parameters[name].flatten() for name in self._adaptation_names])
        return self.regularization_weight * (current - self.initial_parameters).square().mean()

    def forward(self, training: GraphBatchV3):
        self.base.eval()
        with torch.no_grad():
            graph, phases, presence = self.base._encode(0, *graph_batch_tensors(training))
        with torch.autocast(training.x_cont.device.type, enabled=False):
            hardware = torch.cat((training.hardware_cont, training.hardware_missing_mask), dim=-1).float()
            if not torch.isfinite(hardware).all():
                raise ValueError("nonfinite hardware features")
            raw = torch.cat([head(adapter(graph.float(), hardware)) for head, adapter in
                             zip(self.base.prediction_heads, self.adapters, strict=True)], dim=-1)
            prediction = torch.stack([100 * torch.sigmoid(raw[:, index]) if index in SM_INDICES else
                                      F.softplus(raw[:, index]).clamp_min(1e-6) * self.target_scales[index]
                                      for index in range(len(TARGET_NAMES))], dim=-1)
        return SeerOutputV4(prediction, graph[:, None], phases[:, None], presence[:, None])

    def predict_batch(self, batch):
        return self(batch if isinstance(batch, GraphBatchV3) else batch.training)
