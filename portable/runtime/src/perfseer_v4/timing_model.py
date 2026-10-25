"""ABA-1 input contract: hardware-conditioned step timing and explicit epoch length.

Legacy workload normalization and checkpoints are intentionally left unchanged.
Hardware uses the existing fixed-reference values, masks, and categorical ID.
Raw epoch length bypasses the source-fitted (constant-one) workload normalizer.
"""
import torch
from torch import nn

VERSION = "perfseer_aba1_hardware_epoch_v1"
EPOCH_PROTOCOLS = {"measured_single_step", "derived_step_times_count"}


def epoch_time(step_wall_ms, steps_per_epoch, protocol):
    if protocol not in EPOCH_PROTOCOLS:
        raise ValueError("unsupported epoch protocol; measured multi-step needs its own model")
    steps = torch.as_tensor(steps_per_epoch, device=step_wall_ms.device)
    if (steps.shape != step_wall_ms.shape or not torch.isfinite(steps).all()
            or (steps < 1).any() or (steps != steps.floor()).any()):
        raise ValueError("raw steps_per_epoch must be aligned positive integers")
    if protocol == "measured_single_step" and (steps != 1).any():
        raise ValueError("measured_single_step requires exactly one step")
    return step_wall_ms * steps.to(step_wall_ms.dtype)


class TimingResidual(nn.Module):
    """Zero-initialized correction; a frozen graph teacher keeps its memory head.

    Hardware values are masked before encoding; missingness remains observable.
    The category retains known identity when specifications were not recorded.
    A100-only training cannot establish cross-hardware generalization.
    """
    def __init__(self, graph_dim, hardware_dim=62, hidden=128, dropout=.05):
        super().__init__()
        self.config = dict(graph_dim=graph_dim, hardware_dim=hardware_dim,
                           hidden=hidden, dropout=dropout)
        self.hardware_id = nn.Embedding(256, 16)
        self.hardware = nn.Sequential(nn.Linear(2 * hardware_dim + 16, 64), nn.GELU())
        self.graph_norm = nn.LayerNorm(graph_dim)
        self.correction = nn.Sequential(nn.Linear(graph_dim + 64, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 2))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, graph, base_prediction, hardware_cont, hardware_missing_mask,
                hardware_id, raw_steps_per_epoch, epoch_protocol):
        n, d = len(graph), self.config["hardware_dim"]
        if (hardware_cont.shape != (n, d) or hardware_missing_mask.shape != (n, d)
                or hardware_id.shape != (n,) or base_prediction.shape != (n, 6)
                or not torch.isfinite(hardware_cont).all()
                or not ((hardware_missing_mask == 0) | (hardware_missing_mask == 1)).all()
                or (hardware_id < 0).any() or (hardware_id >= 256).any()):
            raise ValueError("invalid hardware or six-target input")
        hw = self.hardware(torch.cat((hardware_cont * (1 - hardware_missing_mask),
                                      hardware_missing_mask, self.hardware_id(hardware_id)), -1))
        delta = self.correction(torch.cat((self.graph_norm(graph), hw), -1))
        step = base_prediction[:, :2] * torch.exp(delta.clamp(-2., 2.))
        epoch = epoch_time(step[:, 0], raw_steps_per_epoch, epoch_protocol)
        return torch.cat((step, epoch[:, None], base_prediction[:, 3:]), -1)


class TimingPredictor(nn.Module):
    def __init__(self, teacher, residual=None):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.residual = residual or TimingResidual(teacher.config.hidden)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, batch, raw_steps_per_epoch, epoch_protocol="measured_single_step"):
        with torch.no_grad():
            output = self.teacher(batch)
        return self.residual(output.graph_embedding[:, 0].float(), output.prediction.float(),
                             batch.hardware_cont.float(), batch.hardware_missing_mask.float(),
                             batch.hardware_id, raw_steps_per_epoch, epoch_protocol)
