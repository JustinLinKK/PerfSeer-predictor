"""Loss functions used by optimized PerfSeer training."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    """Smooth robust loss in standardized log-target space."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(diff + torch.nn.functional.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, device=diff.device)))


def build_loss(name: str = "mse_logstd", huber_delta: float = 1.0) -> nn.Module:
    key = (name or "mse_logstd").lower()
    if key in {"mse", "mse_logstd"}:
        return nn.MSELoss()
    if key in {"huber", "huber_logstd", "smooth_l1"}:
        return nn.HuberLoss(delta=huber_delta)
    if key in {"logcosh", "log_cosh", "logcosh_logstd"}:
        return LogCoshLoss()
    raise ValueError(f"unknown loss {name!r}")


def weighted_metric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return aggregate multi-task loss and one scalar loss per metric."""

    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    losses: list[torch.Tensor] = []
    for idx in range(pred.size(1)):
        losses.append(criterion(pred[:, idx : idx + 1], target[:, idx : idx + 1]))
    if weights is None:
        total = torch.stack(losses).sum()
    else:
        total = torch.stack([losses[i] * weights[i] for i in range(len(losses))]).sum()
    return total, losses
