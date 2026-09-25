"""Small PCGrad implementation for multi-task PerfSeer training."""

from __future__ import annotations

import random

import torch


def _flatten_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _assign_flat_grad(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    pos = 0
    for p in params:
        n = p.numel()
        p.grad = flat[pos : pos + n].view_as(p).clone()
        pos += n


def pcgrad_backward(
    losses: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    weights: torch.Tensor | None = None,
) -> None:
    """Backpropagate projected average gradients for a list of task losses."""

    if not losses:
        return
    grads: list[torch.Tensor] = []
    for idx, loss in enumerate(losses):
        for p in params:
            p.grad = None
        scaled = loss if weights is None else loss * weights[idx]
        scaled.backward(retain_graph=idx < len(losses) - 1)
        grads.append(_flatten_grads(params))

    projected = []
    order = list(range(len(grads)))
    for i, grad in enumerate(grads):
        g = grad.clone()
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            other = grads[j]
            dot = torch.dot(g, other)
            if dot < 0:
                denom = torch.dot(other, other).clamp_min(1e-12)
                g = g - dot / denom * other
        projected.append(g)
    merged = torch.stack(projected, dim=0).mean(dim=0)
    _assign_flat_grad(params, merged)
