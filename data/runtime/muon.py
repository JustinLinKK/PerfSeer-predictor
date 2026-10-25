"""Practical Muon-plus-Adam optimizer support for profiling workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw", "muon"}


@dataclass(frozen=True)
class ParameterPartition:
    """Disjoint trainable parameter groups for Muon and Adam."""

    muon: tuple[nn.Parameter, ...]
    adam: tuple[nn.Parameter, ...]


def partition_muon_parameters(model: nn.Module) -> ParameterPartition:
    """Put 2-D non-embedding weights in Muon and all other parameters in Adam."""

    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag))
        for parameter in module.parameters(recurse=False)
    }
    muon: list[nn.Parameter] = []
    adam: list[nn.Parameter] = []
    seen: set[int] = set()
    for _name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen or not parameter.requires_grad:
            continue
        seen.add(parameter_id)
        if parameter.ndim == 2 and parameter_id not in embedding_parameter_ids:
            muon.append(parameter)
        else:
            adam.append(parameter)
    return ParameterPartition(muon=tuple(muon), adam=tuple(adam))


def _zeroth_power_newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of source gradient ``matrix:[R,C]``."""

    source_dtype = matrix.dtype
    update = matrix.float()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.transpose(0, 1)
    update = update / (update.norm() + eps)
    for _ in range(steps):
        gram = update @ update.transpose(0, 1)
        correction = -4.775 * gram + 2.0315 * (gram @ gram)
        update = 3.4445 * update + correction @ update
    if transposed:
        update = update.transpose(0, 1)
    return update.to(dtype=source_dtype)


class FallbackMuon(torch.optim.Optimizer):
    """Local 2-D Muon implementation used when native PyTorch Muon is absent."""

    def __init__(self, parameters: list[nn.Parameter], lr: float, momentum: float = 0.95) -> None:
        super().__init__(parameters, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            momentum = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.ndim != 2:
                    raise ValueError(f"Muon parameter must be 2-D, got {tuple(gradient.shape)}")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                nesterov_update = gradient.add(buffer, alpha=momentum)
                orthogonal = _zeroth_power_newton_schulz(nesterov_update)
                scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                parameter.add_(orthogonal, alpha=-learning_rate * scale)
        return loss


class CompositeOptimizer(torch.optim.Optimizer):
    """One optimizer interface backed by Muon and Adam child optimizers."""

    def __init__(self, muon_parameters: tuple[nn.Parameter, ...], adam_parameters: tuple[nn.Parameter, ...], lr: float) -> None:
        groups = [
            {"params": list(muon_parameters), "algorithm": "muon"},
            {"params": list(adam_parameters), "algorithm": "adam"},
        ]
        super().__init__(groups, defaults={})
        native_muon = getattr(torch.optim, "Muon", None)
        self.muon_optimizer: torch.optim.Optimizer = (
            native_muon(muon_parameters, lr=lr, weight_decay=0.0)
            if native_muon is not None
            else FallbackMuon(list(muon_parameters), lr=lr)
        )
        self.adam_optimizer = torch.optim.Adam(adam_parameters, lr=lr)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adam_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None) -> Any | None:
        loss = self.muon_optimizer.step(closure=closure)
        self.adam_optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "muon_adam_composite_v1",
            "muon": self.muon_optimizer.state_dict(),
            "adam": self.adam_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("format") != "muon_adam_composite_v1":
            raise ValueError("unsupported composite optimizer state format")
        self.muon_optimizer.load_state_dict(state_dict["muon"])
        self.adam_optimizer.load_state_dict(state_dict["adam"])


def make_training_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer | None:
    """Create a supported optimizer for all trainable parameters in ``model``."""

    optimizer_name = str(name).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unsupported optimizer {optimizer_name!r}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return None
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    partition = partition_muon_parameters(model)
    if not partition.muon:
        return torch.optim.Adam(partition.adam, lr=lr)
    if not partition.adam:
        native_muon = getattr(torch.optim, "Muon", None)
        return native_muon(partition.muon, lr=lr, weight_decay=0.0) if native_muon is not None else FallbackMuon(list(partition.muon), lr=lr)
    return CompositeOptimizer(partition.muon, partition.adam, lr=lr)


def resolve_profile_optimizer(row: dict[str, Any], dataset_spec: dict[str, Any], default: str) -> str:
    """Resolve optimizer precedence: workload training, manifest row, then CLI."""

    training = dataset_spec.get("training") if isinstance(dataset_spec.get("training"), dict) else {}
    name = str(training.get("optimizer") or row.get("optimizer") or default).lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unsupported optimizer {name!r}")
    return name


def resolve_profile_batch_size(row: dict[str, Any], dataset_spec: dict[str, Any], default: int) -> int:
    """Resolve batch precedence: workload training, manifest row, then input default."""

    training = dataset_spec.get("training") if isinstance(dataset_spec.get("training"), dict) else {}
    raw_batch_size = training.get("batch_size")
    if raw_batch_size is None:
        raw_batch_size = row.get("batch_size")
    if raw_batch_size is None:
        raw_batch_size = default
    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    return batch_size
