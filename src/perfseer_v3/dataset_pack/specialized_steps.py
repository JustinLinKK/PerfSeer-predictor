"""Task-level golden fixtures not represented by a standalone quota family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from perfseer_v3.op_registry import OperationRegistry

from .models.base import _canonicalize_targets, _trace_call


class SpecializedStepError(ValueError):
    """Raised when a task-level specialized training fixture is not executable."""


@dataclass(frozen=True)
class SpecializedStepResult:
    version: str
    step_id: str
    task_kind: str
    requested_backend_id: str
    observed_backend_ids: tuple[str, ...]
    faithful_operation_assertions: tuple[str, ...]
    observed_canonical_operations: tuple[str, ...]
    output_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    initial_loss: float
    final_loss: float
    gradient_norm: float
    changed_parameter_groups: tuple[str, ...]
    task_shape_validated: bool
    training_approved: bool

    def validate(self) -> None:
        if self.version != "perfseer_v3_specialized_step_v1":
            raise SpecializedStepError("specialized-step version mismatch")
        if not self.step_id or not self.task_kind:
            raise SpecializedStepError("specialized-step identity must be non-empty")
        if self.observed_backend_ids != (self.requested_backend_id,):
            raise SpecializedStepError("specialized-step backend mismatch")
        if not set(self.faithful_operation_assertions).issubset(
            self.observed_canonical_operations
        ):
            raise SpecializedStepError("specialized step did not observe its declared operations")
        if not self.output_shape or not self.target_shape or not self.task_shape_validated:
            raise SpecializedStepError("specialized-step output/target shape was not validated")
        for value in (self.initial_loss, self.final_loss, self.gradient_norm):
            if not math.isfinite(value):
                raise SpecializedStepError("specialized-step scalar is non-finite")
        if self.final_loss >= self.initial_loss or self.gradient_norm <= 0.0:
            raise SpecializedStepError("specialized task fixture did not learn")
        if not self.changed_parameter_groups:
            raise SpecializedStepError("specialized task fixture changed no parameter group")
        if self.training_approved is not False:
            raise SpecializedStepError("local specialized fixture cannot approve training")


def _run_step(
    *,
    step_id: str,
    task_kind: str,
    parameters: dict[str, torch.nn.Parameter],
    forward: Callable[[], torch.Tensor],
    target: torch.Tensor,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    assertions: tuple[str, ...],
) -> SpecializedStepResult:
    optimizer = torch.optim.AdamW(tuple(parameters.values()), lr=0.03, weight_decay=0.0)
    before = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    output, forward_raw = _trace_call(forward)
    loss, loss_raw = _trace_call(lambda: loss_function(output, target))
    initial = float(loss.detach())
    _, backward_raw = _trace_call(loss.backward)
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().float().pow(2).sum())
            for parameter in parameters.values()
            if parameter.grad is not None
        )
    )
    _, optimizer_raw = _trace_call(optimizer.step)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        current = loss_function(forward(), target)
        current.backward()
        optimizer.step()
    final = float(loss_function(forward(), target).detach())
    raw = tuple(dict.fromkeys((*forward_raw, *loss_raw, *backward_raw, *optimizer_raw)))
    observed = _canonicalize_targets(OperationRegistry.load(), raw)
    changed = tuple(
        sorted(
            name
            for name, parameter in parameters.items()
            if not torch.equal(before[name], parameter.detach())
        )
    )
    result = SpecializedStepResult(
        version="perfseer_v3_specialized_step_v1",
        step_id=step_id,
        task_kind=task_kind,
        requested_backend_id="cpu_eager",
        observed_backend_ids=("cpu_eager",),
        faithful_operation_assertions=assertions,
        observed_canonical_operations=observed,
        output_shape=tuple(output.shape),
        target_shape=tuple(target.shape),
        initial_loss=initial,
        final_loss=final,
        gradient_norm=gradient_norm,
        changed_parameter_groups=changed,
        task_shape_validated=True,
        training_approved=False,
    )
    result.validate()
    return result


def run_volumetric_training_fixture() -> SpecializedStepResult:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(30_003)
    volume = torch.randn((2, 1, 8, 8, 8), generator=generator)
    target = torch.randn((2, 3, 8, 8, 8), generator=generator).mul(0.1)
    weight = torch.nn.Parameter(torch.randn((3, 1, 3, 3, 3), generator=generator).mul(0.05))
    bias = torch.nn.Parameter(torch.zeros(3))
    return _run_step(
        step_id="volumetric_3d_convolution_training",
        task_kind="volumetric_dense_prediction",
        parameters={"volume_encoder": weight, "volume_bias": bias},
        forward=lambda: torch.ops.aten.conv3d.default(
            volume, weight, bias, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1
        ),
        target=target,
        loss_function=F.mse_loss,
        assertions=("aten.convolution.3d",),
    )


def run_segmentation_training_fixture() -> SpecializedStepResult:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20_002)
    image = torch.randn((3, 3, 16, 16), generator=generator)
    target = torch.randint(0, 2, (3, 16, 16), generator=generator)
    weight = torch.nn.Parameter(torch.randn((2, 3, 3, 3), generator=generator).mul(0.05))
    bias = torch.nn.Parameter(torch.zeros(2))
    return _run_step(
        step_id="semantic_segmentation_training",
        task_kind="pixelwise_classification",
        parameters={"segmentation_head": weight, "segmentation_bias": bias},
        forward=lambda: torch.ops.aten.conv2d.default(
            image, weight, bias, [1, 1], [1, 1], [1, 1], 1
        ),
        target=target,
        loss_function=lambda output, labels: torch.ops.aten.cross_entropy_loss.default(
            output, labels, None, 1, -100, 0.0
        ),
        assertions=("aten.convolution.2d", "aten.cross_entropy_loss"),
    )


def run_specialized_training_fixtures() -> tuple[SpecializedStepResult, ...]:
    return (run_volumetric_training_fixture(), run_segmentation_training_fixture())


__all__ = [
    "SpecializedStepError",
    "SpecializedStepResult",
    "run_segmentation_training_fixture",
    "run_specialized_training_fixtures",
    "run_volumetric_training_fixture",
]
