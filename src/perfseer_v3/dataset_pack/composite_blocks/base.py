"""Runtime construction and verification for composite operation blocks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from ..composite_sampler import CompositeBlockSpec, CompositeCandidate
from ..fingerprints import canonical_sha256
from ..operation_benchmarks import (
    OperationBenchmarkInstance,
    build_operation_benchmark,
    tensor_outputs,
    verify_operation_benchmark,
)
from ..operation_sampler import (
    OperationCandidate,
    OperationGeneratorRegistry,
    OperationGeneratorSpec,
    build_operation_generator_registry,
)


class CompositeBlockError(ValueError):
    """Raised when a composite block cannot prove its declared interactions."""


@dataclass(frozen=True)
class CompositeExecutionResult:
    context: str
    interaction_requirements: tuple[str, ...]
    executed_operation_ids: tuple[str, ...]
    environment_unsupported_operation_ids: tuple[str, ...]
    loss_before_step: float
    anchor_after_step: float


@dataclass(frozen=True)
class CompositeVerificationResult:
    block_id: str
    candidate_id: str
    verified_operation_ids: tuple[str, ...]
    environment_unsupported_operation_ids: tuple[str, ...]
    interaction_requirements: tuple[str, ...]


def _operation_candidate(
    block_candidate: CompositeCandidate,
    generator: OperationGeneratorSpec,
    operation_index: int,
) -> OperationCandidate:
    payload = {
        "generator_id": generator.generator_id,
        "canonical_operation_id": generator.canonical_operation_id,
        "raw_target": generator.raw_target,
        "family": generator.family,
        "shape_regime": block_candidate.shape_regime,
        "dtype": block_candidate.dtype,
        "accumulation_dtype": block_candidate.accumulation_dtype,
        "phase": block_candidate.phase,
        "backend": block_candidate.backend,
        "layout": block_candidate.layout,
        "optimizer_context": block_candidate.optimizer_context,
        "architecture_context": block_candidate.context,
        "batch_size": block_candidate.batch_size,
        "variant_seed": block_candidate.variant_seed * 1_000_003 + operation_index,
        "coverage_cell_id": block_candidate.coverage_cell_ids[operation_index],
    }
    result = OperationCandidate(candidate_id=canonical_sha256(payload), **payload)
    result.validate(generator)
    return result


@dataclass
class CompositeBlockInstance:
    block: CompositeBlockSpec
    candidate: CompositeCandidate
    operations: tuple[OperationBenchmarkInstance, ...]

    def validate(self) -> None:
        self.block.validate()
        self.candidate.validate(self.block)
        expected = self.block.canonical_operation_ids
        actual = tuple(row.generator.canonical_operation_id for row in self.operations)
        if actual != expected:
            raise CompositeBlockError("composite operation order differs from its block registry")
        for operation in self.operations:
            operation.validate()

    def run(self) -> CompositeExecutionResult:
        self.validate()
        scalar_values: list[torch.Tensor] = []
        executed: list[str] = []
        unsupported: list[str] = []
        for operation in self.operations:
            if not operation.recipe.portable:
                unsupported.append(operation.generator.canonical_operation_id)
                continue
            value = operation.run()
            outputs = tensor_outputs(value)
            scalar_values.extend(
                tensor.detach().to(dtype=torch.float32).mean().reshape(1)
                for tensor in outputs
                if tensor.numel() > 0
            )
            executed.append(operation.generator.canonical_operation_id)
        if not scalar_values:
            raise CompositeBlockError("composite block produced no tensor values")
        values = torch.cat(scalar_values)
        anchor = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
        optimizer: torch.optim.Optimizer
        if self.candidate.optimizer_context == "adamw":
            optimizer = torch.optim.AdamW((anchor,), lr=1e-3)
        else:
            optimizer = torch.optim.SGD((anchor,), lr=1e-3)

        if self.block.context == "sequential":
            combined = values.cumsum(0)[-1].clone()
        elif self.block.context == "residual_or_branch":
            left = values[0::2].sum()
            right = values[1::2].sum() if values.numel() > 1 else values[0]
            combined = left + right + values[0]
        elif self.block.context == "saved_activation_or_alias":
            materialized = values.clone()
            alias = materialized.view(-1)
            saved_activation = alias[: max(1, alias.numel() // 2)].clone()
            combined = alias.sum() + saved_activation.sum()
        else:
            raise CompositeBlockError("unknown composite context")

        optimizer.zero_grad(set_to_none=True)
        loss = combined * anchor
        loss.backward()
        before = float(loss.detach())
        optimizer.step()
        return CompositeExecutionResult(
            context=self.block.context,
            interaction_requirements=self.block.interaction_requirements,
            executed_operation_ids=tuple(executed),
            environment_unsupported_operation_ids=tuple(unsupported),
            loss_before_step=before,
            anchor_after_step=float(anchor.detach()),
        )


def _build(
    block: CompositeBlockSpec,
    candidate: CompositeCandidate,
    generators: OperationGeneratorRegistry,
) -> CompositeBlockInstance:
    by_operation = {row.canonical_operation_id: row for row in generators.generators}
    operations = []
    for index, operation_id in enumerate(block.canonical_operation_ids):
        generator = by_operation.get(operation_id)
        if generator is None:
            raise CompositeBlockError(f"composite operation {operation_id!r} has no generator")
        operation_candidate = _operation_candidate(candidate, generator, index)
        operations.append(build_operation_benchmark(generator, operation_candidate))
    result = CompositeBlockInstance(block, candidate, tuple(operations))
    result.validate()
    return result


def build_composite_block(
    block: CompositeBlockSpec,
    candidate: CompositeCandidate,
    *,
    generators: OperationGeneratorRegistry | None = None,
) -> CompositeBlockInstance:
    generators = generators or build_operation_generator_registry()
    generators.validate()
    module = importlib.import_module(block.factory_id)
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise CompositeBlockError(f"composite factory {block.factory_id!r} has no build function")
    result = builder(block, candidate, generators)
    if not isinstance(result, CompositeBlockInstance):
        raise CompositeBlockError("composite factory returned the wrong instance type")
    result.validate()
    return result


def verify_composite_block(
    instance: CompositeBlockInstance,
    *,
    generators: OperationGeneratorRegistry | None = None,
) -> CompositeVerificationResult:
    generators = generators or build_operation_generator_registry()
    execution = instance.run()
    verified: list[str] = []
    unsupported: list[str] = []
    for operation in instance.operations:
        result = verify_operation_benchmark(operation, generators=generators)
        target = operation.generator.canonical_operation_id
        if result.status == "verified":
            verified.append(target)
        elif result.status == "environment_unsupported":
            unsupported.append(target)
        else:
            raise CompositeBlockError(f"unexpected verification state for {target}")
    if tuple(verified) != execution.executed_operation_ids:
        raise CompositeBlockError("composite execution and exact verification disagree")
    if tuple(unsupported) != execution.environment_unsupported_operation_ids:
        raise CompositeBlockError("composite unsupported identities disagree")
    return CompositeVerificationResult(
        block_id=instance.block.block_id,
        candidate_id=instance.candidate.candidate_id,
        verified_operation_ids=tuple(verified),
        environment_unsupported_operation_ids=tuple(unsupported),
        interaction_requirements=instance.block.interaction_requirements,
    )


__all__ = [
    "CompositeBlockError",
    "CompositeBlockInstance",
    "CompositeExecutionResult",
    "CompositeVerificationResult",
    "build_composite_block",
    "verify_composite_block",
]
