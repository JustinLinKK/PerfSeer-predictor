"""Runtime types and dynamic dispatch for operation benchmark recipes."""

from __future__ import annotations

import importlib
import operator
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from ..dispatcher_verify import (
    DISPATCH_EVIDENCE_VERSION,
    DISPATCH_REQUEST_VERSION,
    DispatchEvidence,
    DispatchRequest,
    DispatchVerification,
    verify_dispatch,
)
from ..fingerprints import canonical_sha256
from ..operation_sampler import (
    OperationCandidate,
    OperationGeneratorSpec,
    build_operation_generator_registry,
)


class OperationBenchmarkError(ValueError):
    """Raised when a declared generator cannot build or execute its operation."""


class CallableModule(nn.Module):
    def __init__(self, function: Callable[..., Any]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


def _raw_dispatch_target(function: object) -> str:
    value = str(function)
    if "." not in value:
        return value
    namespace, remainder = value.split(".", 1)
    if remainder.endswith(".default"):
        remainder = remainder.removesuffix(".default")
    return f"{namespace}::{remainder}"


class _IdentityTraceMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.raw_targets: list[str] = []

    def __torch_dispatch__(
        self,
        function: object,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.raw_targets.append(_raw_dispatch_target(function))
        return function(*args, **(kwargs or {}))


class _OperatorCallTraceMode(TorchFunctionMode):
    """Capture the requested public/OpOverload call before composite decomposition."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_targets: list[str] = []

    def __torch_function__(
        self,
        function: object,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.raw_targets.append(_raw_dispatch_target(function))
        return function(*args, **(kwargs or {}))


@dataclass
class BenchmarkRecipe:
    module: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    identity_source: str
    portable: bool = True
    unsupported_reason: str | None = None
    semantic_exception: str | None = None
    optimizer_step: Callable[[], torch.Tensor] | None = None

    def validate(self) -> None:
        if self.identity_source not in {
            "dispatcher_trace",
            "capture_semantic_summary",
            "perfseer_phase_annotation",
        }:
            raise OperationBenchmarkError("benchmark recipe identity source is invalid")
        if type(self.portable) is not bool:
            raise OperationBenchmarkError("benchmark recipe portable flag must be boolean")
        if self.portable and self.unsupported_reason is not None:
            raise OperationBenchmarkError("portable benchmark cannot retain an unsupported reason")
        if not self.portable and (type(self.unsupported_reason) is not str or not self.unsupported_reason):
            raise OperationBenchmarkError("non-portable benchmark requires a written reason")
        if self.semantic_exception is not None and (
            type(self.semantic_exception) is not str or not self.semantic_exception
        ):
            raise OperationBenchmarkError("semantic exception must be a non-empty string")

    def run(self) -> Any:
        self.validate()
        if not self.portable:
            raise OperationBenchmarkError(self.unsupported_reason or "benchmark is not portable")
        if self.optimizer_step is not None:
            return self.optimizer_step()
        return self.module(*self.args, **self.kwargs)


@dataclass
class OperationBenchmarkInstance:
    generator: OperationGeneratorSpec
    candidate: OperationCandidate
    recipe: BenchmarkRecipe

    def validate(self) -> None:
        self.generator.validate()
        self.candidate.validate(self.generator)
        self.recipe.validate()
        expected_source = {
            "optimizer_step_annotation": "perfseer_phase_annotation",
            "training_graph_annotation": "perfseer_phase_annotation",
            "capture_observed_python_semantics": "capture_semantic_summary",
        }.get(self.generator.execution_route, "dispatcher_trace")
        if self.recipe.identity_source != expected_source:
            raise OperationBenchmarkError("recipe identity source differs from generator route")

    def run(self) -> Any:
        self.validate()
        result = self.recipe.run()
        if not tensor_outputs(result) and self.recipe.semantic_exception is None:
            raise OperationBenchmarkError(
                f"generator {self.generator.generator_id} produced no tensor output"
            )
        return result


def tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, dict):
        result = []
        for key in sorted(value, key=str):
            result.extend(tensor_outputs(value[key]))
        return tuple(result)
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensor_outputs(item))
        return tuple(result)
    return ()


def build_operation_benchmark(
    generator: OperationGeneratorSpec,
    candidate: OperationCandidate,
) -> OperationBenchmarkInstance:
    generator.validate()
    candidate.validate(generator)
    module_name = generator.factory_id
    module = importlib.import_module(module_name)
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise OperationBenchmarkError(f"operation factory {module_name!r} has no build function")
    recipe = builder(generator, candidate)
    if not isinstance(recipe, BenchmarkRecipe):
        raise OperationBenchmarkError("operation factory returned the wrong recipe type")
    instance = OperationBenchmarkInstance(generator, candidate, recipe)
    instance.validate()
    return instance


def verify_operation_benchmark(
    instance: OperationBenchmarkInstance,
    *,
    backend_id: str = "cpu_eager",
    target_hardware_id: str = "local_cpu_fixture",
    generators=None,
) -> DispatchVerification:
    """Execute one local smoke and bind the actual dispatcher overload/backend."""

    instance.validate()
    generator = instance.generator
    recipe = instance.recipe
    generator_registry = generators or build_operation_generator_registry()
    request = DispatchRequest(
        version=DISPATCH_REQUEST_VERSION,
        canonical_operation_id=generator.canonical_operation_id,
        raw_target=generator.raw_target,
        aliases=generator.aliases,
        generator_registry_sha256=generator_registry.sha256,
        measurement_scope="local_smoke",
        requested_backend_id=backend_id,
        identity_source=recipe.identity_source,
        environment_gated=generator.execution_route == "a10g_environment_gated_dispatcher",
        specialized_backend=generator.execution_route == "a10g_environment_gated_dispatcher",
    )
    workload_sha256 = canonical_sha256(
        {
            "generator": generator.generator_id,
            "candidate": instance.candidate.candidate_id,
        }
    )
    if not recipe.portable:
        evidence = DispatchEvidence(
            version=DISPATCH_EVIDENCE_VERSION,
            request_sha256=canonical_sha256(request),
            target_hardware_id=target_hardware_id,
            measurement_occurrence_id=f"local_smoke:{instance.candidate.candidate_id}",
            accepted_measurement=False,
            supported=False,
            unsupported_reason=recipe.unsupported_reason,
            capture_workload_sha256=workload_sha256,
            profile_workload_sha256=workload_sha256,
            identity_source=recipe.identity_source,
            observed_raw_targets=(),
            observed_backend_ids=(),
        )
        return verify_dispatch(request, evidence, generators=generator_registry)
    if recipe.identity_source == "dispatcher_trace":
        dispatch_trace = _IdentityTraceMode()
        operator_trace = _OperatorCallTraceMode()
        with operator_trace:
            with dispatch_trace:
                instance.run()
        observed = tuple(
            dict.fromkeys((*operator_trace.raw_targets, *dispatch_trace.raw_targets))
        )
    elif recipe.identity_source == "perfseer_phase_annotation":
        instance.run()
        observed = (generator.raw_target,)
    elif recipe.identity_source == "capture_semantic_summary":
        instance.run()
        if generator.canonical_operation_id == "prim.getitem":
            scripted = getattr(recipe.module, "function", None)
            graph = getattr(scripted, "graph", None)
            observed = tuple(
                node.kind()
                for node in graph.nodes()
                if node.kind() == "prim::TupleIndex"
            ) if graph is not None else ()
        elif generator.canonical_operation_id == "operator.getitem":
            graph_module = torch.fx.symbolic_trace(recipe.module.function)
            observed = tuple(
                "operator::getitem"
                for node in graph_module.graph.nodes
                if node.op == "call_function" and node.target is operator.getitem
            )
        elif generator.canonical_operation_id == "aten.sym_size":
            graph_module = make_fx(recipe.module, tracing_mode="symbolic")(
                *recipe.args, **recipe.kwargs
            )
            observed = tuple(
                _raw_dispatch_target(node.target)
                for node in graph_module.graph.nodes
                if node.op == "call_function"
            )
        else:
            raise OperationBenchmarkError("unsupported capture-semantic verification route")
    else:
        raise OperationBenchmarkError("unsupported operation identity source")
    evidence = DispatchEvidence(
        version=DISPATCH_EVIDENCE_VERSION,
        request_sha256=canonical_sha256(request),
        target_hardware_id=target_hardware_id,
        measurement_occurrence_id=f"local_smoke:{instance.candidate.candidate_id}",
        accepted_measurement=False,
        supported=True,
        unsupported_reason=None,
        capture_workload_sha256=workload_sha256,
        profile_workload_sha256=workload_sha256,
        identity_source=recipe.identity_source,
        observed_raw_targets=observed,
        observed_backend_ids=(backend_id,),
    )
    return verify_dispatch(request, evidence, generators=generator_registry)


__all__ = [
    "BenchmarkRecipe",
    "CallableModule",
    "OperationBenchmarkError",
    "OperationBenchmarkInstance",
    "build_operation_benchmark",
    "tensor_outputs",
    "verify_operation_benchmark",
]
