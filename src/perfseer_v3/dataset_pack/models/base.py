"""Runtime model-factory and golden-training contracts for Phase 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from perfseer_v3.op_registry import OperationRegistry

from ..adapters.base import TaskAdapter
from ..fingerprints import canonical_sha256, canonical_value
from .architecture import build_semantic_field_evidence


class ModelFactoryError(ValueError):
    """Raised when a family factory or golden training step is not faithful."""


@dataclass(frozen=True)
class FamilyBuildConfig:
    output_width: int
    task_kind: str
    seed: int = 0
    architecture_parameters: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if type(self.output_width) is not int or self.output_width < 1:
            raise ModelFactoryError("model output width must be positive")
        if type(self.task_kind) is not str or not self.task_kind:
            raise ModelFactoryError("model task kind must be non-empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ModelFactoryError("model seed must be a nonnegative integer")
        canonical_value(dict(self.architecture_parameters or {}))


def _raw_dispatch_target(function: object) -> str:
    value = str(function)
    if value.startswith(("aten.", "prim.", "prims.")):
        namespace, remainder = value.split(".", 1)
        if remainder.endswith(".default"):
            remainder = remainder.removesuffix(".default")
        return f"{namespace}::{remainder}"
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", None) or getattr(function, "__name__", None)
    if type(module) is str and module and type(name) is str and name:
        return f"python::{module}.{name}"
    if type(name) is str and name:
        return f"python::{type(function).__module__}.{type(function).__qualname__}.{name}"
    return f"python::{type(function).__module__}.{type(function).__qualname__}"


class _DispatchTrace(TorchDispatchMode):
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


class _FunctionTrace(TorchFunctionMode):
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


def _trace_call(function: Any) -> tuple[Any, tuple[str, ...]]:
    dispatch = _DispatchTrace()
    public = _FunctionTrace()
    with public:
        with dispatch:
            result = function()
    return result, tuple(dict.fromkeys((*public.raw_targets, *dispatch.raw_targets)))


def tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(tensor_outputs(value[key]))
        return tuple(result)
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensor_outputs(item))
        return tuple(result)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return tensor_outputs(to_dict())
    return ()


class FamilyModel(nn.Module):
    family_id: str

    def __init__(self) -> None:
        super().__init__()
        self.architecture_parameters: Mapping[str, Any] = {}
        self.architecture_semantic_roles: dict[str, str] = {}
        self._architecture_field_evidence: Mapping[str, Mapping[str, Any]] = {}

    def bind_architecture(self, parameters: Mapping[str, Any]) -> None:
        if self.architecture_parameters:
            raise ModelFactoryError("architecture parameters are already bound")
        self.architecture_parameters = canonical_value(dict(parameters))
        self._architecture_field_evidence = build_semantic_field_evidence(
            self.family_id,
            self.architecture_parameters,
            self.architecture_semantic_roles,
        )

    @property
    def architecture_field_evidence(self) -> Mapping[str, Mapping[str, Any]]:
        if not self._architecture_field_evidence:
            raise ModelFactoryError("family architecture parameters were not bound")
        return self._architecture_field_evidence

    @property
    def architecture_execution_sha256(self) -> str:
        return canonical_sha256(
            {
                "family_id": self.family_id,
                "parameters": self.architecture_parameters,
                "field_evidence": self.architecture_field_evidence,
                "state_shapes": {
                    name: list(value.shape) for name, value in self.state_dict().items()
                },
            }
        )

    def compute_loss(
        self,
        output: Any,
        batch: Mapping[str, Any],
        adapter: TaskAdapter,
    ) -> torch.Tensor:
        return adapter.build_loss(output, adapter.build_targets(batch))

    def metric_output(self, output: Any) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise ModelFactoryError("family must select a tensor for the task metric")
        return output


@dataclass(frozen=True)
class GoldenValidationResult:
    version: str
    family_id: str
    task_id: str
    factory_id: str
    fixture_fingerprint: str
    requested_backend_id: str
    observed_backend_ids: tuple[str, ...]
    phase_raw_targets: Mapping[str, tuple[str, ...]]
    phase_canonical_operations: Mapping[str, tuple[str, ...]]
    profiler_operator_names: tuple[str, ...]
    capture_raw_targets: tuple[str, ...]
    capture_canonical_operations: tuple[str, ...]
    faithful_operation_assertions: tuple[str, ...]
    verified_assertions: tuple[str, ...]
    initial_loss: float
    best_loss: float
    final_loss: float
    gradient_norm: float
    changed_parameter_count: int
    changed_parameter_names: tuple[str, ...]
    smoke_metric: float
    eager_replay_equivalent: bool
    capture_route: str
    capture_eager_equivalent: bool
    architecture_parameters: Mapping[str, Any]
    architecture_field_evidence: Mapping[str, Mapping[str, Any]]
    architecture_execution_sha256: str
    pyg_batch_consumed: bool | None
    mixed_precision_status: str
    mixed_precision_dtype: str
    training_approved: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.version != "perfseer_v3_model_golden_v1":
            raise ModelFactoryError("model golden result version mismatch")
        for name in ("family_id", "task_id", "factory_id", "requested_backend_id", "capture_route"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ModelFactoryError(f"golden result {name} must be non-empty")
        if len(self.fixture_fingerprint) != 64:
            raise ModelFactoryError("golden result fixture fingerprint must be SHA-256")
        if self.observed_backend_ids != (self.requested_backend_id,):
            raise ModelFactoryError("golden result backend identity mismatch")
        if tuple(self.phase_raw_targets) != ("forward", "loss", "backward", "optimizer"):
            raise ModelFactoryError("golden result phase attribution is incomplete")
        if tuple(self.phase_canonical_operations) != tuple(self.phase_raw_targets):
            raise ModelFactoryError("golden result canonical phase attribution differs")
        if not self.profiler_operator_names:
            raise ModelFactoryError("golden result requires a profiler trace")
        if not self.capture_raw_targets or not self.capture_canonical_operations:
            raise ModelFactoryError("golden result requires an actual captured graph")
        if self.verified_assertions != self.faithful_operation_assertions:
            raise ModelFactoryError("golden result did not verify every declared operation")
        for value in (self.initial_loss, self.best_loss, self.final_loss, self.gradient_norm, self.smoke_metric):
            if not math.isfinite(value):
                raise ModelFactoryError("golden result contains a non-finite scalar")
        if self.best_loss >= self.initial_loss:
            raise ModelFactoryError("tiny deterministic training did not decrease loss")
        if (
            self.gradient_norm <= 0.0
            or self.changed_parameter_count < 1
            or self.changed_parameter_count != len(self.changed_parameter_names)
            or tuple(sorted(set(self.changed_parameter_names))) != self.changed_parameter_names
        ):
            raise ModelFactoryError("golden training step did not update trainable parameters")
        if not 0.0 <= self.smoke_metric <= 1.0:
            raise ModelFactoryError("golden task metric is outside [0, 1]")
        if self.eager_replay_equivalent is not True or self.capture_eager_equivalent is not True:
            raise ModelFactoryError("golden eager/captured replay is not equivalent")
        if (
            not self.architecture_parameters
            or set(self.architecture_field_evidence) != set(self.architecture_parameters)
            or len(self.architecture_execution_sha256) != 64
        ):
            raise ModelFactoryError("golden architecture fields are not execution-bound")
        if (self.family_id == "cgcnn") != (self.pyg_batch_consumed is True):
            raise ModelFactoryError("CGCNN PyG Batch consumption evidence is invalid")
        if self.family_id != "cgcnn" and self.pyg_batch_consumed is not None:
            raise ModelFactoryError("non-CGCNN family has unexpected PyG consumption evidence")
        if self.mixed_precision_status not in {"verified", "unsupported_local_cpu"}:
            raise ModelFactoryError("golden mixed-precision status is invalid")
        if self.mixed_precision_dtype != "bfloat16":
            raise ModelFactoryError("golden mixed-precision dtype must be bfloat16")
        if self.training_approved is not False:
            raise ModelFactoryError("local golden evidence cannot approve production training")


def _canonicalize_targets(
    registry: OperationRegistry,
    raw_targets: tuple[str, ...],
) -> tuple[str, ...]:
    result = []
    for raw in raw_targets:
        resolved = registry.resolve(raw)
        if resolved.is_known:
            result.append(resolved.canonical_id)
    return tuple(dict.fromkeys(result))


def _nested_close(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.allclose(left, right, rtol=1e-5, atol=1e-6)
    if isinstance(left, Mapping) and isinstance(right, Mapping) and tuple(left) == tuple(right):
        return all(_nested_close(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(_nested_close(a, b) for a, b in zip(left, right))
    return left == right


def _verify_pyg_batch_consumption(
    model: FamilyModel,
    inputs: Mapping[str, Any],
) -> bool | None:
    if model.family_id != "cgcnn":
        return None
    pyg_batch = inputs.get("pyg_batch")
    if pyg_batch is None or not callable(getattr(pyg_batch, "clone", None)):
        return False
    poisoned_explicit = dict(inputs)
    for name in ("node_features", "edge_features"):
        value = poisoned_explicit.get(name)
        if isinstance(value, torch.Tensor):
            poisoned_explicit[name] = torch.full_like(value, 1_000.0)
    if isinstance(poisoned_explicit.get("edge_index"), torch.Tensor):
        poisoned_explicit["edge_index"] = torch.zeros_like(
            poisoned_explicit["edge_index"]
        )
    if isinstance(poisoned_explicit.get("graph_index"), torch.Tensor):
        poisoned_explicit["graph_index"] = torch.zeros_like(
            poisoned_explicit["graph_index"]
        )
    changed_pyg = pyg_batch.clone()
    changed_pyg.x = changed_pyg.x.add(1.0)
    changed_inputs = dict(inputs)
    changed_inputs["pyg_batch"] = changed_pyg
    missing_inputs = dict(inputs)
    missing_inputs["pyg_batch"] = None
    with torch.no_grad():
        baseline = model(inputs)
        explicit_poison = model(poisoned_explicit)
        pyg_changed = model(changed_inputs)
    rejected_missing = False
    try:
        model(missing_inputs)
    except ModelFactoryError:
        rejected_missing = True
    return bool(
        _nested_close(baseline, explicit_poison)
        and not _nested_close(baseline, pyg_changed)
        and rejected_missing
    )


def golden_validate_model(
    *,
    model: FamilyModel,
    adapter: TaskAdapter,
    factory_id: str,
    assertions: tuple[str, ...],
    backend_id: str = "cpu_eager",
    optimization_steps: int = 6,
) -> GoldenValidationResult:
    """Run a deterministic forward/loss/backward/update and strict identity audit."""

    if optimization_steps < 2:
        raise ModelFactoryError("golden validation needs at least two optimization steps")
    prepared = adapter.prepare_dataset()
    batch = adapter.build_collator()(adapter.build_train_dataset())
    inputs = adapter.build_model_inputs(batch)
    target = adapter.build_targets(batch)
    model.train()
    execution_tensors = tensor_outputs((inputs, batch))
    model_state_tensors = tuple(model.parameters()) + tuple(model.buffers())
    derived_backend_id = (
        "cpu_eager"
        if execution_tensors
        and all(tensor.device.type == "cpu" for tensor in execution_tensors)
        and all(tensor.device.type == "cpu" for tensor in model_state_tensors)
        else "unsupported_non_cpu_local_fixture"
    )
    if backend_id != derived_backend_id:
        raise ModelFactoryError(
            f"requested backend {backend_id!r} differs from "
            f"execution-derived {derived_backend_id!r}"
        )
    pyg_batch_consumed = _verify_pyg_batch_consumption(model, inputs)
    learning_rate = float(getattr(model, "golden_learning_rate", 0.01))
    if not 0.0 < learning_rate <= 0.1:
        raise ModelFactoryError("golden learning rate is outside safe fixture bounds")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    optimizer.zero_grad(set_to_none=True)
    output, forward_raw = _trace_call(lambda: model(inputs))
    loss, loss_raw = _trace_call(lambda: model.compute_loss(output, batch, adapter))
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ModelFactoryError(f"family {model.family_id} produced an invalid scalar loss")
    initial_loss = float(loss.detach())
    _, backward_raw = _trace_call(loss.backward)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = math.sqrt(sum(float(gradient.detach().float().pow(2).sum()) for gradient in gradients))
    _, optimizer_raw = _trace_call(optimizer.step)
    changed_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(before[name], parameter.detach())
        )
    )
    changed = len(changed_names)

    mixed_precision_status = "verified"
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_output = model(inputs)
            mixed_loss = model.compute_loss(mixed_output, batch, adapter)
        if not torch.isfinite(mixed_loss):
            raise ModelFactoryError("mixed-precision loss is non-finite")
        mixed_loss.backward()
        mixed_gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not mixed_gradients or not all(
            torch.isfinite(gradient).all() for gradient in mixed_gradients
        ):
            raise ModelFactoryError("mixed-precision gradients are invalid")
    except (RuntimeError, NotImplementedError):
        mixed_precision_status = "unsupported_local_cpu"
    optimizer.zero_grad(set_to_none=True)

    losses = [initial_loss]
    for _ in range(optimization_steps - 1):
        optimizer.zero_grad(set_to_none=True)
        current_output = model(inputs)
        current_loss = model.compute_loss(current_output, batch, adapter)
        if not torch.isfinite(current_loss):
            raise ModelFactoryError(f"family {model.family_id} became non-finite")
        current_loss.backward()
        optimizer.step()
        losses.append(float(current_loss.detach()))

    model.eval()
    with torch.no_grad():
        replay_a, _ = _trace_call(lambda: model(inputs))
        replay_b = model(inputs)
        eager_equivalent = _nested_close(replay_a, replay_b)
        metric_output = model.metric_output(replay_b)
        adapter.validate_output_shape(metric_output, target)
        smoke_metric = adapter.compute_smoke_metric(metric_output, target)
        eager_loss = model.compute_loss(replay_b, batch, adapter)

    captured = make_fx(
        lambda captured_inputs: model.compute_loss(
            model(captured_inputs), batch, adapter
        )
    )(inputs)
    with torch.no_grad():
        captured_loss = captured(inputs)
    capture_eager_equivalent = bool(
        torch.allclose(eager_loss, captured_loss, rtol=1e-5, atol=1e-6)
    )
    capture_raw = tuple(
        dict.fromkeys(
            _raw_dispatch_target(node.target)
            for node in captured.graph.nodes
            if node.op == "call_function"
        )
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
    ) as profiler:
        with torch.no_grad():
            profiled = model(inputs)
            model.compute_loss(profiled, batch, adapter)
    profiler_names = tuple(sorted({event.key for event in profiler.key_averages() if event.count > 0}))

    operation_registry = OperationRegistry.load()
    capture_canonical = _canonicalize_targets(operation_registry, capture_raw)
    phase_raw = {
        "forward": forward_raw,
        "loss": loss_raw,
        "backward": backward_raw,
        "optimizer": optimizer_raw,
    }
    phase_canonical = {
        phase: _canonicalize_targets(operation_registry, raw)
        for phase, raw in phase_raw.items()
    }
    all_canonical = set(capture_canonical).union(
        operation
        for operations_in_phase in phase_canonical.values()
        for operation in operations_in_phase
    )
    structural_observed = {
        raw.replace("::", ".")
        for raw in (
            *capture_raw,
            *(raw for targets in phase_raw.values() for raw in targets),
        )
        if raw.startswith(("aten::", "prim::", "prims::"))
    }
    verified = []
    for assertion in assertions:
        if assertion == "structural:all_tensor_nodes":
            if tensor_outputs(output):
                verified.append(assertion)
            continue
        expected = assertion.removeprefix("structural:")
        if expected in all_canonical or (
            assertion.startswith("structural:") and expected in structural_observed
        ):
            verified.append(assertion)
    result = GoldenValidationResult(
        version="perfseer_v3_model_golden_v1",
        family_id=model.family_id,
        task_id=adapter.task_id,
        factory_id=factory_id,
        fixture_fingerprint=prepared.fingerprint,
        requested_backend_id=backend_id,
        observed_backend_ids=(derived_backend_id,),
        phase_raw_targets=phase_raw,
        phase_canonical_operations=phase_canonical,
        profiler_operator_names=profiler_names,
        capture_raw_targets=capture_raw,
        capture_canonical_operations=capture_canonical,
        faithful_operation_assertions=assertions,
        verified_assertions=tuple(verified),
        initial_loss=initial_loss,
        best_loss=min(losses[1:]),
        final_loss=losses[-1],
        gradient_norm=gradient_norm,
        changed_parameter_count=changed,
        changed_parameter_names=changed_names,
        smoke_metric=smoke_metric,
        eager_replay_equivalent=eager_equivalent,
        capture_route="make_fx_forward_loss_plus_dispatch_and_cpu_profiler",
        capture_eager_equivalent=capture_eager_equivalent,
        architecture_parameters=model.architecture_parameters,
        architecture_field_evidence=model.architecture_field_evidence,
        architecture_execution_sha256=model.architecture_execution_sha256,
        pyg_batch_consumed=pyg_batch_consumed,
        mixed_precision_status=mixed_precision_status,
        mixed_precision_dtype="bfloat16",
        training_approved=False,
    )
    result.validate()
    return result


__all__ = [
    "FamilyBuildConfig",
    "FamilyModel",
    "GoldenValidationResult",
    "ModelFactoryError",
    "golden_validate_model",
    "tensor_outputs",
]
