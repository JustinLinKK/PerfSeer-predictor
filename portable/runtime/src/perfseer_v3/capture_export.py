"""Strict-first ``torch.export`` capture into complete GraphIRV3."""

from __future__ import annotations

import hashlib
import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from .canonicalize import (
    normalize_constraints,
    normalized_node_arguments,
    raw_target_name,
    source_module_stack,
)
from .diagnostics import CaptureFailureV3
from .cost_registry_v3 import estimate_fx_node
from .graph_ir_v3 import (
    CoverageQuality,
    Estimate,
    GraphGlobalFeatures,
    GraphIRV3,
    OperationNodeV3,
    TensorEdgeV3,
)
from .hardware import canonical_hardware_id
from .inputs import input_signature
from .liveness_v3 import apply_liveness
from .op_registry import OperationRegistry
from .schema import build_feature_schema
from .tensor_metadata import (
    clone_inputs,
    compare_output_pytrees,
    flatten_nodes,
    flatten_tensor_values,
    randomized_like,
    tensor_metadata,
)


ReplayInputFactory = Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]
_WIDENED_ACCUMULATION_FAMILIES = frozenset(
    {
        "attention",
        "convolution",
        "dense_matrix",
        "loss_probability",
        "normalization",
        "reduction",
        "training",
        "optimizer",
    }
)


def infer_accumulation_dtype(family: str, output_dtype: str) -> str:
    """Return the explicit analytical accumulation-dtype category."""

    if family in _WIDENED_ACCUMULATION_FAMILIES:
        if output_dtype in {"float16", "bfloat16"}:
            return "float32"
        if output_dtype in {"int8", "uint8"}:
            return "int32"
    return output_dtype


@dataclass(frozen=True)
class CaptureOptions:
    allow_non_strict: bool = True
    replay_samples: int = 3
    replay_rtol: float = 1e-4
    replay_atol: float = 1e-5
    training_mode: bool = False
    precision: str = "float32"
    optimizer_config: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    target_hardware_id: str = "unknown"
    hardware_features: dict[str, Any] | None = None
    apply_selective_decomposition: bool = True

    def __post_init__(self) -> None:
        if self.replay_samples < 3:
            raise ValueError("replay_samples must be >= 3 for validated non-strict capture")
        if self.optimizer_config is not None and not isinstance(self.optimizer_config, dict):
            raise ValueError("optimizer_config must be an object")
        if self.training_config is not None and not isinstance(self.training_config, dict):
            raise ValueError("training_config must be an object")
        if self.hardware_features is not None and not isinstance(self.hardware_features, dict):
            raise ValueError("hardware_features must be an object")


@dataclass(frozen=True)
class CaptureResult:
    graph: GraphIRV3 | None
    exported_program: Any | None
    failures: tuple[CaptureFailureV3, ...]
    model_object_id: int | None = None
    callable_qualname: str | None = None

    @property
    def success(self) -> bool:
        return self.graph is not None


def _model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model.__class__.__module__}.{model.__class__.__qualname__}".encode("utf-8"))
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _placeholder_roles(exported_program: Any) -> dict[str, tuple[str, str | None]]:
    roles: dict[str, tuple[str, str | None]] = {}
    kind_map = {
        "USER_INPUT": "model_input",
        "PARAMETER": "parameter",
        "BUFFER": "buffer",
        "CONSTANT_TENSOR": "constant",
        "CUSTOM_OBJ": "constant",
        "TOKEN": "constant",
    }
    for spec in exported_program.graph_signature.input_specs:
        name = str(getattr(spec.arg, "name", spec.arg))
        kind = str(getattr(spec.kind, "name", spec.kind)).upper()
        roles[name] = (kind_map.get(kind, "constant"), getattr(spec, "target", None))
    return roles


def _output_roles(exported_program: Any) -> dict[str, str]:
    roles: dict[str, str] = {}
    kind_map = {
        "USER_OUTPUT": "model_output",
        "BUFFER_MUTATION": "buffer",
        "GRADIENT_TO_PARAMETER": "gradient",
        "GRADIENT_TO_USER_INPUT": "gradient",
        "LOSS_OUTPUT": "model_output",
    }
    for spec in exported_program.graph_signature.output_specs:
        name = str(getattr(spec.arg, "name", spec.arg))
        kind = str(getattr(spec.kind, "name", spec.kind)).upper()
        roles[name] = kind_map.get(kind, "model_output")
    return roles


def _node_outputs(node: torch.fx.Node) -> tuple[Any, ...]:
    return flatten_tensor_values(node.meta.get("val"))


def _selected_source_outputs(source: torch.fx.Node, consumer: torch.fx.Node) -> tuple[tuple[int, Any], ...]:
    outputs = _node_outputs(source)
    if consumer.target is operator.getitem and consumer.args and consumer.args[0] is source:
        index = consumer.args[1] if len(consumer.args) > 1 else 0
        if isinstance(index, int) and 0 <= index < len(outputs):
            return ((index, outputs[index]),)
    return tuple(enumerate(outputs))


def _input_occurrences(node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
    return (*flatten_nodes(node.args), *flatten_nodes(node.kwargs))


def _is_tensor_operation(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and bool(_node_outputs(node))


def _operation_histograms(
    exported_program: Any,
    registry: OperationRegistry,
) -> tuple[dict[str, int], dict[str, int]]:
    raw_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    for node in exported_program.graph_module.graph.nodes:
        if not _is_tensor_operation(node):
            continue
        raw = raw_target_name(node.target)
        raw_counts[raw] += 1
        canonical_counts[registry.resolve(raw).canonical_id] += 1
    return dict(sorted(raw_counts.items())), dict(sorted(canonical_counts.items()))


def apply_selective_decomposition(
    exported_program: Any,
    registry: OperationRegistry,
) -> tuple[Any, dict[str, Any]]:
    """Functionalize export while decomposing only registry-approved wrappers."""

    pre_raw, pre_canonical = _operation_histograms(exported_program, registry)
    requested_targets: dict[str, Any] = {}
    preserved_targets: set[str] = set()
    for node in exported_program.graph_module.graph.nodes:
        if not _is_tensor_operation(node):
            continue
        raw = raw_target_name(node.target)
        resolved = registry.resolve(raw)
        if resolved.decomposition == "decompose":
            if isinstance(node.target, torch._ops.OpOverload):
                requested_targets[raw] = node.target
        else:
            preserved_targets.add(raw)
    decomposition_table = torch._decomp.get_decompositions(
        tuple(requested_targets.values())
    )
    functional = exported_program.run_decompositions(decomposition_table)
    post_raw, post_canonical = _operation_histograms(functional, registry)
    report = {
        "policy_version": "perfseer_v3_selective_decomposition_v1",
        "functionalization": "torch_export_run_decompositions",
        "requested_targets": sorted(requested_targets),
        "registered_decomposition_targets": sorted(
            raw_target_name(target) for target in decomposition_table
        ),
        "preserved_semantic_targets": sorted(preserved_targets),
        "pre_tensor_nodes": sum(pre_raw.values()),
        "post_tensor_nodes": sum(post_raw.values()),
        "pre_raw_target_histogram": pre_raw,
        "post_raw_target_histogram": post_raw,
        "pre_canonical_histogram": pre_canonical,
        "post_canonical_histogram": post_canonical,
    }
    return functional, report


def _make_edges(
    exported_program: Any,
    operation_nodes: tuple[torch.fx.Node, ...],
    operation_ids: Mapping[str, str],
) -> tuple[tuple[TensorEdgeV3, ...], dict[str, dict[str, int]], dict[str, str]]:
    placeholders = _placeholder_roles(exported_program)
    output_roles = _output_roles(exported_program)
    edges: list[TensorEdgeV3] = []
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"inputs": 0, "parameter_numel": 0, "parameter_bytes": 0, "buffer_numel": 0, "buffer_bytes": 0}
    )
    alias_by_value: dict[tuple[str, int], str] = {}
    next_alias = 0
    view_targets = {
        "prim::getitem",
        "aten::view",
        "aten::transpose.int",
        "aten::permute",
        "aten::expand",
    }
    for source in exported_program.graph_module.graph.nodes:
        outputs = _node_outputs(source)
        inherited_alias: str | None = None
        if source.op == "call_function" and raw_target_name(source.target) in view_targets:
            input_nodes = _input_occurrences(source)
            if input_nodes:
                selected_inputs = _selected_source_outputs(input_nodes[0], source)
                if selected_inputs:
                    source_index = selected_inputs[0][0]
                    inherited_alias = alias_by_value.get((input_nodes[0].name, source_index))
        for output_index, _ in enumerate(outputs):
            key = (source.name, output_index)
            if inherited_alias is not None:
                alias_by_value[key] = inherited_alias
            else:
                alias_by_value[key] = f"a{next_alias}"
                next_alias += 1
    edge_index = 0

    for consumer in operation_nodes:
        consumer_id = operation_ids[consumer.name]
        consumer_input_index = 0
        for source in _input_occurrences(consumer):
            selected = _selected_source_outputs(source, consumer)
            role, target = placeholders.get(source.name, ("activation", None))
            producer_id = operation_ids.get(source.name)
            for producer_output_index, value in selected:
                metadata = tensor_metadata(value)
                alias_group = alias_by_value.get((source.name, producer_output_index))
                if alias_group is None:
                    alias_group = f"a{next_alias}"
                    next_alias += 1
                    alias_by_value[(source.name, producer_output_index)] = alias_group
                is_view = raw_target_name(source.target) in {
                    "prim::getitem",
                    "aten::view",
                    "aten::reshape",
                    "aten::transpose.int",
                    "aten::permute",
                    "aten::expand",
                } if source.op == "call_function" else False
                edges.append(
                    TensorEdgeV3(
                        edge_id=f"e{edge_index}",
                        producer_node_id=producer_id,
                        consumer_node_id=consumer_id,
                        producer_output_index=producer_output_index,
                        consumer_input_index=consumer_input_index,
                        tensor_role=role,
                        shape=metadata.shape,
                        rank=metadata.rank,
                        dtype=metadata.dtype,
                        element_width_bytes=metadata.element_width_bytes,
                        numel=metadata.numel,
                        tensor_bytes=metadata.tensor_bytes,
                        source_name=source.name,
                        stride=metadata.stride,
                        memory_format=metadata.memory_format,
                        alias_group=alias_group,
                        is_view=is_view,
                        is_materialized=not is_view,
                        dynamic_shape_quality="concrete" if metadata.numel is not None else "symbolic",
                    )
                )
                edge_index += 1
                consumer_input_index += 1
                counts[consumer_id]["inputs"] += 1
                if role == "parameter":
                    counts[consumer_id]["parameter_numel"] += metadata.numel or 0
                    counts[consumer_id]["parameter_bytes"] += metadata.tensor_bytes or 0
                if role == "buffer":
                    counts[consumer_id]["buffer_numel"] += metadata.numel or 0
                    counts[consumer_id]["buffer_bytes"] += metadata.tensor_bytes or 0

    output_node = next(node for node in exported_program.graph_module.graph.nodes if node.op == "output")
    output_slot = 0
    for source in flatten_nodes(output_node.args):
        role = output_roles.get(source.name, "model_output")
        producer_id = operation_ids.get(source.name)
        for producer_output_index, value in enumerate(_node_outputs(source)):
            metadata = tensor_metadata(value)
            alias_group = alias_by_value.get((source.name, producer_output_index))
            if alias_group is None:
                alias_group = f"a{next_alias}"
                next_alias += 1
                alias_by_value[(source.name, producer_output_index)] = alias_group
            edges.append(
                TensorEdgeV3(
                    edge_id=f"e{edge_index}",
                    producer_node_id=producer_id,
                    consumer_node_id=None,
                    producer_output_index=producer_output_index,
                    consumer_input_index=output_slot,
                    tensor_role=role,
                    shape=metadata.shape,
                    rank=metadata.rank,
                    dtype=metadata.dtype,
                    element_width_bytes=metadata.element_width_bytes,
                    numel=metadata.numel,
                    tensor_bytes=metadata.tensor_bytes,
                    source_name=source.name,
                    stride=metadata.stride,
                    memory_format=metadata.memory_format,
                    alias_group=alias_group,
                    dynamic_shape_quality="concrete" if metadata.numel is not None else "symbolic",
                )
            )
            edge_index += 1
            output_slot += 1
    placeholder_targets = {name: str(target) for name, (_, target) in placeholders.items() if target is not None}
    return tuple(edges), dict(counts), placeholder_targets


def _depths(nodes: tuple[torch.fx.Node, ...], operation_ids: Mapping[str, str]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node in nodes:
        predecessors = [source for source in _input_occurrences(node) if source.name in operation_ids]
        depths[node.name] = 0 if not predecessors else 1 + max(depths[source.name] for source in predecessors)
    return depths


def exported_program_to_graph_ir(
    exported_program: Any,
    *,
    registry: OperationRegistry,
    capture_mode: str,
    source_fingerprint: str,
    model_fingerprint: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    options: CaptureOptions,
    replay_validated: bool,
    replay_samples: int,
    prior_failures: tuple[CaptureFailureV3, ...] = (),
    capture_metadata: Mapping[str, Any] | None = None,
) -> GraphIRV3:
    feature_schema = build_feature_schema(registry)
    fx_nodes = tuple(node for node in exported_program.graph_module.graph.nodes if _is_tensor_operation(node))
    operation_ids = {node.name: f"n{index}" for index, node in enumerate(fx_nodes)}
    tensor_edges, counts, placeholder_targets = _make_edges(exported_program, fx_nodes, operation_ids)
    depths = _depths(fx_nodes, operation_ids)
    fan_in: Counter[str] = Counter(
        edge.consumer_node_id for edge in tensor_edges if edge.producer_node_id is not None and edge.consumer_node_id
    )
    fan_out: Counter[str] = Counter(
        edge.producer_node_id for edge in tensor_edges if edge.producer_node_id and edge.consumer_node_id is not None
    )
    nodes: list[OperationNodeV3] = []
    unknown_count = 0
    custom_count = 0
    for index, node in enumerate(fx_nodes):
        raw = raw_target_name(node.target)
        resolved = registry.resolve(raw)
        unknown_count += int(not resolved.is_known)
        custom_count += int(resolved.is_custom)
        outputs = _node_outputs(node)
        output_dtype = (
            tensor_metadata(outputs[0]).dtype
            if outputs
            else "unknown"
        )
        stack = source_module_stack(node)
        node_id = operation_ids[node.name]
        resolved_flags = set(resolved.flags)
        if resolved.is_custom:
            resolved_flags.add("custom")
        cost = estimate_fx_node(
            node,
            raw_target=raw,
            cost_formula=resolved.cost_formula,
        )
        nodes.append(
            OperationNodeV3(
                node_id=node_id,
                raw_target=raw,
                canonical_op_id=resolved.canonical_id,
                family_id=resolved.family_id,
                family=resolved.family,
                phase="forward",
                exact_op_id=resolved.exact_id,
                op_hash_bucket=resolved.hash_bucket,
                accumulation_dtype=infer_accumulation_dtype(
                    resolved.family,
                    output_dtype,
                ),
                source_module_path=stack[-1] if stack else None,
                source_module_stack=stack,
                flags={name: True for name in sorted(resolved_flags)},
                normalized_args=normalized_node_arguments(node),
                input_tensor_count=counts.get(node_id, {}).get("inputs", 0),
                output_tensor_count=len(outputs),
                input_numel=cost.input_numel,
                output_numel=cost.output_numel,
                input_bytes=cost.input_bytes,
                output_bytes=cost.output_bytes,
                parameter_numel=counts.get(node_id, {}).get("parameter_numel", 0),
                parameter_bytes=counts.get(node_id, {}).get("parameter_bytes", 0),
                buffer_numel=counts.get(node_id, {}).get("buffer_numel", 0),
                buffer_bytes=counts.get(node_id, {}).get("buffer_bytes", 0),
                flops=cost.flops,
                macs=cost.macs,
                bytes_read=cost.bytes_read,
                bytes_written=cost.bytes_written,
                estimated_workspace_bytes=cost.estimated_workspace_bytes,
                arithmetic_intensity_flops_per_byte=cost.arithmetic_intensity_flops_per_byte,
                topological_index=index,
                depth=depths[node.name],
                fan_in=fan_in[node_id],
                fan_out=fan_out[node_id],
            )
        )
    activation_bytes = sum(
        edge.tensor_bytes or 0
        for edge in tensor_edges
        if edge.tensor_role in {"activation", "model_input", "model_output"}
    )
    total_parameter_numel = sum(node.parameter_numel for node in nodes)
    total_parameter_bytes = sum(node.parameter_bytes for node in nodes)
    total_buffer_bytes = sum(node.buffer_bytes for node in nodes)
    cost_proxy_total = sum(node.output_bytes for node in nodes)
    unknown_cost_proxy = sum(
        node.output_bytes
        for node in nodes
        if node.flops.method == "unknown"
    )
    unknown_output_bytes = sum(
        node.output_bytes
        for node in nodes
        if node.canonical_op_id == "UNK"
    )
    denominator = max(1, len(nodes))
    coverage = CoverageQuality(
        capture_quality="strict" if capture_mode == "strict" else "non_strict_validated",
        backward_capture_quality="estimated",
        tensor_nodes_seen=len(fx_nodes),
        tensor_nodes_encoded=len(nodes),
        unknown_operations=unknown_count,
        custom_operations=custom_count,
        replay_samples=replay_samples,
        replay_validated=replay_validated,
    )
    global_features = GraphGlobalFeatures(
        operation_nodes=len(nodes),
        tensor_edges=len(tensor_edges),
        total_flops=sum(node.flops.value for node in nodes),
        total_macs=sum(node.macs.value for node in nodes),
        total_parameter_numel=total_parameter_numel,
        total_parameter_bytes=total_parameter_bytes,
        total_buffer_bytes=total_buffer_bytes,
        total_activation_bytes=activation_bytes,
        critical_path_length=1 + max(depths.values(), default=-1),
        unknown_operation_fraction=unknown_count / denominator,
        unknown_cost_fraction=(
            unknown_cost_proxy / cost_proxy_total
            if cost_proxy_total > 0
            else float(bool(unknown_cost_proxy))
        ),
        unknown_byte_fraction=(
            unknown_output_bytes / cost_proxy_total
            if cost_proxy_total > 0
            else float(bool(unknown_output_bytes))
        ),
    )
    graph = GraphIRV3.create(
        operator_registry_sha256=registry.sha256,
        feature_schema_sha256=feature_schema["feature_schema_sha256"],
        capture_backend="torch_export",
        capture_mode=capture_mode,
        pytorch_version=torch.__version__,
        source_fingerprint=source_fingerprint,
        model_fingerprint=model_fingerprint,
        input_signature=input_signature(args, kwargs),
        dynamic_constraints=normalize_constraints(exported_program),
        training_mode=options.training_mode,
        precision=options.precision,
        optimizer_config=options.optimizer_config or {},
        training_config=options.training_config or {},
        nodes=tuple(nodes),
        tensor_edges=tensor_edges,
        global_features=global_features,
        coverage=coverage,
        warnings=tuple(
            f"{failure.mode}:{failure.stage}:{failure.exception_type}" for failure in prior_failures
        ),
        failures=tuple(failure.to_dict() for failure in prior_failures),
        metadata={
            "placeholder_targets": placeholder_targets,
            "target_hardware_id": canonical_hardware_id(options.target_hardware_id),
            "hardware_features": dict(options.hardware_features or {}),
            **dict(capture_metadata or {}),
        },
    )
    return apply_liveness(graph)


def _default_replay_factory(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> ReplayInputFactory:
    def factory(index: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator().manual_seed(1729 + index)
        return (
            randomized_like(args, generator=generator),
            randomized_like(dict(kwargs), generator=generator),
        )

    return factory


def validate_non_strict_replay(
    model: nn.Module,
    exported_program: Any,
    *,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    options: CaptureOptions,
    replay_input_factory: ReplayInputFactory | None,
) -> None:
    factory = replay_input_factory or _default_replay_factory(args, kwargs)
    exported_module = exported_program.module()
    original_cpu_rng = torch.random.get_rng_state()
    original_cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    model_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    exported_state = {
        name: value.detach().clone()
        for name, value in exported_module.state_dict().items()
    }

    def restore_rng(
        cpu_state: torch.Tensor,
        cuda_state: list[torch.Tensor] | None,
    ) -> None:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    try:
        for sample_index in range(options.replay_samples):
            sample_args, sample_kwargs = factory(sample_index)
            eager_args, eager_kwargs = clone_inputs(sample_args), clone_inputs(sample_kwargs)
            export_args, export_kwargs = clone_inputs(sample_args), clone_inputs(sample_kwargs)
            sample_cpu_rng = torch.random.get_rng_state()
            sample_cuda_rng = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            model.load_state_dict(model_state)
            restore_rng(sample_cpu_rng, sample_cuda_rng)
            with torch.no_grad():
                eager = model(*eager_args, **eager_kwargs)
            exported_module.load_state_dict(exported_state)
            restore_rng(sample_cpu_rng, sample_cuda_rng)
            with torch.no_grad():
                replayed = exported_module(*export_args, **export_kwargs)
            compare_output_pytrees(
                eager,
                replayed,
                rtol=options.replay_rtol,
                atol=options.replay_atol,
                path=f"sample[{sample_index}]",
            )
    finally:
        model.load_state_dict(model_state)
        exported_module.load_state_dict(exported_state)
        restore_rng(original_cpu_rng, original_cuda_rng)


def capture_export(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: Any | None = None,
    registry: OperationRegistry | None = None,
    options: CaptureOptions | None = None,
    source_fingerprint: str | None = None,
    replay_input_factory: ReplayInputFactory | None = None,
) -> CaptureResult:
    registry = registry or OperationRegistry.load()
    options = options or CaptureOptions()
    kwargs = dict(kwargs or {})
    failures: list[CaptureFailureV3] = []
    if options.training_mode:
        model.train()
    else:
        model.eval()
    model_fingerprint = _model_fingerprint(model)
    if source_fingerprint is None:
        source_fingerprint = hashlib.sha256(
            f"{model.__class__.__module__}.{model.__class__.__qualname__}".encode("utf-8")
        ).hexdigest()

    def make_result(
        graph: GraphIRV3 | None,
        exported_program: Any | None,
        result_failures: tuple[CaptureFailureV3, ...],
    ) -> CaptureResult:
        return CaptureResult(
            graph,
            exported_program,
            result_failures,
            model_object_id=id(model),
            callable_qualname=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        )

    try:
        exported = torch.export.export(
            model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        pre_raw, pre_canonical = _operation_histograms(exported, registry)
        semantic_summary = {
            "tensor_nodes": sum(pre_raw.values()),
            "raw_target_histogram": pre_raw,
            "canonical_histogram": pre_canonical,
        }
        if options.apply_selective_decomposition:
            exported, decomposition_report = apply_selective_decomposition(
                exported,
                registry,
            )
        else:
            decomposition_report = {
                "policy_version": "perfseer_v3_selective_decomposition_v1",
                "functionalization": "disabled_by_capture_option",
                "requested_targets": [],
            }
        graph = exported_program_to_graph_ir(
            exported,
            registry=registry,
            capture_mode="strict",
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            args=args,
            kwargs=kwargs,
            options=options,
            replay_validated=False,
            replay_samples=0,
            capture_metadata={
                "semantic_summary_pre_decomposition": semantic_summary,
                "selective_decomposition": decomposition_report,
            },
        )
        return make_result(graph, exported, ())
    except Exception as exc:
        failures.append(
            CaptureFailureV3.from_exception(
                exc,
                backend="torch_export",
                mode="strict",
                stage="capture_or_convert",
                retryable=options.allow_non_strict,
            )
        )
    if not options.allow_non_strict:
        return make_result(None, None, tuple(failures))
    try:
        exported = torch.export.export(
            model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    except Exception as exc:
        failures.append(
            CaptureFailureV3.from_exception(
                exc,
                backend="torch_export",
                mode="non_strict",
                stage="capture",
            )
        )
        return make_result(None, None, tuple(failures))
    try:
        validate_non_strict_replay(
            model,
            exported,
            args=args,
            kwargs=kwargs,
            options=options,
            replay_input_factory=replay_input_factory,
        )
    except Exception as exc:
        failures.append(
            CaptureFailureV3.from_exception(
                exc,
                backend="torch_export",
                mode="non_strict",
                stage="replay_validation",
            )
        )
        return make_result(None, exported, tuple(failures))
    try:
        pre_raw, pre_canonical = _operation_histograms(exported, registry)
        semantic_summary = {
            "tensor_nodes": sum(pre_raw.values()),
            "raw_target_histogram": pre_raw,
            "canonical_histogram": pre_canonical,
        }
        if options.apply_selective_decomposition:
            exported, decomposition_report = apply_selective_decomposition(
                exported,
                registry,
            )
        else:
            decomposition_report = {
                "policy_version": "perfseer_v3_selective_decomposition_v1",
                "functionalization": "disabled_by_capture_option",
                "requested_targets": [],
            }
        graph = exported_program_to_graph_ir(
            exported,
            registry=registry,
            capture_mode="non_strict",
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            args=args,
            kwargs=kwargs,
            options=options,
            replay_validated=True,
            replay_samples=options.replay_samples,
            prior_failures=tuple(failures),
            capture_metadata={
                "semantic_summary_pre_decomposition": semantic_summary,
                "selective_decomposition": decomposition_report,
            },
        )
        return make_result(graph, exported, tuple(failures))
    except Exception as exc:
        failures.append(
            CaptureFailureV3.from_exception(
                exc,
                backend="torch_export",
                mode="non_strict",
                stage="convert",
            )
        )
        return make_result(None, exported, tuple(failures))


__all__ = [
    "CaptureOptions",
    "CaptureResult",
    "ReplayInputFactory",
    "apply_selective_decomposition",
    "capture_export",
    "exported_program_to_graph_ir",
    "infer_accumulation_dtype",
    "validate_non_strict_replay",
]
