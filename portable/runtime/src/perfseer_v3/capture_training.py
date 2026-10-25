"""Version-pinned AOT Autograd capture with an explicit analytical fallback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from .canonicalize import normalized_node_arguments, raw_target_name, source_module_stack
from .capture_export import (
    CaptureOptions,
    capture_export,
    infer_accumulation_dtype,
)
from .cost_registry_v3 import estimate_fx_node
from .diagnostics import CaptureFailureV3
from .graph_ir_v3 import (
    CoverageQuality,
    Estimate,
    GraphGlobalFeatures,
    GraphIRV3,
    OperationNodeV3,
    TensorEdgeV3,
)
from .liveness_v3 import apply_liveness
from .op_registry import OperationRegistry
from .tensor_metadata import flatten_nodes, flatten_tensor_values, tensor_metadata
from .training_semantics import (
    canonical_optimizer_name,
    optimizer_flops_per_parameter,
    optimizer_state_multiplier,
)


@dataclass(frozen=True)
class TrainingCaptureResult:
    graph: GraphIRV3 | None
    forward_exported_program: Any | None
    backward_backend: str
    failures: tuple[CaptureFailureV3, ...]
    model_object_id: int | None = None
    callable_qualname: str | None = None

    @property
    def success(self) -> bool:
        return self.graph is not None

    @property
    def exported_program(self) -> Any | None:
        return self.forward_exported_program


class _LossWrapper(nn.Module):
    def __init__(self, model: nn.Module, loss_fn: Callable[[Any, Any], torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *flat_args: Any, **kwargs: Any) -> tuple[torch.Tensor]:
        model_args = flat_args[:-1]
        target = flat_args[-1]
        output = self.model(*model_args, **kwargs)
        return (self.loss_fn(output, target),)


def _capture_aot_joint(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    target: Any,
) -> tuple[torch.fx.GraphModule, Any]:
    # This is the only module that imports the evolving internal AOT API.
    from torch._functorch.aot_autograd import aot_export_module

    wrapper = _LossWrapper(model, loss_fn)
    graph_module, signature = aot_export_module(
        wrapper,
        (*args, target),
        kwargs=dict(kwargs),
        trace_joint=True,
        output_loss_index=0,
        pre_dispatch=False,
        dynamic_shapes=False,
    )
    if signature.backward_signature is None:
        raise RuntimeError("AOT Autograd returned no backward signature")
    return graph_module, signature


def _phase_nodes(
    graph_module: torch.fx.GraphModule,
    signature: Any,
) -> tuple[tuple[torch.fx.Node, str], ...]:
    tensor_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and flatten_tensor_values(node.meta.get("val"))
    ]
    loss_name = str(signature.backward_signature.loss_output)
    loss_position = next(
        (index for index, node in enumerate(tensor_nodes) if node.name == loss_name),
        None,
    )
    if loss_position is None:
        raise RuntimeError(f"AOT loss output {loss_name!r} was not found in the joint graph")
    selected = tensor_nodes[loss_position:]
    return tuple(
        (node, "loss" if index == 0 else "backward")
        for index, node in enumerate(selected)
    )


def _append_aot_phases(
    graph: GraphIRV3,
    graph_module: torch.fx.GraphModule,
    signature: Any,
    registry: OperationRegistry,
) -> GraphIRV3:
    phase_nodes = _phase_nodes(graph_module, signature)
    start_node = len(graph.nodes)
    op_ids = {node.name: f"n{start_node + index}" for index, (node, _) in enumerate(phase_nodes)}
    phase_by_name = {node.name: phase for node, phase in phase_nodes}
    parameter_inputs = dict(getattr(signature, "inputs_to_parameters", {}))
    buffer_inputs = dict(getattr(signature, "inputs_to_buffers", {}))
    edge_start = len(graph.tensor_edges)
    edges = list(graph.tensor_edges)
    counts: dict[str, dict[str, int]] = {}
    nodes: list[OperationNodeV3] = []
    edge_counter = edge_start

    for topo_offset, (node, phase) in enumerate(phase_nodes):
        node_id = op_ids[node.name]
        input_count = input_numel = input_bytes = parameter_numel = parameter_bytes = 0
        consumer_slot = 0
        for source in (*flatten_nodes(node.args), *flatten_nodes(node.kwargs)):
            values = flatten_tensor_values(source.meta.get("val"))
            for output_index, value in enumerate(values):
                metadata = tensor_metadata(value)
                producer = op_ids.get(source.name)
                if source.name in parameter_inputs:
                    role = "parameter"
                elif source.name in buffer_inputs:
                    role = "buffer"
                else:
                    role = "activation"
                edges.append(
                    TensorEdgeV3(
                        edge_id=f"e{edge_counter}",
                        producer_node_id=producer,
                        consumer_node_id=node_id,
                        producer_output_index=output_index,
                        consumer_input_index=consumer_slot,
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
                        alias_group=f"aot:{source.name}:{output_index}",
                        dynamic_shape_quality="concrete" if metadata.numel is not None else "symbolic",
                    )
                )
                edge_counter += 1
                consumer_slot += 1
                input_count += 1
                input_numel += metadata.numel or 0
                input_bytes += metadata.tensor_bytes or 0
                if role == "parameter":
                    parameter_numel += metadata.numel or 0
                    parameter_bytes += metadata.tensor_bytes or 0

        raw = raw_target_name(node.target)
        resolved = registry.resolve(raw)
        family = resolved.family
        family_id = resolved.family_id
        if phase == "backward" and not resolved.is_known:
            family = "training"
            family_id = registry.family_to_id["training"]
        cost = estimate_fx_node(node, raw_target=raw, cost_formula=resolved.cost_formula)
        output_values = flatten_tensor_values(node.meta.get("val"))
        output_dtype = (
            tensor_metadata(output_values[0]).dtype
            if output_values
            else "unknown"
        )
        stack = source_module_stack(node)
        flags = set(resolved.flags)
        if phase == "backward":
            flags.add("backward")
        nodes.append(
            OperationNodeV3(
                node_id=node_id,
                raw_target=raw,
                canonical_op_id=resolved.canonical_id,
                family_id=family_id,
                family=family,
                phase=phase,
                exact_op_id=resolved.exact_id,
                op_hash_bucket=resolved.hash_bucket,
                accumulation_dtype=infer_accumulation_dtype(
                    family,
                    output_dtype,
                ),
                source_module_path=stack[-1] if stack else None,
                source_module_stack=stack,
                flags={name: True for name in sorted(flags)},
                normalized_args=normalized_node_arguments(node),
                input_tensor_count=input_count,
                output_tensor_count=len(flatten_tensor_values(node.meta.get("val"))),
                input_numel=input_numel,
                output_numel=cost.output_numel,
                input_bytes=input_bytes,
                output_bytes=cost.output_bytes,
                parameter_numel=parameter_numel,
                parameter_bytes=parameter_bytes,
                flops=cost.flops,
                macs=cost.macs,
                bytes_read=cost.bytes_read,
                bytes_written=cost.bytes_written,
                estimated_workspace_bytes=cost.estimated_workspace_bytes,
                arithmetic_intensity_flops_per_byte=cost.arithmetic_intensity_flops_per_byte,
                topological_index=start_node + topo_offset,
                depth=graph.global_features.critical_path_length + topo_offset,
            )
        )

    gradients = dict(signature.backward_signature.gradients_to_parameters)
    for output_name, parameter_name in sorted(gradients.items()):
        producer_id = op_ids.get(output_name)
        source = next((node for node, _ in phase_nodes if node.name == output_name), None)
        if producer_id is None or source is None:
            continue
        for output_index, value in enumerate(flatten_tensor_values(source.meta.get("val"))):
            metadata = tensor_metadata(value)
            edges.append(
                TensorEdgeV3(
                    edge_id=f"e{edge_counter}",
                    producer_node_id=producer_id,
                    consumer_node_id=None,
                    producer_output_index=output_index,
                    consumer_input_index=output_index,
                    tensor_role="gradient",
                    shape=metadata.shape,
                    rank=metadata.rank,
                    dtype=metadata.dtype,
                    element_width_bytes=metadata.element_width_bytes,
                    numel=metadata.numel,
                    tensor_bytes=metadata.tensor_bytes,
                    source_name=str(parameter_name),
                    stride=metadata.stride,
                    memory_format=metadata.memory_format,
                    alias_group=f"aot-gradient:{parameter_name}",
                )
            )
            edge_counter += 1

    combined_nodes = (*graph.nodes, *nodes)
    unknown_added = sum(node.canonical_op_id == "UNK" for node in nodes)
    custom_added = sum(node.flags.get("custom", False) for node in nodes)
    coverage = replace(
        graph.coverage,
        backward_capture_quality="strict",
        tensor_nodes_seen=graph.coverage.tensor_nodes_seen + len(nodes),
        tensor_nodes_encoded=graph.coverage.tensor_nodes_encoded + len(nodes),
        unknown_operations=graph.coverage.unknown_operations + unknown_added,
        custom_operations=graph.coverage.custom_operations + custom_added,
    )
    globals_updated = replace(
        graph.global_features,
        operation_nodes=len(combined_nodes),
        tensor_edges=len(edges),
        total_flops=graph.global_features.total_flops + sum(node.flops.value for node in nodes),
        total_macs=graph.global_features.total_macs + sum(node.macs.value for node in nodes),
        critical_path_length=max(
            graph.global_features.critical_path_length,
            max((node.depth + 1 for node in nodes), default=0),
        ),
        unknown_operation_fraction=(
            coverage.unknown_operations / max(1, len(combined_nodes))
        ),
    )
    updated = replace(
        graph,
        nodes=tuple(combined_nodes),
        tensor_edges=tuple(edges),
        global_features=globals_updated,
        coverage=coverage,
        metadata={
            **graph.metadata,
            "backward_capture": {
                "backend": "torch._functorch.aot_autograd.aot_export_module",
                "torch_version": torch.__version__,
                "quality": "strict",
            },
        },
    )
    updated.validate()
    return updated


def _precision_width(precision: str) -> tuple[str, int]:
    key = precision.lower()
    if "16" in key or "bf16" in key:
        return ("bfloat16" if "bf" in key else "float16", 2)
    if "64" in key:
        return "float64", 8
    return "float32", 4


def _append_optimizer_summary(
    graph: GraphIRV3,
    registry: OperationRegistry,
    optimizer_name: str,
    optimizer_config: Mapping[str, Any],
) -> GraphIRV3:
    name = canonical_optimizer_name(optimizer_name)
    raw = f"perfseer::optimizer.{name}"
    resolved = registry.resolve(raw)
    parameter_numel = graph.global_features.total_parameter_numel
    parameter_bytes = graph.global_features.total_parameter_bytes
    state_bytes = int(
        round(parameter_bytes * optimizer_state_multiplier(name, optimizer_config))
    )
    gradient_bytes = parameter_bytes
    flops_per_parameter = optimizer_flops_per_parameter(name, optimizer_config)
    flops_value = parameter_numel * flops_per_parameter
    node_id = f"n{len(graph.nodes)}"
    total_traffic = parameter_bytes + gradient_bytes + state_bytes
    foreach = bool(optimizer_config.get("foreach", False))
    fused = bool(optimizer_config.get("fused", False))
    workspace_bytes = parameter_bytes if foreach else 0
    optimizer_node = OperationNodeV3(
        node_id=node_id,
        raw_target=raw,
        canonical_op_id=resolved.canonical_id,
        family_id=resolved.family_id,
        family=resolved.family,
        phase="optimizer",
        exact_op_id=resolved.exact_id,
        op_hash_bucket=resolved.hash_bucket,
        accumulation_dtype=infer_accumulation_dtype(
            resolved.family,
            graph.precision,
        ),
        flags={
            **{flag: True for flag in resolved.flags},
            "estimated": True,
            "foreach": foreach,
            "fused": fused,
        },
        normalized_args={"optimizer": name, **dict(optimizer_config)},
        input_tensor_count=1,
        output_tensor_count=1 if state_bytes else 0,
        input_numel=parameter_numel,
        output_numel=state_bytes // max(1, _precision_width(graph.precision)[1]),
        input_bytes=gradient_bytes,
        output_bytes=state_bytes,
        flops=Estimate(flops_value, "shape_formula", 0.7),
        macs=Estimate(),
        bytes_read=Estimate(total_traffic, "shape_formula", 0.8),
        bytes_written=Estimate(parameter_bytes + state_bytes, "shape_formula", 0.8),
        estimated_workspace_bytes=Estimate(
            workspace_bytes,
            "shape_formula" if workspace_bytes else "exact_formula",
            0.7 if workspace_bytes else 1.0,
        ),
        arithmetic_intensity_flops_per_byte=flops_value / max(1, total_traffic + parameter_bytes),
        optimizer_state_bytes=state_bytes,
        topological_index=len(graph.nodes),
        depth=graph.global_features.critical_path_length,
    )
    edges = list(graph.tensor_edges)
    edge_index = len(edges)
    gradient_source = next(
        (
            edge
            for edge in reversed(graph.tensor_edges)
            if edge.tensor_role == "gradient" and edge.producer_node_id is not None
        ),
        None,
    )
    producer_id = gradient_source.producer_node_id if gradient_source else (
        graph.nodes[-1].node_id if graph.nodes else None
    )
    dtype, width = _precision_width(graph.precision)
    edges.append(
        TensorEdgeV3(
            edge_id=f"e{edge_index}",
            producer_node_id=producer_id,
            consumer_node_id=node_id,
            producer_output_index=0,
            consumer_input_index=0,
            tensor_role="gradient",
            shape=(parameter_numel,),
            rank=1,
            dtype=dtype,
            element_width_bytes=width,
            numel=parameter_numel,
            tensor_bytes=gradient_bytes,
            source_name="aggregate_parameter_gradients",
            stride=(1,),
            memory_format="contiguous",
            alias_group="aggregate:gradients",
            dynamic_shape_quality="estimated",
        )
    )
    if state_bytes:
        state_numel = state_bytes // width
        edges.append(
            TensorEdgeV3(
                edge_id=f"e{edge_index + 1}",
                producer_node_id=node_id,
                consumer_node_id=None,
                producer_output_index=0,
                consumer_input_index=0,
                tensor_role="optimizer_state",
                shape=(state_numel,),
                rank=1,
                dtype=dtype,
                element_width_bytes=width,
                numel=state_numel,
                tensor_bytes=state_bytes,
                source_name=f"{name}_state",
                stride=(1,),
                memory_format="contiguous",
                alias_group="aggregate:optimizer_state",
                dynamic_shape_quality="estimated",
            )
        )
    nodes = (*graph.nodes, optimizer_node)
    updated = replace(
        graph,
        optimizer_config={"name": name, **dict(optimizer_config)},
        nodes=nodes,
        tensor_edges=tuple(edges),
        global_features=replace(
            graph.global_features,
            operation_nodes=len(nodes),
            tensor_edges=len(edges),
            total_flops=graph.global_features.total_flops + flops_value,
            total_optimizer_state_bytes=state_bytes,
            critical_path_length=max(
                graph.global_features.critical_path_length,
                optimizer_node.depth + 1,
            ),
        ),
    )
    updated.validate()
    return updated


def _append_analytical_fallback(
    graph: GraphIRV3,
    registry: OperationRegistry,
    optimizer_name: str,
    optimizer_config: Mapping[str, Any],
    failures: tuple[CaptureFailureV3, ...],
) -> GraphIRV3:
    model_output = next(
        (edge for edge in graph.tensor_edges if edge.tensor_role == "model_output"),
        None,
    )
    output_bytes = model_output.tensor_bytes if model_output and model_output.tensor_bytes else 0
    output_numel = model_output.numel if model_output and model_output.numel else 1
    loss_rule = registry.resolve("perfseer::loss.generic")
    backward_rule = registry.resolve("perfseer::analytical_backward")
    loss_id = f"n{len(graph.nodes)}"
    backward_id = f"n{len(graph.nodes) + 1}"
    saved_bytes = graph.global_features.total_activation_bytes
    loss_node = OperationNodeV3(
        node_id=loss_id,
        raw_target=loss_rule.raw_target,
        canonical_op_id=loss_rule.canonical_id,
        family_id=loss_rule.family_id,
        family=loss_rule.family,
        phase="loss",
        exact_op_id=loss_rule.exact_id,
        op_hash_bucket=loss_rule.hash_bucket,
        accumulation_dtype="float32",
        flags={"estimated": True, "reduction": True},
        input_tensor_count=1,
        output_tensor_count=1,
        input_numel=output_numel,
        output_numel=1,
        input_bytes=output_bytes,
        output_bytes=4,
        flops=Estimate(output_numel, "shape_formula", 0.5),
        bytes_read=Estimate(output_bytes, "shape_formula", 0.7),
        bytes_written=Estimate(4, "shape_formula", 0.7),
        topological_index=len(graph.nodes),
        depth=graph.global_features.critical_path_length,
    )
    backward_node = OperationNodeV3(
        node_id=backward_id,
        raw_target=backward_rule.raw_target,
        canonical_op_id=backward_rule.canonical_id,
        family_id=backward_rule.family_id,
        family=backward_rule.family,
        phase="backward",
        exact_op_id=backward_rule.exact_id,
        op_hash_bucket=backward_rule.hash_bucket,
        accumulation_dtype=infer_accumulation_dtype(
            backward_rule.family,
            graph.precision,
        ),
        flags={"estimated": True},
        input_tensor_count=1,
        output_tensor_count=1,
        input_numel=1,
        output_numel=graph.global_features.total_parameter_numel,
        input_bytes=4,
        output_bytes=graph.global_features.total_parameter_bytes,
        flops=Estimate(
            max(graph.global_features.total_flops * 2.0, 0.0),
            "shape_formula",
            0.35,
        ),
        bytes_read=Estimate(
            saved_bytes + graph.global_features.total_parameter_bytes,
            "shape_formula",
            0.4,
        ),
        bytes_written=Estimate(
            graph.global_features.total_parameter_bytes,
            "shape_formula",
            0.5,
        ),
        saved_for_backward_bytes=saved_bytes,
        topological_index=len(graph.nodes) + 1,
        depth=graph.global_features.critical_path_length + 1,
    )
    edges = list(graph.tensor_edges)
    edge_index = len(edges)
    if model_output is not None:
        edges.append(
            replace(
                model_output,
                edge_id=f"e{edge_index}",
                consumer_node_id=loss_id,
                consumer_input_index=0,
                tensor_role="activation",
            )
        )
        edge_index += 1
    edges.append(
        TensorEdgeV3(
            edge_id=f"e{edge_index}",
            producer_node_id=loss_id,
            consumer_node_id=backward_id,
            producer_output_index=0,
            consumer_input_index=0,
            tensor_role="activation",
            shape=(),
            rank=0,
            dtype="float32",
            element_width_bytes=4,
            numel=1,
            tensor_bytes=4,
            source_name="loss",
            stride=(),
            memory_format="contiguous",
            alias_group="analytical:loss",
            dynamic_shape_quality="estimated",
        )
    )
    nodes = (*graph.nodes, loss_node, backward_node)
    coverage = replace(graph.coverage, backward_capture_quality="estimated")
    updated = replace(
        graph,
        nodes=nodes,
        tensor_edges=tuple(edges),
        global_features=replace(
            graph.global_features,
            operation_nodes=len(nodes),
            tensor_edges=len(edges),
            total_flops=(
                graph.global_features.total_flops
                + loss_node.flops.value
                + backward_node.flops.value
            ),
            total_saved_for_backward_bytes=saved_bytes,
            critical_path_length=backward_node.depth + 1,
        ),
        coverage=coverage,
        warnings=(
            *graph.warnings,
            "backward_capture:estimated_analytical_fallback",
        ),
        failures=(*graph.failures, *(failure.to_dict() for failure in failures)),
        metadata={
            **graph.metadata,
            "backward_capture": {
                "backend": "analytical_fallback",
                "quality": "estimated",
            },
        },
    )
    updated.validate()
    return updated


def capture_training_graph(
    model: nn.Module,
    args: tuple[Any, ...],
    *,
    target: Any,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    optimizer_name: str = "adamw",
    optimizer_config: Mapping[str, Any] | None = None,
    scheduler_name: str = "none",
    scheduler_config: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    target_hardware_id: str = "unknown",
    hardware_features: Mapping[str, Any] | None = None,
    registry: OperationRegistry | None = None,
    allow_analytical_fallback: bool = True,
) -> TrainingCaptureResult:
    registry = registry or OperationRegistry.load()
    kwargs = dict(kwargs or {})
    optimizer_config = dict(optimizer_config or {})
    effective_training_config = dict(training_config or {})
    if "scheduler" not in effective_training_config:
        effective_training_config["scheduler"] = {
            "name": scheduler_name,
            **dict(scheduler_config or {}),
        }
    optimizer_name = optimizer_name.strip().lower()
    if not optimizer_name:
        raise ValueError(
            "optimizer_name cannot be empty; use 'none' for gradient-only capture"
        )
    forward_result = capture_export(
        model,
        args,
        kwargs,
        dynamic_shapes=dynamic_shapes,
        registry=registry,
        options=CaptureOptions(
            training_mode=True,
            precision=str(next(model.parameters(), torch.empty((), dtype=torch.float32)).dtype).removeprefix("torch."),
            optimizer_config={"name": optimizer_name, **optimizer_config},
            training_config=effective_training_config,
            target_hardware_id=target_hardware_id,
            hardware_features=dict(hardware_features or {}),
        ),
    )
    if not forward_result.success:
        return TrainingCaptureResult(
            None,
            forward_result.exported_program,
            "failed",
            forward_result.failures,
            forward_result.model_object_id,
            forward_result.callable_qualname,
        )
    assert forward_result.graph is not None
    failures: list[CaptureFailureV3] = list(forward_result.failures)
    try:
        joint_graph, signature = _capture_aot_joint(model, args, kwargs, loss_fn, target)
        graph = _append_aot_phases(forward_result.graph, joint_graph, signature, registry)
        backend = "aot_autograd_joint"
    except Exception as exc:
        failure = CaptureFailureV3.from_exception(
            exc,
            backend="aot_autograd",
            mode="joint",
            stage="backward_capture",
            retryable=allow_analytical_fallback,
        )
        failures.append(failure)
        if not allow_analytical_fallback:
            return TrainingCaptureResult(
                None,
                forward_result.exported_program,
                "failed",
                tuple(failures),
                forward_result.model_object_id,
                forward_result.callable_qualname,
            )
        graph = _append_analytical_fallback(
            forward_result.graph,
            registry,
            optimizer_name,
            optimizer_config,
            (failure,),
        )
        backend = "analytical_fallback"
    try:
        if optimizer_name != "none":
            graph = _append_optimizer_summary(
                graph,
                registry,
                optimizer_name,
                optimizer_config,
            )
        graph = apply_liveness(graph)
    except Exception as exc:
        failures.append(
            CaptureFailureV3.from_exception(
                exc,
                backend="training_features",
                mode=backend,
                stage="optimizer_or_liveness",
            )
        )
        return TrainingCaptureResult(
            None,
            forward_result.exported_program,
            "failed",
            tuple(failures),
            forward_result.model_object_id,
            forward_result.callable_qualname,
        )
    return TrainingCaptureResult(
        graph,
        forward_result.exported_program,
        backend,
        tuple(failures),
        forward_result.model_object_id,
        forward_result.callable_qualname,
    )


__all__ = ["TrainingCaptureResult", "capture_training_graph"]
