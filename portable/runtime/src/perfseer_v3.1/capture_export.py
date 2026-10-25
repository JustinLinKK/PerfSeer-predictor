"""GraphIR construction retained from v3, using linear-time v3.1 liveness."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import torch

from perfseer_v3.capture_export import (
    CaptureOptions, CaptureFailureV3, OperationRegistry, GraphIRV3,
    OperationNodeV3, GraphGlobalFeatures, CoverageQuality, _is_tensor_operation,
    _make_edges, _depths, _node_outputs, build_feature_schema, raw_target_name,
    tensor_metadata, source_module_stack, estimate_fx_node, infer_accumulation_dtype,
    normalized_node_arguments, input_signature, normalize_constraints, canonical_hardware_id,
)
from .liveness import apply_liveness


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
