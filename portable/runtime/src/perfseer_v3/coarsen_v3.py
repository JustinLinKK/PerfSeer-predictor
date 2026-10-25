"""Deterministic conservative graph coarsening for deployment."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from .graph_ir_v3 import Estimate, GraphIRV3, OperationNodeV3, TensorEdgeV3
from .liveness_v3 import apply_liveness
from .op_registry import OperationRegistry


COARSENING_POLICY_ID = "perfseer_v3_safe_sese_v1"
COARSENING_POLICY_SHA256 = hashlib.sha256(
    (
        COARSENING_POLICY_ID
        + "|view_only_or_same_shape_pointwise|no_branch_join_mutation_dtype_materialization"
    ).encode("utf-8")
).hexdigest()


_NEVER_COLLAPSE_FAMILIES = {
    "convolution",
    "dense_matrix",
    "attention",
    "normalization",
    "reduction",
    "loss_probability",
    "index_scatter",
    "sparse_graph",
    "random_regularization",
    "optimizer",
    "training",
}
_POINTWISE_FAMILIES = {"activation_unary", "elementwise"}


def _operation_edges(graph: GraphIRV3) -> tuple[tuple[str, str], ...]:
    return tuple(
        (edge.producer_node_id, edge.consumer_node_id)
        for edge in graph.tensor_edges
        if edge.producer_node_id is not None and edge.consumer_node_id is not None
    )


def _is_candidate(node: OperationNodeV3) -> bool:
    if node.family in _NEVER_COLLAPSE_FAMILIES:
        return False
    if any(
        node.flags.get(flag, False)
        for flag in ("in_place", "random", "sparse", "quantized", "custom")
    ):
        return False
    if node.flags.get("view_only", False):
        return True
    return (
        node.family in _POINTWISE_FAMILIES
        and node.input_numel == node.output_numel
        and node.input_numel > 0
    )


def _find_regions(graph: GraphIRV3) -> tuple[tuple[str, ...], ...]:
    adjacency_out: dict[str, set[str]] = defaultdict(set)
    adjacency_in: dict[str, set[str]] = defaultdict(set)
    for source, destination in _operation_edges(graph):
        adjacency_out[source].add(destination)
        adjacency_in[destination].add(source)
    by_id = {node.node_id: node for node in graph.nodes}
    assigned: set[str] = set()
    regions: list[tuple[str, ...]] = []
    for node in graph.nodes:
        if node.node_id in assigned or not _is_candidate(node):
            continue
        chain = [node.node_id]
        current = node
        while True:
            successors = adjacency_out.get(current.node_id, set())
            if len(successors) != 1:
                break
            successor_id = next(iter(successors))
            if len(adjacency_in.get(successor_id, set())) != 1:
                break
            successor = by_id[successor_id]
            if (
                successor.node_id in assigned
                or not _is_candidate(successor)
                or successor.phase != current.phase
            ):
                break
            chain.append(successor_id)
            current = successor
        if len(chain) >= 2:
            regions.append(tuple(chain))
            assigned.update(chain)
    return tuple(regions)


def _sum_estimates(values: Iterable[Estimate]) -> Estimate:
    items = tuple(values)
    if not items:
        return Estimate()
    known = [item for item in items if item.method != "unknown"]
    if not known:
        return Estimate()
    method = known[0].method if all(item.method == known[0].method for item in known) else "shape_formula"
    confidence = min(item.confidence for item in known)
    return Estimate(sum(item.value for item in items), method, confidence)


def _collapse_region(
    members: tuple[OperationNodeV3, ...],
    *,
    region_id: str,
    registry: OperationRegistry,
) -> OperationNodeV3:
    first, last = members[0], members[-1]
    exact_histogram = Counter(node.canonical_op_id for node in members)
    family_histogram = Counter(node.family for node in members)
    flags: dict[str, bool] = {}
    for node in members:
        for key, value in node.flags.items():
            flags[key] = flags.get(key, False) or bool(value)
    flops = _sum_estimates(node.flops for node in members)
    bytes_read = _sum_estimates(node.bytes_read for node in members)
    bytes_written = _sum_estimates(node.bytes_written for node in members)
    traffic = bytes_read.value + bytes_written.value
    accumulation_dtypes = {
        node.accumulation_dtype
        for node in members
    }
    return OperationNodeV3(
        node_id=region_id,
        raw_target="perfseer::coarsened_region",
        canonical_op_id="perfseer.coarsened_region",
        family_id=first.family_id if len(family_histogram) == 1 else 0,
        family=first.family if len(family_histogram) == 1 else "unknown_or_custom",
        phase=first.phase,
        exact_op_id=0,
        op_hash_bucket=registry.stable_hash_bucket(
            "|".join(node.canonical_op_id for node in members)
        ),
        accumulation_dtype=(
            next(iter(accumulation_dtypes))
            if len(accumulation_dtypes) == 1
            else "unknown"
        ),
        source_module_path=first.source_module_path,
        source_module_stack=tuple(
            dict.fromkeys(path for node in members for path in node.source_module_stack)
        ),
        flags=flags,
        normalized_args={
            "members": [node.node_id for node in members],
            "member_count": len(members),
            "exact_operation_histogram": dict(sorted(exact_histogram.items())),
            "family_histogram": dict(sorted(family_histogram.items())),
            "first_tensor": {
                "numel": first.input_numel,
                "bytes": first.input_bytes,
            },
            "last_tensor": {
                "numel": last.output_numel,
                "bytes": last.output_bytes,
            },
            "max_live_bytes": max(
                max(node.live_bytes_before, node.live_bytes_after) for node in members
            ),
        },
        input_tensor_count=first.input_tensor_count,
        output_tensor_count=last.output_tensor_count,
        input_numel=first.input_numel,
        output_numel=last.output_numel,
        input_bytes=first.input_bytes,
        output_bytes=last.output_bytes,
        parameter_numel=sum(node.parameter_numel for node in members),
        parameter_bytes=sum(node.parameter_bytes for node in members),
        buffer_numel=sum(node.buffer_numel for node in members),
        buffer_bytes=sum(node.buffer_bytes for node in members),
        flops=flops,
        macs=_sum_estimates(node.macs for node in members),
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        estimated_workspace_bytes=Estimate(
            max(node.estimated_workspace_bytes.value for node in members),
            "shape_formula",
            min(node.estimated_workspace_bytes.confidence for node in members),
        ),
        arithmetic_intensity_flops_per_byte=flops.value / traffic if traffic > 0 else 0.0,
        saved_for_backward_bytes=sum(node.saved_for_backward_bytes for node in members),
        optimizer_state_bytes=sum(node.optimizer_state_bytes for node in members),
        topological_index=first.topological_index,
        depth=first.depth,
        live_bytes_before=first.live_bytes_before,
        live_bytes_after=last.live_bytes_after,
    )


def coarsen_graph(
    graph: GraphIRV3,
    *,
    registry: OperationRegistry | None = None,
) -> GraphIRV3:
    graph.validate()
    registry = registry or OperationRegistry.load()
    regions = _find_regions(graph)
    if not regions:
        return graph
    by_id = {node.node_id: node for node in graph.nodes}
    replacement: dict[str, str] = {}
    region_nodes: dict[str, OperationNodeV3] = {}
    for index, region in enumerate(regions):
        region_id = f"c{index}"
        for member_id in region:
            replacement[member_id] = region_id
        region_nodes[region_id] = _collapse_region(
            tuple(by_id[member_id] for member_id in region),
            region_id=region_id,
            registry=registry,
        )

    ordered_nodes: list[OperationNodeV3] = []
    emitted_regions: set[str] = set()
    for node in graph.nodes:
        replacement_id = replacement.get(node.node_id)
        if replacement_id is None:
            ordered_nodes.append(node)
        elif replacement_id not in emitted_regions:
            ordered_nodes.append(region_nodes[replacement_id])
            emitted_regions.add(replacement_id)

    node_order = {node.node_id: index for index, node in enumerate(ordered_nodes)}
    mapped_edges: list[TensorEdgeV3] = []
    for edge in graph.tensor_edges:
        producer = replacement.get(edge.producer_node_id, edge.producer_node_id)
        consumer = replacement.get(edge.consumer_node_id, edge.consumer_node_id)
        if producer is not None and producer == consumer:
            continue
        mapped_edges.append(
            replace(
                edge,
                edge_id=f"e{len(mapped_edges)}",
                producer_node_id=producer,
                consumer_node_id=consumer,
            )
        )

    fan_in: Counter[str] = Counter(
        edge.consumer_node_id
        for edge in mapped_edges
        if edge.producer_node_id is not None and edge.consumer_node_id is not None
    )
    fan_out: Counter[str] = Counter(
        edge.producer_node_id
        for edge in mapped_edges
        if edge.producer_node_id is not None and edge.consumer_node_id is not None
    )
    normalized_nodes = tuple(
        replace(
            node,
            topological_index=index,
            fan_in=fan_in[node.node_id],
            fan_out=fan_out[node.node_id],
        )
        for index, node in enumerate(ordered_nodes)
    )
    ratio = len(normalized_nodes) / max(1, len(graph.nodes))
    updated = replace(
        graph,
        nodes=normalized_nodes,
        tensor_edges=tuple(mapped_edges),
        global_features=replace(
            graph.global_features,
            operation_nodes=len(normalized_nodes),
            tensor_edges=len(mapped_edges),
            coarsening_ratio=ratio,
        ),
        metadata={
            **graph.metadata,
            "coarsening": {
                "policy": COARSENING_POLICY_ID,
                "policy_sha256": COARSENING_POLICY_SHA256,
                "raw_operation_nodes": len(graph.nodes),
                "coarsened_operation_nodes": len(normalized_nodes),
                "regions": [list(region) for region in regions],
            },
        },
    )
    updated.validate()
    updated = apply_liveness(updated)
    updated = replace(
        updated,
        global_features=replace(
            updated.global_features,
            peak_live_activation_bytes=max(
                graph.global_features.peak_live_activation_bytes,
                updated.global_features.peak_live_activation_bytes,
            ),
        ),
    )
    updated.validate()
    return updated


__all__ = ["COARSENING_POLICY_ID", "COARSENING_POLICY_SHA256", "coarsen_graph"]
