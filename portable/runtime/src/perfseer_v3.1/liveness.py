"""Alias-aware tensor lifetime and peak-live-activation features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from perfseer_v3.graph_ir_v3 import GraphIRV3


_LIVE_ROLES = {"activation", "model_input", "model_output", "gradient"}


def apply_liveness(graph: GraphIRV3) -> GraphIRV3:
    graph.validate()
    node_positions = {node.node_id: node.topological_index for node in graph.nodes}
    output_position = len(graph.nodes)
    groups: dict[str, dict[str, int | str]] = {}
    group_edges: dict[str, list[int]] = defaultdict(list)

    for index, edge in enumerate(graph.tensor_edges):
        group_id = edge.alias_group or f"edge:{edge.edge_id}"
        group_edges[group_id].append(index)
        birth = node_positions.get(edge.producer_node_id, -1)
        use = node_positions.get(edge.consumer_node_id, output_position)
        current = groups.get(group_id)
        if current is None:
            groups[group_id] = {
                "birth": birth,
                "last": use,
                "bytes": edge.tensor_bytes or 0,
                "role": edge.tensor_role,
            }
        else:
            current["birth"] = min(int(current["birth"]), birth)
            current["last"] = max(int(current["last"]), use)
            current["bytes"] = max(int(current["bytes"]), edge.tensor_bytes or 0)
            if edge.tensor_role in _LIVE_ROLES:
                current["role"] = edge.tensor_role

    updated_edges = list(graph.tensor_edges)
    for group_id, indices in group_edges.items():
        interval = groups[group_id]
        birth = int(interval["birth"])
        last = int(interval["last"])
        for index in indices:
            edge = graph.tensor_edges[index]
            use = node_positions.get(edge.consumer_node_id, output_position)
            updated_edges[index] = replace(
                edge,
                first_use_distance=max(0, use - birth),
                last_use_distance=max(0, last - birth),
            )

    births, deaths = defaultdict(int), defaultdict(int)
    for interval in groups.values():
        if interval["role"] in _LIVE_ROLES:
            births[int(interval["birth"])] += int(interval["bytes"])
            deaths[int(interval["last"])] += int(interval["bytes"])
    live = sum(size for position, size in births.items() if position < 0)
    peak = 0
    updated_nodes = []
    for node in graph.nodes:
        position = node.topological_index
        live_before = live
        live_after = live + births[position] - deaths[position]
        live = live_after
        peak = max(peak, live_before, live_after)
        updated_nodes.append(
            replace(node, live_bytes_before=live_before, live_bytes_after=live_after)
        )

    updated = replace(
        graph,
        nodes=tuple(updated_nodes),
        tensor_edges=tuple(updated_edges),
        global_features=replace(
            graph.global_features,
            peak_live_activation_bytes=peak,
        ),
    )
    updated.validate()
    return updated


__all__ = ["apply_liveness"]
