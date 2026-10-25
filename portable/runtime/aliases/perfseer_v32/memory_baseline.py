"""Explicit storage lifetimes and a deliberately approximate static graph baseline."""

from collections import defaultdict
import math

import numpy as np

from .capture import graphs_from_design

TRACE_VERSION = "perfseer_v32_storage_trace_v1"
BASELINE_VERSION = "perfseer_v32_static_memory_baseline_v1"


def storage_peak(trace):
    """Count backing extents once at each inclusive operation boundary.

    Repeated storage IDs describe aliases; nonoverlapping intervals support lazy
    initialization and release. Callers must include backward saves, gradients,
    optimizer state and intra-operation temporaries in their declared trace.
    """
    if trace.get("version") != TRACE_VERSION or trace.get("scope") != "allocated_storage":
        raise ValueError("memory trace version or scope differs")
    storages, points = {}, set()
    for item in trace["storages"]:
        key, size = item["storage_id"], item["storage_bytes"]
        if not key or type(size) is not int or size < 0:
            raise ValueError("backing storage identity and full byte extent are required")
        if key in storages and storages[key]["size"] != size:
            raise ValueError("aliases disagree on backing storage extent")
        entry = storages.setdefault(key, {"size": size, "intervals": []})
        for birth, last in item["lifetimes"]:
            if type(birth) is not int or type(last) is not int or last < birth:
                raise ValueError("invalid storage lifetime")
            entry["intervals"].append((birth, last))
            points.update((birth, last))
    live = [{"position": point, "bytes": sum(item["size"] for item in storages.values()
              if any(birth <= point <= last for birth, last in item["intervals"]))} for point in sorted(points)]
    unsupported = list(trace.get("unsupported_operators", []))
    return {"version": TRACE_VERSION, "scope": "allocated_storage", "peak_bytes": max((item["bytes"] for item in live), default=0),
            "live": live, "unique_storages": len(storages), "supported": not unsupported,
            "unsupported_operators": unsupported, "is_process_memory": False,
            "available_before_execution": bool(trace.get("available_before_execution", False))}


def process_peak(reserved_bytes, external_bytes):
    reserved, external = (np.asarray(value, dtype=np.float64) for value in (reserved_bytes, external_bytes))
    if reserved.ndim != 1 or not len(reserved) or reserved.shape != external.shape or not np.isfinite(reserved).all() or not np.isfinite(external).all() or (reserved < 0).any() or (external < 0).any():
        raise ValueError("process peak requires nonnegative, finite, coincident samples")
    return float(np.max(reserved + external))


def graph_baseline(design):
    """A static descriptor, not a lower bound or complete training simulator.

    GraphIR v3 stores view sizes rather than full backing extents. Keep that
    limitation explicit instead of treating its alias-aware liveness as exact.
    """
    graph = graphs_from_design(design)["training"]
    positions = {node.node_id: node.topological_index for node in graph.nodes}
    intervals = {}
    for edge in graph.tensor_edges:
        if edge.tensor_role not in {"activation", "model_input", "model_output"}:
            continue
        key = edge.alias_group or f"edge:{edge.edge_id}"
        birth = positions.get(edge.producer_node_id, -1)
        last = positions.get(edge.consumer_node_id, len(graph.nodes))
        item = intervals.setdefault(key, {"birth": birth, "last": last, "bytes": 0})
        item.update(birth=min(item["birth"], birth), last=max(item["last"], last),
                    bytes=max(item["bytes"], edge.tensor_bytes or 0))
    births, deaths = defaultdict(int), defaultdict(int)
    for item in intervals.values():
        births[item["birth"]] += item["bytes"]
        deaths[item["last"]] += item["bytes"]
    global_features = graph.global_features
    # Persistent dtypes come from graph byte descriptors, not an AMP-wide cast.
    persistent = (global_features.total_parameter_bytes + global_features.total_buffer_bytes +
                  global_features.total_optimizer_state_bytes + global_features.total_parameter_bytes)
    live, peak = 0, 0
    workspaces = {node.topological_index: node.estimated_workspace_bytes.value or 0 for node in graph.nodes}
    for position in sorted(set(births) | set(deaths) | set(workspaces)):
        live += births[position]
        peak = max(peak, live + workspaces.get(position, 0))
        live -= deaths[position]
    estimate = persistent + peak
    if not math.isfinite(estimate) or estimate < 0:
        raise ValueError("invalid graph memory estimate")
    unsupported = [node.raw_target for node in graph.nodes if node.estimated_workspace_bytes.method == "unknown"]
    return {"version": BASELINE_VERSION, "peak_bytes": float(estimate), "persistent_bytes": persistent,
            "available_before_execution": True, "supported": False, "is_lower_bound": False,
            "scope": "approximate_allocated_storage", "unsupported_operators": sorted(set(unsupported)),
            "uncertainty": ["backing_extents_not_recorded_in_graph_ir", "gradient_bytes_assume_parameter_dtype",
                            "lazy_optimizer_and_saved_activation_lifetimes_approximate",
                            "allocator_reservation_and_external_allocations_unmodeled"]}
