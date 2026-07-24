"""Featurize PerfSeer compute graphs for the 53/3/40 A10 student."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

OP_VOCAB = [
    "Add",
    "Attention",
    "Concat",
    "Conv",
    "DepthwiseConv",
    "DetectorHead",
    "Embedding",
    "Flatten",
    "GRU",
    "Gelu",
    "Gemm",
    "GlobalAveragePool",
    "GraphAttention",
    "GraphMessage",
    "LSTM",
    "LayerNormalization",
    "MaxPool",
    "Relu",
    "SegmentationHead",
    "Silu",
    "Softmax",
    "TabularFeature",
    "Upsample",
]
OP_INDEX = {op: index for index, op in enumerate(OP_VOCAB)}
ARG_KEYS = (
    "conv_kernel_size",
    "conv_stride",
    "conv_padding",
    "conv_dilation",
    "conv_groups",
    "conv_bias",
    "linear_in_features",
    "linear_out_features",
    "linear_bias",
    "pool_kernel_size",
    "pool_stride",
    "pool_padding",
    "pool_ceil_mode",
)
MEMORY_KEYS = (
    "weight_size",
    "input_size",
    "output_size",
    "bytes",
    "input_channels",
    "output_channels",
    "input_features",
    "output_features",
    "spatial_area",
    "sequence_length",
)
PRECISIONS = ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp")
PRECISION_INDEX = {precision: index for index, precision in enumerate(PRECISIONS)}
TARGET_NAMES = (
    "train_util",
    "train_mem",
    "train_time",
    "infer_util",
    "infer_mem",
    "infer_time",
)
NODE_DIM = 53
EDGE_DIM = 3
GLOBAL_DIM = 40


class UnsupportedStudentOperationError(ValueError):
    """Raised when a graph contains an operation with no student feature slot."""


def _log1p_nonnegative(value: object) -> float:
    return math.log1p(max(float(value or 0), 0.0))


def featurize_graph(
    graph: nx.DiGraph,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw ``x``, ``edge_index``, ``edge_attr`` and ``u`` arrays."""

    if precision not in PRECISION_INDEX:
        raise ValueError(f"unsupported precision {precision!r}; expected one of {PRECISIONS}")
    nodes = list(graph.nodes())
    if not nodes:
        raise ValueError("cannot featurize an empty graph")
    unknown_operations = sorted(
        {
            str(graph.nodes[node].get("feature", {}).get("type") or "<missing>")
            for node in nodes
            if str(graph.nodes[node].get("feature", {}).get("type") or "<missing>") not in OP_INDEX
        }
    )
    if unknown_operations:
        raise UnsupportedStudentOperationError(
            "student operation vocabulary does not cover: " + ", ".join(unknown_operations)
        )
    node_index = {node: index for index, node in enumerate(nodes)}
    indegree = np.asarray([graph.in_degree(node) for node in nodes], dtype=np.float32)
    outdegree = np.asarray([graph.out_degree(node) for node in nodes], dtype=np.float32)
    try:
        topological = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        topological = nodes
    topological_position = {node: index for index, node in enumerate(topological)}
    depth = {node: 0 for node in nodes}
    for node in topological:
        for predecessor in graph.predecessors(node):
            depth[node] = max(depth[node], depth.get(predecessor, 0) + 1)
    max_depth = max(depth.values(), default=1)

    one_hot = np.zeros((len(nodes), len(OP_VOCAB)), dtype=np.float32)
    continuous: list[list[float]] = []
    flops: list[float] = []
    weights: list[float] = []
    outputs: list[float] = []
    byte_counts: list[float] = []
    ranks: list[float] = []
    op_histogram = np.zeros(len(OP_VOCAB), dtype=np.float32)

    for node in nodes:
        feature = graph.nodes[node]["feature"]
        op_type = feature.get("type")
        memory = feature.get("memory_info", {})
        arguments = feature.get("args", {})
        if op_type in OP_INDEX:
            one_hot[node_index[node], OP_INDEX[op_type]] = 1.0
            op_histogram[OP_INDEX[op_type]] += 1.0
        row = [float(arguments.get(key, 0) or 0) for key in ARG_KEYS]
        row.extend(
            [
                _log1p_nonnegative(feature.get("flops", 0)),
                float(feature.get("arith_intensity", 0.0) or 0.0),
            ]
        )
        row.extend(_log1p_nonnegative(memory.get(key, 0)) for key in MEMORY_KEYS)
        row.append(float(memory.get("rank", 0) or 0))
        index = node_index[node]
        row.extend(
            [
                float(indegree[index]),
                float(outdegree[index]),
                topological_position.get(node, index) / max(len(nodes) - 1, 1),
                depth[node] / max(max_depth, 1),
            ]
        )
        continuous.append(row)
        flops.append(float(feature.get("flops", 0) or 0))
        weights.append(float(memory.get("weight_size", 0) or 0))
        outputs.append(float(memory.get("output_size", 0) or 0))
        byte_counts.append(float(memory.get("bytes", 0) or 0))
        ranks.append(float(memory.get("rank", 0) or 0))

    edges = list(graph.edges())
    edge_index = np.zeros((2, len(edges)), dtype=np.int64)
    edge_features = np.zeros((len(edges), EDGE_DIM), dtype=np.float32)
    for edge_number, (source, destination) in enumerate(edges):
        edge_index[:, edge_number] = (node_index[source], node_index[destination])
        source_memory = graph.nodes[source]["feature"]["memory_info"]
        destination_memory = graph.nodes[destination]["feature"]["memory_info"]
        edge_features[edge_number] = (
            _log1p_nonnegative(source_memory.get("output_size", 0)),
            _log1p_nonnegative(destination_memory.get("input_size", 0)),
            _log1p_nonnegative(source_memory.get("bytes", 0)),
        )

    branch_count = sum(value > 1 for value in outdegree)
    join_count = sum(value > 1 for value in indegree)
    graph_features = np.asarray(
        [
            _log1p_nonnegative(len(nodes)),
            _log1p_nonnegative(len(edges)),
            len(edges) / max(len(nodes) * (len(nodes) - 1), 1),
            _log1p_nonnegative(sum(flops)),
            _log1p_nonnegative(np.mean(flops) if flops else 0),
            _log1p_nonnegative(max(flops, default=0)),
            _log1p_nonnegative(sum(weights)),
            _log1p_nonnegative(sum(outputs)),
            _log1p_nonnegative(max(byte_counts, default=0)),
            max(ranks, default=0),
            _log1p_nonnegative(branch_count),
            _log1p_nonnegative(join_count),
            float(max_depth),
        ],
        dtype=np.float32,
    )
    global_continuous = np.concatenate([graph_features, np.log1p(op_histogram)])
    precision_one_hot = np.zeros(len(PRECISIONS), dtype=np.float32)
    precision_one_hot[PRECISION_INDEX[precision]] = 1.0

    x = np.concatenate([one_hot, np.asarray(continuous, dtype=np.float32)], axis=1)
    u = np.concatenate([global_continuous, precision_one_hot])[None, :].astype(np.float32)
    if x.shape[1] != NODE_DIM or edge_features.shape[1] != EDGE_DIM or u.shape[1] != GLOBAL_DIM:
        raise RuntimeError(
            f"student schema mismatch: got {x.shape[1]}/{edge_features.shape[1]}/{u.shape[1]}, "
            f"expected {NODE_DIM}/{EDGE_DIM}/{GLOBAL_DIM}"
        )
    return x, edge_index, edge_features, u
