"""Config-driven data module for optimized PerfSeer experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from perfseer.data import ARG_KEYS, NODE_TYPES, list_pairs, parse_graph, parse_label


NUM_TARGETS = 6
TARGET_NAMES: list[str] = [
    "train_util",
    "train_mem",
    "train_time",
    "infer_util",
    "infer_mem",
    "infer_time",
]


class PerfSeerOptimizedData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key == "u":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


try:
    torch.serialization.add_safe_globals([PerfSeerOptimizedData])
except Exception:
    pass


@dataclass(frozen=True)
class FeatureConfig:
    use_operator_type_onehot: bool = True
    topology: bool = False
    critical_path: bool = False
    edge_topology: bool = False
    include_destination_tensor: bool = False
    time_target_mode: str = "raw"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureConfig":
        if data is None:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FeatureLayout:
    node_dim: int
    edge_dim: int
    global_dim: int
    node_std_idx: tuple[int, ...]
    edge_std_idx: tuple[int, ...]
    global_std_idx: tuple[int, ...]
    node_names: tuple[str, ...]
    edge_names: tuple[str, ...]
    global_names: tuple[str, ...]


def feature_layout(cfg: FeatureConfig) -> FeatureLayout:
    node_names: list[str] = []
    node_std: list[int] = []

    def add_node(name: str, standardize: bool) -> None:
        idx = len(node_names)
        node_names.append(name)
        if standardize:
            node_std.append(idx)

    if cfg.use_operator_type_onehot:
        for typ in NODE_TYPES:
            add_node(f"type_{typ}", False)
    for key in ARG_KEYS:
        add_node(f"arg_{key}", True)
    for name in ["log_flops", "log_mac_bytes", "log_weight_size", "arith_intensity"]:
        add_node(name, True)
    for name in ["flops_ratio", "mac_ratio", "weight_ratio"]:
        add_node(name, False)
    if cfg.topology:
        for name in ["in_degree", "out_degree", "topo_index", "forward_depth", "reverse_depth"]:
            add_node(name, True)
        for name in ["is_source", "is_sink", "is_branch", "is_join"]:
            add_node(name, False)
    if cfg.critical_path:
        for name in [
            "longest_depth_from_input",
            "longest_depth_to_output",
            "flop_weighted_depth_from_input",
            "flop_weighted_depth_to_output",
        ]:
            add_node(name, True)
        for name in ["is_on_unweighted_longest_path", "is_on_flop_weighted_longest_path"]:
            add_node(name, False)

    edge_names: list[str] = []
    edge_std: list[int] = []

    def add_edge(name: str, standardize: bool = True) -> None:
        idx = len(edge_names)
        edge_names.append(name)
        if standardize:
            edge_std.append(idx)

    for name in ["edge_tensor_bytes_log1p", "src_batch_log1p", "src_channels_log1p", "src_height_log1p", "src_width_log1p"]:
        add_edge(name, True)
    if cfg.edge_topology:
        for name in ["source_out_degree", "target_in_degree", "source_depth", "target_depth", "depth_delta"]:
            add_edge(name, True)
        add_edge("is_skip_like_edge", False)
    if cfg.include_destination_tensor:
        for name in [
            "destination_input_size_log1p",
            "destination_batch_log1p",
            "destination_channels_log1p",
            "destination_height_log1p",
            "destination_width_log1p",
        ]:
            add_edge(name, True)

    global_names: list[str] = []
    global_std: list[int] = []

    def add_global(name: str, standardize: bool) -> None:
        idx = len(global_names)
        global_names.append(name)
        if standardize:
            global_std.append(idx)

    for name, std in [
        ("num_nodes", True),
        ("num_edges", True),
        ("density", False),
        ("total_flops_log1p", True),
        ("mean_flops_log1p", True),
        ("median_flops_log1p", True),
        ("max_flops_log1p", True),
        ("total_mac_log1p", True),
        ("mean_mac_log1p", True),
        ("median_mac_log1p", True),
        ("max_mac_log1p", True),
        ("total_weight_log1p", True),
        ("mean_weight_log1p", True),
        ("median_weight_log1p", True),
        ("max_weight_log1p", True),
        ("mean_edge_tensor_log1p", True),
        ("model_arith_intensity", True),
        ("batch_size", True),
    ]:
        add_global(name, std)
    if cfg.topology:
        for name in ["max_topological_depth", "mean_topological_depth", "num_branch_nodes", "num_join_nodes", "branch_join_ratio"]:
            add_global(name, True)
    if cfg.critical_path:
        for name in ["max_flop_weighted_path", "critical_path_flops_ratio", "num_nodes_on_longest_path", "num_nodes_on_flop_path"]:
            add_global(name, True)

    return FeatureLayout(
        node_dim=len(node_names),
        edge_dim=len(edge_names),
        global_dim=len(global_names),
        node_std_idx=tuple(node_std),
        edge_std_idx=tuple(edge_std),
        global_std_idx=tuple(global_std),
        node_names=tuple(node_names),
        edge_names=tuple(edge_names),
        global_names=tuple(global_names),
    )


DEFAULT_LAYOUT = feature_layout(FeatureConfig())
NODE_DIM = DEFAULT_LAYOUT.node_dim
EDGE_DIM = DEFAULT_LAYOUT.edge_dim
GLOBAL_DIM = DEFAULT_LAYOUT.global_dim


def _f(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v if np.isfinite(v) else default
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return 0.0
    out = num / den
    return out if np.isfinite(out) else 0.0


def _stats4(arr: np.ndarray) -> tuple[float, float, float, float]:
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.sum(arr)), float(np.mean(arr)), float(np.median(arr)), float(np.max(arr))


def _node_type_onehot(type_str: str) -> list[float]:
    vec = [0.0] * len(NODE_TYPES)
    if type_str in NODE_TYPES:
        vec[NODE_TYPES.index(type_str)] = 1.0
    return vec


def _batch_size_from_features(features: Sequence[dict]) -> float:
    for feat in features:
        mem = feat.get("memory_info", {}) or {}
        bs = _f(mem.get("batch_size"))
        if bs > 0:
            return bs
    return 0.0


def _target_with_mode(label6: Sequence[float], batch_size: float, cfg: FeatureConfig) -> np.ndarray:
    y = np.asarray(label6, dtype=np.float64).reshape(NUM_TARGETS).copy()
    if cfg.time_target_mode == "batch_scaled":
        y[2] *= batch_size
        y[5] *= batch_size
    elif cfg.time_target_mode != "raw":
        raise ValueError(f"unknown time_target_mode {cfg.time_target_mode!r}")
    return y


def _topology_context(g: nx.DiGraph, nodes: list, features: list[dict]) -> dict[str, np.ndarray | float]:
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    try:
        topo_nodes = [node for node in nx.topological_sort(g) if node in index]
        if len(topo_nodes) != n:
            topo_nodes = nodes
    except Exception:
        topo_nodes = nodes

    in_degree = np.asarray([g.in_degree(node) for node in nodes], dtype=np.float64)
    out_degree = np.asarray([g.out_degree(node) for node in nodes], dtype=np.float64)
    rank = np.zeros(n, dtype=np.float64)
    for r, node in enumerate(topo_nodes):
        rank[index[node]] = r

    fwd = np.zeros(n, dtype=np.float64)
    weighted_from = np.zeros(n, dtype=np.float64)
    flops = np.asarray([max(_f(feat.get("flops")), 0.0) for feat in features], dtype=np.float64)
    for node in topo_nodes:
        i = index[node]
        preds = [index[p] for p in g.predecessors(node) if p in index]
        if preds:
            fwd[i] = max(fwd[p] + 1.0 for p in preds)
            weighted_from[i] = max(weighted_from[p] for p in preds) + flops[i]
        else:
            weighted_from[i] = flops[i]

    rev = np.zeros(n, dtype=np.float64)
    weighted_to = np.zeros(n, dtype=np.float64)
    for node in reversed(topo_nodes):
        i = index[node]
        succs = [index[s] for s in g.successors(node) if s in index]
        if succs:
            rev[i] = max(rev[s] + 1.0 for s in succs)
            weighted_to[i] = max(weighted_to[s] for s in succs) + flops[i]
        else:
            weighted_to[i] = flops[i]

    max_depth = float(np.max(fwd)) if n else 0.0
    max_weighted = float(np.max(weighted_from)) if n else 0.0
    on_longest = np.isclose(fwd + rev, max_depth, atol=1e-8).astype(np.float64)
    on_weighted = np.isclose(weighted_from + weighted_to - flops, max_weighted, rtol=1e-6, atol=1e-6).astype(np.float64)

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "topo_index": rank / max(n - 1, 1),
        "forward_depth": fwd / max(max_depth, 1.0),
        "reverse_depth": rev / max(float(np.max(rev)) if n else 0.0, 1.0),
        "fwd_raw": fwd,
        "rev_raw": rev,
        "weighted_from": np.log1p(weighted_from),
        "weighted_to": np.log1p(weighted_to),
        "on_longest": on_longest,
        "on_weighted": on_weighted,
        "max_depth": max_depth,
        "mean_depth": float(np.mean(fwd)) if n else 0.0,
        "max_weighted": max_weighted,
    }


def _node_raw(feat: dict, totals: dict[str, float], cfg: FeatureConfig) -> list[float]:
    args = feat.get("args", {}) or {}
    mem = feat.get("memory_info", {}) or {}
    flops = max(_f(feat.get("flops")), 0.0)
    mac = max(_f(mem.get("bytes")), 0.0)
    weight = max(_f(mem.get("weight_size")), 0.0)
    out: list[float] = []
    if cfg.use_operator_type_onehot:
        out.extend(_node_type_onehot(str(feat.get("type", ""))))
    out.extend(_f(args.get(key)) for key in ARG_KEYS)
    out.extend(
        [
            np.log1p(flops),
            np.log1p(mac),
            np.log1p(weight),
            _f(feat.get("arith_intensity")),
            _safe_div(flops, totals["flops"]),
            _safe_div(mac, totals["mac"]),
            _safe_div(weight, totals["weight"]),
        ]
    )
    return [float(v) for v in out]


def _edge_base(src_feat: dict) -> list[float]:
    mem = src_feat.get("memory_info", {}) or {}
    return [
        np.log1p(max(_f(mem.get("output_size")), 0.0)),
        np.log1p(max(_f(mem.get("batch_size")), 0.0)),
        np.log1p(max(_f(mem.get("output_channels")), 0.0)),
        np.log1p(max(_f(mem.get("output_h")), 0.0)),
        np.log1p(max(_f(mem.get("output_w")), 0.0)),
    ]


def _edge_destination(dst_feat: dict) -> list[float]:
    mem = dst_feat.get("memory_info", {}) or {}
    return [
        np.log1p(max(_f(mem.get("input_size")), 0.0)),
        np.log1p(max(_f(mem.get("batch_size")), 0.0)),
        np.log1p(max(_f(mem.get("input_channels")), 0.0)),
        np.log1p(max(_f(mem.get("input_h")), 0.0)),
        np.log1p(max(_f(mem.get("input_w")), 0.0)),
    ]


def _extract_raw(g: nx.DiGraph, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    layout = feature_layout(cfg)
    nodes = list(g.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    features = [(g.nodes[node].get("feature", {}) or {}) for node in nodes]
    batch_size = _batch_size_from_features(features)

    flops_arr = np.asarray([max(_f(feat.get("flops")), 0.0) for feat in features], dtype=np.float64)
    mac_arr = np.asarray([max(_f((feat.get("memory_info", {}) or {}).get("bytes")), 0.0) for feat in features], dtype=np.float64)
    weight_arr = np.asarray([max(_f((feat.get("memory_info", {}) or {}).get("weight_size")), 0.0) for feat in features], dtype=np.float64)
    totals = {
        "flops": float(np.sum(flops_arr)),
        "mac": float(np.sum(mac_arr)),
        "weight": float(np.sum(weight_arr)),
    }
    topo = _topology_context(g, nodes, features) if (cfg.topology or cfg.critical_path or cfg.edge_topology) else None

    x_raw = np.zeros((n, layout.node_dim), dtype=np.float64)
    for i, feat in enumerate(features):
        row = _node_raw(feat, totals, cfg)
        if cfg.topology and topo is not None:
            row.extend(
                [
                    np.log1p(topo["in_degree"][i]),  # type: ignore[index]
                    np.log1p(topo["out_degree"][i]),  # type: ignore[index]
                    topo["topo_index"][i],  # type: ignore[index]
                    topo["forward_depth"][i],  # type: ignore[index]
                    topo["reverse_depth"][i],  # type: ignore[index]
                    float(topo["in_degree"][i] == 0),  # type: ignore[index]
                    float(topo["out_degree"][i] == 0),  # type: ignore[index]
                    float(topo["out_degree"][i] > 1),  # type: ignore[index]
                    float(topo["in_degree"][i] > 1),  # type: ignore[index]
                ]
            )
        if cfg.critical_path and topo is not None:
            row.extend(
                [
                    topo["fwd_raw"][i],  # type: ignore[index]
                    topo["rev_raw"][i],  # type: ignore[index]
                    topo["weighted_from"][i],  # type: ignore[index]
                    topo["weighted_to"][i],  # type: ignore[index]
                    topo["on_longest"][i],  # type: ignore[index]
                    topo["on_weighted"][i],  # type: ignore[index]
                ]
            )
        x_raw[i] = row

    edges = list(g.edges())
    edge_index = np.zeros((2, len(edges)), dtype=np.int64)
    e_raw = np.zeros((len(edges), layout.edge_dim), dtype=np.float64)
    edge_sizes = np.zeros(len(edges), dtype=np.float64)
    for j, (src, dst) in enumerate(edges):
        si, ti = node_index[src], node_index[dst]
        edge_index[:, j] = [si, ti]
        row = _edge_base(features[si])
        edge_sizes[j] = max(_f((features[si].get("memory_info", {}) or {}).get("output_size")), 0.0)
        if cfg.edge_topology and topo is not None:
            depth_delta = float(topo["fwd_raw"][ti] - topo["fwd_raw"][si])  # type: ignore[index]
            row.extend(
                [
                    np.log1p(topo["out_degree"][si]),  # type: ignore[index]
                    np.log1p(topo["in_degree"][ti]),  # type: ignore[index]
                    topo["forward_depth"][si],  # type: ignore[index]
                    topo["forward_depth"][ti],  # type: ignore[index]
                    depth_delta,
                    float(depth_delta > 1.0),
                ]
            )
        if cfg.include_destination_tensor:
            row.extend(_edge_destination(features[ti]))
        e_raw[j] = row

    u: list[float] = []
    e = len(edges)
    density = _safe_div(float(e), float(n) * (float(n) - 1.0)) if n > 1 else 0.0
    u.extend([float(n), float(e), density])
    for val in _stats4(flops_arr):
        u.append(np.log1p(max(val, 0.0)))
    for val in _stats4(mac_arr):
        u.append(np.log1p(max(val, 0.0)))
    for val in _stats4(weight_arr):
        u.append(np.log1p(max(val, 0.0)))
    u.append(np.log1p(max(float(np.mean(edge_sizes)) if edge_sizes.size else 0.0, 0.0)))
    u.append(_safe_div(totals["flops"], totals["mac"]))
    u.append(batch_size)
    if cfg.topology and topo is not None:
        branch_nodes = float(np.sum(np.asarray(topo["out_degree"]) > 1.0))
        join_nodes = float(np.sum(np.asarray(topo["in_degree"]) > 1.0))
        u.extend(
            [
                float(topo["max_depth"]),
                float(topo["mean_depth"]),
                branch_nodes,
                join_nodes,
                _safe_div(branch_nodes, join_nodes + 1.0),
            ]
        )
    if cfg.critical_path and topo is not None:
        max_weighted = float(topo["max_weighted"])
        u.extend(
            [
                np.log1p(max(max_weighted, 0.0)),
                _safe_div(max_weighted, totals["flops"]),
                float(np.sum(topo["on_longest"])),  # type: ignore[arg-type]
                float(np.sum(topo["on_weighted"])),  # type: ignore[arg-type]
            ]
        )

    u_raw = np.asarray(u, dtype=np.float64)
    if u_raw.shape[0] != layout.global_dim:
        raise ValueError(f"global feature length mismatch: {u_raw.shape[0]} != {layout.global_dim}")
    return x_raw, edge_index, e_raw, u_raw, batch_size


def standardize_targets(y_raw, stats: dict[str, np.ndarray]):
    is_torch = hasattr(y_raw, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_raw.dtype, device=y_raw.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_raw.dtype, device=y_raw.device)
        return (torch.log1p(torch.clamp(y_raw, min=0.0)) - ym) / ys
    y = np.asarray(y_raw, dtype=np.float64)
    return (np.log1p(np.maximum(y, 0.0)) - stats["y_mean"]) / stats["y_std"]


def invert_targets(y_std, stats: dict[str, np.ndarray]):
    is_torch = hasattr(y_std, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_std.dtype, device=y_std.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_std.dtype, device=y_std.device)
        return torch.expm1(y_std * ys + ym)
    y = np.asarray(y_std, dtype=np.float64)
    return np.expm1(y * np.asarray(stats["y_std"]) + np.asarray(stats["y_mean"]))


def build_pyg_data(
    g: nx.DiGraph,
    label6: Sequence[float],
    norm_stats: Optional[dict[str, np.ndarray]] = None,
    feature_config: FeatureConfig | None = None,
) -> Data:
    cfg = feature_config or FeatureConfig()
    layout = feature_layout(cfg)
    x_raw, edge_idx, e_raw, u_raw, batch_size = _extract_raw(g, cfg)
    y_raw = _target_with_mode(label6, batch_size, cfg)

    if norm_stats is not None:
        x = (x_raw - norm_stats["node_mean"]) / norm_stats["node_std"]
        e = (e_raw - norm_stats["edge_mean"]) / norm_stats["edge_std"]
        u = (u_raw - norm_stats["global_mean"]) / norm_stats["global_std"]
        y = standardize_targets(y_raw, norm_stats)
    else:
        x, e, u, y = x_raw, e_raw, u_raw, np.log1p(np.maximum(y_raw, 0.0))

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    data = PerfSeerOptimizedData(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_idx.astype(np.int64)),
        edge_attr=torch.from_numpy(e.astype(np.float32)),
        u=torch.from_numpy(u.astype(np.float32)).view(1, layout.global_dim),
        y=torch.from_numpy(y.astype(np.float32)).view(1, NUM_TARGETS),
        y_raw=torch.from_numpy(y_raw.astype(np.float32)).view(1, NUM_TARGETS),
    )
    data.num_nodes = int(x_raw.shape[0])
    return data


def _num_procs() -> int:
    env = os.environ.get("PERFSEER_NUM_PROC")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _chunkify(seq: Sequence, n_chunks: int) -> list[list]:
    seq = list(seq)
    if not seq:
        return []
    n_chunks = max(1, min(n_chunks, len(seq)))
    k = (len(seq) + n_chunks - 1) // n_chunks
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def _mean_std_from_acc(count: float, s: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros_like(s), np.ones_like(s)
    mean = s / count
    var = np.maximum(ss / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean, std


def _stats_chunk_worker(arg):
    pairs, cfg = arg
    layout = feature_layout(cfg)
    acc = {
        "node": [0.0, np.zeros(layout.node_dim), np.zeros(layout.node_dim)],
        "edge": [0.0, np.zeros(layout.edge_dim), np.zeros(layout.edge_dim)],
        "global": [0.0, np.zeros(layout.global_dim), np.zeros(layout.global_dim)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    for gp, lp in pairs:
        g = parse_graph(gp)
        x_raw, _ei, e_raw, u_raw, batch_size = _extract_raw(g, cfg)
        y_log = np.log1p(np.maximum(_target_with_mode(parse_label(lp), batch_size, cfg), 0.0))
        if x_raw.shape[0]:
            acc["node"][0] += x_raw.shape[0]
            acc["node"][1] += x_raw.sum(0)
            acc["node"][2] += (x_raw * x_raw).sum(0)
        if e_raw.shape[0]:
            acc["edge"][0] += e_raw.shape[0]
            acc["edge"][1] += e_raw.sum(0)
            acc["edge"][2] += (e_raw * e_raw).sum(0)
        acc["global"][0] += 1.0
        acc["global"][1] += u_raw
        acc["global"][2] += u_raw * u_raw
        acc["y"][0] += 1.0
        acc["y"][1] += y_log
        acc["y"][2] += y_log * y_log
    return acc


def compute_norm_stats(train_files: Sequence[Tuple[str, str]], feature_config: FeatureConfig | None = None) -> dict[str, np.ndarray]:
    cfg = feature_config or FeatureConfig()
    layout = feature_layout(cfg)
    pairs = list(train_files)
    total = {
        "node": [0.0, np.zeros(layout.node_dim), np.zeros(layout.node_dim)],
        "edge": [0.0, np.zeros(layout.edge_dim), np.zeros(layout.edge_dim)],
        "global": [0.0, np.zeros(layout.global_dim), np.zeros(layout.global_dim)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    nproc = min(_num_procs(), max(1, len(pairs)))
    chunks = _chunkify(pairs, nproc * 4)
    if nproc > 1 and len(chunks) > 1:
        with get_context("fork").Pool(nproc) as pool:
            partials = pool.map(_stats_chunk_worker, [(c, cfg) for c in chunks])
    else:
        partials = [_stats_chunk_worker((c, cfg)) for c in chunks]
    for part in partials:
        for key in total:
            total[key][0] += part[key][0]
            total[key][1] += part[key][1]
            total[key][2] += part[key][2]

    node_mean, node_std = _mean_std_from_acc(*total["node"])
    edge_mean, edge_std = _mean_std_from_acc(*total["edge"])
    global_mean, global_std = _mean_std_from_acc(*total["global"])
    y_mean, y_std = _mean_std_from_acc(*total["y"])

    node_mean_m = np.zeros(layout.node_dim)
    node_std_m = np.ones(layout.node_dim)
    for i in layout.node_std_idx:
        node_mean_m[i] = node_mean[i]
        node_std_m[i] = node_std[i]
    edge_mean_m = np.zeros(layout.edge_dim)
    edge_std_m = np.ones(layout.edge_dim)
    for i in layout.edge_std_idx:
        edge_mean_m[i] = edge_mean[i]
        edge_std_m[i] = edge_std[i]
    global_mean_m = np.zeros(layout.global_dim)
    global_std_m = np.ones(layout.global_dim)
    for i in layout.global_std_idx:
        global_mean_m[i] = global_mean[i]
        global_std_m[i] = global_std[i]

    return {
        "node_mean": node_mean_m.astype(np.float32),
        "node_std": node_std_m.astype(np.float32),
        "edge_mean": edge_mean_m.astype(np.float32),
        "edge_std": edge_std_m.astype(np.float32),
        "global_mean": global_mean_m.astype(np.float32),
        "global_std": global_std_m.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def split_dataset(
    data_root: str = "dataset",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    pairs = list_pairs(data_root)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n = len(pairs)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx], [pairs[i] for i in test_idx]


def split_hash(pairs: Sequence[Tuple[str, str]]) -> str:
    h = hashlib.sha1()
    for gp, lp in pairs:
        h.update(os.path.basename(gp).encode())
        h.update(os.path.basename(lp).encode())
    return h.hexdigest()


def norm_stats_to_serializable(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: np.asarray(v, dtype=float).reshape(-1).tolist() for k, v in stats.items()}


def save_norm_stats(stats: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_norm_stats(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def _build_chunk_worker(arg):
    pairs, norm_stats, cfg = arg
    out: list[Data] = []
    for gp, lp in pairs:
        try:
            out.append(build_pyg_data(parse_graph(gp), parse_label(lp), norm_stats, cfg))
        except Exception as exc:
            print(f"[PerfSeerOptimizedDataset] skipping {gp}: {exc}", flush=True)
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


class PerfSeerOptimizedDataset(InMemoryDataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        norm_stats: Optional[dict[str, np.ndarray]] = None,
        split_name: Optional[str] = None,
        feature_config: FeatureConfig | None = None,
        force_reprocess: bool = False,
        *,
        root: Optional[str] = None,
        file_list: Optional[Sequence[Tuple[str, str]]] = None,
        split: Optional[str] = None,
    ) -> None:
        data_root = root if root is not None else data_root
        pairs = file_list if file_list is not None else pairs
        split_name = split if split is not None else split_name
        if data_root is None:
            raise ValueError("PerfSeerOptimizedDataset requires data_root/root")
        if pairs is None:
            raise ValueError("PerfSeerOptimizedDataset requires pairs/file_list")
        if norm_stats is None:
            raise ValueError("PerfSeerOptimizedDataset requires norm_stats")
        if split_name is None:
            raise ValueError("PerfSeerOptimizedDataset requires split_name/split")
        self._pairs = list(pairs)
        self._norm_stats = norm_stats
        self._split_name = split_name
        self._feature_config = feature_config or FeatureConfig()
        self._data_root = data_root
        if force_reprocess:
            cache = os.path.join(data_root, "processed", self._cache_filename())
            if os.path.exists(cache):
                os.remove(cache)
        super().__init__(root=data_root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    def _cache_filename(self) -> str:
        h = hashlib.sha1()
        h.update(self._feature_config.signature().encode())
        layout = feature_layout(self._feature_config)
        h.update(f"{layout.node_dim}:{layout.edge_dim}:{layout.global_dim}".encode())
        for gp, lp in self._pairs:
            h.update(os.path.basename(gp).encode())
            h.update(os.path.basename(lp).encode())
        for key in sorted(self._norm_stats):
            h.update(key.encode())
            h.update(np.asarray(self._norm_stats[key], dtype=np.float64).tobytes())
        return f"perfseer_opt_{self._split_name}_{h.hexdigest()[:12]}.pt"

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [self._cache_filename()]

    def download(self) -> None:
        pass

    def process(self) -> None:
        pairs = list(self._pairs)
        nproc = min(_num_procs(), max(1, len(pairs)))
        chunks = _chunkify(pairs, nproc * 4)
        args = [(c, self._norm_stats, self._feature_config) for c in chunks]
        try:
            if nproc > 1 and len(chunks) > 1:
                with get_context("fork").Pool(nproc) as pool:
                    results = pool.map(_build_chunk_worker, args)
            else:
                results = [_build_chunk_worker(a) for a in args]
        except Exception as exc:
            print(f"[PerfSeerOptimizedDataset] parallel build failed ({exc}); falling back to serial")
            results = [_build_chunk_worker(a) for a in args]
        data_list: list[Data] = [d for blob in results for d in pickle.loads(blob)]
        self.save(data_list, self.processed_paths[0])


# Compatibility alias that mirrors the baseline module name.
PerfSeerDataset = PerfSeerOptimizedDataset
