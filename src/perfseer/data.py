"""Data module for the PerfSeer reproduction.

This module implements the feature engineering described in the PerfSeer paper
(arXiv:2502.01206, section 3.1) on the real released dataset:

  - graphs: dataset/cg/cg/*.pkl  -> pickled networkx.DiGraph (53407 files)
  - labels: dataset/label/label/*.txt -> python dict literal, matching filename stems

Each node carries a ``feature`` dict with keys: type, args, memory_info, flops,
arith_intensity. Edges carry NO attributes, so edge features are DERIVED from the
SOURCE node's memory_info (output tensor size + shape).

Pipeline:
  1. parse_graph(path)            -> networkx.DiGraph
  2. parse_label(path)            -> 6-vector of raw targets in the fixed metric order
  3. compute_norm_stats(files)    -> per-channel (mean, std) of log1p-features over TRAIN files
  4. build_pyg_data(g, y6, stats) -> torch_geometric.data.Data
  5. PerfSeerDataset              -> InMemoryDataset that caches processed graphs to disk
  6. split_dataset(seed)          -> (train, val, test) file-stem lists in a 2:1:1 ratio

Standardization: every "large magnitude" feature is passed through log1p and then
z-scored using statistics computed on the TRAIN split only. Targets y are likewise
predicted in log1p space and standardized by train stats; helpers to invert are
provided so eval can recover the original metric space.

IMPORTANT: several memory_info values are stored as STRINGS (e.g. "282594816"),
so every numeric field is coerced with the robust ``_f`` cast below.
"""

from __future__ import annotations

import os
import pickle
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class PerfSeerData(Data):
    """PyG ``Data`` subclass that pins how the graph-global feature ``u`` is
    batched. ``u`` is stored as ``[1, GLOBAL_DIM]`` per graph; we force
    concatenation along dim 0 so a batch of B graphs yields ``u`` of shape
    ``[B, GLOBAL_DIM]`` (one row per graph), which is what ``model.SeerNet``
    expects. Relying on PyG's default ``__cat_dim__`` is fragile for single-node
    graphs (where the leading dim of ``u`` would collide with ``num_nodes``), so
    we make the intent explicit here.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key == "u":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

# ---------------------------------------------------------------------------
# Constants and exported feature dimensions
# ---------------------------------------------------------------------------

# The ~10 node operator types observed in the dataset. Order is fixed so the
# one-hot encoding is stable across processing runs and machines. An unknown
# type falls back to an all-zero one-hot (handled in _node_type_onehot).
NODE_TYPES: List[str] = [
    "Conv",
    "Relu",
    "BatchNormalization",
    "Concat",
    "AveragePool",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Add",
]
NUM_NODE_TYPES = len(NODE_TYPES)  # 10

# Selected hyper-parameter args (v^hp), in a fixed order. These are the 13 ints
# stored in feature['args']. We keep all of them; they are integer-valued and
# small, so they are NOT log1p'd, only z-scored together with the rest.
ARG_KEYS: List[str] = [
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
]
NUM_ARGS = len(ARG_KEYS)  # 13

# --- Node feature layout (v = v^hp || v^c || v^m || v^a || v^p) ---
#   v^hp : one-hot type (NUM_NODE_TYPES) + selected args (NUM_ARGS)
#   v^c  : log1p(flops)                                  -> 1
#   v^m  : log1p(MAC=bytes) + log1p(weight_size)         -> 2
#   v^a  : arith_intensity                               -> 1
#   v^p  : flops/total_flops, mac/total_mac, weight/total_weight -> 3
NODE_DIM = NUM_NODE_TYPES + NUM_ARGS + 1 + 2 + 1 + 3  # = 30

# Indices (within the node vector) of the *continuous* channels that must be
# z-score standardized. The one-hot type block and the proportion block (v^p,
# already in [0, 1]) are left un-standardized. The arg block IS standardized.
# Layout offsets:
_OFF_TYPE = 0
_OFF_ARGS = _OFF_TYPE + NUM_NODE_TYPES        # 10
_OFF_C = _OFF_ARGS + NUM_ARGS                 # 23  (flops)
_OFF_M = _OFF_C + 1                           # 24  (mac, weight)
_OFF_A = _OFF_M + 2                           # 26  (arith_intensity)
_OFF_P = _OFF_A + 1                           # 27  (3 proportions)
# Standardize args + v^c + v^m + v^a; leave one-hot and proportions alone.
NODE_STD_IDX: List[int] = (
    list(range(_OFF_ARGS, _OFF_ARGS + NUM_ARGS))   # 13 arg channels
    + [_OFF_C]                                      # flops
    + [_OFF_M, _OFF_M + 1]                          # mac, weight
    + [_OFF_A]                                      # arith_intensity
)

# --- Edge feature layout (e = e^sz || e^sp) ---
#   e^sz : log1p(source output_size bytes)                       -> 1
#   e^sp : log1p(batch_size, output_channels, output_h, output_w) -> 4
EDGE_DIM = 1 + 4  # = 5
EDGE_STD_IDX: List[int] = list(range(EDGE_DIM))  # all edge channels standardized

# --- Global feature layout (u = u^gp || u^c || u^m || u^a || u^b) ---
#   u^gp : num_nodes, num_edges, density = E/(V*(V-1))           -> 3
#   u^c  : log1p of {total, mean, median, max} flops             -> 4
#   u^m  : log1p of {total, mean, median, max} MAC               -> 4
#          log1p of {total, mean, median, max} weight            -> 4
#          log1p of mean edge tensor size                        -> 1
#   u^a  : model arith intensity = total_flops / total_mac       -> 1
#   u^b  : batch size                                            -> 1
GLOBAL_DIM = 3 + 4 + (4 + 4 + 1) + 1 + 1  # = 18

# Global continuous channels to standardize. We standardize everything EXCEPT
# the density channel (index 2), which is already a ratio in [0, 1].
_GLOBAL_ALL = list(range(GLOBAL_DIM))
GLOBAL_STD_IDX: List[int] = [i for i in _GLOBAL_ALL if i != 2]

# Number of regression targets (the 6 metrics, in the fixed order below).
NUM_TARGETS = 6

# Target order (paper Tables 1-3); kept EXACTLY:
#   [0] train_util = train.average_sm_util
#   [1] train_mem  = train.peak_memory_usuage
#   [2] train_time = train.time
#   [3] infer_util = infer.average_sm_util
#   [4] infer_mem  = infer.peak_memory_usuage
#   [5] infer_time = infer.time
TARGET_NAMES: List[str] = [
    "train_util",
    "train_mem",
    "train_time",
    "infer_util",
    "infer_mem",
    "infer_time",
]

# Default dataset locations (relative to a data-root). The remote layout has a
# doubled directory ("cg/cg", "label/label"); _resolve_dirs handles both.
_CG_SUBDIRS = [os.path.join("cg", "cg"), "cg"]
_LABEL_SUBDIRS = [os.path.join("label", "label"), "label"]


# ---------------------------------------------------------------------------
# Robust scalar casting
# ---------------------------------------------------------------------------

def _f(value, default: float = 0.0) -> float:
    """Robustly cast an arbitrary value (int / str / None / bool) to float.

    The released memory_info dict stores several numbers as STRINGS, and a few
    fields may be missing or empty. This never raises; it returns ``default``
    on failure so processing is resilient to malformed records.
    """
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
    """Division guarded against zero / non-finite denominator."""
    if den == 0 or not np.isfinite(den):
        return 0.0
    out = num / den
    return out if np.isfinite(out) else 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_graph(path: str) -> nx.DiGraph:
    """Load a pickled networkx DiGraph from ``path``."""
    with open(path, "rb") as fh:
        return nx.DiGraph(pickle.load(fh))


def parse_label(path: str) -> np.ndarray:
    """Parse a label .txt file into the 6-target vector (raw, original space).

    The file is a python dict literal: {'train': '<7 fields>', 'infer': '<7>'}
    with fields '|'-separated as:
      time | average_sm_util | average_memory_util | average_memory_usuage |
      peak_sm_util | peak_memory_util | peak_memory_usuage
    """
    with open(path, "r") as fh:
        d = eval(fh.read())  # trusted local dataset format

    def fields(phase: str) -> List[float]:
        parts = str(d[phase]).split("|")
        return [_f(p) for p in parts]

    tr = fields("train")
    inf = fields("infer")
    # Field indices: 0=time, 1=avg_sm_util, ..., 6=peak_memory_usuage
    y = np.array(
        [
            tr[1],   # [0] train_util  = train.average_sm_util
            tr[6],   # [1] train_mem   = train.peak_memory_usuage
            tr[0],   # [2] train_time  = train.time
            inf[1],  # [3] infer_util  = infer.average_sm_util
            inf[6],  # [4] infer_mem   = infer.peak_memory_usuage
            inf[0],  # [5] infer_time  = infer.time
        ],
        dtype=np.float64,
    )
    return y


# ---------------------------------------------------------------------------
# Per-node / per-edge / global raw feature extraction (BEFORE standardization)
# ---------------------------------------------------------------------------

def _node_type_onehot(type_str: str) -> np.ndarray:
    """One-hot over NODE_TYPES; unknown type -> all zeros (robust)."""
    vec = np.zeros(NUM_NODE_TYPES, dtype=np.float64)
    if type_str in NODE_TYPES:
        vec[NODE_TYPES.index(type_str)] = 1.0
    return vec


def _node_raw(feature: dict, totals: Dict[str, float]) -> np.ndarray:
    """Build a single node's raw feature vector (log1p applied, NOT standardized).

    ``totals`` carries the model-level sums needed for the proportion block v^p.
    """
    args = feature.get("args", {}) or {}
    mem = feature.get("memory_info", {}) or {}

    flops = _f(feature.get("flops"))
    mac = _f(mem.get("bytes"))            # MAC proxy = total bytes touched
    weight = _f(mem.get("weight_size"))
    arith = _f(feature.get("arith_intensity"))

    out = np.zeros(NODE_DIM, dtype=np.float64)
    # v^hp: type one-hot
    out[_OFF_TYPE:_OFF_TYPE + NUM_NODE_TYPES] = _node_type_onehot(
        str(feature.get("type", ""))
    )
    # v^hp: selected args (raw int values; standardized later)
    for k, key in enumerate(ARG_KEYS):
        out[_OFF_ARGS + k] = _f(args.get(key))
    # v^c: log1p flops
    out[_OFF_C] = np.log1p(max(flops, 0.0))
    # v^m: log1p mac, log1p weight
    out[_OFF_M] = np.log1p(max(mac, 0.0))
    out[_OFF_M + 1] = np.log1p(max(weight, 0.0))
    # v^a: arith intensity (already a ratio; left as-is, standardized later)
    out[_OFF_A] = arith
    # v^p: proportions w.r.t. model totals (already in [0, 1], NOT standardized)
    out[_OFF_P] = _safe_div(flops, totals["flops"])
    out[_OFF_P + 1] = _safe_div(mac, totals["mac"])
    out[_OFF_P + 2] = _safe_div(weight, totals["weight"])
    return out


def _edge_raw(src_feature: dict) -> np.ndarray:
    """Build a single edge's raw feature vector from the SOURCE node memory_info.

    Edges carry no attributes in the dataset, so the edge tensor (the activation
    flowing along the edge) is the source node's OUTPUT tensor.
    """
    mem = src_feature.get("memory_info", {}) or {}
    out_size = _f(mem.get("output_size"))
    bs = _f(mem.get("batch_size"))
    oc = _f(mem.get("output_channels"))
    ow = _f(mem.get("output_w"))
    oh = _f(mem.get("output_h"))

    out = np.zeros(EDGE_DIM, dtype=np.float64)
    out[0] = np.log1p(max(out_size, 0.0))        # e^sz
    out[1] = np.log1p(max(bs, 0.0))              # e^sp batch
    out[2] = np.log1p(max(oc, 0.0))              # e^sp channels
    out[3] = np.log1p(max(oh, 0.0))              # e^sp height
    out[4] = np.log1p(max(ow, 0.0))              # e^sp width
    return out


def _stats4(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (total, mean, median, max) of a 1-D array; zeros if empty."""
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.sum(arr)),
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.max(arr)),
    )


def _extract_raw(g: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw (un-standardized) node feats, edge_index, edge feats, global feats.

    Returns:
        x_raw    : [N, NODE_DIM]
        edge_idx : [2, E] int64 (PyG COO, [source; target])
        e_raw    : [E, EDGE_DIM]
        u_raw    : [GLOBAL_DIM]
    """
    nodes = list(g.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Per-node raw scalars (pre-totals) for flops / mac / weight / arith.
    flops_arr = np.zeros(n, dtype=np.float64)
    mac_arr = np.zeros(n, dtype=np.float64)
    weight_arr = np.zeros(n, dtype=np.float64)
    features: List[dict] = []
    batch_size = 0.0
    for i, node in enumerate(nodes):
        feat = g.nodes[node].get("feature", {}) or {}
        features.append(feat)
        mem = feat.get("memory_info", {}) or {}
        flops_arr[i] = max(_f(feat.get("flops")), 0.0)
        mac_arr[i] = max(_f(mem.get("bytes")), 0.0)
        weight_arr[i] = max(_f(mem.get("weight_size")), 0.0)
        if batch_size == 0.0:
            batch_size = _f(mem.get("batch_size"))

    totals = {
        "flops": float(np.sum(flops_arr)),
        "mac": float(np.sum(mac_arr)),
        "weight": float(np.sum(weight_arr)),
    }

    # Node feature matrix.
    x_raw = np.zeros((n, NODE_DIM), dtype=np.float64)
    for i, feat in enumerate(features):
        x_raw[i] = _node_raw(feat, totals)

    # Edges -> COO index + derived edge features (from source node output).
    edges = list(g.edges())
    e = len(edges)
    edge_idx = np.zeros((2, e), dtype=np.int64)
    e_raw = np.zeros((e, EDGE_DIM), dtype=np.float64)
    edge_sizes = np.zeros(e, dtype=np.float64)  # raw source output_size for u^m
    for j, (s, t) in enumerate(edges):
        si, ti = node_index[s], node_index[t]
        edge_idx[0, j] = si
        edge_idx[1, j] = ti
        src_feat = features[si]
        e_raw[j] = _edge_raw(src_feat)
        edge_sizes[j] = max(_f((src_feat.get("memory_info", {}) or {}).get("output_size")), 0.0)

    # ----- Global features -----
    u = np.zeros(GLOBAL_DIM, dtype=np.float64)
    p = 0
    # u^gp: num_nodes, num_edges, density
    density = _safe_div(float(e), float(n) * (float(n) - 1.0)) if n > 1 else 0.0
    u[p] = float(n); p += 1
    u[p] = float(e); p += 1
    u[p] = density; p += 1
    # u^c: log1p of (total, mean, median, max) flops
    for v in _stats4(flops_arr):
        u[p] = np.log1p(max(v, 0.0)); p += 1
    # u^m: MAC stats (4), weight stats (4), mean edge tensor size (1)
    for v in _stats4(mac_arr):
        u[p] = np.log1p(max(v, 0.0)); p += 1
    for v in _stats4(weight_arr):
        u[p] = np.log1p(max(v, 0.0)); p += 1
    mean_edge = float(np.mean(edge_sizes)) if edge_sizes.size else 0.0
    u[p] = np.log1p(max(mean_edge, 0.0)); p += 1
    # u^a: model arith intensity = total_flops / total_mac
    u[p] = _safe_div(totals["flops"], totals["mac"]); p += 1
    # u^b: batch size
    u[p] = batch_size; p += 1
    assert p == GLOBAL_DIM, f"global feature length mismatch: {p} != {GLOBAL_DIM}"

    return x_raw, edge_idx, e_raw, u


# ---------------------------------------------------------------------------
# Normalization statistics (computed over the TRAIN split only)
# ---------------------------------------------------------------------------

def _welford_init(dim: int) -> Dict[str, np.ndarray]:
    """Streaming mean/variance accumulator (Welford) over ``dim`` channels."""
    return {
        "n": np.zeros(dim, dtype=np.float64),
        "mean": np.zeros(dim, dtype=np.float64),
        "m2": np.zeros(dim, dtype=np.float64),
    }


def _welford_update(acc: Dict[str, np.ndarray], batch: np.ndarray) -> None:
    """Update accumulator with a batch [K, dim] of observations."""
    if batch.size == 0:
        return
    for row in np.atleast_2d(batch):
        acc["n"] += 1.0
        delta = row - acc["mean"]
        acc["mean"] += delta / acc["n"]
        acc["m2"] += delta * (row - acc["mean"])


def _welford_finalize(acc: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) with std floored to avoid divide-by-zero."""
    n = np.maximum(acc["n"], 1.0)
    mean = acc["mean"].copy()
    var = acc["m2"] / np.maximum(n - 1.0, 1.0)
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-8] = 1.0  # constant channels -> no scaling
    return mean, std


# ---------------------------------------------------------------------------
# Parallel helpers (multiprocessing across CPU cores)
# ---------------------------------------------------------------------------

def _num_procs() -> int:
    """Worker count for parallel parsing (env PERFSEER_NUM_PROC overrides)."""
    env = os.environ.get("PERFSEER_NUM_PROC")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _chunkify(seq: Sequence, n_chunks: int) -> List[list]:
    """Split a sequence into ~n_chunks roughly equal contiguous chunks."""
    seq = list(seq)
    if not seq:
        return []
    n_chunks = max(1, min(n_chunks, len(seq)))
    k = (len(seq) + n_chunks - 1) // n_chunks
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def _stats_chunk_worker(pairs: List[Tuple[str, str]]):
    """Accumulate (count, sum, sumsq) per channel over a chunk of files.

    Values are already log1p-bounded (~[0,30]) so float64 sum/sumsq is numerically
    safe; partials merge by simple addition (exact, order-independent).
    """
    acc = {
        "node": [0.0, np.zeros(NODE_DIM), np.zeros(NODE_DIM)],
        "edge": [0.0, np.zeros(EDGE_DIM), np.zeros(EDGE_DIM)],
        "global": [0.0, np.zeros(GLOBAL_DIM), np.zeros(GLOBAL_DIM)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    for gp, lp in pairs:
        g = parse_graph(gp)
        x_raw, _ei, e_raw, u_raw = _extract_raw(g)
        if x_raw.shape[0] > 0:
            acc["node"][0] += x_raw.shape[0]
            acc["node"][1] += x_raw.sum(0)
            acc["node"][2] += (x_raw * x_raw).sum(0)
        if e_raw.shape[0] > 0:
            acc["edge"][0] += e_raw.shape[0]
            acc["edge"][1] += e_raw.sum(0)
            acc["edge"][2] += (e_raw * e_raw).sum(0)
        u = u_raw.astype(np.float64)
        acc["global"][0] += 1.0
        acc["global"][1] += u
        acc["global"][2] += u * u
        y_log = np.log1p(np.maximum(parse_label(lp), 0.0)).astype(np.float64)
        acc["y"][0] += 1.0
        acc["y"][1] += y_log
        acc["y"][2] += y_log * y_log
    return acc


def _mean_std_from_acc(count: float, s: np.ndarray, ss: np.ndarray):
    """Finalize (mean, std) from accumulated count/sum/sumsq."""
    if count <= 0:
        return np.zeros_like(s), np.ones_like(s)
    mean = s / count
    var = np.maximum(ss / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean, std


def _build_chunk_worker(arg):
    """Build PyG Data objects for a chunk of (graph, label) pairs.

    Returns a pickled-bytes blob (standard pickle) rather than live Data objects.
    Passing torch tensors back through a multiprocessing.Pool triggers torch's
    file-descriptor sharing reducer, which fails with "received 0 items of
    ancdata" under load. Serializing to plain bytes here sidesteps that path.
    """
    pairs, norm_stats = arg
    out: List[Data] = []
    for gp, lp in pairs:
        try:
            g = parse_graph(gp)
            y6 = parse_label(lp)
            out.append(build_pyg_data(g, y6, norm_stats))
        except Exception as exc:  # robust: skip malformed records, keep going
            print(f"[PerfSeerDataset] skipping {gp}: {exc}")
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


def compute_norm_stats(train_files: Sequence[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """Compute per-channel (mean, std) over the TRAIN files.

    Args:
        train_files: list of (graph_path, label_path) tuples for the train split.

    Returns a dict with keys:
        node_mean, node_std   : [NODE_DIM]   (only NODE_STD_IDX are meaningful;
                                               other channels get mean=0,std=1)
        edge_mean, edge_std   : [EDGE_DIM]
        global_mean, global_std : [GLOBAL_DIM]
        y_mean, y_std         : [NUM_TARGETS] (statistics of log1p(targets))
    Standardization is applied ONLY to the *_STD_IDX channels and to targets.
    """
    # Parallel pass: each worker accumulates (count, sum, sumsq) over a chunk;
    # partials merge by addition (exact, order-independent).
    pairs = list(train_files)
    total = {
        "node": [0.0, np.zeros(NODE_DIM), np.zeros(NODE_DIM)],
        "edge": [0.0, np.zeros(EDGE_DIM), np.zeros(EDGE_DIM)],
        "global": [0.0, np.zeros(GLOBAL_DIM), np.zeros(GLOBAL_DIM)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    nproc = min(_num_procs(), max(1, len(pairs)))
    chunks = _chunkify(pairs, nproc * 4)  # more chunks than procs for load balance
    if nproc > 1 and len(chunks) > 1:
        with get_context("fork").Pool(nproc) as pool:
            partials = pool.map(_stats_chunk_worker, chunks)
    else:
        partials = [_stats_chunk_worker(c) for c in chunks]
    for part in partials:
        for key in total:
            total[key][0] += part[key][0]
            total[key][1] += part[key][1]
            total[key][2] += part[key][2]

    node_mean, node_std = _mean_std_from_acc(*total["node"])
    edge_mean, edge_std = _mean_std_from_acc(*total["edge"])
    global_mean, global_std = _mean_std_from_acc(*total["global"])
    y_mean, y_std = _mean_std_from_acc(*total["y"])

    # Mask out non-standardized channels: identity transform (mean 0, std 1).
    node_mean_m = np.zeros(NODE_DIM); node_std_m = np.ones(NODE_DIM)
    for i in NODE_STD_IDX:
        node_mean_m[i] = node_mean[i]; node_std_m[i] = node_std[i]
    edge_mean_m = np.zeros(EDGE_DIM); edge_std_m = np.ones(EDGE_DIM)
    for i in EDGE_STD_IDX:
        edge_mean_m[i] = edge_mean[i]; edge_std_m[i] = edge_std[i]
    global_mean_m = np.zeros(GLOBAL_DIM); global_std_m = np.ones(GLOBAL_DIM)
    for i in GLOBAL_STD_IDX:
        global_mean_m[i] = global_mean[i]; global_std_m[i] = global_std[i]

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


# ---------------------------------------------------------------------------
# Target (de)standardization helpers (used by train.py / eval.py)
# ---------------------------------------------------------------------------

def standardize_targets(y_raw, stats: Dict[str, np.ndarray]):
    """Map raw targets -> standardized log1p space. Accepts numpy or torch."""
    is_torch = hasattr(y_raw, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_raw.dtype, device=y_raw.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_raw.dtype, device=y_raw.device)
        return (torch.log1p(torch.clamp(y_raw, min=0.0)) - ym) / ys
    y = np.asarray(y_raw, dtype=np.float64)
    return (np.log1p(np.maximum(y, 0.0)) - stats["y_mean"]) / stats["y_std"]


def invert_targets(y_std, stats: Dict[str, np.ndarray]):
    """Map standardized log1p predictions -> ORIGINAL metric space.

    Inverse of ``standardize_targets``: y = expm1(y_std * std + mean).
    """
    is_torch = hasattr(y_std, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_std.dtype, device=y_std.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_std.dtype, device=y_std.device)
        return torch.expm1(y_std * ys + ym)
    y = np.asarray(y_std, dtype=np.float64)
    return np.expm1(y * np.asarray(stats["y_std"]) + np.asarray(stats["y_mean"]))


# ---------------------------------------------------------------------------
# Build a single PyG Data object
# ---------------------------------------------------------------------------

def build_pyg_data(
    g: nx.DiGraph,
    label6: Sequence[float],
    norm_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Data:
    """Convert a networkx DiGraph + 6 raw targets into a PyG Data object.

    If ``norm_stats`` is provided, node/edge/global features and targets are
    standardized (log1p was already applied to magnitudes during extraction;
    targets are log1p'd here). If None, RAW (log1p-but-unstandardized) features
    are returned (useful for computing stats / debugging).

    Returned Data fields:
        x         : [N, NODE_DIM]      float32
        edge_index: [2, E]             int64
        edge_attr : [E, EDGE_DIM]      float32
        u         : [1, GLOBAL_DIM]    float32  (graph-level / global feature)
        y         : [1, NUM_TARGETS]   float32  (standardized log space if stats given)
        num_nodes : int
    """
    x_raw, edge_idx, e_raw, u_raw = _extract_raw(g)
    y_raw = np.asarray(label6, dtype=np.float64).reshape(-1)
    assert y_raw.shape[0] == NUM_TARGETS, "label6 must have NUM_TARGETS entries"

    if norm_stats is not None:
        x = (x_raw - norm_stats["node_mean"]) / norm_stats["node_std"]
        e = (e_raw - norm_stats["edge_mean"]) / norm_stats["edge_std"]
        u = (u_raw - norm_stats["global_mean"]) / norm_stats["global_std"]
        y = (np.log1p(np.maximum(y_raw, 0.0)) - norm_stats["y_mean"]) / norm_stats["y_std"]
    else:
        x, e, u, y = x_raw, e_raw, u_raw, np.log1p(np.maximum(y_raw, 0.0))

    # Sanitize any residual non-finite values (defensive).
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    data = PerfSeerData(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_idx.astype(np.int64)),
        edge_attr=torch.from_numpy(e.astype(np.float32)),
        u=torch.from_numpy(u.astype(np.float32)).view(1, GLOBAL_DIM),
        y=torch.from_numpy(y.astype(np.float32)).view(1, NUM_TARGETS),
    )
    data.num_nodes = int(x_raw.shape[0])
    return data


# ---------------------------------------------------------------------------
# File discovery and splitting
# ---------------------------------------------------------------------------

def _resolve_dirs(data_root: str) -> Tuple[str, str]:
    """Resolve (cg_dir, label_dir) under ``data_root``, handling doubled paths."""
    cg_dir = None
    for sub in _CG_SUBDIRS:
        cand = os.path.join(data_root, sub)
        if os.path.isdir(cand):
            cg_dir = cand
            break
    label_dir = None
    for sub in _LABEL_SUBDIRS:
        cand = os.path.join(data_root, sub)
        if os.path.isdir(cand):
            label_dir = cand
            break
    if cg_dir is None or label_dir is None:
        raise FileNotFoundError(
            f"Could not locate cg/ and label/ under {data_root!r}. "
            f"Tried cg subdirs {_CG_SUBDIRS} and label subdirs {_LABEL_SUBDIRS}."
        )
    return cg_dir, label_dir


def list_pairs(data_root: str) -> List[Tuple[str, str]]:
    """Return sorted (graph_path, label_path) pairs whose stems match.

    A graph 'foo.pkl' is paired with label 'foo.txt'. Graphs lacking a matching
    label file are skipped (robust to partial OOM-truncated collection).
    """
    cg_dir, label_dir = _resolve_dirs(data_root)
    pairs: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(cg_dir)):
        if not fname.endswith(".pkl"):
            continue
        stem = fname[: -len(".pkl")]
        lp = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(lp):
            pairs.append((os.path.join(cg_dir, fname), lp))
    return pairs


def split_dataset(
    data_root: Optional[str] = None,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Random 2:1:1 train/val/test split of (graph, label) pairs.

    Args:
        data_root: dataset root containing cg/ and label/. If None, falls back
            to the ``PERFSEER_DATA_ROOT`` environment variable, then ``"dataset"``.
            This lets callers invoke ``split_dataset(seed=...)`` without repeating
            the root (train.py / eval.py rely on this); the root is then resolved
            consistently for both training and evaluation as long as the env var /
            cwd are the same.
        seed: RNG seed for a reproducible permutation.
        ratios: train/val/test fractions (default 2:1:1 == 0.5/0.25/0.25).

    Returns (train_pairs, val_pairs, test_pairs).
    """
    if data_root is None:
        data_root = os.environ.get("PERFSEER_DATA_ROOT", "dataset")
    pairs = list_pairs(data_root)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n = len(pairs)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    sel = lambda idx: [pairs[i] for i in idx]
    return sel(train_idx), sel(val_idx), sel(test_idx)


# ---------------------------------------------------------------------------
# PyG InMemoryDataset with on-disk caching
# ---------------------------------------------------------------------------

class PerfSeerDataset(InMemoryDataset):
    """In-memory PyG dataset that caches processed graphs to disk.

    The cache is keyed by ``split_name`` so train/val/test are stored separately
    under ``<data_root>/processed/perfseer_<split_name>.pt``. Normalization stats
    MUST be supplied (computed from the TRAIN split) so every split is transformed
    with the same train-set statistics.

    Two equivalent construction styles are accepted so the data / train / eval
    modules can interoperate:

      positional (data.py native):
        PerfSeerDataset(data_root, pairs, norm_stats, "train")
      keyword (train.py / eval.py contract):
        PerfSeerDataset(root=..., file_list=..., norm_stats=..., split="train")

    ``root`` aliases ``data_root``, ``file_list`` aliases ``pairs``, and
    ``split`` aliases ``split_name``. Each split is cached separately under
    ``<data_root>/processed/perfseer_<split>.pt``. Normalization stats MUST be
    supplied (computed from the TRAIN split) so every split is transformed with
    the same train-set statistics.

    Usage:
        train_pairs, val_pairs, test_pairs = split_dataset(root, seed)
        stats = compute_norm_stats(train_pairs)   # or load cached stats
        train_ds = PerfSeerDataset(root, train_pairs, stats, "train")
        val_ds   = PerfSeerDataset(root, val_pairs,   stats, "val")
        test_ds  = PerfSeerDataset(root, test_pairs,  stats, "test")
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        split_name: Optional[str] = None,
        force_reprocess: bool = False,
        *,
        root: Optional[str] = None,
        file_list: Optional[Sequence[Tuple[str, str]]] = None,
        split: Optional[str] = None,
    ):
        # Resolve keyword aliases (train.py / eval.py contract) onto the native
        # positional parameters. Keyword values win if both are given.
        data_root = root if root is not None else data_root
        pairs = file_list if file_list is not None else pairs
        split_name = split if split is not None else split_name
        if data_root is None:
            raise ValueError("PerfSeerDataset requires 'data_root' (or 'root').")
        if pairs is None:
            raise ValueError("PerfSeerDataset requires 'pairs' (or 'file_list').")
        if norm_stats is None:
            raise ValueError("PerfSeerDataset requires 'norm_stats'.")
        if split_name is None:
            raise ValueError("PerfSeerDataset requires 'split_name' (or 'split').")
        self._pairs = list(pairs)
        self._norm_stats = norm_stats
        self._split_name = split_name
        self._data_root = data_root
        if force_reprocess:
            cache = os.path.join(data_root, "processed", self._cache_filename())
            if os.path.exists(cache):
                os.remove(cache)
        # InMemoryDataset expects root with raw/ and processed/ subdirs; we use
        # data_root as root so the processed cache lives at <data_root>/processed.
        super().__init__(root=data_root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    def _cache_filename(self) -> str:
        # Key the cache on split name AND a short hash of (pair stems + norm_stats),
        # so changing the seed/split membership or normalization invalidates the
        # cache instead of silently reusing graphs standardized with old stats.
        import hashlib

        h = hashlib.sha1()
        for gp, lp in self._pairs:
            h.update(os.path.basename(gp).encode())
        for key in sorted(self._norm_stats):
            h.update(key.encode())
            h.update(np.asarray(self._norm_stats[key], dtype=np.float64).tobytes())
        return f"perfseer_{self._split_name}_{h.hexdigest()[:12]}.pt"

    @property
    def raw_file_names(self) -> List[str]:
        # Raw files are managed externally (cg/, label/); nothing to download.
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [self._cache_filename()]

    def download(self) -> None:  # pragma: no cover - dataset is pre-downloaded
        pass

    def process(self) -> None:
        # Parallel build across CPU cores; chunks carry the (immutable) norm_stats.
        pairs = list(self._pairs)
        nproc = min(_num_procs(), max(1, len(pairs)))
        chunks = _chunkify(pairs, nproc * 4)
        args = [(c, self._norm_stats) for c in chunks]
        try:
            if nproc > 1 and len(chunks) > 1:
                with get_context("fork").Pool(nproc) as pool:
                    results = pool.map(_build_chunk_worker, args)
            else:
                results = [_build_chunk_worker(a) for a in args]
        except Exception as exc:  # fall back to single-process build on any IPC failure
            print(f"[PerfSeerDataset] parallel build failed ({exc}); falling back to serial")
            results = [_build_chunk_worker(a) for a in args]
        # Workers return pickled-bytes blobs; reconstitute Data objects here.
        data_list: List[Data] = [d for blob in results for d in pickle.loads(blob)]
        self.save(data_list, self.processed_paths[0])


# ---------------------------------------------------------------------------
# Norm-stats (de)serialization helpers
# ---------------------------------------------------------------------------

def save_norm_stats(stats: Dict[str, np.ndarray], path: str) -> None:
    """Persist norm stats to ``path`` (.npz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_norm_stats(path: str) -> Dict[str, np.ndarray]:
    """Load norm stats previously saved with ``save_norm_stats``."""
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}
