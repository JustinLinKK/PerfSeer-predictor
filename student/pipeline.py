"""Self-contained featurization + dataset for the PerfSeer teacher predictor.

No dependency on the (unavailable) `perfseer` package: graphs are rebuilt from the
NODE_SPECS embedded in each reverted calib model via predictor/converter.py, labels
are parsed directly from the RTX-6000 Results label files.
"""
from __future__ import annotations
import ast, glob, math, os, re, sys
from pathlib import Path
import numpy as np
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predictor"))
from converter import convert_generated_source_to_networkx  # noqa: E402

MODELS_DIR = Path(os.environ.get("SEER_MODELS_DIR", ROOT / "dataset/labels/RTX-6000/pack/models"))
LABELS_DIR = Path(os.environ.get("SEER_LABELS_DIR", ROOT / "dataset/labels/RTX-6000/Results/label"))

OP_VOCAB = ["Add", "Attention", "Concat", "Conv", "DepthwiseConv", "DetectorHead",
            "Embedding", "Flatten", "GRU", "Gelu", "Gemm", "GlobalAveragePool",
            "GraphAttention", "GraphMessage", "LSTM", "LayerNormalization", "MaxPool",
            "Relu", "SegmentationHead", "Silu", "Softmax", "TabularFeature", "Upsample"]
OP_INDEX = {op: i for i, op in enumerate(OP_VOCAB)}
N_OP = len(OP_VOCAB)

ARG_KEYS = ("conv_kernel_size", "conv_stride", "conv_padding", "conv_dilation",
            "conv_groups", "conv_bias", "linear_in_features", "linear_out_features",
            "linear_bias", "pool_kernel_size", "pool_stride", "pool_padding",
            "pool_ceil_mode")

PRECISIONS = ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp"]
PREC_INDEX = {p: i for i, p in enumerate(PRECISIONS)}

TARGET_NAMES = ["train_util", "train_mem", "train_time", "infer_util", "infer_mem", "infer_time"]


def _l1p(x):
    return math.log1p(max(float(x), 0.0))


# ---- node continuous feature layout ----
_MEM_KEYS = ["weight_size", "input_size", "output_size", "bytes", "input_channels",
             "output_channels", "input_features", "output_features", "spatial_area",
             "sequence_length"]


def featurize_graph(g: nx.DiGraph):
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # topology
    indeg = np.array([g.in_degree(nd) for nd in nodes], dtype=np.float64)
    outdeg = np.array([g.out_degree(nd) for nd in nodes], dtype=np.float64)
    try:
        topo = list(nx.topological_sort(g))
    except Exception:
        topo = list(nodes)
    topo_pos = {nd: i for i, nd in enumerate(topo)}
    # longest-path depth from sources (iterate in topological order)
    depth = {nd: 0 for nd in nodes}
    for nd in topo:
        for p in g.predecessors(nd):
            depth[nd] = max(depth[nd], depth[p] + 1)
    maxdepth = max(depth.values()) if depth else 1

    x_onehot = np.zeros((n, N_OP), dtype=np.float32)
    cont = []
    flops_all, weight_all, outsz_all, bytes_all, rank_all = [], [], [], [], []
    for nd in nodes:
        f = g.nodes[nd]["feature"]
        t = f.get("type"); mem = f.get("memory_info", {}); args = f.get("args", {})
        if t in OP_INDEX:
            x_onehot[idx[nd], OP_INDEX[t]] = 1.0
        row = [float(args.get(k, 0) or 0) for k in ARG_KEYS]
        row.append(_l1p(f.get("flops", 0)))
        row.append(float(f.get("arith_intensity", 0.0) or 0.0))
        for k in _MEM_KEYS:
            row.append(_l1p(mem.get(k, 0)))
        row.append(float(mem.get("rank", 0) or 0))
        i = idx[nd]
        row += [indeg[i], outdeg[i], topo_pos.get(nd, i) / max(n - 1, 1), depth[nd] / max(maxdepth, 1)]
        cont.append(row)
        flops_all.append(float(f.get("flops", 0) or 0)); weight_all.append(float(mem.get("weight_size", 0) or 0))
        outsz_all.append(float(mem.get("output_size", 0) or 0)); bytes_all.append(float(mem.get("bytes", 0) or 0))
        rank_all.append(float(mem.get("rank", 0) or 0))
    x_cont = np.asarray(cont, dtype=np.float32)

    # edges
    ei = np.zeros((2, g.number_of_edges()), dtype=np.int64)
    e_cont = []
    for j, (u, v) in enumerate(g.edges()):
        ei[0, j] = idx[u]; ei[1, j] = idx[v]
        su = g.nodes[u]["feature"]["memory_info"]; sv = g.nodes[v]["feature"]["memory_info"]
        e_cont.append([_l1p(su.get("output_size", 0)), _l1p(sv.get("input_size", 0)), _l1p(su.get("bytes", 0))])
    if not e_cont:  # guard: no edges (single-node)
        e_cont = np.zeros((0, 3), dtype=np.float32)
    e_cont = np.asarray(e_cont, dtype=np.float32).reshape(-1, 3)

    # op-type histogram (batch-independent op mix) + structural stats
    op_hist = np.zeros(N_OP, dtype=np.float32)
    for nd in nodes:
        t = g.nodes[nd]["feature"].get("type")
        if t in OP_INDEX:
            op_hist[OP_INDEX[t]] += 1.0
    branch = float(sum(1 for d in outdeg if d > 1))
    join = float(sum(1 for d in indeg if d > 1))
    # global-static
    g_base = np.asarray([
        _l1p(n), _l1p(g.number_of_edges()),
        g.number_of_edges() / max(n * (n - 1), 1),
        _l1p(sum(flops_all)), _l1p(np.mean(flops_all) if flops_all else 0), _l1p(max(flops_all) if flops_all else 0),
        _l1p(sum(weight_all)), _l1p(sum(outsz_all)), _l1p(max(bytes_all) if bytes_all else 0),
        max(rank_all) if rank_all else 0.0,
        _l1p(branch), _l1p(join), float(maxdepth),
    ], dtype=np.float32)
    g_cont = np.concatenate([g_base, np.log1p(op_hist)]).astype(np.float32)
    return x_onehot, x_cont, ei, e_cont, g_cont


def parse_label6(path):
    d = ast.literal_eval(Path(path).read_text())
    T = [float(x) for x in str(d["train"]).split("|")]
    I = [float(x) for x in str(d["infer"]).split("|")]
    # per phase: util=field1, mem=field3, time=field0
    return np.asarray([T[1], T[3], T[0], I[1], I[3], I[0]], dtype=np.float64)


def label_files():
    return sorted(glob.glob(str(LABELS_DIR / "calib_*.txt")))


def model_id_and_precision(label_path):
    b = os.path.basename(label_path)
    m = re.match(r"(calib_\d+)_", b)
    mid = m.group(1)
    pm = re.search(r"_(fp32_ieee|tf32|bf16_amp|fp16_amp)_", b)
    return mid, pm.group(1)


if __name__ == "__main__":
    # smoke: featurize one graph
    f = MODELS_DIR / "calib_0000.py"
    g = convert_generated_source_to_networkx(str(f))
    xo, xc, ei, ec, gc = featurize_graph(g)
    print("nodes", xo.shape[0], "onehot", xo.shape[1], "cont", xc.shape[1], "edges", ei.shape[1], "e_cont", ec.shape[1], "g_cont", gc.shape[0])
    lbls = label_files()
    print("label files", len(lbls))
    print("sample label6", parse_label6(lbls[0]), model_id_and_precision(lbls[0]))
