"""Build teacher cache from A10 label_v3 JSONL shards (mirrors build_cache.py).

Array shapes (n = nodes in one graph, e = edges in one graph, s = samples, t = 6 targets):
  x_onehot(n, N_OP)   one-hot op type per node
  x_cont(n, c)        continuous node features, c = 30
  ei(2, e)            edge index (src row 0, dst row 1)
  e_cont(e, 3)        continuous edge features
  g_cont(d,)          global graph features, d = 10 + N_OP
  label6(t,)          [train_util %, train_mem MiB, train_time ms, infer_util %, infer_mem MiB, infer_time ms]
  y(s, t)             stacked train labels for normalization stats
"""
from __future__ import annotations
import json, glob, pickle, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P
from converter import convert_generated_source_to_networkx

ROOT = Path(__file__).resolve().parents[1]
A10_DIR = ROOT / "dataset/labels/A10/extracted"
MODELS_DIR = A10_DIR / "models"
OUT = Path(__file__).resolve().parent / "cache_a10"
OUT.mkdir(exist_ok=True)
VAL_FRAC = 0.1
SEED = 42

TARGET_KEYS = ["train_avg_sm_util_percent", "train_peak_vram_mib", "train_step_wall_ms",
               "infer_avg_sm_util_percent", "infer_peak_vram_mib", "infer_step_wall_ms"]


def main():
    samples = []
    mids = set()
    for shard in sorted(glob.glob(str(A10_DIR / "label_v3_shard*.jsonl"))):
        for line in open(shard):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            label6 = np.asarray([float(r["targets"][k]) for k in TARGET_KEYS], dtype=np.float64)
            samples.append((r["model_id"], r["precision_config"], label6))
            mids.add(r["model_id"])
    mids = sorted(mids)
    print(f"labels={len(samples)} models={len(mids)}", flush=True)

    graphs = {}
    t0 = time.time()
    for i, mid in enumerate(mids):
        g = convert_generated_source_to_networkx(str(MODELS_DIR / f"{mid}.py"))
        graphs[mid] = P.featurize_graph(g)
        if (i + 1) % 1000 == 0 or i + 1 == len(mids):
            print(f"featurized {i+1}/{len(mids)} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(mids))
    n_val = int(len(mids) * VAL_FRAC)
    val_mids = set(mids[j] for j in perm[:n_val])
    train_idx = [k for k, s in enumerate(samples) if s[0] not in val_mids]
    val_idx = [k for k, s in enumerate(samples) if s[0] in val_mids]
    print(f"train_samples={len(train_idx)} val_samples={len(val_idx)} val_models={len(val_mids)}", flush=True)

    train_graph_mids = sorted({samples[k][0] for k in train_idx})
    xc = np.concatenate([graphs[m][1] for m in train_graph_mids], axis=0)
    ec = np.concatenate([graphs[m][3] for m in train_graph_mids if graphs[m][3].shape[0] > 0], axis=0)
    gc = np.stack([graphs[m][4] for m in train_graph_mids], axis=0)
    stats = {
        "x_mean": xc.mean(0), "x_std": xc.std(0) + 1e-6,
        "e_mean": ec.mean(0), "e_std": ec.std(0) + 1e-6,
        "g_mean": gc.mean(0), "g_std": gc.std(0) + 1e-6,
    }
    ytr = np.stack([samples[k][2] for k in train_idx], axis=0)
    ylog = np.log1p(np.maximum(ytr, 0.0))
    stats["y_mean"] = ylog.mean(0)
    stats["y_std"] = ylog.std(0) + 1e-6
    stats["node_dim"] = P.N_OP + xc.shape[1]
    stats["edge_dim"] = 3
    stats["global_dim"] = gc.shape[1] + len(P.PRECISIONS)

    with open(OUT / "graphs.pkl", "wb") as f:
        pickle.dump(graphs, f, protocol=4)
    with open(OUT / "meta.pkl", "wb") as f:
        pickle.dump({"samples": samples, "train_idx": train_idx, "val_idx": val_idx,
                     "val_mids": sorted(val_mids), "stats": stats}, f, protocol=4)
    print("saved cache. dims:", {k: stats[k] for k in ("node_dim", "edge_dim", "global_dim")}, flush=True)
    print("y_mean(log1p)", np.round(stats["y_mean"], 3), "y_std", np.round(stats["y_std"], 3), flush=True)


if __name__ == "__main__":
    main()
