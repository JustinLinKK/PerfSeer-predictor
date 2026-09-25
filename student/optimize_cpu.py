"""Benchmark and export CPU-optimized inference variants of the student predictor.

Backends compared (all CPU): eager fp32, dynamic-int8 quantized Linear,
TorchScript trace (fp32 + int8), torch.compile, ONNX Runtime (fp32 + int8).
Each backend is scored on the full A10 validation split (10Acc per target)
and single-graph latency; the fastest accuracy-preserving artifact is saved.

Array shapes (n = nodes, e = edges, b = graphs in batch, t = 6 targets):
  x(n, 53)          normalized node features
  edge_index(2, e)  directed edges
  edge_attr(e, 3)   normalized edge features
  u(b, 40)          normalized global features + precision one-hot
  batch(n,)         graph id per node
  y(b, t)           raw targets; predictions compared in raw space

Usage:
  SEER_CACHE=$PWD/teacher/cache_a10 python student/optimize_cpu.py \
      --ckpt student/student_A10.pt --threads 4
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import SeerNetMulti, SeerNetConfig, count_parameters  # noqa: E402
import pipeline as P  # noqa: E402

sys.path.insert(0, str(ROOT / "predictor"))
from metrics import x_acc  # noqa: E402


class Batch:
    pass


class TensorWrapper(nn.Module):
    """Tensor-arg wrapper so the model can be traced / exported."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, u, batch):
        d = Batch()
        d.x = x
        d.edge_index = edge_index
        d.edge_attr = edge_attr
        d.u = u
        d.batch = batch
        d.num_graphs = int(u.shape[0])
        return self.model(d)


def load_cache(cache_dir: Path):
    graphs = pickle.load(open(cache_dir / "graphs.pkl", "rb"))
    meta = pickle.load(open(cache_dir / "meta.pkl", "rb"))
    return graphs, meta


def make_batch(idxs, samples, G, st):
    xs, eis, es, us, ys, bat = [], [], [], [], [], []
    off = 0
    for bi, k in enumerate(idxs):
        mid, prec, y6 = samples[k]
        xo, xc, ei, ec, gc = G[mid]
        x = np.concatenate([xo, (xc - st["x_mean"]) / st["x_std"]], 1).astype(np.float32)
        e = ((ec - st["e_mean"]) / st["e_std"]).astype(np.float32) if ec.shape[0] else np.zeros((0, 3), np.float32)
        g = ((gc - st["g_mean"]) / st["g_std"]).astype(np.float32)
        oh = np.zeros(len(P.PRECISIONS), np.float32)
        oh[P.PREC_INDEX[prec]] = 1.0
        xs.append(x)
        es.append(e)
        eis.append(ei + off)
        us.append(np.concatenate([g, oh]))
        bat.append(np.full(x.shape[0], bi, np.int64))
        ys.append(y6)
        off += x.shape[0]
    args = (
        torch.from_numpy(np.concatenate(xs)),
        torch.from_numpy(np.concatenate(eis, 1)),
        torch.from_numpy(np.concatenate(es)),
        torch.from_numpy(np.stack(us)),
        torch.from_numpy(np.concatenate(bat)),
    )
    return args, np.stack(ys)


def invert(pred_std, st):
    return np.maximum(np.expm1(pred_std * st["y_std"] + st["y_mean"]), 0.0)


def eval_backend(run_fn, val_batches, st):
    preds, trues = [], []
    for args, y in val_batches:
        preds.append(run_fn(args))
        trues.append(y)
    pred_raw = invert(np.concatenate(preds), st)
    y_raw = np.concatenate(trues)
    return np.array([x_acc(y_raw[:, j], pred_raw[:, j], 10.0) for j in range(6)])


def bench_latency(run_fn, args, iters=300, warmup=30):
    for _ in range(warmup):
        run_fn(args)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_fn(args)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(ts.mean()), float(np.percentile(ts, 95))


def torch_runner(module):
    def run(args):
        with torch.inference_mode():
            return module(*args).numpy()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "student/student_A10.pt"))
    ap.add_argument("--cache", default=os.environ.get("SEER_CACHE", str(ROOT / "teacher/cache_a10")))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512, help="validation eval batch size")
    ap.add_argument("--out-dir", default=str(ROOT / "student"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    st = ck["stats"]
    net = SeerNetMulti(SeerNetConfig(**ck["cfg"]))
    net.load_state_dict(ck["model"])
    net.eval()
    wrapper = TensorWrapper(net).eval()
    print(f"student params={count_parameters(net)} threads={args.threads}", flush=True)

    graphs, meta = load_cache(Path(args.cache))
    samples, va = meta["samples"], meta["val_idx"]
    val_batches = [make_batch(va[i:i + args.bs], samples, graphs, st) for i in range(0, len(va), args.bs)]
    # traced/exported artifacts specialize num_graphs to the example's batch size,
    # so they are single-graph deployables and must be validated at bs=1
    val_singles = [make_batch([k], samples, graphs, st) for k in va]
    single_args, _ = make_batch(va[:1], samples, graphs, st)
    print(f"val_samples={len(va)} eval_batches={len(val_batches)}", flush=True)

    out_dir = Path(args.out_dir)
    results = {}

    def record(name, run_fn, artifact=None, single_only=False):
        accs = eval_backend(run_fn, val_singles if single_only else val_batches, st)
        mean_ms, p95_ms = bench_latency(run_fn, single_args)
        size_mb = os.path.getsize(artifact) / 1e6 if artifact else os.path.getsize(args.ckpt) / 1e6
        results[name] = {"accs": accs, "min": float(accs.min()), "mean_ms": mean_ms,
                         "p95_ms": p95_ms, "artifact": artifact, "size_mb": size_mb}
        print(f"{name:22s} min10Acc={accs.min():.4f} acc=[{' '.join(f'{a:.3f}' for a in accs)}] "
              f"lat mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms size={size_mb:.1f}MB", flush=True)

    # 1) eager fp32 baseline
    record("eager_fp32", torch_runner(wrapper))

    # 2) dynamic int8 quantization of Linear layers
    qnet = torch.ao.quantization.quantize_dynamic(TensorWrapper(net).eval(), {nn.Linear}, dtype=torch.qint8)
    record("dynamic_int8", torch_runner(qnet))

    # 3) TorchScript trace fp32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_fp32 = torch.jit.trace(wrapper, single_args, strict=False)
        ts_fp32 = torch.jit.freeze(ts_fp32.eval())
    ts_path = str(out_dir / "student_a10_cpu_ts.pt")
    torch.jit.save(ts_fp32, ts_path)
    ts_fp32 = torch.jit.load(ts_path).eval()
    record("torchscript_fp32", torch_runner(ts_fp32), ts_path, single_only=True)

    # 4) TorchScript trace of the int8-quantized model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts_int8 = torch.jit.trace(qnet, single_args, strict=False)
            ts_int8 = torch.jit.freeze(ts_int8.eval())
        ts8_path = str(out_dir / "student_a10_cpu_ts_int8.pt")
        torch.jit.save(ts_int8, ts8_path)
        ts_int8 = torch.jit.load(ts8_path).eval()
        record("torchscript_int8", torch_runner(ts_int8), ts8_path, single_only=True)
    except Exception as exc:
        print(f"torchscript_int8 skipped: {exc}", flush=True)

    # 5) torch.compile (inductor, dynamic shapes)
    try:
        cnet = torch.compile(TensorWrapper(net).eval(), dynamic=True)
        run = torch_runner(cnet)
        run(single_args)
        record("torch_compile", run)
    except Exception as exc:
        print(f"torch_compile skipped: {exc}", flush=True)

    # 6) ONNX Runtime fp32 (+ dynamic int8)
    try:
        import onnxruntime as ort

        onnx_path = str(out_dir / "student_a10_cpu.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper, single_args, onnx_path, opset_version=18,
                input_names=["x", "edge_index", "edge_attr", "u", "batch"],
                output_names=["pred"],
                dynamic_axes={"x": {0: "n"}, "edge_index": {1: "e"}, "edge_attr": {0: "e"},
                              "u": {0: "b"}, "batch": {0: "n"}, "pred": {0: "b"}},
                dynamo=False,
            )

        def ort_runner(path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = args.threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            names = ["x", "edge_index", "edge_attr", "u", "batch"]

            def run(targs):
                feed = {k: v.numpy() for k, v in zip(names, targs)}
                return sess.run(None, feed)[0]
            return run

        record("onnxruntime_fp32", ort_runner(onnx_path), onnx_path, single_only=True)

        from onnxruntime.quantization import QuantType, quantize_dynamic

        onnx8_path = str(out_dir / "student_a10_cpu_int8.onnx")
        quantize_dynamic(onnx_path, onnx8_path, weight_type=QuantType.QInt8)
        record("onnxruntime_int8", ort_runner(onnx8_path), onnx8_path, single_only=True)
    except Exception as exc:
        print(f"onnxruntime skipped: {exc}", flush=True)

    # selection: fastest backend whose min10Acc is within 0.005 of eager fp32
    base = results["eager_fp32"]["min"]
    ok = {k: v for k, v in results.items() if v["min"] >= base - 0.005}
    best = min(ok, key=lambda k: ok[k]["mean_ms"])
    print(f"\nBEST={best} min10Acc={results[best]['min']:.4f} "
          f"mean={results[best]['mean_ms']:.3f}ms p95={results[best]['p95_ms']:.3f}ms "
          f"artifact={results[best]['artifact'] or args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
