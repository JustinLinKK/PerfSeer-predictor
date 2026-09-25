"""Encode a pure PyTorch source file (nn.Module) into SeerNet predictor input.

Bridges converter.convert_source_to_networkx (torch.fx trace + shape propagation)
to the teacher/student featurization (teacher/pipeline.py), producing the exact
batch object SeerNetMulti consumes, normalized with the stats stored in a
trained checkpoint.

Array shapes (n = graph nodes, e = graph edges, o = N_OP op vocab, t = 6 targets):
  x(n, 53)          node features: o one-hot + 30 normalized continuous
  edge_index(2, e)  directed edges (row 0 = src, row 1 = dst)
  edge_attr(e, 3)   normalized edge features
  u(1, 40)          normalized global features + 4-dim precision one-hot
  batch(n,)         graph id per node (all zeros: single graph)
  pred(t,)          de-normalized predictions
    [train_util %, train_mem MiB, train_time, infer_util %, infer_mem MiB, infer_time]

Usage:
  from encoder import encode, predict
  d = encode("my_model.py", "MyNet", [[8, 3, 224, 224]], "fp32_ieee", stats)
  pred = predict("my_model.py", "MyNet", [[8, 3, 224, 224]], "fp32_ieee",
                 ckpt_path="student/student_A10.pt")

CLI:
  python student/encoder.py my_model.py --entry MyNet \
      --input-shapes 8,3,224,224 --precision fp32_ieee \
      --ckpt student/student_A10.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter import SourceModelSpec, convert_source_to_networkx  # noqa: E402
import pipeline as P  # noqa: E402


class Batch:
    """Duck-typed stand-in for the PyG batch object SeerNetMulti expects."""

    def __init__(self, x, edge_index, edge_attr, u, batch, num_graphs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.u = u
        self.batch = batch
        self.num_graphs = num_graphs


def encode(source_path, entry, input_shapes, precision, stats,
           constructor_args=(), constructor_kwargs=None, input_dtypes=("float32",),
           device="cpu"):
    """Trace a PyTorch source file and return the normalized SeerNet input batch."""
    if precision not in P.PREC_INDEX:
        raise ValueError(f"precision {precision!r} not in {P.PRECISIONS}")
    spec = SourceModelSpec(
        source_path=source_path,
        entry=entry,
        input_shapes=input_shapes,
        constructor_args=tuple(constructor_args),
        constructor_kwargs=dict(constructor_kwargs or {}),
        input_dtypes=tuple(input_dtypes),
    )
    graph = convert_source_to_networkx(spec)
    xo, xc, ei, ec, gc = P.featurize_graph(graph)

    x = np.concatenate([xo, (xc - stats["x_mean"]) / stats["x_std"]], axis=1).astype(np.float32)
    e = ((ec - stats["e_mean"]) / stats["e_std"]).astype(np.float32) if ec.shape[0] else np.zeros((0, 3), np.float32)
    g = ((gc - stats["g_mean"]) / stats["g_std"]).astype(np.float32)
    prec_oh = np.zeros(len(P.PRECISIONS), dtype=np.float32)
    prec_oh[P.PREC_INDEX[precision]] = 1.0
    u = np.concatenate([g, prec_oh])[None, :]

    dev = torch.device(device)
    return Batch(
        x=torch.from_numpy(x).to(dev),
        edge_index=torch.from_numpy(ei).to(dev),
        edge_attr=torch.from_numpy(e).to(dev),
        u=torch.from_numpy(u).to(dev),
        batch=torch.zeros(x.shape[0], dtype=torch.long, device=dev),
        num_graphs=1,
    )


def load_predictor(ckpt_path, device="cpu"):
    """Load a trained SeerNetMulti checkpoint; returns (net, stats)."""
    from model import SeerNetMulti, SeerNetConfig

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = SeerNetMulti(SeerNetConfig(**ck["cfg"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck["stats"]


def predict(source_path, entry, input_shapes, precision, ckpt_path,
            constructor_args=(), constructor_kwargs=None, input_dtypes=("float32",),
            device="cpu"):
    """End to end: source file -> 6 de-normalized target predictions."""
    net, stats = load_predictor(ckpt_path, device=device)
    d = encode(source_path, entry, input_shapes, precision, stats,
               constructor_args=constructor_args, constructor_kwargs=constructor_kwargs,
               input_dtypes=input_dtypes, device=device)
    with torch.no_grad():
        pred_std = net(d).cpu().numpy()[0]
    ylog = pred_std * stats["y_std"] + stats["y_mean"]
    return np.maximum(np.expm1(ylog), 0.0)


def _parse_shapes(values):
    return [[int(x) for x in v.split(",")] for v in values]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="path to a pure PyTorch .py file")
    ap.add_argument("--entry", required=True,
                    help="nn.Module class, instance, or factory name inside the file")
    ap.add_argument("--input-shapes", nargs="+", required=True,
                    help="one or more comma-separated shapes, e.g. 8,3,224,224")
    ap.add_argument("--input-dtypes", nargs="+", default=["float32"])
    ap.add_argument("--precision", default="fp32_ieee", choices=P.PRECISIONS)
    ap.add_argument("--ckpt", default=str(ROOT / "student/student_A10.pt"),
                    help="trained checkpoint supplying norm stats (and weights for --predict)")
    ap.add_argument("--no-predict", action="store_true",
                    help="only encode and print input tensor shapes")
    ap.add_argument("--onnx", default=None,
                    help="run inference with an ONNX artifact (e.g. student/student_a10_cpu_int8.onnx); "
                         "--ckpt still supplies normalization stats")
    args = ap.parse_args()

    shapes = _parse_shapes(args.input_shapes)
    net, stats = load_predictor(args.ckpt)
    d = encode(args.source, args.entry, shapes, args.precision, stats,
               input_dtypes=tuple(args.input_dtypes))
    print(f"encoded: x{tuple(d.x.shape)} edge_index{tuple(d.edge_index.shape)} "
          f"edge_attr{tuple(d.edge_attr.shape)} u{tuple(d.u.shape)}")
    if args.no_predict:
        return
    if args.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
        feed = {"x": d.x.numpy(), "edge_index": d.edge_index.numpy(),
                "edge_attr": d.edge_attr.numpy(), "u": d.u.numpy(),
                "batch": d.batch.numpy()}
        pred_std = sess.run(None, feed)[0][0]
    else:
        with torch.no_grad():
            pred_std = net(d).cpu().numpy()[0]
    pred = np.maximum(np.expm1(pred_std * stats["y_std"] + stats["y_mean"]), 0.0)
    for name, val in zip(P.TARGET_NAMES, pred):
        print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()
