"""Consecutive converter -> predictor pipeline for the student model.

One object owns the whole chain: PyTorch source file -> torch.fx compute graph
(converter) -> featurization (pipeline) -> z-score normalization (training
mean/std) -> SeerNet student inference -> de-normalized 6-target prediction.

Normalization stats travel with the artifact: torch checkpoints store them in
ck["stats"]; ONNX artifacts carry them in metadata_props["seer_stats"], so a
single file is sufficient for deployment.

Array shapes (n = nodes, e = edges, o = op vocab 23, t = 6 targets):
  x(n, 53)          node features: o one-hot + 30 normalized continuous
  edge_index(2, e)  directed edges
  edge_attr(e, 3)   normalized edge features
  u(1, 40)          normalized global features + 4-dim precision one-hot
  pred(t,)          [train_util %, train_mem MiB, train_time, infer_util %,
                     infer_mem MiB, infer_time] (time in A10 label units, ms)

Usage:
  from predict import SeerPredictor
  p = SeerPredictor("student/student_a10_cpu_int8.onnx")   # or student_A10.pt
  out = p.predict_source("my_model.py", "MyNet", [[8, 3, 224, 224]], "fp32_ieee")

CLI:
  python student/predict.py my_model.py --entry MyNet \
      --input-shapes 8,3,224,224 --precision fp32_ieee \
      --artifact student/student_a10_cpu_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter import SourceModelSpec, convert_source_to_networkx  # noqa: E402
import pipeline as P  # noqa: E402


class _Batch:
    pass


class SeerPredictor:
    """End-to-end predictor bundling converter, featurizer, stats, and backend."""

    def __init__(self, artifact: str | Path, threads: int = 4):
        self.artifact = str(artifact)
        if self.artifact.endswith(".onnx"):
            import onnx
            import onnxruntime as ort

            meta = {p.key: p.value for p in onnx.load(self.artifact).metadata_props}
            if "seer_stats" not in meta:
                raise ValueError(f"{artifact} has no seer_stats metadata; re-embed training mean/std")
            raw = json.loads(meta["seer_stats"])
            self.target_names = raw.pop("target_names")
            self.stats = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
                          for k, v in raw.items()}
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(self.artifact, so, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
        else:
            from model import SeerNetMulti, SeerNetConfig

            torch.set_num_threads(threads)
            ck = torch.load(self.artifact, map_location="cpu", weights_only=False)
            self.stats = ck["stats"]
            self.target_names = ck.get("targets", P.TARGET_NAMES)
            self._net = SeerNetMulti(SeerNetConfig(**ck["cfg"]))
            self._net.load_state_dict(ck["model"])
            self._net.eval()
            self.backend = "torch"

    def encode_source(self, source_path, entry, input_shapes, precision,
                      constructor_args=(), constructor_kwargs=None, input_dtypes=("float32",)):
        """Converter + featurizer + normalization -> input arrays."""
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
        st = self.stats
        x = np.concatenate([xo, (xc - st["x_mean"]) / st["x_std"]], 1).astype(np.float32)
        e = ((ec - st["e_mean"]) / st["e_std"]).astype(np.float32) if ec.shape[0] else np.zeros((0, 3), np.float32)
        g = ((gc - st["g_mean"]) / st["g_std"]).astype(np.float32)
        oh = np.zeros(len(P.PRECISIONS), np.float32)
        oh[P.PREC_INDEX[precision]] = 1.0
        u = np.concatenate([g, oh]).astype(np.float32)[None, :]
        batch = np.zeros(x.shape[0], np.int64)
        return x, ei.astype(np.int64), e, u, batch

    def predict_arrays(self, x, edge_index, edge_attr, u, batch):
        """Backend inference + de-normalization."""
        if self.backend == "onnxruntime":
            pred_std = self._sess.run(None, {"x": x, "edge_index": edge_index,
                                             "edge_attr": edge_attr, "u": u, "batch": batch})[0][0]
        else:
            d = _Batch()
            d.x = torch.from_numpy(x)
            d.edge_index = torch.from_numpy(edge_index)
            d.edge_attr = torch.from_numpy(edge_attr)
            d.u = torch.from_numpy(u)
            d.batch = torch.from_numpy(batch)
            d.num_graphs = 1
            with torch.inference_mode():
                pred_std = self._net(d).numpy()[0]
        st = self.stats
        return np.maximum(np.expm1(pred_std * st["y_std"] + st["y_mean"]), 0.0)

    def predict_source(self, source_path, entry, input_shapes, precision, **kw):
        """Full consecutive pipeline: source file -> 6-target prediction dict."""
        arrays = self.encode_source(source_path, entry, input_shapes, precision, **kw)
        pred = self.predict_arrays(*arrays)
        return dict(zip(self.target_names, (float(v) for v in pred)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("--entry", required=True)
    ap.add_argument("--input-shapes", nargs="+", required=True)
    ap.add_argument("--input-dtypes", nargs="+", default=["float32"])
    ap.add_argument("--precision", default="fp32_ieee", choices=P.PRECISIONS)
    ap.add_argument("--artifact", default=str(ROOT / "student/student_a10_cpu_int8.onnx"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    p = SeerPredictor(args.artifact, threads=args.threads)
    shapes = [[int(v) for v in s.split(",")] for s in args.input_shapes]
    out = p.predict_source(args.source, args.entry, shapes, args.precision,
                           input_dtypes=tuple(args.input_dtypes))
    print(f"backend={p.backend} artifact={Path(args.artifact).name}")
    for k, v in out.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
