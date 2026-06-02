"""Convert example source models and run PerfSeer predictions.

Usage:
  python examples/predict_source_models.py
  python examples/predict_source_models.py --ckpt runs/.../seernet_multi.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_optimized.calibration import LinearCalibrator
from perfseer_optimized.data import TARGET_NAMES
from perfseer_optimized.eval import invert_all, load_model, stats_from_metadata
from perfseer_source_converter import SourceModelSpec, convert_source_to_networkx, convert_source_to_pyg_data


DEFAULT_CKPT = ROOT / "runs/experiments/full_20260601_122347/checkpoints/deploy_distill_student_128/seernet_multi.pt"


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}")
    return shape


def discover_checkpoint() -> Path:
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    matches = sorted(glob.glob(str(ROOT / "runs/**/seernet_multi.pt"), recursive=True))
    if not matches:
        raise FileNotFoundError("no seernet_multi.pt checkpoint found; pass --ckpt explicitly")
    return Path(matches[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-converter prediction examples.")
    parser.add_argument("--ckpt", help="Optimized multi-output checkpoint. Defaults to a local seernet_multi.pt when available.")
    parser.add_argument("--mlp-shape", type=parse_shape, default=(1, 784), help="Input shape for SimpleMLP")
    parser.add_argument("--vgg-shape", type=parse_shape, default=(1, 3, 32, 32), help="Input shape for TinyVGG")
    return parser.parse_args()


def predict_one(name: str, spec: SourceModelSpec, ckpt_path: Path, model, ckpt: dict[str, Any], stats: dict[str, np.ndarray]) -> None:
    graph = convert_source_to_networkx(spec)
    data = convert_source_to_pyg_data(spec, ckpt_path=ckpt_path)
    with torch.no_grad():
        pred_std = model(data).detach().cpu().numpy()

    calibrator = LinearCalibrator.from_dict(ckpt.get("calibration"))
    if calibrator is not None:
        pred_std = calibrator.apply(pred_std)
    pred = invert_all(pred_std, stats)[0]

    node_types = [node_data["feature"]["type"] for _, node_data in graph.nodes(data=True)]
    print(f"\n{name}")
    print(f"  graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")
    print(f"  node types: {', '.join(node_types)}")
    print(f"  tensor input: x={tuple(data.x.shape)} edge_index={tuple(data.edge_index.shape)} edge_attr={tuple(data.edge_attr.shape)} u={tuple(data.u.shape)}")
    print("  predictions:")
    for metric_name, value in zip(TARGET_NAMES, pred):
        print(f"    {metric_name:>10}: {float(value):.6g}")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve() if args.ckpt else discover_checkpoint()
    model, ckpt, is_multi = load_model(str(ckpt_path), torch.device("cpu"))
    if not is_multi:
        raise ValueError(f"{ckpt_path} is a single-metric checkpoint; use a seernet_multi.pt checkpoint")
    stats = stats_from_metadata(ckpt)

    examples = [
        (
            "SimpleMLP",
            SourceModelSpec(
                source_path=ROOT / "examples/simple_mlp.py",
                entry="SimpleMLP",
                input_shapes=(args.mlp_shape,),
            ),
        ),
        (
            "TinyVGG",
            SourceModelSpec(
                source_path=ROOT / "examples/simple_vgg.py",
                entry="TinyVGG",
                input_shapes=(args.vgg_shape,),
            ),
        ),
    ]

    print(f"checkpoint: {os.path.relpath(ckpt_path, ROOT)}")
    for name, spec in examples:
        predict_one(name, spec, ckpt_path, model, ckpt, stats)


if __name__ == "__main__":
    main()
