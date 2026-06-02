"""CLI for converting PyTorch model source into PerfSeer graph inputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import torch

from .converter import SourceModelSpec, convert_source_to_networkx, convert_source_to_pyg_data


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid input shape {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"input shape must contain positive integers: {value!r}")
    return shape


def parse_json_arg(value: str, expected: type, label: str):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be valid JSON") from exc
    if not isinstance(parsed, expected):
        raise argparse.ArgumentTypeError(f"{label} must decode to {expected.__name__}")
    return parsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PyTorch model source to PerfSeer graph input.")
    parser.add_argument("--source", required=True, help="Python file containing the model definition")
    parser.add_argument("--entry", required=True, help="Model class, instance, or factory name inside --source")
    parser.add_argument("--input-shape", action="append", type=parse_shape, required=True, help="Input tensor shape, e.g. 1,3,224,224. Repeat for multiple inputs.")
    parser.add_argument("--input-dtype", action="append", help="Input dtype name. Repeat or provide once. Default: float32")
    parser.add_argument("--constructor-args", default="[]", help="JSON list passed positionally to the entry constructor/factory")
    parser.add_argument("--constructor-kwargs", default="{}", help="JSON object passed as keyword args to the entry constructor/factory")
    parser.add_argument("--ckpt", help="Optimized PerfSeer checkpoint supplying norm_stats and feature_config")
    parser.add_argument("--out", help="Path to save a torch-serialized PerfSeerOptimizedData object")
    parser.add_argument("--graph-out", help="Path to save the intermediate networkx graph pickle")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    constructor_args = tuple(parse_json_arg(args.constructor_args, list, "--constructor-args"))
    constructor_kwargs = parse_json_arg(args.constructor_kwargs, dict, "--constructor-kwargs")
    spec = SourceModelSpec(
        source_path=args.source,
        entry=args.entry,
        input_shapes=tuple(args.input_shape),
        constructor_args=constructor_args,
        constructor_kwargs=constructor_kwargs,
        input_dtypes=tuple(args.input_dtype or ("float32",)),
    )

    graph = None
    if args.graph_out:
        graph = convert_source_to_networkx(spec)
        graph_path = Path(args.graph_out)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with graph_path.open("wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if args.out:
        data = convert_source_to_pyg_data(spec, ckpt_path=args.ckpt)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)

    if not args.out and not args.graph_out:
        graph = graph or convert_source_to_networkx(spec)
        print(f"converted graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges", flush=True)


if __name__ == "__main__":
    main()
