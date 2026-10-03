"""Prepare, train, convert, and verify training-only PerfSeer artifacts."""

import argparse
import json
from pathlib import Path
import sys

import torch
from perfseer_v31.io import atomic_write, read_json


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "train":
        from .runner import main as train
        return train(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Project a verified v3.2 dataset to training-only graphs")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=1)
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset", type=Path)
    verify.add_argument("--predictions", type=Path)
    convert = commands.add_parser("convert")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--designs", type=Path, required=True, help="A JSON list of training-only designs")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    commands.add_parser("train", help="Train a fresh teacher/student pair; see train --help")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .dataset import prepare
        result = prepare(args.source, args.output, workers=args.workers)
    elif args.command == "verify":
        if bool(args.dataset) == bool(args.predictions):
            parser.error("verify requires exactly one of --dataset or --predictions")
        if args.dataset:
            from .dataset import verify
            result = verify(args.dataset)
        else:
            from .verification import verify_predictions
            result = verify_predictions(args.predictions)
    elif args.command == "convert":
        from .conversion import convert_file
        convert_file(args.source, args.output)
        result = {"status": "converted", "output": str(args.output), "accuracy_gate_claimed": False}
    else:
        from .inference import export_model, predict
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.command == "export":
            export_model(payload, args.output)
        else:
            atomic_write(args.output, predict(payload, read_json(args.designs), device=args.device))
        result = {"status": "completed", "output": str(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
