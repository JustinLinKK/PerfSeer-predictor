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
    predict.add_argument("--calibration", type=Path)
    predict.add_argument("--environment", type=Path)
    transfer = commands.add_parser("transfer", help="Fit candidates and seal validation selection; does not score test labels")
    transfer.add_argument("--dataset", type=Path, required=True)
    transfer.add_argument("--source", type=Path, required=True)
    transfer.add_argument("--resource-source", type=Path)
    transfer.add_argument("--environment", type=Path, required=True)
    transfer.add_argument("--output", type=Path, required=True)
    transfer.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256])
    transfer.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    transfer.add_argument("--variants", nargs="+", default=["source", "constant", "affine", "v4.1", "v4.0-finetune", "v4.2", "v4.3"])
    transfer.add_argument("--device", default="cpu", help="Neural adaptation device; residual source predictions use CPU FP32")
    transfer.add_argument("--epochs", type=int, default=100)
    transfer.add_argument("--microbatch", type=int, default=4)
    evaluation = commands.add_parser("evaluate", help="Evaluate a sealed selection on held-out test rows")
    evaluation.add_argument("--selection", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--microbatch", type=int, default=4)
    resource = commands.add_parser("train-resource", help="Explicitly train the v4.3 source MLP")
    resource.add_argument("--dataset", type=Path, required=True)
    resource.add_argument("--output", type=Path, required=True)
    resource.add_argument("--device", default="cuda")
    resource.add_argument("--epochs", type=int, default=100)
    resource.add_argument("--microbatch", type=int, default=256)
    resource.add_argument("--seed", type=int, default=11)
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
    elif args.command == "transfer":
        from .experiments import select
        selection = select(args.dataset, args.source, args.resource_source, read_json(args.environment), args.output,
                           budgets=args.budgets, seeds=args.seeds, variants=args.variants,
                           device=args.device, epochs=args.epochs, microbatch=args.microbatch)
        result = {"status": selection["stage"], "trials": len(selection["trials"]),
                  "selection": str(args.output / "selection.json"), "deployment_approved": False}
    elif args.command == "evaluate":
        from .experiments import evaluate_selection
        report = evaluate_selection(args.selection, args.output, device=args.device, microbatch=args.microbatch)
        result = {"status": report["status"], "trials": len(report["trials"]), "output": str(args.output), "deployment_approved": False}
    elif args.command == "train-resource":
        from .transfer import train_resource
        result = train_resource(args)
    else:
        from .inference import export_model, predict
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.command == "export":
            export_model(payload, args.output)
        else:
            from .residuals import load_adapter
            atomic_write(args.output, predict(payload, read_json(args.designs), device=args.device,
                         calibration=load_adapter(args.calibration) if args.calibration else None,
                         environment=read_json(args.environment) if args.environment else None))
        result = {"status": "completed", "output": str(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
