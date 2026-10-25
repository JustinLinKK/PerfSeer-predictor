"""Campaign command-line interface, shared by local CUDA and Nautilus."""

import argparse
import json
from pathlib import Path

from perfseer_v31.io import atomic_write, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-campaign-data")
    prepare.add_argument("--repo", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    prepare.add_argument("--smoke-rows-per-split", type=int, default=0)
    prepare.add_argument("--feature-workers", type=int, default=0)
    config = commands.add_parser("campaign-config")
    config.add_argument("--data", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    config.add_argument("--save", type=Path, required=True)
    config.add_argument("--smoke", action="store_true")
    for name in ("campaign", "campaign-stage", "verify-campaign", "evaluate-campaign"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"campaign", "campaign-stage"}:
            command.add_argument("--resume", action="store_true")
        if name == "campaign-stage":
            command.add_argument("--source", choices=("A100", "A10"), required=True)
            command.add_argument("--proposal", choices=("v4.0", "v4.1", "v4.2", "v4.3"), required=True)
            command.add_argument("--stage", choices=("source_teacher", "source_student", "target_teacher", "target_student"), required=True)
            command.add_argument("--probe", action="store_true")
            command.add_argument("--result", type=Path)
            command.add_argument("--memory-fraction", type=float)
    package = commands.add_parser("package-campaign")
    package.add_argument("--config", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--repo", type=Path, default=Path.cwd())
    render = commands.add_parser("render-campaign-job")
    render.add_argument("--package", type=Path, required=True)
    render.add_argument("--image", required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--namespace", default="ecepxie")
    render.add_argument("--name", default="perfseer-v4-eight-pair-a100")
    args = parser.parse_args(argv)
    if args.command == "prepare-campaign-data":
        from .campaign_data import prepare
        if args.smoke_rows_per_split < 0:
            parser.error("smoke rows must be nonnegative")
        result = prepare(args.repo, args.output, device=args.device, smoke_rows=args.smoke_rows_per_split,
                         feature_workers=args.feature_workers)
        from .campaign import source_normalization
        for source in ("A100", "A10"):
            source_normalization(args.output / source)
        result = {key: value["fingerprint"] for key, value in result.items()}
    elif args.command == "campaign-config":
        from .campaign import default_config, validate_config
        result = validate_config(default_config(args.data.resolve(), args.output.resolve(), smoke=args.smoke))
        atomic_write(args.save, result)
    elif args.command in {"package-campaign", "render-campaign-job"}:
        from .campaign_packaging import package, render_job
        result = package(args.repo, read_json(args.config), args.output) if args.command == "package-campaign" else render_job(
            args.package, args.image, args.output, namespace=args.namespace, name=args.name)
    else:
        from .campaign import validate_config, run_stage, verify_campaign, final_evaluate
        config = validate_config(read_json(args.config))
        if args.command == "campaign-stage":
            try:
                result = run_stage(config, args.source, args.proposal, args.stage, resume=args.resume,
                                   probe=args.probe, memory_fraction=args.memory_fraction)
            except InterruptedError:
                raise SystemExit(75)
            if args.result:
                atomic_write(args.result, result)
        elif args.command == "verify-campaign":
            result = verify_campaign(config)
        elif args.command == "evaluate-campaign":
            result = final_evaluate(config)
        else:
            from .campaign_controller import run
            result = run(config, args.config.resolve(), resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
