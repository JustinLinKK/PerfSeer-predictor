#!/usr/bin/env python3
"""Run the gated PerfSeer v3 teacher or student training workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.training_runner import run_smoke, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    smoke = subparsers.add_parser("smoke", help="run a non-production local smoke test")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=42)

    for stage in (
        "base_teacher",
        "base_student",
        "target_teacher_adapter",
        "target_student_adapter",
    ):
        command = subparsers.add_parser(stage)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument(
            "--amp", choices=("none", "float16", "bfloat16"), default="bfloat16"
        )
        command.add_argument("--epochs", type=int)
        command.add_argument(
            "--pretrain-epochs",
            type=int,
            help="override the base-teacher Stage A epoch count from the config",
        )
        if stage in {"base_student", "target_student_adapter"}:
            command.add_argument("--teacher-artifact", type=Path, required=True)
        if stage in {"target_teacher_adapter", "target_student_adapter"}:
            command.add_argument("--base-artifact", type=Path, required=True)
            command.add_argument("--resume-artifact", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "smoke":
        report = run_smoke(output_path=args.output, seed=args.seed)
    else:
        report = run_training(
            stage=args.stage,
            config_path=args.config,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            amp=args.amp,
            epochs=args.epochs,
            pretrain_epochs=args.pretrain_epochs,
            teacher_artifact=getattr(args, "teacher_artifact", None),
            base_artifact=getattr(args, "base_artifact", None),
            resume_artifact=getattr(args, "resume_artifact", None),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
