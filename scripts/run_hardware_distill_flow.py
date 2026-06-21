#!/usr/bin/env python
"""Run or print the scratch hardware teacher-to-student workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_TEACHER_CONFIG = "src/perfseer-optimized/configs/train_hardware_teacher/large_teacher.yaml"
DEFAULT_STUDENT_CONFIG = "src/perfseer-optimized/configs/train_deploy_model/distill_student_128.yaml"
DEFAULT_EVAL_PROFILE = "src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the scratch one-hardware teacher and distilled-student flow.")
    p.add_argument("--python", default=sys.executable, help="Python executable for module commands.")
    p.add_argument("--data-root", default="dataset", help="Materialized profiling dataset root.")
    p.add_argument("--hardware-id", required=True, help="Single hardware id to train for, for example rtx4090.")
    p.add_argument("--teacher-config", default=DEFAULT_TEACHER_CONFIG)
    p.add_argument("--student-config", default=DEFAULT_STUDENT_CONFIG)
    p.add_argument("--eval-profile", default=DEFAULT_EVAL_PROFILE)
    p.add_argument("--deploy-eval-profile", help="Optional deployment eval profile.")
    p.add_argument("--out-dir", default="runs/optimized")
    p.add_argument("--results-path", default="runs/results.jsonl")
    p.add_argument("--teacher-run-id", help="Defaults to hardware_large_teacher_<hardware-id>.")
    p.add_argument("--student-run-id", help="Defaults to hardware_distill_student_128_<hardware-id>.")
    p.add_argument("--teacher-ckpt-dir", help="Existing teacher directory for --skip-teacher distillation.")
    p.add_argument("--limit", type=int, help="Optional data limit for train/eval commands.")
    p.add_argument("--split-unit", choices=("pair", "graph", "graph_signature", "graph_family"), help="Override data split unit.")
    p.add_argument("--teacher-epochs", type=int, help="Override teacher epochs.")
    p.add_argument("--student-epochs", type=int, help="Override student epochs.")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--skip-distill", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-deploy-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args(argv)


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("+", quote_cmd(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def train_cmd(
    args: argparse.Namespace,
    config: str,
    run_id: str,
    *,
    epochs: int | None = None,
    teacher_ckpt_dir: Path | None = None,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.train",
        "--config",
        config,
        "--run-id",
        run_id,
        "--data-root",
        args.data_root,
        "--hardware-id",
        args.hardware_id,
        "--out",
        args.out_dir,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.split_unit:
        cmd += ["--split-unit", args.split_unit]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if teacher_ckpt_dir is not None:
        cmd += ["--teacher-ckpt-dir", str(teacher_ckpt_dir)]
    return cmd


def eval_cmd(args: argparse.Namespace, ckpt_dir: Path, label: str) -> list[str]:
    print(f"# evaluate {label}", flush=True)
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.eval",
        "--eval-profile",
        args.eval_profile,
        "--ckpt-dir",
        str(ckpt_dir),
        "--data-root",
        args.data_root,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def deploy_eval_cmd(args: argparse.Namespace, ckpt_dir: Path) -> list[str]:
    if not args.deploy_eval_profile:
        raise ValueError("--deploy-eval-profile is required")
    print("# evaluate deployment student", flush=True)
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.eval_deploy",
        "--eval-profile",
        args.deploy_eval_profile,
        "--ckpt-dir",
        str(ckpt_dir),
        "--data-root",
        args.data_root,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    teacher_run_id = args.teacher_run_id or f"hardware_large_teacher_{args.hardware_id}"
    student_run_id = args.student_run_id or f"hardware_distill_student_128_{args.hardware_id}"
    out_root = Path(args.out_dir)
    teacher_dir = Path(args.teacher_ckpt_dir) if args.teacher_ckpt_dir else out_root / teacher_run_id
    student_dir = out_root / student_run_id

    if not args.skip_teacher:
        print("# train scratch hardware teacher", flush=True)
        run_cmd(train_cmd(args, args.teacher_config, teacher_run_id, epochs=args.teacher_epochs), dry_run=args.dry_run)

    if not args.skip_distill:
        print("# distill hardware student", flush=True)
        run_cmd(
            train_cmd(
                args,
                args.student_config,
                student_run_id,
                epochs=args.student_epochs,
                teacher_ckpt_dir=teacher_dir,
            ),
            dry_run=args.dry_run,
        )

    if not args.skip_eval:
        if not args.skip_teacher:
            run_cmd(eval_cmd(args, teacher_dir, "hardware teacher"), dry_run=args.dry_run)
        if not args.skip_distill:
            run_cmd(eval_cmd(args, student_dir, "hardware student"), dry_run=args.dry_run)
            if args.deploy_eval_profile and not args.skip_deploy_eval:
                run_cmd(deploy_eval_cmd(args, student_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
