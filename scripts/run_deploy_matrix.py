#!/usr/bin/env python
"""Evaluate one checkpoint directory across deployment runtime profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROFILES = [
    "src/perfseer-optimized/configs/eval_profiles/cpu_pytorch_fp32.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_pytorch_dynamic_int8.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_torchscript_fp32.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_onnx_fp32.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_onnx_int8.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_openvino_fp32.yaml",
    "src/perfseer-optimized/configs/eval_profiles/cpu_openvino_int8.yaml",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-bench-graphs", type=int)
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--results-path", default="runs/results.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for profile in args.profiles:
        if not Path(profile).exists():
            raise SystemExit(f"missing eval profile: {profile}")
        cmd = [
            sys.executable,
            "-m",
            "perfseer_optimized.eval_deploy",
            "--eval-profile",
            profile,
            "--ckpt-dir",
            args.ckpt_dir,
            "--data-root",
            args.data_root,
            "--results-path",
            args.results_path,
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.num_bench_graphs is not None:
            cmd += ["--num-bench-graphs", str(args.num_bench_graphs)]
        if args.cpu_threads is not None:
            cmd += ["--cpu-threads", str(args.cpu_threads)]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
