#!/usr/bin/env python
"""Write a resolved experiment config for the unattended runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a PerfSeer experiment config.")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--results-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--teacher-dir")
    p.add_argument("--init-checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.src, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg.setdefault("run", {})
    cfg["run"]["out_dir"] = args.checkpoint_root
    cfg["run"]["results_path"] = args.results_path
    if args.run_id:
        cfg["run"]["run_id"] = args.run_id

    cfg.setdefault("data", {})
    cfg["data"]["root"] = args.data_root
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit

    cfg.setdefault("train", {})
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    if args.teacher_dir:
        cfg.setdefault("distillation", {})
        cfg["distillation"]["enabled"] = True
        cfg["distillation"]["teacher_ckpt_dir"] = args.teacher_dir

    if args.init_checkpoint:
        cfg.setdefault("train", {})
        cfg["train"]["init_checkpoint"] = args.init_checkpoint

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(cfg.get("run", {}).get("run_id") or Path(args.src).stem)


if __name__ == "__main__":
    main()
