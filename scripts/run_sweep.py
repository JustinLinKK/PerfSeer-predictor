#!/usr/bin/env python
"""Run a list of optimized PerfSeer configs sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--metric")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for cfg in args.configs:
        cmd = [sys.executable, "-m", "perfseer_optimized.train", "--config", cfg]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.metric is not None:
            cmd += ["--metric", args.metric]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
