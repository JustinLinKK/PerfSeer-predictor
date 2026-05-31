#!/usr/bin/env python
"""Summarize optimized PerfSeer result ledger rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="runs/results.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"missing results file: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "eval_complete":
            rows.append(row)
    rows.sort(key=lambda r: r.get("mean_mape", float("inf")))
    print(f"{'run_id':<28} {'mean_mape':>10} {'params':>12} {'cpu_p95':>10}")
    print("-" * 64)
    for row in rows:
        cpu = row.get("cpu_forward") or {}
        print(
            f"{row.get('run_id', ''):<28} "
            f"{row.get('mean_mape', float('nan')):>10.3f} "
            f"{row.get('params', 0):>12,} "
            f"{cpu.get('p95_ms', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    main()
