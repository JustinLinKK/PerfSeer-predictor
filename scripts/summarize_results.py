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
        if row.get("event") in {"eval_complete", "eval_deploy_complete"}:
            rows.append(row)
    rows.sort(key=lambda r: (r.get("track", "accuracy"), r.get("mean_mape", float("inf"))))
    print(
        f"{'track':<12} {'run_id':<32} {'backend':<24} "
        f"{'mean_mape':>10} {'params':>12} {'p95_ms':>10} {'size_mb':>9}"
    )
    print("-" * 118)
    for row in rows:
        cpu = row.get("cpu_forward") or {}
        backend = row.get("runtime_backend_actual") or row.get("runtime_backend") or "pytorch"
        p95 = row.get("latency_forward_ms_p95", cpu.get("p95_ms", float("nan")))
        params = row.get("model_params", row.get("params", 0))
        size_mb = row.get("artifact_size_mb", float("nan"))
        print(
            f"{row.get('track', 'accuracy'):<12} "
            f"{row.get('run_id', ''):<32} "
            f"{backend:<24} "
            f"{row.get('mean_mape', float('nan')):>10.3f} "
            f"{params:>12,} "
            f"{p95:>10.3f} "
            f"{size_mb:>9.2f}"
        )


if __name__ == "__main__":
    main()
