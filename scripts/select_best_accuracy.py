#!/usr/bin/env python
"""Select the best accuracy-track run id from a result ledger."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best accuracy run by GPU mean MAPE.")
    p.add_argument("--results", required=True)
    p.add_argument("--prefix", default="accuracy_")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results)
    best: tuple[float, str] | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "eval_complete":
            continue
        if row.get("track", "accuracy") != "accuracy":
            continue
        run_id = str(row.get("run_id", ""))
        if args.prefix and not run_id.startswith(args.prefix):
            continue
        value = float(row.get("mean_mape", "nan"))
        if math.isfinite(value) and (best is None or value < best[0]):
            best = (value, run_id)
    if best is None:
        raise SystemExit(f"no accuracy eval rows found in {path}")
    print(best[1])


if __name__ == "__main__":
    main()
