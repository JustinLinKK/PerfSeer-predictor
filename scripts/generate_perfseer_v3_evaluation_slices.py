"""Generate the deterministic PerfSeer v3 evaluation-slice manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.splits import (
    evaluation_slice_manifest_payload,
    grouped_split,
)
from perfseer_v3.workloads import (
    default_composites,
    default_microbenchmarks,
    default_source_workloads,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_evaluation_slices.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    descriptors = (
        *default_microbenchmarks(),
        *default_composites(),
        *default_source_workloads(),
    )
    payload = evaluation_slice_manifest_payload(
        grouped_split(descriptors, seed=args.seed)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete={payload['complete']} "
        f"missing={len(payload['missing_required_slices'])} "
        f"sha256={payload['sha256']}"
    )
    return 0 if payload["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
