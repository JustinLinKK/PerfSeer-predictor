"""Build the deterministic bootstrap workload and grouped-split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.splits import grouped_split, split_manifest_payload
from perfseer_v3.workloads import (
    default_composites,
    default_microbenchmarks,
    default_source_workloads,
    manifest_payload,
)


def build_payload(seed: int) -> dict[str, object]:
    descriptors = (
        *default_microbenchmarks(),
        *default_composites(),
        *default_source_workloads(),
    )
    manifest = manifest_payload(descriptors)
    split_manifest = split_manifest_payload(grouped_split(descriptors, seed=seed))
    return {
        "report_version": "perfseer_v3_workload_and_split_manifest_v1",
        "manifest": manifest,
        "split_manifest": split_manifest,
        "summary": {
            "workloads": len(descriptors),
            "source_groups": len({row.source_group for row in descriptors}),
            "data_layers": dict(
                sorted(Counter(row.data_layer for row in descriptors).items())
            ),
            "shape_regimes": dict(
                sorted(Counter(row.shape_regime for row in descriptors).items())
            ),
            "precisions": dict(
                sorted(Counter(row.dtype for row in descriptors).items())
            ),
            "phases": dict(
                sorted(Counter(row.phase for row in descriptors).items())
            ),
            "optimizers": dict(
                sorted(Counter(row.optimizer or "none" for row in descriptors).items())
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_workload_manifest.json",
    )
    args = parser.parse_args(argv)
    payload = build_payload(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"workloads={payload['summary']['workloads']} "
        f"manifest_sha256={payload['manifest']['sha256']} "
        f"split_sha256={payload['split_manifest']['sha256']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
