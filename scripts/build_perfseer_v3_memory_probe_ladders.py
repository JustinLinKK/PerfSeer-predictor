#!/usr/bin/env python3
"""Build deterministic larger-batch candidates for frozen target memory probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.dataset_pack.sampler import target_candidate_from_dict
from perfseer_v3.dataset_pack.transfer_labeling import (
    TARGET_MEMORY_PROBE_PLAN_VERSION,
    build_memory_probe_candidates,
)
from perfseer_v3.version import TRANSFER_MANIFEST_VERSION


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-microbatch-size", type=int)
    args = parser.parse_args(argv)

    subset = dict(_load_object(args.subset))
    if subset.get("manifest_version") != TRANSFER_MANIFEST_VERSION:
        raise ValueError("memory-probe subset version mismatch")
    declared_subset_sha256 = str(subset.pop("subset_sha256", ""))
    observed_subset_sha256 = hashlib.sha256(
        canonical_json(subset).encode("utf-8")
    ).hexdigest()
    if declared_subset_sha256 != observed_subset_sha256:
        raise ValueError("memory-probe subset content hash mismatch")
    subset["subset_sha256"] = declared_subset_sha256

    probe_ids = tuple(
        str(row.get("configuration_id", ""))
        for row in subset.get("memory_boundary_probes", ())
    )
    if not probe_ids or any(not value for value in probe_ids):
        raise ValueError("subset contains no valid memory-boundary probes")
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("subset repeats a memory-boundary probe")

    candidate_output = args.output.parent / "memory_probe_candidates"
    candidate_output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for base_configuration_id in probe_ids:
        candidate_path = args.candidate_directory / f"{base_configuration_id}.json"
        candidate = target_candidate_from_dict(_load_object(candidate_path))
        if candidate.candidate_id != base_configuration_id:
            raise ValueError("frozen probe candidate file has another configuration ID")
        probes = build_memory_probe_candidates(
            candidate,
            maximum_microbatch_size=args.maximum_microbatch_size,
        )
        output_paths = []
        for probe in probes:
            output_path = candidate_output / f"{probe.candidate_id}.json"
            output_path.write_text(
                json.dumps(probe.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_paths.append(str(output_path.resolve()))
        entries.append(
            {
                "base_configuration_id": base_configuration_id,
                "paired_microbatch_size": candidate.microbatch_size,
                "probe_microbatch_sizes": [probe.microbatch_size for probe in probes],
                "probe_configuration_ids": [probe.candidate_id for probe in probes],
                "candidate_paths": output_paths,
                "stop_after_first_oom": True,
            }
        )

    payload: dict[str, Any] = {
        "plan_version": TARGET_MEMORY_PROBE_PLAN_VERSION,
        "source_subset_sha256": declared_subset_sha256,
        "target_hardware_id": subset["target_hardware_id"],
        "paired_batch_must_run_first": True,
        "entries": entries,
    }
    payload["plan_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "plan_sha256": payload["plan_sha256"],
                "probe_base_count": len(entries),
                "probe_candidate_count": sum(
                    len(entry["probe_configuration_ids"]) for entry in entries
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
