"""Immutable non-vision V1 to Disaster V2 candidate lineage crosswalk."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .a10_crosswalk import semantic_distribution_signature
from .a10_speech_crosswalk import task_independent_compute_signature
from .disaster_substitution import (
    DISASTER_TASK_ID,
    HISTORICAL_TASK_ID,
    load_disaster_substitution_contract,
)
from .fingerprints import canonical_sha256, canonical_value


PREDECESSOR_PROFILE = "native_a10_nonvision_4gpu_v1"
PREDECESSOR_MANIFEST_SHA256 = (
    "05e4d98b94c55d787a3d325b9c35f8ab3ff5dfbbb211be22844ff49bd881515b"
)
PREDECESSOR_TASK_REGISTRY_SHA256 = (
    "dddc68689d6fa3e59426ea7f9cccd37a5787a633112492e1903cfad7bc60e298"
)
PREDECESSOR_MODEL_REGISTRY_SHA256 = (
    "9953e4d249688f244ddbb9cf9475b988476144ffed2e4e2870df36056b18f513"
)
PREDECESSOR_CROSSWALK_SHA256 = (
    "6556f56f799c83772eb9fbac8639aa4c7be7476fdf3616b519cbcfd89f1be999"
)
CROSSWALK_VERSION = "perfseer_v3_nrp_a10_nonvision_disaster_crosswalk_v2"
CROSSWALK_ROW_VERSION = "perfseer_v3_nrp_a10_nonvision_disaster_crosswalk_row_v2"
EXPECTED_CLASSIFICATIONS = {
    "unchanged": 9_050,
    "nlp_registry_rebound": 1_075,
    "dataset_substitution": 1_075,
}


class A10DisasterCrosswalkError(RuntimeError):
    """Raised when V1-to-V2 lineage changes outside the contracted rows."""


def _emit(path: Path) -> None:
    from .model_registry import load_model_registry
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    manifest = build_target_manifest()
    summary = {
        "kind": "summary",
        "manifest_sha256": manifest.sha256,
        "task_registry_sha256": load_task_registry().sha256,
        "model_registry_sha256": load_model_registry().sha256,
        "candidate_count": len(manifest.candidates),
    }
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, sort_keys=True) + "\n")
        for candidate in manifest.candidates:
            stream.write(
                json.dumps(
                    canonical_value(
                        {
                            "kind": "candidate",
                            "ordinal": candidate.ordinal,
                            "candidate_id": candidate.candidate_id,
                            "task_id": candidate.task_id,
                            "family_id": candidate.family_id,
                            "semantic_signature": semantic_distribution_signature(candidate),
                            "compute_signature": task_independent_compute_signature(candidate),
                        }
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _predecessor_emission() -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    with tempfile.TemporaryDirectory(prefix="perfseer-disaster-crosswalk-") as directory:
        output = Path(directory) / "nonvision-v1.jsonl"
        environment = os.environ.copy()
        environment["PERFSEER_LABELER_PROFILE"] = PREDECESSOR_PROFILE
        subprocess.run(
            [sys.executable, "-m", __name__, "--emit", str(output)],
            check=True,
            env=environment,
            timeout=300,
        )
        with output.open(encoding="utf-8") as stream:
            summary = json.loads(next(stream))
            rows = tuple(json.loads(line) for line in stream if line.strip())
    if summary.pop("kind", None) != "summary" or any(
        row.pop("kind", None) != "candidate" for row in rows
    ):
        raise A10DisasterCrosswalkError("predecessor emission schema differs")
    return summary, rows


@dataclass(frozen=True)
class A10DisasterCrosswalk:
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        if self.summary.get("version") != CROSSWALK_VERSION or len(self.rows) != 11_200:
            raise A10DisasterCrosswalkError("crosswalk identity or row count differs")
        if tuple(row.get("ordinal") for row in self.rows) != tuple(range(11_200)):
            raise A10DisasterCrosswalkError("crosswalk ordinals differ")
        if Counter(row.get("row_classification") for row in self.rows) != EXPECTED_CLASSIFICATIONS:
            raise A10DisasterCrosswalkError("crosswalk classification counts differ")
        if any(
            row.get("old_compute_signature") != row.get("new_compute_signature")
            for row in self.rows
        ):
            raise A10DisasterCrosswalkError("substitution changed a compute signature")
        for row in self.rows:
            changed = row["v1_candidate_id"] != row["v2_candidate_id"]
            if changed != (row["row_classification"] != "unchanged"):
                raise A10DisasterCrosswalkError("candidate-ID classification differs")
            substituted = row["row_classification"] == "dataset_substitution"
            if substituted != (
                row["old_task_id"] == HISTORICAL_TASK_ID
                and row["new_task_id"] == DISASTER_TASK_ID
            ):
                raise A10DisasterCrosswalkError("dataset substitution task pairing differs")
            for key in (
                "v1_candidate_id",
                "v2_candidate_id",
                "old_semantic_signature",
                "new_semantic_signature",
                "old_compute_signature",
                "new_compute_signature",
            ):
                if not isinstance(row.get(key), str) or len(row[key]) != 64:
                    raise A10DisasterCrosswalkError(f"crosswalk {key} is not a digest")
        if len({row["v2_candidate_id"] for row in self.rows}) != 11_200:
            raise A10DisasterCrosswalkError("V2 candidate IDs are not unique")
        if self.summary.get("rows_sha256") != canonical_sha256(self.rows):
            raise A10DisasterCrosswalkError("crosswalk row hash differs")
        unhashed = dict(self.summary)
        declared = unhashed.pop("crosswalk_sha256", None)
        if declared != canonical_sha256(unhashed):
            raise A10DisasterCrosswalkError("crosswalk summary hash differs")


def build_crosswalk(manifest: Any | None = None) -> A10DisasterCrosswalk:
    from .labeler_profile import PROFILE
    from .model_registry import load_model_registry
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    if not PROFILE.uses_disaster_v2:
        raise A10DisasterCrosswalkError("Disaster crosswalk requires its V2 profile")
    current = manifest or build_target_manifest()
    current.validate()
    old_summary, old_rows = _predecessor_emission()
    if old_summary != {
        "manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "task_registry_sha256": PREDECESSOR_TASK_REGISTRY_SHA256,
        "model_registry_sha256": PREDECESSOR_MODEL_REGISTRY_SHA256,
        "candidate_count": 11_200,
    }:
        raise A10DisasterCrosswalkError("non-vision V1 predecessor identity drifted")
    if len(old_rows) != len(current.candidates):
        raise A10DisasterCrosswalkError("predecessor/current counts differ")
    rows = []
    for old, new in zip(old_rows, current.candidates, strict=True):
        if old["ordinal"] != new.ordinal or old["family_id"] != new.family_id:
            raise A10DisasterCrosswalkError("candidate ordering/family distribution changed")
        if old["task_id"] == HISTORICAL_TASK_ID:
            classification = "dataset_substitution"
        elif old["candidate_id"] != new.candidate_id:
            classification = "nlp_registry_rebound"
        else:
            classification = "unchanged"
        rows.append(
            canonical_value(
                {
                    "version": CROSSWALK_ROW_VERSION,
                    "ordinal": new.ordinal,
                    "v1_candidate_id": old["candidate_id"],
                    "v2_candidate_id": new.candidate_id,
                    "row_classification": classification,
                    "old_task_id": old["task_id"],
                    "new_task_id": new.task_id,
                    "family_id": new.family_id,
                    "old_semantic_signature": old["semantic_signature"],
                    "new_semantic_signature": semantic_distribution_signature(new),
                    "old_compute_signature": old["compute_signature"],
                    "new_compute_signature": task_independent_compute_signature(new),
                }
            )
        )
    immutable_rows = tuple(rows)
    summary: dict[str, Any] = {
        "version": CROSSWALK_VERSION,
        "predecessor_profile": PREDECESSOR_PROFILE,
        "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "predecessor_task_registry_sha256": PREDECESSOR_TASK_REGISTRY_SHA256,
        "predecessor_model_registry_sha256": PREDECESSOR_MODEL_REGISTRY_SHA256,
        "predecessor_crosswalk_sha256": PREDECESSOR_CROSSWALK_SHA256,
        "v2_manifest_sha256": current.sha256,
        "v2_task_registry_sha256": load_task_registry().sha256,
        "v2_model_registry_sha256": load_model_registry().sha256,
        "substitution_contract_sha256": load_disaster_substitution_contract().sha256,
        "row_count": 11_200,
        "classification_counts": EXPECTED_CLASSIFICATIONS,
        "rows_sha256": canonical_sha256(immutable_rows),
    }
    summary["crosswalk_sha256"] = canonical_sha256(summary)
    result = A10DisasterCrosswalk(canonical_value(summary), immutable_rows)
    result.validate()
    return result


def freeze_crosswalk(
    workspace: str | Path, manifest: Any | None = None
) -> A10DisasterCrosswalk:
    from .storage import atomic_write_json, atomic_write_jsonl

    state = Path(workspace).resolve() / "state"
    summary_path = state / "a10_disaster_crosswalk_summary.json"
    rows_path = state / "a10_disaster_crosswalk.jsonl"
    if summary_path.exists() or rows_path.exists():
        if not summary_path.is_file() or not rows_path.is_file():
            raise A10DisasterCrosswalkError("crosswalk state is partially present")
        result = A10DisasterCrosswalk(
            json.loads(summary_path.read_text(encoding="utf-8")),
            tuple(
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
        )
        result.validate()
        current = manifest or __import__(
            "perfseer_v3.dataset_pack.sampler", fromlist=["build_target_manifest"]
        ).build_target_manifest()
        if result.summary.get("v2_manifest_sha256") != current.sha256:
            raise A10DisasterCrosswalkError("workspace crosswalk differs from V2")
        return result
    result = build_crosswalk(manifest)
    atomic_write_jsonl(rows_path, result.rows)
    atomic_write_json(summary_path, result.summary)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=Path, required=True)
    arguments = parser.parse_args(argv)
    _emit(arguments.emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A10DisasterCrosswalk",
    "A10DisasterCrosswalkError",
    "EXPECTED_CLASSIFICATIONS",
    "PREDECESSOR_CROSSWALK_SHA256",
    "PREDECESSOR_MANIFEST_SHA256",
    "build_crosswalk",
    "freeze_crosswalk",
]
