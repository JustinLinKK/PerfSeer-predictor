"""Immutable Speech-V2 to non-vision four-A10 lineage projection."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .a10_crosswalk import semantic_distribution_signature
from .fingerprints import canonical_sha256, canonical_value


PREDECESSOR_PROFILE = "native_a10_speech_v2"
PREDECESSOR_MANIFEST_SHA256 = (
    "c8fb3fe0d51073c63645f539a53247204bde45cf199671263db7cab5e79f5e39"
)
CROSSWALK_VERSION = "perfseer_v3_nrp_a10_nonvision_crosswalk_v1"
CROSSWALK_ROW_VERSION = "perfseer_v3_nrp_a10_nonvision_crosswalk_row_v1"
EXPECTED_DISPOSITIONS = {"excluded": 6_800, "retained": 11_200}
EXPECTED_EXCLUSIONS = {
    "generated_vision_source": 650,
    "vision_family": 6_150,
}


class A10NonvisionCrosswalkError(RuntimeError):
    """Raised when the projected corpus is not an exact non-vision lineage."""


def _nonbatch_payload(candidate: Any) -> Mapping[str, Any]:
    return {
        "task_id": candidate.task_id,
        "family_id": candidate.family_id,
        "quota_modality": candidate.quota_modality,
        "source_modality": candidate.source_modality,
        "regime": candidate.regime,
        "source_lineage": candidate.source_lineage,
        "source_split": candidate.source_split,
        "architecture_parameters": candidate.architecture_parameters,
        "input_signature": candidate.input_signature,
        "training_step_id": candidate.training_step_id,
        "precision_policy": candidate.precision_policy,
        "optimizer": candidate.optimizer,
        "scheduler": candidate.scheduler,
        "execution": candidate.execution,
        "activation_checkpointing": candidate.activation_checkpointing,
        "seed_policy": candidate.seed_policy,
        "coverage_cell_specs": candidate.coverage_cell_specs,
    }


def nonbatch_semantic_signature(candidate: Any) -> str:
    return canonical_sha256(_nonbatch_payload(candidate))


def _emit_predecessor(path: Path) -> None:
    from .a10_speech_crosswalk import build_crosswalk
    from .sampler import build_target_manifest

    manifest = build_target_manifest()
    crosswalk = build_crosswalk(manifest)
    by_v2 = {row["native_v2_candidate_id"]: row for row in crosswalk.rows}
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "summary",
                    "manifest_sha256": manifest.sha256,
                    "crosswalk_sha256": crosswalk.summary["crosswalk_sha256"],
                    "candidate_count": len(manifest.candidates),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for candidate in manifest.candidates:
            lineage = by_v2[candidate.candidate_id]
            stream.write(
                json.dumps(
                    canonical_value(
                        {
                            "kind": "candidate",
                            "ordinal": candidate.ordinal,
                            "candidate_id": candidate.candidate_id,
                            "original_a10g_candidate_id": lineage[
                                "original_a10g_candidate_id"
                            ],
                            "native_v1_candidate_id": lineage[
                                "native_v1_candidate_id"
                            ],
                            "quota_modality": candidate.quota_modality,
                            "source_modality": candidate.source_modality,
                            "semantic_signature": semantic_distribution_signature(
                                candidate
                            ),
                            "nonbatch_semantic_signature": nonbatch_semantic_signature(
                                candidate
                            ),
                        }
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _predecessor_emission() -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    with tempfile.TemporaryDirectory(prefix="perfseer-nonvision-crosswalk-") as directory:
        path = Path(directory) / "speech-v2.jsonl"
        environment = os.environ.copy()
        environment["PERFSEER_LABELER_PROFILE"] = PREDECESSOR_PROFILE
        subprocess.run(
            [sys.executable, "-m", __name__, "--emit-predecessor", str(path)],
            check=True,
            env=environment,
            timeout=600,
        )
        with path.open(encoding="utf-8") as stream:
            summary = json.loads(next(stream))
            rows = tuple(json.loads(line) for line in stream if line.strip())
    if summary.pop("kind", None) != "summary" or any(
        row.pop("kind", None) != "candidate" for row in rows
    ):
        raise A10NonvisionCrosswalkError("predecessor emission schema differs")
    return summary, rows


@dataclass(frozen=True)
class A10NonvisionCrosswalk:
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        if self.summary.get("version") != CROSSWALK_VERSION or len(self.rows) != 18_000:
            raise A10NonvisionCrosswalkError("crosswalk version or row count differs")
        if tuple(row.get("historical_ordinal") for row in self.rows) != tuple(
            range(18_000)
        ):
            raise A10NonvisionCrosswalkError("historical ordinals are not contiguous")
        if Counter(row.get("disposition") for row in self.rows) != EXPECTED_DISPOSITIONS:
            raise A10NonvisionCrosswalkError("crosswalk disposition totals differ")
        exclusions = Counter(
            row.get("exclusion_reason")
            for row in self.rows
            if row.get("disposition") == "excluded"
        )
        if exclusions != EXPECTED_EXCLUSIONS:
            raise A10NonvisionCrosswalkError("vision exclusion totals differ")
        retained = tuple(row for row in self.rows if row["disposition"] == "retained")
        if tuple(row["nonvision_ordinal"] for row in retained) != tuple(range(11_200)):
            raise A10NonvisionCrosswalkError("non-vision ordinals are not contiguous")
        digest_fields = (
            "predecessor_candidate_id",
            "original_a10g_candidate_id",
            "native_v1_candidate_id",
        )
        if any(
            not isinstance(row.get(key), str) or len(row[key]) != 64
            for row in self.rows
            for key in digest_fields
        ):
            raise A10NonvisionCrosswalkError("historical candidate digest is invalid")
        if any(
            not isinstance(row.get("nonvision_candidate_id"), str)
            or len(row["nonvision_candidate_id"]) != 64
            or row["old_nonbatch_semantic_signature"]
            != row["new_nonbatch_semantic_signature"]
            for row in retained
        ):
            raise A10NonvisionCrosswalkError("retained semantic projection differs")
        if len({row["nonvision_candidate_id"] for row in retained}) != 11_200:
            raise A10NonvisionCrosswalkError("non-vision candidate IDs are not unique")
        if self.summary.get("rows_sha256") != canonical_sha256(self.rows):
            raise A10NonvisionCrosswalkError("crosswalk rows hash differs")
        unhashed = dict(self.summary)
        declared = unhashed.pop("crosswalk_sha256", None)
        if declared != canonical_sha256(unhashed):
            raise A10NonvisionCrosswalkError("crosswalk summary hash differs")


def build_crosswalk(manifest: Any | None = None) -> A10NonvisionCrosswalk:
    from .labeler_profile import PROFILE
    from .model_registry import load_model_registry
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    if not PROFILE.is_nonvision_4gpu:
        raise A10NonvisionCrosswalkError("non-vision crosswalk requires its baked profile")
    current = manifest or build_target_manifest()
    current.validate()
    predecessor_summary, predecessor = _predecessor_emission()
    if (
        predecessor_summary.get("manifest_sha256") != PREDECESSOR_MANIFEST_SHA256
        or predecessor_summary.get("candidate_count") != 18_000
    ):
        raise A10NonvisionCrosswalkError("Speech V2 predecessor identity drifted")
    retained_predecessor = tuple(
        row for row in predecessor if row["source_modality"] != "vision"
    )
    if len(retained_predecessor) != len(current.candidates):
        raise A10NonvisionCrosswalkError("retained predecessor count differs")
    by_historical_ordinal = {
        row["ordinal"]: (new_ordinal, candidate)
        for new_ordinal, (row, candidate) in enumerate(
            zip(retained_predecessor, current.candidates, strict=True)
        )
    }
    rows = []
    for old in predecessor:
        retained = old["ordinal"] in by_historical_ordinal
        if retained:
            new_ordinal, candidate = by_historical_ordinal[old["ordinal"]]
            new_id: str | None = candidate.candidate_id
            new_signature: str | None = semantic_distribution_signature(candidate)
            new_nonbatch: str | None = nonbatch_semantic_signature(candidate)
            exclusion_reason = None
        else:
            new_ordinal = None
            new_id = None
            new_signature = None
            new_nonbatch = None
            exclusion_reason = (
                "vision_family"
                if old["quota_modality"] == "vision"
                else "generated_vision_source"
            )
        rows.append(
            canonical_value(
                {
                    "version": CROSSWALK_ROW_VERSION,
                    "historical_ordinal": old["ordinal"],
                    "disposition": "retained" if retained else "excluded",
                    "exclusion_reason": exclusion_reason,
                    "predecessor_candidate_id": old["candidate_id"],
                    "original_a10g_candidate_id": old["original_a10g_candidate_id"],
                    "native_v1_candidate_id": old["native_v1_candidate_id"],
                    "nonvision_ordinal": new_ordinal,
                    "nonvision_candidate_id": new_id,
                    "old_semantic_signature": old["semantic_signature"],
                    "new_semantic_signature": new_signature,
                    "old_nonbatch_semantic_signature": old[
                        "nonbatch_semantic_signature"
                    ],
                    "new_nonbatch_semantic_signature": new_nonbatch,
                }
            )
        )
    immutable_rows = tuple(rows)
    summary: dict[str, Any] = {
        "version": CROSSWALK_VERSION,
        "predecessor_profile": PREDECESSOR_PROFILE,
        "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "predecessor_crosswalk_sha256": predecessor_summary["crosswalk_sha256"],
        "nonvision_manifest_sha256": current.sha256,
        "nonvision_task_registry_sha256": load_task_registry().sha256,
        "nonvision_model_registry_sha256": load_model_registry().sha256,
        "row_count": 18_000,
        "retained_count": 11_200,
        "excluded_count": 6_800,
        "disposition_counts": EXPECTED_DISPOSITIONS,
        "exclusion_counts": EXPECTED_EXCLUSIONS,
        "rows_sha256": canonical_sha256(immutable_rows),
    }
    summary["crosswalk_sha256"] = canonical_sha256(summary)
    result = A10NonvisionCrosswalk(canonical_value(summary), immutable_rows)
    result.validate()
    return result


def freeze_crosswalk(
    workspace: str | Path, manifest: Any | None = None
) -> A10NonvisionCrosswalk:
    from .storage import atomic_write_json, atomic_write_jsonl

    state = Path(workspace).resolve() / "state"
    summary_path = state / "a10_nonvision_crosswalk_summary.json"
    rows_path = state / "a10_nonvision_crosswalk.jsonl"
    if summary_path.exists() or rows_path.exists():
        if not summary_path.is_file() or not rows_path.is_file():
            raise A10NonvisionCrosswalkError("crosswalk state is partially present")
        stored = A10NonvisionCrosswalk(
            json.loads(summary_path.read_text(encoding="utf-8")),
            tuple(
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
        )
        stored.validate()
        from .model_registry import load_model_registry
        from .sampler import build_target_manifest
        from .task_registry import load_task_registry

        current = manifest or build_target_manifest()
        if (
            stored.summary.get("nonvision_manifest_sha256") != current.sha256
            or stored.summary.get("nonvision_task_registry_sha256")
            != load_task_registry().sha256
            or stored.summary.get("nonvision_model_registry_sha256")
            != load_model_registry().sha256
        ):
            raise A10NonvisionCrosswalkError("workspace crosswalk differs")
        return stored
    result = build_crosswalk(manifest)
    atomic_write_jsonl(rows_path, result.rows)
    atomic_write_json(summary_path, result.summary)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-predecessor", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.emit_predecessor is None:
        raise SystemExit("--emit-predecessor is required for the internal emitter")
    _emit_predecessor(arguments.emit_predecessor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A10NonvisionCrosswalk",
    "A10NonvisionCrosswalkError",
    "CROSSWALK_VERSION",
    "EXPECTED_DISPOSITIONS",
    "EXPECTED_EXCLUSIONS",
    "PREDECESSOR_MANIFEST_SHA256",
    "build_crosswalk",
    "freeze_crosswalk",
    "nonbatch_semantic_signature",
]
