"""Three-way AWS A10G / native A10 V1 / speech V2 lineage crosswalk."""

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

from .a10_crosswalk import (
    EXPECTED_PRECISION_COUNTS,
    REFERENCE_MANIFEST_SHA256,
    REFERENCE_MODEL_REGISTRY_SHA256,
    REFERENCE_SOURCE_COMMIT,
    REFERENCE_TASK_REGISTRY_SHA256,
    semantic_distribution_signature,
)
from .fingerprints import canonical_sha256, canonical_value
from .speech_substitution import (
    HISTORICAL_WHALE_TASK_ID,
    SPEECH_TASK_ID,
    load_speech_substitution_contract,
)


NATIVE_V1_MANIFEST_SHA256 = (
    "361f31ed6fe6b75c508c7e5405aad6c893cc69f5a3cfe33d71fcc823c88aaba5"
)
NATIVE_V1_CROSSWALK_SHA256 = (
    "d66c4f098665f9bab003825abde591c6d142dd68d09178e93475bac2e3db9a69"
)
CROSSWALK_VERSION = "perfseer_v3_nrp_a10_speech_three_way_crosswalk_v2"
CROSSWALK_ROW_VERSION = "perfseer_v3_nrp_a10_speech_crosswalk_row_v2"
EXPECTED_CLASS_COUNTS = {
    "unchanged": 16_700,
    "audio_registry_rebound": 650,
    "dataset_substitution": 650,
}


class A10SpeechCrosswalkError(RuntimeError):
    """Raised if historical identity or the narrowly scoped rebound drifts."""


def task_independent_compute_payload(candidate: Any) -> Mapping[str, Any]:
    """Bind every compute field while intentionally excluding dataset identity."""

    batch = asdict(candidate.batch_plan)
    batch.pop("version")
    return {
        "family_id": candidate.family_id,
        "quota_modality": candidate.quota_modality,
        "source_modality": candidate.source_modality,
        "regime": candidate.regime,
        "source_lineage": candidate.source_lineage,
        "source_split": candidate.source_split,
        "architecture_parameters": candidate.architecture_parameters,
        "input_signature": candidate.input_signature,
        "task_schema_sha256": candidate.task_schema_sha256,
        "target_width": candidate.target_width,
        "training_step_id": candidate.training_step_id,
        "precision_policy": candidate.precision_policy,
        "optimizer": candidate.optimizer,
        "scheduler": candidate.scheduler,
        "execution": candidate.execution,
        "batch": {
            **batch,
            "microbatch_size": candidate.microbatch_size,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        },
        "activation_checkpointing": candidate.activation_checkpointing,
        "seed_policy": candidate.seed_policy,
        "coverage_cell_specs": candidate.coverage_cell_specs,
    }


def task_independent_compute_signature(candidate: Any) -> str:
    return canonical_sha256(task_independent_compute_payload(candidate))


def _distribution_payload(candidates: Sequence[Any]) -> Mapping[str, Mapping[str, int]]:
    dimensions = {
        "family": lambda row: row.family_id,
        "modality": lambda row: row.quota_modality,
        "architecture": lambda row: json.dumps(
            row.architecture_parameters, sort_keys=True, separators=(",", ":")
        ),
        "precision": lambda row: row.precision_policy["policy_id"],
        "optimizer": lambda row: row.optimizer["name"],
        "scheduler": lambda row: row.scheduler["name"],
        "execution": lambda row: row.execution["mode"],
        "batch": lambda row: json.dumps(
            {
                "microbatch_size": row.microbatch_size,
                "gradient_accumulation_steps": row.gradient_accumulation_steps,
                "tier": row.batch_plan.effective_tier,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "seed": lambda row: str(row.seed_policy["seed"]),
        "checkpoint": lambda row: str(bool(row.activation_checkpointing["enabled"])),
        "regime": lambda row: row.regime,
        "coverage": lambda row: ",".join(row.coverage_cell_ids),
    }
    return {
        name: dict(sorted(Counter(key(row) for row in candidates).items()))
        for name, key in dimensions.items()
    }


def _emit_profile(path: Path) -> None:
    from .labeler_profile import PROFILE
    from .model_registry import load_model_registry
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    manifest = build_target_manifest()
    summary = {
        "profile": PROFILE.name,
        "manifest_sha256": manifest.sha256,
        "task_registry_sha256": load_task_registry().sha256,
        "model_registry_sha256": load_model_registry().sha256,
        "candidate_count": len(manifest.candidates),
        "precision_counts": dict(
            sorted(
                Counter(
                    row.precision_policy["policy_id"] for row in manifest.candidates
                ).items()
            )
        ),
        "distribution_sha256": canonical_sha256(
            _distribution_payload(manifest.candidates)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": "summary", **summary}, sort_keys=True) + "\n")
        for row in manifest.candidates:
            stream.write(
                json.dumps(
                    canonical_value(
                        {
                            "kind": "candidate",
                            "ordinal": row.ordinal,
                            "candidate_id": row.candidate_id,
                            "task_id": row.task_id,
                            "semantic_distribution_signature": semantic_distribution_signature(
                                row
                            ),
                            "task_independent_compute_signature": task_independent_compute_signature(
                                row
                            ),
                        }
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _read_emission(
    path: Path,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    with path.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
        rows = tuple(json.loads(line) for line in stream if line.strip())
    if first.pop("kind", None) != "summary" or any(
        row.pop("kind", None) != "candidate" for row in rows
    ):
        raise A10SpeechCrosswalkError("profile emission schema differs")
    return first, rows


def _profile_emission(
    profile: str,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    with tempfile.TemporaryDirectory(prefix="perfseer-a10-speech-crosswalk-") as directory:
        output = Path(directory) / f"{profile}.jsonl"
        environment = os.environ.copy()
        environment["PERFSEER_LABELER_PROFILE"] = profile
        subprocess.run(
            [sys.executable, "-m", __name__, "--emit", str(output)],
            check=True,
            env=environment,
            timeout=300,
        )
        return _read_emission(output)


@dataclass(frozen=True)
class A10SpeechCrosswalk:
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        if self.summary.get("version") != CROSSWALK_VERSION:
            raise A10SpeechCrosswalkError("crosswalk version differs")
        if len(self.rows) != 18_000 or tuple(
            row.get("ordinal") for row in self.rows
        ) != tuple(range(18_000)):
            raise A10SpeechCrosswalkError("crosswalk rows/ordinals differ")
        expected_keys = {
            "version",
            "ordinal",
            "original_a10g_candidate_id",
            "native_v1_candidate_id",
            "native_v2_candidate_id",
            "row_classification",
            "old_task_id",
            "new_task_id",
            "old_semantic_signature",
            "new_semantic_signature",
            "task_independent_compute_signature",
        }
        for row in self.rows:
            if set(row) != expected_keys or row["version"] != CROSSWALK_ROW_VERSION:
                raise A10SpeechCrosswalkError("crosswalk row schema differs")
            for key in (
                "original_a10g_candidate_id",
                "native_v1_candidate_id",
                "native_v2_candidate_id",
                "old_semantic_signature",
                "new_semantic_signature",
                "task_independent_compute_signature",
            ):
                value = row[key]
                if not isinstance(value, str) or len(value) != 64:
                    raise A10SpeechCrosswalkError(f"crosswalk {key} is not a digest")
        for key in (
            "original_a10g_candidate_id",
            "native_v1_candidate_id",
            "native_v2_candidate_id",
        ):
            if len({row[key] for row in self.rows}) != 18_000:
                raise A10SpeechCrosswalkError(f"crosswalk {key} is not bijective")
        if dict(Counter(row["row_classification"] for row in self.rows)) != EXPECTED_CLASS_COUNTS:
            raise A10SpeechCrosswalkError("crosswalk classification counts differ")
        if self.summary.get("rows_sha256") != canonical_sha256(self.rows):
            raise A10SpeechCrosswalkError("crosswalk row hash differs")
        unhashed = dict(self.summary)
        declared = unhashed.pop("crosswalk_sha256", None)
        if declared != canonical_sha256(unhashed):
            raise A10SpeechCrosswalkError("crosswalk summary hash differs")


def build_crosswalk(v2_manifest: Any | None = None) -> A10SpeechCrosswalk:
    from .labeler_profile import PROFILE
    from .model_registry import load_model_registry
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    if PROFILE.name != "native_a10_speech_v2":
        raise A10SpeechCrosswalkError(
            "speech crosswalk requires the native_a10_speech_v2 process profile"
        )
    manifest = v2_manifest or build_target_manifest()
    manifest.validate()
    legacy_summary, legacy_rows = _profile_emission("legacy_a10g")
    v1_summary, v1_rows = _profile_emission("native_a10")
    expected_legacy = {
        "profile": "legacy_a10g",
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "task_registry_sha256": REFERENCE_TASK_REGISTRY_SHA256,
        "model_registry_sha256": REFERENCE_MODEL_REGISTRY_SHA256,
        "candidate_count": 18_000,
        "precision_counts": EXPECTED_PRECISION_COUNTS,
    }
    if any(legacy_summary.get(key) != value for key, value in expected_legacy.items()):
        raise A10SpeechCrosswalkError("AWS A10G reference emission drifted")
    if (
        v1_summary.get("profile") != "native_a10"
        or v1_summary.get("manifest_sha256") != NATIVE_V1_MANIFEST_SHA256
        or v1_summary.get("candidate_count") != 18_000
        or v1_summary.get("precision_counts") != EXPECTED_PRECISION_COUNTS
    ):
        raise A10SpeechCrosswalkError("native A10 V1 emission drifted")
    if not len(legacy_rows) == len(v1_rows) == len(manifest.candidates):
        raise A10SpeechCrosswalkError("profile emission row counts differ")
    rows = []
    for legacy, v1, v2 in zip(legacy_rows, v1_rows, manifest.candidates, strict=True):
        if legacy["ordinal"] != v1["ordinal"] or v1["ordinal"] != v2.ordinal:
            raise A10SpeechCrosswalkError("profile ordinals differ")
        if legacy["semantic_distribution_signature"] != v1["semantic_distribution_signature"]:
            raise A10SpeechCrosswalkError("historical native row differs from AWS semantics")
        v2_compute = task_independent_compute_signature(v2)
        if v1["task_independent_compute_signature"] != v2_compute:
            raise A10SpeechCrosswalkError(
                f"task-independent compute signature differs at ordinal {v2.ordinal}"
            )
        if v1["candidate_id"] == v2.candidate_id:
            classification = "unchanged"
            if v1["task_id"] != v2.task_id:
                raise A10SpeechCrosswalkError("an unchanged ID changed task identity")
        elif v1["task_id"] == "mlsp-2013-birds" and v2.task_id == "mlsp-2013-birds":
            classification = "audio_registry_rebound"
        elif (
            v1["task_id"] == HISTORICAL_WHALE_TASK_ID
            and v2.task_id == SPEECH_TASK_ID
        ):
            classification = "dataset_substitution"
        else:
            raise A10SpeechCrosswalkError(
                f"unexpected V2 identity change at ordinal {v2.ordinal}"
            )
        rows.append(
            canonical_value(
                {
                    "version": CROSSWALK_ROW_VERSION,
                    "ordinal": v2.ordinal,
                    "original_a10g_candidate_id": legacy["candidate_id"],
                    "native_v1_candidate_id": v1["candidate_id"],
                    "native_v2_candidate_id": v2.candidate_id,
                    "row_classification": classification,
                    "old_task_id": v1["task_id"],
                    "new_task_id": v2.task_id,
                    "old_semantic_signature": v1[
                        "semantic_distribution_signature"
                    ],
                    "new_semantic_signature": semantic_distribution_signature(v2),
                    "task_independent_compute_signature": v2_compute,
                }
            )
        )
    immutable_rows = tuple(rows)
    class_counts = dict(Counter(row["row_classification"] for row in immutable_rows))
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise A10SpeechCrosswalkError("V2 changed IDs outside the approved audio scope")
    distributions = _distribution_payload(manifest.candidates)
    if canonical_sha256(distributions) != v1_summary["distribution_sha256"]:
        raise A10SpeechCrosswalkError("task-independent aggregate distributions drifted")
    substitution = load_speech_substitution_contract()
    summary: dict[str, Any] = {
        "version": CROSSWALK_VERSION,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "original_a10g_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "original_a10g_task_registry_sha256": REFERENCE_TASK_REGISTRY_SHA256,
        "native_v1_manifest_sha256": NATIVE_V1_MANIFEST_SHA256,
        "native_v1_crosswalk_sha256": NATIVE_V1_CROSSWALK_SHA256,
        "native_v2_manifest_sha256": manifest.sha256,
        "native_v2_task_registry_sha256": load_task_registry().sha256,
        "native_v2_model_registry_sha256": load_model_registry().sha256,
        "substitution_contract_sha256": substitution.sha256,
        "row_count": len(immutable_rows),
        "measured_epoch_count": len(immutable_rows) * 3,
        "precision_counts": EXPECTED_PRECISION_COUNTS,
        "classification_counts": class_counts,
        "task_independent_distribution_sha256": canonical_sha256(distributions),
        "rows_sha256": canonical_sha256(immutable_rows),
    }
    summary["crosswalk_sha256"] = canonical_sha256(summary)
    result = A10SpeechCrosswalk(canonical_value(summary), immutable_rows)
    result.validate()
    return result


def freeze_crosswalk(
    workspace: str | Path, v2_manifest: Any | None = None
) -> A10SpeechCrosswalk:
    from .storage import atomic_write_json, atomic_write_jsonl

    root = Path(workspace).resolve() / "state"
    result = build_crosswalk(v2_manifest)
    summary_path = root / "a10_speech_v2_crosswalk_summary.json"
    rows_path = root / "a10_speech_v2_crosswalk.jsonl"
    if summary_path.exists() or rows_path.exists():
        if not summary_path.is_file() or not rows_path.is_file():
            raise A10SpeechCrosswalkError("V2 crosswalk state is partially present")
        stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stored_rows = tuple(
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        stored = A10SpeechCrosswalk(stored_summary, stored_rows)
        stored.validate()
        if stored != result:
            raise A10SpeechCrosswalkError("workspace V2 crosswalk differs")
        return stored
    atomic_write_jsonl(rows_path, result.rows)
    atomic_write_json(summary_path, result.summary)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.emit is None:
        raise SystemExit("--emit is required for the internal profile emitter")
    _emit_profile(arguments.emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A10SpeechCrosswalk",
    "A10SpeechCrosswalkError",
    "CROSSWALK_VERSION",
    "EXPECTED_CLASS_COUNTS",
    "NATIVE_V1_CROSSWALK_SHA256",
    "NATIVE_V1_MANIFEST_SHA256",
    "build_crosswalk",
    "freeze_crosswalk",
    "task_independent_compute_payload",
    "task_independent_compute_signature",
]
