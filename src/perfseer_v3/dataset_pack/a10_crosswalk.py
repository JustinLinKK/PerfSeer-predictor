"""Frozen AWS-A10G to native NRP-A10 identity crosswalk."""

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
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_SOURCE_COMMIT = "d7abb69b3c79e65e2f3834065ce284484a020ad4"
REFERENCE_MANIFEST_SHA256 = "bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e"
REFERENCE_TASK_REGISTRY_SHA256 = "781b93ddc020d7cbc77d816458fc213065088a27b9e72d42290ddd8e656de049"
REFERENCE_MODEL_REGISTRY_SHA256 = "ad1027aa906ff8dbc23c24b921ea85496b473b7eeb88fcce10628d142a35c9e2"
CROSSWALK_VERSION = "perfseer_v3_nrp_a10_18k_crosswalk_v1"
CROSSWALK_ROW_VERSION = "perfseer_v3_nrp_a10_crosswalk_row_v1"
EXPECTED_PRECISION_COUNTS = {
    "bf16": 5_232,
    "fp16_grad_scaler": 4_171,
    "fp32_tf32": 5_193,
    "mixed_structured": 3_404,
}
SEMANTIC_DISTRIBUTION_FIELDS = (
    "task_id",
    "family_id",
    "quota_modality",
    "source_modality",
    "architecture_parameters",
    "precision_policy",
    "optimizer",
    "scheduler",
    "execution",
    "batch",
    "seed_policy",
    "coverage_cell_specs",
)


class A10CrosswalkError(RuntimeError):
    """Raised when the frozen reference or native bijection drifts."""


def semantic_distribution_payload(candidate: Any) -> Mapping[str, Any]:
    """Return identity-independent fields that define one workload distribution row."""

    batch = asdict(candidate.batch_plan)
    batch.pop("version")
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
        "batch": {
            **batch,
            "microbatch_size": candidate.microbatch_size,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        },
        "activation_checkpointing": candidate.activation_checkpointing,
        "seed_policy": candidate.seed_policy,
        "coverage_cell_specs": candidate.coverage_cell_specs,
    }


def semantic_distribution_signature(candidate: Any) -> str:
    from .fingerprints import canonical_sha256

    return canonical_sha256(semantic_distribution_payload(candidate))


def _distribution_counters(candidates: Iterable[Any]) -> Mapping[str, Mapping[str, int]]:
    rows = tuple(candidates)
    dimensions = {
        "task": lambda row: row.task_id,
        "family": lambda row: row.family_id,
        "quota_modality": lambda row: row.quota_modality,
        "source_modality": lambda row: row.source_modality,
        "architecture": lambda row: json.dumps(row.architecture_parameters, sort_keys=True, separators=(",", ":")),
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
        "coverage": lambda row: ",".join(row.coverage_cell_ids),
    }
    return {
        name: dict(sorted(Counter(key(row) for row in rows).items()))
        for name, key in dimensions.items()
    }


def _emit_profile(path: Path) -> None:
    from .fingerprints import canonical_sha256, canonical_value
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
            sorted(Counter(row.precision_policy["policy_id"] for row in manifest.candidates).items())
        ),
        "distribution_sha256": canonical_sha256(_distribution_counters(manifest.candidates)),
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
                            "semantic_distribution_signature": semantic_distribution_signature(row),
                        }
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _read_emission(path: Path) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if first.pop("kind", None) != "summary" or any(row.pop("kind", None) != "candidate" for row in rows):
        raise A10CrosswalkError("profile emission schema differs")
    return first, tuple(rows)


def _legacy_emission() -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    with tempfile.TemporaryDirectory(prefix="perfseer-a10-crosswalk-") as directory:
        output = Path(directory) / "legacy.jsonl"
        environment = os.environ.copy()
        environment["PERFSEER_LABELER_PROFILE"] = "legacy_a10g"
        subprocess.run(
            [sys.executable, "-m", __name__, "--emit", str(output)],
            check=True,
            env=environment,
            timeout=300,
        )
        return _read_emission(output)


@dataclass(frozen=True)
class A10Crosswalk:
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        from .fingerprints import canonical_sha256

        if self.summary.get("version") != CROSSWALK_VERSION:
            raise A10CrosswalkError("crosswalk version differs")
        if len(self.rows) != 18_000:
            raise A10CrosswalkError("crosswalk must contain exactly 18,000 rows")
        if tuple(row.get("ordinal") for row in self.rows) != tuple(range(18_000)):
            raise A10CrosswalkError("crosswalk ordinals are not contiguous")
        for key in ("original_a10g_candidate_id", "native_nrp_a10_candidate_id"):
            values = tuple(str(row.get(key, "")) for row in self.rows)
            if len(set(values)) != 18_000 or any(len(value) != 64 for value in values):
                raise A10CrosswalkError(f"crosswalk {key} values are not a bijection")
        if any(
            row.get("version") != CROSSWALK_ROW_VERSION
            or len(str(row.get("semantic_distribution_signature", ""))) != 64
            for row in self.rows
        ):
            raise A10CrosswalkError("crosswalk row schema differs")
        expected_hash = canonical_sha256(self.rows)
        if self.summary.get("rows_sha256") != expected_hash:
            raise A10CrosswalkError("crosswalk row hash differs")
        unhashed = dict(self.summary)
        declared = unhashed.pop("crosswalk_sha256", None)
        if declared != canonical_sha256(unhashed):
            raise A10CrosswalkError("crosswalk summary hash differs")


def build_crosswalk(native_manifest: Any | None = None) -> A10Crosswalk:
    from .fingerprints import canonical_sha256, canonical_value
    from .labeler_profile import PROFILE
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    if PROFILE.name != "native_a10":
        raise A10CrosswalkError("crosswalk construction requires the native_a10 process profile")
    native = native_manifest or build_target_manifest()
    native.validate()
    legacy_summary, legacy_rows = _legacy_emission()
    expected_reference = {
        "profile": "legacy_a10g",
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "task_registry_sha256": REFERENCE_TASK_REGISTRY_SHA256,
        "model_registry_sha256": REFERENCE_MODEL_REGISTRY_SHA256,
        "candidate_count": 18_000,
        "precision_counts": EXPECTED_PRECISION_COUNTS,
    }
    if any(legacy_summary.get(key) != value for key, value in expected_reference.items()):
        raise A10CrosswalkError("legacy emission differs from the frozen source commit")
    native_signatures = tuple(semantic_distribution_signature(row) for row in native.candidates)
    legacy_signatures = tuple(str(row["semantic_distribution_signature"]) for row in legacy_rows)
    if native_signatures != legacy_signatures:
        mismatch = next(index for index, pair in enumerate(zip(native_signatures, legacy_signatures)) if pair[0] != pair[1])
        raise A10CrosswalkError(f"native semantic distribution differs at ordinal {mismatch}")
    native_distributions = _distribution_counters(native.candidates)
    if canonical_sha256(native_distributions) != legacy_summary["distribution_sha256"]:
        raise A10CrosswalkError("native aggregate distributions differ from the reference")
    rows = tuple(
        canonical_value(
            {
                "version": CROSSWALK_ROW_VERSION,
                "ordinal": native_row.ordinal,
                "original_a10g_candidate_id": legacy_row["candidate_id"],
                "native_nrp_a10_candidate_id": native_row.candidate_id,
                "semantic_distribution_signature": signature,
            }
        )
        for native_row, legacy_row, signature in zip(
            native.candidates, legacy_rows, native_signatures, strict=True
        )
    )
    summary: dict[str, Any] = {
        "version": CROSSWALK_VERSION,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "reference_task_registry_sha256": REFERENCE_TASK_REGISTRY_SHA256,
        "reference_model_registry_sha256": REFERENCE_MODEL_REGISTRY_SHA256,
        "native_manifest_sha256": native.sha256,
        "native_task_registry_sha256": load_task_registry().sha256,
        "row_count": len(rows),
        "measured_epoch_count": len(rows) * 3,
        "precision_counts": EXPECTED_PRECISION_COUNTS,
        "distribution_sha256": canonical_sha256(native_distributions),
        "rows_sha256": canonical_sha256(rows),
    }
    summary["crosswalk_sha256"] = canonical_sha256(summary)
    result = A10Crosswalk(canonical_value(summary), rows)
    result.validate()
    return result


def freeze_crosswalk(workspace: str | Path, native_manifest: Any | None = None) -> A10Crosswalk:
    from .storage import atomic_write_json, atomic_write_jsonl

    root = Path(workspace).resolve() / "state"
    result = build_crosswalk(native_manifest)
    summary_path = root / "a10_crosswalk_summary.json"
    rows_path = root / "a10_crosswalk.jsonl"
    if summary_path.exists() or rows_path.exists():
        if not summary_path.is_file() or not rows_path.is_file():
            raise A10CrosswalkError("crosswalk state path is not a regular file")
        stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _, stored_rows = _read_emission_with_crosswalk_rows(rows_path)
        stored = A10Crosswalk(stored_summary, stored_rows)
        stored.validate()
        if stored != result:
            raise A10CrosswalkError("workspace crosswalk differs from the frozen contract")
        return stored
    atomic_write_jsonl(rows_path, result.rows)
    atomic_write_json(summary_path, result.summary)
    return result


def _read_emission_with_crosswalk_rows(path: Path) -> tuple[None, tuple[Mapping[str, Any], ...]]:
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None, rows


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
    "A10Crosswalk",
    "A10CrosswalkError",
    "CROSSWALK_VERSION",
    "EXPECTED_PRECISION_COUNTS",
    "REFERENCE_MANIFEST_SHA256",
    "REFERENCE_MODEL_REGISTRY_SHA256",
    "REFERENCE_SOURCE_COMMIT",
    "REFERENCE_TASK_REGISTRY_SHA256",
    "build_crosswalk",
    "freeze_crosswalk",
    "semantic_distribution_payload",
    "semantic_distribution_signature",
]
