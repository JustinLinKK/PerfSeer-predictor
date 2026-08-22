#!/usr/bin/env python3
"""Exhaustively validate every production-reachable Disaster V2 replacement."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os


PROFILE = "native_a10_nonvision_disaster_v2"
os.environ.setdefault("PERFSEER_LABELER_PROFILE", PROFILE)
if os.environ["PERFSEER_LABELER_PROFILE"] != PROFILE:
    raise SystemExit(f"PERFSEER_LABELER_PROFILE must equal {PROFILE}")

from perfseer_v3.dataset_pack.compatibility import DEPLOYMENT_OPTIMIZERS
from perfseer_v3.dataset_pack.contracts import FailureStage
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.repair import (
    QUARANTINE_VERSION,
    QuarantineRecord,
    make_quota_replacement,
)
from perfseer_v3.dataset_pack.sampler import TargetCandidate, build_target_manifest


def _audit_quarantine(target: TargetCandidate) -> QuarantineRecord:
    draft = QuarantineRecord(
        version=QUARANTINE_VERSION,
        quarantine_id="0" * 64,
        root_candidate_id=target.candidate_id,
        terminal_candidate_id=target.candidate_id,
        terminal_failure_record_sha256="1" * 64,
        oom_repair_attempt_ids=(),
        failure_stage=FailureStage.FORWARD.value,
        reason_code="exhaustive_replacement_audit",
        family_id=target.family_id,
        task_id=target.task_id,
        regime=target.regime,
        coverage_cell_ids=target.coverage_cell_ids,
        batch_one_exhausted=False,
    )
    return replace(
        draft,
        quarantine_id=canonical_sha256(draft.unhashed_payload()),
    )


def main() -> int:
    manifest = build_target_manifest()
    candidate_ids: list[str] = []
    substitution_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    optimizer_counts: Counter[str] = Counter()
    fallback_counts: Counter[int] = Counter()
    for target in manifest.candidates:
        quarantine = _audit_quarantine(target)
        for replacement_index in range(3):
            replacement = make_quota_replacement(
                target,
                quarantine,
                replacement_index=replacement_index,
            )
            candidate = replacement.candidate
            candidate_ids.append(candidate.candidate_id)
            substitution_ids.append(replacement.substitution_id)
            family_counts[candidate.family_id] += 1
            optimizer_counts[str(candidate.optimizer["name"])] += 1
            proposal_index = int(
                candidate.mutation_specification["quota_replacement"].get(
                    "proposal_index",
                    0,
                )
            )
            fallback_counts[proposal_index] += 1
            if candidate.family_id == "fasttext_embeddingbag":
                sparse_gradients = candidate.architecture_parameters[
                    "sparse_gradients"
                ]
                if sparse_gradients is not (
                    candidate.optimizer["name"] == "sparse_adam"
                ):
                    raise AssertionError("fastText replacement optimizer contract drifted")

    expected = len(manifest.candidates) * 3
    if expected != 33_600:
        raise AssertionError("Disaster V2 replacement audit expected 33,600 rows")
    if len(candidate_ids) != expected or len(set(candidate_ids)) != expected:
        raise AssertionError("replacement candidate identities are incomplete or duplicated")
    if len(substitution_ids) != expected or len(set(substitution_ids)) != expected:
        raise AssertionError("replacement substitution identities are incomplete or duplicated")
    if set(optimizer_counts) != set(DEPLOYMENT_OPTIMIZERS):
        raise AssertionError("replacement optimizer coverage is incomplete")

    summary = {
        "candidate_count": len(manifest.candidates),
        "candidate_manifest_sha256": manifest.sha256,
        "family_counts": dict(sorted(family_counts.items())),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "optimizer_counts": dict(sorted(optimizer_counts.items())),
        "replacement_count": expected,
        "replacement_set_sha256": canonical_sha256(
            {
                "candidate_ids": candidate_ids,
                "substitution_ids": substitution_ids,
            }
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
