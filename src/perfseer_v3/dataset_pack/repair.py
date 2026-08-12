"""Strict OOM batch descent, quarantine, replacement, and exact quota filling."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

from .contracts import AttemptStatus, FailureStage, LabelRunRecord
from .fingerprints import canonical_sha256, canonical_value
from .sampler import (
    BATCH_LADDERS,
    BATCH_PLAN_VERSION,
    BatchPlan,
    TargetCandidate,
    TargetManifest,
    _architecture_parameters,
    _batch_plan,
    _input_signature,
    target_candidate_from_dict,
)
from .compatibility import CompatibilityRequest
from .generated_lineages import build_generated_lineage_registry


REPAIR_VERSION = "perfseer_v3_v100_oom_batch_repair_v2"
QUARANTINE_VERSION = "perfseer_v3_v100_quarantine_v2"
SUBSTITUTION_VERSION = "perfseer_v3_v100_quota_substitution_v2"
QUOTA_FILL_VERSION = "perfseer_v3_v100_quota_fill_v2"
OOM_REPAIR_ACTION = "next_lower_power_of_two"


class RepairError(ValueError):
    """Raised when OOM repair or same-cell substitution changes forbidden fields."""


def _sha256(value: str, *, context: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RepairError(f"{context} must be a lowercase SHA-256 digest")


def _failure_stage(value: FailureStage | str) -> str:
    raw = value.value if isinstance(value, FailureStage) else value
    if raw not in {member.value for member in FailureStage}:
        raise RepairError(f"unknown failure stage {raw!r}")
    return raw


def _rehash(candidate: TargetCandidate) -> TargetCandidate:
    draft = replace(candidate, candidate_id="0" * 64)
    return replace(draft, candidate_id=canonical_sha256(draft.unhashed_payload()))


def _bound_failure_sha256(
    record: LabelRunRecord,
    candidate: TargetCandidate,
) -> str:
    from .task_registry import load_task_registry

    task = next(
        (row for row in load_task_registry().entries if row.task_id == candidate.task_id),
        None,
    )
    if task is None:
        raise RepairError("failed record task is absent from the frozen task registry")
    try:
        return record.bound_record_sha256(candidate, task)
    except ValueError as error:
        raise RepairError(
            "failed record is not bound to its terminal configuration/task"
        ) from error


def _tier_for_repaired_batch(parent_tier: str, batch_size: int) -> str:
    if batch_size in BATCH_LADDERS[parent_tier]:
        return parent_tier
    if batch_size >= 8:
        return "standard"
    return "heavy"


def _repaired_batch_plan(parent: BatchPlan, next_batch: int) -> BatchPlan:
    effective_tier = _tier_for_repaired_batch(parent.effective_tier, next_batch)
    ladder = BATCH_LADDERS[effective_tier]
    eligible = tuple(
        value for value in ladder if value <= parent.maximum_batch_for_eight_batches
    )
    reasons = tuple(
        dict.fromkeys(
            (
                *parent.promotion_reasons,
                f"oom_reclassified_from_{parent.effective_tier}"
                if effective_tier != parent.effective_tier
                else "oom_batch_descent",
            )
        )
    )
    plan = BatchPlan(
        version=BATCH_PLAN_VERSION,
        base_tier=parent.base_tier,
        effective_tier=effective_tier,
        full_ladder=ladder,
        eligible_ladder=eligible,
        expected_train_examples=parent.expected_train_examples,
        maximum_batch_for_eight_batches=parent.maximum_batch_for_eight_batches,
        promotion_reasons=reasons,
        selected_microbatch=next_batch,
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class OomRepairAttempt:
    version: str
    attempt_id: str
    root_candidate_id: str
    repair_index: int
    action: str
    reason_code: str
    authorizing_failure_record_sha256: str
    authorizing_run_id: str
    authorizing_status: str
    authorizing_failure_stage: str
    authorizing_cleanup_passed: bool
    parent_candidate: TargetCandidate
    candidate: TargetCandidate

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("attempt_id")
        return payload

    def validate(self) -> None:
        if self.version != REPAIR_VERSION or self.action != OOM_REPAIR_ACTION:
            raise RepairError("OOM repair version or action mismatch")
        _sha256(self.root_candidate_id, context="root candidate ID")
        if type(self.repair_index) is not int or self.repair_index < 0:
            raise RepairError("repair index must be nonnegative")
        if not self.reason_code:
            raise RepairError("OOM reason code cannot be empty")
        _sha256(
            self.authorizing_failure_record_sha256,
            context="authorizing failure record SHA-256",
        )
        if not self.authorizing_run_id:
            raise RepairError("authorizing failure run ID cannot be empty")
        if (
            self.authorizing_status != AttemptStatus.OOM.value
            or self.authorizing_failure_stage != FailureStage.ALLOCATOR.value
            or self.authorizing_cleanup_passed is not True
        ):
            raise RepairError(
                "OOM repair must retain allocator-OOM status and passing cleanup proof"
            )
        self.parent_candidate.validate()
        self.candidate.validate()
        before = self.parent_candidate.microbatch_size
        after = self.candidate.microbatch_size
        if after != before // 2 or before <= 1:
            raise RepairError("OOM repair must use the next lower power of two")
        if before & (before - 1) or after & (after - 1):
            raise RepairError("OOM repair sizes must be powers of two")
        preserved = (
            "quota_modality",
            "source_modality",
            "family_id",
            "regime",
            "source_lineage",
            "source_sha256",
            "source_split",
            "factory_id",
            "architecture_parameters",
            "task_id",
            "task_schema_sha256",
            "target_width",
            "dataset_revision",
            "input_signature",
            "training_step_id",
            "gradient_accumulation_steps",
            "precision_policy",
            "optimizer",
            "scheduler",
            "activation_checkpointing",
            "execution",
            "coverage_cell_ids",
            "compatibility_request_sha256",
            "observation_protocol_sha256",
            "target_hardware_id",
        )
        if any(
            getattr(self.parent_candidate, name) != getattr(self.candidate, name)
            for name in preserved
        ):
            raise RepairError("OOM repair changed a non-batch execution or quota field")
        history = self.candidate.mutation_specification.get("oom_repair_history", ())
        if not history or history[-1].get("parent_candidate_id") != self.parent_candidate.candidate_id:
            raise RepairError("OOM repair history does not bind its parent")
        if self.attempt_id != canonical_sha256(self.unhashed_payload()):
            raise RepairError("OOM repair attempt identity drifted")


def oom_repair_attempt_from_dict(value: Mapping[str, Any]) -> OomRepairAttempt:
    if not isinstance(value, Mapping) or set(value) != set(OomRepairAttempt.__dataclass_fields__):
        raise RepairError("serialized OOM repair attempt schema differs")
    result = OomRepairAttempt(
        **{
            **dict(value),
            "parent_candidate": target_candidate_from_dict(value["parent_candidate"]),
            "candidate": target_candidate_from_dict(value["candidate"]),
        }
    )
    result.validate()
    return result


def next_oom_repair(
    parent: TargetCandidate,
    failed_attempt: LabelRunRecord,
    *,
    repair_index: int,
    reason_code: str = "cuda_out_of_memory",
    root_candidate_id: str | None = None,
) -> OomRepairAttempt:
    """Create the only allowed unchanged-workload retry: halve batch size."""

    parent.validate()
    failed_attempt.validate()
    if (
        failed_attempt.configuration_id != parent.candidate_id
        or failed_attempt.status != AttemptStatus.OOM
        or failed_attempt.failure_stage != FailureStage.ALLOCATOR
        or not failed_attempt.cleanup.passed
        or failed_attempt.targets is not None
    ):
        raise RepairError(
            "OOM repair requires a persisted allocator-OOM attempt with null targets and passing cleanup"
        )
    if parent.microbatch_size <= 1:
        raise RepairError("batch 1 OOM must be quarantined and replaced")
    if type(repair_index) is not int or repair_index < 0:
        raise RepairError("repair index must be nonnegative")
    root = root_candidate_id or parent.candidate_id
    _sha256(root, context="root candidate ID")
    next_batch = parent.microbatch_size // 2
    if next_batch < 1 or parent.microbatch_size & (parent.microbatch_size - 1):
        raise RepairError("OOM repair requires a power-of-two parent batch")
    history = list(parent.mutation_specification.get("oom_repair_history", ()))
    history.append(
        {
            "parent_candidate_id": parent.candidate_id,
            "root_candidate_id": root,
            "repair_index": repair_index,
            "before_batch": parent.microbatch_size,
            "after_batch": next_batch,
            "reason_code": reason_code,
        }
    )
    mutation = dict(parent.mutation_specification)
    mutation["oom_repair_history"] = history
    candidate = _rehash(
        replace(
            parent,
            batch_plan=_repaired_batch_plan(parent.batch_plan, next_batch),
            microbatch_size=next_batch,
            mutation_specification=canonical_value(mutation),
        )
    )
    failure_record_sha256 = _bound_failure_sha256(failed_attempt, parent)
    draft = OomRepairAttempt(
        version=REPAIR_VERSION,
        attempt_id="0" * 64,
        root_candidate_id=root,
        repair_index=repair_index,
        action=OOM_REPAIR_ACTION,
        reason_code=reason_code,
        authorizing_failure_record_sha256=failure_record_sha256,
        authorizing_run_id=failed_attempt.run_id,
        authorizing_status=failed_attempt.status.value,
        authorizing_failure_stage=failed_attempt.failure_stage.value,
        authorizing_cleanup_passed=failed_attempt.cleanup.passed,
        parent_candidate=parent,
        candidate=candidate,
    )
    result = replace(draft, attempt_id=canonical_sha256(draft.unhashed_payload()))
    result.validate()
    return result


@dataclass(frozen=True)
class QuarantineRecord:
    version: str
    quarantine_id: str
    root_candidate_id: str
    terminal_candidate_id: str
    terminal_failure_record_sha256: str
    oom_repair_attempt_ids: tuple[str, ...]
    failure_stage: str
    reason_code: str
    family_id: str
    task_id: str
    regime: str
    coverage_cell_ids: tuple[str, ...]
    batch_one_exhausted: bool

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("quarantine_id")
        return payload

    def validate(self) -> None:
        if self.version != QUARANTINE_VERSION:
            raise RepairError("quarantine version mismatch")
        _sha256(self.root_candidate_id, context="quarantine root candidate ID")
        _sha256(self.terminal_candidate_id, context="quarantine terminal candidate ID")
        _sha256(
            self.terminal_failure_record_sha256,
            context="terminal failure record SHA-256",
        )
        for value in self.oom_repair_attempt_ids:
            _sha256(value, context="OOM repair attempt ID")
        _failure_stage(self.failure_stage)
        if not all((self.reason_code, self.family_id, self.task_id, self.regime, self.coverage_cell_ids)):
            raise RepairError("quarantine reason and quota identity must be complete")
        if type(self.batch_one_exhausted) is not bool:
            raise RepairError("batch_one_exhausted must be boolean")
        if self.quarantine_id != canonical_sha256(self.unhashed_payload()):
            raise RepairError("quarantine identity drifted")


def quarantine_record_from_dict(value: Mapping[str, Any]) -> QuarantineRecord:
    if not isinstance(value, Mapping) or set(value) != set(QuarantineRecord.__dataclass_fields__):
        raise RepairError("serialized quarantine record schema differs")
    result = QuarantineRecord(
        **{
            **dict(value),
            "oom_repair_attempt_ids": tuple(value["oom_repair_attempt_ids"]),
            "coverage_cell_ids": tuple(value["coverage_cell_ids"]),
        }
    )
    result.validate()
    return result


def quarantine_candidate(
    root: TargetCandidate,
    terminal_failed_attempt: LabelRunRecord,
    attempts: tuple[OomRepairAttempt, ...] = (),
    *,
    failure_stage: FailureStage | str,
    reason_code: str,
) -> QuarantineRecord:
    """Quarantine instability immediately, or OOM only after reaching batch one."""

    root.validate()
    parent = root
    for index, attempt in enumerate(attempts):
        attempt.validate()
        if (
            attempt.root_candidate_id != root.candidate_id
            or attempt.repair_index != index
            or attempt.parent_candidate.candidate_id != parent.candidate_id
        ):
            raise RepairError("OOM repair chain is not contiguous")
        parent = attempt.candidate
    stage = _failure_stage(failure_stage)
    batch_one_exhausted = stage == FailureStage.ALLOCATOR.value and parent.microbatch_size == 1
    if stage == FailureStage.ALLOCATOR.value and not batch_one_exhausted:
        raise RepairError("OOM may be quarantined only after batch 1 fails")
    terminal_failure_record_sha256 = _bound_failure_sha256(
        terminal_failed_attempt,
        parent,
    )
    if (
        terminal_failed_attempt.configuration_id != parent.candidate_id
        or terminal_failed_attempt.targets is not None
        or not terminal_failed_attempt.cleanup.passed
        or terminal_failed_attempt.failure_stage.value != stage
        or terminal_failed_attempt.status == AttemptStatus.ACCEPTED
    ):
        raise RepairError(
            "quarantine requires a persisted terminal failure for the terminal candidate with passing cleanup"
        )
    if stage == FailureStage.ALLOCATOR.value and terminal_failed_attempt.status != AttemptStatus.OOM:
        raise RepairError("allocator quarantine requires a persisted OOM terminal failure")
    draft = QuarantineRecord(
        version=QUARANTINE_VERSION,
        quarantine_id="0" * 64,
        root_candidate_id=root.candidate_id,
        terminal_candidate_id=parent.candidate_id,
        terminal_failure_record_sha256=terminal_failure_record_sha256,
        oom_repair_attempt_ids=tuple(row.attempt_id for row in attempts),
        failure_stage=stage,
        reason_code=reason_code,
        family_id=root.family_id,
        task_id=root.task_id,
        regime=root.regime,
        coverage_cell_ids=root.coverage_cell_ids,
        batch_one_exhausted=batch_one_exhausted,
    )
    result = replace(draft, quarantine_id=canonical_sha256(draft.unhashed_payload()))
    result.validate()
    return result


@dataclass(frozen=True)
class QuotaSubstitution:
    version: str
    substitution_id: str
    target_candidate_id: str
    quarantine_id: str
    replacement_index: int
    candidate: TargetCandidate

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("substitution_id")
        return payload

    def validate(self, target: TargetCandidate) -> None:
        if self.version != SUBSTITUTION_VERSION:
            raise RepairError("quota substitution version mismatch")
        if self.target_candidate_id != target.candidate_id:
            raise RepairError("quota substitution targets another candidate")
        _sha256(self.quarantine_id, context="quota substitution quarantine ID")
        if type(self.replacement_index) is not int or self.replacement_index < 0:
            raise RepairError("replacement index must be nonnegative")
        self.candidate.validate()
        if self.candidate.candidate_id == target.candidate_id:
            raise RepairError("replacement must have a new configuration ID")
        preserved = (
            "family_id",
            "quota_modality",
            "source_modality",
            "source_lineage",
            "source_sha256",
            "source_split",
            "factory_id",
            "task_id",
            "task_schema_sha256",
            "target_width",
            "dataset_revision",
            "regime",
            "training_step_id",
            "gradient_accumulation_steps",
            "precision_policy",
            "optimizer",
            "scheduler",
            "activation_checkpointing",
            "execution",
            "coverage_cell_ids",
            "observation_protocol_sha256",
            "target_hardware_id",
        )
        if any(getattr(self.candidate, name) != getattr(target, name) for name in preserved):
            raise RepairError("replacement changed family, task, regime, or quota cell")
        if self.substitution_id != canonical_sha256(self.unhashed_payload()):
            raise RepairError("quota substitution identity drifted")


def quota_substitution_from_dict(
    value: Mapping[str, Any],
    target: TargetCandidate,
) -> QuotaSubstitution:
    if not isinstance(value, Mapping) or set(value) != set(QuotaSubstitution.__dataclass_fields__):
        raise RepairError("serialized quota substitution schema differs")
    result = QuotaSubstitution(
        **{
            **dict(value),
            "candidate": target_candidate_from_dict(value["candidate"]),
        }
    )
    result.validate(target)
    return result


def make_quota_replacement(
    target: TargetCandidate,
    quarantine: QuarantineRecord,
    *,
    replacement_index: int,
) -> QuotaSubstitution:
    quarantine.validate()
    if (
        quarantine.family_id,
        quarantine.task_id,
        quarantine.regime,
        quarantine.coverage_cell_ids,
    ) != (
        target.family_id,
        target.task_id,
        target.regime,
        target.coverage_cell_ids,
    ):
        raise RepairError("quarantine and replacement quota cells differ")
    if type(replacement_index) is not int or replacement_index < 0:
        raise RepairError("replacement index must be nonnegative")
    seed_policy = dict(target.seed_policy)
    seed_policy["seed"] = int(
        canonical_sha256(
            {
                "target_candidate_id": target.candidate_id,
                "quarantine_id": quarantine.quarantine_id,
                "replacement_index": replacement_index,
            }
        )[:8],
        16,
    )
    mutation = dict(target.mutation_specification)
    mutation["quota_replacement"] = {
        "quarantine_id": quarantine.quarantine_id,
        "replacement_index": replacement_index,
        "target_candidate_id": target.candidate_id,
    }
    replacement_ordinal = 18_001 + target.ordinal * 1_000 + replacement_index
    generated_lineage = None
    if target.family_id == "independent_generated":
        generated_lineage = next(
            (
                row
                for row in build_generated_lineage_registry().lineages
                if row.lineage_id == target.source_lineage
            ),
            None,
        )
        if generated_lineage is None:
            raise RepairError("generated replacement root lineage is unregistered")
    architecture = _architecture_parameters(
        target.family_id,
        replacement_ordinal,
        target.regime,
        defaults=target.architecture_parameters,
        generated_lineage=generated_lineage,
    )
    input_signature = _input_signature(
        target.source_modality,
        architecture,
        replacement_ordinal,
        target.regime,
        target_width=target.target_width,
    )
    batch_plan = _batch_plan(
        target.family_id,
        architecture,
        input_signature,
        replacement_ordinal,
        str(target.precision_policy["policy_id"]),
        bool(target.activation_checkpointing["enabled"]),
        str(target.optimizer["name"]),
        target.batch_plan.expected_train_examples,
    )
    request = CompatibilityRequest(
        family_id=target.family_id,
        modality=target.source_modality,
        architecture_parameters=architecture,
        input_signature=input_signature,
        precision_id=str(target.precision_policy["policy_id"]),
        optimizer_id=str(target.optimizer["name"]),
        scheduler_id=str(target.scheduler["name"]),
        execution_mode=str(target.execution["mode"]),
        backend_id=str(target.execution["backend_id"]),
    )
    candidate = _rehash(
        replace(
            target,
            ordinal=replacement_ordinal,
            architecture_parameters=architecture,
            input_signature=input_signature,
            batch_plan=batch_plan,
            microbatch_size=batch_plan.selected_microbatch,
            compatibility_request_sha256=request.sha256,
            seed_policy=canonical_value(seed_policy),
            mutation_specification=canonical_value(mutation),
        )
    )
    draft = QuotaSubstitution(
        version=SUBSTITUTION_VERSION,
        substitution_id="0" * 64,
        target_candidate_id=target.candidate_id,
        quarantine_id=quarantine.quarantine_id,
        replacement_index=replacement_index,
        candidate=candidate,
    )
    result = replace(draft, substitution_id=canonical_sha256(draft.unhashed_payload()))
    result.validate(target)
    return result


@dataclass(frozen=True)
class QuotaFillResult:
    version: str
    target_manifest_sha256: str
    accepted_candidates: tuple[TargetCandidate, ...]
    substitution_ids: tuple[str, ...]
    exact_quota_filled: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self, manifest: TargetManifest | None = None) -> None:
        if self.version != QUOTA_FILL_VERSION:
            raise RepairError("quota-fill result version mismatch")
        _sha256(self.target_manifest_sha256, context="target manifest SHA")
        if manifest is not None and self.target_manifest_sha256 != canonical_sha256(asdict(manifest)):
            raise RepairError("quota-fill result targets another manifest")
        if len(self.accepted_candidates) != 18_000:
            raise RepairError("quota fill must contain exactly 18,000 accepted candidates")
        if len({row.candidate_id for row in self.accepted_candidates}) != 18_000:
            raise RepairError("quota-fill candidate IDs must be unique")
        if any(row.training_approved for row in self.accepted_candidates):
            raise RepairError("quota planning cannot approve training")
        if self.exact_quota_filled is not True:
            raise RepairError("quota-fill result must be exact")
        if manifest is not None:
            expected = Counter((row.family_id, row.task_id, row.regime) for row in manifest.candidates)
            actual = Counter((row.family_id, row.task_id, row.regime) for row in self.accepted_candidates)
            if actual != expected:
                raise RepairError("quota-fill family/task/regime totals drifted")


def fill_to_quota(
    manifest: TargetManifest,
    accepted_candidate_ids: Iterable[str],
    substitutions: Iterable[QuotaSubstitution] = (),
) -> QuotaFillResult:
    """Resolve every frozen quota slot to exactly one successful configuration."""

    manifest.validate()
    targets = {row.candidate_id: row for row in manifest.candidates}
    substitution_rows = tuple(substitutions)
    by_target: dict[str, list[QuotaSubstitution]] = {}
    candidate_to_target: dict[str, str] = {}
    for row in substitution_rows:
        target = targets.get(row.target_candidate_id)
        if target is None:
            raise RepairError("quota substitution references an unknown target")
        row.validate(target)
        by_target.setdefault(row.target_candidate_id, []).append(row)
        if row.candidate.candidate_id in candidate_to_target:
            raise RepairError("replacement candidate is reused across quota slots")
        candidate_to_target[row.candidate.candidate_id] = row.target_candidate_id
    accepted = tuple(accepted_candidate_ids)
    if len(set(accepted)) != len(accepted):
        raise RepairError("accepted candidate IDs contain duplicates")
    if not set(accepted) <= set(targets) | set(candidate_to_target):
        raise RepairError("accepted set contains an unknown candidate ID")
    accepted_set = set(accepted)
    selected: list[TargetCandidate] = []
    selected_substitutions: list[str] = []
    for target in manifest.candidates:
        options: list[tuple[TargetCandidate, str | None]] = []
        if target.candidate_id in accepted_set:
            options.append((target, None))
        options.extend(
            (row.candidate, row.substitution_id)
            for row in by_target.get(target.candidate_id, ())
            if row.candidate.candidate_id in accepted_set
        )
        if len(options) != 1:
            raise RepairError(
                "each quota slot must resolve to exactly one accepted configuration"
            )
        candidate, substitution_id = options[0]
        selected.append(candidate)
        if substitution_id is not None:
            selected_substitutions.append(substitution_id)
    result = QuotaFillResult(
        version=QUOTA_FILL_VERSION,
        target_manifest_sha256=canonical_sha256(asdict(manifest)),
        accepted_candidates=tuple(selected),
        substitution_ids=tuple(selected_substitutions),
        exact_quota_filled=True,
    )
    result.validate(manifest)
    return result


__all__ = [
    "OOM_REPAIR_ACTION",
    "QUARANTINE_VERSION",
    "QUOTA_FILL_VERSION",
    "REPAIR_VERSION",
    "SUBSTITUTION_VERSION",
    "OomRepairAttempt",
    "QuotaFillResult",
    "QuotaSubstitution",
    "QuarantineRecord",
    "RepairError",
    "fill_to_quota",
    "make_quota_replacement",
    "next_oom_repair",
    "oom_repair_attempt_from_dict",
    "quota_substitution_from_dict",
    "quarantine_candidate",
    "quarantine_record_from_dict",
]
