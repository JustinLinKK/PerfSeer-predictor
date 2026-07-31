"""Fail-closed finalization for the exact 18K AWS A10G label campaign."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from perfseer_v3.artifact import TargetTransformV3

from .contracts import (
    AttemptStatus,
    LabelRunRecord,
    TARGET_NAMES,
    label_run_record_from_dict,
)
from .fingerprints import canonical_sha256, canonical_value
from .materialization import (
    TaskCompletionReceipt,
    freeze_initial_target_manifest,
    load_task_loop_state,
)
from .quota import load_quota_plan
from .repair import (
    OomRepairAttempt,
    QuarantineRecord,
    QuotaSubstitution,
    oom_repair_attempt_from_dict,
    quarantine_record_from_dict,
    quota_substitution_from_dict,
)
from .sampler import TargetCandidate, TargetManifest
from .storage import atomic_write_bytes
from .task_registry import TaskRegistryEntry, load_task_registry


FINALIZATION_VERSION = "perfseer_v3_a10g_18k_finalization_v1"
FINAL_LABEL_ROW_VERSION = "perfseer_v3_a10g_final_label_row_v1"
TARGET_TRANSFORM_RECORD_VERSION = "perfseer_v3_a10g_target_transform_v1"
FINAL_RECEIPT_VERSION = "perfseer_v3_a10g_final_receipt_v1"
SPLIT_COUNTS = {"train": 14_400, "validation": 1_800, "test": 1_800}


class FinalizationError(RuntimeError):
    """Raised when campaign evidence cannot prove the exact final contract."""


def _digest(value: str, *, context: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FinalizationError(f"{context} must be a lowercase SHA-256 digest")


_PRESERVED_SLOT_FIELDS = (
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


@dataclass(frozen=True)
class ResolvedAcceptedSlot:
    """One frozen quota root and its complete failure/repair/acceptance chain."""

    root_candidate: TargetCandidate
    attempted_candidates: tuple[TargetCandidate, ...]
    accepted_record: LabelRunRecord
    oom_repair_attempt_ids: tuple[str, ...]
    quarantine_ids: tuple[str, ...]
    substitution_ids: tuple[str, ...]
    repair_lineage_sha256: str

    @classmethod
    def direct(
        cls,
        candidate: TargetCandidate,
        record: LabelRunRecord,
    ) -> "ResolvedAcceptedSlot":
        payload = {
            "root_candidate_id": candidate.candidate_id,
            "attempted_candidate_ids": (candidate.candidate_id,),
            "oom_repair_attempt_ids": (),
            "quarantine_ids": (),
            "substitution_ids": (),
        }
        return cls(candidate, (candidate,), record, (), (), (), canonical_sha256(payload))

    @property
    def accepted_candidate(self) -> TargetCandidate:
        if not self.attempted_candidates:
            raise FinalizationError("resolved slot has no attempted candidates")
        return self.attempted_candidates[-1]

    @property
    def lineage_payload(self) -> Mapping[str, Any]:
        return {
            "root_candidate_id": self.root_candidate.candidate_id,
            "attempted_candidate_ids": tuple(
                row.candidate_id for row in self.attempted_candidates
            ),
            "oom_repair_attempt_ids": self.oom_repair_attempt_ids,
            "quarantine_ids": self.quarantine_ids,
            "substitution_ids": self.substitution_ids,
        }

    def validate(self, task: TaskRegistryEntry) -> None:
        self.root_candidate.validate()
        if not self.attempted_candidates:
            raise FinalizationError("resolved slot must retain at least one attempt")
        candidate_ids: list[str] = []
        for candidate in self.attempted_candidates:
            candidate.validate()
            candidate_ids.append(candidate.candidate_id)
            if any(
                getattr(candidate, name) != getattr(self.root_candidate, name)
                for name in _PRESERVED_SLOT_FIELDS
            ):
                raise FinalizationError(
                    "resolved attempt changed a source, execution, or frozen quota field"
                )
        if self.attempted_candidates[0] != self.root_candidate:
            raise FinalizationError("resolved attempt chain does not start at its frozen root")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise FinalizationError("resolved attempt chain reuses a configuration ID")
        if len(set(self.oom_repair_attempt_ids)) != len(self.oom_repair_attempt_ids):
            raise FinalizationError("resolved slot repeats an OOM repair identity")
        if len(set(self.quarantine_ids)) != len(self.quarantine_ids):
            raise FinalizationError("resolved slot repeats a quarantine identity")
        if len(set(self.substitution_ids)) != len(self.substitution_ids):
            raise FinalizationError("resolved slot repeats a substitution identity")
        if len(self.quarantine_ids) != len(self.substitution_ids):
            raise FinalizationError("every quarantine must have exactly one substitution")
        self.accepted_record.validate_against_configuration(
            self.accepted_candidate,
            task,
        )
        if self.accepted_record.status != AttemptStatus.ACCEPTED:
            raise FinalizationError("resolved slot terminal record is not accepted")
        if self.accepted_record.fingerprints.source_sha256 != self.accepted_candidate.source_sha256:
            raise FinalizationError("accepted source fingerprint differs from its candidate")
        if (
            self.accepted_record.capture_workload_sha256
            != self.accepted_candidate.candidate_id
            or self.accepted_record.profile_workload_sha256
            != self.accepted_candidate.candidate_id
        ):
            raise FinalizationError("accepted workload fingerprint differs from its configuration")
        if self.repair_lineage_sha256 != canonical_sha256(self.lineage_payload):
            raise FinalizationError("resolved repair-lineage hash differs")


@dataclass(frozen=True)
class FinalLabelRow:
    version: str
    ordinal: int
    root_configuration_id: str
    configuration_id: str
    run_id: str
    accepted_record_sha256: str
    repair_lineage_sha256: str
    split: str
    source_group: str
    source_lineage: str
    source_sha256: str
    graph_sha256: str
    family_id: str
    quota_modality: str
    task_id: str
    regime: str
    precision_id: str
    optimizer_id: str
    scheduler_id: str
    execution_mode: str
    initial_microbatch_size: int
    final_microbatch_size: int
    oom_repair_count: int
    replacement_count: int
    target_values: tuple[float, ...]
    dataset_sha256: str
    environment_sha256: str
    hardware_sha256: str
    support_contract_sha256: str
    row_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("row_sha256")
        return payload

    def validate(self) -> None:
        if self.version != FINAL_LABEL_ROW_VERSION:
            raise FinalizationError("final label row version differs")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 18_000:
            raise FinalizationError("final label row ordinal is invalid")
        for name in (
            "root_configuration_id",
            "configuration_id",
            "accepted_record_sha256",
            "repair_lineage_sha256",
            "source_group",
            "source_sha256",
            "graph_sha256",
            "dataset_sha256",
            "environment_sha256",
            "hardware_sha256",
            "support_contract_sha256",
            "row_sha256",
        ):
            _digest(getattr(self, name), context=name)
        if self.split not in SPLIT_COUNTS:
            raise FinalizationError("final label row split is invalid")
        for name in (
            "run_id",
            "source_lineage",
            "family_id",
            "quota_modality",
            "task_id",
            "regime",
            "precision_id",
            "optimizer_id",
            "scheduler_id",
            "execution_mode",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise FinalizationError(f"final label row {name} is empty")
        for name in (
            "initial_microbatch_size",
            "final_microbatch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise FinalizationError(f"final label row {name} is invalid")
        for name in ("oom_repair_count", "replacement_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise FinalizationError(f"final label row {name} is invalid")
        if len(self.target_values) != len(TARGET_NAMES) or any(
            not math.isfinite(value) or value < 0 for value in self.target_values
        ):
            raise FinalizationError("final label row targets violate the six-target contract")
        if self.row_sha256 != canonical_sha256(self.unhashed_payload()):
            raise FinalizationError("final label row hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


@dataclass(frozen=True)
class TargetTransformRecord:
    version: str
    target_names: tuple[str, ...]
    transform: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    constant_target_names: tuple[str, ...]
    fit_split: str
    fit_row_count: int
    split_fingerprint: str
    fit_targets_sha256: str
    transform_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("transform_sha256")
        return payload

    def validate(self) -> None:
        if self.version != TARGET_TRANSFORM_RECORD_VERSION:
            raise FinalizationError("target-transform version differs")
        if self.target_names != TARGET_NAMES or self.transform != "log1p_standardized":
            raise FinalizationError("target-transform order or semantics differ")
        if self.fit_split != "train" or self.fit_row_count != SPLIT_COUNTS["train"]:
            raise FinalizationError("target transform was not fit on the exact train split")
        for value in (*self.mean, *self.std):
            if not math.isfinite(value):
                raise FinalizationError("target transform contains a non-finite value")
        if any(value <= 0 for value in self.std):
            raise FinalizationError("target transform standard deviations must be positive")
        if not set(self.constant_target_names) <= set(TARGET_NAMES):
            raise FinalizationError("target transform constant-target set is invalid")
        _digest(self.split_fingerprint, context="target transform split fingerprint")
        _digest(self.fit_targets_sha256, context="target transform training targets")
        _digest(self.transform_sha256, context="target transform")
        TargetTransformV3(self.mean, self.std, self.transform).validate()
        if self.transform_sha256 != canonical_sha256(self.unhashed_payload()):
            raise FinalizationError("target-transform hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


@dataclass(frozen=True)
class CampaignFinalization:
    rows: tuple[FinalLabelRow, ...]
    failure_index: tuple[Mapping[str, Any], ...]
    target_transform: TargetTransformRecord
    audit_report: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]

    def validate(self) -> None:
        if len(self.rows) != 18_000:
            raise FinalizationError("final campaign must contain exactly 18,000 rows")
        for row in self.rows:
            row.validate()
        if tuple(row.ordinal for row in self.rows) != tuple(range(18_000)):
            raise FinalizationError("final campaign rows are not in frozen ordinal order")
        if len({row.configuration_id for row in self.rows}) != 18_000:
            raise FinalizationError("final campaign configuration IDs are not unique")
        if len({row.run_id for row in self.rows}) != 18_000:
            raise FinalizationError("final campaign run IDs are not unique")
        if Counter(row.split for row in self.rows) != Counter(SPLIT_COUNTS):
            raise FinalizationError("final campaign split counts differ from 80/10/10")
        self.target_transform.validate()
        split_fingerprint = _split_fingerprint(self.rows)
        if self.target_transform != _fit_target_transform(self.rows, split_fingerprint):
            raise FinalizationError("stored target transform differs from train-only recomputation")
        failure_run_ids: set[str] = set()
        failure_configuration_ids: set[str] = set()
        expected_failure_keys = {
            "run_id",
            "configuration_id",
            "status",
            "failure_stage",
            "task_id",
            "record_sha256",
        }
        for row in self.failure_index:
            if not isinstance(row, Mapping) or set(row) != expected_failure_keys:
                raise FinalizationError("failure-index row schema differs")
            _digest(str(row["record_sha256"]), context="failure-index record")
            run_id = str(row["run_id"])
            configuration_id = str(row["configuration_id"])
            if run_id in failure_run_ids or configuration_id in failure_configuration_ids:
                raise FinalizationError("failure-index identities are duplicated")
            failure_run_ids.add(run_id)
            failure_configuration_ids.add(configuration_id)
        report = dict(self.audit_report)
        report_hash = report.pop("report_sha256", None)
        if report_hash != canonical_sha256(report):
            raise FinalizationError("campaign audit report hash differs")
        manifest = dict(self.dataset_manifest)
        manifest_hash = manifest.pop("manifest_sha256", None)
        if manifest_hash != canonical_sha256(manifest):
            raise FinalizationError("final dataset manifest hash differs")
        if (
            self.dataset_manifest.get("accepted_rows_sha256")
            != canonical_sha256(tuple(row.to_dict() for row in self.rows))
            or self.dataset_manifest.get("failure_index_sha256")
            != canonical_sha256(self.failure_index)
            or self.dataset_manifest.get("target_transform_sha256")
            != self.target_transform.transform_sha256
            or self.dataset_manifest.get("audit_report_sha256")
            != self.audit_report.get("report_sha256")
            or self.dataset_manifest.get("split_fingerprint")
            != split_fingerprint
        ):
            raise FinalizationError("final dataset manifest does not bind its artifacts")
        if (
            self.audit_report.get("accepted_run_count") != 18_000
            or self.audit_report.get("measured_epoch_record_count") != 54_000
            or self.audit_report.get("target_normalization")
            != self.target_transform.to_dict()
        ):
            raise FinalizationError("campaign audit does not reflect final count/normalization evidence")


@dataclass(frozen=True)
class _SplitComponent:
    component_id: str
    source_groups: tuple[str, ...]
    row_indexes: tuple[int, ...]
    forced_test: bool

    @property
    def size(self) -> int:
        return len(self.row_indexes)


def _source_group(candidate: TargetCandidate) -> str:
    return canonical_sha256(
        {
            "source_lineage": candidate.source_lineage,
            "source_sha256": candidate.source_sha256,
        }
    )


def _split_components(
    candidates: Sequence[TargetCandidate],
    graph_sha256s: Sequence[str],
) -> tuple[_SplitComponent, ...]:
    source_groups = tuple(_source_group(row) for row in candidates)
    parent = {value: value for value in source_groups}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        before, after = sorted((left_root, right_root))
        parent[after] = before

    graph_sources: dict[str, set[str]] = defaultdict(set)
    for source_group, graph_sha256 in zip(source_groups, graph_sha256s, strict=True):
        graph_sources[graph_sha256].add(source_group)
    for linked in graph_sources.values():
        ordered = sorted(linked)
        for source_group in ordered[1:]:
            union(ordered[0], source_group)

    component_sources: dict[str, set[str]] = defaultdict(set)
    component_rows: dict[str, list[int]] = defaultdict(list)
    for index, source_group in enumerate(source_groups):
        component = find(source_group)
        component_sources[component].add(source_group)
        component_rows[component].append(index)
    results = []
    for component in sorted(component_rows):
        indexes = tuple(component_rows[component])
        sources = tuple(sorted(component_sources[component]))
        identifier = canonical_sha256({"source_groups": sources})
        forced_test = any(candidates[index].source_split == "held_out" for index in indexes)
        results.append(_SplitComponent(identifier, sources, indexes, forced_test))
    return tuple(results)


def _exact_subset(
    components: Sequence[_SplitComponent],
    target: int,
) -> tuple[str, ...]:
    if target < 0:
        raise FinalizationError("forced split groups exceed their exact row budget")
    states: dict[int, tuple[str, ...]] = {0: ()}
    by_id = {row.component_id: row for row in components}
    for component in sorted(components, key=lambda row: row.component_id):
        for total, selected in tuple(sorted(states.items(), reverse=True)):
            next_total = total + component.size
            if next_total > target:
                continue
            proposal = (*selected, component.component_id)
            current = states.get(next_total)
            if current is None or proposal < current:
                states[next_total] = proposal
    selected = states.get(target)
    if selected is None:
        sizes = Counter(row.size for row in components)
        raise FinalizationError(
            f"whole leakage groups cannot fill exact split size {target}; sizes={dict(sizes)}"
        )
    if sum(by_id[value].size for value in selected) != target:
        raise FinalizationError("exact split subset has an internal size mismatch")
    return selected


def _assign_splits(
    candidates: Sequence[TargetCandidate],
    graph_sha256s: Sequence[str],
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    components = _split_components(candidates, graph_sha256s)
    forced = tuple(row for row in components if row.forced_test)
    development = tuple(row for row in components if not row.forced_test)
    forced_count = sum(row.size for row in forced)
    test_extra = set(_exact_subset(development, SPLIT_COUNTS["test"] - forced_count))
    test_ids = {row.component_id for row in forced} | test_extra
    after_test = tuple(row for row in development if row.component_id not in test_ids)
    validation_ids = set(_exact_subset(after_test, SPLIT_COUNTS["validation"]))
    split_by_component = {
        row.component_id: (
            "test"
            if row.component_id in test_ids
            else "validation"
            if row.component_id in validation_ids
            else "train"
        )
        for row in components
    }
    assignments = [""] * len(candidates)
    for component in components:
        for index in component.row_indexes:
            assignments[index] = split_by_component[component.component_id]
    counts = Counter(assignments)
    if counts != Counter(SPLIT_COUNTS):
        raise FinalizationError(f"exact grouped split counts differ: {dict(counts)}")
    held_out_lineages = sorted(
        {
            row.source_lineage
            for row, split in zip(candidates, assignments, strict=True)
            if row.family_id == "independent_generated"
            and row.source_split == "held_out"
            and split == "test"
        }
    )
    required = load_quota_plan().generated_lineages_held_out_minimum
    if len(held_out_lineages) < required:
        raise FinalizationError("generated held-out lineage minimum is not in the test split")
    for lineage in held_out_lineages:
        lineage_splits = {
            split
            for row, split in zip(candidates, assignments, strict=True)
            if row.source_lineage == lineage
        }
        if lineage_splits != {"test"}:
            raise FinalizationError("a held-out generated lineage crosses final splits")
    return tuple(assignments), canonical_value(
        {
            "counts": dict(sorted(counts.items())),
            "component_count": len(components),
            "source_group_count": len({_source_group(row) for row in candidates}),
            "graph_identity_count": len(set(graph_sha256s)),
            "held_out_generated_lineages": held_out_lineages,
            "source_leakage": False,
            "graph_leakage": False,
        }
    )


def _split_fingerprint(rows: Sequence[FinalLabelRow]) -> str:
    return canonical_sha256(
        tuple(
            {
                "configuration_id": row.configuration_id,
                "source_group": row.source_group,
                "graph_sha256": row.graph_sha256,
                "split": row.split,
            }
            for row in rows
        )
    )


def _fit_target_transform(
    rows: Sequence[FinalLabelRow],
    split_fingerprint: str,
) -> TargetTransformRecord:
    training = tuple(row for row in rows if row.split == "train")
    if len(training) != SPLIT_COUNTS["train"]:
        raise FinalizationError("target transform requires the exact train split")
    columns = tuple(
        tuple(math.log1p(row.target_values[index]) for row in training)
        for index in range(len(TARGET_NAMES))
    )
    means = tuple(math.fsum(column) / len(column) for column in columns)
    raw_std = tuple(
        math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column))
        for column, mean in zip(columns, means, strict=True)
    )
    constant = tuple(
        name for name, value in zip(TARGET_NAMES, raw_std, strict=True) if value <= 1e-12
    )
    std = tuple(1.0 if value <= 1e-12 else value for value in raw_std)
    fit_hash = canonical_sha256(
        tuple((row.configuration_id, row.target_values) for row in training)
    )
    draft = TargetTransformRecord(
        version=TARGET_TRANSFORM_RECORD_VERSION,
        target_names=TARGET_NAMES,
        transform="log1p_standardized",
        mean=means,
        std=std,
        constant_target_names=constant,
        fit_split="train",
        fit_row_count=len(training),
        split_fingerprint=split_fingerprint,
        fit_targets_sha256=fit_hash,
        transform_sha256="",
    )
    result = replace(draft, transform_sha256=canonical_sha256(draft.unhashed_payload()))
    result.validate()
    return result


def _counts(values: Iterable[str]) -> Mapping[str, int]:
    return dict(sorted(Counter(values).items()))


def _identity_summary(values: Iterable[str]) -> Mapping[str, Any]:
    unique = tuple(sorted(set(values)))
    return {"unique_count": len(unique), "identity_set_sha256": canonical_sha256(unique)}


def build_campaign_finalization(
    manifest: TargetManifest,
    resolved_slots: Sequence[ResolvedAcceptedSlot],
    failed_records: Sequence[LabelRunRecord] = (),
    *,
    campaign_provenance: Mapping[str, Any] | None = None,
) -> CampaignFinalization:
    """Build the deterministic final label pack without reading or fabricating labels."""

    manifest.validate()
    quota = load_quota_plan()
    tasks = load_task_registry()
    task_by_id = {row.task_id: row for row in tasks.entries}
    roots = {row.candidate_id: row for row in manifest.candidates}
    slots_by_root = {row.root_candidate.candidate_id: row for row in resolved_slots}
    if len(slots_by_root) != len(resolved_slots) or set(slots_by_root) != set(roots):
        raise FinalizationError("resolved slots do not match all 18,000 frozen roots exactly")
    ordered_slots = tuple(slots_by_root[row.candidate_id] for row in manifest.candidates)
    attempted_ids: set[str] = set()
    attempted_candidates_by_id: dict[str, TargetCandidate] = {}
    expected_failed_ids: set[str] = set()
    all_sidecar_ids: set[str] = set()
    for slot in ordered_slots:
        task = task_by_id.get(slot.root_candidate.task_id)
        if task is None:
            raise FinalizationError("resolved slot task is absent from the frozen registry")
        slot.validate(task)
        identifiers = {row.candidate_id for row in slot.attempted_candidates}
        if attempted_ids & identifiers:
            raise FinalizationError("one attempted configuration is reused across quota slots")
        attempted_ids.update(identifiers)
        attempted_candidates_by_id.update(
            {row.candidate_id: row for row in slot.attempted_candidates}
        )
        expected_failed_ids.update(row.candidate_id for row in slot.attempted_candidates[:-1])
        sidecars = {
            *slot.oom_repair_attempt_ids,
            *slot.quarantine_ids,
            *slot.substitution_ids,
        }
        if all_sidecar_ids & sidecars:
            raise FinalizationError("one repair sidecar is reused across quota slots")
        all_sidecar_ids.update(sidecars)

    failures_by_configuration: dict[str, LabelRunRecord] = {}
    failure_run_ids: set[str] = set()
    for record in failed_records:
        record.validate()
        if record.status == AttemptStatus.ACCEPTED or record.targets is not None:
            raise FinalizationError("failure diagnostics contain an accepted/targeted record")
        if record.configuration_id in failures_by_configuration:
            raise FinalizationError("one configuration has multiple terminal failures")
        if record.run_id in failure_run_ids:
            raise FinalizationError("failed run IDs are duplicated")
        candidate = attempted_candidates_by_id.get(record.configuration_id)
        if candidate is None:
            raise FinalizationError("failure diagnostics reference an unknown attempt")
        task = task_by_id[candidate.task_id]
        record.validate_against_configuration(candidate, task)
        if record.fingerprints.source_sha256 != candidate.source_sha256:
            raise FinalizationError("failed source fingerprint differs from its candidate")
        if (
            record.capture_workload_sha256 != candidate.candidate_id
            or record.profile_workload_sha256 != candidate.candidate_id
        ):
            raise FinalizationError("failed workload fingerprint differs from its configuration")
        failures_by_configuration[record.configuration_id] = record
        failure_run_ids.add(record.run_id)
    if set(failures_by_configuration) != expected_failed_ids:
        raise FinalizationError("failed attempts do not exactly match nonterminal repair attempts")

    accepted_records = tuple(row.accepted_record for row in ordered_slots)
    accepted_ids = {row.configuration_id for row in accepted_records}
    accepted_run_ids = {row.run_id for row in accepted_records}
    if len(accepted_ids) != 18_000 or len(accepted_run_ids) != 18_000:
        raise FinalizationError("accepted configuration/run IDs are not exactly unique 18K sets")
    if accepted_run_ids & failure_run_ids:
        raise FinalizationError("accepted and failed run IDs overlap")
    if accepted_ids & set(failures_by_configuration):
        raise FinalizationError("accepted and failed configuration IDs overlap")
    if sum(len(row.epoch_measurements) for row in accepted_records) != 54_000:
        raise FinalizationError("accepted records do not contain exactly 54,000 epoch records")

    final_candidates = tuple(row.accepted_candidate for row in ordered_slots)
    quota_axes = {
        "family": lambda row: row.family_id,
        "modality": lambda row: row.quota_modality,
        "task": lambda row: row.task_id,
        "regime": lambda row: row.regime,
        "precision": lambda row: str(row.precision_policy["policy_id"]),
        "optimizer": lambda row: str(row.optimizer["name"]),
        "scheduler": lambda row: str(row.scheduler["name"]),
        "execution": lambda row: str(row.execution["mode"]),
        "generated_lineage": lambda row: (
            row.source_lineage if row.family_id == "independent_generated" else "not_generated"
        ),
    }
    quota_report: dict[str, Any] = {}
    for name, accessor in quota_axes.items():
        expected = _counts(accessor(row) for row in manifest.candidates)
        actual = _counts(accessor(row) for row in final_candidates)
        if actual != expected:
            raise FinalizationError(f"final {name} quotas drifted from the frozen manifest")
        quota_report[name] = {"expected": expected, "accepted": actual, "exact": True}

    assignments, split_report = _assign_splits(
        final_candidates,
        tuple(row.fingerprints.graph_sha256 for row in accepted_records),
    )
    rows: list[FinalLabelRow] = []
    for slot, candidate, record, split in zip(
        ordered_slots,
        final_candidates,
        accepted_records,
        assignments,
        strict=True,
    ):
        if record.targets is None:
            raise FinalizationError("accepted record has no target vector")
        task = task_by_id[candidate.task_id]
        record_sha = record.bound_record_sha256(candidate, task)
        draft = FinalLabelRow(
            version=FINAL_LABEL_ROW_VERSION,
            ordinal=slot.root_candidate.ordinal,
            root_configuration_id=slot.root_candidate.candidate_id,
            configuration_id=candidate.candidate_id,
            run_id=record.run_id,
            accepted_record_sha256=record_sha,
            repair_lineage_sha256=slot.repair_lineage_sha256,
            split=split,
            source_group=_source_group(candidate),
            source_lineage=candidate.source_lineage,
            source_sha256=candidate.source_sha256,
            graph_sha256=record.fingerprints.graph_sha256,
            family_id=candidate.family_id,
            quota_modality=candidate.quota_modality,
            task_id=candidate.task_id,
            regime=candidate.regime,
            precision_id=str(candidate.precision_policy["policy_id"]),
            optimizer_id=str(candidate.optimizer["name"]),
            scheduler_id=str(candidate.scheduler["name"]),
            execution_mode=str(candidate.execution["mode"]),
            initial_microbatch_size=slot.root_candidate.microbatch_size,
            final_microbatch_size=candidate.microbatch_size,
            oom_repair_count=len(slot.oom_repair_attempt_ids),
            replacement_count=len(slot.substitution_ids),
            target_values=tuple(float(value) for value in record.targets.values),
            dataset_sha256=str(record.fingerprints.dataset_sha256),
            environment_sha256=record.fingerprints.environment_sha256,
            hardware_sha256=record.fingerprints.hardware_sha256,
            support_contract_sha256=record.fingerprints.support_contract_sha256,
            row_sha256="",
        )
        row = replace(draft, row_sha256=canonical_sha256(draft.unhashed_payload()))
        row.validate()
        rows.append(row)
    final_rows = tuple(rows)
    split_fingerprint = _split_fingerprint(final_rows)
    transform = _fit_target_transform(final_rows, split_fingerprint)

    failure_index = tuple(
        canonical_value(
            {
                "run_id": record.run_id,
                "configuration_id": record.configuration_id,
                "status": record.status.value,
                "failure_stage": record.failure_stage.value,
                "task_id": record.task_id,
                "record_sha256": canonical_sha256(asdict(record)),
            }
        )
        for record in sorted(failed_records, key=lambda row: row.run_id)
    )
    spreads = tuple(float(row.accepted_record.epoch_time_relative_spread) for row in ordered_slots)
    repair_report = {
        "slots_with_oom_repair": sum(bool(row.oom_repair_attempt_ids) for row in ordered_slots),
        "slots_with_replacement": sum(bool(row.substitution_ids) for row in ordered_slots),
        "oom_repair_attempt_count": sum(len(row.oom_repair_attempt_ids) for row in ordered_slots),
        "quarantine_count": sum(len(row.quarantine_ids) for row in ordered_slots),
        "replacement_count": sum(len(row.substitution_ids) for row in ordered_slots),
        "batch_changed_row_count": sum(
            row.root_candidate.microbatch_size != row.accepted_candidate.microbatch_size
            for row in ordered_slots
        ),
        "initial_batch_counts": _counts(str(row.root_candidate.microbatch_size) for row in ordered_slots),
        "final_batch_counts": _counts(str(row.accepted_candidate.microbatch_size) for row in ordered_slots),
        "initial_tier_counts": _counts(row.root_candidate.batch_plan.effective_tier for row in ordered_slots),
        "final_tier_counts": _counts(row.accepted_candidate.batch_plan.effective_tier for row in ordered_slots),
    }
    coverage_report = {
        name: _counts(accessor(row) for row in final_candidates)
        for name, accessor in quota_axes.items()
        if name != "generated_lineage"
    }
    coverage_report["coverage_cell_identity"] = _identity_summary(
        cell for row in final_candidates for cell in row.coverage_cell_ids
    )
    from .local_validation import signature_factors

    coverage_report["execution_signature_identity"] = _identity_summary(
        canonical_sha256(signature_factors(row)) for row in final_candidates
    )
    coverage_report["configuration_count"] = len(final_candidates)
    provenance_report = {
        "source": _identity_summary(row.source_sha256 for row in final_rows),
        "graph": _identity_summary(row.graph_sha256 for row in final_rows),
        "dataset": _identity_summary(row.dataset_sha256 for row in final_rows),
        "environment": _identity_summary(row.environment_sha256 for row in final_rows),
        "hardware": _identity_summary(row.hardware_sha256 for row in final_rows),
        "support_contract": _identity_summary(
            row.support_contract_sha256 for row in final_rows
        ),
        "campaign_evidence": canonical_value(dict(campaign_provenance or {})),
    }
    audit_payload = canonical_value(
        {
            "version": FINALIZATION_VERSION,
            "target_manifest_sha256": manifest.sha256,
            "accepted_run_count": len(final_rows),
            "measured_epoch_record_count": 54_000,
            "quota": quota_report,
            "failure": {
                "attempt_count": len(failure_index),
                "status_counts": _counts(row.status.value for row in failed_records),
                "stage_counts": _counts(row.failure_stage.value for row in failed_records),
                "excluded_from_success_manifest": True,
            },
            "batch_repair": repair_report,
            "stability": {
                "maximum_allowed_relative_spread": quota.protocol.maximum_epoch_time_relative_spread,
                "minimum": min(spreads),
                "mean": math.fsum(spreads) / len(spreads),
                "maximum": max(spreads),
                "all_accepted": max(spreads)
                <= quota.protocol.maximum_epoch_time_relative_spread,
            },
            "execution_signature_coverage": coverage_report,
            "provenance": provenance_report,
            "deduplication": {
                "successful_rows_retained": len(final_rows),
                "source_identity": provenance_report["source"],
                "graph_identity": provenance_report["graph"],
                "identities_canonicalized_without_dropping_configurations": True,
            },
            "split": {**split_report, "split_fingerprint": split_fingerprint},
            "target_normalization": transform.to_dict(),
        }
    )
    audit_report = {
        **audit_payload,
        "report_sha256": canonical_sha256(audit_payload),
    }
    accepted_rows_sha = canonical_sha256(tuple(row.to_dict() for row in final_rows))
    failure_index_sha = canonical_sha256(failure_index)
    manifest_payload = canonical_value(
        {
            "version": FINALIZATION_VERSION,
            "artifact_scope": "a10g_label_pack",
            "accepted_a10g_measurement": True,
            "target_hardware_id": quota.target_hardware_id,
            "target_manifest_sha256": manifest.sha256,
            "target_names": TARGET_NAMES,
            "accepted_run_count": 18_000,
            "measured_epoch_record_count": 54_000,
            "split_counts": SPLIT_COUNTS,
            "split_fingerprint": split_fingerprint,
            "accepted_rows_sha256": accepted_rows_sha,
            "failure_index_sha256": failure_index_sha,
            "target_transform_sha256": transform.transform_sha256,
            "audit_report_sha256": audit_report["report_sha256"],
            "graph_ir_materialized": False,
        }
    )
    dataset_manifest = {
        **manifest_payload,
        "manifest_sha256": canonical_sha256(manifest_payload),
    }
    result = CampaignFinalization(
        final_rows,
        failure_index,
        transform,
        audit_report,
        dataset_manifest,
    )
    result.validate()
    return result


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            canonical_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _file_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_campaign_finalization(
    finalization: CampaignFinalization,
    output_directory: str | Path,
    *,
    verify_only: bool = False,
) -> Mapping[str, Any]:
    """Atomically publish artifacts and write the completion receipt last."""

    finalization.validate()
    root = Path(output_directory)
    payloads = {
        "accepted_labels.jsonl": _jsonl_bytes(row.to_dict() for row in finalization.rows),
        "failed_attempts.jsonl": _jsonl_bytes(finalization.failure_index),
        "target_transform.json": _json_bytes(finalization.target_transform.to_dict()),
        "audit_report.json": _json_bytes(finalization.audit_report),
        "dataset_manifest.json": _json_bytes(finalization.dataset_manifest),
    }
    receipt_payload = canonical_value(
        {
            "version": FINAL_RECEIPT_VERSION,
            "dataset_manifest_sha256": finalization.dataset_manifest["manifest_sha256"],
            "artifact_file_sha256s": {
                name: _file_digest(payload) for name, payload in sorted(payloads.items())
            },
        }
    )
    receipt = {
        **receipt_payload,
        "receipt_sha256": canonical_sha256(receipt_payload),
    }
    payloads["completion_receipt.json"] = _json_bytes(receipt)
    if verify_only:
        for name, expected in payloads.items():
            path = root / name
            try:
                actual = path.read_bytes()
            except OSError as error:
                raise FinalizationError(f"final artifact {name!r} is missing") from error
            if actual != expected:
                raise FinalizationError(f"final artifact {name!r} differs from recomputation")
        return receipt
    for name in sorted(name for name in payloads if name != "completion_receipt.json"):
        atomic_write_bytes(root / name, payloads[name])
    atomic_write_bytes(root / "completion_receipt.json", payloads["completion_receipt.json"])
    return receipt


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot load campaign artifact {path}") from error
    if not isinstance(value, Mapping):
        raise FinalizationError(f"campaign artifact {path} is not a JSON object")
    return value


def _load_records(workspace: Path) -> tuple[dict[str, LabelRunRecord], tuple[LabelRunRecord, ...]]:
    accepted: dict[str, LabelRunRecord] = {}
    accepted_run_ids: set[str] = set()
    for path in sorted((workspace / "attempts" / "accepted").glob("*.json")):
        record = label_run_record_from_dict(_load_json(path))
        if record.status != AttemptStatus.ACCEPTED or path.stem != record.configuration_id:
            raise FinalizationError("accepted record directory contains a mismatched record")
        if record.configuration_id in accepted or record.run_id in accepted_run_ids:
            raise FinalizationError("accepted record configuration/run ID is duplicated")
        accepted[record.configuration_id] = record
        accepted_run_ids.add(record.run_id)
    failures: list[LabelRunRecord] = []
    failure_run_ids: set[str] = set()
    for path in sorted((workspace / "attempts" / "failed").glob("*.json")):
        record = label_run_record_from_dict(_load_json(path))
        if record.status == AttemptStatus.ACCEPTED or path.stem != record.run_id:
            raise FinalizationError("failure record directory contains a mismatched record")
        if record.run_id in failure_run_ids:
            raise FinalizationError("failed record run ID is duplicated")
        failures.append(record)
        failure_run_ids.add(record.run_id)
    return accepted, tuple(failures)


def _load_repair_artifacts(
    workspace: Path,
    roots: Mapping[str, TargetCandidate],
) -> tuple[
    Mapping[str, OomRepairAttempt],
    Mapping[str, QuarantineRecord],
    Mapping[str, QuotaSubstitution],
]:
    repair_root = workspace / "attempts" / "repairs"
    oom: dict[str, OomRepairAttempt] = {}
    for path in sorted((repair_root / "oom").glob("*.json")):
        row = oom_repair_attempt_from_dict(_load_json(path))
        if path.stem != row.attempt_id or row.attempt_id in oom:
            raise FinalizationError("OOM repair artifact identity is duplicated/mismatched")
        oom[row.attempt_id] = row
    quarantines: dict[str, QuarantineRecord] = {}
    for path in sorted((repair_root / "quarantine").glob("*.json")):
        row = quarantine_record_from_dict(_load_json(path))
        if path.stem != row.quarantine_id or row.quarantine_id in quarantines:
            raise FinalizationError("quarantine artifact identity is duplicated/mismatched")
        quarantines[row.quarantine_id] = row
    substitutions: dict[str, QuotaSubstitution] = {}
    for path in sorted((repair_root / "substitution").glob("*.json")):
        raw = _load_json(path)
        target = roots.get(str(raw.get("target_candidate_id")))
        if target is None:
            raise FinalizationError("substitution references an unknown frozen root")
        row = quota_substitution_from_dict(raw, target)
        if path.stem != row.substitution_id or row.substitution_id in substitutions:
            raise FinalizationError("substitution artifact identity is duplicated/mismatched")
        substitutions[row.substitution_id] = row
    return oom, quarantines, substitutions


def _record_sha(record: LabelRunRecord, candidate: TargetCandidate, task: TaskRegistryEntry) -> str:
    try:
        return record.bound_record_sha256(candidate, task)
    except ValueError as error:
        raise FinalizationError("failure record is not bound to its attempted candidate") from error


def _resolve_workspace_slots(
    workspace: Path,
    manifest: TargetManifest,
    resolutions: Mapping[str, str],
    accepted: Mapping[str, LabelRunRecord],
    failures: Sequence[LabelRunRecord],
) -> tuple[ResolvedAcceptedSlot, ...]:
    from .workflow import _load_slot

    roots = {row.candidate_id: row for row in manifest.candidates}
    task_by_id = {row.task_id: row for row in load_task_registry().entries}
    failures_by_configuration = {row.configuration_id: row for row in failures}
    if len(failures_by_configuration) != len(failures):
        raise FinalizationError("workspace has multiple terminal failures for one configuration")
    oom, quarantines, substitutions = _load_repair_artifacts(workspace, roots)
    substitutions_by_root: dict[str, list[QuotaSubstitution]] = defaultdict(list)
    for row in substitutions.values():
        substitutions_by_root[row.target_candidate_id].append(row)
    oom_by_chain_root: dict[str, list[OomRepairAttempt]] = defaultdict(list)
    for row in oom.values():
        oom_by_chain_root[row.root_candidate_id].append(row)
    referenced_oom: set[str] = set()
    referenced_quarantine: set[str] = set()
    referenced_substitution: set[str] = set()
    referenced_failures: set[str] = set()
    resolved: list[ResolvedAcceptedSlot] = []
    for root in manifest.candidates:
        expected_final_id = resolutions.get(root.candidate_id)
        if expected_final_id is None:
            raise FinalizationError("workspace receipt omitted a frozen root")
        slot_path = workspace / "state" / "slots" / f"{root.candidate_id}.json"
        state = _load_slot(slot_path, root)
        substitutions_for_root = sorted(
            substitutions_by_root.get(root.candidate_id, ()),
            key=lambda row: row.replacement_index,
        )
        if tuple(row.replacement_index for row in substitutions_for_root) != tuple(
            range(len(substitutions_for_root))
        ):
            raise FinalizationError("slot substitution indexes are not contiguous")
        if state.replacement_index != len(substitutions_for_root):
            raise FinalizationError("slot replacement count differs from persisted substitutions")
        chain_roots = [root, *(row.candidate for row in substitutions_for_root)]
        attempted: list[TargetCandidate] = []
        slot_oom_ids: list[str] = []
        slot_quarantine_ids: list[str] = []
        slot_substitution_ids: list[str] = []
        for chain_index, chain_root in enumerate(chain_roots):
            attempts = sorted(
                oom_by_chain_root.get(chain_root.candidate_id, ()),
                key=lambda row: row.repair_index,
            )
            parent = chain_root
            attempted.append(parent)
            for repair_index, repair in enumerate(attempts):
                repair.validate()
                if repair.repair_index != repair_index or repair.parent_candidate != parent:
                    raise FinalizationError("workspace OOM repair chain is not contiguous")
                failure = failures_by_configuration.get(parent.candidate_id)
                task = task_by_id[parent.task_id]
                if (
                    failure is None
                    or repair.authorizing_run_id != failure.run_id
                    or repair.authorizing_failure_record_sha256
                    != _record_sha(failure, parent, task)
                ):
                    raise FinalizationError("OOM repair is not authorized by its failure record")
                referenced_failures.add(parent.candidate_id)
                referenced_oom.add(repair.attempt_id)
                slot_oom_ids.append(repair.attempt_id)
                parent = repair.candidate
                attempted.append(parent)
            if chain_index < len(substitutions_for_root):
                substitution = substitutions_for_root[chain_index]
                quarantine = quarantines.get(substitution.quarantine_id)
                if quarantine is None:
                    raise FinalizationError("substitution has no matching quarantine artifact")
                terminal_failure = failures_by_configuration.get(parent.candidate_id)
                task = task_by_id[parent.task_id]
                if (
                    quarantine.root_candidate_id != chain_root.candidate_id
                    or quarantine.terminal_candidate_id != parent.candidate_id
                    or quarantine.oom_repair_attempt_ids
                    != tuple(row.attempt_id for row in attempts)
                    or terminal_failure is None
                    or quarantine.terminal_failure_record_sha256
                    != _record_sha(terminal_failure, parent, task)
                ):
                    raise FinalizationError("quarantine does not close its exact failed chain")
                referenced_failures.add(parent.candidate_id)
                referenced_quarantine.add(quarantine.quarantine_id)
                referenced_substitution.add(substitution.substitution_id)
                slot_quarantine_ids.append(quarantine.quarantine_id)
                slot_substitution_ids.append(substitution.substitution_id)
        if (
            state.chain_root_candidate != chain_roots[-1]
            or state.current_candidate != attempted[-1]
            or tuple(row.attempt_id for row in state.oom_attempts)
            != tuple(row.attempt_id for row in sorted(
                oom_by_chain_root.get(chain_roots[-1].candidate_id, ()),
                key=lambda row: row.repair_index,
            ))
        ):
            raise FinalizationError("slot state differs from reconstructed repair lineage")
        if attempted[-1].candidate_id != expected_final_id:
            raise FinalizationError("task receipt final configuration differs from slot state")
        record = accepted.get(expected_final_id)
        if record is None:
            raise FinalizationError("resolved final configuration has no accepted record")
        lineage_payload = {
            "root_candidate_id": root.candidate_id,
            "attempted_candidate_ids": tuple(row.candidate_id for row in attempted),
            "oom_repair_attempt_ids": tuple(slot_oom_ids),
            "quarantine_ids": tuple(slot_quarantine_ids),
            "substitution_ids": tuple(slot_substitution_ids),
        }
        resolved.append(
            ResolvedAcceptedSlot(
                root,
                tuple(attempted),
                record,
                tuple(slot_oom_ids),
                tuple(slot_quarantine_ids),
                tuple(slot_substitution_ids),
                canonical_sha256(lineage_payload),
            )
        )
    if referenced_oom != set(oom):
        raise FinalizationError("workspace contains an unreferenced OOM repair artifact")
    if referenced_quarantine != set(quarantines):
        raise FinalizationError("workspace contains an unreferenced quarantine artifact")
    if referenced_substitution != set(substitutions):
        raise FinalizationError("workspace contains an unreferenced substitution artifact")
    if referenced_failures != set(failures_by_configuration):
        raise FinalizationError("workspace contains an unreferenced failure record")
    return tuple(resolved)


def _verify_provenance_sidecars(
    workspace: Path,
    records: Iterable[LabelRunRecord],
) -> Mapping[str, Any]:
    required = {
        "environment": {row.fingerprints.environment_sha256 for row in records},
        "hardware": {row.fingerprints.hardware_sha256 for row in records},
    }
    for category, digests in required.items():
        directory = workspace / "provenance" / category
        actual_paths = {path.stem: path for path in directory.glob("*.json")}
        if not digests <= set(actual_paths):
            raise FinalizationError(
                f"{category} provenance sidecars do not cover accepted hashes"
            )
        for digest in digests:
            path = actual_paths[digest]
            payload = _load_json(path)
            if canonical_sha256(payload) != digest:
                raise FinalizationError(f"{category} provenance sidecar hash differs")
    return {
        "payload_sidecars_verified": True,
        "environment_sidecar_identity": _identity_summary(required["environment"]),
        "hardware_sidecar_identity": _identity_summary(required["hardware"]),
    }


def finalize_workspace(
    workspace: str | Path,
    *,
    verify_only: bool = False,
) -> Mapping[str, Any]:
    """Revalidate the completed task loop and publish/verify the final pack."""

    root = Path(workspace).resolve()
    manifest, _ = freeze_initial_target_manifest(
        root,
        create_if_missing=not verify_only,
    )
    tasks = load_task_registry()
    order = tuple(row.task_id for row in tasks.entries)
    loop = load_task_loop_state(
        root / "state" / "task_loop.json",
        manifest=manifest,
        create_if_missing=not verify_only,
    )
    if loop.active_task_id is not None or loop.completed_task_ids != order:
        raise FinalizationError("all 22 task receipts must complete before finalization")
    resolutions: dict[str, str] = {}
    receipt_record_hashes: dict[str, str] = {}
    receipt_datasets: dict[str, str] = {}
    receipt_hashes: list[str] = []
    roots_by_id = {row.candidate_id: row for row in manifest.candidates}
    for task_id, expected_sha in zip(
        loop.completed_task_ids,
        loop.completed_receipt_sha256s,
        strict=True,
    ):
        receipt = TaskCompletionReceipt.from_dict(
            _load_json(root / "state" / "completed_materializations" / f"{task_id}.json")
        )
        if (
            receipt.task_id != task_id
            or receipt.receipt_sha256 != expected_sha
            or receipt.target_manifest_sha256 != manifest.sha256
        ):
            raise FinalizationError("task completion receipt differs from the frozen loop")
        for (root_id, accepted_id), record_sha256 in zip(
            receipt.resolutions,
            receipt.accepted_record_sha256s,
            strict=True,
        ):
            if root_id in resolutions:
                raise FinalizationError("one frozen root appears in multiple task receipts")
            root_candidate = roots_by_id.get(root_id)
            if root_candidate is None or root_candidate.task_id != task_id:
                raise FinalizationError("task receipt resolves a root from another task")
            resolutions[root_id] = accepted_id
            if accepted_id in receipt_record_hashes:
                raise FinalizationError("one accepted configuration appears in multiple receipts")
            receipt_record_hashes[accepted_id] = record_sha256
        receipt_datasets[task_id] = receipt.dataset_fingerprint
        receipt_hashes.append(receipt.receipt_sha256)
    if set(resolutions) != {row.candidate_id for row in manifest.candidates}:
        raise FinalizationError("task receipts do not resolve exactly 18,000 frozen roots")
    accepted, failures = _load_records(root)
    if set(accepted) != set(resolutions.values()):
        raise FinalizationError("accepted directory differs from the completed task receipts")
    for configuration_id, record in accepted.items():
        if (
            canonical_sha256(asdict(record)) != receipt_record_hashes.get(configuration_id)
            or record.fingerprints.dataset_sha256 != receipt_datasets.get(record.task_id)
        ):
            raise FinalizationError("accepted record hash/dataset differs from its task receipt")
    resolved = _resolve_workspace_slots(root, manifest, resolutions, accepted, failures)
    sidecar_provenance = _verify_provenance_sidecars(root, accepted.values())
    campaign_provenance = {
        **sidecar_provenance,
        "task_receipt_identity": _identity_summary(receipt_hashes),
        "task_receipt_count": len(receipt_hashes),
    }
    finalization = build_campaign_finalization(
        manifest,
        resolved,
        failures,
        campaign_provenance=campaign_provenance,
    )
    receipt = publish_campaign_finalization(
        finalization,
        root / "final",
        verify_only=verify_only,
    )
    return receipt


__all__ = [
    "CampaignFinalization",
    "FINALIZATION_VERSION",
    "FINAL_LABEL_ROW_VERSION",
    "FINAL_RECEIPT_VERSION",
    "FinalLabelRow",
    "FinalizationError",
    "ResolvedAcceptedSlot",
    "SPLIT_COUNTS",
    "TARGET_TRANSFORM_RECORD_VERSION",
    "TargetTransformRecord",
    "build_campaign_finalization",
    "finalize_workspace",
    "publish_campaign_finalization",
]
