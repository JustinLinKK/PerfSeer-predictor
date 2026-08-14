"""One resumable, task-by-task command for the exact V100 18K campaign."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import AttemptStatus, FailureStage, LabelRunRecord, label_run_record_from_dict
from .fingerprints import canonical_sha256, canonical_value
from .kaggle import KaggleCliClient
from .materialization import (
    TaskCompletionReceipt,
    TaskLoopState,
    TaskMaterializationError,
    TaskMaterializer,
    freeze_initial_target_manifest,
    load_task_loop_state,
    save_task_loop_state,
    verify_task_completion,
)
from .mlebench_bridge import PinnedMleBenchPreparer
from .repair import (
    OomRepairAttempt,
    make_quota_replacement,
    next_oom_repair,
    oom_repair_attempt_from_dict,
    quarantine_candidate,
)
from .sampler import TargetCandidate, target_candidate_from_dict
from .labeler_profile import PROFILE
from .storage import atomic_write_json
from .supervisor import (
    AttemptSupervisor,
    GpuProbe,
    discover_v100_probes,
    lock_campaign_environment,
)
from .task_registry import load_task_registry


WORKFLOW_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_11200_workflow_v1"
    if PROFILE.is_nonvision_4gpu
    else "perfseer_v3_v100_18k_workflow_v2"
)


class WorkflowError(RuntimeError):
    pass


class SlotExhaustedError(WorkflowError):
    """Raised after a quota slot consumes its bounded replacement budget."""


@dataclass(frozen=True)
class SlotState:
    version: str
    root_candidate_id: str
    chain_root_candidate: TargetCandidate
    current_candidate: TargetCandidate
    oom_attempts: tuple[OomRepairAttempt, ...]
    repair_index: int
    replacement_index: int
    state_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("state_sha256")
        return payload

    def validate(self) -> None:
        if self.version != WORKFLOW_VERSION or len(self.root_candidate_id) != 64:
            raise WorkflowError("slot state identity/version is invalid")
        self.chain_root_candidate.validate()
        self.current_candidate.validate()
        if min(self.repair_index, self.replacement_index) < 0:
            raise WorkflowError("slot repair/replacement indexes must be nonnegative")
        if self.repair_index != len(self.oom_attempts):
            raise WorkflowError("slot repair index differs from persisted OOM history")
        parent = self.chain_root_candidate
        for index, attempt in enumerate(self.oom_attempts):
            attempt.validate()
            if (
                attempt.root_candidate_id != self.chain_root_candidate.candidate_id
                or attempt.repair_index != index
                or attempt.parent_candidate.candidate_id != parent.candidate_id
            ):
                raise WorkflowError("slot OOM repair history is not contiguous")
            parent = attempt.candidate
        if self.current_candidate.candidate_id != parent.candidate_id:
            raise WorkflowError("slot current candidate differs from its OOM history")
        if self.state_sha256 != canonical_sha256(self.unhashed_payload()):
            raise WorkflowError("slot state hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


def _new_slot(root: TargetCandidate) -> SlotState:
    draft = SlotState(
        version=WORKFLOW_VERSION,
        root_candidate_id=root.candidate_id,
        chain_root_candidate=root,
        current_candidate=root,
        oom_attempts=(),
        repair_index=0,
        replacement_index=0,
        state_sha256="",
    )
    return replace(draft, state_sha256=canonical_sha256(draft.unhashed_payload()))


def _load_slot(path: Path, root: TargetCandidate) -> SlotState:
    if not path.exists():
        return _new_slot(root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError("slot resume state is unreadable") from error
    if not isinstance(raw, Mapping) or set(raw) != set(SlotState.__dataclass_fields__):
        raise WorkflowError("slot resume state schema differs")
    state = SlotState(
        version=raw["version"],
        root_candidate_id=raw["root_candidate_id"],
        chain_root_candidate=target_candidate_from_dict(raw["chain_root_candidate"]),
        current_candidate=target_candidate_from_dict(raw["current_candidate"]),
        oom_attempts=tuple(
            oom_repair_attempt_from_dict(value) for value in raw["oom_attempts"]
        ),
        repair_index=raw["repair_index"],
        replacement_index=raw["replacement_index"],
        state_sha256=raw["state_sha256"],
    )
    state.validate()
    if state.root_candidate_id != root.candidate_id:
        raise WorkflowError("slot resume state targets another frozen quota slot")
    preserved = ("family_id", "task_id", "regime", "coverage_cell_ids")
    if any(
        getattr(state.chain_root_candidate, name) != getattr(root, name)
        for name in preserved
    ):
        raise WorkflowError("slot chain root left its frozen quota cell")
    if state.replacement_index == 0 and state.chain_root_candidate.candidate_id != root.candidate_id:
        raise WorkflowError("unreplaced slot chain root differs from its frozen root")
    return state


def _save_slot(path: Path, state: SlotState) -> SlotState:
    draft = SlotState(
        state.version,
        state.root_candidate_id,
        state.chain_root_candidate,
        state.current_candidate,
        state.oom_attempts,
        state.repair_index,
        state.replacement_index,
        "",
    )
    final = replace(draft, state_sha256=canonical_sha256(draft.unhashed_payload()))
    atomic_write_json(path, final.to_dict())
    return final


def _load_record(path: Path) -> LabelRunRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot load persisted label record {path.name!r}") from error
    return label_run_record_from_dict(payload)


def _record_indexes(workspace: Path) -> tuple[dict[str, LabelRunRecord], dict[str, list[LabelRunRecord]]]:
    accepted: dict[str, LabelRunRecord] = {}
    for path in sorted((workspace / "attempts" / "accepted").glob("*.json")):
        record = _load_record(path)
        if record.status != AttemptStatus.ACCEPTED:
            raise WorkflowError("accepted output directory contains a rejected record")
        if path.stem != record.configuration_id:
            raise WorkflowError("accepted record filename differs from its configuration ID")
        if record.configuration_id in accepted:
            raise WorkflowError("accepted configuration ID is duplicated")
        accepted[record.configuration_id] = record
    failures: dict[str, list[LabelRunRecord]] = {}
    failure_run_ids: set[str] = set()
    for path in sorted((workspace / "attempts" / "failed").glob("*.json")):
        record = _load_record(path)
        if record.status == AttemptStatus.ACCEPTED:
            raise WorkflowError("failure output directory contains an accepted record")
        if path.stem != record.run_id or record.run_id in failure_run_ids:
            raise WorkflowError("failed run filename/identity is mismatched or duplicated")
        if record.configuration_id in accepted:
            raise WorkflowError("one configuration has both accepted and failed terminal records")
        failure_run_ids.add(record.run_id)
        failures.setdefault(record.configuration_id, []).append(record)
    terminal_configuration_ids = {*accepted, *failures}
    for configuration_id in terminal_configuration_ids:
        (workspace / "attempts" / "staging" / f"{configuration_id}.json").unlink(
            missing_ok=True
        )
        (workspace / "state" / "dispatch" / f"{configuration_id}.json").unlink(
            missing_ok=True
        )
    return accepted, failures


def _pilot_factor_tokens(candidate: TargetCandidate) -> tuple[tuple[str, str], ...]:
    return (
        ("family", candidate.family_id),
        ("mode", str(candidate.execution.get("mode", ""))),
        ("batch_tier", candidate.batch_plan.effective_tier),
        ("precision", str(candidate.precision_policy.get("policy_id", ""))),
        (
            "checkpoint",
            str(bool(candidate.activation_checkpointing.get("enabled", False))),
        ),
        ("optimizer", str(candidate.optimizer.get("name", ""))),
        ("scheduler", str(candidate.scheduler.get("name", ""))),
        ("regime", candidate.regime),
        ("training_step", candidate.training_step_id),
    )


def _pilot_candidate_order(
    candidates: Sequence[TargetCandidate],
    covering_prefix_size: int,
) -> tuple[TargetCandidate, ...]:
    """Put a deterministic factor-covering pilot prefix before canonical rows."""

    remaining = list(candidates)
    selected: list[TargetCandidate] = []
    covered_values: set[tuple[str, str]] = set()
    covered_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    while remaining and len(selected) < min(covering_prefix_size, len(candidates)):
        scored = []
        for candidate in remaining:
            values = _pilot_factor_tokens(candidate)
            pairs = set(combinations(values, 2))
            new_values = len(set(values) - covered_values)
            new_pairs = len(pairs - covered_pairs)
            scored.append((new_values * 1_000 + new_pairs, -candidate.ordinal, candidate))
        _, _, chosen = max(scored, key=lambda row: (row[0], row[1]))
        values = _pilot_factor_tokens(chosen)
        covered_values.update(values)
        covered_pairs.update(combinations(values, 2))
        selected.append(chosen)
        remaining.remove(chosen)
    selected_ids = {row.candidate_id for row in selected}
    return (*selected, *(row for row in candidates if row.candidate_id not in selected_ids))


def _latest_failure(rows: Sequence[LabelRunRecord]) -> LabelRunRecord | None:
    if len(rows) > 1:
        raise WorkflowError("unchanged configuration has multiple terminal failures")
    return rows[0] if rows else None


def _advance_failed_slot(
    workspace: Path,
    slot: SlotState,
    failure: LabelRunRecord,
    root: TargetCandidate,
) -> SlotState:
    if not failure.cleanup.passed:
        raise WorkflowError("GPU cleanup failed; this physical worker must not be reused")
    current = slot.current_candidate
    if slot.root_candidate_id != root.candidate_id:
        raise WorkflowError("slot failure advance targets another frozen quota slot")
    artifact_root = workspace / "attempts" / "repairs"
    if failure.status == AttemptStatus.OOM and current.microbatch_size > 1:
        repair = next_oom_repair(
            current,
            failure,
            repair_index=slot.repair_index,
            root_candidate_id=slot.chain_root_candidate.candidate_id,
        )
        atomic_write_json(artifact_root / "oom" / f"{repair.attempt_id}.json", asdict(repair))
        return SlotState(
            version=WORKFLOW_VERSION,
            root_candidate_id=slot.root_candidate_id,
            chain_root_candidate=slot.chain_root_candidate,
            current_candidate=repair.candidate,
            oom_attempts=(*slot.oom_attempts, repair),
            repair_index=slot.repair_index + 1,
            replacement_index=slot.replacement_index,
            state_sha256="",
        )
    quarantine = quarantine_candidate(
        slot.chain_root_candidate,
        failure,
        slot.oom_attempts,
        failure_stage=failure.failure_stage,
        reason_code=f"{failure.status.value}:{failure.failure_stage.value}",
    )
    maximum_replacements = 3 if PROFILE.is_nonvision_4gpu else 100
    if slot.replacement_index >= maximum_replacements:
        raise SlotExhaustedError(
            f"quota slot exceeded {maximum_replacements} deterministic replacements"
        )
    replacement = make_quota_replacement(
        root,
        quarantine,
        replacement_index=slot.replacement_index,
    )
    atomic_write_json(
        artifact_root / "quarantine" / f"{quarantine.quarantine_id}.json",
        asdict(quarantine),
    )
    atomic_write_json(
        artifact_root / "substitution" / f"{replacement.substitution_id}.json",
        asdict(replacement),
    )
    return SlotState(
        version=WORKFLOW_VERSION,
        root_candidate_id=slot.root_candidate_id,
        chain_root_candidate=replacement.candidate,
        current_candidate=replacement.candidate,
        oom_attempts=(),
        repair_index=0,
        replacement_index=slot.replacement_index + 1,
        state_sha256="",
    )


def _verify_recovered_receipt(
    workspace: Path,
    receipt: TaskCompletionReceipt,
    manifest: Any,
) -> None:
    receipt.validate()
    roots = tuple(
        row.candidate_id for row in manifest.candidates if row.task_id == receipt.task_id
    )
    if receipt.target_manifest_sha256 != manifest.sha256 or tuple(
        row[0] for row in receipt.resolutions
    ) != roots:
        raise WorkflowError("recovered completion receipt differs from frozen targets")
    for (_, candidate_id), expected_sha in zip(
        receipt.resolutions, receipt.accepted_record_sha256s, strict=True
    ):
        record = _load_record(workspace / "attempts" / "accepted" / f"{candidate_id}.json")
        if canonical_sha256(asdict(record)) != expected_sha:
            raise WorkflowError("recovered completion record hash differs")


def _verify_completed_prefix(
    workspace: Path,
    loop: TaskLoopState,
    manifest: Any,
) -> None:
    """Revalidate all previously completed tasks before trusting resume state."""

    for task_id, expected_receipt_sha256 in zip(
        loop.completed_task_ids,
        loop.completed_receipt_sha256s,
        strict=True,
    ):
        path = (
            workspace
            / "state"
            / "completed_materializations"
            / f"{task_id}.json"
        )
        try:
            receipt = TaskCompletionReceipt.from_dict(json.loads(path.read_text()))
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TaskMaterializationError,
        ) as error:
            raise WorkflowError(
                f"completed task receipt {task_id!r} is missing or invalid"
            ) from error
        if (
            receipt.task_id != task_id
            or receipt.receipt_sha256 != expected_receipt_sha256
        ):
            raise WorkflowError("completed task receipt differs from task-loop state")
        _verify_recovered_receipt(workspace, receipt, manifest)


def run_task_workflow(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    kaggle_executable: str = "kaggle",
    materialize_only: bool = False,
    probes: Sequence[GpuProbe] | None = None,
    task_group: str | None = None,
    max_new_accepted: int | None = None,
) -> None:
    """Resume the full campaign or one contracted whole-task shard."""

    if max_new_accepted is not None and (
        type(max_new_accepted) is not int or max_new_accepted < 1
    ):
        raise WorkflowError("maximum new accepted count must be a positive integer")
    if materialize_only and max_new_accepted is not None:
        raise WorkflowError("materialize-only mode cannot use an accepted-row limit")
    root = Path(workspace).resolve()
    manifest, _ = freeze_initial_target_manifest(root)
    tasks = load_task_registry()
    if task_group is None:
        if (
            (root / "state" / "shard_contract.json").exists()
            or (root / "state" / "shard_task_loop.json").exists()
        ):
            raise WorkflowError("unsharded workflow cannot resume a shard-bound workspace")
        order = tuple(row.task_id for row in tasks.entries)
        loop_path = root / "state" / "task_loop.json"
        loop = load_task_loop_state(loop_path, manifest=manifest)

        def persist_loop(value: Any) -> None:
            save_task_loop_state(loop_path, value, order)

        shard_contract = None
    else:
        from .sharding import (
            freeze_production_source_lock,
            freeze_shard_contract,
            load_shard_task_loop,
            save_shard_task_loop,
        )

        shard_contract = freeze_shard_contract(root, manifest, task_group)
        freeze_production_source_lock(root, repository_root)
        order = shard_contract.task_ids
        loop_path = root / "state" / "shard_task_loop.json"
        loop = load_shard_task_loop(loop_path, shard_contract)

        def persist_loop(value: Any) -> None:
            save_shard_task_loop(loop_path, value, shard_contract)
    materializer = TaskMaterializer(
        workspace=root,
        repository_root=Path(repository_root),
        kaggle=KaggleCliClient(executable=kaggle_executable),
        preparer=PinnedMleBenchPreparer(Path(mlebench_checkout)),
    )
    _verify_completed_prefix(root, loop, manifest)
    workers = tuple(probes or (() if materialize_only else discover_v100_probes()))
    if not materialize_only:
        lock_campaign_environment(root)
    accepted_at_start_index = (
        _record_indexes(root)[0] if max_new_accepted is not None else {}
    )
    accepted_at_start = len(accepted_at_start_index)
    while True:
        task_id = loop.select_next(order)
        if task_id is None:
            if not materialize_only:
                if shard_contract is None:
                    from .finalization import finalize_workspace

                    finalize_workspace(root)
                else:
                    from .sharding import finalize_shard_workspace

                    finalize_shard_workspace(root, repository_root)
            return
        receipt_path = root / "state" / "completed_materializations" / f"{task_id}.json"
        if loop.active_task_id == task_id and receipt_path.is_file():
            receipt = TaskCompletionReceipt.from_dict(json.loads(receipt_path.read_text()))
            _verify_recovered_receipt(root, receipt, manifest)
            materializer.cleanup_recovered_task(receipt)
            loop = loop.complete(task_id, receipt, order)
            persist_loop(loop)
            continue
        if loop.active_task_id is None:
            loop = loop.begin(task_id, order)
            persist_loop(loop)
        entry = next(row for row in tasks.entries if row.task_id == task_id)
        materialized = materializer.materialize(entry)
        if materialize_only:
            return
        roots = tuple(row for row in manifest.candidates if row.task_id == task_id)
        slot_paths = {
            row.candidate_id: root / "state" / "slots" / f"{row.candidate_id}.json"
            for row in roots
        }
        slots = {
            row.candidate_id: _load_slot(slot_paths[row.candidate_id], row)
            for row in roots
        }
        processing_roots = roots
        if max_new_accepted is not None:
            already_accepted_here = sum(
                slot.current_candidate.candidate_id in accepted_at_start_index
                for slot in slots.values()
            )
            processing_roots = _pilot_candidate_order(
                roots,
                already_accepted_here + max_new_accepted + len(workers),
            )
        supervisor = AttemptSupervisor(root)
        while True:
            accepted, failures = _record_indexes(root)
            unresolved = []
            for root_candidate in processing_roots:
                slot = slots[root_candidate.candidate_id]
                if slot.current_candidate.candidate_id in accepted:
                    continue
                failure = _latest_failure(failures.get(slot.current_candidate.candidate_id, ()))
                if failure is not None:
                    slot = _advance_failed_slot(root, slot, failure, root_candidate)
                    slot = _save_slot(slot_paths[root_candidate.candidate_id], slot)
                    slots[root_candidate.candidate_id] = slot
                unresolved.append(root_candidate.candidate_id)
            if not unresolved:
                break
            batch_roots = unresolved[: len(workers)]
            if not batch_roots:
                raise WorkflowError("label workflow has no qualified physical V100 workers")
            with ThreadPoolExecutor(max_workers=len(batch_roots)) as executor:
                futures = []
                for probe, root_id in zip(workers, batch_roots, strict=False):
                    slot = slots[root_id]
                    attempt_count = len(
                        failures.get(slot.current_candidate.candidate_id, ())
                    )
                    futures.append(
                        executor.submit(
                            supervisor.run,
                            slot.current_candidate,
                            entry,
                            materialized.view_manifest,
                            public_directory=materialized.public,
                            prepared_directory=materialized.prepared_view,
                            archive_sha256=materialized.inventory.archive_sha256,
                            probe=probe,
                            attempt_index=attempt_count,
                        )
                    )
                for future in futures:
                    future.result()
            if max_new_accepted is not None:
                accepted_now, _ = _record_indexes(root)
                if len(accepted_now) - accepted_at_start >= max_new_accepted:
                    return
        accepted, _ = _record_indexes(root)
        resolutions = {
            root_candidate.candidate_id: slots[root_candidate.candidate_id].current_candidate
            for root_candidate in roots
        }
        records = tuple(
            accepted[candidate.candidate_id] for candidate in resolutions.values()
        )
        receipt = verify_task_completion(
            materialized,
            manifest,
            resolutions,
            records,
            workspace=root,
        )
        materializer.cleanup_completed_task(materialized, completion=receipt)
        loop = loop.complete(task_id, receipt, order)
        persist_loop(loop)


__all__ = [
    "SlotState",
    "SlotExhaustedError",
    "WORKFLOW_VERSION",
    "WorkflowError",
    "run_task_workflow",
]
