"""Resumable one-GPU labeling workflow for one canonical rest modality."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .fingerprints import canonical_sha256, canonical_value
from .family_sharding import (
    FamilyShardError,
    freeze_family_contract,
    verify_family_workspace,
)
from .kaggle import KaggleCliClient
from .materialization import TaskMaterializer, freeze_initial_target_manifest
from .mlebench_bridge import PinnedMleBenchPreparer
from .modality_sharding import (
    ModalityContract,
    ModalityShardError,
    freeze_modality_contract,
    freeze_run_identity,
    verify_modality_workspace,
)
from .storage import atomic_write_json
from .supervisor import AttemptSupervisor, GpuProbe, discover_a10g_probes, lock_campaign_environment
from .task_registry import load_task_registry
from .workflow import (
    WorkflowError,
    _advance_failed_slot,
    _latest_failure,
    _load_record,
    _load_slot,
    _pilot_candidate_order,
    _record_indexes,
    _save_slot,
)


MODALITY_TASK_LOOP_VERSION = "perfseer_v3_nautilus_a10_modality_task_loop_v1"
MODALITY_TASK_RECEIPT_VERSION = "perfseer_v3_nautilus_a10_modality_task_receipt_v1"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot load modality state {path.name!r}") from error
    if not isinstance(value, Mapping):
        raise WorkflowError(f"modality state {path.name!r} is not an object")
    return value


def _new_loop(contract: ModalityContract) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "version": MODALITY_TASK_LOOP_VERSION,
        "contract_sha256": contract.contract_sha256,
        "completed_task_ids": [],
        "active_task_id": None,
    }
    value["state_sha256"] = canonical_sha256(value)
    return value


def _validate_loop(value: Mapping[str, Any], contract: ModalityContract) -> None:
    if set(value) != {
        "version",
        "contract_sha256",
        "completed_task_ids",
        "active_task_id",
        "state_sha256",
    }:
        raise WorkflowError("modality task-loop schema differs")
    unhashed = dict(value)
    declared_hash = unhashed.pop("state_sha256")
    completed = value["completed_task_ids"]
    if (
        value["version"] != MODALITY_TASK_LOOP_VERSION
        or value["contract_sha256"] != contract.contract_sha256
        or not isinstance(completed, list)
        or completed != list(contract.task_ids[: len(completed)])
        or declared_hash != canonical_sha256(unhashed)
    ):
        raise WorkflowError("modality task-loop identity, prefix, or hash differs")
    active = value["active_task_id"]
    if active is not None:
        if len(completed) >= len(contract.task_ids) or active != contract.task_ids[len(completed)]:
            raise WorkflowError("active modality task is not the next frozen task")


def _load_loop(path: Path, contract: ModalityContract) -> Mapping[str, Any]:
    if not path.exists():
        value = _new_loop(contract)
        atomic_write_json(path, value)
        return value
    value = _load_json(path)
    _validate_loop(value, contract)
    return value


def _save_loop(
    path: Path,
    contract: ModalityContract,
    *,
    completed: Sequence[str],
    active: str | None,
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "version": MODALITY_TASK_LOOP_VERSION,
        "contract_sha256": contract.contract_sha256,
        "completed_task_ids": list(completed),
        "active_task_id": active,
    }
    value["state_sha256"] = canonical_sha256(value)
    _validate_loop(value, contract)
    atomic_write_json(path, value)
    return value


def _build_task_receipt(
    *,
    workspace: Path,
    contract: ModalityContract,
    task_id: str,
    dataset_fingerprint: str,
    roots: Sequence[Any],
    slots: Mapping[str, Any],
) -> Mapping[str, Any]:
    accepted, _ = _record_indexes(workspace)
    resolutions: list[list[str]] = []
    record_hashes: list[str] = []
    for root in roots:
        candidate = slots[root.candidate_id].current_candidate
        if (
            candidate.task_id,
            candidate.family_id,
            candidate.regime,
            candidate.quota_modality,
            candidate.coverage_cell_ids,
        ) != (
            root.task_id,
            root.family_id,
            root.regime,
            root.quota_modality,
            root.coverage_cell_ids,
        ):
            raise WorkflowError("modality replacement left its frozen quota cell")
        record = accepted.get(candidate.candidate_id)
        if record is None:
            raise WorkflowError("modality task has an unresolved quota slot")
        payload = canonical_value(asdict(record))
        persisted = _load_json(
            workspace / "attempts" / "accepted" / f"{candidate.candidate_id}.json"
        )
        if persisted != payload:
            raise WorkflowError("durable modality record differs from validated memory")
        resolutions.append([root.candidate_id, candidate.candidate_id])
        record_hashes.append(canonical_sha256(payload))
    receipt: dict[str, Any] = {
        "version": MODALITY_TASK_RECEIPT_VERSION,
        "modality": contract.modality,
        "contract_sha256": contract.contract_sha256,
        "task_id": task_id,
        "dataset_fingerprint": dataset_fingerprint,
        "resolutions": resolutions,
        "accepted_record_sha256s": record_hashes,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _verify_task_receipt(
    workspace: Path,
    contract: ModalityContract,
    task_id: str,
) -> Mapping[str, Any]:
    receipt = _load_json(
        workspace / "state" / "modality_task_receipts" / f"{task_id}.json"
    )
    unhashed = dict(receipt)
    declared = unhashed.pop("receipt_sha256", None)
    candidate_by_id = {row.candidate_id: row for row in build_target_manifest_cached()}
    expected_roots = tuple(
        candidate_id
        for candidate_id in contract.candidate_ids
        if candidate_by_id[candidate_id].task_id == task_id
    )
    resolutions = receipt.get("resolutions")
    if not isinstance(resolutions, list) or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(type(value) is not str for value in row)
        for row in resolutions
    ):
        raise WorkflowError("modality task receipt resolution schema differs")
    if (
        receipt.get("version") != MODALITY_TASK_RECEIPT_VERSION
        or receipt.get("modality") != contract.modality
        or receipt.get("contract_sha256") != contract.contract_sha256
        or receipt.get("task_id") != task_id
        or tuple(row[0] for row in resolutions) != expected_roots
        or len({row[1] for row in resolutions}) != len(resolutions)
        or declared != canonical_sha256(unhashed)
    ):
        raise WorkflowError("modality task receipt identity, roots, or hash differs")
    hashes = receipt.get("accepted_record_sha256s")
    if not isinstance(hashes, list) or len(hashes) != len(resolutions):
        raise WorkflowError("modality task receipt record hashes differ")
    task_entry = next(
        row for row in load_task_registry().entries if row.task_id == task_id
    )
    for row, expected_hash in zip(resolutions, hashes, strict=True):
        if not isinstance(row, list) or len(row) != 2:
            raise WorkflowError("modality task receipt resolution schema differs")
        root_candidate = candidate_by_id[row[0]]
        slot = _load_slot(
            workspace / "state" / "slots" / f"{root_candidate.candidate_id}.json",
            root_candidate,
        )
        if slot.current_candidate.candidate_id != row[1]:
            raise WorkflowError("modality task receipt differs from durable slot resolution")
        record = _load_record(
            workspace / "attempts" / "accepted" / f"{row[1]}.json"
        )
        record.validate_against_configuration(slot.current_candidate, task_entry)
        if (
            record.fingerprints.dataset_sha256 != receipt.get("dataset_fingerprint")
            or canonical_sha256(asdict(record)) != expected_hash
        ):
            raise WorkflowError("modality task record changed after receipt publication")
    return receipt


_MANIFEST_CANDIDATES: tuple[Any, ...] | None = None


def build_target_manifest_cached() -> tuple[Any, ...]:
    global _MANIFEST_CANDIDATES
    if _MANIFEST_CANDIDATES is None:
        from .sampler import build_target_manifest

        _MANIFEST_CANDIDATES = build_target_manifest().candidates
    return _MANIFEST_CANDIDATES


def _remove_verified_task_cache(materializer: TaskMaterializer) -> None:
    cache = materializer.task_cache
    if not cache.exists() and not cache.is_symlink():
        return
    if cache.is_symlink() or cache.resolve() in {
        materializer.workspace.resolve(),
        materializer.repository_root.resolve(),
    }:
        raise WorkflowError("refusing to remove an unsafe modality task cache")
    shutil.rmtree(cache)


def _run_contract_workflow(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    repository_revision: str,
    image_digest: str,
    contract: Any,
    completion_verifier: Any,
    selection_name: str,
    kaggle_executable: str = "kaggle",
    max_new_accepted: int | None = None,
    probes: Sequence[GpuProbe] | None = None,
) -> None:
    """Run one exact contract while preserving the established task workflow."""

    if max_new_accepted is not None and (
        type(max_new_accepted) is not int or max_new_accepted < 1
    ):
        raise WorkflowError("maximum new accepted count must be a positive integer")
    if os.environ.get("PERFSEER_ALLOW_A10_FAMILY") != "1":
        raise WorkflowError("Nautilus modality workflow requires explicit A10-family opt-in")
    root = Path(workspace).resolve()
    freeze_run_identity(
        root,
        repository_revision=repository_revision,
        image_digest=image_digest,
    )
    manifest, _ = freeze_initial_target_manifest(root)
    if manifest.sha256 != contract.target_manifest_sha256:
        raise WorkflowError("selection contract and frozen target manifest differ")
    tasks = load_task_registry()
    entries = {entry.task_id: entry for entry in tasks.entries}
    loop_path = root / "state" / "modality_task_loop.json"
    loop = _load_loop(loop_path, contract)
    materializer = TaskMaterializer(
        workspace=root,
        repository_root=Path(repository_root),
        kaggle=KaggleCliClient(executable=kaggle_executable),
        preparer=PinnedMleBenchPreparer(Path(mlebench_checkout)),
    )
    workers = tuple(probes or discover_a10g_probes())
    if len(workers) != 1:
        raise WorkflowError("selection workflow requires exactly one visible A10-family GPU")
    lock_campaign_environment(root)
    accepted_at_start = len(_record_indexes(root)[0])
    while len(loop["completed_task_ids"]) < len(contract.task_ids):
        task_id = contract.task_ids[len(loop["completed_task_ids"])]
        receipt_path = root / "state" / "modality_task_receipts" / f"{task_id}.json"
        if receipt_path.is_file():
            _verify_task_receipt(root, contract, task_id)
            _remove_verified_task_cache(materializer)
            completed = [*loop["completed_task_ids"], task_id]
            loop = _save_loop(
                loop_path,
                contract,
                completed=completed,
                active=None,
            )
            continue
        if loop["active_task_id"] is None:
            loop = _save_loop(
                loop_path,
                contract,
                completed=loop["completed_task_ids"],
                active=task_id,
            )
        materialized = materializer.materialize(entries[task_id])
        roots = tuple(
            row
            for row in manifest.candidates
            if row.task_id == task_id and row.candidate_id in set(contract.candidate_ids)
        )
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
            processing_roots = _pilot_candidate_order(
                roots,
                max_new_accepted + 1,
            )
        supervisor = AttemptSupervisor(root)
        while True:
            accepted, failures = _record_indexes(root)
            unresolved: list[str] = []
            for root_candidate in processing_roots:
                slot = slots[root_candidate.candidate_id]
                if slot.current_candidate.candidate_id in accepted:
                    continue
                failure = _latest_failure(
                    failures.get(slot.current_candidate.candidate_id, ())
                )
                if failure is not None:
                    slot = _advance_failed_slot(root, slot, failure, root_candidate)
                    slot = _save_slot(slot_paths[root_candidate.candidate_id], slot)
                    slots[root_candidate.candidate_id] = slot
                unresolved.append(root_candidate.candidate_id)
            if not unresolved:
                break
            root_id = unresolved[0]
            slot = slots[root_id]
            supervisor.run(
                slot.current_candidate,
                entries[task_id],
                materialized.view_manifest,
                public_directory=materialized.public,
                prepared_directory=materialized.prepared_view,
                archive_sha256=materialized.inventory.archive_sha256,
                probe=workers[0],
                attempt_index=len(failures.get(slot.current_candidate.candidate_id, ())),
            )
            if max_new_accepted is not None:
                accepted_now = len(_record_indexes(root)[0])
                if accepted_now - accepted_at_start >= max_new_accepted:
                    return
        receipt = _build_task_receipt(
            workspace=root,
            contract=contract,
            task_id=task_id,
            dataset_fingerprint=materialized.view_manifest.dataset_fingerprint,
            roots=roots,
            slots=slots,
        )
        atomic_write_json(receipt_path, receipt)
        _verify_task_receipt(root, contract, task_id)
        _remove_verified_task_cache(materializer)
        completed = [*loop["completed_task_ids"], task_id]
        loop = _save_loop(loop_path, contract, completed=completed, active=None)
    try:
        completion_verifier(root)
    except (FamilyShardError, ModalityShardError) as error:
        raise WorkflowError(
            f"completed {selection_name} workspace failed final verification"
        ) from error


def run_modality_workflow(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    modality: str,
    repository_revision: str,
    image_digest: str,
    kaggle_executable: str = "kaggle",
    max_new_accepted: int | None = None,
    probes: Sequence[GpuProbe] | None = None,
) -> None:
    """Run or resume one modality; publish completion only at its exact quota."""

    root = Path(workspace).resolve()
    contract = freeze_modality_contract(root, modality)
    _run_contract_workflow(
        workspace=root,
        repository_root=repository_root,
        mlebench_checkout=mlebench_checkout,
        repository_revision=repository_revision,
        image_digest=image_digest,
        contract=contract,
        completion_verifier=lambda value: verify_modality_workspace(
            value,
            modality,
            repository_revision=repository_revision,
            image_digest=image_digest,
        ),
        selection_name="modality",
        kaggle_executable=kaggle_executable,
        max_new_accepted=max_new_accepted,
        probes=probes,
    )


def run_family_workflow(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    family_id: str,
    repository_revision: str,
    image_digest: str,
    kaggle_executable: str = "kaggle",
    max_new_accepted: int | None = None,
    probes: Sequence[GpuProbe] | None = None,
) -> None:
    """Run or resume one family; publish completion only at its exact quota."""

    root = Path(workspace).resolve()
    contract = freeze_family_contract(root, family_id)
    _run_contract_workflow(
        workspace=root,
        repository_root=repository_root,
        mlebench_checkout=mlebench_checkout,
        repository_revision=repository_revision,
        image_digest=image_digest,
        contract=contract,
        completion_verifier=lambda value: verify_family_workspace(
            value,
            family_id,
            repository_revision=repository_revision,
            image_digest=image_digest,
        ),
        selection_name="family",
        kaggle_executable=kaggle_executable,
        max_new_accepted=max_new_accepted,
        probes=probes,
    )


__all__ = [
    "MODALITY_TASK_LOOP_VERSION",
    "MODALITY_TASK_RECEIPT_VERSION",
    "run_family_workflow",
    "run_modality_workflow",
]
