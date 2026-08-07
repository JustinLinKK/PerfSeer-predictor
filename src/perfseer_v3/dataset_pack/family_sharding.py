"""Fail-closed single-family projections for Nautilus A10 labeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import canonical_sha256, canonical_value
from .modality_sharding import (
    A10_FAMILY_ID,
    LOGICAL_TARGET_HARDWARE_ID,
    MEASURED_EPOCHS_PER_LABEL,
    ModalityShardError,
    _digest,
    _frozen_registry_identities,
    _image_digest,
    _load_json,
    _revision,
    _verify_record,
    candidates_for_modality,
    freeze_run_identity,
)
from .quota import load_quota_plan
from .sharding import task_entries_for_shard
from .storage import atomic_write_json
from .task_registry import load_task_registry


SUPPORTED_FAMILIES = ("panns_cnn14",)
FAMILY_CONTRACT_VERSION = "perfseer_v3_nautilus_a10_family_contract_v1"
FAMILY_COMPLETION_VERSION = "perfseer_v3_nautilus_a10_family_completion_v1"


class FamilyShardError(ModalityShardError):
    """Raised when a family projection or completion is not exact."""


def _quota_cell(family_id: str) -> Any:
    if family_id not in SUPPORTED_FAMILIES:
        raise FamilyShardError(f"unsupported Nautilus family {family_id!r}")
    matches = tuple(
        cell for cell in load_quota_plan().cells if cell.family_id == family_id
    )
    if len(matches) != 1:
        raise FamilyShardError("family must resolve to exactly one frozen quota cell")
    return matches[0]


@lru_cache(maxsize=len(SUPPORTED_FAMILIES))
def candidates_for_family(family_id: str) -> tuple[Any, ...]:
    """Return one exact, registry-order family projection."""

    cell = _quota_cell(family_id)
    rows = tuple(
        row
        for row in candidates_for_modality(cell.modality)
        if row.family_id == family_id
    )
    if len(rows) != cell.accepted_configurations:
        raise FamilyShardError("family projection count differs from its frozen quota")
    if any(
        row.family_id != family_id or row.quota_modality != cell.modality
        for row in rows
    ):
        raise FamilyShardError("family projection escaped its frozen quota cell")
    return rows


@dataclass(frozen=True)
class FamilyContract:
    version: str
    family_id: str
    display_name: str
    modality: str
    accepted_configurations: int
    hardware_family_id: str
    logical_target_hardware_id: str
    target_manifest_sha256: str
    task_registry_sha256: str
    task_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_ids_sha256: str
    candidate_count: int
    measured_epoch_count: int
    compressed_download_estimate_bytes: int
    contract_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("contract_sha256")
        return payload

    def validate(self) -> None:
        cell = _quota_cell(self.family_id)
        rows = candidates_for_family(self.family_id)
        tasks = load_task_registry()
        manifest_sha256, task_registry_sha256 = _frozen_registry_identities()
        expected_task_ids = tuple(
            entry.task_id
            for entry in task_entries_for_shard("rest")
            if any(row.task_id == entry.task_id for row in rows)
        )
        task_by_id = {entry.task_id: entry for entry in tasks.entries}
        expected_ids = tuple(row.candidate_id for row in rows)
        if (
            self.version != FAMILY_CONTRACT_VERSION
            or self.display_name != cell.display_name
            or self.modality != cell.modality
            or self.accepted_configurations != cell.accepted_configurations
            or self.hardware_family_id != A10_FAMILY_ID
            or self.logical_target_hardware_id != LOGICAL_TARGET_HARDWARE_ID
            or self.target_manifest_sha256 != manifest_sha256
            or self.task_registry_sha256 != task_registry_sha256
        ):
            raise FamilyShardError("family contract identity or registry binding differs")
        if (
            self.task_ids != expected_task_ids
            or self.candidate_ids != expected_ids
            or self.candidate_ids_sha256 != canonical_sha256(expected_ids)
            or self.candidate_count != len(expected_ids)
            or self.candidate_count != cell.accepted_configurations
            or self.measured_epoch_count
            != self.candidate_count * MEASURED_EPOCHS_PER_LABEL
            or self.compressed_download_estimate_bytes
            != sum(task_by_id[task_id].compressed_size_hint_bytes for task_id in expected_task_ids)
        ):
            raise FamilyShardError("family contract projection or count differs")
        if self.contract_sha256 != canonical_sha256(self.unhashed_payload()):
            raise FamilyShardError("family contract hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FamilyContract":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise FamilyShardError("serialized family contract schema differs")
        contract = cls(
            **{
                **dict(value),
                "task_ids": tuple(value["task_ids"]),
                "candidate_ids": tuple(value["candidate_ids"]),
            }
        )
        contract.validate()
        return contract


@lru_cache(maxsize=len(SUPPORTED_FAMILIES))
def build_family_contract(family_id: str) -> FamilyContract:
    cell = _quota_cell(family_id)
    rows = candidates_for_family(family_id)
    tasks = load_task_registry()
    manifest_sha256, task_registry_sha256 = _frozen_registry_identities()
    task_ids = tuple(
        entry.task_id
        for entry in task_entries_for_shard("rest")
        if any(row.task_id == entry.task_id for row in rows)
    )
    task_by_id = {entry.task_id: entry for entry in tasks.entries}
    candidate_ids = tuple(row.candidate_id for row in rows)
    draft = FamilyContract(
        version=FAMILY_CONTRACT_VERSION,
        family_id=family_id,
        display_name=cell.display_name,
        modality=cell.modality,
        accepted_configurations=cell.accepted_configurations,
        hardware_family_id=A10_FAMILY_ID,
        logical_target_hardware_id=LOGICAL_TARGET_HARDWARE_ID,
        target_manifest_sha256=manifest_sha256,
        task_registry_sha256=task_registry_sha256,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        candidate_ids_sha256=canonical_sha256(candidate_ids),
        candidate_count=len(candidate_ids),
        measured_epoch_count=len(candidate_ids) * MEASURED_EPOCHS_PER_LABEL,
        compressed_download_estimate_bytes=sum(
            task_by_id[task_id].compressed_size_hint_bytes for task_id in task_ids
        ),
        contract_sha256="",
    )
    contract = replace(draft, contract_sha256=canonical_sha256(draft.unhashed_payload()))
    contract.validate()
    return contract


def family_analysis_summary(family_id: str) -> Mapping[str, Any]:
    contract = build_family_contract(family_id)
    return contract.to_dict()


def freeze_family_contract(workspace: str | Path, family_id: str) -> FamilyContract:
    root = Path(workspace).resolve()
    path = root / "state" / "family_contract.json"
    contract = build_family_contract(family_id)
    if path.is_file():
        stored = FamilyContract.from_dict(_load_json(path))
        if stored != contract:
            raise FamilyShardError("workspace is pinned to a different family contract")
    elif path.exists() or path.is_symlink():
        raise FamilyShardError("family contract path is not a regular file")
    else:
        atomic_write_json(path, contract.to_dict())
    return contract


@dataclass(frozen=True)
class FamilyCompletion:
    version: str
    family_id: str
    modality: str
    hardware_family_id: str
    logical_target_hardware_id: str
    contract_sha256: str
    target_manifest_sha256: str
    repository_revision: str
    image_digest: str
    candidate_ids: tuple[str, ...]
    resolved_candidate_ids: tuple[str, ...]
    accepted_record_sha256s: tuple[str, ...]
    hardware_provenance_sha256s: tuple[str, ...]
    candidate_count: int
    measured_epoch_count: int
    completion_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("completion_sha256")
        return payload

    def validate(self) -> None:
        contract = build_family_contract(self.family_id)
        _revision(self.repository_revision)
        _image_digest(self.image_digest)
        if (
            self.version != FAMILY_COMPLETION_VERSION
            or self.modality != contract.modality
            or self.hardware_family_id != A10_FAMILY_ID
            or self.logical_target_hardware_id != LOGICAL_TARGET_HARDWARE_ID
            or self.contract_sha256 != contract.contract_sha256
            or self.target_manifest_sha256 != contract.target_manifest_sha256
            or self.candidate_ids != contract.candidate_ids
            or self.candidate_count != contract.candidate_count
            or self.measured_epoch_count != contract.measured_epoch_count
            or len(self.resolved_candidate_ids) != self.candidate_count
            or len(set(self.resolved_candidate_ids)) != self.candidate_count
            or len(self.accepted_record_sha256s) != self.candidate_count
            or len(self.hardware_provenance_sha256s) != self.candidate_count
        ):
            raise FamilyShardError("family completion binding or count differs")
        for value in (
            *self.resolved_candidate_ids,
            *self.accepted_record_sha256s,
            *self.hardware_provenance_sha256s,
        ):
            _digest(value, context="family completion record/provenance")
        if self.completion_sha256 != canonical_sha256(self.unhashed_payload()):
            raise FamilyShardError("family completion hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


def verify_family_workspace(
    workspace: str | Path,
    family_id: str,
    *,
    repository_revision: str,
    image_digest: str,
    publish: bool = True,
) -> FamilyCompletion:
    root = Path(workspace).resolve()
    contract = freeze_family_contract(root, family_id)
    identity = freeze_run_identity(
        root,
        repository_revision=repository_revision,
        image_digest=image_digest,
    )
    resolutions: dict[str, str] = {}
    receipt_root = root / "state" / "modality_task_receipts"
    for receipt_path in sorted(receipt_root.glob("*.json")):
        receipt = _load_json(receipt_path)
        if receipt.get("contract_sha256") != contract.contract_sha256:
            raise FamilyShardError("task receipt is bound to another family contract")
        raw_resolutions = receipt.get("resolutions")
        if not isinstance(raw_resolutions, list):
            raise FamilyShardError("task receipt has no resolution list")
        for raw in raw_resolutions:
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or any(type(value) is not str for value in raw)
                or raw[0] in resolutions
            ):
                raise FamilyShardError("task receipt resolution schema or identity differs")
            resolutions[raw[0]] = raw[1]
    if resolutions:
        if set(resolutions) != set(contract.candidate_ids):
            raise FamilyShardError("task receipts do not resolve every family quota slot")
        resolved_ids = tuple(resolutions[candidate_id] for candidate_id in contract.candidate_ids)
    else:
        resolved_ids = contract.candidate_ids
    accepted_root = root / "attempts" / "accepted"
    if {path.stem for path in accepted_root.glob("*.json")} != set(resolved_ids):
        raise FamilyShardError("workspace accepted-record set differs from its family quota")
    record_hashes: list[str] = []
    provenance_hashes: list[str] = []
    for candidate_id in resolved_ids:
        record_sha, provenance_sha = _verify_record(
            accepted_root / f"{candidate_id}.json",
            candidate_id=candidate_id,
            workspace=root,
        )
        record_hashes.append(record_sha)
        provenance_hashes.append(provenance_sha)
    draft = FamilyCompletion(
        version=FAMILY_COMPLETION_VERSION,
        family_id=family_id,
        modality=contract.modality,
        hardware_family_id=A10_FAMILY_ID,
        logical_target_hardware_id=LOGICAL_TARGET_HARDWARE_ID,
        contract_sha256=contract.contract_sha256,
        target_manifest_sha256=contract.target_manifest_sha256,
        repository_revision=identity["repository_revision"],
        image_digest=identity["image_digest"],
        candidate_ids=contract.candidate_ids,
        resolved_candidate_ids=resolved_ids,
        accepted_record_sha256s=tuple(record_hashes),
        hardware_provenance_sha256s=tuple(provenance_hashes),
        candidate_count=contract.candidate_count,
        measured_epoch_count=contract.measured_epoch_count,
        completion_sha256="",
    )
    completion = replace(
        draft,
        completion_sha256=canonical_sha256(draft.unhashed_payload()),
    )
    completion.validate()
    if publish:
        atomic_write_json(root / "state" / "family_completion.json", completion.to_dict())
    return completion


__all__ = [
    "FamilyCompletion",
    "FamilyContract",
    "FamilyShardError",
    "SUPPORTED_FAMILIES",
    "build_family_contract",
    "candidates_for_family",
    "family_analysis_summary",
    "freeze_family_contract",
    "verify_family_workspace",
]
