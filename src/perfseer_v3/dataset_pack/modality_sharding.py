"""Fail-closed modality projections and merge receipts for Nautilus A10 work.

The original 18K target manifest remains immutable.  This module projects the
canonical 5,500-row ``rest`` task shard into four execution workspaces without
changing any candidate identity, then proves their exact union before a merged
index is published.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .fingerprints import canonical_sha256, canonical_value
from .sampler import TargetCandidate, build_target_manifest
from .sharding import task_entries_for_shard
from .storage import atomic_write_json
from .task_registry import load_task_registry


MODALITIES = ("audio", "tabular", "graph", "generated")
EXPECTED_MODALITY_COUNTS = {
    "audio": 1_300,
    "tabular": 950,
    "graph": 1_300,
    "generated": 1_950,
}
EXPECTED_REST_COUNT = 5_500
MEASURED_EPOCHS_PER_LABEL = 3
EXPECTED_MEASURED_EPOCHS = 16_500
A10_FAMILY_ID = "nvidia_a10_24gb_family_v1"
LOGICAL_TARGET_HARDWARE_ID = "nvidia_a10g_24gb_aws_g5"
MODALITY_CONTRACT_VERSION = "perfseer_v3_nautilus_a10_modality_contract_v1"
MODALITY_COMPLETION_VERSION = "perfseer_v3_nautilus_a10_modality_completion_v1"
MODALITY_MERGE_VERSION = "perfseer_v3_nautilus_a10_modality_merge_v1"


class ModalityShardError(RuntimeError):
    """Raised when a modality projection, completion, or merge is not exact."""


def _digest(value: str, *, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ModalityShardError(f"{context} must be a lowercase SHA-256 digest")


@lru_cache(maxsize=1)
def _frozen_registry_identities() -> tuple[str, str]:
    """Pay the full 18K validation cost once per verifier process."""

    return build_target_manifest().sha256, load_task_registry().sha256


def _revision(value: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ModalityShardError("repository revision must be a full lowercase Git commit")


def _image_digest(value: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ModalityShardError("container image must use an immutable sha256 digest")
    _digest(value.removeprefix("sha256:"), context="container image digest")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModalityShardError(f"cannot load JSON artifact {path.name!r}") from error
    if not isinstance(value, Mapping):
        raise ModalityShardError(f"JSON artifact {path.name!r} is not an object")
    return value


@lru_cache(maxsize=1)
def _rest_candidates() -> tuple[TargetCandidate, ...]:
    manifest = build_target_manifest()
    task_ids = {entry.task_id for entry in task_entries_for_shard("rest")}
    rows = tuple(row for row in manifest.candidates if row.task_id in task_ids)
    if len(rows) != EXPECTED_REST_COUNT:
        raise ModalityShardError("canonical rest shard no longer contains 5,500 rows")
    return rows


@lru_cache(maxsize=len(MODALITIES))
def candidates_for_modality(modality: str) -> tuple[TargetCandidate, ...]:
    """Return one canonical, registry-order modality projection."""

    if modality not in MODALITIES:
        raise ModalityShardError(f"unknown modality {modality!r}")
    rows = tuple(row for row in _rest_candidates() if row.quota_modality == modality)
    if len(rows) != EXPECTED_MODALITY_COUNTS[modality]:
        raise ModalityShardError(f"{modality} projection count differs from its frozen quota")
    if modality == "generated":
        if any(row.family_id != "independent_generated" for row in rows):
            raise ModalityShardError("generated projection contains a native family")
    elif any(row.family_id == "independent_generated" for row in rows):
        raise ModalityShardError(f"native {modality} projection contains generated work")
    return rows


@dataclass(frozen=True)
class ModalityContract:
    version: str
    modality: str
    hardware_family_id: str
    logical_target_hardware_id: str
    target_manifest_sha256: str
    task_registry_sha256: str
    task_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_ids_sha256: str
    candidate_count: int
    measured_epoch_count: int
    family_counts: Mapping[str, int]
    compressed_download_estimate_bytes: int
    contract_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("contract_sha256")
        return payload

    def validate(self) -> None:
        tasks = load_task_registry()
        manifest_sha256, task_registry_sha256 = _frozen_registry_identities()
        rows = candidates_for_modality(self.modality)
        expected_task_ids = tuple(
            entry.task_id
            for entry in task_entries_for_shard("rest")
            if any(row.task_id == entry.task_id for row in rows)
        )
        task_by_id = {entry.task_id: entry for entry in tasks.entries}
        expected_download = sum(
            task_by_id[task_id].compressed_size_hint_bytes
            for task_id in expected_task_ids
        )
        expected_families = dict(sorted(Counter(row.family_id for row in rows).items()))
        expected_ids = tuple(row.candidate_id for row in rows)
        if (
            self.version != MODALITY_CONTRACT_VERSION
            or self.modality not in MODALITIES
            or self.hardware_family_id != A10_FAMILY_ID
            or self.logical_target_hardware_id != LOGICAL_TARGET_HARDWARE_ID
            or self.target_manifest_sha256 != manifest_sha256
            or self.task_registry_sha256 != task_registry_sha256
        ):
            raise ModalityShardError("modality contract identity or registry binding differs")
        if (
            self.task_ids != expected_task_ids
            or self.candidate_ids != expected_ids
            or self.candidate_ids_sha256 != canonical_sha256(expected_ids)
            or self.candidate_count != len(expected_ids)
            or self.candidate_count != EXPECTED_MODALITY_COUNTS[self.modality]
            or self.measured_epoch_count
            != self.candidate_count * MEASURED_EPOCHS_PER_LABEL
            or dict(self.family_counts) != expected_families
            or self.compressed_download_estimate_bytes != expected_download
        ):
            raise ModalityShardError("modality contract projection or count differs")
        if self.contract_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ModalityShardError("modality contract hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModalityContract":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ModalityShardError("serialized modality contract schema differs")
        contract = cls(
            **{
                **dict(value),
                "task_ids": tuple(value["task_ids"]),
                "candidate_ids": tuple(value["candidate_ids"]),
            }
        )
        contract.validate()
        return contract


@lru_cache(maxsize=len(MODALITIES))
def build_modality_contract(modality: str) -> ModalityContract:
    rows = candidates_for_modality(modality)
    tasks = load_task_registry()
    manifest_sha256, task_registry_sha256 = _frozen_registry_identities()
    task_ids = tuple(
        entry.task_id
        for entry in task_entries_for_shard("rest")
        if any(row.task_id == entry.task_id for row in rows)
    )
    task_by_id = {entry.task_id: entry for entry in tasks.entries}
    candidate_ids = tuple(row.candidate_id for row in rows)
    draft = ModalityContract(
        version=MODALITY_CONTRACT_VERSION,
        modality=modality,
        hardware_family_id=A10_FAMILY_ID,
        logical_target_hardware_id=LOGICAL_TARGET_HARDWARE_ID,
        target_manifest_sha256=manifest_sha256,
        task_registry_sha256=task_registry_sha256,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        candidate_ids_sha256=canonical_sha256(candidate_ids),
        candidate_count=len(candidate_ids),
        measured_epoch_count=len(candidate_ids) * MEASURED_EPOCHS_PER_LABEL,
        family_counts=dict(sorted(Counter(row.family_id for row in rows).items())),
        compressed_download_estimate_bytes=sum(
            task_by_id[task_id].compressed_size_hint_bytes for task_id in task_ids
        ),
        contract_sha256="",
    )
    contract = replace(draft, contract_sha256=canonical_sha256(draft.unhashed_payload()))
    contract.validate()
    return contract


def analysis_summary() -> Mapping[str, Any]:
    contracts = tuple(build_modality_contract(modality) for modality in MODALITIES)
    ids = tuple(candidate for contract in contracts for candidate in contract.candidate_ids)
    if len(ids) != len(set(ids)) or set(ids) != {
        row.candidate_id for row in _rest_candidates()
    }:
        raise ModalityShardError("four modality projections are not an exact disjoint union")
    return canonical_value(
        {
            "hardware_family_id": A10_FAMILY_ID,
            "logical_target_hardware_id": LOGICAL_TARGET_HARDWARE_ID,
            "target_manifest_sha256": _frozen_registry_identities()[0],
            "rest_candidate_count": len(ids),
            "rest_measured_epoch_count": len(ids) * MEASURED_EPOCHS_PER_LABEL,
            "modalities": {
                contract.modality: {
                    "candidate_count": contract.candidate_count,
                    "measured_epoch_count": contract.measured_epoch_count,
                    "family_counts": contract.family_counts,
                    "task_ids": contract.task_ids,
                    "compressed_download_estimate_bytes": contract.compressed_download_estimate_bytes,
                    "contract_sha256": contract.contract_sha256,
                }
                for contract in contracts
            },
        }
    )


def freeze_modality_contract(workspace: str | Path, modality: str) -> ModalityContract:
    root = Path(workspace).resolve()
    path = root / "state" / "modality_contract.json"
    contract = build_modality_contract(modality)
    if path.is_file():
        stored = ModalityContract.from_dict(_load_json(path))
        if stored != contract:
            raise ModalityShardError("workspace is pinned to a different modality contract")
    elif path.exists() or path.is_symlink():
        raise ModalityShardError("modality contract path is not a regular file")
    else:
        atomic_write_json(path, contract.to_dict())
    return contract


def freeze_run_identity(
    workspace: str | Path,
    *,
    repository_revision: str,
    image_digest: str,
) -> Mapping[str, Any]:
    _revision(repository_revision)
    _image_digest(image_digest)
    root = Path(workspace).resolve()
    identity = {
        "version": "perfseer_v3_nautilus_a10_run_identity_v1",
        "repository_revision": repository_revision,
        "image_digest": image_digest,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    path = root / "state" / "run_identity.json"
    if path.is_file():
        if _load_json(path) != identity:
            raise ModalityShardError("workspace run identity changed")
    elif path.exists() or path.is_symlink():
        raise ModalityShardError("run identity path is not a regular file")
    else:
        atomic_write_json(path, identity)
    return identity


def _verify_hardware_provenance(value: Mapping[str, Any], expected_sha256: str) -> None:
    if canonical_sha256(value) != expected_sha256:
        raise ModalityShardError("hardware provenance hash differs from its label record")
    name = str(value.get("name", "")).upper().replace(" ", "")
    capability = value.get("compute_capability")
    memory = value.get("total_memory_bytes")
    if (
        "A10" not in name
        or capability not in ([8, 6], (8, 6))
        or type(memory) is not int
        or not 22 * 1024**3 <= memory <= 26 * 1024**3
        or not str(value.get("uuid", ""))
    ):
        raise ModalityShardError("label record is not backed by physical A10/A10G provenance")


def _verify_record(
    path: Path,
    *,
    candidate_id: str,
    workspace: Path,
) -> tuple[str, str]:
    value = _load_json(path)
    if (
        path.stem != candidate_id
        or value.get("configuration_id") != candidate_id
        or value.get("status") != "accepted"
        or value.get("target_hardware_id") != LOGICAL_TARGET_HARDWARE_ID
        or value.get("measured_epochs") != [3, 4, 5]
    ):
        raise ModalityShardError("accepted label identity, status, target, or epochs differ")
    fingerprints = value.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ModalityShardError("accepted label has no fingerprint bundle")
    hardware_sha256 = fingerprints.get("hardware_sha256")
    _digest(hardware_sha256, context="hardware provenance")
    provenance = _load_json(
        workspace / "provenance" / "hardware" / f"{hardware_sha256}.json"
    )
    _verify_hardware_provenance(provenance, hardware_sha256)
    return canonical_sha256(value), hardware_sha256


@dataclass(frozen=True)
class ModalityCompletion:
    version: str
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
        contract = build_modality_contract(self.modality)
        _revision(self.repository_revision)
        _image_digest(self.image_digest)
        if (
            self.version != MODALITY_COMPLETION_VERSION
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
            raise ModalityShardError("modality completion binding or count differs")
        for value in (
            *self.resolved_candidate_ids,
            *self.accepted_record_sha256s,
            *self.hardware_provenance_sha256s,
        ):
            _digest(value, context="completion record/provenance")
        if self.completion_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ModalityShardError("modality completion hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModalityCompletion":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ModalityShardError("serialized modality completion schema differs")
        completion = cls(
            **{
                **dict(value),
                "candidate_ids": tuple(value["candidate_ids"]),
                "resolved_candidate_ids": tuple(value["resolved_candidate_ids"]),
                "accepted_record_sha256s": tuple(value["accepted_record_sha256s"]),
                "hardware_provenance_sha256s": tuple(
                    value["hardware_provenance_sha256s"]
                ),
            }
        )
        completion.validate()
        return completion


def verify_modality_workspace(
    workspace: str | Path,
    modality: str,
    *,
    repository_revision: str,
    image_digest: str,
    publish: bool = True,
) -> ModalityCompletion:
    root = Path(workspace).resolve()
    contract = freeze_modality_contract(root, modality)
    identity = freeze_run_identity(
        root,
        repository_revision=repository_revision,
        image_digest=image_digest,
    )
    accepted_root = root / "attempts" / "accepted"
    resolutions: dict[str, str] = {}
    receipt_root = root / "state" / "modality_task_receipts"
    for receipt_path in sorted(receipt_root.glob("*.json")):
        receipt = _load_json(receipt_path)
        if receipt.get("contract_sha256") != contract.contract_sha256:
            raise ModalityShardError("task receipt is bound to another modality contract")
        raw_resolutions = receipt.get("resolutions")
        if not isinstance(raw_resolutions, list):
            raise ModalityShardError("task receipt has no resolution list")
        for raw in raw_resolutions:
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or any(type(value) is not str for value in raw)
                or raw[0] in resolutions
            ):
                raise ModalityShardError("task receipt resolution schema or identity differs")
            resolutions[raw[0]] = raw[1]
    if resolutions:
        if set(resolutions) != set(contract.candidate_ids):
            raise ModalityShardError("task receipts do not resolve every modality quota slot")
        resolved_ids = tuple(resolutions[candidate_id] for candidate_id in contract.candidate_ids)
    else:
        # Identity resolutions support legacy/imported workspaces that completed
        # without any OOM repair or quota substitution.
        resolved_ids = contract.candidate_ids
    actual_names = {path.stem for path in accepted_root.glob("*.json")}
    if actual_names != set(resolved_ids):
        raise ModalityShardError("workspace accepted-record set differs from its modality quota")
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
    draft = ModalityCompletion(
        version=MODALITY_COMPLETION_VERSION,
        modality=modality,
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
        atomic_write_json(root / "state" / "modality_completion.json", completion.to_dict())
    return completion


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str) -> None:
    value = _load_json(source)
    if canonical_sha256(value) != expected_sha256:
        raise ModalityShardError("source artifact changed after completion publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def merge_modality_workspaces(
    workspaces: Mapping[str, str | Path],
    output: str | Path,
) -> Mapping[str, Any]:
    if set(workspaces) != set(MODALITIES):
        raise ModalityShardError("merge requires exactly audio, tabular, graph, and generated")
    output_root = Path(output).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ModalityShardError("merged-rest output must not already exist")
    completions: dict[str, ModalityCompletion] = {}
    roots: dict[str, Path] = {}
    for modality in MODALITIES:
        root = Path(workspaces[modality]).resolve()
        roots[modality] = root
        completion = ModalityCompletion.from_dict(
            _load_json(root / "state" / "modality_completion.json")
        )
        if completion.modality != modality:
            raise ModalityShardError("completion was supplied under the wrong modality")
        current = verify_modality_workspace(
            root,
            modality,
            repository_revision=completion.repository_revision,
            image_digest=completion.image_digest,
            publish=False,
        )
        if current != completion:
            raise ModalityShardError("modality workspace changed after completion")
        completions[modality] = completion
    revisions = {row.repository_revision for row in completions.values()}
    images = {row.image_digest for row in completions.values()}
    manifests = {row.target_manifest_sha256 for row in completions.values()}
    if len(revisions) != 1 or len(images) != 1 or len(manifests) != 1:
        raise ModalityShardError("modality workspaces use different source, image, or manifest pins")
    all_ids = tuple(
        candidate_id
        for modality in MODALITIES
        for candidate_id in completions[modality].candidate_ids
    )
    expected_ids = tuple(row.candidate_id for row in _rest_candidates())
    if len(all_ids) != len(set(all_ids)):
        raise ModalityShardError("modality completion candidate sets overlap")
    if set(all_ids) != set(expected_ids) or len(all_ids) != EXPECTED_REST_COUNT:
        raise ModalityShardError("modality completion union differs from canonical rest")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        for modality in MODALITIES:
            completion = completions[modality]
            source_root = roots[modality]
            for candidate_id, record_sha in zip(
                completion.resolved_candidate_ids,
                completion.accepted_record_sha256s,
                strict=True,
            ):
                _copy_verified_file(
                    source_root / "attempts" / "accepted" / f"{candidate_id}.json",
                    staging / "attempts" / "accepted" / f"{candidate_id}.json",
                    record_sha,
                )
            for provenance_sha in set(completion.hardware_provenance_sha256s):
                destination = staging / "provenance" / "hardware" / f"{provenance_sha}.json"
                if not destination.exists():
                    _copy_verified_file(
                        source_root / "provenance" / "hardware" / f"{provenance_sha}.json",
                        destination,
                        provenance_sha,
                    )
            atomic_write_json(
                staging / "state" / "modality_completions" / f"{modality}.json",
                completion.to_dict(),
            )
        receipt = {
            "version": MODALITY_MERGE_VERSION,
            "hardware_family_id": A10_FAMILY_ID,
            "logical_target_hardware_id": LOGICAL_TARGET_HARDWARE_ID,
            "target_manifest_sha256": next(iter(manifests)),
            "repository_revision": next(iter(revisions)),
            "image_digest": next(iter(images)),
            "modality_completion_sha256s": {
                modality: completions[modality].completion_sha256
                for modality in MODALITIES
            },
            "candidate_ids": expected_ids,
            "candidate_ids_sha256": canonical_sha256(expected_ids),
            "candidate_count": len(expected_ids),
            "measured_epoch_count": len(expected_ids) * MEASURED_EPOCHS_PER_LABEL,
        }
        receipt["merge_sha256"] = canonical_sha256(receipt)
        atomic_write_json(staging / "state" / "modality_merge.json", receipt)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return canonical_value(receipt)


__all__ = [
    "A10_FAMILY_ID",
    "EXPECTED_MEASURED_EPOCHS",
    "EXPECTED_MODALITY_COUNTS",
    "EXPECTED_REST_COUNT",
    "LOGICAL_TARGET_HARDWARE_ID",
    "MODALITIES",
    "ModalityCompletion",
    "ModalityContract",
    "ModalityShardError",
    "analysis_summary",
    "build_modality_contract",
    "candidates_for_modality",
    "freeze_modality_contract",
    "freeze_run_identity",
    "merge_modality_workspaces",
    "verify_modality_workspace",
]
