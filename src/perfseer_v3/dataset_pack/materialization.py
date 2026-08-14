"""Resumable one-task-at-a-time materialization for the V100 label runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from .fingerprints import canonical_sha256, canonical_value
from .labeler_profile import PROFILE
from .contracts import LabelRunRecord
from .kaggle import (
    ArchiveInventory,
    KaggleCompetitionProbe,
    KaggleCredentialEvidence,
    extract_zip_archive,
    inspect_nested_zip_archives,
    inspect_zip_archive,
    validate_external_credentials,
)
from .mlebench_bridge import PinnedMleBenchPreparer
from .prepared_view import (
    PreparedViewManifest,
    build_shared_prepared_view,
    load_and_verify_prepared_view,
)
from .sampler import TargetCandidate, TargetManifest, build_target_manifest
from .storage import (
    FreeSpaceGuard,
    atomic_write_json,
    atomic_write_jsonl,
    non_task_workspace_bytes,
)
from .task_registry import TaskRegistryEntry, load_task_registry
from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract


_SPEECH_V2 = PROFILE.uses_speech_v2
MATERIALIZATION_STATE_VERSION = (
    "perfseer_v3_nrp_a10_speech_task_materialization_state_v2"
    if _SPEECH_V2
    else "perfseer_v3_v100_task_materialization_state_v1"
)
TASK_LOOP_STATE_VERSION = (
    "perfseer_v3_nrp_a10_speech_task_loop_state_v2"
    if _SPEECH_V2
    else "perfseer_v3_v100_task_loop_state_v2"
)
TASK_COMPLETION_VERSION = (
    "perfseer_v3_nrp_a10_speech_task_completion_v2"
    if _SPEECH_V2
    else "perfseer_v3_v100_task_completion_v1"
)
SPEECH_SOURCE_LOCK_VERSION = "perfseer_v3_nrp_a10_speech_source_lock_v2"
MATERIALIZATION_STAGES = (
    "selected",
    "downloaded",
    "inspected",
    "extracted",
    "preparing",
    "mlebench_prepared",
    "view_ready",
)


class TaskMaterializationError(RuntimeError):
    """Raised when resume state or a task materialization stage is invalid."""


@dataclass(frozen=True)
class SpeechSourceLock:
    version: str
    task_id: str
    kaggle_slug: str
    remote_inventory_sha256: str
    archive_sha256: str
    archive_inventory_sha256: str
    substitution_contract_sha256: str
    lock_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("lock_sha256")
        return payload

    def validate(self) -> None:
        if (
            self.version != SPEECH_SOURCE_LOCK_VERSION
            or self.task_id != SPEECH_TASK_ID
            or self.kaggle_slug != "tensorflow-speech-recognition-challenge"
        ):
            raise TaskMaterializationError("speech source-lock identity differs")
        for name in (
            "remote_inventory_sha256",
            "archive_sha256",
            "archive_inventory_sha256",
            "substitution_contract_sha256",
            "lock_sha256",
        ):
            _digest(getattr(self, name), context=f"speech source lock {name}")
        if self.substitution_contract_sha256 != load_speech_substitution_contract().sha256:
            raise TaskMaterializationError("speech source lock uses another substitution")
        if self.lock_sha256 != canonical_sha256(self.unhashed_payload()):
            raise TaskMaterializationError("speech source lock hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def build(
        cls,
        *,
        remote_inventory_sha256: str,
        archive_sha256: str,
        archive_inventory_sha256: str,
    ) -> "SpeechSourceLock":
        draft = cls(
            version=SPEECH_SOURCE_LOCK_VERSION,
            task_id=SPEECH_TASK_ID,
            kaggle_slug="tensorflow-speech-recognition-challenge",
            remote_inventory_sha256=remote_inventory_sha256,
            archive_sha256=archive_sha256,
            archive_inventory_sha256=archive_inventory_sha256,
            substitution_contract_sha256=load_speech_substitution_contract().sha256,
            lock_sha256="",
        )
        result = replace(draft, lock_sha256=canonical_sha256(draft.unhashed_payload()))
        result.validate()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpeechSourceLock":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise TaskMaterializationError("serialized speech source-lock schema differs")
        result = cls(**dict(value))
        result.validate()
        return result


class KaggleAcquirer(Protocol):
    def authenticate(self) -> None: ...

    def probe_competition(self, slug: str) -> KaggleCompetitionProbe: ...

    def download_competition(self, slug: str, destination_archive: str | Path) -> Path: ...


class TaskPreparer(Protocol):
    def prepare(
        self,
        entry: TaskRegistryEntry,
        raw: str | Path,
        public: str | Path,
        private: str | Path,
    ) -> None: ...


def _digest(value: str | None, *, context: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TaskMaterializationError(f"{context} must be a lowercase SHA-256 digest")


def _outside_repository(path: Path, repository: Path) -> bool:
    try:
        path.resolve().relative_to(repository.resolve())
    except ValueError:
        return True
    return False


@dataclass(frozen=True)
class TaskMaterializationState:
    version: str
    task_id: str
    kaggle_slug: str
    stage: str
    credential_source: str
    remote_inventory_sha256: str
    archive_sha256: str | None = None
    archive_inventory_sha256: str | None = None
    nested_inventory_sha256: str | None = None
    dataset_fingerprint: str | None = None

    def validate(self) -> None:
        if self.version != MATERIALIZATION_STATE_VERSION:
            raise TaskMaterializationError("materialization state version mismatch")
        if not self.task_id or not self.kaggle_slug or self.stage not in MATERIALIZATION_STAGES:
            raise TaskMaterializationError("materialization state identity/stage is invalid")
        if self.credential_source not in {
            "api_token_environment",
            "legacy_environment",
            "external_file",
        }:
            raise TaskMaterializationError("materialization credential source is invalid")
        _digest(self.remote_inventory_sha256, context="remote inventory")
        for name in (
            "archive_sha256",
            "archive_inventory_sha256",
            "nested_inventory_sha256",
            "dataset_fingerprint",
        ):
            _digest(getattr(self, name), context=name, optional=True)
        stage_index = MATERIALIZATION_STAGES.index(self.stage)
        required_after = {
            "archive_sha256": 1,
            "archive_inventory_sha256": 2,
            "nested_inventory_sha256": 3,
            "dataset_fingerprint": 6,
        }
        for name, minimum in required_after.items():
            if (getattr(self, name) is not None) != (stage_index >= minimum):
                raise TaskMaterializationError(f"materialization state {name} disagrees with stage")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = canonical_value(asdict(self))
        payload["state_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskMaterializationState":
        expected = {*cls.__dataclass_fields__, "state_sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TaskMaterializationError("serialized materialization state schema differs")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        if value["state_sha256"] != canonical_sha256(payload):
            raise TaskMaterializationError("materialization state hash mismatch")
        state = cls(**payload)
        state.validate()
        return state


@dataclass(frozen=True)
class MaterializedTask:
    entry: TaskRegistryEntry
    archive: Path
    extracted: Path
    public: Path
    private: Path
    prepared_view: Path
    inventory: ArchiveInventory
    view_manifest: PreparedViewManifest
    state: TaskMaterializationState
    source_lock: SpeechSourceLock | None = None


def seal_worker_inputs(materialized: MaterializedTask) -> None:
    """Remove write bits from the two trees exposed to label workers."""

    for root in (materialized.public, materialized.prepared_view):
        if not root.is_dir() or root.is_symlink():
            raise TaskMaterializationError("worker input root is not a regular directory")
        paths = (root, *tuple(sorted(root.rglob("*"))))
        for path in paths:
            if path.is_symlink():
                raise TaskMaterializationError("worker input tree contains a symlink")
            if path.is_dir():
                os.chmod(path, 0o555)
            elif path.is_file():
                os.chmod(path, 0o444)
            else:
                raise TaskMaterializationError("worker input tree contains a special file")


@dataclass(frozen=True)
class TaskCompletionReceipt:
    version: str
    task_id: str
    target_manifest_sha256: str
    dataset_fingerprint: str
    resolutions: tuple[tuple[str, str], ...]
    accepted_record_sha256s: tuple[str, ...]
    records_manifest_sha256: str
    receipt_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("receipt_sha256")
        return payload

    def validate(self) -> None:
        if self.version != TASK_COMPLETION_VERSION or not self.task_id:
            raise TaskMaterializationError("task completion identity/version is invalid")
        for name in (
            "target_manifest_sha256",
            "dataset_fingerprint",
            "records_manifest_sha256",
            "receipt_sha256",
        ):
            _digest(getattr(self, name), context=name)
        if not self.resolutions or len(self.resolutions) != len(self.accepted_record_sha256s):
            raise TaskMaterializationError("task completion resolution/record counts differ")
        roots = tuple(row[0] for row in self.resolutions)
        accepted = tuple(row[1] for row in self.resolutions)
        if len(set(roots)) != len(roots) or len(set(accepted)) != len(accepted):
            raise TaskMaterializationError("task completion IDs are duplicated")
        for value in (*roots, *accepted, *self.accepted_record_sha256s):
            _digest(value, context="task completion identity")
        expected_manifest = canonical_sha256(
            {
                "resolutions": self.resolutions,
                "accepted_record_sha256s": self.accepted_record_sha256s,
            }
        )
        if self.records_manifest_sha256 != expected_manifest:
            raise TaskMaterializationError("task completion records manifest hash differs")
        if self.receipt_sha256 != canonical_sha256(self.unhashed_payload()):
            raise TaskMaterializationError("task completion receipt hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskCompletionReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise TaskMaterializationError("serialized task completion schema differs")
        receipt = cls(
            **{
                **dict(value),
                "resolutions": tuple(tuple(row) for row in value["resolutions"]),
                "accepted_record_sha256s": tuple(value["accepted_record_sha256s"]),
            }
        )
        receipt.validate()
        return receipt


def verify_task_completion(
    materialized: MaterializedTask,
    target_manifest: TargetManifest,
    slot_resolutions: Mapping[str, TargetCandidate],
    accepted_records: Sequence[LabelRunRecord],
    *,
    workspace: str | Path,
) -> TaskCompletionReceipt:
    """Bind every task quota slot to one durable, validated accepted record."""

    target_manifest.validate()
    roots = tuple(
        row for row in target_manifest.candidates if row.task_id == materialized.entry.task_id
    )
    if set(slot_resolutions) != {row.candidate_id for row in roots}:
        raise TaskMaterializationError("task completion does not resolve every frozen quota slot")
    accepted_by_id = {row.configuration_id: row for row in accepted_records}
    if len(accepted_by_id) != len(accepted_records):
        raise TaskMaterializationError("task completion accepted records are duplicated")
    ordered_resolutions: list[tuple[str, str]] = []
    ordered_hashes: list[str] = []
    output_root = Path(workspace).resolve() / "attempts" / "accepted"
    for root in roots:
        candidate = slot_resolutions[root.candidate_id]
        candidate.validate()
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
            raise TaskMaterializationError("task completion replacement left its quota cell")
        record = accepted_by_id.get(candidate.candidate_id)
        if record is None:
            raise TaskMaterializationError("task completion candidate has no accepted record")
        record.validate_against_configuration(candidate, materialized.entry)
        expected_payload = canonical_value(asdict(record))
        record_path = output_root / f"{candidate.candidate_id}.json"
        if _load_json(record_path) != expected_payload:
            raise TaskMaterializationError("durable accepted record differs from validated memory")
        ordered_resolutions.append((root.candidate_id, candidate.candidate_id))
        ordered_hashes.append(canonical_sha256(expected_payload))
    if set(accepted_by_id) != {row[1] for row in ordered_resolutions}:
        raise TaskMaterializationError("task completion contains an extra accepted record")
    records_manifest = canonical_sha256(
        {
            "resolutions": ordered_resolutions,
            "accepted_record_sha256s": ordered_hashes,
        }
    )
    draft = TaskCompletionReceipt(
        version=TASK_COMPLETION_VERSION,
        task_id=materialized.entry.task_id,
        target_manifest_sha256=target_manifest.sha256,
        dataset_fingerprint=materialized.view_manifest.dataset_fingerprint,
        resolutions=tuple(ordered_resolutions),
        accepted_record_sha256s=tuple(ordered_hashes),
        records_manifest_sha256=records_manifest,
        receipt_sha256="",
    )
    receipt = replace(draft, receipt_sha256=canonical_sha256(draft.unhashed_payload()))
    receipt.validate()
    return receipt


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TaskMaterializationError(f"cannot load resume state {path.name!r}") from error
    if not isinstance(value, Mapping):
        raise TaskMaterializationError("resume state must be a JSON object")
    return value


@dataclass
class TaskMaterializer:
    workspace: Path
    repository_root: Path
    kaggle: KaggleAcquirer
    preparer: TaskPreparer
    disk_guard: FreeSpaceGuard = FreeSpaceGuard()

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.repository_root = Path(self.repository_root).resolve()
        if not _outside_repository(self.workspace, self.repository_root):
            raise TaskMaterializationError(
                "Kaggle task workspace must be outside the repository checkout"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def task_cache(self) -> Path:
        return self.workspace / "task_cache"

    @property
    def state_path(self) -> Path:
        return self.task_cache / "materialization_state.json"

    def _save_state(self, state: TaskMaterializationState) -> None:
        atomic_write_json(self.state_path, state.to_dict())

    def _resume_state(self, entry: TaskRegistryEntry) -> TaskMaterializationState | None:
        if not self.state_path.is_file():
            return None
        state = TaskMaterializationState.from_dict(_load_json(self.state_path))
        if state.task_id != entry.task_id or state.kaggle_slug != entry.kaggle_slug:
            raise TaskMaterializationError(
                "task cache belongs to another task; complete or clean it first"
            )
        return state

    def _disk_values(self) -> tuple[int, int]:
        retained = non_task_workspace_bytes(self.workspace, self.task_cache)
        free = shutil.disk_usage(self.workspace).free
        return retained, free

    def _freeze_speech_source_lock(
        self,
        entry: TaskRegistryEntry,
        state: TaskMaterializationState,
        inventory: ArchiveInventory,
    ) -> SpeechSourceLock | None:
        if not PROFILE.uses_speech_v2 or entry.task_id != SPEECH_TASK_ID:
            return None
        if state.archive_inventory_sha256 is None:
            raise TaskMaterializationError("speech source cannot lock before archive inspection")
        expected = SpeechSourceLock.build(
            remote_inventory_sha256=state.remote_inventory_sha256,
            archive_sha256=inventory.archive_sha256,
            archive_inventory_sha256=state.archive_inventory_sha256,
        )
        path = self.workspace / "state" / "source_locks" / f"{entry.task_id}.json"
        if path.is_file():
            actual = SpeechSourceLock.from_dict(_load_json(path))
            if actual != expected:
                raise TaskMaterializationError(
                    "speech source archive or remote inventory drifted from the PVC lock"
                )
            return actual
        if path.exists() or path.is_symlink():
            raise TaskMaterializationError("speech source lock path is not a regular file")
        atomic_write_json(path, expected.to_dict())
        return expected

    def materialize(self, entry: TaskRegistryEntry) -> MaterializedTask:
        entry.validate()
        credentials = validate_external_credentials(self.repository_root)
        self.kaggle.authenticate()
        probe = self.kaggle.probe_competition(entry.kaggle_slug)
        probe.validate()
        if PROFILE.uses_speech_v2 and entry.task_id == SPEECH_TASK_ID:
            expected_inventory = sorted(
                (
                    str(row["name"]),
                    int(row["size_bytes"]),
                )
                for row in load_speech_substitution_contract().payload["new_task"][
                    "source_inventory"
                ]
            )
            actual_inventory = sorted((row.name, row.size_bytes) for row in probe.files)
            if actual_inventory != expected_inventory:
                raise TaskMaterializationError(
                    "TensorFlow Speech Recognition remote inventory differs from V2"
                )
        self.task_cache.mkdir(parents=True, exist_ok=True)
        state = self._resume_state(entry)
        resumed_preparing = state is not None and state.stage == "preparing"
        if state is None:
            state = TaskMaterializationState(
                version=MATERIALIZATION_STATE_VERSION,
                task_id=entry.task_id,
                kaggle_slug=entry.kaggle_slug,
                stage="selected",
                credential_source=credentials.source,
                remote_inventory_sha256=probe.sha256,
            )
            self._save_state(state)
        elif state.remote_inventory_sha256 != probe.sha256:
            raise TaskMaterializationError("Kaggle remote inventory changed during resume")

        archive = self.task_cache / f"{entry.kaggle_slug}.zip"
        stage_index = MATERIALIZATION_STAGES.index(state.stage)
        if stage_index < MATERIALIZATION_STAGES.index("downloaded"):
            retained, free = self._disk_values()
            estimated_archive_bytes = max(
                entry.compressed_size_hint_bytes, probe.compressed_bytes
            )
            admission = self.disk_guard.assess(
                task_id=entry.task_id,
                stage="before_download",
                current_non_task_bytes=retained,
                task_archive_bytes=estimated_archive_bytes,
                extracted_bytes=entry.maximum_extracted_bytes,
                # The exact largest-member/5% value is unavailable until the ZIP is
                # present.  Reserve a non-zero transfer/extraction estimate here and
                # replace it with exact inventory values at the next two gates.
                extraction_temporary_bytes=max(1, estimated_archive_bytes // 8),
                observed_free_bytes=free,
            ).require()
            atomic_write_json(
                self.workspace / "state" / "admissions" / f"{entry.task_id}-download.json",
                admission.to_dict(),
            )
            # A file without a state-bound checksum is reconstructible and untrusted.
            # Removing only this exact task archive forces a clean redownload after an
            # interrupted or corrupt transfer.
            if archive.exists() or archive.is_symlink():
                archive.unlink()
            downloaded = self.kaggle.download_competition(entry.kaggle_slug, archive)
            if downloaded.resolve() != archive.resolve():
                raise TaskMaterializationError("Kaggle client returned an unexpected archive path")
            inventory = inspect_zip_archive(
                archive, maximum_uncompressed_bytes=entry.maximum_extracted_bytes
            )
            state = replace(state, stage="downloaded", archive_sha256=inventory.archive_sha256)
            self._save_state(state)
        if not archive.is_file() or state.archive_sha256 is None:
            raise TaskMaterializationError("resume archive is missing")
        inventory = inspect_zip_archive(
            archive, maximum_uncompressed_bytes=entry.maximum_extracted_bytes
        )
        if inventory.archive_sha256 != state.archive_sha256:
            raise TaskMaterializationError("resume archive checksum changed")
        stage_index = MATERIALIZATION_STAGES.index(state.stage)
        if stage_index < MATERIALIZATION_STAGES.index("inspected"):
            retained, free = self._disk_values()
            admission = self.disk_guard.assess(
                task_id=entry.task_id,
                stage="before_extraction",
                current_non_task_bytes=retained,
                task_archive_bytes=inventory.archive_bytes,
                extracted_bytes=inventory.uncompressed_bytes,
                extraction_temporary_bytes=inventory.extraction_temporary_bytes,
                observed_free_bytes=free,
                already_present_task_bytes=inventory.archive_bytes,
            ).require()
            atomic_write_json(
                self.workspace / "state" / "admissions" / f"{entry.task_id}-extraction.json",
                admission.to_dict(),
            )
            state = replace(
                state,
                stage="inspected",
                archive_inventory_sha256=inventory.inventory_sha256,
            )
            self._save_state(state)
        elif state.archive_inventory_sha256 != inventory.inventory_sha256:
            raise TaskMaterializationError("resume archive inventory changed")
        source_lock = self._freeze_speech_source_lock(entry, state, inventory)

        extracted = self.task_cache / "extracted"
        stage_index = MATERIALIZATION_STAGES.index(state.stage)
        if stage_index < MATERIALIZATION_STAGES.index("extracted"):
            if extracted.exists() or extracted.is_symlink():
                if extracted.is_dir() and not extracted.is_symlink():
                    shutil.rmtree(extracted)
                else:
                    extracted.unlink()
            extract_zip_archive(archive, extracted, inventory)
            nested = inspect_nested_zip_archives(
                extracted,
                maximum_uncompressed_bytes=entry.maximum_extracted_bytes,
            )
            nested_sha = canonical_sha256([row.to_dict() for row in nested])
            state = replace(
                state,
                stage="extracted",
                nested_inventory_sha256=nested_sha,
            )
            self._save_state(state)
        elif resumed_preparing:
            # Pinned preparers are allowed to extract/delete within raw.  A crash may
            # therefore leave it dirty, so a retry starts again from the immutable,
            # checksum-bound outer archive.
            if extracted.is_dir() and not extracted.is_symlink():
                shutil.rmtree(extracted)
            elif extracted.exists() or extracted.is_symlink():
                extracted.unlink()
            extract_zip_archive(archive, extracted, inventory)
        if not extracted.is_dir() or state.nested_inventory_sha256 is None:
            raise TaskMaterializationError("resume extracted task is missing")
        nested_inventories = inspect_nested_zip_archives(
            extracted,
            maximum_uncompressed_bytes=entry.maximum_extracted_bytes,
        )
        current_nested_sha = canonical_sha256([row.to_dict() for row in nested_inventories])
        if current_nested_sha != state.nested_inventory_sha256:
            raise TaskMaterializationError("nested archive inventory changed during resume")

        mlebench = self.task_cache / "mlebench"
        public = mlebench / "public"
        private = mlebench / "private"
        stage_index = MATERIALIZATION_STAGES.index(state.stage)
        if stage_index < MATERIALIZATION_STAGES.index("mlebench_prepared"):
            if state.stage == "extracted":
                # Account for raw outer files, concurrent nested expansion, and one
                # public/private prepared copy before invoking opaque pinned code.
                nested_expanded = sum(row.uncompressed_bytes for row in nested_inventories)
                prepared_copy_bound = max(inventory.uncompressed_bytes, nested_expanded)
                effective_extracted = (
                    inventory.uncompressed_bytes + nested_expanded + prepared_copy_bound
                )
                temporary = max(
                    [
                        inventory.extraction_temporary_bytes,
                        *(row.extraction_temporary_bytes for row in nested_inventories),
                    ]
                )
                retained, free = self._disk_values()
                admission = self.disk_guard.assess(
                    task_id=entry.task_id,
                    stage="before_preparation",
                    current_non_task_bytes=retained,
                    task_archive_bytes=inventory.archive_bytes,
                    extracted_bytes=effective_extracted,
                    extraction_temporary_bytes=temporary,
                    observed_free_bytes=free,
                    already_present_task_bytes=inventory.archive_bytes + inventory.uncompressed_bytes,
                ).require()
                atomic_write_json(
                    self.workspace / "state" / "admissions" / f"{entry.task_id}-preparation.json",
                    admission.to_dict(),
                )
                state = replace(state, stage="preparing")
                self._save_state(state)
            if mlebench.exists() or mlebench.is_symlink():
                if mlebench.is_dir() and not mlebench.is_symlink():
                    shutil.rmtree(mlebench)
                else:
                    mlebench.unlink()
            staging = Path(tempfile.mkdtemp(prefix=".mlebench.", dir=self.task_cache))
            try:
                self.preparer.prepare(entry, extracted, staging / "public", staging / "private")
                os.replace(staging, mlebench)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            state = replace(state, stage="mlebench_prepared")
            self._save_state(state)
        if not public.is_dir() or not private.is_dir():
            raise TaskMaterializationError("resume MLE-bench prepared directories are missing")

        prepared_view = self.task_cache / "prepared"
        stage_index = MATERIALIZATION_STAGES.index(state.stage)
        if stage_index < MATERIALIZATION_STAGES.index("view_ready"):
            if prepared_view.exists() or prepared_view.is_symlink():
                if prepared_view.is_dir() and not prepared_view.is_symlink():
                    shutil.rmtree(prepared_view)
                else:
                    prepared_view.unlink()
            view_manifest = build_shared_prepared_view(
                entry,
                public,
                prepared_view,
                archive_sha256=inventory.archive_sha256,
            )
            state = replace(
                state,
                stage="view_ready",
                dataset_fingerprint=view_manifest.dataset_fingerprint,
            )
            self._save_state(state)
        view_manifest = load_and_verify_prepared_view(
            entry,
            public,
            prepared_view,
            archive_sha256=inventory.archive_sha256,
        )
        if view_manifest.dataset_fingerprint != state.dataset_fingerprint:
            raise TaskMaterializationError("resume prepared-view fingerprint changed")
        return MaterializedTask(
            entry=entry,
            archive=archive,
            extracted=extracted,
            public=public,
            private=private,
            prepared_view=prepared_view,
            inventory=inventory,
            view_manifest=view_manifest,
            state=state,
            source_lock=source_lock,
        )

    def cleanup_completed_task(
        self,
        materialized: MaterializedTask,
        *,
        completion: TaskCompletionReceipt,
    ) -> Path:
        completion.validate()
        current = self._resume_state(materialized.entry)
        if current is None or current.stage != "view_ready":
            raise TaskMaterializationError("only a view-ready task can be cleaned")
        if current.dataset_fingerprint != materialized.view_manifest.dataset_fingerprint:
            raise TaskMaterializationError("cleanup task fingerprint mismatch")
        if (
            completion.task_id != materialized.entry.task_id
            or completion.dataset_fingerprint != materialized.view_manifest.dataset_fingerprint
        ):
            raise TaskMaterializationError("cleanup completion receipt targets another task/view")
        self._verify_cleanup_durability(
            completion,
            task_id=materialized.entry.task_id,
        )
        receipt_path = (
            self.workspace
            / "state"
            / "completed_materializations"
            / f"{materialized.entry.task_id}.json"
        )
        atomic_write_json(receipt_path, completion.to_dict())
        self._delete_task_cache()
        return receipt_path

    def _verify_cleanup_durability(
        self,
        completion: TaskCompletionReceipt,
        *,
        task_id: str,
    ) -> None:
        current_manifest = build_target_manifest()
        expected_roots = tuple(
            row.candidate_id
            for row in current_manifest.candidates
            if row.task_id == task_id
        )
        if (
            completion.task_id != task_id
            or completion.target_manifest_sha256 != current_manifest.sha256
            or tuple(row[0] for row in completion.resolutions) != expected_roots
        ):
            raise TaskMaterializationError("cleanup receipt differs from the frozen task slots")
        durable_root = self.workspace / "attempts" / "accepted"
        for (_, configuration_id), expected_sha in zip(
            completion.resolutions,
            completion.accepted_record_sha256s,
            strict=True,
        ):
            if canonical_sha256(_load_json(durable_root / f"{configuration_id}.json")) != expected_sha:
                raise TaskMaterializationError("durable task record changed after completion verification")

    def _delete_task_cache(self) -> None:
        if not self.task_cache.exists() and not self.task_cache.is_symlink():
            return
        if self.task_cache.is_symlink():
            raise TaskMaterializationError("refusing to delete a symlinked task cache")
        if self.task_cache.resolve() == self.workspace.resolve():
            raise TaskMaterializationError("refusing to delete the workspace root")
        shutil.rmtree(self.task_cache)

    def cleanup_recovered_task(self, completion: TaskCompletionReceipt) -> Path:
        """Finish a receipt-first cleanup interrupted before cache deletion."""

        completion.validate()
        receipt_path = (
            self.workspace
            / "state"
            / "completed_materializations"
            / f"{completion.task_id}.json"
        )
        if not receipt_path.is_file() or TaskCompletionReceipt.from_dict(
            _load_json(receipt_path)
        ) != completion:
            raise TaskMaterializationError("recovered cleanup receipt differs on disk")
        if self.task_cache.exists() or self.task_cache.is_symlink():
            entry = next(
                (
                    row
                    for row in load_task_registry().entries
                    if row.task_id == completion.task_id
                ),
                None,
            )
            if entry is None:
                raise TaskMaterializationError("recovered task is absent from the registry")
            current = self._resume_state(entry)
            if (
                current is None
                or current.stage != "view_ready"
                or current.dataset_fingerprint != completion.dataset_fingerprint
            ):
                raise TaskMaterializationError(
                    "recovered task cache is not the completed view"
                )
            self._verify_cleanup_durability(completion, task_id=completion.task_id)
            self._delete_task_cache()
        return receipt_path


@dataclass(frozen=True)
class TaskLoopState:
    version: str
    target_manifest_sha256: str
    task_registry_sha256: str
    completed_task_ids: tuple[str, ...]
    completed_receipt_sha256s: tuple[str, ...]
    active_task_id: str | None

    def validate(self, task_order: Sequence[str]) -> None:
        if self.version != TASK_LOOP_STATE_VERSION:
            raise TaskMaterializationError("task-loop state version mismatch")
        _digest(self.target_manifest_sha256, context="target manifest")
        _digest(self.task_registry_sha256, context="task registry")
        if len(set(self.completed_task_ids)) != len(self.completed_task_ids):
            raise TaskMaterializationError("completed task IDs are duplicated")
        expected_prefix = tuple(task_order[: len(self.completed_task_ids)])
        if self.completed_task_ids != expected_prefix:
            raise TaskMaterializationError("completed tasks are not a frozen-order prefix")
        if len(self.completed_receipt_sha256s) != len(self.completed_task_ids):
            raise TaskMaterializationError("completed tasks and receipts are not one-to-one")
        for digest in self.completed_receipt_sha256s:
            _digest(digest, context="completed task receipt")
        if self.active_task_id is not None:
            next_index = len(self.completed_task_ids)
            if next_index >= len(task_order) or self.active_task_id != task_order[next_index]:
                raise TaskMaterializationError("active task is not the next frozen task")

    def to_dict(self, task_order: Sequence[str]) -> dict[str, Any]:
        self.validate(task_order)
        payload = canonical_value(asdict(self))
        payload["state_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        task_order: Sequence[str],
    ) -> "TaskLoopState":
        expected = {*cls.__dataclass_fields__, "state_sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TaskMaterializationError("serialized task-loop state schema differs")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        if value["state_sha256"] != canonical_sha256(payload):
            raise TaskMaterializationError("task-loop state hash mismatch")
        state = cls(
            version=str(payload["version"]),
            target_manifest_sha256=str(payload["target_manifest_sha256"]),
            task_registry_sha256=str(payload["task_registry_sha256"]),
            completed_task_ids=tuple(payload["completed_task_ids"]),
            completed_receipt_sha256s=tuple(payload["completed_receipt_sha256s"]),
            active_task_id=payload["active_task_id"],
        )
        state.validate(task_order)
        return state

    def select_next(self, task_order: Sequence[str]) -> str | None:
        self.validate(task_order)
        if self.active_task_id is not None:
            return self.active_task_id
        if len(self.completed_task_ids) == len(task_order):
            return None
        return task_order[len(self.completed_task_ids)]

    def begin(self, task_id: str, task_order: Sequence[str]) -> "TaskLoopState":
        expected = self.select_next(task_order)
        if expected is None or task_id != expected:
            raise TaskMaterializationError("task-loop begin is not the next frozen task")
        if self.active_task_id is not None:
            return self
        state = replace(self, active_task_id=task_id)
        state.validate(task_order)
        return state

    def complete(
        self,
        task_id: str,
        receipt: TaskCompletionReceipt,
        task_order: Sequence[str],
    ) -> "TaskLoopState":
        self.validate(task_order)
        if self.active_task_id != task_id:
            raise TaskMaterializationError("only the active task can be completed")
        receipt.validate()
        if receipt.task_id != task_id:
            raise TaskMaterializationError("task loop completion receipt targets another task")
        state = replace(
            self,
            completed_task_ids=(*self.completed_task_ids, task_id),
            completed_receipt_sha256s=(
                *self.completed_receipt_sha256s,
                receipt.receipt_sha256,
            ),
            active_task_id=None,
        )
        state.validate(task_order)
        return state


def freeze_initial_target_manifest(
    workspace: str | Path,
    *,
    create_if_missing: bool = True,
) -> tuple[TargetManifest, Path]:
    """Create once or verify the exact source-generated 18K target manifest."""

    root = Path(workspace)
    manifest = build_target_manifest()
    path = root / "manifests" / "initial_targets.jsonl"
    metadata_path = root / "manifests" / "initial_targets.json"
    expected_metadata = manifest.to_summary()
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise TaskMaterializationError("frozen target manifest is partially present")
        actual_metadata = _load_json(metadata_path)
        if actual_metadata != expected_metadata:
            raise TaskMaterializationError("frozen target manifest differs from current source")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(manifest.candidates):
            raise TaskMaterializationError("frozen target manifest row count changed")
        try:
            actual_rows = tuple(json.loads(line) for line in lines)
        except json.JSONDecodeError as error:
            raise TaskMaterializationError("frozen target manifest JSONL is malformed") from error
        expected_rows = tuple(row.to_dict() for row in manifest.candidates)
        if actual_rows != expected_rows:
            raise TaskMaterializationError("frozen target manifest rows changed")
        return manifest, path
    if not create_if_missing:
        raise TaskMaterializationError("frozen target manifest is missing")
    atomic_write_jsonl(path, (row.to_dict() for row in manifest.candidates))
    atomic_write_json(metadata_path, expected_metadata)
    return manifest, path


def initial_task_loop_state(manifest: TargetManifest) -> TaskLoopState:
    tasks = load_task_registry()
    state = TaskLoopState(
        version=TASK_LOOP_STATE_VERSION,
        target_manifest_sha256=manifest.sha256,
        task_registry_sha256=tasks.sha256,
        completed_task_ids=(),
        completed_receipt_sha256s=(),
        active_task_id=None,
    )
    state.validate([entry.task_id for entry in tasks.entries])
    return state


def save_task_loop_state(
    path: str | Path,
    state: TaskLoopState,
    task_order: Sequence[str],
) -> None:
    atomic_write_json(path, state.to_dict(task_order))


def load_task_loop_state(
    path: str | Path,
    *,
    manifest: TargetManifest,
    create_if_missing: bool = True,
) -> TaskLoopState:
    tasks = load_task_registry()
    order = [entry.task_id for entry in tasks.entries]
    state_path = Path(path)
    if not state_path.exists():
        if not create_if_missing:
            raise TaskMaterializationError("task-loop state is missing")
        state = initial_task_loop_state(manifest)
        save_task_loop_state(state_path, state, order)
        return state
    state = TaskLoopState.from_dict(_load_json(state_path), order)
    if state.target_manifest_sha256 != manifest.sha256:
        raise TaskMaterializationError("task-loop state targets a different manifest")
    if state.task_registry_sha256 != tasks.sha256:
        raise TaskMaterializationError("task-loop state targets a different task registry")
    return state


__all__ = [
    "MATERIALIZATION_STATE_VERSION",
    "MaterializedTask",
    "TASK_LOOP_STATE_VERSION",
    "TASK_COMPLETION_VERSION",
    "TaskCompletionReceipt",
    "TaskLoopState",
    "TaskMaterializationError",
    "TaskMaterializationState",
    "TaskMaterializer",
    "freeze_initial_target_manifest",
    "initial_task_loop_state",
    "load_task_loop_state",
    "save_task_loop_state",
    "verify_task_completion",
    "seal_worker_inputs",
]
