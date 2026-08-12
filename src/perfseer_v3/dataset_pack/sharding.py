"""Deterministic three-workspace sharding and strict campaign merge.

The full 18K target manifest and per-task completion receipts remain the
canonical production identities.  A shard is only a task-order projection of
that manifest; final dataset construction still happens exactly once after a
strict union into a new canonical workspace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .contracts import LabelRunRecord
from .fingerprints import canonical_sha256, canonical_value, file_sha256
from .materialization import (
    TASK_LOOP_STATE_VERSION,
    TaskCompletionReceipt,
    freeze_initial_target_manifest,
)
from .sampler import TargetCandidate, TargetManifest
from .storage import atomic_write_bytes, atomic_write_json
from .task_registry import TaskRegistry, load_task_registry


SHARD_CONTRACT_VERSION = "perfseer_v3_v100_task_shard_contract_v1"
SHARD_GROUPING_RULE_VERSION = "perfseer_v3_source_task_modality_partition_v1"
SHARD_TASK_LOOP_VERSION = "perfseer_v3_v100_shard_task_loop_v1"
SHARD_COMPLETION_VERSION = "perfseer_v3_v100_shard_completion_v1"
PRODUCTION_SOURCE_LOCK_VERSION = "perfseer_v3_v100_production_source_lock_v1"
MERGE_JOURNAL_VERSION = "perfseer_v3_v100_shard_merge_journal_v1"
MERGE_RECEIPT_VERSION = "perfseer_v3_v100_shard_merge_receipt_v1"

SHARD_IDS = ("nlp", "vision", "rest")
EXPECTED_SHARD_TASK_COUNTS = {"nlp": 6, "vision": 10, "rest": 6}
EXPECTED_SHARD_ROOT_COUNTS = {"nlp": 5_700, "vision": 6_800, "rest": 5_500}
_SHARD_MODALITIES = {
    "nlp": frozenset({"nlp"}),
    "vision": frozenset({"vision"}),
    "rest": frozenset({"audio", "graph", "tabular"}),
}
_PRODUCTION_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "scripts/finalize_v100_18k_pack.py",
    "scripts/merge_v100_18k_shards.py",
    "scripts/run_v100_18k_pack.py",
    "src/perfseer_v3",
)


class ShardError(RuntimeError):
    """Raised when a shard or merged campaign is not an exact frozen union."""


def _digest(value: str, *, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ShardError(f"{context} must be a lowercase SHA-256 digest")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShardError(f"cannot load shard artifact {path}") from error
    if not isinstance(value, Mapping):
        raise ShardError(f"shard artifact {path} is not a JSON object")
    return value


def _task_ids_for_shard(tasks: TaskRegistry, shard_id: str) -> tuple[str, ...]:
    if shard_id not in SHARD_IDS:
        raise ShardError(f"unknown task shard {shard_id!r}")
    modalities = _SHARD_MODALITIES[shard_id]
    return tuple(row.task_id for row in tasks.entries if row.modality in modalities)


def task_entries_for_shard(shard_id: str) -> tuple[Any, ...]:
    """Return the frozen registry-order task entries for one public shard ID."""

    tasks = load_task_registry()
    ids = set(_task_ids_for_shard(tasks, shard_id))
    return tuple(row for row in tasks.entries if row.task_id in ids)


@dataclass(frozen=True)
class ShardContract:
    version: str
    grouping_rule_version: str
    shard_id: str
    target_manifest_sha256: str
    task_registry_sha256: str
    task_ids: tuple[str, ...]
    task_order_sha256: str
    task_count: int
    root_candidate_ids: tuple[str, ...]
    root_candidate_ids_sha256: str
    root_candidate_count: int
    contract_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("contract_sha256")
        return payload

    def validate(self, manifest: TargetManifest, tasks: TaskRegistry) -> None:
        manifest.validate()
        tasks.validate()
        if (
            self.version != SHARD_CONTRACT_VERSION
            or self.grouping_rule_version != SHARD_GROUPING_RULE_VERSION
            or self.shard_id not in SHARD_IDS
            or self.target_manifest_sha256 != manifest.sha256
            or self.task_registry_sha256 != tasks.sha256
        ):
            raise ShardError("shard contract identity/version differs")
        expected_tasks = _task_ids_for_shard(tasks, self.shard_id)
        expected_roots = tuple(
            row.candidate_id
            for row in manifest.candidates
            if row.task_id in set(expected_tasks)
        )
        if (
            self.task_ids != expected_tasks
            or self.task_count != len(expected_tasks)
            or self.task_count != EXPECTED_SHARD_TASK_COUNTS[self.shard_id]
            or self.task_order_sha256 != canonical_sha256(expected_tasks)
        ):
            raise ShardError("shard contract task projection differs")
        if (
            self.root_candidate_ids != expected_roots
            or self.root_candidate_count != len(expected_roots)
            or self.root_candidate_count != EXPECTED_SHARD_ROOT_COUNTS[self.shard_id]
            or self.root_candidate_ids_sha256 != canonical_sha256(expected_roots)
        ):
            raise ShardError("shard contract root projection differs")
        root_by_id = {row.candidate_id: row for row in manifest.candidates}
        allowed_modalities = _SHARD_MODALITIES[self.shard_id]
        if any(
            root_by_id[root_id].source_modality not in allowed_modalities
            for root_id in expected_roots
        ):
            raise ShardError("shard contract contains a cross-modality root")
        if self.contract_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ShardError("shard contract hash differs")

    def to_dict(self, manifest: TargetManifest, tasks: TaskRegistry) -> Mapping[str, Any]:
        self.validate(manifest, tasks)
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        manifest: TargetManifest,
        tasks: TaskRegistry,
    ) -> "ShardContract":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ShardError("serialized shard contract schema differs")
        contract = cls(
            **{
                **dict(value),
                "task_ids": tuple(value["task_ids"]),
                "root_candidate_ids": tuple(value["root_candidate_ids"]),
            }
        )
        contract.validate(manifest, tasks)
        return contract


def build_shard_contract(
    manifest: TargetManifest,
    shard_id: str,
    *,
    tasks: TaskRegistry | None = None,
) -> ShardContract:
    registry = tasks or load_task_registry()
    task_ids = _task_ids_for_shard(registry, shard_id)
    selected = set(task_ids)
    root_ids = tuple(
        row.candidate_id for row in manifest.candidates if row.task_id in selected
    )
    draft = ShardContract(
        version=SHARD_CONTRACT_VERSION,
        grouping_rule_version=SHARD_GROUPING_RULE_VERSION,
        shard_id=shard_id,
        target_manifest_sha256=manifest.sha256,
        task_registry_sha256=registry.sha256,
        task_ids=task_ids,
        task_order_sha256=canonical_sha256(task_ids),
        task_count=len(task_ids),
        root_candidate_ids=root_ids,
        root_candidate_ids_sha256=canonical_sha256(root_ids),
        root_candidate_count=len(root_ids),
        contract_sha256="",
    )
    contract = replace(
        draft, contract_sha256=canonical_sha256(draft.unhashed_payload())
    )
    contract.validate(manifest, registry)
    return contract


def validate_shard_partition(
    manifest: TargetManifest,
    contracts: Sequence[ShardContract],
    *,
    tasks: TaskRegistry | None = None,
) -> None:
    registry = tasks or load_task_registry()
    if {row.shard_id for row in contracts} != set(SHARD_IDS) or len(contracts) != 3:
        raise ShardError("completed shard set must contain nlp, vision, and rest exactly once")
    for contract in contracts:
        contract.validate(manifest, registry)
    all_tasks = [task for row in contracts for task in row.task_ids]
    all_roots = [root for row in contracts for root in row.root_candidate_ids]
    if (
        len(all_tasks) != len(set(all_tasks))
        or set(all_tasks) != {row.task_id for row in registry.entries}
        or len(all_roots) != len(set(all_roots))
        or tuple(sorted(all_roots))
        != tuple(sorted(row.candidate_id for row in manifest.candidates))
    ):
        raise ShardError("shard contracts do not partition the frozen campaign")


def freeze_shard_contract(
    workspace: str | Path,
    manifest: TargetManifest,
    shard_id: str,
    *,
    create_if_missing: bool = True,
) -> ShardContract:
    root = Path(workspace).resolve()
    path = root / "state" / "shard_contract.json"
    expected = build_shard_contract(manifest, shard_id)
    if path.is_file():
        actual = ShardContract.from_dict(
            _load_json(path), manifest, load_task_registry()
        )
        if actual != expected:
            raise ShardError("persisted shard contract differs from the requested shard")
        return actual
    if path.exists() or path.is_symlink():
        raise ShardError("shard contract path is not a regular file")
    if not create_if_missing:
        raise ShardError("shard contract is missing")
    if (root / "state" / "task_loop.json").exists():
        raise ShardError("cannot convert an unsharded workspace into a shard workspace")
    atomic_write_json(path, expected.to_dict(manifest, load_task_registry()))
    return expected


@dataclass(frozen=True)
class ShardTaskLoopState:
    version: str
    shard_contract_sha256: str
    task_order_sha256: str
    target_manifest_sha256: str
    task_registry_sha256: str
    completed_task_ids: tuple[str, ...]
    completed_receipt_sha256s: tuple[str, ...]
    active_task_id: str | None

    def validate(self, contract: ShardContract) -> None:
        if (
            self.version != SHARD_TASK_LOOP_VERSION
            or self.shard_contract_sha256 != contract.contract_sha256
            or self.task_order_sha256 != contract.task_order_sha256
            or self.target_manifest_sha256 != contract.target_manifest_sha256
            or self.task_registry_sha256 != contract.task_registry_sha256
        ):
            raise ShardError("shard task-loop identity differs from its contract")
        if self.completed_task_ids != contract.task_ids[: len(self.completed_task_ids)]:
            raise ShardError("completed shard tasks are not a frozen-order prefix")
        if len(self.completed_receipt_sha256s) != len(self.completed_task_ids):
            raise ShardError("completed shard tasks and receipts are not one-to-one")
        for value in self.completed_receipt_sha256s:
            _digest(value, context="shard task receipt")
        if self.active_task_id is not None:
            index = len(self.completed_task_ids)
            if index >= len(contract.task_ids) or self.active_task_id != contract.task_ids[index]:
                raise ShardError("active shard task is not the next contracted task")

    def to_dict(self, contract: ShardContract) -> Mapping[str, Any]:
        self.validate(contract)
        payload = canonical_value(asdict(self))
        payload["state_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], contract: ShardContract
    ) -> "ShardTaskLoopState":
        expected = {*cls.__dataclass_fields__, "state_sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ShardError("serialized shard task-loop schema differs")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        if value["state_sha256"] != canonical_sha256(payload):
            raise ShardError("shard task-loop hash differs")
        state = cls(
            **{
                **payload,
                "completed_task_ids": tuple(payload["completed_task_ids"]),
                "completed_receipt_sha256s": tuple(
                    payload["completed_receipt_sha256s"]
                ),
            }
        )
        state.validate(contract)
        return state

    def select_next(self, task_order: Sequence[str]) -> str | None:
        order = tuple(task_order)
        if (
            canonical_sha256(order) != self.task_order_sha256
            or self.completed_task_ids != order[: len(self.completed_task_ids)]
        ):
            raise ShardError("shard task-loop received another task order")
        if self.active_task_id is not None:
            return self.active_task_id
        if len(self.completed_task_ids) == len(order):
            return None
        return order[len(self.completed_task_ids)]

    def begin(self, task_id: str, task_order: Sequence[str]) -> "ShardTaskLoopState":
        order = tuple(task_order)
        if canonical_sha256(order) != self.task_order_sha256:
            raise ShardError("shard task-loop begin received another task order")
        index = len(self.completed_task_ids)
        expected = self.active_task_id or (order[index] if index < len(order) else None)
        if expected is None or task_id != expected:
            raise ShardError("shard task-loop begin is not the next contracted task")
        if self.active_task_id is not None:
            return self
        return replace(self, active_task_id=task_id)

    def complete(
        self,
        task_id: str,
        receipt: TaskCompletionReceipt,
        task_order: Sequence[str],
    ) -> "ShardTaskLoopState":
        order = tuple(task_order)
        index = len(self.completed_task_ids)
        if (
            canonical_sha256(order) != self.task_order_sha256
            or self.active_task_id != task_id
            or index >= len(order)
            or order[index] != task_id
        ):
            raise ShardError("only the active contracted shard task can complete")
        receipt.validate()
        if receipt.task_id != task_id:
            raise ShardError("shard loop receipt targets another task")
        return replace(
            self,
            completed_task_ids=(*self.completed_task_ids, task_id),
            completed_receipt_sha256s=(
                *self.completed_receipt_sha256s,
                receipt.receipt_sha256,
            ),
            active_task_id=None,
        )


def initial_shard_task_loop(contract: ShardContract) -> ShardTaskLoopState:
    state = ShardTaskLoopState(
        version=SHARD_TASK_LOOP_VERSION,
        shard_contract_sha256=contract.contract_sha256,
        task_order_sha256=contract.task_order_sha256,
        target_manifest_sha256=contract.target_manifest_sha256,
        task_registry_sha256=contract.task_registry_sha256,
        completed_task_ids=(),
        completed_receipt_sha256s=(),
        active_task_id=None,
    )
    state.validate(contract)
    return state


def save_shard_task_loop(
    path: str | Path, state: ShardTaskLoopState, contract: ShardContract
) -> None:
    atomic_write_json(path, state.to_dict(contract))


def load_shard_task_loop(
    path: str | Path,
    contract: ShardContract,
    *,
    create_if_missing: bool = True,
) -> ShardTaskLoopState:
    state_path = Path(path)
    if not state_path.exists():
        if not create_if_missing:
            raise ShardError("shard task-loop state is missing")
        state = initial_shard_task_loop(contract)
        save_shard_task_loop(state_path, state, contract)
        return state
    if not state_path.is_file():
        raise ShardError("shard task-loop path is not a regular file")
    return ShardTaskLoopState.from_dict(_load_json(state_path), contract)


@dataclass(frozen=True)
class ProductionSourceLock:
    version: str
    git_commit: str
    file_manifest: tuple[tuple[str, int, str], ...]
    file_manifest_sha256: str
    source_lock_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("source_lock_sha256")
        return payload

    def validate(self) -> None:
        if self.version != PRODUCTION_SOURCE_LOCK_VERSION or len(self.git_commit) != 40:
            raise ShardError("production source lock version/commit is invalid")
        if not self.file_manifest or tuple(sorted(self.file_manifest)) != self.file_manifest:
            raise ShardError("production source file manifest is empty or unordered")
        for path, length, digest in self.file_manifest:
            if not path or type(length) is not int or length < 0:
                raise ShardError("production source file identity is invalid")
            _digest(digest, context="production source file")
        if self.file_manifest_sha256 != canonical_sha256(self.file_manifest):
            raise ShardError("production source file-manifest hash differs")
        if self.source_lock_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ShardError("production source lock hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionSourceLock":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ShardError("serialized production source lock schema differs")
        lock = cls(
            **{
                **dict(value),
                "file_manifest": tuple(tuple(row) for row in value["file_manifest"]),
            }
        )
        lock.validate()
        return lock


def build_production_source_lock(repository_root: str | Path) -> ProductionSourceLock:
    root = Path(repository_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *_PRODUCTION_SOURCE_PATHS,
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", *_PRODUCTION_SOURCE_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ShardError("cannot resolve the reviewed production source identity") from error
    if status.strip():
        raise ShardError("sharded production labeling requires a clean reviewed source tree")
    relative_paths = tuple(
        sorted(row.decode("utf-8") for row in tracked if row)
    )
    if not relative_paths:
        raise ShardError("production source closure is empty")
    file_manifest = tuple(
        (path, (root / path).stat().st_size, file_sha256(root / path))
        for path in relative_paths
    )
    draft = ProductionSourceLock(
        version=PRODUCTION_SOURCE_LOCK_VERSION,
        git_commit=commit,
        file_manifest=file_manifest,
        file_manifest_sha256=canonical_sha256(file_manifest),
        source_lock_sha256="",
    )
    lock = replace(
        draft, source_lock_sha256=canonical_sha256(draft.unhashed_payload())
    )
    lock.validate()
    return lock


def freeze_production_source_lock(
    workspace: str | Path,
    repository_root: str | Path,
    *,
    create_if_missing: bool = True,
) -> ProductionSourceLock:
    expected = build_production_source_lock(repository_root)
    path = Path(workspace).resolve() / "state" / "production_source.json"
    if path.is_file():
        actual = ProductionSourceLock.from_dict(_load_json(path))
        if actual != expected:
            raise ShardError("production source changed since shard collection started")
        return actual
    if path.exists() or path.is_symlink():
        raise ShardError("production source lock path is not a regular file")
    if not create_if_missing:
        raise ShardError("production source lock is missing")
    atomic_write_json(path, expected.to_dict())
    return expected


@dataclass(frozen=True)
class DurableArtifact:
    relative_path: str
    byte_length: int
    sha256: str

    def validate(self) -> None:
        path = Path(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
            or type(self.byte_length) is not int
            or self.byte_length < 0
        ):
            raise ShardError("durable shard artifact path/length is invalid")
        _digest(self.sha256, context="durable shard artifact")


@dataclass(frozen=True)
class ShardCompletionReceipt:
    version: str
    shard_id: str
    shard_contract_sha256: str
    target_manifest_sha256: str
    task_registry_sha256: str
    production_source_lock_sha256: str
    campaign_environment_sha256: str
    task_ids: tuple[str, ...]
    task_receipt_sha256s: tuple[str, ...]
    root_candidate_count: int
    root_candidate_ids_sha256: str
    accepted_configuration_count: int
    accepted_configuration_ids_sha256: str
    accepted_records_manifest_sha256: str
    failed_attempt_count: int
    measured_epoch_record_count: int
    durable_artifacts: tuple[DurableArtifact, ...]
    durable_artifacts_sha256: str
    receipt_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("receipt_sha256")
        return payload

    def validate(self, contract: ShardContract) -> None:
        if (
            self.version != SHARD_COMPLETION_VERSION
            or self.shard_id != contract.shard_id
            or self.shard_contract_sha256 != contract.contract_sha256
            or self.target_manifest_sha256 != contract.target_manifest_sha256
            or self.task_registry_sha256 != contract.task_registry_sha256
            or self.task_ids != contract.task_ids
            or len(self.task_receipt_sha256s) != len(contract.task_ids)
            or self.root_candidate_count != contract.root_candidate_count
            or self.root_candidate_ids_sha256 != contract.root_candidate_ids_sha256
            or self.accepted_configuration_count != contract.root_candidate_count
            or self.measured_epoch_record_count != 3 * contract.root_candidate_count
            or self.failed_attempt_count < 0
        ):
            raise ShardError("shard completion counts/identity differ from its contract")
        for value in (
            self.production_source_lock_sha256,
            self.campaign_environment_sha256,
            self.accepted_configuration_ids_sha256,
            self.accepted_records_manifest_sha256,
            self.durable_artifacts_sha256,
            self.receipt_sha256,
            *self.task_receipt_sha256s,
        ):
            _digest(value, context="shard completion identity")
        if not self.durable_artifacts:
            raise ShardError("shard completion durable artifact manifest is empty")
        for row in self.durable_artifacts:
            row.validate()
        paths = tuple(row.relative_path for row in self.durable_artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ShardError("shard completion durable artifacts are unordered/duplicated")
        if self.durable_artifacts_sha256 != canonical_sha256(self.durable_artifacts):
            raise ShardError("shard completion durable artifact hash differs")
        if self.receipt_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ShardError("shard completion receipt hash differs")

    def to_dict(self, contract: ShardContract) -> Mapping[str, Any]:
        self.validate(contract)
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], contract: ShardContract
    ) -> "ShardCompletionReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ShardError("serialized shard completion schema differs")
        receipt = cls(
            **{
                **dict(value),
                "task_ids": tuple(value["task_ids"]),
                "task_receipt_sha256s": tuple(value["task_receipt_sha256s"]),
                "durable_artifacts": tuple(
                    DurableArtifact(**row) for row in value["durable_artifacts"]
                ),
            }
        )
        receipt.validate(contract)
        return receipt


_DURABLE_FIXED_PATHS = (
    "manifests/initial_targets.json",
    "manifests/initial_targets.jsonl",
    "state/campaign_environment.json",
    "state/production_source.json",
    "state/shard_contract.json",
    "state/shard_task_loop.json",
)
_DURABLE_DIRECTORY_PATHS = (
    "state/completed_materializations",
    "state/slots",
    "attempts/accepted",
    "attempts/failed",
    "attempts/repairs/oom",
    "attempts/repairs/quarantine",
    "attempts/repairs/substitution",
    "provenance/environment",
    "provenance/hardware",
)
_PROHIBITED_LIVE_PATHS = (
    "task_cache",
    "attempts/staging",
    "state/dispatch",
    "state/credential_sandboxes",
    "cache/compiler_attempts",
    "final",
)


def _files_below(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_dir():
        raise ShardError(f"shard artifact directory is unsafe: {path}")
    files = []
    for child in path.rglob("*"):
        if child.is_symlink() or (child.exists() and not child.is_file() and not child.is_dir()):
            raise ShardError(f"shard artifact tree contains an unsafe entry: {child}")
        if child.is_file():
            files.append(child)
    return tuple(sorted(files))


def _direct_json_files(workspace: Path, relative: str) -> tuple[Path, ...]:
    directory = workspace / relative
    files = _files_below(directory)
    if any(path.parent != directory or path.suffix != ".json" for path in files):
        raise ShardError(
            f"durable shard directory must contain only direct JSON files: {relative}"
        )
    return files


def _assert_quiescent_workspace(workspace: Path) -> None:
    for relative in _PROHIBITED_LIVE_PATHS:
        path = workspace / relative
        if path.is_symlink() or _files_below(path):
            raise ShardError(
                f"completed shard retains live or non-mergeable state under {relative!r}"
            )


def _durable_artifact_manifest(workspace: Path) -> tuple[DurableArtifact, ...]:
    _assert_quiescent_workspace(workspace)
    paths: list[Path] = []
    for relative in _DURABLE_FIXED_PATHS:
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise ShardError(f"required durable shard artifact is missing: {relative}")
        paths.append(path)
    for relative in _DURABLE_DIRECTORY_PATHS:
        paths.extend(_direct_json_files(workspace, relative))
    artifacts = tuple(
        DurableArtifact(
            relative_path=path.relative_to(workspace).as_posix(),
            byte_length=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in sorted(set(paths))
    )
    for row in artifacts:
        row.validate()
    return artifacts


def _verify_artifact_manifest(
    workspace: Path, artifacts: Sequence[DurableArtifact]
) -> None:
    actual = _durable_artifact_manifest(workspace)
    if tuple(artifacts) != actual:
        raise ShardError("completed shard durable artifact manifest changed")


def _load_campaign_environment(workspace: Path) -> tuple[Mapping[str, Any], str]:
    payload = _load_json(workspace / "state" / "campaign_environment.json")
    if set(payload) != {"version", "environment", "environment_sha256"}:
        raise ShardError("campaign environment lock schema differs")
    environment = payload["environment"]
    if not isinstance(environment, Mapping):
        raise ShardError("campaign environment payload is not an object")
    digest = str(payload["environment_sha256"])
    if (
        payload["version"] != "perfseer_v3_v100_campaign_environment_v1"
        or digest != canonical_sha256(environment)
    ):
        raise ShardError("campaign environment lock hash/version differs")
    return payload, digest


def _verify_exact_provenance(
    workspace: Path, records: Iterable[LabelRunRecord]
) -> None:
    rows = tuple(records)
    expected = {
        "environment": {row.fingerprints.environment_sha256 for row in rows},
        "hardware": {row.fingerprints.hardware_sha256 for row in rows},
    }
    for category, identities in expected.items():
        actual = {
            path.stem: path
            for path in _files_below(workspace / "provenance" / category)
            if path.suffix == ".json"
        }
        if set(actual) != identities:
            raise ShardError(f"{category} provenance is not an exact shard closure")
        for identity, path in actual.items():
            payload = _load_json(path)
            if canonical_sha256(payload) != identity:
                raise ShardError(f"{category} provenance sidecar hash differs")
            if category == "hardware" and (
                payload.get("target_hardware_id") != "nvidia_tesla_v100_sxm2_32gb_nrp"
                or "TESLAV100SXM2" not in re.sub(
                    r"[^A-Z0-9]", "", str(payload.get("name", "")).upper()
                )
                or payload.get("compute_capability") != [7, 0]
                or not 30 * 1024**3
                <= int(payload.get("total_memory_bytes", 0))
                <= 34 * 1024**3
                or not str(payload.get("uuid", ""))
            ):
                raise ShardError(
                    "hardware provenance is not a qualified Tesla V100 SXM2 32GB"
                )


def _assert_exact_semantic_paths(
    workspace: Path,
    contract: ShardContract,
    accepted: Mapping[str, LabelRunRecord],
    failures: Sequence[LabelRunRecord],
) -> None:
    expected_exact = {
        "state/completed_materializations": set(contract.task_ids),
        "attempts/accepted": set(accepted),
        "attempts/failed": {row.run_id for row in failures},
    }
    for relative, expected_stems in expected_exact.items():
        actual_stems = {path.stem for path in _direct_json_files(workspace, relative)}
        if actual_stems != expected_stems:
            raise ShardError(f"{relative} is not the exact contracted artifact closure")
    slot_stems = {
        path.stem for path in _direct_json_files(workspace, "state/slots")
    }
    if not slot_stems <= set(contract.root_candidate_ids):
        raise ShardError("state/slots contains a root outside the contracted shard")


def finalize_shard_workspace(
    workspace: str | Path,
    repository_root: str | Path,
    *,
    verify_only: bool = False,
) -> ShardCompletionReceipt:
    """Revalidate a completed shard and publish/verify its separate receipt."""

    from .finalization import (
        _load_records,
        _resolve_workspace_slots,
    )

    root = Path(workspace).resolve()
    manifest, _ = freeze_initial_target_manifest(root, create_if_missing=False)
    tasks = load_task_registry()
    contract_path = root / "state" / "shard_contract.json"
    if not contract_path.is_file():
        raise ShardError("shard contract is missing")
    contract = ShardContract.from_dict(_load_json(contract_path), manifest, tasks)
    source_lock = freeze_production_source_lock(
        root,
        repository_root,
        create_if_missing=False,
    )
    loop = load_shard_task_loop(
        root / "state" / "shard_task_loop.json",
        contract,
        create_if_missing=False,
    )
    if loop.active_task_id is not None or loop.completed_task_ids != contract.task_ids:
        raise ShardError("all contracted shard tasks must complete before shard finalization")
    roots_by_id = {row.candidate_id: row for row in manifest.candidates}
    selected_roots = tuple(roots_by_id[root_id] for root_id in contract.root_candidate_ids)
    contract_root_ids = set(contract.root_candidate_ids)
    resolutions: dict[str, str] = {}
    record_hashes: dict[str, str] = {}
    dataset_hashes: dict[str, str] = {}
    ordered_task_receipts: list[str] = []
    for task_id, expected_receipt_sha in zip(
        loop.completed_task_ids,
        loop.completed_receipt_sha256s,
        strict=True,
    ):
        receipt = TaskCompletionReceipt.from_dict(
            _load_json(
                root / "state" / "completed_materializations" / f"{task_id}.json"
            )
        )
        if (
            receipt.task_id != task_id
            or receipt.receipt_sha256 != expected_receipt_sha
            or receipt.target_manifest_sha256 != manifest.sha256
        ):
            raise ShardError("task receipt differs from the completed shard loop")
        for (root_id, accepted_id), record_sha in zip(
            receipt.resolutions,
            receipt.accepted_record_sha256s,
            strict=True,
        ):
            candidate = roots_by_id.get(root_id)
            if (
                candidate is None
                or candidate.task_id != task_id
                or root_id not in contract_root_ids
                or root_id in resolutions
                or accepted_id in record_hashes
            ):
                raise ShardError("task receipt crosses or duplicates a shard boundary")
            resolutions[root_id] = accepted_id
            record_hashes[accepted_id] = record_sha
        dataset_hashes[task_id] = receipt.dataset_fingerprint
        ordered_task_receipts.append(receipt.receipt_sha256)
    if set(resolutions) != set(contract.root_candidate_ids):
        raise ShardError("task receipts do not resolve the exact contracted roots")
    accepted, failures = _load_records(root)
    if set(accepted) != set(resolutions.values()):
        raise ShardError("accepted directory differs from shard task receipts")
    environment_lock, environment_sha = _load_campaign_environment(root)
    del environment_lock
    ordered_record_manifest = []
    for root_id in contract.root_candidate_ids:
        configuration_id = resolutions[root_id]
        record = accepted[configuration_id]
        if (
            canonical_sha256(asdict(record)) != record_hashes[configuration_id]
            or record.fingerprints.dataset_sha256 != dataset_hashes.get(record.task_id)
            or record.fingerprints.environment_sha256 != environment_sha
        ):
            raise ShardError("accepted shard record differs from its receipt/environment")
        ordered_record_manifest.append(
            (root_id, configuration_id, record_hashes[configuration_id])
        )
    _resolve_workspace_slots(
        root,
        manifest,
        resolutions,
        accepted,
        failures,
        root_candidates=selected_roots,
    )
    _verify_exact_provenance(root, accepted.values())
    _assert_exact_semantic_paths(root, contract, accepted, failures)
    durable = _durable_artifact_manifest(root)
    accepted_ids = tuple(resolutions[root_id] for root_id in contract.root_candidate_ids)
    draft = ShardCompletionReceipt(
        version=SHARD_COMPLETION_VERSION,
        shard_id=contract.shard_id,
        shard_contract_sha256=contract.contract_sha256,
        target_manifest_sha256=manifest.sha256,
        task_registry_sha256=tasks.sha256,
        production_source_lock_sha256=source_lock.source_lock_sha256,
        campaign_environment_sha256=environment_sha,
        task_ids=contract.task_ids,
        task_receipt_sha256s=tuple(ordered_task_receipts),
        root_candidate_count=contract.root_candidate_count,
        root_candidate_ids_sha256=contract.root_candidate_ids_sha256,
        accepted_configuration_count=len(accepted),
        accepted_configuration_ids_sha256=canonical_sha256(accepted_ids),
        accepted_records_manifest_sha256=canonical_sha256(
            tuple(ordered_record_manifest)
        ),
        failed_attempt_count=len(failures),
        measured_epoch_record_count=sum(
            len(row.epoch_measurements) for row in accepted.values()
        ),
        durable_artifacts=durable,
        durable_artifacts_sha256=canonical_sha256(durable),
        receipt_sha256="",
    )
    receipt = replace(
        draft, receipt_sha256=canonical_sha256(draft.unhashed_payload())
    )
    receipt.validate(contract)
    path = root / "state" / "shard_completion.json"
    if verify_only:
        actual = ShardCompletionReceipt.from_dict(_load_json(path), contract)
        if actual != receipt:
            raise ShardError("persisted shard completion differs from recomputation")
    else:
        atomic_write_json(path, receipt.to_dict(contract))
    return receipt


@dataclass(frozen=True)
class _VerifiedShard:
    workspace: Path
    manifest: TargetManifest
    contract: ShardContract
    completion: ShardCompletionReceipt
    source_lock: ProductionSourceLock
    environment_lock: Mapping[str, Any]
    completion_artifact: DurableArtifact


@dataclass(frozen=True)
class MergeJournal:
    version: str
    merge_identity_sha256: str
    target_manifest_sha256: str
    task_registry_sha256: str
    production_source_lock_sha256: str
    campaign_environment_sha256: str
    shard_receipt_sha256s: tuple[tuple[str, str], ...]
    journal_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("journal_sha256")
        return payload

    def validate(self) -> None:
        if self.version != MERGE_JOURNAL_VERSION:
            raise ShardError("merge journal version differs")
        if tuple(row[0] for row in self.shard_receipt_sha256s) != SHARD_IDS:
            raise ShardError("merge journal shard order differs")
        for value in (
            self.merge_identity_sha256,
            self.target_manifest_sha256,
            self.task_registry_sha256,
            self.production_source_lock_sha256,
            self.campaign_environment_sha256,
            self.journal_sha256,
            *(row[1] for row in self.shard_receipt_sha256s),
        ):
            _digest(value, context="merge journal identity")
        if self.journal_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ShardError("merge journal hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MergeJournal":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ShardError("serialized merge journal schema differs")
        journal = cls(
            **{
                **dict(value),
                "shard_receipt_sha256s": tuple(
                    tuple(row) for row in value["shard_receipt_sha256s"]
                ),
            }
        )
        journal.validate()
        return journal


@dataclass(frozen=True)
class MergeReceipt:
    version: str
    merge_identity_sha256: str
    target_manifest_sha256: str
    task_registry_sha256: str
    production_source_lock_sha256: str
    campaign_environment_sha256: str
    shard_receipt_sha256s: tuple[tuple[str, str], ...]
    task_count: int
    root_candidate_count: int
    accepted_configuration_count: int
    measured_epoch_record_count: int
    merged_artifact_count: int
    merged_artifacts_sha256: str
    receipt_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("receipt_sha256")
        return payload

    def validate(self) -> None:
        if (
            self.version != MERGE_RECEIPT_VERSION
            or tuple(row[0] for row in self.shard_receipt_sha256s) != SHARD_IDS
            or self.task_count != 22
            or self.root_candidate_count != 18_000
            or self.accepted_configuration_count != 18_000
            or self.measured_epoch_record_count != 54_000
            or self.merged_artifact_count < 1
        ):
            raise ShardError("merge receipt version/counts differ")
        for value in (
            self.merge_identity_sha256,
            self.target_manifest_sha256,
            self.task_registry_sha256,
            self.production_source_lock_sha256,
            self.campaign_environment_sha256,
            self.merged_artifacts_sha256,
            self.receipt_sha256,
            *(row[1] for row in self.shard_receipt_sha256s),
        ):
            _digest(value, context="merge receipt identity")
        if self.receipt_sha256 != canonical_sha256(self.unhashed_payload()):
            raise ShardError("merge receipt hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MergeReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ShardError("serialized merge receipt schema differs")
        receipt = cls(
            **{
                **dict(value),
                "shard_receipt_sha256s": tuple(
                    tuple(row) for row in value["shard_receipt_sha256s"]
                ),
            }
        )
        receipt.validate()
        return receipt


def _verified_shard(
    workspace: str | Path, repository_root: str | Path
) -> _VerifiedShard:
    root = Path(workspace).resolve()
    completion = finalize_shard_workspace(
        root,
        repository_root,
        verify_only=True,
    )
    manifest, _ = freeze_initial_target_manifest(root, create_if_missing=False)
    contract = ShardContract.from_dict(
        _load_json(root / "state" / "shard_contract.json"),
        manifest,
        load_task_registry(),
    )
    source_lock = ProductionSourceLock.from_dict(
        _load_json(root / "state" / "production_source.json")
    )
    completion_path = root / "state" / "shard_completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ShardError("shard completion receipt path is missing or unsafe")
    if ShardCompletionReceipt.from_dict(
        _load_json(completion_path), contract
    ) != completion:
        raise ShardError("shard completion receipt changed during verification")
    completion_artifact = DurableArtifact(
        "state/shard_completion.json",
        completion_path.stat().st_size,
        file_sha256(completion_path),
    )
    completion_artifact.validate()
    environment_lock, environment_sha = _load_campaign_environment(root)
    if (
        completion.production_source_lock_sha256 != source_lock.source_lock_sha256
        or completion.campaign_environment_sha256 != environment_sha
    ):
        raise ShardError("shard completion lock identities differ from durable locks")
    _verify_artifact_manifest(root, completion.durable_artifacts)
    return _VerifiedShard(
        workspace=root,
        manifest=manifest,
        contract=contract,
        completion=completion,
        source_lock=source_lock,
        environment_lock=environment_lock,
        completion_artifact=completion_artifact,
    )


def _build_merge_journal(shards: Sequence[_VerifiedShard]) -> MergeJournal:
    ordered = tuple(sorted(shards, key=lambda row: SHARD_IDS.index(row.contract.shard_id)))
    shard_receipts = tuple(
        (row.contract.shard_id, row.completion.receipt_sha256) for row in ordered
    )
    identity_payload = {
        "version": MERGE_JOURNAL_VERSION,
        "target_manifest_sha256": ordered[0].manifest.sha256,
        "task_registry_sha256": ordered[0].contract.task_registry_sha256,
        "production_source_lock_sha256": ordered[0].source_lock.source_lock_sha256,
        "campaign_environment_sha256": ordered[0].completion.campaign_environment_sha256,
        "shard_receipt_sha256s": shard_receipts,
    }
    draft = MergeJournal(
        version=MERGE_JOURNAL_VERSION,
        merge_identity_sha256=canonical_sha256(identity_payload),
        target_manifest_sha256=ordered[0].manifest.sha256,
        task_registry_sha256=ordered[0].contract.task_registry_sha256,
        production_source_lock_sha256=ordered[0].source_lock.source_lock_sha256,
        campaign_environment_sha256=ordered[0].completion.campaign_environment_sha256,
        shard_receipt_sha256s=shard_receipts,
        journal_sha256="",
    )
    journal = replace(
        draft, journal_sha256=canonical_sha256(draft.unhashed_payload())
    )
    journal.validate()
    return journal


def _validate_merge_sources(shards: Sequence[_VerifiedShard]) -> MergeJournal:
    if len(shards) != 3:
        raise ShardError("merge requires exactly three shard workspaces")
    manifest = shards[0].manifest
    contracts = tuple(row.contract for row in shards)
    validate_shard_partition(manifest, contracts)
    if any(row.manifest.sha256 != manifest.sha256 for row in shards):
        raise ShardError("shard target manifests differ")
    if len({row.contract.task_registry_sha256 for row in shards}) != 1:
        raise ShardError("shard task registries differ")
    if len({row.source_lock.source_lock_sha256 for row in shards}) != 1:
        raise ShardError("shard production source locks differ")
    if len({canonical_sha256(row.environment_lock) for row in shards}) != 1:
        raise ShardError("shard campaign environment locks differ")
    accepted_ids: set[str] = set()
    run_ids: set[str] = set()
    artifact_ids: dict[str, set[str]] = {
        "oom": set(),
        "quarantine": set(),
        "substitution": set(),
    }
    for shard in shards:
        for path in _files_below(shard.workspace / "attempts" / "accepted"):
            payload = _load_json(path)
            configuration_id = str(payload.get("configuration_id", ""))
            run_id = str(payload.get("run_id", ""))
            if configuration_id in accepted_ids or run_id in run_ids:
                raise ShardError("accepted configuration/run identity overlaps shards")
            accepted_ids.add(configuration_id)
            run_ids.add(run_id)
        for path in _files_below(shard.workspace / "attempts" / "failed"):
            run_id = str(_load_json(path).get("run_id", ""))
            if run_id in run_ids:
                raise ShardError("failed run identity overlaps another shard")
            run_ids.add(run_id)
        for category in artifact_ids:
            for path in _files_below(
                shard.workspace / "attempts" / "repairs" / category
            ):
                if path.stem in artifact_ids[category]:
                    raise ShardError(f"{category} repair identity overlaps shards")
                artifact_ids[category].add(path.stem)
    if len(accepted_ids) != 18_000:
        raise ShardError("shard accepted-record union is not exactly 18,000")
    return _build_merge_journal(shards)


_SHARD_ONLY_DURABLE_PATHS = {
    "state/shard_contract.json",
    "state/shard_task_loop.json",
}
_DEDUPLICATED_MERGE_PATHS = {
    "manifests/initial_targets.json",
    "manifests/initial_targets.jsonl",
    "state/campaign_environment.json",
    "state/production_source.json",
}


def _merge_copy_plan(
    shards: Sequence[_VerifiedShard],
) -> Mapping[str, tuple[Path, DurableArtifact]]:
    plan: dict[str, tuple[Path, DurableArtifact]] = {}
    for shard in shards:
        for artifact in shard.completion.durable_artifacts:
            relative = artifact.relative_path
            if relative in _SHARD_ONLY_DURABLE_PATHS:
                continue
            source = shard.workspace / relative
            previous = plan.get(relative)
            if previous is not None:
                deduplicable = (
                    relative in _DEDUPLICATED_MERGE_PATHS
                    or relative.startswith("provenance/environment/")
                    or relative.startswith("provenance/hardware/")
                )
                if not deduplicable or previous[1] != artifact:
                    raise ShardError(f"non-deduplicable or conflicting merge path {relative!r}")
                if previous[0].read_bytes() != source.read_bytes():
                    raise ShardError(f"same-hash merge path has different bytes: {relative!r}")
                continue
            plan[relative] = (source, artifact)
    return dict(sorted(plan.items()))


def _copy_artifact(source: Path, destination: Path, expected: DurableArtifact) -> None:
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size != expected.byte_length
        or file_sha256(source) != expected.sha256
    ):
        raise ShardError(
            f"source shard artifact changed before copy: {expected.relative_path}"
        )
    payload = source.read_bytes()
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.read_bytes() != payload
        ):
            raise ShardError(
                f"resumable merge destination conflicts: {expected.relative_path}"
            )
        return
    atomic_write_bytes(destination, payload)


def _named_durable_artifact(
    shard: _VerifiedShard, relative_path: str
) -> DurableArtifact:
    matches = tuple(
        row
        for row in shard.completion.durable_artifacts
        if row.relative_path == relative_path
    )
    if len(matches) != 1:
        raise ShardError(f"shard receipt does not bind {relative_path!r} exactly once")
    return matches[0]


def _copy_shard_identity_artifacts(
    shard: _VerifiedShard, destination_root: Path
) -> None:
    group = shard.contract.shard_id
    _copy_artifact(
        shard.workspace / "state" / "shard_contract.json",
        destination_root / "state" / "shard_contracts" / f"{group}.json",
        _named_durable_artifact(shard, "state/shard_contract.json"),
    )
    _copy_artifact(
        shard.workspace / "state" / "shard_completion.json",
        destination_root / "state" / "shard_completions" / f"{group}.json",
        shard.completion_artifact,
    )


def _verify_copied_shard_identities(
    shard: _VerifiedShard, destination_root: Path
) -> None:
    group = shard.contract.shard_id
    _copy_shard_identity_artifacts(shard, destination_root)
    copied_contract = ShardContract.from_dict(
        _load_json(destination_root / "state" / "shard_contracts" / f"{group}.json"),
        shard.manifest,
        load_task_registry(),
    )
    copied_completion = ShardCompletionReceipt.from_dict(
        _load_json(
            destination_root / "state" / "shard_completions" / f"{group}.json"
        ),
        copied_contract,
    )
    if copied_contract != shard.contract or copied_completion != shard.completion:
        raise ShardError("copied shard contract/completion differs from its source")


def _artifact_rows_for_paths(root: Path, paths: Iterable[str]) -> tuple[DurableArtifact, ...]:
    rows = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ShardError(f"merged artifact is missing or unsafe: {relative}")
        rows.append(DurableArtifact(relative, path.stat().st_size, file_sha256(path)))
    return tuple(rows)


def _expected_merged_paths(
    copy_plan: Mapping[str, tuple[Path, DurableArtifact]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *copy_plan,
                "state/task_loop.json",
                "state/merge_journal.json",
                *(f"state/shard_contracts/{group}.json" for group in SHARD_IDS),
                *(f"state/shard_completions/{group}.json" for group in SHARD_IDS),
            }
        )
    )


_FINAL_ARTIFACT_PATHS = {
    "final/accepted_labels.jsonl",
    "final/failed_attempts.jsonl",
    "final/target_transform.json",
    "final/audit_report.json",
    "final/dataset_manifest.json",
    "final/completion_receipt.json",
}


def _assert_merged_workspace_allowlist(
    root: Path,
    expected_paths: Sequence[str],
    *,
    finalized: bool,
) -> None:
    base = {
        *expected_paths,
        "state/merge_receipt.json",
    }
    allowed = {*base, *_FINAL_ARTIFACT_PATHS}
    actual = {
        path.relative_to(root).as_posix()
        for path in _files_below(root)
    }
    valid = actual == allowed if finalized else base <= actual <= allowed
    if not valid:
        required = allowed if finalized else base
        missing = sorted(required - actual)
        extra = sorted(actual - allowed)
        raise ShardError(
            f"merged workspace allowlist differs; missing={missing[:5]}, extra={extra[:5]}"
        )


def _build_merge_receipt(
    root: Path,
    journal: MergeJournal,
    expected_paths: Sequence[str],
) -> MergeReceipt:
    artifacts = _artifact_rows_for_paths(root, expected_paths)
    draft = MergeReceipt(
        version=MERGE_RECEIPT_VERSION,
        merge_identity_sha256=journal.merge_identity_sha256,
        target_manifest_sha256=journal.target_manifest_sha256,
        task_registry_sha256=journal.task_registry_sha256,
        production_source_lock_sha256=journal.production_source_lock_sha256,
        campaign_environment_sha256=journal.campaign_environment_sha256,
        shard_receipt_sha256s=journal.shard_receipt_sha256s,
        task_count=22,
        root_candidate_count=18_000,
        accepted_configuration_count=18_000,
        measured_epoch_record_count=54_000,
        merged_artifact_count=len(artifacts),
        merged_artifacts_sha256=canonical_sha256(artifacts),
        receipt_sha256="",
    )
    receipt = replace(
        draft, receipt_sha256=canonical_sha256(draft.unhashed_payload())
    )
    receipt.validate()
    return receipt


def _verify_published_merge(
    output: Path,
    shards: Sequence[_VerifiedShard],
    journal: MergeJournal,
) -> Mapping[str, Any]:
    from .finalization import finalize_workspace

    copy_plan = _merge_copy_plan(shards)
    expected_paths = _expected_merged_paths(copy_plan)
    stored_journal = MergeJournal.from_dict(
        _load_json(output / "state" / "merge_journal.json")
    )
    if stored_journal != journal:
        raise ShardError("published merge journal differs from source shards")
    expected_receipt = _build_merge_receipt(output, journal, expected_paths)
    actual_receipt = MergeReceipt.from_dict(
        _load_json(output / "state" / "merge_receipt.json")
    )
    if actual_receipt != expected_receipt:
        raise ShardError("published merge receipt differs from recomputation")
    _assert_merged_workspace_allowlist(output, expected_paths, finalized=True)
    for relative, (source, artifact) in copy_plan.items():
        destination = output / relative
        if (
            not destination.is_file()
            or destination.stat().st_size != artifact.byte_length
            or file_sha256(destination) != artifact.sha256
            or source.read_bytes() != destination.read_bytes()
        ):
            raise ShardError(f"published merged artifact differs: {relative}")
    for shard in shards:
        _verify_copied_shard_identities(shard, output)
    final_receipt = finalize_workspace(output, verify_only=True)
    for shard in shards:
        _verify_artifact_manifest(
            shard.workspace, shard.completion.durable_artifacts
        )
    return final_receipt


def merge_shard_workspaces(
    shard_workspaces: Mapping[str, str | Path],
    output_workspace: str | Path,
    repository_root: str | Path,
    *,
    verify_only: bool = False,
) -> Mapping[str, Any]:
    """Strictly union three verified shards and run the unchanged finalizer."""

    from .finalization import finalize_workspace
    from .materialization import TaskLoopState, save_task_loop_state

    if set(shard_workspaces) != set(SHARD_IDS):
        raise ShardError("merge inputs must name nlp, vision, and rest exactly once")
    shards = tuple(
        _verified_shard(shard_workspaces[shard_id], repository_root)
        for shard_id in SHARD_IDS
    )
    if any(
        row.contract.shard_id != expected
        for row, expected in zip(shards, SHARD_IDS, strict=True)
    ):
        raise ShardError("merge workspace is assigned to the wrong shard argument")
    journal = _validate_merge_sources(shards)
    output = Path(output_workspace).resolve()
    source_roots = {row.workspace for row in shards}
    if output in source_roots or any(
        output.is_relative_to(root) or root.is_relative_to(output)
        for root in source_roots
    ):
        raise ShardError("merge output must be separate from every source shard workspace")
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ShardError("merge output path is not a safe directory")
        return _verify_published_merge(output, shards, journal)
    if verify_only:
        raise ShardError("verified merged output workspace is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.merge-{journal.merge_identity_sha256[:16]}"
    if staging.exists() and (not staging.is_dir() or staging.is_symlink()):
        raise ShardError("merge staging path is unsafe")
    staging.mkdir(parents=False, exist_ok=True)
    journal_path = staging / "state" / "merge_journal.json"
    if journal_path.exists():
        if MergeJournal.from_dict(_load_json(journal_path)) != journal:
            raise ShardError("existing merge staging journal targets another shard set")
    else:
        atomic_write_json(journal_path, journal.to_dict())
    copy_plan = _merge_copy_plan(shards)
    for relative, (source, artifact) in copy_plan.items():
        _copy_artifact(source, staging / relative, artifact)
    manifest = shards[0].manifest
    tasks = load_task_registry()
    task_receipts: dict[str, TaskCompletionReceipt] = {}
    for task in tasks.entries:
        receipt = TaskCompletionReceipt.from_dict(
            _load_json(
                staging
                / "state"
                / "completed_materializations"
                / f"{task.task_id}.json"
            )
        )
        task_receipts[task.task_id] = receipt
    full_order = tuple(row.task_id for row in tasks.entries)
    full_loop = TaskLoopState(
        version=TASK_LOOP_STATE_VERSION,
        target_manifest_sha256=manifest.sha256,
        task_registry_sha256=tasks.sha256,
        completed_task_ids=full_order,
        completed_receipt_sha256s=tuple(
            task_receipts[task_id].receipt_sha256 for task_id in full_order
        ),
        active_task_id=None,
    )
    save_task_loop_state(staging / "state" / "task_loop.json", full_loop, full_order)
    for shard in shards:
        _copy_shard_identity_artifacts(shard, staging)
    expected_paths = _expected_merged_paths(copy_plan)
    merge_receipt = _build_merge_receipt(staging, journal, expected_paths)
    atomic_write_json(staging / "state" / "merge_receipt.json", merge_receipt.to_dict())
    _assert_merged_workspace_allowlist(staging, expected_paths, finalized=False)
    finalize_workspace(staging)
    finalize_workspace(staging, verify_only=True)
    _assert_merged_workspace_allowlist(staging, expected_paths, finalized=True)
    for shard in shards:
        _verify_copied_shard_identities(shard, staging)
        _verify_artifact_manifest(
            shard.workspace, shard.completion.durable_artifacts
        )
    os.replace(staging, output)
    return _verify_published_merge(output, shards, journal)


__all__ = [
    "EXPECTED_SHARD_ROOT_COUNTS",
    "EXPECTED_SHARD_TASK_COUNTS",
    "MERGE_JOURNAL_VERSION",
    "MERGE_RECEIPT_VERSION",
    "PRODUCTION_SOURCE_LOCK_VERSION",
    "SHARD_COMPLETION_VERSION",
    "SHARD_CONTRACT_VERSION",
    "SHARD_GROUPING_RULE_VERSION",
    "SHARD_IDS",
    "SHARD_TASK_LOOP_VERSION",
    "DurableArtifact",
    "MergeJournal",
    "MergeReceipt",
    "ProductionSourceLock",
    "ShardCompletionReceipt",
    "ShardContract",
    "ShardError",
    "ShardTaskLoopState",
    "build_production_source_lock",
    "build_shard_contract",
    "finalize_shard_workspace",
    "freeze_production_source_lock",
    "freeze_shard_contract",
    "initial_shard_task_loop",
    "load_shard_task_loop",
    "merge_shard_workspaces",
    "save_shard_task_loop",
    "task_entries_for_shard",
    "validate_shard_partition",
]
