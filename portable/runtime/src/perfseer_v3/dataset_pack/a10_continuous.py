"""Single-submission orchestration for the Disaster V2 A10 campaign."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .fingerprints import canonical_sha256, canonical_value, file_sha256
from .storage import atomic_write_json


CONTINUOUS_PROGRESS_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_continuous_progress_v1"
)
CONTINUOUS_RECEIPT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_continuous_receipt_v1"
)
CONTINUOUS_CHUNK_COUNT = 44
CONTINUOUS_TOTAL_LABELS = 11_200
CONTINUOUS_PILOT_LABELS = 32
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class A10ContinuousError(RuntimeError):
    """Raised when continuous orchestration cannot safely advance."""


@dataclass(frozen=True)
class ContinuousDependencies:
    """Injectable phase operations used by the production runner and unit tests."""

    run_phase: Callable[..., Mapping[str, Any]]
    verify_campaign: Callable[..., Mapping[str, Any]]
    export_release: Callable[..., Mapping[str, Any]]
    verify_release: Callable[..., Mapping[str, Any]]


def continuous_progress_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / "state" / "continuous_campaign_progress.json"


def continuous_receipt_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / "state" / "continuous_campaign_receipt.json"


@contextmanager
def _continuous_controller_lock(workspace: Path):
    """Reject overlapping controller Pods while allowing phase-level locks."""

    path = workspace / "state" / "continuous-campaign-controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise A10ContinuousError(
                "another writer holds the continuous campaign controller lock"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _default_dependencies() -> ContinuousDependencies:
    from .a10_campaign import run_campaign, verify_campaign
    from .a10_export import export_release, verify_release

    return ContinuousDependencies(
        run_phase=run_campaign,
        verify_campaign=verify_campaign,
        export_release=export_release,
        verify_release=verify_release,
    )


def _emit(
    workspace: Path,
    *,
    event: str,
    phase: str,
    status: str,
    completed_chunks: int,
    accepted_labels: int,
    details: Mapping[str, Any] | None = None,
    event_sink: Callable[[str], None] | None = print,
) -> Mapping[str, Any]:
    if not 0 <= completed_chunks <= CONTINUOUS_CHUNK_COUNT:
        raise A10ContinuousError("continuous completed-chunk count is invalid")
    if not 0 <= accepted_labels <= CONTINUOUS_TOTAL_LABELS:
        raise A10ContinuousError("continuous accepted-label count is invalid")
    payload: dict[str, Any] = {
        "version": CONTINUOUS_PROGRESS_VERSION,
        "event": event,
        "phase": phase,
        "status": status,
        "completed_chunks": completed_chunks,
        "accepted_labels": accepted_labels,
        "details": dict(details or {}),
    }
    payload["progress_sha256"] = canonical_sha256(payload)
    value = canonical_value(payload)
    atomic_write_json(continuous_progress_path(workspace), value)
    if event_sink is not None:
        event_sink(json.dumps({"perfseer_continuous": value}, sort_keys=True))
    return value


def _receipt_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(payload) != claimed:
        raise A10ContinuousError("continuous receipt hash differs")
    return payload


def _load_terminal_receipt(
    path: Path,
    *,
    repository_revision: str,
    image_digest: str,
    output_directory: Path,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise A10ContinuousError("continuous receipt path is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise A10ContinuousError("continuous receipt is unreadable") from error
    if not isinstance(value, Mapping):
        raise A10ContinuousError("continuous receipt is not an object")
    payload = _receipt_payload(value)
    if (
        payload.get("version") != CONTINUOUS_RECEIPT_VERSION
        or payload.get("repository_revision") != repository_revision
        or payload.get("image_digest") != image_digest
        or payload.get("completed_chunks") != CONTINUOUS_CHUNK_COUNT
        or payload.get("accepted_labels") != CONTINUOUS_TOTAL_LABELS
    ):
        raise A10ContinuousError("continuous receipt identity differs")
    archive = Path(str(payload.get("archive", ""))).resolve()
    try:
        archive.relative_to(output_directory)
    except ValueError as error:
        raise A10ContinuousError("continuous receipt archive is outside the release path") from error
    if not archive.is_file() or file_sha256(archive) != payload.get("archive_sha256"):
        raise A10ContinuousError("continuous receipt archive differs")
    return canonical_value(value)


def _phase_hash(receipt: Mapping[str, Any], *, phase: str) -> str:
    value = receipt.get("receipt_sha256")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise A10ContinuousError(f"{phase} returned no valid receipt hash")
    return value


def _accepted_count(verification: Mapping[str, Any], *, minimum: int) -> int:
    value = verification.get("accepted_labels")
    if type(value) is not int or not minimum <= value <= CONTINUOUS_TOTAL_LABELS:
        raise A10ContinuousError("campaign verification accepted-label count differs")
    return value


def run_continuous_campaign(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    repository_revision: str,
    image_digest: str,
    output_directory: str | Path,
    kaggle_executable: str = "kaggle",
    preflight: Callable[[], Any] | None = None,
    dependencies: ContinuousDependencies | None = None,
    event_sink: Callable[[str], None] | None = print,
) -> Mapping[str, Any]:
    """Run pilot, all chunks, verification, and export in one resumable process."""

    root = Path(workspace).resolve()
    release_root = Path(output_directory).resolve()
    if release_root != root / "releases":
        raise A10ContinuousError("continuous releases must be written under the workspace")
    dependencies = dependencies or _default_dependencies()
    terminal_path = continuous_receipt_path(root)
    terminal = _load_terminal_receipt(
        terminal_path,
        repository_revision=repository_revision,
        image_digest=image_digest,
        output_directory=release_root,
    )
    if terminal is not None:
        verification = dependencies.verify_campaign(root, complete=True)
        if _accepted_count(verification, minimum=CONTINUOUS_TOTAL_LABELS) != CONTINUOUS_TOTAL_LABELS:
            raise A10ContinuousError("terminal campaign verification is incomplete")
        archive_verification = dependencies.verify_release(terminal["archive"])
        if archive_verification.get("archive_sha256") != terminal["archive_sha256"]:
            raise A10ContinuousError("terminal archive verification differs")
        _emit(
            root,
            event="terminal_verified",
            phase="complete",
            status="complete",
            completed_chunks=CONTINUOUS_CHUNK_COUNT,
            accepted_labels=CONTINUOUS_TOTAL_LABELS,
            details={"archive_sha256": terminal["archive_sha256"]},
            event_sink=event_sink,
        )
        return terminal

    phase = "controller-lock"
    completed_chunks = 0
    accepted_labels = 0
    phase_receipts: list[Mapping[str, Any]] = []
    controller_lock_entered = False
    try:
        controller_lock = _continuous_controller_lock(root)
        controller_lock.__enter__()
        controller_lock_entered = True
        phase = "preflight"
        _emit(
            root,
            event="phase_start",
            phase=phase,
            status="running",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            event_sink=event_sink,
        )
        if preflight is not None:
            preflight()
        _emit(
            root,
            event="phase_complete",
            phase=phase,
            status="passed",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            event_sink=event_sink,
        )

        phase = "pilot"
        _emit(
            root,
            event="phase_start",
            phase=phase,
            status="running",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            event_sink=event_sink,
        )
        pilot_receipt = dependencies.run_phase(
            workspace=root,
            repository_root=Path(repository_root).resolve(),
            mlebench_checkout=Path(mlebench_checkout).resolve(),
            repository_revision=repository_revision,
            image_digest=image_digest,
            kaggle_executable=kaggle_executable,
            pilot=True,
            max_new_accepted=256,
        )
        phase_receipts.append(
            {"phase": phase, "receipt_sha256": _phase_hash(pilot_receipt, phase=phase)}
        )
        verification = dependencies.verify_campaign(root, complete=False)
        accepted_labels = _accepted_count(
            verification, minimum=CONTINUOUS_PILOT_LABELS
        )
        _emit(
            root,
            event="phase_complete",
            phase=phase,
            status="passed",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            details={"receipt_sha256": phase_receipts[-1]["receipt_sha256"]},
            event_sink=event_sink,
        )

        for chunk_index in range(CONTINUOUS_CHUNK_COUNT):
            phase = f"chunk-{chunk_index:02d}"
            _emit(
                root,
                event="phase_start",
                phase=phase,
                status="running",
                completed_chunks=completed_chunks,
                accepted_labels=accepted_labels,
                event_sink=event_sink,
            )
            chunk_receipt = dependencies.run_phase(
                workspace=root,
                repository_root=Path(repository_root).resolve(),
                mlebench_checkout=Path(mlebench_checkout).resolve(),
                repository_revision=repository_revision,
                image_digest=image_digest,
                kaggle_executable=kaggle_executable,
                chunk_index=chunk_index,
                max_new_accepted=256,
            )
            phase_receipts.append(
                {
                    "phase": phase,
                    "receipt_sha256": _phase_hash(chunk_receipt, phase=phase),
                }
            )
            verification = dependencies.verify_campaign(root, complete=False)
            completed_chunks = chunk_index + 1
            expected_accepted = min(
                CONTINUOUS_PILOT_LABELS + completed_chunks * 256,
                CONTINUOUS_TOTAL_LABELS,
            )
            accepted_labels = _accepted_count(
                verification, minimum=expected_accepted
            )
            _emit(
                root,
                event="phase_complete",
                phase=phase,
                status="passed",
                completed_chunks=completed_chunks,
                accepted_labels=accepted_labels,
                details={"receipt_sha256": phase_receipts[-1]["receipt_sha256"]},
                event_sink=event_sink,
            )

        phase = "complete-verification"
        _emit(
            root,
            event="phase_start",
            phase=phase,
            status="running",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            event_sink=event_sink,
        )
        complete_verification = dependencies.verify_campaign(root, complete=True)
        accepted_labels = _accepted_count(
            complete_verification, minimum=CONTINUOUS_TOTAL_LABELS
        )
        if accepted_labels != CONTINUOUS_TOTAL_LABELS:
            raise A10ContinuousError("continuous campaign did not reach 11,200 labels")
        _emit(
            root,
            event="phase_complete",
            phase=phase,
            status="passed",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            details={
                "campaign_contract_sha256": complete_verification.get(
                    "campaign_contract_sha256"
                )
            },
            event_sink=event_sink,
        )

        phase = "export"
        _emit(
            root,
            event="phase_start",
            phase=phase,
            status="running",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            event_sink=event_sink,
        )
        export = dependencies.export_release(
            root,
            Path(repository_root).resolve(),
            release_root,
            complete=True,
        )
        archive = Path(str(export.get("archive", ""))).resolve()
        try:
            archive.relative_to(release_root)
        except ValueError as error:
            raise A10ContinuousError("export archive is outside the release path") from error
        archive_sha256 = export.get("archive_sha256")
        if (
            export.get("complete") is not True
            or export.get("label_count") != CONTINUOUS_TOTAL_LABELS
            or not archive.is_file()
            or not isinstance(archive_sha256, str)
            or file_sha256(archive) != archive_sha256
        ):
            raise A10ContinuousError("continuous export archive differs")
        archive_verification = dependencies.verify_release(archive)
        if archive_verification.get("archive_sha256") != archive_sha256:
            raise A10ContinuousError("continuous export verification differs")

        receipt_payload: dict[str, Any] = {
            "version": CONTINUOUS_RECEIPT_VERSION,
            "repository_revision": repository_revision,
            "image_digest": image_digest,
            "completed_chunks": completed_chunks,
            "accepted_labels": accepted_labels,
            "campaign_contract_sha256": complete_verification.get(
                "campaign_contract_sha256"
            ),
            "phase_receipts": phase_receipts,
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "release_sha256": export.get("release_sha256"),
        }
        receipt_payload["receipt_sha256"] = canonical_sha256(receipt_payload)
        terminal = canonical_value(receipt_payload)
        atomic_write_json(terminal_path, terminal)
        _emit(
            root,
            event="phase_complete",
            phase=phase,
            status="passed",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            details={"archive": str(archive), "archive_sha256": archive_sha256},
            event_sink=event_sink,
        )
        _emit(
            root,
            event="campaign_complete",
            phase="complete",
            status="complete",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            details={
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "receipt_sha256": terminal["receipt_sha256"],
            },
            event_sink=event_sink,
        )
        return terminal
    except BaseException as error:
        _emit(
            root,
            event="campaign_failed",
            phase=phase,
            status="failed",
            completed_chunks=completed_chunks,
            accepted_labels=accepted_labels,
            details={"error_type": type(error).__name__, "message": str(error)},
            event_sink=event_sink,
        )
        raise
    finally:
        if controller_lock_entered:
            controller_lock.__exit__(None, None, None)


__all__ = [
    "A10ContinuousError",
    "CONTINUOUS_CHUNK_COUNT",
    "CONTINUOUS_PILOT_LABELS",
    "CONTINUOUS_PROGRESS_VERSION",
    "CONTINUOUS_RECEIPT_VERSION",
    "CONTINUOUS_TOTAL_LABELS",
    "ContinuousDependencies",
    "continuous_progress_path",
    "continuous_receipt_path",
    "run_continuous_campaign",
]
