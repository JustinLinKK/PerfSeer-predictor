"""Minimal local-disk admission and atomic state writes for the 18K pack.

There is deliberately no artifact ledger or storage controller.  The AWS
runner keeps one reconstructible Kaggle task at a time and applies the one
projection formula defined by the dataset design report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .fingerprints import canonical_sha256, canonical_value


GIB = 1024**3
DEFAULT_STORAGE_BUDGET_BYTES = 600 * GIB
DEFAULT_SAFETY_HEADROOM_BYTES = 40 * GIB
DISK_GUARD_VERSION = "perfseer_v3_a10g_disk_guard_v1"


class DiskAdmissionError(RuntimeError):
    """Raised when a task cannot fit or a disk projection is malformed."""


def _nonnegative_int(value: int, *, context: str) -> None:
    if type(value) is not int or value < 0:
        raise DiskAdmissionError(f"{context} must be a nonnegative integer")


def atomic_write_bytes(path: str | Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Publish bytes with a same-directory temporary file, fsync, and replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        canonical_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, encoded)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = b"".join(
        json.dumps(
            canonical_value(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, encoded)


def path_size_bytes(path: str | Path) -> int:
    """Return allocated file lengths without following directory symlinks."""

    root = Path(path)
    if not root.exists() and not root.is_symlink():
        return 0
    if root.is_symlink() or root.is_file():
        return root.lstat().st_size
    total = root.lstat().st_size
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in directories:
            child = current_path / name
            total += child.lstat().st_size
            if not child.is_symlink():
                retained_directories.append(name)
        directories[:] = retained_directories
        for name in files:
            total += (current_path / name).lstat().st_size
    return total


def non_task_workspace_bytes(workspace: str | Path, task_cache: str | Path) -> int:
    """Measure retained workspace data while excluding the current task cache."""

    root = Path(workspace).resolve()
    excluded = Path(task_cache).resolve()
    try:
        excluded.relative_to(root)
    except ValueError as error:
        raise DiskAdmissionError("task cache must be inside the workspace") from error
    if excluded == root:
        raise DiskAdmissionError("task cache cannot be the workspace root")
    if not root.exists():
        return 0
    total = root.lstat().st_size
    for child in root.iterdir():
        resolved = child.resolve()
        if resolved == excluded or excluded.is_relative_to(resolved):
            continue
        total += path_size_bytes(child)
    return total


@dataclass(frozen=True)
class DiskAdmission:
    version: str
    task_id: str
    stage: str
    current_non_task_bytes: int
    task_archive_bytes: int
    extracted_bytes: int
    extraction_temporary_bytes: int
    already_present_task_bytes: int
    safety_headroom_bytes: int
    storage_budget_bytes: int
    observed_free_bytes: int
    projected_peak_bytes: int
    additional_required_bytes: int
    accepted: bool
    refusal_reasons: tuple[str, ...]

    def validate(self) -> None:
        if self.version != DISK_GUARD_VERSION:
            raise DiskAdmissionError("disk admission version mismatch")
        if not self.task_id or self.stage not in {
            "before_download",
            "before_extraction",
            "before_preparation",
        }:
            raise DiskAdmissionError("disk admission identity or stage is invalid")
        for name in (
            "current_non_task_bytes",
            "task_archive_bytes",
            "extracted_bytes",
            "extraction_temporary_bytes",
            "already_present_task_bytes",
            "safety_headroom_bytes",
            "storage_budget_bytes",
            "observed_free_bytes",
            "projected_peak_bytes",
            "additional_required_bytes",
        ):
            _nonnegative_int(getattr(self, name), context=name)
        expected_peak = (
            self.current_non_task_bytes
            + self.task_archive_bytes
            + self.extracted_bytes
            + self.extraction_temporary_bytes
            + self.safety_headroom_bytes
        )
        if self.projected_peak_bytes != expected_peak:
            raise DiskAdmissionError("projected peak does not match the disk formula")
        task_peak = (
            self.task_archive_bytes
            + self.extracted_bytes
            + self.extraction_temporary_bytes
        )
        expected_additional = max(0, task_peak - self.already_present_task_bytes)
        if self.additional_required_bytes != expected_additional:
            raise DiskAdmissionError("additional disk requirement is inconsistent")
        expected_reasons = []
        if expected_peak >= self.storage_budget_bytes:
            expected_reasons.append("projected_peak_reaches_storage_budget")
        if expected_additional + self.safety_headroom_bytes > self.observed_free_bytes:
            expected_reasons.append("observed_free_space_cannot_preserve_headroom")
        if tuple(expected_reasons) != self.refusal_reasons:
            raise DiskAdmissionError("disk refusal reasons are inconsistent")
        if self.accepted != (not expected_reasons):
            raise DiskAdmissionError("disk admission decision is inconsistent")

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_value(asdict(self))
        payload["admission_sha256"] = self.sha256
        return payload

    def require(self) -> "DiskAdmission":
        self.validate()
        if not self.accepted:
            reasons = ", ".join(self.refusal_reasons)
            raise DiskAdmissionError(
                f"task {self.task_id!r} cannot enter {self.stage}: {reasons}; "
                f"projected={self.projected_peak_bytes} bytes, "
                f"budget={self.storage_budget_bytes} bytes, "
                f"free={self.observed_free_bytes} bytes"
            )
        return self


@dataclass(frozen=True)
class FreeSpaceGuard:
    storage_budget_bytes: int = DEFAULT_STORAGE_BUDGET_BYTES
    safety_headroom_bytes: int = DEFAULT_SAFETY_HEADROOM_BYTES

    def __post_init__(self) -> None:
        _nonnegative_int(self.storage_budget_bytes, context="storage budget")
        _nonnegative_int(self.safety_headroom_bytes, context="safety headroom")
        if self.storage_budget_bytes <= self.safety_headroom_bytes:
            raise DiskAdmissionError("storage budget must exceed safety headroom")

    def assess(
        self,
        *,
        task_id: str,
        stage: str,
        current_non_task_bytes: int,
        task_archive_bytes: int,
        extracted_bytes: int,
        extraction_temporary_bytes: int,
        observed_free_bytes: int,
        already_present_task_bytes: int = 0,
    ) -> DiskAdmission:
        values = {
            "current_non_task_bytes": current_non_task_bytes,
            "task_archive_bytes": task_archive_bytes,
            "extracted_bytes": extracted_bytes,
            "extraction_temporary_bytes": extraction_temporary_bytes,
            "observed_free_bytes": observed_free_bytes,
            "already_present_task_bytes": already_present_task_bytes,
        }
        for name, value in values.items():
            _nonnegative_int(value, context=name)
        task_peak = task_archive_bytes + extracted_bytes + extraction_temporary_bytes
        projected = current_non_task_bytes + task_peak + self.safety_headroom_bytes
        additional = max(0, task_peak - already_present_task_bytes)
        reasons = []
        if projected >= self.storage_budget_bytes:
            reasons.append("projected_peak_reaches_storage_budget")
        if additional + self.safety_headroom_bytes > observed_free_bytes:
            reasons.append("observed_free_space_cannot_preserve_headroom")
        result = DiskAdmission(
            version=DISK_GUARD_VERSION,
            task_id=task_id,
            stage=stage,
            current_non_task_bytes=current_non_task_bytes,
            task_archive_bytes=task_archive_bytes,
            extracted_bytes=extracted_bytes,
            extraction_temporary_bytes=extraction_temporary_bytes,
            already_present_task_bytes=already_present_task_bytes,
            safety_headroom_bytes=self.safety_headroom_bytes,
            storage_budget_bytes=self.storage_budget_bytes,
            observed_free_bytes=observed_free_bytes,
            projected_peak_bytes=projected,
            additional_required_bytes=additional,
            accepted=not reasons,
            refusal_reasons=tuple(reasons),
        )
        result.validate()
        return result


__all__ = [
    "DEFAULT_SAFETY_HEADROOM_BYTES",
    "DEFAULT_STORAGE_BUDGET_BYTES",
    "DISK_GUARD_VERSION",
    "DiskAdmission",
    "DiskAdmissionError",
    "FreeSpaceGuard",
    "GIB",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "non_task_workspace_bytes",
    "path_size_bytes",
]
