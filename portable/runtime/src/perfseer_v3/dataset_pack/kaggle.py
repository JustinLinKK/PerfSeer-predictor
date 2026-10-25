"""Direct Kaggle CLI acquisition and fail-closed ZIP handling.

This module has no EC2 identity check and no NRP API integration.  Credentials
remain owned by the operator outside the repository.  The runner downloads one
competition archive, records its hash, validates every path, and extracts it
into the external task workspace.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

from .fingerprints import canonical_sha256, canonical_value, file_sha256


KAGGLE_MATERIALIZATION_VERSION = "perfseer_v3_v100_kaggle_materialization_v2"
KAGGLE_INVENTORY_PAGE_SIZE = 200
_NEXT_PAGE_TOKEN_PREFIX = "Next Page Token = "


class KaggleMaterializationError(RuntimeError):
    """Raised for missing credentials, CLI failures, or unsafe archives."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class KaggleCredentialEvidence:
    source: str
    location: str | None

    def validate(self) -> None:
        if self.source not in {"api_token_environment", "legacy_environment", "external_file"}:
            raise KaggleMaterializationError("Kaggle credential source is invalid")
        if self.source == "external_file" and not self.location:
            raise KaggleMaterializationError("external Kaggle credential path is missing")
        if self.source != "external_file" and self.location is not None:
            raise KaggleMaterializationError("environment credentials must not expose a location")


def validate_external_credentials(
    repository_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> KaggleCredentialEvidence:
    """Prove only the credential source/location; never read or return a secret."""

    env = os.environ if environment is None else environment
    if env.get("KAGGLE_API_TOKEN"):
        evidence = KaggleCredentialEvidence("api_token_environment", None)
        evidence.validate()
        return evidence
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        evidence = KaggleCredentialEvidence("legacy_environment", None)
        evidence.validate()
        return evidence
    config_dir = Path(env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    credential = (config_dir / "kaggle.json").expanduser()
    if _inside(credential, Path(repository_root)):
        raise KaggleMaterializationError(
            "Kaggle credentials must be outside the repository checkout"
        )
    if not credential.is_file():
        raise KaggleMaterializationError(
            "Kaggle credentials are missing; set KAGGLE_API_TOKEN, set the legacy "
            "environment pair, or place kaggle.json outside the repository"
        )
    if credential.stat().st_mode & 0o077:
        raise KaggleMaterializationError(
            "external kaggle.json must not be readable by group or other users"
        )
    evidence = KaggleCredentialEvidence("external_file", str(credential.resolve()))
    evidence.validate()
    return evidence


def _parse_size(value: str) -> int:
    text = value.strip().replace(",", "")
    if text.isdigit():
        return int(text)
    suffixes = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    upper = text.upper()
    for suffix in sorted(suffixes, key=len, reverse=True):
        if upper.endswith(suffix):
            number = upper[: -len(suffix)].strip()
            try:
                parsed = float(number)
            except ValueError as error:
                raise KaggleMaterializationError(
                    f"cannot parse Kaggle file size {value!r}"
                ) from error
            if not parsed >= 0:
                break
            return int(parsed * suffixes[suffix])
    raise KaggleMaterializationError(f"cannot parse Kaggle file size {value!r}")


@dataclass(frozen=True)
class KaggleRemoteFile:
    name: str
    size_bytes: int
    creation_timestamp: str

    def validate(self) -> None:
        if not self.name or self.name.startswith(("/", "\\")) or "\\" in self.name:
            raise KaggleMaterializationError("Kaggle remote file name is unsafe")
        if ".." in PurePosixPath(self.name).parts:
            raise KaggleMaterializationError("Kaggle remote file traverses its root")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise KaggleMaterializationError("Kaggle remote file size is invalid")
        if type(self.creation_timestamp) is not str:
            raise KaggleMaterializationError("Kaggle remote timestamp must be text")


@dataclass(frozen=True)
class KaggleCompetitionProbe:
    slug: str
    files: tuple[KaggleRemoteFile, ...]

    def validate(self) -> None:
        if not self.slug or not self.files:
            raise KaggleMaterializationError("Kaggle competition inventory is empty")
        for row in self.files:
            row.validate()
        if len({row.name.casefold() for row in self.files}) != len(self.files):
            raise KaggleMaterializationError("Kaggle inventory has duplicate paths")

    @property
    def compressed_bytes(self) -> int:
        self.validate()
        return sum(row.size_bytes for row in self.files)

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256([asdict(row) for row in self.files])


@dataclass(frozen=True)
class KaggleDownloadProbe:
    slug: str
    remote_name: str
    advertised_bytes: int
    downloaded_bytes: int
    downloaded_sha256: str

    def validate(self) -> None:
        KaggleRemoteFile(self.remote_name, self.advertised_bytes, "").validate()
        if not self.slug or self.downloaded_bytes < 1:
            raise KaggleMaterializationError("Kaggle download probe is empty")
        if len(self.downloaded_sha256) != 64:
            raise KaggleMaterializationError("Kaggle download probe hash is invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


class KaggleCliClient:
    """Small secret-safe wrapper around the official `kaggle` executable."""

    def __init__(
        self,
        *,
        executable: str = "kaggle",
        timeout_seconds: int = 3600,
        maximum_attempts: int = 4,
        initial_backoff_seconds: float = 1.0,
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise KaggleMaterializationError("the official Kaggle CLI is not installed")
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise KaggleMaterializationError("Kaggle CLI timeout must be positive")
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 8:
            raise KaggleMaterializationError("Kaggle retry count must be between one and eight")
        if (
            isinstance(initial_backoff_seconds, bool)
            or not isinstance(initial_backoff_seconds, (int, float))
            or not 0 <= float(initial_backoff_seconds) <= 10
        ):
            raise KaggleMaterializationError("Kaggle retry backoff must be between zero and ten seconds")
        self.executable = resolved
        self.timeout_seconds = timeout_seconds
        self.maximum_attempts = maximum_attempts
        self.initial_backoff_seconds = float(initial_backoff_seconds)

    @staticmethod
    def _failure_message(stderr: str, stdout: str) -> str:
        # Inspect only for classification.  Never include CLI output because it may
        # contain operator-specific paths or credential details.
        lowered = f"{stderr}\n{stdout}".lower()
        if "401" in lowered or "unauthorized" in lowered or "credential" in lowered:
            return "Kaggle authentication failed; verify the external credential source"
        if "403" in lowered or "forbidden" in lowered or "accept" in lowered:
            return "Kaggle competition rules are not accepted for this account"
        if "404" in lowered or "not found" in lowered:
            return "the frozen Kaggle competition is unavailable"
        if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
            return "Kaggle rate limit remained active after bounded retry/backoff"
        return "the Kaggle CLI request failed; verify CLI installation and network access"

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        for attempt in range(self.maximum_attempts):
            try:
                result = subprocess.run(
                    [self.executable, *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as error:
                raise KaggleMaterializationError("the Kaggle CLI request timed out") from error
            if result.returncode == 0:
                return result
            lowered = f"{result.stderr}\n{result.stdout}".lower()
            retryable = any(
                token in lowered
                for token in ("429", "too many requests", "rate limit")
            )
            if not retryable or attempt + 1 == self.maximum_attempts:
                raise KaggleMaterializationError(
                    self._failure_message(result.stderr, result.stdout)
                )
            time.sleep(min(10.0, self.initial_backoff_seconds * (2**attempt)))
        raise AssertionError("bounded Kaggle retry loop did not terminate")

    def authenticate(self) -> None:
        self._run(["competitions", "list", "--page-size", "1", "--csv"])

    @staticmethod
    def _parse_files_page(
        payload: str,
    ) -> tuple[tuple[KaggleRemoteFile, ...], str | None]:
        lines = payload.splitlines()
        next_page_token = None
        if lines and lines[0].startswith(_NEXT_PAGE_TOKEN_PREFIX):
            next_page_token = lines.pop(0)[len(_NEXT_PAGE_TOKEN_PREFIX) :].strip()
            if not next_page_token:
                raise KaggleMaterializationError("Kaggle returned an empty next-page token")
        csv_payload = "\n".join(lines)
        rows = []
        for raw in csv.DictReader(io.StringIO(csv_payload)):
            name = str(raw.get("name", raw.get("Name", ""))).strip()
            size = str(raw.get("size", raw.get("Size", ""))).strip()
            created = str(
                raw.get("creationDate", raw.get("CreationDate", raw.get("date", "")))
            )
            if not name or not size:
                raise KaggleMaterializationError("Kaggle file inventory schema is unrecognized")
            row = KaggleRemoteFile(name, _parse_size(size), created)
            row.validate()
            rows.append(row)
        if not rows:
            raise KaggleMaterializationError("Kaggle file inventory page is empty")
        return tuple(rows), next_page_token

    @staticmethod
    def _parse_files_csv(payload: str) -> tuple[KaggleRemoteFile, ...]:
        rows, next_page_token = KaggleCliClient._parse_files_page(payload)
        if next_page_token is not None:
            raise KaggleMaterializationError("Kaggle file inventory requires pagination")
        return tuple(sorted(rows, key=lambda row: row.name))

    def probe_competition(self, slug: str) -> KaggleCompetitionProbe:
        files: list[KaggleRemoteFile] = []
        next_page_token = None
        seen_page_tokens: set[str] = set()
        while True:
            arguments = [
                "competitions",
                "files",
                "--competition",
                slug,
                "--page-size",
                str(KAGGLE_INVENTORY_PAGE_SIZE),
                "--csv",
                "--quiet",
            ]
            if next_page_token is not None:
                arguments.extend(("--page-token", next_page_token))
            result = self._run(arguments)
            page, returned_token = self._parse_files_page(result.stdout)
            files.extend(page)
            if returned_token is None:
                break
            if returned_token in seen_page_tokens:
                raise KaggleMaterializationError("Kaggle repeated an inventory page token")
            seen_page_tokens.add(returned_token)
            next_page_token = returned_token
        probe = KaggleCompetitionProbe(
            slug,
            tuple(sorted(files, key=lambda row: row.name)),
        )
        probe.validate()
        return probe

    def download_smallest_file(self, slug: str) -> KaggleDownloadProbe:
        """Actually download and hash the smallest advertised competition file.

        Listing files alone does not prove that the account accepted competition
        rules.  The temporary payload is deleted before this method returns.
        """

        inventory = self.probe_competition(slug)
        selected = min(inventory.files, key=lambda row: (row.size_bytes, row.name))
        with tempfile.TemporaryDirectory(prefix="perfseer-kaggle-access-") as directory:
            destination = Path(directory)
            self._run(
                [
                    "competitions",
                    "download",
                    "--competition",
                    slug,
                    "--file",
                    selected.name,
                    "--path",
                    str(destination),
                    "--quiet",
                ]
            )
            downloaded = tuple(
                path
                for path in destination.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
            if len(downloaded) != 1:
                raise KaggleMaterializationError(
                    "smallest-file access probe did not produce exactly one regular file"
                )
            payload = downloaded[0]
            result = KaggleDownloadProbe(
                slug=slug,
                remote_name=selected.name,
                advertised_bytes=selected.size_bytes,
                downloaded_bytes=payload.stat().st_size,
                downloaded_sha256=file_sha256(payload),
            )
            result.validate()
            return result

    def download_competition(
        self,
        slug: str,
        destination_archive: str | Path,
    ) -> Path:
        destination = Path(destination_archive)
        if destination.name != f"{slug}.zip":
            raise KaggleMaterializationError("destination archive name must match the frozen slug")
        if destination.exists() or destination.is_symlink():
            raise KaggleMaterializationError(
                "destination archive must be absent so a download cannot be trusted implicitly"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / ".kaggle-download"
        staging.mkdir(parents=True, exist_ok=True)
        downloaded = staging / f"{slug}.zip"
        if downloaded.exists() or downloaded.is_symlink():
            downloaded.unlink()
        self._run(
            [
                "competitions",
                "download",
                "--competition",
                slug,
                "--path",
                str(staging),
                "--quiet",
            ]
        )
        if not downloaded.is_file():
            raise KaggleMaterializationError(
                "Kaggle download returned success without the expected archive"
            )
        os.replace(downloaded, destination)
        try:
            staging.rmdir()
        except OSError:
            pass
        return destination


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    is_directory: bool


@dataclass(frozen=True)
class ArchiveInventory:
    version: str
    archive_name: str
    archive_sha256: str
    archive_bytes: int
    compressed_member_bytes: int
    uncompressed_bytes: int
    largest_member_bytes: int
    members: tuple[ArchiveMember, ...]
    inventory_sha256: str

    def validate(self) -> None:
        if self.version != KAGGLE_MATERIALIZATION_VERSION or not self.archive_name:
            raise KaggleMaterializationError("archive inventory identity is invalid")
        if len(self.archive_sha256) != 64 or len(self.inventory_sha256) != 64:
            raise KaggleMaterializationError("archive inventory hashes are invalid")
        for name in (
            "archive_bytes",
            "compressed_member_bytes",
            "uncompressed_bytes",
            "largest_member_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise KaggleMaterializationError(f"{name} is invalid")
        if not self.members:
            raise KaggleMaterializationError("archive contains no members")
        if self.compressed_member_bytes != sum(row.compressed_bytes for row in self.members):
            raise KaggleMaterializationError("compressed member total is inconsistent")
        if self.uncompressed_bytes != sum(row.uncompressed_bytes for row in self.members):
            raise KaggleMaterializationError("uncompressed member total is inconsistent")
        if self.largest_member_bytes != max(row.uncompressed_bytes for row in self.members):
            raise KaggleMaterializationError("largest archive member is inconsistent")
        payload = [asdict(row) for row in self.members]
        if self.inventory_sha256 != canonical_sha256(payload):
            raise KaggleMaterializationError("archive inventory hash is inconsistent")

    @property
    def extraction_temporary_bytes(self) -> int:
        self.validate()
        return max(self.largest_member_bytes, (self.uncompressed_bytes + 19) // 20)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


def _validated_member_path(info: zipfile.ZipInfo) -> tuple[str, bool]:
    name = info.filename
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise KaggleMaterializationError("ZIP contains an unsafe member path")
    pure = PurePosixPath(name.rstrip("/"))
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise KaggleMaterializationError("ZIP member path escapes or aliases its root")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    is_directory = info.is_dir()
    if stat.S_ISLNK(mode):
        raise KaggleMaterializationError("ZIP symlink members are forbidden")
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise KaggleMaterializationError("ZIP special-file members are forbidden")
    if is_directory != (kind == stat.S_IFDIR) and kind != 0:
        raise KaggleMaterializationError("ZIP member type metadata is inconsistent")
    if info.flag_bits & 0x1:
        raise KaggleMaterializationError("encrypted ZIP members are unsupported")
    return pure.as_posix(), is_directory


def inspect_zip_archive(
    archive_path: str | Path,
    *,
    maximum_uncompressed_bytes: int,
) -> ArchiveInventory:
    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise KaggleMaterializationError("Kaggle archive is missing")
    if type(maximum_uncompressed_bytes) is not int or maximum_uncompressed_bytes < 1:
        raise KaggleMaterializationError("archive extraction ceiling must be positive")
    members: list[ArchiveMember] = []
    seen: set[str] = set()
    files: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                normalized, is_directory = _validated_member_path(info)
                identity = normalized.casefold()
                if identity in seen:
                    raise KaggleMaterializationError("ZIP has duplicate normalized paths")
                seen.add(identity)
                parents = PurePosixPath(normalized).parents
                if any(parent.as_posix().casefold() in files for parent in parents):
                    raise KaggleMaterializationError("ZIP file path conflicts with a directory")
                if not is_directory:
                    prefix = identity + "/"
                    if any(existing.startswith(prefix) for existing in seen - {identity}):
                        raise KaggleMaterializationError("ZIP file path conflicts with a directory")
                    files.add(identity)
                members.append(
                    ArchiveMember(
                        path=normalized,
                        compressed_bytes=info.compress_size,
                        uncompressed_bytes=info.file_size,
                        crc32=info.CRC,
                        is_directory=is_directory,
                    )
                )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise KaggleMaterializationError("Kaggle archive is not a readable ZIP") from error
    if not members:
        raise KaggleMaterializationError("Kaggle archive is empty")
    uncompressed = sum(row.uncompressed_bytes for row in members)
    if uncompressed > maximum_uncompressed_bytes:
        raise KaggleMaterializationError(
            "ZIP uncompressed total exceeds the frozen task extraction ceiling"
        )
    inventory_payload = [asdict(row) for row in members]
    result = ArchiveInventory(
        version=KAGGLE_MATERIALIZATION_VERSION,
        archive_name=archive.name,
        archive_sha256=file_sha256(archive),
        archive_bytes=archive.stat().st_size,
        compressed_member_bytes=sum(row.compressed_bytes for row in members),
        uncompressed_bytes=uncompressed,
        largest_member_bytes=max(row.uncompressed_bytes for row in members),
        members=tuple(members),
        inventory_sha256=canonical_sha256(inventory_payload),
    )
    result.validate()
    return result


def extract_zip_archive(
    archive_path: str | Path,
    destination: str | Path,
    inventory: ArchiveInventory,
) -> Path:
    """Extract a previously inspected ZIP into a newly published directory."""

    inventory.validate()
    archive = Path(archive_path)
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise KaggleMaterializationError("extraction destination already exists")
    if archive.name != inventory.archive_name or file_sha256(archive) != inventory.archive_sha256:
        raise KaggleMaterializationError("archive changed after safety inspection")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    written = 0
    try:
        with zipfile.ZipFile(archive) as stream:
            infos = stream.infolist()
            if len(infos) != len(inventory.members):
                raise KaggleMaterializationError("archive changed after inventory creation")
            for info, expected in zip(infos, inventory.members, strict=True):
                normalized, is_directory = _validated_member_path(info)
                if normalized != expected.path or is_directory != expected.is_directory:
                    raise KaggleMaterializationError("archive member changed after inspection")
                output = staging.joinpath(*PurePosixPath(normalized).parts)
                if is_directory:
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                digest_crc = 0
                with stream.open(info, "r") as source, output.open("xb") as sink:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > inventory.uncompressed_bytes:
                            raise KaggleMaterializationError(
                                "archive expanded beyond its inspected size"
                            )
                        digest_crc = zipfile.crc32(block, digest_crc)
                        sink.write(block)
                    sink.flush()
                    os.fsync(sink.fileno())
                if output.stat().st_size != expected.uncompressed_bytes:
                    raise KaggleMaterializationError("extracted member size mismatch")
                if digest_crc & 0xFFFFFFFF != expected.crc32:
                    raise KaggleMaterializationError("extracted member CRC mismatch")
        if written != sum(
            row.uncompressed_bytes for row in inventory.members if not row.is_directory
        ):
            raise KaggleMaterializationError("extracted byte total mismatch")
        os.replace(staging, target)
        if file_sha256(archive) != inventory.archive_sha256:
            shutil.rmtree(target)
            raise KaggleMaterializationError("archive changed during extraction")
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def inspect_nested_zip_archives(
    extracted_root: str | Path,
    *,
    maximum_uncompressed_bytes: int,
) -> tuple[ArchiveInventory, ...]:
    """Validate nested ZIPs before a pinned preparer is allowed to open them."""

    root = Path(extracted_root)
    inventories = []
    for archive in sorted(root.rglob("*.zip")):
        if archive.is_symlink() or not archive.is_file():
            raise KaggleMaterializationError("nested archive path is not a regular file")
        inventories.append(
            inspect_zip_archive(
                archive,
                maximum_uncompressed_bytes=maximum_uncompressed_bytes,
            )
        )
    return tuple(inventories)


__all__ = [
    "ArchiveInventory",
    "ArchiveMember",
    "KAGGLE_MATERIALIZATION_VERSION",
    "KaggleCliClient",
    "KaggleCompetitionProbe",
    "KaggleCredentialEvidence",
    "KaggleDownloadProbe",
    "KaggleMaterializationError",
    "KaggleRemoteFile",
    "extract_zip_archive",
    "inspect_nested_zip_archives",
    "inspect_zip_archive",
    "validate_external_credentials",
]
