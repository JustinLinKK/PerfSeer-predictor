"""Verified, weight-free release export for the non-vision A10 corpus."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

try:
    import pyzstd as _zstd_backend

    def _zstd_compress(value: bytes) -> bytes:
        return _zstd_backend.compress(value, level_or_option=19)

    def _zstd_decompress(value: bytes) -> bytes:
        return _zstd_backend.decompress(value)
except ImportError:  # Local verifier environments may provide python-zstandard.
    import zstandard as _zstd_backend

    def _zstd_compress(value: bytes) -> bytes:
        return _zstd_backend.ZstdCompressor(level=19).compress(value)

    def _zstd_decompress(value: bytes) -> bytes:
        return _zstd_backend.ZstdDecompressor().decompress(value)

from .a10_campaign import (
    TOTAL_CANDIDATES,
    _load_record_for_root,
    build_campaign_contract,
)
from .contracts import AttemptStatus, label_run_record_from_dict
from .fingerprints import canonical_sha256, canonical_value, file_sha256
from .labeler_profile import PROFILE
from .sampler import build_target_manifest, target_candidate_from_dict
from .storage import atomic_write_json, atomic_write_jsonl
from .task_registry import load_task_registry


EXPORT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_release_v2"
    if PROFILE.uses_disaster_v2
    else "perfseer_v3_nrp_a10_nonvision_release_v1"
)
SOURCE_BUNDLE_VERSION = "perfseer_v3_content_addressed_source_bundle_v1"
FORBIDDEN_WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INDEX_KEYS = {
    "root_candidate_id",
    "resolved_candidate_id",
    "label_line",
    "configuration_path",
    "configuration_sha256",
    "factory_entrypoint",
    "runtime_source_bundle_sha256",
    "runtime_source_bundle_path",
    "model_source_bundle_sha256",
    "model_source_bundle_path",
}
_RELEASE_MANIFEST_KEYS = {
    "version",
    "status",
    "label_count",
    "candidate_index_count",
    "campaign_contract_sha256",
    "target_manifest_sha256",
    "runtime_source_bundle_sha256",
    "model_source_bundle_count",
    "contains_model_weights",
    "release_manifest_sha256",
}
_CONFIGURATION_KEYS = {
    "version",
    "root_candidate_id",
    "resolved_candidate_id",
    "family_id",
    "factory_entrypoint",
    "task_id",
    "task_kind",
    "target_width",
    "architecture",
    "optimizer",
    "scheduler",
    "precision",
    "requested_effective_batch",
    "runtime_microbatch",
    "gradient_accumulation",
    "repair_history",
    "execution_mode",
    "activation_checkpointing",
    "seed_policy",
    "regime",
    "source_sha256",
    "resolved_candidate",
}


class A10ExportError(RuntimeError):
    """Raised when a release is incomplete, mutable, or not reconstructable."""


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise A10ExportError(f"{context} is not a lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, *, context: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise A10ExportError(f"{context} is not a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path == PurePosixPath(".")
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise A10ExportError(f"{context} is not a safe relative POSIX path")
    return path


def _load_json_object(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise A10ExportError(f"{context} is unreadable") from error
    if not isinstance(value, Mapping):
        raise A10ExportError(f"{context} must contain one JSON object")
    return value


def _load_jsonl_objects(path: Path, *, context: str) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise A10ExportError(f"{context} is unreadable") from error
    if not lines or any(not line.strip() for line in lines):
        raise A10ExportError(f"{context} must contain non-empty JSONL rows")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise A10ExportError(
                f"{context} line {line_number} is not valid JSON"
            ) from error
        if not isinstance(value, Mapping):
            raise A10ExportError(f"{context} line {line_number} is not an object")
        rows.append(value)
    return tuple(rows)


def _source_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _copy_bundle(
    *,
    repository_root: Path,
    release_root: Path,
    relative_files: Sequence[Path],
    bundle_kind: str,
) -> tuple[str, str]:
    hashes = {
        path.as_posix(): file_sha256(repository_root / path)
        for path in sorted(set(relative_files))
    }
    bundle_sha256 = canonical_sha256(
        {
            "version": SOURCE_BUNDLE_VERSION,
            "kind": bundle_kind,
            "files": hashes,
        }
    )
    relative_bundle = Path("source-bundles") / bundle_kind / bundle_sha256
    destination = release_root / relative_bundle
    for relative in sorted(hashes):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative, target)
    manifest: dict[str, Any] = {
        "version": SOURCE_BUNDLE_VERSION,
        "kind": bundle_kind,
        "bundle_sha256": bundle_sha256,
        "files": hashes,
    }
    atomic_write_json(destination / "bundle-manifest.json", manifest)
    return bundle_sha256, relative_bundle.as_posix()


def _canonical_training_configuration(
    root_candidate: Any,
    resolved_candidate: Any,
    task_entry: Any,
) -> Mapping[str, Any]:
    requested = root_candidate.microbatch_size * root_candidate.gradient_accumulation_steps
    final_effective = (
        resolved_candidate.microbatch_size
        * resolved_candidate.gradient_accumulation_steps
    )
    if requested != final_effective or requested not in {32, 64, 128, 256, 512}:
        raise A10ExportError("resolved candidate did not preserve requested effective batch")
    return canonical_value(
        {
            "version": (
                "perfseer_v3_nrp_a10_nonvision_disaster_training_configuration_v2"
                if PROFILE.uses_disaster_v2
                else "perfseer_v3_nrp_a10_nonvision_training_configuration_v1"
            ),
            "root_candidate_id": root_candidate.candidate_id,
            "resolved_candidate_id": resolved_candidate.candidate_id,
            "family_id": resolved_candidate.family_id,
            "factory_entrypoint": f"{resolved_candidate.factory_id}:build_model",
            "task_id": resolved_candidate.task_id,
            "task_kind": str(task_entry.target_schema["kind"]),
            "target_width": resolved_candidate.target_width,
            "architecture": resolved_candidate.architecture_parameters,
            "optimizer": resolved_candidate.optimizer,
            "scheduler": resolved_candidate.scheduler,
            "precision": resolved_candidate.precision_policy,
            "requested_effective_batch": requested,
            "runtime_microbatch": resolved_candidate.microbatch_size,
            "gradient_accumulation": resolved_candidate.gradient_accumulation_steps,
            "repair_history": resolved_candidate.mutation_specification,
            "execution_mode": resolved_candidate.execution,
            "activation_checkpointing": resolved_candidate.activation_checkpointing,
            "seed_policy": resolved_candidate.seed_policy,
            "regime": resolved_candidate.regime,
            "source_sha256": resolved_candidate.source_sha256,
            "resolved_candidate": resolved_candidate.to_dict(),
        }
    )


def _write_sha256sums(root: Path) -> None:
    rows = []
    for path in _source_files(root):
        if path.name == "SHA256SUMS":
            continue
        rows.append(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _deterministic_tar(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as stream:
        tar_path = Path(stream.name)
    try:
        with tarfile.open(tar_path, "w") as archive:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as file_stream:
                        archive.addfile(info, file_stream)
                else:
                    archive.addfile(info)
        compressed = _zstd_compress(tar_path.read_bytes())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_bytes(compressed)
        os.replace(temporary, destination)
    finally:
        tar_path.unlink(missing_ok=True)


def export_release(
    workspace: str | Path,
    repository_root: str | Path,
    output_directory: str | Path,
    *,
    complete: bool,
) -> Mapping[str, Any]:
    if not PROFILE.is_nonvision_4gpu:
        raise A10ExportError("non-vision export requires its baked image profile")
    workspace = Path(workspace).resolve()
    repository_root = Path(repository_root).resolve()
    output_directory = Path(output_directory).resolve()
    manifest = build_target_manifest()
    tasks = {row.task_id: row for row in load_task_registry().entries}
    resolved_rows = []
    for root_candidate in manifest.candidates:
        resolved = _load_record_for_root(workspace, root_candidate)
        if resolved is not None:
            slot, record = resolved
            record.validate()
            if record.status != AttemptStatus.ACCEPTED:
                raise A10ExportError("release contains a non-accepted label")
            resolved_rows.append((root_candidate, slot.current_candidate, record))
    if complete and len(resolved_rows) != TOTAL_CANDIDATES:
        raise A10ExportError("complete export requires all 11,200 accepted labels")
    if not resolved_rows:
        raise A10ExportError("release has no accepted labels")
    with tempfile.TemporaryDirectory(prefix="perfseer-nonvision-release-") as directory:
        release = Path(directory) / "release"
        release.mkdir()
        source_root = repository_root / "src" / "perfseer_v3"
        runtime_files = tuple(
            path.relative_to(repository_root) for path in _source_files(source_root)
        )
        runtime_sha, runtime_path = _copy_bundle(
            repository_root=repository_root,
            release_root=release,
            relative_files=runtime_files,
            bundle_kind="runtime",
        )
        family_bundles: dict[str, tuple[str, str]] = {}
        for family in sorted({row.family_id for _, row, _ in resolved_rows}):
            models = Path("src/perfseer_v3/dataset_pack/models")
            files = (
                models / "__init__.py",
                models / "base.py",
                models / "module_api.py",
                models / "factory.py",
                models / "architecture.py",
                models / f"{family}.py",
            )
            if family == "independent_generated":
                files = (*files, Path("src/perfseer_v3/dataset_pack/generated_lineages.py"))
            family_bundles[family] = _copy_bundle(
                repository_root=repository_root,
                release_root=release,
                relative_files=files,
                bundle_kind="model",
            )
        labels = []
        indexes = []
        for line_number, (root_candidate, resolved_candidate, record) in enumerate(
            resolved_rows, 1
        ):
            labels.append(canonical_value(asdict(record)))
            configuration = _canonical_training_configuration(
                root_candidate, resolved_candidate, tasks[resolved_candidate.task_id]
            )
            configuration_path = (
                Path("configurations") / f"{root_candidate.candidate_id}.json"
            )
            atomic_write_json(release / configuration_path, configuration)
            model_sha, model_path = family_bundles[resolved_candidate.family_id]
            indexes.append(
                {
                    "root_candidate_id": root_candidate.candidate_id,
                    "resolved_candidate_id": resolved_candidate.candidate_id,
                    "label_line": line_number,
                    "configuration_path": configuration_path.as_posix(),
                    "configuration_sha256": canonical_sha256(configuration),
                    "factory_entrypoint": f"{resolved_candidate.factory_id}:build_model",
                    "runtime_source_bundle_sha256": runtime_sha,
                    "runtime_source_bundle_path": runtime_path,
                    "model_source_bundle_sha256": model_sha,
                    "model_source_bundle_path": model_path,
                }
            )
        atomic_write_jsonl(release / "labels.jsonl", labels)
        atomic_write_jsonl(release / "candidate-index.jsonl", indexes)
        for source in (
            workspace / "attempts" / "failed",
            workspace / "attempts" / "repairs",
            workspace / "attempts" / "diagnostics",
        ):
            if source.is_dir():
                shutil.copytree(source, release / "failure-artifacts" / source.name)
        for pattern in (
            "state/*receipt*.json",
            "state/**/*receipt*.json",
            "state/*manifest*.json",
            "state/*ledger*.json",
            "state/**/*ledger*.json",
        ):
            for source in sorted(workspace.glob(pattern)):
                if source.is_file():
                    relative = source.relative_to(workspace)
                    target = release / "campaign-artifacts" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        release_manifest: dict[str, Any] = {
            "version": EXPORT_VERSION,
            "status": "complete" if complete else "partial",
            "label_count": len(labels),
            "candidate_index_count": len(indexes),
            "campaign_contract_sha256": build_campaign_contract()["contract_sha256"],
            "target_manifest_sha256": manifest.sha256,
            "runtime_source_bundle_sha256": runtime_sha,
            "model_source_bundle_count": len(family_bundles),
            "contains_model_weights": False,
        }
        release_manifest["release_manifest_sha256"] = canonical_sha256(release_manifest)
        atomic_write_json(release / "release-manifest.json", release_manifest)
        forbidden = [
            path
            for path in _source_files(release)
            if path.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
            or "checkpoint" in path.name.lower()
        ]
        if forbidden:
            raise A10ExportError("release unexpectedly contains weights/checkpoints")
        _write_sha256sums(release)
        release_hash = canonical_sha256(
            {
                path.relative_to(release).as_posix(): file_sha256(path)
                for path in _source_files(release)
            }
        )
        archive = output_directory / (
            f"perfseer-v3-a10-nonvision-"
            f"{'disaster-v2-' if PROFILE.uses_disaster_v2 else ''}"
            f"{'complete' if complete else 'partial'}-"
            f"{release_hash[:16]}.tar.zst"
        )
        _deterministic_tar(release, archive)
    result = {
        "version": EXPORT_VERSION,
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
        "release_sha256": release_hash,
        "label_count": len(resolved_rows),
        "complete": complete,
    }
    atomic_write_json(archive.with_suffix(archive.suffix + ".json"), result)
    return canonical_value(result)


def _safe_extract(archive: Path, destination: Path) -> None:
    raw = _zstd_decompress(archive.read_bytes())
    tar_path = destination / "release.tar"
    tar_path.write_bytes(raw)
    with tarfile.open(tar_path, "r") as bundle:
        members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        member_paths: set[str] = set()
        file_paths: set[str] = set()
        for member in bundle.getmembers():
            if not (member.isfile() or member.isdir()):
                raise A10ExportError("release archive contains an unsafe member type")
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = _safe_relative_path(name, context="release archive member")
            normalized = path.as_posix()
            if normalized in member_paths:
                raise A10ExportError("release archive contains a duplicate member")
            member_paths.add(normalized)
            if member.isfile():
                file_paths.add(normalized)
            members.append((member, path))
        for path in member_paths:
            parent = PurePosixPath(path).parent
            while parent != PurePosixPath("."):
                if parent.as_posix() in file_paths:
                    raise A10ExportError("release archive member has a file parent")
                parent = parent.parent
        extraction_root = destination / "release"
        for member, relative in members:
            target = extraction_root.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise A10ExportError("release archive file has no payload")
            with source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    tar_path.unlink()


def _release_file_map(release: Path) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(release.rglob("*")):
        if path.is_symlink():
            raise A10ExportError("release contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise A10ExportError("release contains a non-regular path")
        relative = path.relative_to(release).as_posix()
        _safe_relative_path(relative, context="release file")
        result[relative] = path
    return result


def _verify_sha256sums(release: Path) -> Mapping[str, Path]:
    files = _release_file_map(release)
    sums_path = files.get("SHA256SUMS")
    if sums_path is None:
        raise A10ExportError("release has no SHA256SUMS")
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise A10ExportError("release SHA256SUMS is unreadable") from error
    if not lines or any(not line for line in lines):
        raise A10ExportError("release SHA256SUMS contains an empty row")
    checksums: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line in lines:
        digest, separator, relative = line.partition("  ")
        _require_sha256(digest, context="release checksum")
        safe = _safe_relative_path(relative, context="release checksum path")
        normalized = safe.as_posix()
        if separator != "  " or normalized == "SHA256SUMS" or normalized in checksums:
            raise A10ExportError("release SHA256SUMS row is malformed or duplicated")
        checksums[normalized] = digest
        ordered_paths.append(normalized)
    expected = set(files) - {"SHA256SUMS"}
    if set(checksums) != expected or ordered_paths != sorted(ordered_paths):
        raise A10ExportError("release SHA256SUMS coverage differs from archive files")
    for relative, digest in checksums.items():
        if file_sha256(files[relative]) != digest:
            raise A10ExportError(f"release checksum differs for {relative}")
    forbidden = [
        relative
        for relative in files
        if (
            PurePosixPath(relative).suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
            or "checkpoint" in PurePosixPath(relative).name.lower()
            or "__pycache__" in PurePosixPath(relative).parts
            or PurePosixPath(relative).suffix.lower() in {".pyc", ".pyo"}
        )
    ]
    if forbidden:
        raise A10ExportError("release contains weights, checkpoints, or bytecode")
    allowed_roots = {
        "SHA256SUMS",
        "labels.jsonl",
        "candidate-index.jsonl",
        "release-manifest.json",
        "configurations",
        "source-bundles",
        "failure-artifacts",
        "campaign-artifacts",
    }
    if any(PurePosixPath(relative).parts[0] not in allowed_roots for relative in files):
        raise A10ExportError("release contains a file outside the artifact contract")
    return files


def _verify_source_bundle(
    release: Path,
    *,
    relative_path: Any,
    claimed_sha256: Any,
    kind: str,
) -> Mapping[str, Any]:
    digest = _require_sha256(claimed_sha256, context=f"{kind} source bundle digest")
    relative = _safe_relative_path(relative_path, context=f"{kind} source bundle path")
    expected_path = PurePosixPath("source-bundles") / kind / digest
    if relative != expected_path:
        raise A10ExportError(f"{kind} source bundle path differs from its identity")
    bundle = release.joinpath(*relative.parts)
    if not bundle.is_dir() or bundle.is_symlink():
        raise A10ExportError(f"{kind} source bundle is not a directory")
    manifest = _load_json_object(
        bundle / "bundle-manifest.json", context=f"{kind} source bundle manifest"
    )
    if set(manifest) != {"version", "kind", "bundle_sha256", "files"}:
        raise A10ExportError(f"{kind} source bundle manifest schema differs")
    if (
        manifest["version"] != SOURCE_BUNDLE_VERSION
        or manifest["kind"] != kind
        or manifest["bundle_sha256"] != digest
        or not isinstance(manifest["files"], Mapping)
        or not manifest["files"]
    ):
        raise A10ExportError(f"{kind} source bundle manifest identity differs")
    hashes: dict[str, str] = {}
    for raw_relative, raw_digest in manifest["files"].items():
        source_relative = _safe_relative_path(
            raw_relative, context=f"{kind} source file path"
        ).as_posix()
        source_digest = _require_sha256(
            raw_digest, context=f"{kind} source file digest"
        )
        if source_relative == "bundle-manifest.json" or source_relative in hashes:
            raise A10ExportError(f"{kind} source bundle contains a duplicate file")
        hashes[source_relative] = source_digest
    payload = {"version": SOURCE_BUNDLE_VERSION, "kind": kind, "files": hashes}
    if canonical_sha256(payload) != digest:
        raise A10ExportError(f"{kind} source bundle digest differs")
    actual = {
        path.relative_to(bundle).as_posix(): path
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    }
    if set(actual) != {*hashes, "bundle-manifest.json"}:
        raise A10ExportError(f"{kind} source bundle file coverage differs")
    for source_relative, source_digest in hashes.items():
        if file_sha256(actual[source_relative]) != source_digest:
            raise A10ExportError(
                f"{kind} source bundle checksum differs for {source_relative}"
            )
    return manifest


def _verify_release_manifest(release: Path) -> Mapping[str, Any]:
    manifest = _load_json_object(
        release / "release-manifest.json", context="release manifest"
    )
    if set(manifest) != _RELEASE_MANIFEST_KEYS:
        raise A10ExportError("release manifest schema differs")
    unhashed = dict(manifest)
    claimed = _require_sha256(
        unhashed.pop("release_manifest_sha256"), context="release manifest digest"
    )
    if canonical_sha256(unhashed) != claimed:
        raise A10ExportError("release manifest digest differs")
    label_count = manifest["label_count"]
    index_count = manifest["candidate_index_count"]
    model_bundle_count = manifest["model_source_bundle_count"]
    if (
        manifest["version"] != EXPORT_VERSION
        or manifest["status"] not in {"partial", "complete"}
        or type(label_count) is not int
        or type(index_count) is not int
        or type(model_bundle_count) is not int
        or label_count < 1
        or index_count != label_count
        or model_bundle_count < 1
        or manifest["contains_model_weights"] is not False
    ):
        raise A10ExportError("release manifest values differ")
    if manifest["status"] == "complete" and label_count != TOTAL_CANDIDATES:
        raise A10ExportError("complete release does not contain 11,200 labels")
    if manifest["status"] == "partial" and label_count > TOTAL_CANDIDATES:
        raise A10ExportError("partial release exceeds the campaign size")
    for key in (
        "campaign_contract_sha256",
        "target_manifest_sha256",
        "runtime_source_bundle_sha256",
    ):
        _require_sha256(manifest[key], context=key)
    return manifest


def _verify_configuration(
    configuration: Mapping[str, Any],
    index: Mapping[str, Any],
    label: Any,
    task_kind: str,
) -> Any:
    if set(configuration) != _CONFIGURATION_KEYS:
        raise A10ExportError("candidate configuration schema differs")
    try:
        candidate = target_candidate_from_dict(configuration["resolved_candidate"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise A10ExportError("resolved candidate is invalid") from error
    factory_entrypoint = f"{candidate.factory_id}:build_model"
    requested_batch = configuration["requested_effective_batch"]
    if (
        configuration["version"]
        != (
            "perfseer_v3_nrp_a10_nonvision_disaster_training_configuration_v2"
            if PROFILE.uses_disaster_v2
            else "perfseer_v3_nrp_a10_nonvision_training_configuration_v1"
        )
        or configuration["root_candidate_id"] != index["root_candidate_id"]
        or configuration["resolved_candidate_id"] != index["resolved_candidate_id"]
        or candidate.candidate_id != index["resolved_candidate_id"]
        or label.configuration_id != index["resolved_candidate_id"]
        or configuration["family_id"] != candidate.family_id
        or configuration["factory_entrypoint"] != factory_entrypoint
        or index["factory_entrypoint"] != factory_entrypoint
        or configuration["task_id"] != candidate.task_id
        or configuration["task_kind"] != task_kind
        or configuration["target_width"] != candidate.target_width
        or configuration["architecture"] != candidate.architecture_parameters
        or configuration["optimizer"] != candidate.optimizer
        or configuration["scheduler"] != candidate.scheduler
        or configuration["precision"] != candidate.precision_policy
        or configuration["runtime_microbatch"] != candidate.microbatch_size
        or configuration["gradient_accumulation"]
        != candidate.gradient_accumulation_steps
        or configuration["repair_history"] != candidate.mutation_specification
        or configuration["execution_mode"] != candidate.execution
        or configuration["activation_checkpointing"]
        != candidate.activation_checkpointing
        or configuration["seed_policy"] != candidate.seed_policy
        or configuration["regime"] != candidate.regime
        or configuration["source_sha256"] != candidate.source_sha256
        or requested_batch not in {32, 64, 128, 256, 512}
        or requested_batch
        != candidate.microbatch_size * candidate.gradient_accumulation_steps
    ):
        raise A10ExportError("release label/configuration/index join differs")
    return candidate


def _verify_release_tree(release: Path) -> tuple[Mapping[str, Any], ...]:
    release_files = _verify_sha256sums(release)
    manifest = _verify_release_manifest(release)
    labels = _load_jsonl_objects(release / "labels.jsonl", context="release labels")
    indexes = _load_jsonl_objects(
        release / "candidate-index.jsonl", context="candidate index"
    )
    if len(labels) != len(indexes) or len(labels) != manifest["label_count"]:
        raise A10ExportError("release label/index counts differ")
    root_ids: set[str] = set()
    resolved_ids: set[str] = set()
    configuration_paths: set[str] = set()
    runtime_bundles: set[tuple[str, str]] = set()
    model_bundles: set[tuple[str, str]] = set()
    family_bundles: dict[str, tuple[str, str]] = {}
    verified_bundles: set[tuple[str, str, str]] = set()
    tasks = {row.task_id: row for row in load_task_registry().entries}
    for line_number, (raw_label, index) in enumerate(
        zip(labels, indexes, strict=True), 1
    ):
        if set(index) != _INDEX_KEYS or index["label_line"] != line_number:
            raise A10ExportError("candidate index schema or label order differs")
        root_id = _require_sha256(index["root_candidate_id"], context="root candidate ID")
        resolved_id = _require_sha256(
            index["resolved_candidate_id"], context="resolved candidate ID"
        )
        configuration_digest = _require_sha256(
            index["configuration_sha256"], context="configuration digest"
        )
        configuration_relative = _safe_relative_path(
            index["configuration_path"], context="configuration path"
        )
        if configuration_relative != PurePosixPath("configurations") / f"{root_id}.json":
            raise A10ExportError("configuration path differs from its root candidate")
        if (
            root_id in root_ids
            or resolved_id in resolved_ids
            or configuration_relative.as_posix() in configuration_paths
        ):
            raise A10ExportError("release candidate identity is duplicated")
        root_ids.add(root_id)
        resolved_ids.add(resolved_id)
        configuration_paths.add(configuration_relative.as_posix())
        try:
            label = label_run_record_from_dict(raw_label)
            label.validate()
        except (KeyError, TypeError, ValueError) as error:
            raise A10ExportError("release label is invalid") from error
        if label.status != AttemptStatus.ACCEPTED:
            raise A10ExportError("release contains a non-accepted label")
        configuration = _load_json_object(
            release.joinpath(*configuration_relative.parts),
            context="candidate configuration",
        )
        if canonical_sha256(configuration) != configuration_digest:
            raise A10ExportError("candidate configuration digest differs")
        if configuration.get("task_id") not in tasks:
            raise A10ExportError("candidate configuration references an unknown task")
        task_kind = str(tasks[str(configuration["task_id"])].target_schema["kind"])
        candidate = _verify_configuration(configuration, index, label, task_kind)
        for kind in ("runtime", "model"):
            path_key = f"{kind}_source_bundle_path"
            digest_key = f"{kind}_source_bundle_sha256"
            bundle_identity = (str(index[path_key]), str(index[digest_key]))
            if kind == "runtime":
                runtime_bundles.add(bundle_identity)
            else:
                model_bundles.add(bundle_identity)
                previous = family_bundles.setdefault(candidate.family_id, bundle_identity)
                if previous != bundle_identity:
                    raise A10ExportError("one family references multiple model source bundles")
            verification_key = (kind, *bundle_identity)
            if verification_key not in verified_bundles:
                source_manifest = _verify_source_bundle(
                    release,
                    relative_path=index[path_key],
                    claimed_sha256=index[digest_key],
                    kind=kind,
                )
                files = source_manifest["files"]
                if kind == "runtime" and (
                    "src/perfseer_v3/dataset_pack/models/factory.py" not in files
                ):
                    raise A10ExportError("runtime source bundle lacks the model factory")
                if kind == "model" and (
                    f"src/perfseer_v3/dataset_pack/models/{candidate.family_id}.py"
                    not in files
                ):
                    raise A10ExportError("model source bundle lacks its family module")
                verified_bundles.add(verification_key)
    if (
        len(runtime_bundles) != 1
        or next(iter(runtime_bundles))[1]
        != manifest["runtime_source_bundle_sha256"]
        or len(model_bundles) != manifest["model_source_bundle_count"]
        or len(model_bundles) != len(family_bundles)
    ):
        raise A10ExportError("release source bundle counts or identities differ")
    actual_configuration_paths = {
        relative
        for relative in release_files
        if PurePosixPath(relative).parts[0] == "configurations"
    }
    referenced_bundle_paths = {path for path, _ in {*runtime_bundles, *model_bundles}}
    actual_bundle_paths = {
        PurePosixPath(*PurePosixPath(relative).parts[:3]).as_posix()
        for relative in release_files
        if PurePosixPath(relative).parts[0] == "source-bundles"
    }
    if actual_configuration_paths != configuration_paths:
        raise A10ExportError("release configuration file coverage differs")
    if actual_bundle_paths != referenced_bundle_paths:
        raise A10ExportError("release contains an unreferenced source bundle")
    return indexes


def verify_release(archive: str | Path, *, reconstruct: bool = True) -> Mapping[str, Any]:
    archive = Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix="perfseer-release-verify-") as directory:
        root = Path(directory)
        _safe_extract(archive, root)
        release = root / "release"
        indexes = _verify_release_tree(release)
        if reconstruct:
            runtime = release / indexes[0]["runtime_source_bundle_path"] / "src"
            script = """
import gc, importlib, json, pathlib, sys
runtime, release = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(runtime))
for line in (release/'candidate-index.jsonl').read_text().splitlines():
 row=json.loads(line); cfg=json.loads((release/row['configuration_path']).read_text())
 module_name, callable_name=cfg['factory_entrypoint'].split(':',1)
 factory=getattr(importlib.import_module(module_name), callable_name)
 candidate=cfg['resolved_candidate']
 model=factory(output_width=cfg['target_width'],task_kind=cfg['task_kind'],seed=int(candidate['seed_policy']['seed']),architecture_parameters=cfg['architecture'])
 if getattr(model,'family_id',None) != cfg['family_id']:
  raise RuntimeError('factory reconstructed another family')
 del model; gc.collect()
"""
            try:
                subprocess.run(
                    [sys.executable, "-I", "-c", script, str(runtime), str(release)],
                    check=True,
                    timeout=max(300, len(indexes) * 3),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise A10ExportError("release model reconstruction failed") from error
    return canonical_value(
        {
            "status": "passed",
            "archive_sha256": file_sha256(archive),
            "label_count": len(indexes),
            "reconstructed_configuration_count": len(indexes) if reconstruct else 0,
        }
    )


__all__ = [
    "A10ExportError",
    "EXPORT_VERSION",
    "export_release",
    "verify_release",
]
