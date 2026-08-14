"""Verified, weight-free release export for the non-vision A10 corpus."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
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


class A10ExportError(RuntimeError):
    """Raised when a release is incomplete, mutable, or not reconstructable."""


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
        members = bundle.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise A10ExportError("release archive contains an unsafe member")
        extraction_root = destination / "release"
        for member in members:
            target = extraction_root / member.name
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


def verify_release(archive: str | Path, *, reconstruct: bool = True) -> Mapping[str, Any]:
    archive = Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix="perfseer-release-verify-") as directory:
        root = Path(directory)
        _safe_extract(archive, root)
        release = root / "release"
        sums = (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        for row in sums:
            digest, relative = row.split("  ", 1)
            path = release / relative
            if not path.is_file() or file_sha256(path) != digest:
                raise A10ExportError(f"release checksum differs for {relative}")
        manifest = json.loads((release / "release-manifest.json").read_text())
        labels = tuple(
            json.loads(line)
            for line in (release / "labels.jsonl").read_text().splitlines()
            if line.strip()
        )
        indexes = tuple(
            json.loads(line)
            for line in (release / "candidate-index.jsonl").read_text().splitlines()
            if line.strip()
        )
        if len(labels) != len(indexes) or len(labels) != manifest["label_count"]:
            raise A10ExportError("release label/index counts differ")
        for raw_label, index in zip(labels, indexes, strict=True):
            label = label_run_record_from_dict(raw_label)
            label.validate()
            configuration_path = release / index["configuration_path"]
            configuration = json.loads(configuration_path.read_text())
            if (
                canonical_sha256(configuration) != index["configuration_sha256"]
                or label.configuration_id != configuration["resolved_candidate_id"]
                or target_candidate_from_dict(configuration["resolved_candidate"]).candidate_id
                != label.configuration_id
            ):
                raise A10ExportError("release label/configuration/index join differs")
            for key in ("runtime_source_bundle_path", "model_source_bundle_path"):
                bundle = release / index[key]
                source_manifest = json.loads(
                    (bundle / "bundle-manifest.json").read_text()
                )
                digest_key = key.replace("_path", "_sha256")
                if source_manifest["bundle_sha256"] != index[digest_key]:
                    raise A10ExportError("candidate source bundle identity differs")
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
            subprocess.run(
                [sys.executable, "-I", "-c", script, str(runtime), str(release)],
                check=True,
                timeout=max(300, len(indexes) * 3),
            )
    return canonical_value(
        {
            "status": "passed",
            "archive_sha256": file_sha256(archive),
            "label_count": len(labels),
            "reconstructed_configuration_count": len(indexes) if reconstruct else 0,
        }
    )


__all__ = [
    "A10ExportError",
    "EXPORT_VERSION",
    "export_release",
    "verify_release",
]
