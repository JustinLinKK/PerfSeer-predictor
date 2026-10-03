"""Reproducible metadata snapshot for the v2 comparison baseline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


_SAMPLED_FILE_BYTES = 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, sample_large_files: bool = True) -> str:
    """Hash a file, sampling both ends for large dataset archives.

    The digest is prefixed so a sampled fingerprint can never be mistaken for a
    full content hash. Checkpoints/configs always use full hashes.
    """

    size = path.stat().st_size
    if sample_large_files and size > 2 * _SAMPLED_FILE_BYTES:
        with path.open("rb") as handle:
            first = handle.read(_SAMPLED_FILE_BYTES)
            handle.seek(-_SAMPLED_FILE_BYTES, 2)
            last = handle.read(_SAMPLED_FILE_BYTES)
        payload = b"sampled-sha256-v1\0" + str(size).encode("ascii") + b"\0" + first + last
        return "sampled:" + sha256_bytes(payload)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_SAMPLED_FILE_BYTES), b""):
            digest.update(block)
    return "full:" + digest.hexdigest()


def tree_fingerprint(
    root: Path,
    *,
    suffixes: Iterable[str] | None = None,
    sample_large_files: bool = True,
) -> dict[str, Any]:
    suffix_set = {suffix.lower() for suffix in suffixes} if suffixes else None
    if not root.exists():
        return {"status": "missing", "algorithm": "sha256-tree-v1", "sha256": None, "files": 0}
    records: list[dict[str, Any]] = []
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        if suffix_set is not None and path.suffix.lower() not in suffix_set:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "digest": sha256_file(path, sample_large_files=sample_large_files),
            }
        )
    payload = canonical_json(records).encode("utf-8")
    return {
        "status": "ok",
        "algorithm": "sha256-tree-v1",
        "sha256": sha256_bytes(payload),
        "files": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "large_file_policy": "first_and_last_1MiB" if sample_large_files else "full",
    }


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class BaselineSnapshot:
    baseline_version: str
    commit_sha: str
    branch: str
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    config_fingerprint: dict[str, Any]
    dataset_fingerprint: dict[str, Any]
    checkpoint_fingerprint: dict[str, Any]
    evaluation_fingerprint: dict[str, Any]
    v2_feature_schema_version: str = "perfseer_graph_v1"
    v2_node_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["snapshot_sha256"] = sha256_bytes(canonical_json(data).encode("utf-8"))
        return data


def collect_v2_baseline(repo_root: str | Path) -> BaselineSnapshot:
    from perfseer.architecture_schema import (
        FEATURE_SCHEMA_VERSION as V2_FEATURE_SCHEMA_VERSION,
        NODE_TYPES as V2_NODE_TYPES,
    )

    root = Path(repo_root).resolve()
    return BaselineSnapshot(
        baseline_version="perfseer_v2_baseline_v1",
        commit_sha=_git(root, "rev-parse", "HEAD"),
        branch=_git(root, "branch", "--show-current"),
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        config_fingerprint=tree_fingerprint(
            root / "src" / "perfseer-optimized" / "configs",
            suffixes={".yaml", ".yml", ".json"},
            sample_large_files=False,
        ),
        dataset_fingerprint=tree_fingerprint(root / "dataset", sample_large_files=True),
        checkpoint_fingerprint=tree_fingerprint(
            root / "models",
            suffixes={".pt", ".pth", ".ckpt", ".onnx", ".json"},
            sample_large_files=False,
        ),
        evaluation_fingerprint=tree_fingerprint(
            root / "runs",
            suffixes={".json", ".jsonl", ".yaml", ".txt", ".csv"},
            sample_large_files=False,
        ),
        v2_feature_schema_version=V2_FEATURE_SCHEMA_VERSION,
        v2_node_types=V2_NODE_TYPES,
    )


def write_baseline(snapshot: BaselineSnapshot, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
