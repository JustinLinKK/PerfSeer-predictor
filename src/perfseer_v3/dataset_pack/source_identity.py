"""Deterministic identity of the source closure copied into the labeler image."""

from __future__ import annotations

from pathlib import Path

from .fingerprints import canonical_sha256, file_sha256
from .labeler_profile import PROFILE


V100_SOURCE_ROOTS = (
    "pyproject.toml",
    "scripts/run_perfseer_v3_v100_labeling.py",
    "src/perfseer_v3",
    "containers/v100-labeler/Dockerfile",
    "containers/v100-labeler/requirements.in",
    "containers/v100-labeler/requirements.lock",
)
A10_SOURCE_ROOTS = (
    "pyproject.toml",
    "scripts/run_perfseer_v3_a10_labeling.py",
    "src/perfseer_v3",
    "containers/a10-labeler/Dockerfile",
    "containers/a10-labeler/requirements.in",
    "containers/a10-labeler/requirements.lock",
)
SOURCE_ROOTS = A10_SOURCE_ROOTS if PROFILE.name == "native_a10" else V100_SOURCE_ROOTS


def _is_source_file(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def source_tree_sha256(repository_root: str | Path) -> str:
    root = Path(repository_root).resolve()
    files: list[Path] = []
    for relative in SOURCE_ROOTS:
        path = root / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in sorted(path.rglob("*")) if _is_source_file(item))
        else:
            raise RuntimeError(f"image source path is missing: {relative}")
    if len(files) != len(set(files)):
        raise RuntimeError("image source closure contains duplicate paths")
    return canonical_sha256(
        {path.relative_to(root).as_posix(): file_sha256(path) for path in files}
    )


__all__ = ["A10_SOURCE_ROOTS", "SOURCE_ROOTS", "V100_SOURCE_ROOTS", "source_tree_sha256"]
