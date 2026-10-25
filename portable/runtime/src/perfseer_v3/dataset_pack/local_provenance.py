"""Immutable source and environment identities for the local RTX gate."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
from typing import Any

import torch

from .fingerprints import canonical_sha256, file_sha256


LOCAL_VALIDATION_HARNESS_VERSION = "perfseer_v3_local_validation_harness_v1"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _harness_paths() -> tuple[Path, ...]:
    root = _repository_root()
    package = root / "src/perfseer_v3/dataset_pack"
    # Hash the full executable dataset-pack source closure. Over-invalidation is
    # preferable to reusing a pass after a helper/registry/contract behavior
    # changes, and including this module binds the provenance algorithm itself.
    return (
        *tuple(sorted(package.rglob("*.py"))),
        root / "scripts/validate_v100_18k_local.py",
    )


def validation_harness_sha256() -> str:
    root = _repository_root()
    files = _harness_paths()
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("local validation harness source set is incomplete")
    return canonical_sha256(
        {
            "version": LOCAL_VALIDATION_HARNESS_VERSION,
            "files": {
                path.relative_to(root).as_posix(): file_sha256(path)
                for path in files
            },
        }
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        value: Any = pynvml.nvmlSystemGetDriverVersion()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    except Exception:
        return None


def validation_environment_sha256(
    device: str | torch.device,
    compile_backend: str,
) -> str:
    resolved = torch.device(device)
    device_payload: dict[str, Any] = {"type": resolved.type}
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA validation environment is unavailable")
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_payload.update(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "driver_version": _driver_version(),
            }
        )
    return canonical_sha256(
        {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "compile_backend": compile_backend,
            "device": device_payload,
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "pillow", "pynvml", "torch-geometric", "triton")
            },
        }
    )


__all__ = [
    "LOCAL_VALIDATION_HARNESS_VERSION",
    "validation_environment_sha256",
    "validation_harness_sha256",
]
