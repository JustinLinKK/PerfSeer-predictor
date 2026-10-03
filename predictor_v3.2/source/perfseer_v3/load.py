"""Trusted local source loading for the v3 capture entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch.nn as nn


def source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_source(path: str | Path) -> ModuleType:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    module_name = f"_perfseer_v3_source_{source_sha256(resolved)[:20]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_model_entry(
    source_path: str | Path,
    entry: str,
    *,
    constructor_args: tuple[Any, ...] = (),
    constructor_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    module = import_source(source_path)
    target: Any = module
    for component in entry.split("."):
        if not component:
            raise AttributeError(f"source entry {entry!r} contains an empty component")
        if not hasattr(target, component):
            raise AttributeError(f"{source_path} does not define {entry!r}")
        target = getattr(target, component)
    if isinstance(target, nn.Module):
        value = target
    elif isinstance(target, type) and issubclass(target, nn.Module):
        value = target(*constructor_args, **(constructor_kwargs or {}))
    elif callable(target):
        value = target(*constructor_args, **(constructor_kwargs or {}))
    else:
        value = target
    if not isinstance(value, nn.Module):
        raise TypeError(f"source entry {entry!r} did not produce torch.nn.Module")
    return value


__all__ = ["import_source", "load_model_entry", "source_sha256"]
