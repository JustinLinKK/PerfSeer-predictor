"""Canonical content fingerprints shared by every dataset-pack artifact."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping



def canonical_json(value: Any) -> str:
    """Serialize canonical values without importing the torch-heavy baseline."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_value(value: Any) -> Any:
    """Convert supported typed values into canonical JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return canonical_value(asdict(value))
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical fingerprint mappings require string keys")
        return {
            key: canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical fingerprints reject non-finite floats")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported canonical fingerprint value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(canonical_value(value)).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, block_bytes: int = 1024 * 1024) -> str:
    """Return a full content hash; dataset revisions never use sampled hashes."""

    if block_bytes < 1:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def joined_sha256(**fingerprints: str) -> str:
    if not fingerprints:
        raise ValueError("at least one fingerprint is required")
    for name, digest in fingerprints.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return canonical_sha256(fingerprints)


__all__ = [
    "canonical_bytes",
    "canonical_sha256",
    "canonical_value",
    "file_sha256",
    "joined_sha256",
]
