"""Canonical per-GPU model-pair identity helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping


_NON_SPECIFIC_IDS = frozenset({"", "unknown", "any", "all", "mixed", "generic", "*"})


def canonical_hardware_id(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "unknown"


def require_specific_hardware_id(value: Any, *, context: str) -> str:
    hardware_id = canonical_hardware_id(value)
    if hardware_id in _NON_SPECIFIC_IDS:
        raise ValueError(
            f"{context} must identify exactly one concrete GPU type; got {value!r}"
        )
    return hardware_id


def graph_hardware_id(metadata: Mapping[str, Any]) -> str:
    return canonical_hardware_id(
        metadata.get("target_hardware_id", metadata.get("hardware_id", "unknown"))
    )


__all__ = [
    "canonical_hardware_id",
    "graph_hardware_id",
    "require_specific_hardware_id",
]
