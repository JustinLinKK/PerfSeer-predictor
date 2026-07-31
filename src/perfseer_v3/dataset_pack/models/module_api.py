"""Shared public signature used by each lineage-specific factory module."""

from __future__ import annotations

from typing import Any, Mapping

from .base import FamilyBuildConfig, FamilyModel
from .factory import build_family_model


def build(
    family_id: str,
    *,
    output_width: int,
    task_kind: str,
    seed: int = 0,
    architecture_parameters: Mapping[str, Any] | None = None,
) -> FamilyModel:
    return build_family_model(
        family_id,
        FamilyBuildConfig(
            output_width=output_width,
            task_kind=task_kind,
            seed=seed,
            architecture_parameters=architecture_parameters,
        ),
    )


__all__ = ["build"]
