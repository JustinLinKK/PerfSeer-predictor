"""Execution-sensitive architecture-field binding for every model lineage."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..fingerprints import canonical_sha256, canonical_value


class ArchitectureBindingError(ValueError):
    """Raised when a factory ignores or invents an adjustable architecture field."""


_MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "a10g_model_registry.yaml"
_DEFAULTS: Mapping[str, Any] = {
    "activation": "elu",
    "architecture_specification": "residual_conv_gelu_v1",
    "atom_features": 6,
    "aux_logits": False,
    "base_channels": 8,
    "capacity_factor": 1.0,
    "channel_dim": 8,
    "channels": 8,
    "clip_samples": 256,
    "components": 3,
    "decoder_layers": 1,
    "depth": 2,
    "depth_multiplier": 0.5,
    "depths": [1, 1],
    "discriminator_width": 8,
    "embedding_dim": 8,
    "encoder_layers": 1,
    "expert_count": 3,
    "feature_count": 8,
    "generator_width": 8,
    "heads": 2,
    "hidden_size": 8,
    "input_resolution": 16,
    "kernel_size": 3,
    "kv_heads": 1,
    "latent_rank": 4,
    "layer_count": 1,
    "layers": 1,
    "levels": 2,
    "localization_width": 4,
    "mel_bins": 16,
    "message_layers": 1,
    "modality": "vision",
    "neighbor_count": 4,
    "neighbor_limit": 8,
    "ngram_buckets": 64,
    "node_count": 8,
    "patch_size": 4,
    "sample_rate": 16_000,
    "sequence_length": 8,
    "source_lineage": "generated_golden_000",
    "sparse_gradients": False,
    "stage_depths": [1, 1],
    "state_size": 8,
    "student_layers": 1,
    "tag_count": 3,
    "temperature": 2.0,
    "token_dim": 8,
    "top_k": 1,
    "variant": "b0",
    "width": 8,
    "width_multiplier": 0.25,
    "window_size": 2,
}


@lru_cache(maxsize=None)
def declared_architecture_fields(family_id: str) -> tuple[str, ...]:
    root = yaml.safe_load(_MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    try:
        row = next(entry for entry in root["entries"] if entry["family_id"] == family_id)
    except StopIteration as error:
        raise ArchitectureBindingError(f"unknown architecture family {family_id!r}") from error
    fields = tuple(row["fields"])
    if not fields or len(set(fields)) != len(fields):
        raise ArchitectureBindingError("architecture field declaration is empty or duplicated")
    return fields


def resolve_architecture_parameters(
    family_id: str,
    provided: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields = declared_architecture_fields(family_id)
    provided = dict(provided or {})
    unknown = set(provided) - set(fields)
    if unknown:
        raise ArchitectureBindingError(
            f"family {family_id} received undeclared architecture fields {sorted(unknown)}"
        )
    missing_defaults = set(fields) - set(_DEFAULTS)
    if missing_defaults:
        raise ArchitectureBindingError(
            f"architecture defaults are missing for {sorted(missing_defaults)}"
        )
    result = {field: provided.get(field, _DEFAULTS[field]) for field in fields}
    canonical_value(result)
    return result


def _effect_kind(field: str) -> str:
    if field in {
        "input_resolution", "patch_size", "window_size", "sequence_length",
        "clip_samples", "sample_rate", "mel_bins", "feature_count", "node_count",
        "neighbor_count", "neighbor_limit", "atom_features", "kernel_size",
    }:
        return "input_or_shape_contract"
    if any(token in field for token in ("depth", "layers", "levels", "count")):
        return "structural_depth_or_cardinality"
    if any(token in field for token in ("width", "hidden", "channel", "embedding", "rank", "heads")):
        return "structural_width_or_projection"
    return "execution_branch_or_policy"


def build_semantic_field_evidence(
    family_id: str,
    parameters: Mapping[str, Any],
    semantic_roles: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    fields = declared_architecture_fields(family_id)
    if set(parameters) != set(fields) or set(semantic_roles) != set(fields):
        raise ArchitectureBindingError(
            f"family {family_id} did not semantically consume every declared field"
        )
    evidence = {}
    for field in sorted(fields):
        role = semantic_roles[field]
        if type(role) is not str or not role:
            raise ArchitectureBindingError(
                f"family {family_id} has no semantic role for {field}"
            )
        evidence[field] = {
            "field_name": field,
            "value_sha256": canonical_sha256(parameters[field]),
            "effect_kind": _effect_kind(field),
            "semantic_role": role,
        }
    return evidence


__all__ = [
    "ArchitectureBindingError",
    "build_semantic_field_evidence",
    "declared_architecture_fields",
    "resolve_architecture_parameters",
]
