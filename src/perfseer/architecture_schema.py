"""Shared architecture schema for PerfSeer graph features and catalogs."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


FEATURE_SCHEMA_VERSION = "perfseer_graph_v1"

NODE_TYPES: tuple[str, ...] = (
    "Conv",
    "DepthwiseConv",
    "ConvTranspose",
    "BatchNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "Embedding",
    "Gemm",
    "MatMul",
    "Bmm",
    "Attention",
    "MultiHeadAttention",
    "RNN",
    "GRU",
    "LSTM",
    "GraphMessage",
    "GraphAttention",
    "Relu",
    "Gelu",
    "Silu",
    "Softmax",
    "Sigmoid",
    "Mul",
    "Add",
    "Concat",
    "Flatten",
    "Reshape",
    "Transpose",
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "Upsample",
    "DetectorHead",
    "SegmentationHead",
    "TabularFeature",
)

ARCHITECTURE_FAMILY_QUOTAS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("resnet_cnn", 2160),
        ("efficientnet_cnn", 1520),
        ("bert_encoder", 1200),
        ("vit_encoder", 1200),
        ("yolo_detector", 560),
        ("unet_encoder_decoder", 480),
        ("ast_audio_transformer", 480),
        ("gru_temporal", 480),
        ("lstm_temporal", 320),
        ("vgg_cnn", 400),
        ("gat_graph", 240),
        ("mpnn_graph", 240),
        ("t5_encoder_decoder", 240),
        ("wav2vec2_audio", 240),
        ("ft_transformer_tabular", 240),
    ]
)

ARCHITECTURE_FAMILIES: tuple[str, ...] = tuple(ARCHITECTURE_FAMILY_QUOTAS)
MODALITIES: tuple[str, ...] = ("image", "text", "audio", "temporal", "graph", "tabular")
VARIANT_KINDS: tuple[str, ...] = (
    "canonical",
    "added_depth",
    "dropped_depth",
    "width_shape",
    "mixed_stress",
)
VARIANT_MIX: tuple[tuple[str, float], ...] = (
    ("canonical", 0.10),
    ("added_depth", 0.30),
    ("dropped_depth", 0.30),
    ("width_shape", 0.20),
    ("mixed_stress", 0.10),
)


def node_types_for_schema(feature_schema_version: str | None = None) -> tuple[str, ...]:
    """Return the canonical operator vocabulary for this branch."""

    version = feature_schema_version or FEATURE_SCHEMA_VERSION
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema {version!r}; expected {FEATURE_SCHEMA_VERSION!r}")
    return NODE_TYPES


def feature_schema_signature(feature_schema_version: str | None) -> str:
    """Stable digest used for cache/checkpoint compatibility checks."""

    version = feature_schema_version or FEATURE_SCHEMA_VERSION
    payload = {
        "feature_schema_version": version,
        "node_types": node_types_for_schema(version),
        "architecture_families": ARCHITECTURE_FAMILIES,
        "modalities": MODALITIES,
        "variant_kinds": VARIANT_KINDS,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def variant_counts_for_family(total: int) -> dict[str, int]:
    """Allocate a family quota across the fixed mutation mix."""

    counts: dict[str, int] = {}
    assigned = 0
    for idx, (kind, fraction) in enumerate(VARIANT_MIX):
        if idx == len(VARIANT_MIX) - 1:
            count = total - assigned
        else:
            count = int(round(total * fraction))
            assigned += count
        counts[kind] = count
    return counts
