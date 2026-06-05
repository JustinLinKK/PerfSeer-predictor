"""Config-driven data module for optimized PerfSeer experiments."""

from __future__ import annotations

import hashlib
import ast
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from perfseer.data import ARG_KEYS, NODE_TYPES, list_pairs, parse_graph, parse_label
from perfseer.data import _resolve_dirs as resolve_dataset_dirs


NUM_TARGETS = 6
TARGET_NAMES: list[str] = [
    "train_util",
    "train_mem",
    "train_time",
    "infer_util",
    "infer_mem",
    "infer_time",
]

SOURCE_UNKNOWN_PRECISION_CONFIG = "source_domain_unknown"
DTYPE_VOCAB = ("fp32", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "unknown")
TENSORCORE_MODES = ("none", "tf32", "bf16", "fp16", "fp8", "unknown")
FP8_FORMATS = ("none", "e4m3", "e5m2", "hybrid_e4m3_e5m2", "unknown")
PRECISION_CONFIG_VOCAB = (
    "fp32_ieee",
    "tf32",
    "bf16_amp",
    "fp16_amp",
    "fp8_te_hybrid",
    "fp8_e4m3",
    "fp8_e5m2",
    SOURCE_UNKNOWN_PRECISION_CONFIG,
)
RESOURCE_REGIME_VOCAB = ("small_overhead", "memory_bound", "balanced", "compute_bound")
LABEL_DOMAIN_VOCAB = ("unknown", "source", "precision_profile", "pseudo")
HARDWARE_FEATURE_FIELDS = (
    "compute_capability",
    "architecture_id",
    "sm_count",
    "memory_bandwidth_gbps",
    "vram_gib",
    "l2_cache_mib",
    "peak_fp32_tflops",
    "peak_tf32_tflops",
    "peak_fp16_bf16_tflops",
    "peak_fp8_tflops",
)
DTYPE_BYTES = {
    "unknown": 0.0,
    "fp32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
}
PRECISION_PRESETS: dict[str, dict[str, str]] = {
    SOURCE_UNKNOWN_PRECISION_CONFIG: {
        "weight_dtype": "unknown",
        "activation_dtype": "unknown",
        "grad_dtype": "unknown",
        "accum_dtype": "unknown",
        "optimizer_state_dtype": "unknown",
        "tensorcore_mode": "unknown",
        "fp8_format": "unknown",
    },
    "fp32_ieee": {
        "weight_dtype": "fp32",
        "activation_dtype": "fp32",
        "grad_dtype": "fp32",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "none",
        "fp8_format": "none",
    },
    "tf32": {
        "weight_dtype": "fp32",
        "activation_dtype": "fp32",
        "grad_dtype": "fp32",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "tf32",
        "fp8_format": "none",
    },
    "bf16_amp": {
        "weight_dtype": "fp32",
        "activation_dtype": "bf16",
        "grad_dtype": "bf16",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "bf16",
        "fp8_format": "none",
    },
    "fp16_amp": {
        "weight_dtype": "fp32",
        "activation_dtype": "fp16",
        "grad_dtype": "fp16",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "fp16",
        "fp8_format": "none",
    },
    "fp8_te_hybrid": {
        "weight_dtype": "fp8_e4m3",
        "activation_dtype": "fp8_e4m3",
        "grad_dtype": "fp8_e5m2",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "fp8",
        "fp8_format": "hybrid_e4m3_e5m2",
    },
    "fp8_e4m3": {
        "weight_dtype": "fp8_e4m3",
        "activation_dtype": "fp8_e4m3",
        "grad_dtype": "fp8_e4m3",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "fp8",
        "fp8_format": "e4m3",
    },
    "fp8_e5m2": {
        "weight_dtype": "fp8_e5m2",
        "activation_dtype": "fp8_e5m2",
        "grad_dtype": "fp8_e5m2",
        "accum_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "tensorcore_mode": "fp8",
        "fp8_format": "e5m2",
    },
}


class PerfSeerOptimizedData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key == "u":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


try:
    torch.serialization.add_safe_globals([PerfSeerOptimizedData])
except Exception:
    pass


@dataclass(frozen=True)
class FeatureConfig:
    use_operator_type_onehot: bool = True
    topology: bool = False
    critical_path: bool = False
    edge_topology: bool = False
    include_destination_tensor: bool = False
    time_target_mode: str = "raw"
    target_mode: str = "absolute"
    include_precision_features: bool = False
    include_hardware_features: bool = False
    precision_config: str = "fp32_ieee"
    hardware_id: str = "unknown"
    weight_dtype: str = ""
    activation_dtype: str = ""
    grad_dtype: str = ""
    accum_dtype: str = ""
    optimizer_state_dtype: str = ""
    tensorcore_mode: str = ""
    fp8_format: str = ""
    compute_capability: float = 0.0
    architecture_id: float = 0.0
    sm_count: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    vram_gib: float = 0.0
    l2_cache_mib: float = 0.0
    peak_fp32_tflops: float = 0.0
    peak_tf32_tflops: float = 0.0
    peak_fp16_bf16_tflops: float = 0.0
    peak_fp8_tflops: float = 0.0
    base_label_weight: float = 1.0
    precision_label_weight: float = 1.0
    pseudo_label_weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureConfig":
        if data is None:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FeatureLayout:
    node_dim: int
    edge_dim: int
    global_dim: int
    node_std_idx: tuple[int, ...]
    edge_std_idx: tuple[int, ...]
    global_std_idx: tuple[int, ...]
    node_names: tuple[str, ...]
    edge_names: tuple[str, ...]
    global_names: tuple[str, ...]


def feature_layout(cfg: FeatureConfig) -> FeatureLayout:
    node_names: list[str] = []
    node_std: list[int] = []

    def add_node(name: str, standardize: bool) -> None:
        idx = len(node_names)
        node_names.append(name)
        if standardize:
            node_std.append(idx)

    if cfg.use_operator_type_onehot:
        for typ in NODE_TYPES:
            add_node(f"type_{typ}", False)
    for key in ARG_KEYS:
        add_node(f"arg_{key}", True)
    for name in ["log_flops", "log_mac_bytes", "log_weight_size", "arith_intensity"]:
        add_node(name, True)
    for name in ["flops_ratio", "mac_ratio", "weight_ratio"]:
        add_node(name, False)
    if cfg.topology:
        for name in ["in_degree", "out_degree", "topo_index", "forward_depth", "reverse_depth"]:
            add_node(name, True)
        for name in ["is_source", "is_sink", "is_branch", "is_join"]:
            add_node(name, False)
    if cfg.critical_path:
        for name in [
            "longest_depth_from_input",
            "longest_depth_to_output",
            "flop_weighted_depth_from_input",
            "flop_weighted_depth_to_output",
        ]:
            add_node(name, True)
        for name in ["is_on_unweighted_longest_path", "is_on_flop_weighted_longest_path"]:
            add_node(name, False)

    edge_names: list[str] = []
    edge_std: list[int] = []

    def add_edge(name: str, standardize: bool = True) -> None:
        idx = len(edge_names)
        edge_names.append(name)
        if standardize:
            edge_std.append(idx)

    for name in ["edge_tensor_bytes_log1p", "src_batch_log1p", "src_channels_log1p", "src_height_log1p", "src_width_log1p"]:
        add_edge(name, True)
    if cfg.edge_topology:
        for name in ["source_out_degree", "target_in_degree", "source_depth", "target_depth", "depth_delta"]:
            add_edge(name, True)
        add_edge("is_skip_like_edge", False)
    if cfg.include_destination_tensor:
        for name in [
            "destination_input_size_log1p",
            "destination_batch_log1p",
            "destination_channels_log1p",
            "destination_height_log1p",
            "destination_width_log1p",
        ]:
            add_edge(name, True)

    global_names: list[str] = []
    global_std: list[int] = []

    def add_global(name: str, standardize: bool) -> None:
        idx = len(global_names)
        global_names.append(name)
        if standardize:
            global_std.append(idx)

    for name, std in [
        ("num_nodes", True),
        ("num_edges", True),
        ("density", False),
        ("total_flops_log1p", True),
        ("mean_flops_log1p", True),
        ("median_flops_log1p", True),
        ("max_flops_log1p", True),
        ("total_mac_log1p", True),
        ("mean_mac_log1p", True),
        ("median_mac_log1p", True),
        ("max_mac_log1p", True),
        ("total_weight_log1p", True),
        ("mean_weight_log1p", True),
        ("median_weight_log1p", True),
        ("max_weight_log1p", True),
        ("mean_edge_tensor_log1p", True),
        ("model_arith_intensity", True),
        ("batch_size", True),
    ]:
        add_global(name, std)
    if cfg.topology:
        for name in ["max_topological_depth", "mean_topological_depth", "num_branch_nodes", "num_join_nodes", "branch_join_ratio"]:
            add_global(name, True)
    if cfg.critical_path:
        for name in ["max_flop_weighted_path", "critical_path_flops_ratio", "num_nodes_on_longest_path", "num_nodes_on_flop_path"]:
            add_global(name, True)
    if cfg.include_precision_features:
        for field in ("weight_dtype", "activation_dtype", "grad_dtype", "accum_dtype", "optimizer_state_dtype"):
            for dtype in DTYPE_VOCAB:
                add_global(f"{field}_{dtype}", False)
        for mode in TENSORCORE_MODES:
            add_global(f"tensorcore_mode_{mode}", False)
        for fp8_format in FP8_FORMATS:
            add_global(f"fp8_format_{fp8_format}", False)
        for name in [
            "estimated_activation_bytes_log1p",
            "estimated_weight_bytes_log1p",
            "estimated_grad_bytes_log1p",
            "estimated_optimizer_state_bytes_log1p",
            "estimated_master_weight_bytes_log1p",
        ]:
            add_global(name, True)
    if cfg.include_hardware_features:
        for name in [
            "hardware_compute_capability",
            "hardware_architecture_id",
            "hardware_sm_count",
            "hardware_memory_bandwidth_gbps",
            "hardware_vram_gib",
            "hardware_l2_cache_mib",
            "hardware_peak_fp32_tflops",
            "hardware_peak_tf32_tflops",
            "hardware_peak_fp16_bf16_tflops",
            "hardware_peak_fp8_tflops",
        ]:
            add_global(name, True)

    return FeatureLayout(
        node_dim=len(node_names),
        edge_dim=len(edge_names),
        global_dim=len(global_names),
        node_std_idx=tuple(node_std),
        edge_std_idx=tuple(edge_std),
        global_std_idx=tuple(global_std),
        node_names=tuple(node_names),
        edge_names=tuple(edge_names),
        global_names=tuple(global_names),
    )


DEFAULT_LAYOUT = feature_layout(FeatureConfig())
NODE_DIM = DEFAULT_LAYOUT.node_dim
EDGE_DIM = DEFAULT_LAYOUT.edge_dim
GLOBAL_DIM = DEFAULT_LAYOUT.global_dim


def _f(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v if np.isfinite(v) else default
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return 0.0
    out = num / den
    return out if np.isfinite(out) else 0.0


def _stats4(arr: np.ndarray) -> tuple[float, float, float, float]:
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.sum(arr)), float(np.mean(arr)), float(np.median(arr)), float(np.max(arr))


def _node_type_onehot(type_str: str) -> list[float]:
    vec = [0.0] * len(NODE_TYPES)
    if type_str in NODE_TYPES:
        vec[NODE_TYPES.index(type_str)] = 1.0
    return vec


def _onehot(value: str, vocab: Sequence[str]) -> list[float]:
    key = (value or "").lower()
    return [1.0 if key == item else 0.0 for item in vocab]


def _precision_settings(cfg: FeatureConfig) -> dict[str, str]:
    preset = PRECISION_PRESETS.get((cfg.precision_config or "fp32_ieee").lower(), PRECISION_PRESETS["fp32_ieee"])
    settings = dict(preset)
    for field in (
        "weight_dtype",
        "activation_dtype",
        "grad_dtype",
        "accum_dtype",
        "optimizer_state_dtype",
        "tensorcore_mode",
        "fp8_format",
    ):
        value = getattr(cfg, field)
        if value:
            settings[field] = str(value).lower()
    return settings


def precision_hardware_config(cfg: FeatureConfig) -> dict[str, Any]:
    fields = (
        "include_precision_features",
        "include_hardware_features",
        "precision_config",
        "hardware_id",
        "weight_dtype",
        "activation_dtype",
        "grad_dtype",
        "accum_dtype",
        "optimizer_state_dtype",
        "tensorcore_mode",
        "fp8_format",
        "compute_capability",
        "architecture_id",
        "sm_count",
        "memory_bandwidth_gbps",
        "vram_gib",
        "l2_cache_mib",
        "peak_fp32_tflops",
        "peak_tf32_tflops",
        "peak_fp16_bf16_tflops",
        "peak_fp8_tflops",
    )
    out = {field: getattr(cfg, field) for field in fields}
    out["resolved_precision"] = _precision_settings(cfg)
    return out


def normalize_precision_config(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    aliases = {
        "fp32": "fp32_ieee",
        "float32": "fp32_ieee",
        "fp32_ieee": "fp32_ieee",
        "tf32": "tf32",
        "bf16": "bf16_amp",
        "bf16_amp": "bf16_amp",
        "fp16": "fp16_amp",
        "float16": "fp16_amp",
        "fp16_amp": "fp16_amp",
        "fp8": "fp8_te_hybrid",
        "fp8_te": "fp8_te_hybrid",
        "fp8_te_hybrid": "fp8_te_hybrid",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "source_unknown": SOURCE_UNKNOWN_PRECISION_CONFIG,
        "source_domain_unknown": SOURCE_UNKNOWN_PRECISION_CONFIG,
    }
    if key == "bf32":
        raise ValueError("bf32 is ambiguous; use tf32 or bf16_amp")
    return aliases.get(key, key)


def precision_config_index(value: str) -> int:
    normalized = normalize_precision_config(value)
    try:
        return PRECISION_CONFIG_VOCAB.index(normalized)
    except ValueError:
        return -1


def normalize_label_domain(value: str | None) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key in {"source", "base", "base_label", "source_domain"}:
        return "source"
    if key in {"precision", "precision_profile", "profile", "measured", "golden"}:
        return "precision_profile"
    if key in {"pseudo", "teacher", "teacher_soft", "soft"}:
        return "pseudo"
    return "unknown"


def label_domain_index(value: str | None) -> int:
    normalized = normalize_label_domain(value)
    try:
        return LABEL_DOMAIN_VOCAB.index(normalized)
    except ValueError:
        return 0


def _metadata_candidate_paths(label_path: str) -> tuple[str, ...]:
    path = os.path.abspath(label_path)
    label_dir = os.path.dirname(path)
    label_parent = os.path.dirname(label_dir)
    data_root = os.path.dirname(label_parent)
    return (
        os.path.join(label_dir, "precision_metadata.jsonl"),
        os.path.join(label_parent, "precision_metadata.jsonl"),
        os.path.join(data_root, "precision_metadata.jsonl"),
    )


@lru_cache(maxsize=32)
def _metadata_index(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    index: dict[str, dict[str, Any]] = {}
    with open(path, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            keys = {
                str(row.get("label_file", "")),
                os.path.basename(str(row.get("label_file", ""))),
                str(row.get("label_stem", "")),
            }
            for key in keys:
                if key:
                    index[key] = row
    return index


def precision_metadata_for_label(label_path: str) -> dict[str, Any] | None:
    basename = os.path.basename(label_path)
    stem = os.path.splitext(basename)[0]
    for path in _metadata_candidate_paths(label_path):
        index = _metadata_index(path)
        row = index.get(basename) or index.get(stem)
        if row is not None:
            return row
    return None


def target_mode_key(cfg: FeatureConfig | None) -> str:
    return str(getattr(cfg, "target_mode", "absolute") or "absolute").lower()


def is_log_ratio_target(cfg: FeatureConfig | None) -> bool:
    return target_mode_key(cfg) in {"log_ratio_to_source", "log_ratio", "ratio_delta"}


def precision_from_label_path(graph_path: str, label_path: str) -> str | None:
    metadata = precision_metadata_for_label(label_path)
    if metadata and metadata.get("precision_config"):
        return normalize_precision_config(str(metadata["precision_config"]))
    graph_stem = os.path.splitext(os.path.basename(graph_path))[0]
    label_stem = os.path.splitext(os.path.basename(label_path))[0]
    prefix = graph_stem + "_"
    if not label_stem.startswith(prefix):
        return None
    raw = label_stem[len(prefix) :]
    return normalize_precision_config(raw) if raw else None


def _dataset_root_from_label_path(label_path: str) -> str:
    label_dir = os.path.dirname(os.path.abspath(label_path))
    label_parent = os.path.dirname(label_dir)
    return os.path.dirname(label_parent)


def _resolve_dataset_relative(label_path: str, rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_dataset_root_from_label_path(label_path), rel_path)


def base_label_path_for_pair(graph_path: str, label_path: str) -> str | None:
    metadata = precision_metadata_for_label(label_path) or {}
    if metadata.get("is_base_label") or str(metadata.get("label_domain", "")).lower() == "source":
        return label_path
    if metadata.get("base_label_file"):
        return _resolve_dataset_relative(label_path, str(metadata["base_label_file"]))
    graph_stem = os.path.splitext(os.path.basename(graph_path))[0]
    candidate = os.path.join(os.path.dirname(os.path.abspath(label_path)), f"{graph_stem}.txt")
    return candidate if os.path.isfile(candidate) else None


def base_label_for_pair(cfg: FeatureConfig, graph_path: str, label_path: str) -> np.ndarray | None:
    base_path = base_label_path_for_pair(graph_path, label_path)
    if not base_path or not os.path.isfile(base_path):
        if is_log_ratio_target(cfg):
            raise FileNotFoundError(f"target_mode={cfg.target_mode!r} requires a base label for {label_path}")
        return None
    return parse_label(base_path)


def feature_config_for_pair(cfg: FeatureConfig, graph_path: str, label_path: str) -> FeatureConfig:
    metadata = precision_metadata_for_label(label_path) or {}
    precision = precision_from_label_path(graph_path, label_path)
    has_updates = bool(precision and precision != cfg.precision_config) or bool(metadata)
    if not has_updates:
        return cfg
    data = cfg.to_dict()
    if precision:
        data["precision_config"] = precision
    if metadata.get("hardware_id"):
        data["hardware_id"] = str(metadata["hardware_id"])
    hardware = metadata.get("hardware_features") if isinstance(metadata.get("hardware_features"), dict) else metadata
    for field in HARDWARE_FEATURE_FIELDS:
        if field in hardware:
            data[field] = _f(hardware.get(field))
    if "multi_processor_count" in hardware and "sm_count" not in hardware:
        data["sm_count"] = _f(hardware.get("multi_processor_count"))
    if "total_memory_mib" in hardware and "vram_gib" not in hardware:
        data["vram_gib"] = _f(hardware.get("total_memory_mib")) / 1024.0
    if "compute_capability" in hardware:
        data["compute_capability"] = _f(str(hardware.get("compute_capability")).replace(".", ""))
        try:
            data["compute_capability"] = float(str(hardware.get("compute_capability")))
        except (TypeError, ValueError):
            pass
    return FeatureConfig.from_dict(data)


def sample_weight_for_pair(cfg: FeatureConfig, graph_path: str, label_path: str) -> float:
    metadata = precision_metadata_for_label(label_path) or {}
    label_domain = normalize_label_domain(str(metadata.get("label_domain", "")))
    if metadata.get("is_base_label") or label_domain == "source":
        return float(cfg.base_label_weight)
    if label_domain == "pseudo":
        return float(cfg.pseudo_label_weight)
    if precision_from_label_path(graph_path, label_path):
        return float(cfg.precision_label_weight)
    return float(cfg.base_label_weight)


def label_domain_for_pair(graph_path: str, label_path: str) -> str:
    metadata = precision_metadata_for_label(label_path) or {}
    domain = normalize_label_domain(str(metadata.get("label_domain", "")))
    if domain != "unknown":
        return domain
    if metadata.get("is_base_label"):
        return "source"
    if precision_from_label_path(graph_path, label_path):
        return "precision_profile"
    return "source"


def supported_precision_hardware_summary(pairs: Sequence[tuple[str, str]], cfg: FeatureConfig) -> dict[str, Any]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    label_domains: set[str] = set()
    for graph_path, label_path in pairs:
        pair_cfg = feature_config_for_pair(cfg, graph_path, label_path)
        precision_config = normalize_precision_config(pair_cfg.precision_config)
        hardware_id = str(pair_cfg.hardware_id or "unknown")
        label_domain = label_domain_for_pair(graph_path, label_path)
        label_domains.add(label_domain)
        key = (precision_config, hardware_id)
        record = records.setdefault(
            key,
            {
                "precision_config": precision_config,
                "hardware_id": hardware_id,
                "count": 0,
                "label_domains": set(),
            },
        )
        record["count"] += 1
        record["label_domains"].add(label_domain)

    pairs_out = []
    for record in sorted(records.values(), key=lambda item: (item["precision_config"], item["hardware_id"])):
        pairs_out.append(
            {
                "precision_config": record["precision_config"],
                "hardware_id": record["hardware_id"],
                "count": int(record["count"]),
                "label_domains": sorted(record["label_domains"]),
            }
        )
    return {
        "precision_configs": sorted({item["precision_config"] for item in pairs_out}),
        "hardware_ids": sorted({item["hardware_id"] for item in pairs_out}),
        "precision_hardware_pairs": pairs_out,
        "label_domains": sorted(label_domains),
    }


def validate_precision_hardware_request(
    cfg: FeatureConfig,
    supported: dict[str, Any] | None,
    *,
    context: str = "inference",
) -> None:
    if not supported:
        return
    precision_config = normalize_precision_config(cfg.precision_config)
    hardware_id = str(cfg.hardware_id or "unknown")
    supported_precisions = {str(value) for value in supported.get("precision_configs", [])}
    supported_hardware = {str(value) for value in supported.get("hardware_ids", [])}
    pair_records = supported.get("precision_hardware_pairs", []) or []
    supported_pairs = {
        (str(row.get("precision_config")), str(row.get("hardware_id")))
        for row in pair_records
        if isinstance(row, dict)
    }

    problems: list[str] = []
    if supported_pairs:
        if (precision_config, hardware_id) not in supported_pairs:
            allowed = ", ".join(f"{precision}/{hardware}" for precision, hardware in sorted(supported_pairs))
            problems.append(f"{precision_config}/{hardware_id} is not one of the trained precision/hardware pairs: {allowed}")
    else:
        if supported_precisions and precision_config not in supported_precisions:
            problems.append(f"precision_config={precision_config!r} is not in trained precisions {sorted(supported_precisions)}")
        if supported_hardware and hardware_id not in supported_hardware:
            problems.append(f"hardware_id={hardware_id!r} is not in trained hardware ids {sorted(supported_hardware)}")
    if problems:
        raise ValueError(f"unsupported precision/hardware request for {context}: " + "; ".join(problems))


def validate_precision_hardware_pairs(
    pairs: Sequence[tuple[str, str]],
    cfg: FeatureConfig,
    supported: dict[str, Any] | None,
    *,
    context: str = "dataset",
) -> None:
    if not supported:
        return
    for graph_path, label_path in pairs:
        pair_cfg = feature_config_for_pair(cfg, graph_path, label_path)
        validate_precision_hardware_request(
            pair_cfg,
            supported,
            context=f"{context} label {os.path.basename(label_path)}",
        )


def _metadata_rows_for_root(data_root: str) -> list[dict[str, Any]]:
    candidates = (
        os.path.join(data_root, "precision_metadata.jsonl"),
        os.path.join(data_root, "label", "precision_metadata.jsonl"),
        os.path.join(data_root, "label", "label", "precision_metadata.jsonl"),
    )
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen_paths or not os.path.isfile(abs_path):
            continue
        seen_paths.add(abs_path)
        with open(abs_path, "r") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def list_precision_pairs(data_root: str, include_base_pairs: bool = True) -> list[tuple[str, str]]:
    pairs = list_pairs(data_root) if include_base_pairs else []
    seen = {(os.path.abspath(gp), os.path.abspath(lp)) for gp, lp in pairs}
    cg_dir, label_dir = resolve_dataset_dirs(data_root)
    graph_by_stem = {
        os.path.splitext(fname)[0]: os.path.join(cg_dir, fname)
        for fname in os.listdir(cg_dir)
        if fname.endswith(".pkl")
    }
    precision_suffix = re.compile(r"^(?P<stem>.+)_(?P<precision>fp32_ieee|tf32|bf16_amp|fp16_amp|fp8_te_hybrid|fp8_e4m3|fp8_e5m2)\.txt$")
    for fname in sorted(os.listdir(label_dir)):
        match = precision_suffix.match(fname)
        if not match:
            continue
        graph_path = graph_by_stem.get(match.group("stem"))
        if graph_path is None:
            continue
        label_path = os.path.join(label_dir, fname)
        key = (os.path.abspath(graph_path), os.path.abspath(label_path))
        if key not in seen:
            pairs.append((graph_path, label_path))
            seen.add(key)
    for row in _metadata_rows_for_root(data_root):
        label_file = str(row.get("label_file", ""))
        graph_file = str(row.get("graph_file", ""))
        label_path = label_file if os.path.isabs(label_file) else os.path.join(data_root, label_file)
        graph_path = graph_file if os.path.isabs(graph_file) else os.path.join(data_root, graph_file)
        if not graph_file:
            graph_id = str(row.get("graph_id", "") or row.get("model_id", ""))
            graph_path = graph_by_stem.get(graph_id, graph_path)
        if not os.path.isfile(graph_path) or not os.path.isfile(label_path):
            continue
        key = (os.path.abspath(graph_path), os.path.abspath(label_path))
        if key not in seen:
            pairs.append((graph_path, label_path))
            seen.add(key)
    return sorted(pairs, key=lambda pair: (os.path.basename(pair[0]), os.path.basename(pair[1])))


def _dtype_bytes(dtype: str) -> float:
    return DTYPE_BYTES.get((dtype or "fp32").lower(), 4.0)


def _scaled_bytes(fp32_bytes: float, dtype: str) -> float:
    return max(fp32_bytes, 0.0) * (_dtype_bytes(dtype) / 4.0)


def resource_regime_index(name: str) -> int:
    try:
        return RESOURCE_REGIME_VOCAB.index(name)
    except ValueError:
        return -1


def resource_regime_for_totals(node_count: int, total_flops: float, total_memory: float) -> str:
    if total_flops <= 1e6 and total_memory <= 1e6 and node_count <= 8:
        return "small_overhead"
    intensity = _safe_div(total_flops, max(total_memory, 1.0))
    if intensity < 4.0:
        return "memory_bound"
    if intensity > 64.0:
        return "compute_bound"
    return "balanced"


def graph_signature_bucket(node_count: float, edge_count: float) -> str:
    if not np.isfinite(node_count) or node_count <= 0:
        return "unknown"
    nodes = float(node_count)
    edges = max(float(edge_count), 0.0) if np.isfinite(edge_count) else 0.0
    if nodes <= 8:
        size = "tiny"
    elif nodes <= 32:
        size = "small"
    elif nodes <= 128:
        size = "medium"
    else:
        size = "large"

    edge_ratio = edges / max(nodes, 1.0)
    if edge_ratio <= 1.05:
        shape = "chain_like"
    elif edge_ratio <= 2.0:
        shape = "branched"
    else:
        shape = "dense"
    return f"{size}_{shape}"


def graph_signature_for_graph(g: nx.DiGraph) -> str:
    return graph_signature_bucket(float(g.number_of_nodes()), float(g.number_of_edges()))


def graph_family_from_stem(stem: str) -> tuple[str, ...]:
    try:
        raw = stem.split("_bnum", 1)[0].split("_s", 1)[1]
        family = ast.literal_eval(raw)
        if isinstance(family, (list, tuple)):
            return tuple(str(item) for item in family)
    except Exception:
        return ()
    return ()


def graph_family_bucket(family: Sequence[str]) -> str:
    parts = [str(part) for part in family if str(part)]
    if not parts:
        return "unknown"
    unique = tuple(dict.fromkeys(parts))
    if len(unique) == 1:
        return f"pure:{unique[0]}"
    return "mixed:" + "|".join(unique)


def graph_family_for_path(graph_path: str) -> str:
    stem = os.path.splitext(os.path.basename(graph_path))[0]
    return graph_family_bucket(graph_family_from_stem(stem))


def graph_slice_metadata(g: nx.DiGraph) -> dict[str, float | int]:
    features = [(data.get("feature", {}) or {}) for _node, data in g.nodes(data=True)]
    total_flops = 0.0
    total_memory = 0.0
    for feat in features:
        mem = feat.get("memory_info", {}) or {}
        total_flops += max(_f(feat.get("flops")), 0.0)
        total_memory += max(_f(mem.get("bytes")), 0.0)
    batch_size = _batch_size_from_features(features)
    regime = resource_regime_for_totals(len(features), total_flops, total_memory)
    return {
        "batch_size": float(batch_size),
        "resource_regime_idx": int(resource_regime_index(regime)),
    }


def _batch_size_from_features(features: Sequence[dict]) -> float:
    for feat in features:
        mem = feat.get("memory_info", {}) or {}
        bs = _f(mem.get("batch_size"))
        if bs > 0:
            return bs
    return 0.0


def _target_with_mode(label6: Sequence[float], batch_size: float, cfg: FeatureConfig) -> np.ndarray:
    y = np.asarray(label6, dtype=np.float64).reshape(NUM_TARGETS).copy()
    if cfg.time_target_mode == "batch_scaled":
        y[2] *= batch_size
        y[5] *= batch_size
    elif cfg.time_target_mode != "raw":
        raise ValueError(f"unknown time_target_mode {cfg.time_target_mode!r}")
    return y


def _target_for_mode(
    label6: Sequence[float],
    batch_size: float,
    cfg: FeatureConfig,
    base_label6: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    absolute = _target_with_mode(label6, batch_size, cfg)
    base = _target_with_mode(base_label6 if base_label6 is not None else label6, batch_size, cfg)
    mode = target_mode_key(cfg)
    if mode == "absolute":
        return absolute, absolute, base
    if is_log_ratio_target(cfg):
        eps = 1e-9
        target = np.log(np.maximum(absolute, eps)) - np.log(np.maximum(base, eps))
        return target, absolute, base
    raise ValueError(f"unknown target_mode {cfg.target_mode!r}")


def target_stat_values(y_target_raw, cfg: FeatureConfig | None = None):
    if is_log_ratio_target(cfg):
        return y_target_raw
    is_torch = hasattr(y_target_raw, "detach")
    if is_torch:
        return torch.log1p(torch.clamp(y_target_raw, min=0.0))
    return np.log1p(np.maximum(y_target_raw, 0.0))


def _topology_context(g: nx.DiGraph, nodes: list, features: list[dict]) -> dict[str, np.ndarray | float]:
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    try:
        topo_nodes = [node for node in nx.topological_sort(g) if node in index]
        if len(topo_nodes) != n:
            topo_nodes = nodes
    except Exception:
        topo_nodes = nodes

    in_degree = np.asarray([g.in_degree(node) for node in nodes], dtype=np.float64)
    out_degree = np.asarray([g.out_degree(node) for node in nodes], dtype=np.float64)
    rank = np.zeros(n, dtype=np.float64)
    for r, node in enumerate(topo_nodes):
        rank[index[node]] = r

    fwd = np.zeros(n, dtype=np.float64)
    weighted_from = np.zeros(n, dtype=np.float64)
    flops = np.asarray([max(_f(feat.get("flops")), 0.0) for feat in features], dtype=np.float64)
    for node in topo_nodes:
        i = index[node]
        preds = [index[p] for p in g.predecessors(node) if p in index]
        if preds:
            fwd[i] = max(fwd[p] + 1.0 for p in preds)
            weighted_from[i] = max(weighted_from[p] for p in preds) + flops[i]
        else:
            weighted_from[i] = flops[i]

    rev = np.zeros(n, dtype=np.float64)
    weighted_to = np.zeros(n, dtype=np.float64)
    for node in reversed(topo_nodes):
        i = index[node]
        succs = [index[s] for s in g.successors(node) if s in index]
        if succs:
            rev[i] = max(rev[s] + 1.0 for s in succs)
            weighted_to[i] = max(weighted_to[s] for s in succs) + flops[i]
        else:
            weighted_to[i] = flops[i]

    max_depth = float(np.max(fwd)) if n else 0.0
    max_weighted = float(np.max(weighted_from)) if n else 0.0
    on_longest = np.isclose(fwd + rev, max_depth, atol=1e-8).astype(np.float64)
    on_weighted = np.isclose(weighted_from + weighted_to - flops, max_weighted, rtol=1e-6, atol=1e-6).astype(np.float64)

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "topo_index": rank / max(n - 1, 1),
        "forward_depth": fwd / max(max_depth, 1.0),
        "reverse_depth": rev / max(float(np.max(rev)) if n else 0.0, 1.0),
        "fwd_raw": fwd,
        "rev_raw": rev,
        "weighted_from": np.log1p(weighted_from),
        "weighted_to": np.log1p(weighted_to),
        "on_longest": on_longest,
        "on_weighted": on_weighted,
        "max_depth": max_depth,
        "mean_depth": float(np.mean(fwd)) if n else 0.0,
        "max_weighted": max_weighted,
    }


def _node_raw(feat: dict, totals: dict[str, float], cfg: FeatureConfig) -> list[float]:
    args = feat.get("args", {}) or {}
    mem = feat.get("memory_info", {}) or {}
    flops = max(_f(feat.get("flops")), 0.0)
    mac = max(_f(mem.get("bytes")), 0.0)
    weight = max(_f(mem.get("weight_size")), 0.0)
    out: list[float] = []
    if cfg.use_operator_type_onehot:
        out.extend(_node_type_onehot(str(feat.get("type", ""))))
    out.extend(_f(args.get(key)) for key in ARG_KEYS)
    out.extend(
        [
            np.log1p(flops),
            np.log1p(mac),
            np.log1p(weight),
            _f(feat.get("arith_intensity")),
            _safe_div(flops, totals["flops"]),
            _safe_div(mac, totals["mac"]),
            _safe_div(weight, totals["weight"]),
        ]
    )
    return [float(v) for v in out]


def _edge_base(src_feat: dict) -> list[float]:
    mem = src_feat.get("memory_info", {}) or {}
    return [
        np.log1p(max(_f(mem.get("output_size")), 0.0)),
        np.log1p(max(_f(mem.get("batch_size")), 0.0)),
        np.log1p(max(_f(mem.get("output_channels")), 0.0)),
        np.log1p(max(_f(mem.get("output_h")), 0.0)),
        np.log1p(max(_f(mem.get("output_w")), 0.0)),
    ]


def _edge_destination(dst_feat: dict) -> list[float]:
    mem = dst_feat.get("memory_info", {}) or {}
    return [
        np.log1p(max(_f(mem.get("input_size")), 0.0)),
        np.log1p(max(_f(mem.get("batch_size")), 0.0)),
        np.log1p(max(_f(mem.get("input_channels")), 0.0)),
        np.log1p(max(_f(mem.get("input_h")), 0.0)),
        np.log1p(max(_f(mem.get("input_w")), 0.0)),
    ]


def _extract_raw(g: nx.DiGraph, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    layout = feature_layout(cfg)
    nodes = list(g.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    features = [(g.nodes[node].get("feature", {}) or {}) for node in nodes]
    batch_size = _batch_size_from_features(features)

    flops_arr = np.asarray([max(_f(feat.get("flops")), 0.0) for feat in features], dtype=np.float64)
    mac_arr = np.asarray([max(_f((feat.get("memory_info", {}) or {}).get("bytes")), 0.0) for feat in features], dtype=np.float64)
    weight_arr = np.asarray([max(_f((feat.get("memory_info", {}) or {}).get("weight_size")), 0.0) for feat in features], dtype=np.float64)
    totals = {
        "flops": float(np.sum(flops_arr)),
        "mac": float(np.sum(mac_arr)),
        "weight": float(np.sum(weight_arr)),
    }
    topo = _topology_context(g, nodes, features) if (cfg.topology or cfg.critical_path or cfg.edge_topology) else None

    x_raw = np.zeros((n, layout.node_dim), dtype=np.float64)
    for i, feat in enumerate(features):
        row = _node_raw(feat, totals, cfg)
        if cfg.topology and topo is not None:
            row.extend(
                [
                    np.log1p(topo["in_degree"][i]),  # type: ignore[index]
                    np.log1p(topo["out_degree"][i]),  # type: ignore[index]
                    topo["topo_index"][i],  # type: ignore[index]
                    topo["forward_depth"][i],  # type: ignore[index]
                    topo["reverse_depth"][i],  # type: ignore[index]
                    float(topo["in_degree"][i] == 0),  # type: ignore[index]
                    float(topo["out_degree"][i] == 0),  # type: ignore[index]
                    float(topo["out_degree"][i] > 1),  # type: ignore[index]
                    float(topo["in_degree"][i] > 1),  # type: ignore[index]
                ]
            )
        if cfg.critical_path and topo is not None:
            row.extend(
                [
                    topo["fwd_raw"][i],  # type: ignore[index]
                    topo["rev_raw"][i],  # type: ignore[index]
                    topo["weighted_from"][i],  # type: ignore[index]
                    topo["weighted_to"][i],  # type: ignore[index]
                    topo["on_longest"][i],  # type: ignore[index]
                    topo["on_weighted"][i],  # type: ignore[index]
                ]
            )
        x_raw[i] = row

    edges = list(g.edges())
    edge_index = np.zeros((2, len(edges)), dtype=np.int64)
    e_raw = np.zeros((len(edges), layout.edge_dim), dtype=np.float64)
    edge_sizes = np.zeros(len(edges), dtype=np.float64)
    for j, (src, dst) in enumerate(edges):
        si, ti = node_index[src], node_index[dst]
        edge_index[:, j] = [si, ti]
        row = _edge_base(features[si])
        edge_sizes[j] = max(_f((features[si].get("memory_info", {}) or {}).get("output_size")), 0.0)
        if cfg.edge_topology and topo is not None:
            depth_delta = float(topo["fwd_raw"][ti] - topo["fwd_raw"][si])  # type: ignore[index]
            row.extend(
                [
                    np.log1p(topo["out_degree"][si]),  # type: ignore[index]
                    np.log1p(topo["in_degree"][ti]),  # type: ignore[index]
                    topo["forward_depth"][si],  # type: ignore[index]
                    topo["forward_depth"][ti],  # type: ignore[index]
                    depth_delta,
                    float(depth_delta > 1.0),
                ]
            )
        if cfg.include_destination_tensor:
            row.extend(_edge_destination(features[ti]))
        e_raw[j] = row

    u: list[float] = []
    e = len(edges)
    density = _safe_div(float(e), float(n) * (float(n) - 1.0)) if n > 1 else 0.0
    u.extend([float(n), float(e), density])
    for val in _stats4(flops_arr):
        u.append(np.log1p(max(val, 0.0)))
    for val in _stats4(mac_arr):
        u.append(np.log1p(max(val, 0.0)))
    for val in _stats4(weight_arr):
        u.append(np.log1p(max(val, 0.0)))
    u.append(np.log1p(max(float(np.mean(edge_sizes)) if edge_sizes.size else 0.0, 0.0)))
    u.append(_safe_div(totals["flops"], totals["mac"]))
    u.append(batch_size)
    if cfg.topology and topo is not None:
        branch_nodes = float(np.sum(np.asarray(topo["out_degree"]) > 1.0))
        join_nodes = float(np.sum(np.asarray(topo["in_degree"]) > 1.0))
        u.extend(
            [
                float(topo["max_depth"]),
                float(topo["mean_depth"]),
                branch_nodes,
                join_nodes,
                _safe_div(branch_nodes, join_nodes + 1.0),
            ]
        )
    if cfg.critical_path and topo is not None:
        max_weighted = float(topo["max_weighted"])
        u.extend(
            [
                np.log1p(max(max_weighted, 0.0)),
                _safe_div(max_weighted, totals["flops"]),
                float(np.sum(topo["on_longest"])),  # type: ignore[arg-type]
                float(np.sum(topo["on_weighted"])),  # type: ignore[arg-type]
            ]
        )
    if cfg.include_precision_features:
        precision = _precision_settings(cfg)
        for field in ("weight_dtype", "activation_dtype", "grad_dtype", "accum_dtype", "optimizer_state_dtype"):
            u.extend(_onehot(precision[field], DTYPE_VOCAB))
        u.extend(_onehot(precision["tensorcore_mode"], TENSORCORE_MODES))
        u.extend(_onehot(precision["fp8_format"], FP8_FORMATS))
        activation_bytes = _scaled_bytes(totals["mac"], precision["activation_dtype"])
        weight_bytes = _scaled_bytes(totals["weight"], precision["weight_dtype"])
        grad_bytes = _scaled_bytes(totals["weight"], precision["grad_dtype"])
        optimizer_state_bytes = 2.0 * _scaled_bytes(totals["weight"], precision["optimizer_state_dtype"])
        master_weight_bytes = totals["weight"] if _dtype_bytes(precision["weight_dtype"]) < 4.0 else 0.0
        for value in (activation_bytes, weight_bytes, grad_bytes, optimizer_state_bytes, master_weight_bytes):
            u.append(np.log1p(max(value, 0.0)))
    if cfg.include_hardware_features:
        u.extend(
            [
                _f(cfg.compute_capability),
                _f(cfg.architecture_id),
                _f(cfg.sm_count),
                _f(cfg.memory_bandwidth_gbps),
                _f(cfg.vram_gib),
                _f(cfg.l2_cache_mib),
                _f(cfg.peak_fp32_tflops),
                _f(cfg.peak_tf32_tflops),
                _f(cfg.peak_fp16_bf16_tflops),
                _f(cfg.peak_fp8_tflops),
            ]
        )

    u_raw = np.asarray(u, dtype=np.float64)
    if u_raw.shape[0] != layout.global_dim:
        raise ValueError(f"global feature length mismatch: {u_raw.shape[0]} != {layout.global_dim}")
    return x_raw, edge_index, e_raw, u_raw, batch_size


def standardize_targets(y_raw, stats: dict[str, np.ndarray], cfg: FeatureConfig | None = None):
    is_torch = hasattr(y_raw, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_raw.dtype, device=y_raw.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_raw.dtype, device=y_raw.device)
        return (target_stat_values(y_raw, cfg) - ym) / ys
    y = np.asarray(y_raw, dtype=np.float64)
    return (target_stat_values(y, cfg) - stats["y_mean"]) / stats["y_std"]


def invert_targets(y_std, stats: dict[str, np.ndarray], cfg: FeatureConfig | None = None, base_raw=None):
    is_torch = hasattr(y_std, "detach")
    if is_torch:
        ym = torch.as_tensor(stats["y_mean"], dtype=y_std.dtype, device=y_std.device)
        ys = torch.as_tensor(stats["y_std"], dtype=y_std.dtype, device=y_std.device)
        value = y_std * ys + ym
        if is_log_ratio_target(cfg):
            if base_raw is None:
                raise ValueError("base_raw is required to invert log-ratio targets")
            base = torch.as_tensor(base_raw, dtype=y_std.dtype, device=y_std.device)
            return torch.exp(value) * base
        return torch.expm1(value)
    y = np.asarray(y_std, dtype=np.float64)
    value = y * np.asarray(stats["y_std"]) + np.asarray(stats["y_mean"])
    if is_log_ratio_target(cfg):
        if base_raw is None:
            raise ValueError("base_raw is required to invert log-ratio targets")
        return np.exp(value) * np.asarray(base_raw, dtype=np.float64)
    return np.expm1(value)


def build_pyg_data(
    g: nx.DiGraph,
    label6: Sequence[float],
    norm_stats: Optional[dict[str, np.ndarray]] = None,
    feature_config: FeatureConfig | None = None,
    sample_weight: float = 1.0,
    base_label6: Sequence[float] | None = None,
    label_domain: str = "unknown",
    graph_family: str = "unknown",
) -> Data:
    cfg = feature_config or FeatureConfig()
    layout = feature_layout(cfg)
    x_raw, edge_idx, e_raw, u_raw, batch_size = _extract_raw(g, cfg)
    slice_meta = graph_slice_metadata(g)
    y_raw, y_eval_raw, y_base_raw = _target_for_mode(label6, batch_size, cfg, base_label6)

    if norm_stats is not None:
        x = (x_raw - norm_stats["node_mean"]) / norm_stats["node_std"]
        e = (e_raw - norm_stats["edge_mean"]) / norm_stats["edge_std"]
        u = (u_raw - norm_stats["global_mean"]) / norm_stats["global_std"]
        y = standardize_targets(y_raw, norm_stats, cfg)
    else:
        x, e, u, y = x_raw, e_raw, u_raw, target_stat_values(y_raw, cfg)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    data = PerfSeerOptimizedData(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_idx.astype(np.int64)),
        edge_attr=torch.from_numpy(e.astype(np.float32)),
        u=torch.from_numpy(u.astype(np.float32)).view(1, layout.global_dim),
        y=torch.from_numpy(y.astype(np.float32)).view(1, NUM_TARGETS),
        y_raw=torch.from_numpy(y_raw.astype(np.float32)).view(1, NUM_TARGETS),
        y_eval_raw=torch.from_numpy(y_eval_raw.astype(np.float32)).view(1, NUM_TARGETS),
        y_base_raw=torch.from_numpy(y_base_raw.astype(np.float32)).view(1, NUM_TARGETS),
        sample_weight=torch.tensor([float(sample_weight)], dtype=torch.float32),
        label_domain_idx=torch.tensor([label_domain_index(label_domain)], dtype=torch.long),
        precision_config_idx=torch.tensor([precision_config_index(cfg.precision_config)], dtype=torch.long),
        batch_size_raw=torch.tensor([float(slice_meta["batch_size"])], dtype=torch.float32),
        resource_regime_idx=torch.tensor([int(slice_meta["resource_regime_idx"])], dtype=torch.long),
        hardware_id_name=str(cfg.hardware_id or "unknown"),
        graph_family_name=str(graph_family or "unknown"),
    )
    data.num_nodes = int(x_raw.shape[0])
    return data


def build_pyg_inference_data(
    g: nx.DiGraph,
    norm_stats: Optional[dict[str, np.ndarray]] = None,
    feature_config: FeatureConfig | None = None,
    supported_precision_hardware: dict[str, Any] | None = None,
) -> Data:
    """Convert a feature-bearing compute graph into predictor input tensors.

    This mirrors ``build_pyg_data`` for online inference, but intentionally does
    not attach ``y`` or ``y_raw`` labels.
    """

    cfg = feature_config or FeatureConfig()
    validate_precision_hardware_request(cfg, supported_precision_hardware)
    layout = feature_layout(cfg)
    x_raw, edge_idx, e_raw, u_raw, _batch_size = _extract_raw(g, cfg)
    slice_meta = graph_slice_metadata(g)

    if norm_stats is not None:
        x = (x_raw - norm_stats["node_mean"]) / norm_stats["node_std"]
        e = (e_raw - norm_stats["edge_mean"]) / norm_stats["edge_std"]
        u = (u_raw - norm_stats["global_mean"]) / norm_stats["global_std"]
    else:
        x, e, u = x_raw, e_raw, u_raw

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    data = PerfSeerOptimizedData(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_idx.astype(np.int64)),
        edge_attr=torch.from_numpy(e.astype(np.float32)),
        u=torch.from_numpy(u.astype(np.float32)).view(1, layout.global_dim),
        precision_config_idx=torch.tensor([precision_config_index(cfg.precision_config)], dtype=torch.long),
        batch_size_raw=torch.tensor([float(slice_meta["batch_size"])], dtype=torch.float32),
        resource_regime_idx=torch.tensor([int(slice_meta["resource_regime_idx"])], dtype=torch.long),
        graph_family_name="unknown",
    )
    data.num_nodes = int(x_raw.shape[0])
    return data


def _num_procs() -> int:
    env = os.environ.get("PERFSEER_NUM_PROC")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _chunkify(seq: Sequence, n_chunks: int) -> list[list]:
    seq = list(seq)
    if not seq:
        return []
    n_chunks = max(1, min(n_chunks, len(seq)))
    k = (len(seq) + n_chunks - 1) // n_chunks
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def _mean_std_from_acc(count: float, s: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros_like(s), np.ones_like(s)
    mean = s / count
    var = np.maximum(ss / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean, std


def _stats_chunk_worker(arg):
    pairs, cfg = arg
    layout = feature_layout(cfg)
    acc = {
        "node": [0.0, np.zeros(layout.node_dim), np.zeros(layout.node_dim)],
        "edge": [0.0, np.zeros(layout.edge_dim), np.zeros(layout.edge_dim)],
        "global": [0.0, np.zeros(layout.global_dim), np.zeros(layout.global_dim)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    for gp, lp in pairs:
        g = parse_graph(gp)
        pair_cfg = feature_config_for_pair(cfg, gp, lp)
        x_raw, _ei, e_raw, u_raw, batch_size = _extract_raw(g, pair_cfg)
        base_label = base_label_for_pair(pair_cfg, gp, lp)
        y_raw, _y_eval_raw, _y_base_raw = _target_for_mode(parse_label(lp), batch_size, pair_cfg, base_label)
        y_stat = target_stat_values(y_raw, pair_cfg)
        if x_raw.shape[0]:
            acc["node"][0] += x_raw.shape[0]
            acc["node"][1] += x_raw.sum(0)
            acc["node"][2] += (x_raw * x_raw).sum(0)
        if e_raw.shape[0]:
            acc["edge"][0] += e_raw.shape[0]
            acc["edge"][1] += e_raw.sum(0)
            acc["edge"][2] += (e_raw * e_raw).sum(0)
        acc["global"][0] += 1.0
        acc["global"][1] += u_raw
        acc["global"][2] += u_raw * u_raw
        acc["y"][0] += 1.0
        acc["y"][1] += y_stat
        acc["y"][2] += y_stat * y_stat
    return acc


def compute_norm_stats(train_files: Sequence[Tuple[str, str]], feature_config: FeatureConfig | None = None) -> dict[str, np.ndarray]:
    cfg = feature_config or FeatureConfig()
    layout = feature_layout(cfg)
    pairs = list(train_files)
    measured_pairs = [pair for pair in pairs if label_domain_for_pair(*pair) != "pseudo"]
    if measured_pairs:
        pairs = measured_pairs
    total = {
        "node": [0.0, np.zeros(layout.node_dim), np.zeros(layout.node_dim)],
        "edge": [0.0, np.zeros(layout.edge_dim), np.zeros(layout.edge_dim)],
        "global": [0.0, np.zeros(layout.global_dim), np.zeros(layout.global_dim)],
        "y": [0.0, np.zeros(NUM_TARGETS), np.zeros(NUM_TARGETS)],
    }
    nproc = min(_num_procs(), max(1, len(pairs)))
    chunks = _chunkify(pairs, nproc * 4)
    if nproc > 1 and len(chunks) > 1:
        with get_context("fork").Pool(nproc) as pool:
            partials = pool.map(_stats_chunk_worker, [(c, cfg) for c in chunks])
    else:
        partials = [_stats_chunk_worker((c, cfg)) for c in chunks]
    for part in partials:
        for key in total:
            total[key][0] += part[key][0]
            total[key][1] += part[key][1]
            total[key][2] += part[key][2]

    node_mean, node_std = _mean_std_from_acc(*total["node"])
    edge_mean, edge_std = _mean_std_from_acc(*total["edge"])
    global_mean, global_std = _mean_std_from_acc(*total["global"])
    y_mean, y_std = _mean_std_from_acc(*total["y"])

    node_mean_m = np.zeros(layout.node_dim)
    node_std_m = np.ones(layout.node_dim)
    for i in layout.node_std_idx:
        node_mean_m[i] = node_mean[i]
        node_std_m[i] = node_std[i]
    edge_mean_m = np.zeros(layout.edge_dim)
    edge_std_m = np.ones(layout.edge_dim)
    for i in layout.edge_std_idx:
        edge_mean_m[i] = edge_mean[i]
        edge_std_m[i] = edge_std[i]
    global_mean_m = np.zeros(layout.global_dim)
    global_std_m = np.ones(layout.global_dim)
    for i in layout.global_std_idx:
        global_mean_m[i] = global_mean[i]
        global_std_m[i] = global_std[i]

    return {
        "node_mean": node_mean_m.astype(np.float32),
        "node_std": node_std_m.astype(np.float32),
        "edge_mean": edge_mean_m.astype(np.float32),
        "edge_std": edge_std_m.astype(np.float32),
        "global_mean": global_mean_m.astype(np.float32),
        "global_std": global_std_m.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def split_dataset(
    data_root: str = "dataset",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
    split_unit: str = "pair",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    all_pairs = list_precision_pairs(data_root)
    pseudo_pairs = [pair for pair in all_pairs if label_domain_for_pair(*pair) == "pseudo"]
    pairs = [pair for pair in all_pairs if label_domain_for_pair(*pair) != "pseudo"]
    unit = (split_unit or "pair").lower()
    if unit not in {"pair", "graph", "graph_signature", "graph_family"}:
        raise ValueError("split_unit must be 'pair', 'graph', 'graph_signature', or 'graph_family'")
    rng = np.random.default_rng(seed)
    if unit in {"graph", "graph_signature", "graph_family"}:
        groups: dict[str, list[tuple[str, str]]] = {}
        signature_cache: dict[str, str] = {}
        for gp, lp in pairs:
            if unit == "graph":
                key = os.path.basename(gp)
            elif unit == "graph_family":
                key = graph_family_for_path(gp)
            else:
                key = signature_cache.get(gp)
                if key is None:
                    key = graph_signature_for_graph(parse_graph(gp))
                    signature_cache[gp] = key
            groups.setdefault(key, []).append((gp, lp))
        keys = sorted(groups)
        perm_keys = [keys[i] for i in rng.permutation(len(keys))]
        n = len(perm_keys)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))

        def flatten(selected: Sequence[str]) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for key in selected:
                out.extend(sorted(groups[key], key=lambda pair: os.path.basename(pair[1])))
            return out

        train_keys = perm_keys[:n_train]
        val_keys = perm_keys[n_train : n_train + n_val]
        test_keys = perm_keys[n_train + n_val :]
        train = flatten(train_keys)
        train.extend(sorted(pseudo_pairs, key=lambda pair: (os.path.basename(pair[0]), os.path.basename(pair[1]))))
        return train, flatten(val_keys), flatten(test_keys)

    perm = rng.permutation(len(pairs))
    n = len(pairs)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    train = [pairs[i] for i in train_idx]
    train.extend(sorted(pseudo_pairs, key=lambda pair: (os.path.basename(pair[0]), os.path.basename(pair[1]))))
    return train, [pairs[i] for i in val_idx], [pairs[i] for i in test_idx]


def split_hash(pairs: Sequence[Tuple[str, str]]) -> str:
    h = hashlib.sha1()
    for gp, lp in pairs:
        h.update(os.path.basename(gp).encode())
        h.update(os.path.basename(lp).encode())
    return h.hexdigest()


def norm_stats_to_serializable(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: np.asarray(v, dtype=float).reshape(-1).tolist() for k, v in stats.items()}


def save_norm_stats(stats: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_norm_stats(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def _build_chunk_worker(arg):
    pairs, norm_stats, cfg = arg
    out: list[Data] = []
    for gp, lp in pairs:
        try:
            pair_cfg = feature_config_for_pair(cfg, gp, lp)
            weight = sample_weight_for_pair(cfg, gp, lp)
            label_domain = label_domain_for_pair(gp, lp)
            base_label = base_label_for_pair(pair_cfg, gp, lp)
            out.append(
                build_pyg_data(
                    parse_graph(gp),
                    parse_label(lp),
                    norm_stats,
                    pair_cfg,
                    sample_weight=weight,
                    base_label6=base_label,
                    label_domain=label_domain,
                    graph_family=graph_family_for_path(gp),
                )
            )
        except Exception as exc:
            print(f"[PerfSeerOptimizedDataset] skipping {gp}: {exc}", flush=True)
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


class PerfSeerOptimizedDataset(InMemoryDataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        norm_stats: Optional[dict[str, np.ndarray]] = None,
        split_name: Optional[str] = None,
        feature_config: FeatureConfig | None = None,
        force_reprocess: bool = False,
        *,
        root: Optional[str] = None,
        file_list: Optional[Sequence[Tuple[str, str]]] = None,
        split: Optional[str] = None,
    ) -> None:
        data_root = root if root is not None else data_root
        pairs = file_list if file_list is not None else pairs
        split_name = split if split is not None else split_name
        if data_root is None:
            raise ValueError("PerfSeerOptimizedDataset requires data_root/root")
        if pairs is None:
            raise ValueError("PerfSeerOptimizedDataset requires pairs/file_list")
        if norm_stats is None:
            raise ValueError("PerfSeerOptimizedDataset requires norm_stats")
        if split_name is None:
            raise ValueError("PerfSeerOptimizedDataset requires split_name/split")
        self._pairs = list(pairs)
        self._norm_stats = norm_stats
        self._split_name = split_name
        self._feature_config = feature_config or FeatureConfig()
        self._data_root = data_root
        if force_reprocess:
            cache = os.path.join(data_root, "processed", self._cache_filename())
            if os.path.exists(cache):
                os.remove(cache)
        super().__init__(root=data_root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    def _cache_filename(self) -> str:
        h = hashlib.sha1()
        h.update(self._feature_config.signature().encode())
        layout = feature_layout(self._feature_config)
        h.update(f"{layout.node_dim}:{layout.edge_dim}:{layout.global_dim}".encode())
        for gp, lp in self._pairs:
            h.update(os.path.basename(gp).encode())
            h.update(os.path.basename(lp).encode())
        for key in sorted(self._norm_stats):
            h.update(key.encode())
            h.update(np.asarray(self._norm_stats[key], dtype=np.float64).tobytes())
        return f"perfseer_opt_{self._split_name}_{h.hexdigest()[:12]}.pt"

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [self._cache_filename()]

    def download(self) -> None:
        pass

    def process(self) -> None:
        pairs = list(self._pairs)
        nproc = min(_num_procs(), max(1, len(pairs)))
        chunks = _chunkify(pairs, nproc * 4)
        args = [(c, self._norm_stats, self._feature_config) for c in chunks]
        try:
            if nproc > 1 and len(chunks) > 1:
                with get_context("fork").Pool(nproc) as pool:
                    results = pool.map(_build_chunk_worker, args)
            else:
                results = [_build_chunk_worker(a) for a in args]
        except Exception as exc:
            print(f"[PerfSeerOptimizedDataset] parallel build failed ({exc}); falling back to serial")
            results = [_build_chunk_worker(a) for a in args]
        data_list: list[Data] = [d for blob in results for d in pickle.loads(blob)]
        self.save(data_list, self.processed_paths[0])


# Compatibility alias that mirrors the baseline module name.
PerfSeerDataset = PerfSeerOptimizedDataset
