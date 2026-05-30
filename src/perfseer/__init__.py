"""PerfSeer reproduction package.

Exports the shared DATA contract symbols so model/train/eval modules can import
feature dimensions and dataset utilities directly from ``perfseer``.
"""

from .data import (  # noqa: F401
    NODE_TYPES,
    NODE_DIM,
    EDGE_DIM,
    GLOBAL_DIM,
    NUM_TARGETS,
    TARGET_NAMES,
    ARG_KEYS,
    parse_graph,
    parse_label,
    build_pyg_data,
    compute_norm_stats,
    standardize_targets,
    invert_targets,
    split_dataset,
    list_pairs,
    PerfSeerDataset,
    save_norm_stats,
    load_norm_stats,
)
from . import metrics  # noqa: F401

__all__ = [
    "NODE_TYPES",
    "NODE_DIM",
    "EDGE_DIM",
    "GLOBAL_DIM",
    "NUM_TARGETS",
    "TARGET_NAMES",
    "ARG_KEYS",
    "parse_graph",
    "parse_label",
    "build_pyg_data",
    "compute_norm_stats",
    "standardize_targets",
    "invert_targets",
    "split_dataset",
    "list_pairs",
    "PerfSeerDataset",
    "save_norm_stats",
    "load_norm_stats",
    "metrics",
]
