"""Optimized PerfSeer research package.

The source folder intentionally mirrors the user-facing workspace name
``src/perfseer-optimized``. ``pyproject.toml`` maps it to the importable package
name ``perfseer_optimized``.
"""

from .data import (
    FeatureConfig,
    NUM_TARGETS,
    PRECISION_CONFIG_VOCAB,
    RESOURCE_REGIME_VOCAB,
    TARGET_NAMES,
    build_pyg_inference_data,
    list_precision_pairs,
    precision_hardware_config,
    supported_precision_hardware_summary,
    validate_precision_hardware_request,
)
from .model import SeerNet, SeerNetConfig, SeerNetMulti, count_parameters

__all__ = [
    "FeatureConfig",
    "NUM_TARGETS",
    "PRECISION_CONFIG_VOCAB",
    "RESOURCE_REGIME_VOCAB",
    "TARGET_NAMES",
    "build_pyg_inference_data",
    "list_precision_pairs",
    "precision_hardware_config",
    "supported_precision_hardware_summary",
    "validate_precision_hardware_request",
    "SeerNet",
    "SeerNetConfig",
    "SeerNetMulti",
    "count_parameters",
]

from .converter import (
    SourceModelSpec,
    UnsupportedOpError,
    convert_generated_source_to_networkx,
    convert_source_to_networkx,
    convert_source_to_pyg_data,
    graph_from_generated_node_specs,
)

__all__ = [
    "SourceModelSpec",
    "UnsupportedOpError",
    "convert_generated_source_to_networkx",
    "convert_source_to_networkx",
    "convert_source_to_pyg_data",
    "graph_from_generated_node_specs",
]
