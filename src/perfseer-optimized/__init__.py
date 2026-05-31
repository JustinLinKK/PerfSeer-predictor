"""Optimized PerfSeer research package.

The source folder intentionally mirrors the user-facing workspace name
``src/perfseer-optimized``. ``pyproject.toml`` maps it to the importable package
name ``perfseer_optimized``.
"""

from .data import FeatureConfig, NUM_TARGETS, TARGET_NAMES
from .model import SeerNet, SeerNetConfig, SeerNetMulti, count_parameters

__all__ = [
    "FeatureConfig",
    "NUM_TARGETS",
    "TARGET_NAMES",
    "SeerNet",
    "SeerNetConfig",
    "SeerNetMulti",
    "count_parameters",
]
