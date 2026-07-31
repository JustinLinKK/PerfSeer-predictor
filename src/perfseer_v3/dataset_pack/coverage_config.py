"""Strict loader for frozen A10G operation-coverage gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fingerprints import canonical_sha256


DEFAULT_COVERAGE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "a10g_operation_coverage.yaml"
)
FROZEN_SHAPE_REGIMES = ("tiny", "small", "medium", "large", "boundary")
FROZEN_COVERAGE_DIMENSIONS = (
    "operation",
    "shape_regime",
    "dtype_and_accumulation_policy",
    "phase",
    "backend",
    "layout",
    "optimizer",
    "architecture_context",
)


class CoverageConfigError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CoverageConfigError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CoverageConfigError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise CoverageConfigError(
            f"{context} keys differ from schema; missing={missing}, unknown={unknown}"
        )


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise CoverageConfigError(f"{context} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise CoverageConfigError(f"{context} must not contain duplicates")
    return tuple(value)


def _exact_number(value: Any, expected: int | float, *, context: str) -> int | float:
    if type(value) is not type(expected) or value != expected:
        raise CoverageConfigError(f"{context} must be exactly {expected!r}")
    return value


@dataclass(frozen=True)
class CoverageGates:
    silently_dropped_tensor_producing_operations: int
    structurally_encoded_node_fraction: float
    minimum_strict_complete_capture_rate: float
    minimum_complete_tensor_node_encoding_rate: float
    maximum_unknown_or_custom_gpu_time_fraction: float
    minimum_exact_vocabulary_gpu_time_fraction: float
    capture_profile_workload_fingerprint_mismatches: int

    def validate(self) -> None:
        expected: dict[str, int | float] = {
            "silently_dropped_tensor_producing_operations": 0,
            "structurally_encoded_node_fraction": 1.0,
            "minimum_strict_complete_capture_rate": 0.95,
            "minimum_complete_tensor_node_encoding_rate": 0.99,
            "maximum_unknown_or_custom_gpu_time_fraction": 0.02,
            "minimum_exact_vocabulary_gpu_time_fraction": 0.95,
            "capture_profile_workload_fingerprint_mismatches": 0,
        }
        for name, expected_value in expected.items():
            _exact_number(getattr(self, name), expected_value, context=f"gates.{name}")


@dataclass(frozen=True)
class OperationCoverageConfig:
    version: str
    target_hardware_id: str
    training_approved: bool
    gates: CoverageGates
    required_shape_regimes: tuple[str, ...]
    minimum_composite_contexts_when_semantically_possible: int
    coverage_dimensions: tuple[str, ...]

    def validate(self) -> None:
        if self.version != "perfseer_v3_a10g_operation_coverage_gates_v1":
            raise CoverageConfigError("operation coverage config version mismatch")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise CoverageConfigError("operation coverage config must target AWS A10G")
        if type(self.training_approved) is not bool or self.training_approved:
            raise CoverageConfigError("local coverage config must remain explicitly unapproved")
        self.gates.validate()
        if self.required_shape_regimes != FROZEN_SHAPE_REGIMES:
            raise CoverageConfigError("coverage config must require all five frozen shape regimes")
        if (
            type(self.minimum_composite_contexts_when_semantically_possible) is not int
            or self.minimum_composite_contexts_when_semantically_possible != 3
        ):
            raise CoverageConfigError("coverage config must require exactly three composite contexts")
        if self.coverage_dimensions != FROZEN_COVERAGE_DIMENSIONS:
            raise CoverageConfigError("coverage dimensions differ from the frozen contract")

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))


def load_operation_coverage_config(
    path: str | Path = DEFAULT_COVERAGE_CONFIG_PATH,
) -> OperationCoverageConfig:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _mapping(raw, context="coverage config root")
    _exact_keys(
        root,
        {
            "version",
            "target_hardware_id",
            "training_approved",
            "gates",
            "required_shape_regimes",
            "minimum_composite_contexts_when_semantically_possible",
            "coverage_dimensions",
        },
        context="coverage config root",
    )
    gates = _mapping(root["gates"], context="coverage gates")
    gate_names = {
        "silently_dropped_tensor_producing_operations",
        "structurally_encoded_node_fraction",
        "minimum_strict_complete_capture_rate",
        "minimum_complete_tensor_node_encoding_rate",
        "maximum_unknown_or_custom_gpu_time_fraction",
        "minimum_exact_vocabulary_gpu_time_fraction",
        "capture_profile_workload_fingerprint_mismatches",
    }
    _exact_keys(gates, gate_names, context="coverage gates")
    config = OperationCoverageConfig(
        version=root["version"],
        target_hardware_id=root["target_hardware_id"],
        training_approved=root["training_approved"],
        gates=CoverageGates(**{name: gates[name] for name in gate_names}),
        required_shape_regimes=_string_tuple(
            root["required_shape_regimes"], context="required_shape_regimes"
        ),
        minimum_composite_contexts_when_semantically_possible=root[
            "minimum_composite_contexts_when_semantically_possible"
        ],
        coverage_dimensions=_string_tuple(
            root["coverage_dimensions"], context="coverage_dimensions"
        ),
    )
    config.validate()
    return config


__all__ = [
    "CoverageConfigError",
    "CoverageGates",
    "DEFAULT_COVERAGE_CONFIG_PATH",
    "FROZEN_COVERAGE_DIMENSIONS",
    "FROZEN_SHAPE_REGIMES",
    "OperationCoverageConfig",
    "load_operation_coverage_config",
]
