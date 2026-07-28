"""Typed, deterministic, multiedge graph IR for PerfSeer v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .baseline import canonical_json
from .version import FEATURE_SCHEMA_VERSION, GRAPH_IR_VERSION, OP_REGISTRY_VERSION


PHASES = ("forward", "loss", "backward", "optimizer")
TENSOR_ROLES = (
    "activation",
    "parameter",
    "buffer",
    "gradient",
    "optimizer_state",
    "constant",
    "model_input",
    "model_output",
)
CAPTURE_QUALITIES = ("strict", "non_strict_validated", "estimated", "failed")
ESTIMATE_METHODS = ("exact_formula", "shape_formula", "profiled_prior", "unknown")


class GraphValidationError(ValueError):
    """Raised when serialized graph state violates the v3 contract."""


@dataclass(frozen=True)
class Estimate:
    value: float = 0.0
    method: str = "unknown"
    confidence: float = 0.0

    def validate(self, name: str) -> None:
        if self.value < 0:
            raise GraphValidationError(f"{name}.value must be nonnegative")
        if self.method not in ESTIMATE_METHODS:
            raise GraphValidationError(f"{name}.method {self.method!r} is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise GraphValidationError(f"{name}.confidence must be in [0, 1]")


@dataclass(frozen=True)
class OperationNodeV3:
    node_id: str
    raw_target: str
    canonical_op_id: str
    family_id: int
    family: str
    phase: str
    exact_op_id: int
    op_hash_bucket: int
    accumulation_dtype: str = "unknown"
    source_module_path: str | None = None
    source_module_stack: tuple[str, ...] = ()
    flags: dict[str, bool] = field(default_factory=dict)
    normalized_args: dict[str, Any] = field(default_factory=dict)
    input_tensor_count: int = 0
    output_tensor_count: int = 0
    input_numel: int = 0
    output_numel: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    parameter_numel: int = 0
    parameter_bytes: int = 0
    buffer_numel: int = 0
    buffer_bytes: int = 0
    flops: Estimate = field(default_factory=Estimate)
    macs: Estimate = field(default_factory=Estimate)
    bytes_read: Estimate = field(default_factory=Estimate)
    bytes_written: Estimate = field(default_factory=Estimate)
    estimated_workspace_bytes: Estimate = field(default_factory=Estimate)
    arithmetic_intensity_flops_per_byte: float = 0.0
    saved_for_backward_bytes: int = 0
    optimizer_state_bytes: int = 0
    topological_index: int = 0
    depth: int = 0
    fan_in: int = 0
    fan_out: int = 0
    live_bytes_before: int = 0
    live_bytes_after: int = 0

    def validate(self) -> None:
        if not self.node_id or not self.raw_target or not self.canonical_op_id:
            raise GraphValidationError("node_id, raw_target, and canonical_op_id are required")
        if self.phase not in PHASES:
            raise GraphValidationError(f"node {self.node_id!r} has invalid phase {self.phase!r}")
        if not self.accumulation_dtype:
            raise GraphValidationError(
                f"node {self.node_id!r} accumulation dtype is empty"
            )
        integers = (
            self.family_id,
            self.exact_op_id,
            self.op_hash_bucket,
            self.input_tensor_count,
            self.output_tensor_count,
            self.input_numel,
            self.output_numel,
            self.input_bytes,
            self.output_bytes,
            self.parameter_numel,
            self.parameter_bytes,
            self.buffer_numel,
            self.buffer_bytes,
            self.saved_for_backward_bytes,
            self.optimizer_state_bytes,
            self.topological_index,
            self.depth,
            self.fan_in,
            self.fan_out,
            self.live_bytes_before,
            self.live_bytes_after,
        )
        if any(value < 0 for value in integers):
            raise GraphValidationError(f"node {self.node_id!r} contains a negative count/ID")
        for name in (
            "flops",
            "macs",
            "bytes_read",
            "bytes_written",
            "estimated_workspace_bytes",
        ):
            getattr(self, name).validate(f"{self.node_id}.{name}")
        if self.arithmetic_intensity_flops_per_byte < 0:
            raise GraphValidationError(
                f"node {self.node_id!r} arithmetic intensity must be nonnegative"
            )


@dataclass(frozen=True)
class TensorEdgeV3:
    edge_id: str
    producer_node_id: str | None
    consumer_node_id: str | None
    producer_output_index: int
    consumer_input_index: int
    tensor_role: str
    shape: tuple[int | str, ...]
    rank: int
    dtype: str
    element_width_bytes: int
    numel: int | None
    tensor_bytes: int | None
    source_name: str | None = None
    stride: tuple[int | str, ...] = ()
    memory_format: str = "unknown"
    alias_group: str | None = None
    is_view: bool = False
    is_materialized: bool = True
    first_use_distance: int = 0
    last_use_distance: int = 0
    dynamic_shape_quality: str = "concrete"

    def validate(self, node_ids: set[str]) -> None:
        if not self.edge_id:
            raise GraphValidationError("edge_id is required")
        if (
            self.producer_node_id is None
            and self.consumer_node_id is None
            and (self.tensor_role != "model_output" or not self.source_name)
        ):
            raise GraphValidationError(
                f"edge {self.edge_id!r} has no producer/consumer and is not a named pass-through output"
            )
        if self.producer_node_id is not None and self.producer_node_id not in node_ids:
            raise GraphValidationError(f"edge {self.edge_id!r} has unknown producer")
        if self.consumer_node_id is not None and self.consumer_node_id not in node_ids:
            raise GraphValidationError(f"edge {self.edge_id!r} has unknown consumer")
        if self.tensor_role not in TENSOR_ROLES:
            raise GraphValidationError(f"edge {self.edge_id!r} has invalid role {self.tensor_role!r}")
        if self.producer_output_index < 0 or self.consumer_input_index < 0:
            raise GraphValidationError(f"edge {self.edge_id!r} has a negative tensor slot")
        if self.rank != len(self.shape):
            raise GraphValidationError(f"edge {self.edge_id!r} rank does not match shape")
        if self.stride and len(self.stride) != self.rank:
            raise GraphValidationError(f"edge {self.edge_id!r} stride does not match rank")
        if self.element_width_bytes <= 0:
            raise GraphValidationError(f"edge {self.edge_id!r} element width must be positive")
        if self.numel is not None and self.numel < 0:
            raise GraphValidationError(f"edge {self.edge_id!r} numel must be nonnegative")
        if self.tensor_bytes is not None and self.tensor_bytes < 0:
            raise GraphValidationError(f"edge {self.edge_id!r} bytes must be nonnegative")
        if (
            self.numel is not None
            and self.tensor_bytes is not None
            and self.tensor_bytes != self.numel * self.element_width_bytes
        ):
            raise GraphValidationError(f"edge {self.edge_id!r} byte units disagree with numel and dtype")


@dataclass(frozen=True)
class GraphGlobalFeatures:
    operation_nodes: int = 0
    tensor_edges: int = 0
    total_flops: float = 0.0
    total_macs: float = 0.0
    total_parameter_numel: int = 0
    total_parameter_bytes: int = 0
    total_buffer_bytes: int = 0
    total_activation_bytes: int = 0
    total_saved_for_backward_bytes: int = 0
    total_optimizer_state_bytes: int = 0
    peak_live_activation_bytes: int = 0
    critical_path_length: int = 0
    unknown_operation_fraction: float = 0.0
    unknown_cost_fraction: float = 0.0
    unknown_byte_fraction: float = 0.0
    coarsening_ratio: float = 1.0

    def validate(self, node_count: int, edge_count: int) -> None:
        if self.operation_nodes != node_count or self.tensor_edges != edge_count:
            raise GraphValidationError("graph-global node/edge counts disagree with graph contents")
        values = asdict(self)
        if any(float(value) < 0 for value in values.values()):
            raise GraphValidationError("graph-global features must be nonnegative")
        for key in ("unknown_operation_fraction", "unknown_cost_fraction", "unknown_byte_fraction"):
            if float(values[key]) > 1.0:
                raise GraphValidationError(f"{key} must be in [0, 1]")
        if not 0.0 < self.coarsening_ratio <= 1.0:
            raise GraphValidationError("coarsening_ratio must be in (0, 1]")


@dataclass(frozen=True)
class CoverageQuality:
    capture_quality: str = "strict"
    backward_capture_quality: str = "estimated"
    tensor_nodes_seen: int = 0
    tensor_nodes_encoded: int = 0
    unknown_operations: int = 0
    custom_operations: int = 0
    replay_samples: int = 0
    replay_validated: bool = False

    def validate(self) -> None:
        if self.capture_quality not in CAPTURE_QUALITIES:
            raise GraphValidationError(f"invalid capture quality {self.capture_quality!r}")
        if self.backward_capture_quality not in CAPTURE_QUALITIES:
            raise GraphValidationError(
                f"invalid backward capture quality {self.backward_capture_quality!r}"
            )
        counts = (
            self.tensor_nodes_seen,
            self.tensor_nodes_encoded,
            self.unknown_operations,
            self.custom_operations,
            self.replay_samples,
        )
        if any(value < 0 for value in counts):
            raise GraphValidationError("coverage counts must be nonnegative")
        if self.tensor_nodes_seen != self.tensor_nodes_encoded:
            raise GraphValidationError("captured tensor-producing operations were not completely encoded")
        if self.capture_quality == "non_strict_validated":
            if not self.replay_validated or self.replay_samples < 3:
                raise GraphValidationError("non-strict capture requires at least three validated replays")


@dataclass(frozen=True)
class GraphIRV3:
    graph_ir_version: str
    feature_schema_version: str
    operator_registry_version: str
    operator_registry_sha256: str
    feature_schema_sha256: str
    capture_backend: str
    capture_mode: str
    pytorch_version: str
    source_fingerprint: str
    model_fingerprint: str
    input_signature: dict[str, Any]
    dynamic_constraints: dict[str, Any]
    training_mode: bool
    precision: str
    optimizer_config: dict[str, Any]
    nodes: tuple[OperationNodeV3, ...]
    tensor_edges: tuple[TensorEdgeV3, ...]
    global_features: GraphGlobalFeatures
    coverage: CoverageQuality
    training_config: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failures: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        operator_registry_sha256: str,
        feature_schema_sha256: str,
        capture_backend: str,
        capture_mode: str,
        pytorch_version: str,
        source_fingerprint: str,
        model_fingerprint: str,
        input_signature: Mapping[str, Any],
        dynamic_constraints: Mapping[str, Any],
        training_mode: bool,
        precision: str,
        optimizer_config: Mapping[str, Any],
        nodes: tuple[OperationNodeV3, ...],
        tensor_edges: tuple[TensorEdgeV3, ...],
        global_features: GraphGlobalFeatures,
        coverage: CoverageQuality,
        training_config: Mapping[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        failures: tuple[dict[str, Any], ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "GraphIRV3":
        graph = cls(
            graph_ir_version=GRAPH_IR_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            operator_registry_version=OP_REGISTRY_VERSION,
            operator_registry_sha256=operator_registry_sha256,
            feature_schema_sha256=feature_schema_sha256,
            capture_backend=capture_backend,
            capture_mode=capture_mode,
            pytorch_version=pytorch_version,
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            input_signature=dict(input_signature),
            dynamic_constraints=dict(dynamic_constraints),
            training_mode=bool(training_mode),
            precision=precision,
            optimizer_config=dict(optimizer_config),
            nodes=nodes,
            tensor_edges=tensor_edges,
            global_features=global_features,
            coverage=coverage,
            training_config=dict(training_config or {}),
            warnings=warnings,
            failures=failures,
            metadata=dict(metadata or {}),
        )
        graph.validate()
        return graph

    @property
    def graph_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if self.graph_ir_version != GRAPH_IR_VERSION:
            raise GraphValidationError(f"expected graph IR {GRAPH_IR_VERSION!r}")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise GraphValidationError(f"expected feature schema {FEATURE_SCHEMA_VERSION!r}")
        if self.operator_registry_version != OP_REGISTRY_VERSION:
            raise GraphValidationError(f"expected registry {OP_REGISTRY_VERSION!r}")
        for name in ("operator_registry_sha256", "feature_schema_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise GraphValidationError(f"{name} must be a lowercase SHA-256")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GraphValidationError("operation node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.tensor_edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise GraphValidationError("tensor edge IDs must be unique")
        for node in self.nodes:
            node.validate()
        node_set = set(node_ids)
        for edge in self.tensor_edges:
            edge.validate(node_set)
        if not isinstance(self.optimizer_config, dict):
            raise GraphValidationError("optimizer_config must be an object")
        if not isinstance(self.training_config, dict):
            raise GraphValidationError("training_config must be an object")
        self.global_features.validate(len(self.nodes), len(self.tensor_edges))
        self.coverage.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        data = self.to_dict()
        if indent is None:
            return canonical_json(data)
        return json.dumps(data, indent=indent, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(indent=2), encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphIRV3":
        raw = dict(data)
        raw["nodes"] = tuple(
            OperationNodeV3(
                **{
                    **node,
                    "source_module_stack": tuple(node.get("source_module_stack", ())),
                    "flops": Estimate(**node.get("flops", {})),
                    "macs": Estimate(**node.get("macs", {})),
                    "bytes_read": Estimate(**node.get("bytes_read", {})),
                    "bytes_written": Estimate(**node.get("bytes_written", {})),
                    "estimated_workspace_bytes": Estimate(**node.get("estimated_workspace_bytes", {})),
                }
            )
            for node in raw["nodes"]
        )
        raw["tensor_edges"] = tuple(
            TensorEdgeV3(
                **{
                    **edge,
                    "shape": tuple(edge.get("shape", ())),
                    "stride": tuple(edge.get("stride", ())),
                }
            )
            for edge in raw["tensor_edges"]
        )
        raw["global_features"] = GraphGlobalFeatures(**raw["global_features"])
        raw["coverage"] = CoverageQuality(**raw["coverage"])
        raw["optimizer_config"] = dict(raw.get("optimizer_config", {}))
        raw["training_config"] = dict(raw.get("training_config", {}))
        raw["warnings"] = tuple(raw.get("warnings", ()))
        raw["failures"] = tuple(raw.get("failures", ()))
        graph = cls(**raw)
        graph.validate()
        return graph

    @classmethod
    def load(cls, path: str | Path) -> "GraphIRV3":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def graph_ir_json_schema() -> dict[str, Any]:
    """Return the JSON Schema envelope used by offline validators."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://perfseer.invalid/schemas/perfseer_ir_v3.json",
        "title": "PerfSeer Graph IR v3",
        "type": "object",
        "required": [
            "graph_ir_version",
            "feature_schema_version",
            "operator_registry_version",
            "operator_registry_sha256",
            "feature_schema_sha256",
            "capture_backend",
            "capture_mode",
            "nodes",
            "tensor_edges",
            "global_features",
            "coverage",
        ],
        "properties": {
            "graph_ir_version": {"const": GRAPH_IR_VERSION},
            "feature_schema_version": {"const": FEATURE_SCHEMA_VERSION},
            "operator_registry_version": {"const": OP_REGISTRY_VERSION},
            "operator_registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "feature_schema_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "nodes": {"type": "array", "items": {"type": "object"}},
            "tensor_edges": {"type": "array", "items": {"type": "object"}},
            "global_features": {"type": "object"},
            "coverage": {"type": "object"},
        },
        "additionalProperties": True,
    }


__all__ = [
    "CAPTURE_QUALITIES",
    "ESTIMATE_METHODS",
    "GraphGlobalFeatures",
    "GraphIRV3",
    "GraphValidationError",
    "CoverageQuality",
    "Estimate",
    "OperationNodeV3",
    "PHASES",
    "TENSOR_ROLES",
    "TensorEdgeV3",
    "graph_ir_json_schema",
]
