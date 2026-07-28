"""Typed v3 feature tensors, train-only normalization, and cache contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .baseline import canonical_json
from .graph_ir_v3 import GraphIRV3, PHASES, TENSOR_ROLES
from .hardware import graph_hardware_id
from .op_registry import OperationRegistry
from .schema import (
    CAPTURE_BACKENDS,
    CAPTURE_MODES,
    DTYPES,
    DYNAMIC_SHAPE_POLICIES,
    EDGE_ALIAS_CLASSES,
    EDGE_CONTINUOUS_FIELDS,
    EDGE_DYNAMIC_QUALITIES,
    EDGE_FLAG_FIELDS,
    FEATURE_QUALITIES,
    GLOBAL_CONTINUOUS_FIELDS,
    HARDWARE_HASH_BUCKETS,
    LAYOUTS,
    NODE_CONTINUOUS_FIELDS,
    NODE_FLAG_FIELDS,
    OPERATOR_BACKENDS,
    OPTIMIZERS,
    OPTIMIZER_FAMILIES,
    OPTIMIZER_HASH_BUCKETS,
    PHASE_TRANSITIONS,
    QUALITY_FIELDS,
    SCHEDULERS,
    SCHEDULER_FAMILIES,
    SCHEDULER_HASH_BUCKETS,
    SLOT_BUCKETS,
    TRANSFORMS,
    build_feature_schema,
)
from .training_semantics import (
    optimizer_identity,
    scheduler_identity,
    stable_category_bucket,
    training_hyperparameter_values,
)


@dataclass(frozen=True)
class FeatureLayoutV3:
    feature_schema_version: str
    feature_schema_sha256: str
    operator_registry_sha256: str
    node_continuous_fields: tuple[str, ...]
    edge_continuous_fields: tuple[str, ...]
    global_continuous_fields: tuple[str, ...]
    node_flag_fields: tuple[str, ...]
    edge_flag_fields: tuple[str, ...]
    quality_fields: tuple[str, ...]

    @property
    def layout_sha256(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphFeaturesV3:
    x_cont: torch.Tensor
    op_exact_id: torch.Tensor
    op_family_id: torch.Tensor
    op_hash_id: torch.Tensor
    op_overload_hash_id: torch.Tensor
    phase_id: torch.Tensor
    input_dtype_id: torch.Tensor
    dtype_id: torch.Tensor
    accumulation_dtype_id: torch.Tensor
    backend_id: torch.Tensor
    feature_quality_id: torch.Tensor
    layout_id: torch.Tensor
    rank_id: torch.Tensor
    node_flags: torch.Tensor
    edge_index: torch.Tensor
    edge_cont: torch.Tensor
    edge_role_id: torch.Tensor
    edge_source_slot_id: torch.Tensor
    edge_destination_slot_id: torch.Tensor
    edge_dtype_id: torch.Tensor
    edge_layout_id: torch.Tensor
    edge_rank_id: torch.Tensor
    edge_alias_id: torch.Tensor
    edge_dynamic_quality_id: torch.Tensor
    edge_phase_transition_id: torch.Tensor
    edge_flags: torch.Tensor
    u_cont: torch.Tensor
    hardware_id: torch.Tensor
    precision_id: torch.Tensor
    optimizer_id: torch.Tensor
    optimizer_family_id: torch.Tensor
    optimizer_hash_id: torch.Tensor
    scheduler_id: torch.Tensor
    scheduler_family_id: torch.Tensor
    scheduler_hash_id: torch.Tensor
    capture_mode_id: torch.Tensor
    capture_backend_id: torch.Tensor
    dynamic_shape_id: torch.Tensor
    training_mode_id: torch.Tensor
    quality: torch.Tensor
    layout: FeatureLayoutV3
    graph_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        node_count = self.x_cont.size(0)
        edge_count = self.edge_index.size(1)
        if self.x_cont.shape != (node_count, len(self.layout.node_continuous_fields)):
            raise ValueError("x_cont does not match the named node feature layout")
        for tensor in (
            self.op_exact_id,
            self.op_family_id,
            self.op_hash_id,
            self.op_overload_hash_id,
            self.phase_id,
            self.input_dtype_id,
            self.dtype_id,
            self.accumulation_dtype_id,
            self.backend_id,
            self.feature_quality_id,
            self.layout_id,
            self.rank_id,
        ):
            if tensor.shape != (node_count,) or tensor.dtype != torch.long:
                raise ValueError("node categorical tensors must be int64 vectors")
        if self.node_flags.shape != (node_count, len(self.layout.node_flag_fields)):
            raise ValueError("node_flags does not match the named flag layout")
        if self.edge_index.shape[0] != 2 or self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2, E] and dtype int64")
        if self.edge_cont.shape != (edge_count, len(self.layout.edge_continuous_fields)):
            raise ValueError("edge_cont does not match the named edge feature layout")
        for tensor in (
            self.edge_role_id,
            self.edge_source_slot_id,
            self.edge_destination_slot_id,
            self.edge_dtype_id,
            self.edge_layout_id,
            self.edge_rank_id,
            self.edge_alias_id,
            self.edge_dynamic_quality_id,
            self.edge_phase_transition_id,
        ):
            if tensor.shape != (edge_count,) or tensor.dtype != torch.long:
                raise ValueError("edge categorical tensors must be int64 vectors")
        if self.edge_flags.shape != (edge_count, len(self.layout.edge_flag_fields)):
            raise ValueError("edge_flags does not match the named flag layout")
        if self.u_cont.shape != (1, len(self.layout.global_continuous_fields)):
            raise ValueError("u_cont does not match the named global feature layout")
        for tensor in (
            self.hardware_id,
            self.precision_id,
            self.optimizer_id,
            self.optimizer_family_id,
            self.optimizer_hash_id,
            self.scheduler_id,
            self.scheduler_family_id,
            self.scheduler_hash_id,
            self.capture_mode_id,
            self.capture_backend_id,
            self.dynamic_shape_id,
            self.training_mode_id,
        ):
            if tensor.shape != (1,) or tensor.dtype != torch.long:
                raise ValueError("global categorical tensors must be int64 singleton vectors")
        if self.quality.shape != (1, len(self.layout.quality_fields)):
            raise ValueError("quality does not match the named quality layout")
        for tensor in (
            self.x_cont,
            self.node_flags,
            self.edge_cont,
            self.edge_flags,
            self.u_cont,
            self.quality,
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError("continuous and quality features must be finite")
        if node_count and int(self.op_hash_id.min()) < 0:
            raise ValueError("operation hash IDs must be nonnegative")


@dataclass(frozen=True)
class NormalizationBlock:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    clip_low: tuple[float, ...]
    clip_high: tuple[float, ...]


@dataclass(frozen=True)
class NormalizationStatsV3:
    feature_schema_sha256: str
    operator_registry_sha256: str
    layout_sha256: str
    split_name: str
    split_fingerprint: str
    quantiles: tuple[float, float]
    node: NormalizationBlock
    edge: NormalizationBlock
    global_features: NormalizationBlock

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


def feature_layout(registry: OperationRegistry) -> FeatureLayoutV3:
    schema = build_feature_schema(registry)
    return FeatureLayoutV3(
        feature_schema_version=schema["feature_schema_version"],
        feature_schema_sha256=schema["feature_schema_sha256"],
        operator_registry_sha256=registry.sha256,
        node_continuous_fields=NODE_CONTINUOUS_FIELDS,
        edge_continuous_fields=EDGE_CONTINUOUS_FIELDS,
        global_continuous_fields=GLOBAL_CONTINUOUS_FIELDS,
        node_flag_fields=NODE_FLAG_FIELDS,
        edge_flag_fields=EDGE_FLAG_FIELDS,
        quality_fields=QUALITY_FIELDS,
    )


def _numeric_dimensions(shape: Sequence[int | str]) -> tuple[float, ...]:
    return tuple(float(value) for value in shape if isinstance(value, int))


def _dimension_summary(shape: Sequence[int | str]) -> tuple[float, float, float]:
    values = _numeric_dimensions(shape)
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), max(values), sum(values) / len(values)


def _numeric_arguments(value: Any) -> tuple[float, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        number = float(value)
        return (number,) if np.isfinite(number) else ()
    if isinstance(value, Mapping):
        return tuple(
            number
            for key in sorted(value, key=str)
            for number in _numeric_arguments(value[key])
        )
    if isinstance(value, (tuple, list)):
        return tuple(number for item in value for number in _numeric_arguments(item))
    return ()


def _stable_bucket(value: str, buckets: int) -> int:
    if not value or value == "unknown":
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (buckets - 1)


def _operator_backend(raw_target: str) -> str:
    namespace = raw_target.split("::", 1)[0].lower() if "::" in raw_target else ""
    if namespace in {"aten", "prims", "prim", "torch", "triton", "perfseer"}:
        return namespace
    if "transformer_engine" in namespace or namespace.startswith("te"):
        return "transformer_engine"
    if any(token in namespace for token in ("pyg", "torch_scatter", "torch_sparse")):
        return "pyg"
    return "custom" if namespace else "unknown"


def _feature_quality(node: Any) -> str:
    if node.canonical_op_id == "UNK":
        return "unknown"
    if node.flags.get("estimated", False):
        return "estimated"
    methods = {
        node.flops.method,
        node.bytes_read.method,
        node.bytes_written.method,
        node.estimated_workspace_bytes.method,
    }
    if "unknown" in methods:
        return "unknown"
    if "profiled_prior" in methods:
        return "profiled_prior"
    if "shape_formula" in methods:
        return "shape_formula"
    return "exact_formula"


def _node_edges(graph: GraphIRV3, node_id: str) -> tuple[list[Any], list[Any]]:
    incoming = sorted(
        (edge for edge in graph.tensor_edges if edge.consumer_node_id == node_id),
        key=lambda edge: (edge.consumer_input_index, edge.edge_id),
    )
    outgoing = sorted(
        (edge for edge in graph.tensor_edges if edge.producer_node_id == node_id),
        key=lambda edge: (edge.producer_output_index, edge.edge_id),
    )
    return incoming, outgoing


def _node_categorical_metadata(
    graph: GraphIRV3,
    node_id: str,
) -> tuple[str, str, str, int, bool, bool]:
    incoming, outgoing = _node_edges(graph, node_id)
    input_dtypes = {edge.dtype for edge in incoming}
    input_dtype = incoming[0].dtype if incoming else "unknown"
    selected = outgoing[0] if outgoing else (incoming[0] if incoming else None)
    if selected is None:
        return input_dtype, "unknown", "unknown", 0, len(input_dtypes) > 1, False
    output_dtype = selected.dtype
    return (
        input_dtype,
        output_dtype,
        selected.memory_format,
        selected.rank,
        len(input_dtypes) > 1,
        input_dtype != "unknown" and output_dtype != "unknown" and input_dtype != output_dtype,
    )


_FLOATING_DTYPES = frozenset(
    {
        "float8_e4m3fn",
        "float8_e5m2",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    }
)


def graph_precision_category(graph: GraphIRV3) -> str:
    """Return the actual tensor precision policy, including heterogeneous graphs."""

    dtypes = {
        edge.dtype
        for edge in graph.tensor_edges
        if edge.dtype in _FLOATING_DTYPES
    }
    if len(dtypes) > 1:
        return "mixed"
    if dtypes:
        return next(iter(dtypes))
    declared = str(graph.precision).removeprefix("torch.").lower()
    return declared if declared in DTYPES else "unknown"


def _tensor_core_alignment(edges: Sequence[Any]) -> float:
    eligible = 0
    aligned = 0
    for edge in edges:
        dimensions = _numeric_dimensions(edge.shape)
        if len(dimensions) < 2:
            continue
        eligible += 1
        if int(dimensions[-1]) % 8 == 0 and int(dimensions[-2]) % 8 == 0:
            aligned += 1
    return aligned / eligible if eligible else 0.0


def _node_continuous(graph: GraphIRV3, node: Any) -> list[float]:
    incoming, outgoing = _node_edges(graph, node.node_id)
    input_dimensions = tuple(
        dimension for edge in incoming for dimension in _numeric_dimensions(edge.shape)
    )
    output_dimensions = tuple(
        dimension for edge in outgoing for dimension in _numeric_dimensions(edge.shape)
    )
    input_dim = (
        (min(input_dimensions), max(input_dimensions), sum(input_dimensions) / len(input_dimensions))
        if input_dimensions
        else (0.0, 0.0, 0.0)
    )
    output_dim = (
        (min(output_dimensions), max(output_dimensions), sum(output_dimensions) / len(output_dimensions))
        if output_dimensions
        else (0.0, 0.0, 0.0)
    )
    arguments = _numeric_arguments(node.normalized_args)
    lifetimes = [
        max(0, edge.last_use_distance - edge.first_use_distance)
        for edge in incoming
    ]
    values = {
        "input_numel": node.input_numel,
        "output_numel": node.output_numel,
        "parameter_numel": node.parameter_numel,
        "input_bytes": node.input_bytes,
        "output_bytes": node.output_bytes,
        "parameter_bytes": node.parameter_bytes,
        "buffer_bytes": node.buffer_bytes,
        "flops": node.flops.value,
        "macs": node.macs.value,
        "bytes_read": node.bytes_read.value,
        "bytes_written": node.bytes_written.value,
        "arithmetic_intensity_flops_per_byte": node.arithmetic_intensity_flops_per_byte,
        "estimated_workspace_bytes": node.estimated_workspace_bytes.value,
        "saved_for_backward_bytes": node.saved_for_backward_bytes,
        "optimizer_state_bytes": node.optimizer_state_bytes,
        "topological_index": node.topological_index,
        "depth": node.depth,
        "fan_in": node.fan_in,
        "fan_out": node.fan_out,
        "live_bytes_before": node.live_bytes_before,
        "live_bytes_after": node.live_bytes_after,
        "input_rank_max": max((edge.rank for edge in incoming), default=0),
        "output_rank_max": max((edge.rank for edge in outgoing), default=0),
        "input_dimension_min": input_dim[0],
        "input_dimension_max": input_dim[1],
        "input_dimension_mean": input_dim[2],
        "output_dimension_min": output_dim[0],
        "output_dimension_max": output_dim[1],
        "output_dimension_mean": output_dim[2],
        "dynamic_dimension_count": sum(
            not isinstance(dimension, int)
            for edge in (*incoming, *outgoing)
            for dimension in edge.shape
        ),
        "tensor_core_aligned_fraction": _tensor_core_alignment((*incoming, *outgoing)),
        "argument_numeric_count": len(arguments),
        "argument_numeric_sum_abs": sum(abs(value) for value in arguments),
        "argument_numeric_max_abs": max((abs(value) for value in arguments), default=0.0),
        "critical_path_fraction": node.depth / max(1, graph.global_features.critical_path_length - 1),
        "mean_input_reuse_distance": (
            sum(edge.first_use_distance for edge in incoming) / len(incoming)
            if incoming
            else 0.0
        ),
        "max_input_lifetime": max(lifetimes, default=0),
        "flop_estimate_confidence": node.flops.confidence,
        "byte_estimate_confidence": (
            node.bytes_read.confidence + node.bytes_written.confidence
        ) / 2.0,
        "workspace_estimate_confidence": node.estimated_workspace_bytes.confidence,
    }
    return [float(values[name]) for name in NODE_CONTINUOUS_FIELDS]


def _edge_continuous(edge: Any, producer_fan_out: int) -> list[float]:
    dimension_min, dimension_max, dimension_mean = _dimension_summary(edge.shape)
    strides = _numeric_dimensions(edge.stride)
    stride_min = min(strides, default=0.0)
    stride_max = max(strides, default=0.0)
    stride_mean = sum(strides) / len(strides) if strides else 0.0
    values = {
        "rank": edge.rank,
        "element_width_bytes": edge.element_width_bytes,
        "numel": edge.numel or 0,
        "tensor_bytes": edge.tensor_bytes or 0,
        "first_use_distance": edge.first_use_distance,
        "last_use_distance": edge.last_use_distance,
        "dimension_min": dimension_min,
        "dimension_max": dimension_max,
        "dimension_mean": dimension_mean,
        "stride_min": stride_min,
        "stride_max": stride_max,
        "stride_mean": stride_mean,
        "lifetime_distance": max(0, edge.last_use_distance - edge.first_use_distance),
        "producer_fan_out": producer_fan_out,
    }
    return [float(values[name]) for name in EDGE_CONTINUOUS_FIELDS]


def _global_continuous(graph: GraphIRV3) -> list[float]:
    globals_dict = asdict(graph.global_features)
    phase_nodes = {phase: [node for node in graph.nodes if node.phase == phase] for phase in PHASES}
    model_inputs = [edge for edge in graph.tensor_edges if edge.tensor_role == "model_input"]
    hardware = graph.metadata.get("hardware_features", {})
    if not isinstance(hardware, Mapping):
        hardware = {}
    optimizer = graph.optimizer_config
    floating_edges = [
        edge for edge in graph.tensor_edges if edge.dtype in _FLOATING_DTYPES
    ]
    floating_denominator = max(1, len(floating_edges))
    batch_size = graph.metadata.get("batch_size")
    if batch_size is None:
        batch_size = next(
            (
                edge.shape[0]
                for edge in model_inputs
                if edge.shape and isinstance(edge.shape[0], int)
            ),
            1,
        )
    values: dict[str, float | int] = {
        **globals_dict,
        "batch_size": batch_size,
        "gradient_accumulation_steps": optimizer.get("gradient_accumulation_steps", 1),
        "gradient_clip_norm": optimizer.get("gradient_clip_norm", 0.0) or 0.0,
        "loss_scale": optimizer.get("loss_scale", 1.0) or 1.0,
        "activation_checkpointing": float(
            bool(optimizer.get("activation_checkpointing", False))
        ),
        "optimizer_foreach": float(bool(optimizer.get("foreach", False))),
        "optimizer_fused": float(bool(optimizer.get("fused", False))),
        "model_input_numel": sum(edge.numel or 0 for edge in model_inputs),
        "model_input_bytes": sum(edge.tensor_bytes or 0 for edge in model_inputs),
        "dynamic_symbol_count": sum(
            not isinstance(dimension, int)
            for edge in graph.tensor_edges
            for dimension in edge.shape
        ),
        "custom_operation_fraction": graph.coverage.custom_operations / max(1, len(graph.nodes)),
        "capture_replay_samples": graph.coverage.replay_samples,
        "hardware_memory_bytes": hardware.get("memory_bytes", 0),
        "hardware_sm_count": hardware.get("sm_count", 0),
        "hardware_compute_capability": hardware.get("compute_capability", 0),
        "hardware_memory_bandwidth_bytes_per_second": hardware.get(
            "memory_bandwidth_bytes_per_second",
            0,
        ),
        "hardware_peak_flops": hardware.get("peak_flops", 0),
        "distinct_floating_dtype_count": len(
            {edge.dtype for edge in floating_edges}
        ),
        "float32_tensor_fraction": sum(
            edge.dtype == "float32" for edge in floating_edges
        )
        / floating_denominator,
        "bfloat16_tensor_fraction": sum(
            edge.dtype == "bfloat16" for edge in floating_edges
        )
        / floating_denominator,
        "float16_tensor_fraction": sum(
            edge.dtype == "float16" for edge in floating_edges
        )
        / floating_denominator,
        "float8_tensor_fraction": sum(
            edge.dtype in {"float8_e4m3fn", "float8_e5m2"}
            for edge in floating_edges
        )
        / floating_denominator,
    }
    values.update(training_hyperparameter_values(optimizer, graph.training_config))
    for phase, nodes in phase_nodes.items():
        values[f"{phase}_node_count"] = len(nodes)
        values[f"{phase}_flops"] = sum(node.flops.value for node in nodes)
        values[f"{phase}_bytes_read"] = sum(node.bytes_read.value for node in nodes)
        values[f"{phase}_bytes_written"] = sum(node.bytes_written.value for node in nodes)
    return [float(values[name]) for name in GLOBAL_CONTINUOUS_FIELDS]


def build_graph_features(
    graph: GraphIRV3,
    *,
    registry: OperationRegistry | None = None,
) -> GraphFeaturesV3:
    graph.validate()
    registry = registry or OperationRegistry.load()
    layout = feature_layout(registry)
    if graph.operator_registry_sha256 != registry.sha256:
        raise ValueError("graph and feature builder operator registry hashes differ")
    if graph.feature_schema_sha256 != layout.feature_schema_sha256:
        raise ValueError("graph and feature builder feature schema hashes differ")
    dtype_to_id = {name: index for index, name in enumerate(DTYPES)}
    layout_to_id = {name: index for index, name in enumerate(LAYOUTS)}
    phase_to_id = {name: index for index, name in enumerate(PHASES)}
    role_to_id = {name: index for index, name in enumerate(TENSOR_ROLES)}
    backend_to_id = {name: index for index, name in enumerate(OPERATOR_BACKENDS)}
    feature_quality_to_id = {
        name: index for index, name in enumerate(FEATURE_QUALITIES)
    }
    alias_to_id = {name: index for index, name in enumerate(EDGE_ALIAS_CLASSES)}
    dynamic_quality_to_id = {
        name: index for index, name in enumerate(EDGE_DYNAMIC_QUALITIES)
    }
    phase_transition_to_id = {
        name: index for index, name in enumerate(PHASE_TRANSITIONS)
    }
    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    node_by_id = {node.node_id: node for node in graph.nodes}

    x_cont = torch.tensor(
        [_node_continuous(graph, node) for node in graph.nodes],
        dtype=torch.float32,
    ).reshape(len(graph.nodes), len(NODE_CONTINUOUS_FIELDS))
    categorical = [
        _node_categorical_metadata(graph, node.node_id)
        for node in graph.nodes
    ]
    op_exact = torch.tensor([node.exact_op_id for node in graph.nodes], dtype=torch.long)
    op_family = torch.tensor([node.family_id for node in graph.nodes], dtype=torch.long)
    op_hash = torch.tensor([node.op_hash_bucket for node in graph.nodes], dtype=torch.long)
    op_overload_hash = torch.tensor(
        [registry.stable_hash_bucket(node.raw_target) for node in graph.nodes],
        dtype=torch.long,
    )
    phase = torch.tensor([phase_to_id[node.phase] for node in graph.nodes], dtype=torch.long)
    input_dtype_ids = torch.tensor(
        [dtype_to_id.get(input_dtype, 0) for input_dtype, _, _, _, _, _ in categorical],
        dtype=torch.long,
    )
    dtype_ids = torch.tensor(
        [dtype_to_id.get(dtype, 0) for _, dtype, _, _, _, _ in categorical],
        dtype=torch.long,
    )
    accumulation_dtype_ids = torch.tensor(
        [
            dtype_to_id.get(node.accumulation_dtype, 0)
            for node in graph.nodes
        ],
        dtype=torch.long,
    )
    backend_ids = torch.tensor(
        [backend_to_id[_operator_backend(node.raw_target)] for node in graph.nodes],
        dtype=torch.long,
    )
    feature_quality_ids = torch.tensor(
        [feature_quality_to_id[_feature_quality(node)] for node in graph.nodes],
        dtype=torch.long,
    )
    layout_ids = torch.tensor(
        [
            layout_to_id.get(memory_format, 0)
            for _, _, memory_format, _, _, _ in categorical
        ],
        dtype=torch.long,
    )
    rank_ids = torch.tensor(
        [min(16, max(0, rank)) for _, _, _, rank, _, _ in categorical],
        dtype=torch.long,
    )
    mixed_input_dtype = {
        node.node_id: categorical[index][4]
        for index, node in enumerate(graph.nodes)
    }
    changes_dtype = {
        node.node_id: categorical[index][5]
        for index, node in enumerate(graph.nodes)
    }

    def node_flag(node: Any, name: str) -> bool:
        if name == "unknown":
            return node.canonical_op_id == "UNK"
        if name == "cost_unknown":
            return node.flops.method == "unknown"
        if name == "byte_unknown":
            return (
                node.bytes_read.method == "unknown"
                or node.bytes_written.method == "unknown"
            )
        if name == "mixed_input_dtype":
            return mixed_input_dtype[node.node_id]
        if name == "changes_dtype":
            return changes_dtype[node.node_id]
        return bool(node.flags.get(name, False))

    node_flags = torch.tensor(
        [
            [float(node_flag(node, name)) for name in NODE_FLAG_FIELDS]
            for node in graph.nodes
        ],
        dtype=torch.float32,
    ).reshape(len(graph.nodes), len(NODE_FLAG_FIELDS))

    # External model/parameter/buffer/output edges are represented as typed
    # self-loops on their attached operation. This retains their tensor role
    # and metadata without inventing untyped virtual operation nodes.
    model_edges = [
        edge
        for edge in graph.tensor_edges
        if edge.producer_node_id is not None or edge.consumer_node_id is not None
    ]

    def attached_source(edge: Any) -> str:
        return edge.producer_node_id or edge.consumer_node_id

    def attached_destination(edge: Any) -> str:
        return edge.consumer_node_id or edge.producer_node_id

    edge_index = torch.tensor(
        [
            [node_index[attached_source(edge)] for edge in model_edges],
            [node_index[attached_destination(edge)] for edge in model_edges],
        ],
        dtype=torch.long,
    ).reshape(2, len(model_edges))
    alias_counts: dict[str, int] = {}
    for edge in graph.tensor_edges:
        if edge.alias_group:
            alias_counts[edge.alias_group] = alias_counts.get(edge.alias_group, 0) + 1
    edge_cont = torch.tensor(
        [
            _edge_continuous(
                edge,
                node_by_id[edge.producer_node_id].fan_out
                if edge.producer_node_id is not None
                else 0,
            )
            for edge in model_edges
        ],
        dtype=torch.float32,
    ).reshape(len(model_edges), len(EDGE_CONTINUOUS_FIELDS))
    edge_roles = torch.tensor(
        [role_to_id[edge.tensor_role] for edge in model_edges],
        dtype=torch.long,
    )
    edge_source_slots = torch.tensor(
        [min(SLOT_BUCKETS - 1, edge.producer_output_index) for edge in model_edges],
        dtype=torch.long,
    )
    edge_destination_slots = torch.tensor(
        [min(SLOT_BUCKETS - 1, edge.consumer_input_index) for edge in model_edges],
        dtype=torch.long,
    )
    edge_dtype_ids = torch.tensor(
        [dtype_to_id.get(edge.dtype, 0) for edge in model_edges],
        dtype=torch.long,
    )
    edge_layout_ids = torch.tensor(
        [layout_to_id.get(edge.memory_format, 0) for edge in model_edges],
        dtype=torch.long,
    )
    edge_rank_ids = torch.tensor(
        [min(16, max(0, edge.rank)) for edge in model_edges],
        dtype=torch.long,
    )

    def alias_class(edge: Any) -> str:
        if edge.is_view:
            return "view"
        if edge.alias_group is None:
            return "unknown"
        if alias_counts.get(edge.alias_group, 0) > 1:
            return "aliased_materialized"
        return "unique_materialized"

    edge_alias_ids = torch.tensor(
        [alias_to_id[alias_class(edge)] for edge in model_edges],
        dtype=torch.long,
    )
    edge_dynamic_ids = torch.tensor(
        [
            dynamic_quality_to_id.get(edge.dynamic_shape_quality, 0)
            for edge in model_edges
        ],
        dtype=torch.long,
    )

    def phase_transition(edge: Any) -> str:
        if edge.producer_node_id is None or edge.consumer_node_id is None:
            return "unknown"
        return (
            f"{node_by_id[edge.producer_node_id].phase}->"
            f"{node_by_id[edge.consumer_node_id].phase}"
        )

    edge_phase_transition_ids = torch.tensor(
        [phase_transition_to_id[phase_transition(edge)] for edge in model_edges],
        dtype=torch.long,
    )

    def edge_flag(edge: Any, name: str) -> bool:
        if name == "is_view":
            return edge.is_view
        if name == "is_materialized":
            return edge.is_materialized
        if name == "is_contiguous":
            return edge.memory_format in {"contiguous", "channels_last", "channels_last_3d"}
        if name == "is_phase_boundary":
            return (
                edge.producer_node_id is not None
                and edge.consumer_node_id is not None
                and node_by_id[edge.producer_node_id].phase
                != node_by_id[edge.consumer_node_id].phase
            )
        if name == "has_symbolic_shape":
            return any(not isinstance(dimension, int) for dimension in edge.shape)
        raise KeyError(name)

    edge_flags = torch.tensor(
        [
            [float(edge_flag(edge, name)) for name in EDGE_FLAG_FIELDS]
            for edge in model_edges
        ],
        dtype=torch.float32,
    ).reshape(len(model_edges), len(EDGE_FLAG_FIELDS))
    u_cont = torch.tensor(
        [_global_continuous(graph)],
        dtype=torch.float32,
    )
    custom_fraction = graph.coverage.custom_operations / max(1, len(graph.nodes))
    dynamic_edges = sum(
        any(not isinstance(dimension, int) for dimension in edge.shape)
        for edge in graph.tensor_edges
    )
    quality = torch.tensor(
        [
            [
                graph.global_features.unknown_operation_fraction,
                graph.global_features.unknown_cost_fraction,
                graph.global_features.unknown_byte_fraction,
                custom_fraction,
                float(graph.coverage.capture_quality == "strict"),
                float(graph.coverage.backward_capture_quality == "strict"),
                float(graph.coverage.replay_validated),
                dynamic_edges / max(1, len(graph.tensor_edges)),
            ]
        ],
        dtype=torch.float32,
    )
    hardware_name = graph_hardware_id(graph.metadata)
    precision_name = graph_precision_category(graph)
    optimizer_name, optimizer_family, optimizer_signature = optimizer_identity(
        graph.optimizer_config
    )
    scheduler_name, scheduler_family, scheduler_signature = scheduler_identity(
        graph.training_config
    )
    capture_mode_name = graph.capture_mode if graph.capture_mode in CAPTURE_MODES else "unknown"
    capture_backend_name = (
        graph.capture_backend if graph.capture_backend in CAPTURE_BACKENDS else "unknown"
    )
    if graph.dynamic_constraints:
        dynamic_shape_name = "constrained"
    elif dynamic_edges:
        dynamic_shape_name = "symbolic"
    else:
        dynamic_shape_name = "none"
    features = GraphFeaturesV3(
        x_cont=x_cont,
        op_exact_id=op_exact,
        op_family_id=op_family,
        op_hash_id=op_hash,
        op_overload_hash_id=op_overload_hash,
        phase_id=phase,
        input_dtype_id=input_dtype_ids,
        dtype_id=dtype_ids,
        accumulation_dtype_id=accumulation_dtype_ids,
        backend_id=backend_ids,
        feature_quality_id=feature_quality_ids,
        layout_id=layout_ids,
        rank_id=rank_ids,
        node_flags=node_flags,
        edge_index=edge_index,
        edge_cont=edge_cont,
        edge_role_id=edge_roles,
        edge_source_slot_id=edge_source_slots,
        edge_destination_slot_id=edge_destination_slots,
        edge_dtype_id=edge_dtype_ids,
        edge_layout_id=edge_layout_ids,
        edge_rank_id=edge_rank_ids,
        edge_alias_id=edge_alias_ids,
        edge_dynamic_quality_id=edge_dynamic_ids,
        edge_phase_transition_id=edge_phase_transition_ids,
        edge_flags=edge_flags,
        u_cont=u_cont,
        hardware_id=torch.tensor(
            [_stable_bucket(hardware_name, HARDWARE_HASH_BUCKETS)],
            dtype=torch.long,
        ),
        precision_id=torch.tensor([dtype_to_id.get(precision_name, 0)], dtype=torch.long),
        optimizer_id=torch.tensor([OPTIMIZERS.index(optimizer_name)], dtype=torch.long),
        optimizer_family_id=torch.tensor(
            [OPTIMIZER_FAMILIES.index(optimizer_family)],
            dtype=torch.long,
        ),
        optimizer_hash_id=torch.tensor(
            [stable_category_bucket(optimizer_signature, OPTIMIZER_HASH_BUCKETS)],
            dtype=torch.long,
        ),
        scheduler_id=torch.tensor([SCHEDULERS.index(scheduler_name)], dtype=torch.long),
        scheduler_family_id=torch.tensor(
            [SCHEDULER_FAMILIES.index(scheduler_family)],
            dtype=torch.long,
        ),
        scheduler_hash_id=torch.tensor(
            [stable_category_bucket(scheduler_signature, SCHEDULER_HASH_BUCKETS)],
            dtype=torch.long,
        ),
        capture_mode_id=torch.tensor([CAPTURE_MODES.index(capture_mode_name)], dtype=torch.long),
        capture_backend_id=torch.tensor(
            [CAPTURE_BACKENDS.index(capture_backend_name)],
            dtype=torch.long,
        ),
        dynamic_shape_id=torch.tensor(
            [DYNAMIC_SHAPE_POLICIES.index(dynamic_shape_name)],
            dtype=torch.long,
        ),
        training_mode_id=torch.tensor([int(graph.training_mode)], dtype=torch.long),
        quality=quality,
        layout=layout,
        graph_sha256=graph.graph_sha256,
        metadata={
            "capture_quality": graph.coverage.capture_quality,
            "backward_capture_quality": graph.coverage.backward_capture_quality,
            "coarsening_ratio": graph.global_features.coarsening_ratio,
            "external_edges_as_typed_self_loops": sum(
                edge.producer_node_id is None or edge.consumer_node_id is None
                for edge in model_edges
            ),
            "unattached_passthrough_edges": sum(
                edge.producer_node_id is None and edge.consumer_node_id is None
                for edge in graph.tensor_edges
            ),
            "hardware_id": hardware_name,
            "precision": precision_name,
            "declared_precision": str(graph.precision),
            "optimizer": optimizer_name,
            "optimizer_family": optimizer_family,
            "optimizer_signature": optimizer_signature,
            "scheduler": scheduler_name,
            "scheduler_family": scheduler_family,
            "scheduler_signature": scheduler_signature,
            "dynamic_shape_policy": dynamic_shape_name,
        },
    )
    features.validate()
    return features


def _transform(array: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    result = array.astype(np.float64, copy=True)
    for index, name in enumerate(fields):
        if TRANSFORMS.get(name) == "log1p_nonnegative":
            result[:, index] = np.log1p(np.maximum(result[:, index], 0.0))
    return result


def _fit_block(
    arrays: Iterable[np.ndarray],
    fields: Sequence[str],
    quantiles: tuple[float, float],
) -> NormalizationBlock:
    usable = [array for array in arrays if array.shape[0] > 0]
    width = len(fields)
    if not usable:
        zeros = tuple(0.0 for _ in range(width))
        ones = tuple(1.0 for _ in range(width))
        return NormalizationBlock(zeros, ones, zeros, zeros)
    values = _transform(np.concatenate(usable, axis=0), fields)
    low = np.quantile(values, quantiles[0], axis=0)
    high = np.quantile(values, quantiles[1], axis=0)
    clipped = np.clip(values, low, high)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return NormalizationBlock(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
        tuple(float(value) for value in low),
        tuple(float(value) for value in high),
    )


def fit_normalization(
    samples: Sequence[GraphFeaturesV3],
    *,
    split_name: str,
    split_fingerprint: str,
    quantiles: tuple[float, float] = (0.001, 0.999),
) -> NormalizationStatsV3:
    if split_name != "train":
        raise ValueError("normalization statistics may only be fit on the training split")
    if not samples:
        raise ValueError("cannot fit normalization without training samples")
    first = samples[0]
    if any(sample.layout != first.layout for sample in samples):
        raise ValueError("normalization samples have incompatible feature layouts")
    return NormalizationStatsV3(
        feature_schema_sha256=first.layout.feature_schema_sha256,
        operator_registry_sha256=first.layout.operator_registry_sha256,
        layout_sha256=first.layout.layout_sha256,
        split_name=split_name,
        split_fingerprint=split_fingerprint,
        quantiles=quantiles,
        node=_fit_block(
            (sample.x_cont.cpu().numpy() for sample in samples),
            first.layout.node_continuous_fields,
            quantiles,
        ),
        edge=_fit_block(
            (sample.edge_cont.cpu().numpy() for sample in samples),
            first.layout.edge_continuous_fields,
            quantiles,
        ),
        global_features=_fit_block(
            (sample.u_cont.cpu().numpy() for sample in samples),
            first.layout.global_continuous_fields,
            quantiles,
        ),
    )


def _normalize_tensor(
    tensor: torch.Tensor,
    fields: Sequence[str],
    block: NormalizationBlock,
) -> tuple[torch.Tensor, int, int]:
    if tensor.numel() == 0:
        return tensor.clone(), 0, 0
    array = _transform(tensor.detach().cpu().numpy(), fields)
    low = np.asarray(block.clip_low)
    high = np.asarray(block.clip_high)
    below = int((array < low).sum())
    above = int((array > high).sum())
    clipped = np.clip(array, low, high)
    normalized = (clipped - np.asarray(block.mean)) / np.asarray(block.std)
    return torch.as_tensor(normalized, dtype=torch.float32), below + above, array.size


def apply_normalization(
    sample: GraphFeaturesV3,
    stats: NormalizationStatsV3,
) -> GraphFeaturesV3:
    if sample.layout.feature_schema_sha256 != stats.feature_schema_sha256:
        raise ValueError("normalization feature schema hash mismatch")
    if sample.layout.operator_registry_sha256 != stats.operator_registry_sha256:
        raise ValueError("normalization operator registry hash mismatch")
    if sample.layout.layout_sha256 != stats.layout_sha256:
        raise ValueError("normalization ordered feature layout mismatch")
    x_cont, x_clipped, x_total = _normalize_tensor(
        sample.x_cont,
        sample.layout.node_continuous_fields,
        stats.node,
    )
    edge_cont, edge_clipped, edge_total = _normalize_tensor(
        sample.edge_cont,
        sample.layout.edge_continuous_fields,
        stats.edge,
    )
    u_cont, u_clipped, u_total = _normalize_tensor(
        sample.u_cont,
        sample.layout.global_continuous_fields,
        stats.global_features,
    )
    result = replace(
        sample,
        x_cont=x_cont,
        edge_cont=edge_cont,
        u_cont=u_cont,
        metadata={
            **sample.metadata,
            "normalization_sha256": stats.sha256,
            "clip_frequency": (
                (x_clipped + edge_clipped + u_clipped)
                / max(1, x_total + edge_total + u_total)
            ),
        },
    )
    result.validate()
    return result


def validate_checkpoint_layout(
    sample: GraphFeaturesV3,
    metadata: Mapping[str, Any],
) -> None:
    expected = {
        "feature_schema_sha256": sample.layout.feature_schema_sha256,
        "operator_registry_sha256": sample.layout.operator_registry_sha256,
        "layout_sha256": sample.layout.layout_sha256,
        "node_continuous_fields": list(sample.layout.node_continuous_fields),
        "edge_continuous_fields": list(sample.layout.edge_continuous_fields),
        "global_continuous_fields": list(sample.layout.global_continuous_fields),
        "node_flag_fields": list(sample.layout.node_flag_fields),
        "edge_flag_fields": list(sample.layout.edge_flag_fields),
        "quality_fields": list(sample.layout.quality_fields),
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "checkpoint feature layout mismatch: " + ", ".join(sorted(mismatches))
        )


def sample_cache_key(
    sample: GraphFeaturesV3,
    *,
    coarsening_sha256: str,
    split_fingerprint: str,
    normalization_sha256: str,
) -> str:
    payload = {
        "graph_sha256": sample.graph_sha256,
        "feature_schema_sha256": sample.layout.feature_schema_sha256,
        "operator_registry_sha256": sample.layout.operator_registry_sha256,
        "layout_sha256": sample.layout.layout_sha256,
        "coarsening_sha256": coarsening_sha256,
        "split_fingerprint": split_fingerprint,
        "normalization_sha256": normalization_sha256,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphBatchV3:
    x_cont: torch.Tensor
    op_exact_id: torch.Tensor
    op_family_id: torch.Tensor
    op_hash_id: torch.Tensor
    op_overload_hash_id: torch.Tensor
    phase_id: torch.Tensor
    input_dtype_id: torch.Tensor
    dtype_id: torch.Tensor
    accumulation_dtype_id: torch.Tensor
    backend_id: torch.Tensor
    feature_quality_id: torch.Tensor
    layout_id: torch.Tensor
    rank_id: torch.Tensor
    node_flags: torch.Tensor
    edge_index: torch.Tensor
    edge_cont: torch.Tensor
    edge_role_id: torch.Tensor
    edge_source_slot_id: torch.Tensor
    edge_destination_slot_id: torch.Tensor
    edge_dtype_id: torch.Tensor
    edge_layout_id: torch.Tensor
    edge_rank_id: torch.Tensor
    edge_alias_id: torch.Tensor
    edge_dynamic_quality_id: torch.Tensor
    edge_phase_transition_id: torch.Tensor
    edge_flags: torch.Tensor
    u_cont: torch.Tensor
    hardware_id: torch.Tensor
    precision_id: torch.Tensor
    optimizer_id: torch.Tensor
    optimizer_family_id: torch.Tensor
    optimizer_hash_id: torch.Tensor
    scheduler_id: torch.Tensor
    scheduler_family_id: torch.Tensor
    scheduler_hash_id: torch.Tensor
    capture_mode_id: torch.Tensor
    capture_backend_id: torch.Tensor
    dynamic_shape_id: torch.Tensor
    training_mode_id: torch.Tensor
    quality: torch.Tensor
    batch: torch.Tensor
    layout: FeatureLayoutV3


def batch_graph_features(samples: Sequence[GraphFeaturesV3]) -> GraphBatchV3:
    if not samples:
        raise ValueError("cannot batch zero graph samples")
    layout = samples[0].layout
    if any(sample.layout != layout for sample in samples):
        raise ValueError("cannot batch incompatible feature layouts")
    node_offset = 0
    edge_indices = []
    batches = []
    for graph_index, sample in enumerate(samples):
        sample.validate()
        edge_indices.append(sample.edge_index + node_offset)
        batches.append(torch.full((sample.x_cont.size(0),), graph_index, dtype=torch.long))
        node_offset += sample.x_cont.size(0)
    concatenate = lambda name: torch.cat([getattr(sample, name) for sample in samples], dim=0)
    return GraphBatchV3(
        x_cont=concatenate("x_cont"),
        op_exact_id=concatenate("op_exact_id"),
        op_family_id=concatenate("op_family_id"),
        op_hash_id=concatenate("op_hash_id"),
        op_overload_hash_id=concatenate("op_overload_hash_id"),
        phase_id=concatenate("phase_id"),
        input_dtype_id=concatenate("input_dtype_id"),
        dtype_id=concatenate("dtype_id"),
        accumulation_dtype_id=concatenate("accumulation_dtype_id"),
        backend_id=concatenate("backend_id"),
        feature_quality_id=concatenate("feature_quality_id"),
        layout_id=concatenate("layout_id"),
        rank_id=concatenate("rank_id"),
        node_flags=concatenate("node_flags"),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_cont=concatenate("edge_cont"),
        edge_role_id=concatenate("edge_role_id"),
        edge_source_slot_id=concatenate("edge_source_slot_id"),
        edge_destination_slot_id=concatenate("edge_destination_slot_id"),
        edge_dtype_id=concatenate("edge_dtype_id"),
        edge_layout_id=concatenate("edge_layout_id"),
        edge_rank_id=concatenate("edge_rank_id"),
        edge_alias_id=concatenate("edge_alias_id"),
        edge_dynamic_quality_id=concatenate("edge_dynamic_quality_id"),
        edge_phase_transition_id=concatenate("edge_phase_transition_id"),
        edge_flags=concatenate("edge_flags"),
        u_cont=concatenate("u_cont"),
        hardware_id=concatenate("hardware_id"),
        precision_id=concatenate("precision_id"),
        optimizer_id=concatenate("optimizer_id"),
        optimizer_family_id=concatenate("optimizer_family_id"),
        optimizer_hash_id=concatenate("optimizer_hash_id"),
        scheduler_id=concatenate("scheduler_id"),
        scheduler_family_id=concatenate("scheduler_family_id"),
        scheduler_hash_id=concatenate("scheduler_hash_id"),
        capture_mode_id=concatenate("capture_mode_id"),
        capture_backend_id=concatenate("capture_backend_id"),
        dynamic_shape_id=concatenate("dynamic_shape_id"),
        training_mode_id=concatenate("training_mode_id"),
        quality=concatenate("quality"),
        batch=torch.cat(batches),
        layout=layout,
    )


__all__ = [
    "FeatureLayoutV3",
    "GraphBatchV3",
    "GraphFeaturesV3",
    "NormalizationBlock",
    "NormalizationStatsV3",
    "apply_normalization",
    "batch_graph_features",
    "build_graph_features",
    "feature_layout",
    "fit_normalization",
    "graph_precision_category",
    "sample_cache_key",
    "validate_checkpoint_layout",
]
