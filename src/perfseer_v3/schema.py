"""Generated feature-layout schema for v3 graph tensors and checkpoints."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .baseline import canonical_json
from .graph_ir_v3 import PHASES, TENSOR_ROLES, graph_ir_json_schema
from .op_registry import OperationRegistry
from .training_semantics import (
    OPTIMIZERS,
    OPTIMIZER_FAMILIES,
    OPTIMIZER_HASH_BUCKETS,
    SCHEDULERS,
    SCHEDULER_FAMILIES,
    SCHEDULER_HASH_BUCKETS,
)
from .version import FEATURE_SCHEMA_VERSION, GRAPH_IR_VERSION, OP_REGISTRY_VERSION


NODE_CONTINUOUS_FIELDS: tuple[str, ...] = (
    "input_numel",
    "output_numel",
    "parameter_numel",
    "input_bytes",
    "output_bytes",
    "parameter_bytes",
    "buffer_bytes",
    "flops",
    "macs",
    "bytes_read",
    "bytes_written",
    "arithmetic_intensity_flops_per_byte",
    "estimated_workspace_bytes",
    "saved_for_backward_bytes",
    "optimizer_state_bytes",
    "topological_index",
    "depth",
    "fan_in",
    "fan_out",
    "live_bytes_before",
    "live_bytes_after",
    "input_rank_max",
    "output_rank_max",
    "input_dimension_min",
    "input_dimension_max",
    "input_dimension_mean",
    "output_dimension_min",
    "output_dimension_max",
    "output_dimension_mean",
    "dynamic_dimension_count",
    "tensor_core_aligned_fraction",
    "argument_numeric_count",
    "argument_numeric_sum_abs",
    "argument_numeric_max_abs",
    "critical_path_fraction",
    "mean_input_reuse_distance",
    "max_input_lifetime",
    "flop_estimate_confidence",
    "byte_estimate_confidence",
    "workspace_estimate_confidence",
)
EDGE_CONTINUOUS_FIELDS: tuple[str, ...] = (
    "rank",
    "element_width_bytes",
    "numel",
    "tensor_bytes",
    "first_use_distance",
    "last_use_distance",
    "dimension_min",
    "dimension_max",
    "dimension_mean",
    "stride_min",
    "stride_max",
    "stride_mean",
    "lifetime_distance",
    "producer_fan_out",
)
EDGE_FLAG_FIELDS: tuple[str, ...] = (
    "is_view",
    "is_materialized",
    "is_contiguous",
    "is_phase_boundary",
    "has_symbolic_shape",
)
GLOBAL_CONTINUOUS_FIELDS: tuple[str, ...] = (
    "operation_nodes",
    "tensor_edges",
    "total_flops",
    "total_macs",
    "total_parameter_numel",
    "total_parameter_bytes",
    "total_buffer_bytes",
    "total_activation_bytes",
    "total_saved_for_backward_bytes",
    "total_optimizer_state_bytes",
    "peak_live_activation_bytes",
    "critical_path_length",
    "unknown_operation_fraction",
    "unknown_cost_fraction",
    "unknown_byte_fraction",
    "coarsening_ratio",
    "batch_size",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "loss_scale",
    "activation_checkpointing",
    "optimizer_foreach",
    "optimizer_fused",
    "model_input_numel",
    "model_input_bytes",
    "dynamic_symbol_count",
    "custom_operation_fraction",
    "capture_replay_samples",
    "forward_node_count",
    "loss_node_count",
    "backward_node_count",
    "optimizer_node_count",
    "forward_flops",
    "loss_flops",
    "backward_flops",
    "optimizer_flops",
    "forward_bytes_read",
    "loss_bytes_read",
    "backward_bytes_read",
    "optimizer_bytes_read",
    "forward_bytes_written",
    "loss_bytes_written",
    "backward_bytes_written",
    "optimizer_bytes_written",
    "hardware_memory_bytes",
    "hardware_sm_count",
    "hardware_compute_capability",
    "hardware_memory_bandwidth_bytes_per_second",
    "hardware_peak_flops",
    "total_epochs",
    "current_epoch",
    "steps_per_epoch",
    "total_training_steps",
    "current_training_step",
    "learning_rate_initial",
    "learning_rate_current",
    "learning_rate_min",
    "learning_rate_max",
    "parameter_group_learning_rate_min",
    "parameter_group_learning_rate_max",
    "parameter_group_learning_rate_mean",
    "parameter_group_learning_rate_std",
    "weight_decay",
    "parameter_group_weight_decay_min",
    "parameter_group_weight_decay_max",
    "parameter_group_weight_decay_mean",
    "parameter_group_weight_decay_std",
    "optimizer_momentum",
    "optimizer_dampening",
    "optimizer_beta1",
    "optimizer_beta2",
    "optimizer_beta3",
    "optimizer_epsilon",
    "optimizer_rho",
    "optimizer_alpha",
    "optimizer_trust_coefficient",
    "optimizer_clip_threshold",
    "optimizer_decay_rate",
    "optimizer_ns_steps",
    "optimizer_parameter_group_count",
    "optimizer_component_count",
    "scheduler_warmup_steps",
    "scheduler_warmup_epochs",
    "scheduler_warmup_ratio",
    "scheduler_decay_rate",
    "scheduler_decay_steps",
    "scheduler_patience",
    "scheduler_threshold",
    "scheduler_cosine_cycles",
    "scheduler_polynomial_power",
    "scheduler_cooldown_steps",
    "optimizer_nesterov",
    "optimizer_amsgrad",
    "optimizer_maximize",
    "optimizer_capturable",
    "optimizer_differentiable",
    "optimizer_decoupled_weight_decay",
    "optimizer_relative_step",
    "optimizer_scale_parameter",
    "optimizer_warmup_init",
    "distinct_floating_dtype_count",
    "float32_tensor_fraction",
    "bfloat16_tensor_fraction",
    "float16_tensor_fraction",
    "float8_tensor_fraction",
)
NODE_FLAG_FIELDS: tuple[str, ...] = (
    "in_place",
    "view_only",
    "materializing",
    "reduction",
    "broadcast",
    "random",
    "sparse",
    "quantized",
    "fused",
    "custom",
    "transposed",
    "grouped",
    "adaptive",
    "bidirectional",
    "estimated",
    "unknown",
    "cost_unknown",
    "byte_unknown",
    "mixed_input_dtype",
    "changes_dtype",
    "foreach",
)
QUALITY_FIELDS: tuple[str, ...] = (
    "unknown_operation_fraction",
    "unknown_cost_fraction",
    "unknown_byte_fraction",
    "custom_operation_fraction",
    "strict_capture",
    "exact_backward_capture",
    "replay_validated",
    "dynamic_tensor_fraction",
)
DTYPES: tuple[str, ...] = (
    "unknown",
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "mixed",
)
LAYOUTS: tuple[str, ...] = (
    "unknown",
    "contiguous",
    "channels_last",
    "channels_last_3d",
    "strided",
    "sparse_coo",
    "sparse_csr",
)
OPERATOR_BACKENDS: tuple[str, ...] = (
    "unknown",
    "aten",
    "prims",
    "prim",
    "torch",
    "triton",
    "transformer_engine",
    "pyg",
    "perfseer",
    "custom",
)
FEATURE_QUALITIES: tuple[str, ...] = (
    "unknown",
    "exact_formula",
    "shape_formula",
    "profiled_prior",
    "estimated",
)
EDGE_ALIAS_CLASSES: tuple[str, ...] = (
    "unknown",
    "unique_materialized",
    "aliased_materialized",
    "view",
)
EDGE_DYNAMIC_QUALITIES: tuple[str, ...] = (
    "unknown",
    "concrete",
    "symbolic",
    "estimated",
)
CAPTURE_MODES: tuple[str, ...] = (
    "unknown",
    "strict",
    "non_strict",
    "estimated",
)
CAPTURE_BACKENDS: tuple[str, ...] = (
    "unknown",
    "torch_export",
    "aot_autograd",
    "compiled_autograd",
    "legacy_fx",
    "analytical",
)
DYNAMIC_SHAPE_POLICIES: tuple[str, ...] = (
    "none",
    "constrained",
    "symbolic",
    "unknown",
)
PHASE_TRANSITIONS: tuple[str, ...] = (
    "unknown",
    *(f"{source}->{destination}" for source in PHASES for destination in PHASES),
)
HARDWARE_HASH_BUCKETS = 256
SLOT_BUCKETS = 65

_NODE_IDENTITY_FIELDS = {
    "tensor_core_aligned_fraction",
    "critical_path_fraction",
    "flop_estimate_confidence",
    "byte_estimate_confidence",
    "workspace_estimate_confidence",
}
_GLOBAL_IDENTITY_FIELDS = {
    "unknown_operation_fraction",
    "unknown_cost_fraction",
    "unknown_byte_fraction",
    "coarsening_ratio",
    "custom_operation_fraction",
    "optimizer_momentum",
    "optimizer_dampening",
    "optimizer_beta1",
    "optimizer_beta2",
    "optimizer_beta3",
    "optimizer_rho",
    "optimizer_alpha",
    "optimizer_trust_coefficient",
    "optimizer_clip_threshold",
    "optimizer_decay_rate",
    "scheduler_warmup_ratio",
    "scheduler_decay_rate",
    "scheduler_threshold",
    "optimizer_nesterov",
    "optimizer_amsgrad",
    "optimizer_maximize",
    "optimizer_capturable",
    "optimizer_differentiable",
    "optimizer_decoupled_weight_decay",
    "optimizer_relative_step",
    "optimizer_scale_parameter",
    "optimizer_warmup_init",
    "float32_tensor_fraction",
    "bfloat16_tensor_fraction",
    "float16_tensor_fraction",
    "float8_tensor_fraction",
}
TRANSFORMS: dict[str, str] = {
    **{
        name: ("identity" if name in _NODE_IDENTITY_FIELDS else "log1p_nonnegative")
        for name in NODE_CONTINUOUS_FIELDS
    },
    **{name: "log1p_nonnegative" for name in EDGE_CONTINUOUS_FIELDS},
    **{
        name: ("identity" if name in _GLOBAL_IDENTITY_FIELDS else "log1p_nonnegative")
        for name in GLOBAL_CONTINUOUS_FIELDS
    },
}


def build_feature_schema(
    registry: OperationRegistry,
    *,
    node_continuous_fields: Sequence[str] = NODE_CONTINUOUS_FIELDS,
    edge_continuous_fields: Sequence[str] = EDGE_CONTINUOUS_FIELDS,
    global_continuous_fields: Sequence[str] = GLOBAL_CONTINUOUS_FIELDS,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "graph_ir_version": GRAPH_IR_VERSION,
        "operator_registry_version": OP_REGISTRY_VERSION,
        "operator_registry_sha256": registry.sha256,
        "ordered_layout": {
            "node_continuous": list(node_continuous_fields),
            "edge_continuous": list(edge_continuous_fields),
            "global_continuous": list(global_continuous_fields),
            "node_flags": list(NODE_FLAG_FIELDS),
            "edge_flags": list(EDGE_FLAG_FIELDS),
            "quality": list(QUALITY_FIELDS),
        },
        "categorical_mappings": {
            "operation_family": list(registry.families),
            "phase": list(PHASES),
            "dtype": list(DTYPES),
            "accumulation_dtype": list(DTYPES),
            "layout": list(LAYOUTS),
            "operator_backend": list(OPERATOR_BACKENDS),
            "feature_quality": list(FEATURE_QUALITIES),
            "edge_role": list(TENSOR_ROLES),
            "edge_alias": list(EDGE_ALIAS_CLASSES),
            "edge_dynamic_quality": list(EDGE_DYNAMIC_QUALITIES),
            "phase_transition": list(PHASE_TRANSITIONS),
            "capture_mode": list(CAPTURE_MODES),
            "capture_backend": list(CAPTURE_BACKENDS),
            "optimizer": list(OPTIMIZERS),
            "optimizer_family": list(OPTIMIZER_FAMILIES),
            "optimizer_hash_buckets": OPTIMIZER_HASH_BUCKETS,
            "scheduler": list(SCHEDULERS),
            "scheduler_family": list(SCHEDULER_FAMILIES),
            "scheduler_hash_buckets": SCHEDULER_HASH_BUCKETS,
            "dynamic_shape_policy": list(DYNAMIC_SHAPE_POLICIES),
            "operation_hash_buckets": registry.hash_buckets,
            "operation_overload_hash_buckets": registry.hash_buckets,
            "hardware_hash_buckets": HARDWARE_HASH_BUCKETS,
            "tensor_slot_buckets": SLOT_BUCKETS,
        },
        "normalization": {
            "fit_split": "train_only",
            "transforms": {name: TRANSFORMS.get(name, "identity") for name in TRANSFORMS},
            "clipping": "training_quantiles",
        },
        "graph_ir_json_schema": graph_ir_json_schema(),
    }
    payload["feature_schema_sha256"] = feature_schema_hash(payload)
    return payload


def feature_schema_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("feature_schema_sha256", None)
    return hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()


def validate_feature_schema(payload: Mapping[str, Any], registry: OperationRegistry) -> None:
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature schema version mismatch")
    if payload.get("operator_registry_sha256") != registry.sha256:
        raise ValueError("feature schema registry hash mismatch")
    if payload.get("feature_schema_sha256") != feature_schema_hash(payload):
        raise ValueError("feature schema content hash mismatch")
    layout = payload.get("ordered_layout")
    if not isinstance(layout, Mapping):
        raise ValueError("feature schema ordered_layout is missing")
    for key in (
        "node_continuous",
        "edge_continuous",
        "global_continuous",
        "node_flags",
        "edge_flags",
        "quality",
    ):
        values = list(layout.get(key, ()))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"feature layout {key!r} must be nonempty and unique")


__all__ = [
    "CAPTURE_BACKENDS",
    "CAPTURE_MODES",
    "DTYPES",
    "DYNAMIC_SHAPE_POLICIES",
    "EDGE_ALIAS_CLASSES",
    "EDGE_CONTINUOUS_FIELDS",
    "EDGE_DYNAMIC_QUALITIES",
    "EDGE_FLAG_FIELDS",
    "FEATURE_QUALITIES",
    "GLOBAL_CONTINUOUS_FIELDS",
    "HARDWARE_HASH_BUCKETS",
    "LAYOUTS",
    "NODE_CONTINUOUS_FIELDS",
    "NODE_FLAG_FIELDS",
    "OPERATOR_BACKENDS",
    "OPTIMIZERS",
    "OPTIMIZER_FAMILIES",
    "OPTIMIZER_HASH_BUCKETS",
    "PHASE_TRANSITIONS",
    "QUALITY_FIELDS",
    "SLOT_BUCKETS",
    "SCHEDULERS",
    "SCHEDULER_FAMILIES",
    "SCHEDULER_HASH_BUCKETS",
    "TRANSFORMS",
    "build_feature_schema",
    "feature_schema_hash",
    "validate_feature_schema",
]
