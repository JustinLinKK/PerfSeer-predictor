"""Canonical NVIDIA hardware profiles and fixed-reference normalization."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .baseline import canonical_json
from .version import HARDWARE_NORMALIZATION_VERSION, HARDWARE_PROFILE_VERSION


_NON_SPECIFIC_IDS = frozenset({"", "unknown", "any", "all", "mixed", "generic", "*"})

# These fields use the units named in the key. Compatibility values are parsed
# version numbers (major + minor / 1000), never categorical hash buckets.
HARDWARE_STATIC_FIELDS: tuple[str, ...] = (
    "memory_bytes",
    "sm_count",
    "compute_capability",
    "memory_bandwidth_bytes_per_second",
    "l2_cache_bytes",
    "peak_fp32_flops",
    "peak_tf32_flops",
    "peak_fp16_flops",
    "peak_bf16_flops",
    "peak_fp8_flops",
    "peak_fp4_flops",
    "tensor_core_generation",
    "pcie_generation",
    "nvlink_bandwidth_bytes_per_second",
    "power_limit_watts",
    "max_core_clock_hz",
    "max_memory_clock_hz",
    "cuda_compatibility",
    "cudnn_compatibility",
    "pytorch_compatibility",
    "triton_compatibility",
    "driver_compatibility",
)

# Forty bounded, cheap signature points spanning launch, bandwidth, kernels,
# optimizers, attention, and allocator/workspace behavior.
HARDWARE_MICROBENCHMARK_FIELDS: tuple[str, ...] = (
    "launch_latency_us_1",
    "launch_latency_us_32",
    "pointwise_gbps_small",
    "pointwise_gbps_medium",
    "pointwise_gbps_large",
    "reduction_gbps_small",
    "reduction_gbps_medium",
    "reduction_gbps_large",
    "host_to_device_gbps",
    "device_to_device_gbps",
    "gemm_fp32_tflops_small",
    "gemm_fp32_tflops_medium",
    "gemm_fp32_tflops_large",
    "gemm_tf32_tflops_small",
    "gemm_tf32_tflops_medium",
    "gemm_tf32_tflops_large",
    "gemm_fp16_tflops_small",
    "gemm_fp16_tflops_medium",
    "gemm_fp16_tflops_large",
    "gemm_bf16_tflops_small",
    "gemm_bf16_tflops_medium",
    "gemm_bf16_tflops_large",
    "conv_fp32_tflops_small",
    "conv_fp32_tflops_large",
    "conv_amp_tflops_small",
    "conv_amp_tflops_large",
    "sdpa_fp32_tflops_short",
    "sdpa_fp32_tflops_long",
    "sdpa_amp_tflops_short",
    "sdpa_amp_tflops_long",
    "adamw_mparams_per_second",
    "sgd_mparams_per_second",
    "layernorm_gbps",
    "softmax_gbps",
    "embedding_glookups_per_second",
    "allocator_allocs_per_second",
    "allocator_fragmentation_ratio",
    "workspace_peak_ratio",
    "mixed_precision_tflops",
    "tensor_core_utilization_ratio",
)

HARDWARE_CONTINUOUS_FIELDS = HARDWARE_STATIC_FIELDS + HARDWARE_MICROBENCHMARK_FIELDS
HARDWARE_MISSING_MASK_FIELDS = tuple(f"{name}_missing" for name in HARDWARE_CONTINUOUS_FIELDS)

_ENVIRONMENT_TO_COMPATIBILITY = {
    "cuda_version": "cuda_compatibility",
    "cudnn_version": "cudnn_compatibility",
    "pytorch_version": "pytorch_compatibility",
    "triton_version": "triton_compatibility",
    "driver_version": "driver_compatibility",
}

# Bounds deliberately cover broad current/future NVIDIA ranges. Values outside
# are clipped and reported, rather than fitted to one GPU's empirical variance.
_LINEAR_BOUNDS: dict[str, tuple[float, float]] = {
    "compute_capability": (5.0, 20.0),
    "tensor_core_generation": (0.0, 10.0),
    "pcie_generation": (0.0, 10.0),
    "cuda_compatibility": (8.0, 30.0),
    "cudnn_compatibility": (7.0, 30.0),
    "pytorch_compatibility": (1.0, 20.0),
    "triton_compatibility": (1.0, 20.0),
    "driver_compatibility": (400.0, 2000.0),
    "allocator_fragmentation_ratio": (0.0, 2.0),
    "workspace_peak_ratio": (0.0, 4.0),
    "tensor_core_utilization_ratio": (0.0, 2.0),
}

_LOG_UPPER_BOUNDS: dict[str, float] = {
    "memory_bytes": float(1024**4),
    "sm_count": 1024.0,
    "memory_bandwidth_bytes_per_second": 20e12,
    "l2_cache_bytes": float(4 * 1024**3),
    "peak_fp32_flops": 10e15,
    "peak_tf32_flops": 20e15,
    "peak_fp16_flops": 40e15,
    "peak_bf16_flops": 40e15,
    "peak_fp8_flops": 80e15,
    "peak_fp4_flops": 160e15,
    "nvlink_bandwidth_bytes_per_second": 10e12,
    "power_limit_watts": 5000.0,
    "max_core_clock_hz": 10e9,
    "max_memory_clock_hz": 20e9,
}

# Public because the scripted model must invert the fixed-reference memory
# transform when forming its explicit predicted-VRAM/capacity OOM feature.
HARDWARE_MEMORY_BYTES_UPPER_BOUND = _LOG_UPPER_BOUNDS["memory_bytes"]


def canonical_hardware_id(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "unknown"


def require_specific_hardware_id(value: Any, *, context: str) -> str:
    hardware_id = canonical_hardware_id(value)
    if hardware_id in _NON_SPECIFIC_IDS:
        raise ValueError(
            f"{context} must identify exactly one concrete GPU type; got {value!r}"
        )
    return hardware_id


def _gpu_model_tokens(value: Any) -> frozenset[str]:
    canonical = canonical_hardware_id(value)
    return frozenset(
        match.group(1)
        for match in re.finditer(
            r"(?:^|_)(rtx_[a-z]?\d{4}(?:_ti)?|gb\d{2,4}|[ahlbpvt]\d{1,4}[a-z]?)(?:_|$)",
            canonical,
        )
    )


def assert_physical_hardware_identity(
    declared_hardware_id: Any,
    physical_device_name: Any,
) -> None:
    """Reject a profile/label run whose declared GPU model is not observable."""

    declared = require_specific_hardware_id(
        declared_hardware_id,
        context="declared target hardware ID",
    )
    physical = str(physical_device_name or "").strip()
    declared_tokens = _gpu_model_tokens(declared)
    physical_tokens = _gpu_model_tokens(physical)
    if not declared_tokens:
        raise ValueError(
            f"declared target hardware ID {declared!r} has no recognizable GPU model token"
        )
    if not physical_tokens or declared_tokens.isdisjoint(physical_tokens):
        raise ValueError(
            f"declared target hardware {declared!r} does not match physical device {physical!r}"
        )


def graph_hardware_id(metadata: Mapping[str, Any]) -> str:
    return canonical_hardware_id(
        metadata.get("target_hardware_id", metadata.get("hardware_id", "unknown"))
    )


def _optional_nonnegative(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hardware field {field_name!r} must be numeric or null") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"hardware field {field_name!r} must be finite and nonnegative")
    return number


def _compatibility_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    matches = re.findall(r"\d+", str(value))
    if not matches:
        return None
    major = float(matches[0])
    minor = float(matches[1]) if len(matches) > 1 else 0.0
    patch = float(matches[2]) if len(matches) > 2 else 0.0
    return major + minor / 1000.0 + patch / 1_000_000.0


@dataclass(frozen=True)
class HardwareProfileV3:
    hardware_id: str
    static: dict[str, float | None]
    microbenchmarks: dict[str, float | None]
    environment: dict[str, str | None]
    profile_version: str = HARDWARE_PROFILE_VERSION

    def validate(self, *, require_complete_signature: bool = False) -> None:
        hardware_id = canonical_hardware_id(self.hardware_id)
        if hardware_id != "unknown":
            require_specific_hardware_id(hardware_id, context="hardware profile hardware_id")
        if self.profile_version != HARDWARE_PROFILE_VERSION:
            raise ValueError("hardware profile version mismatch")
        unknown_static = set(self.static) - set(HARDWARE_STATIC_FIELDS)
        unknown_benchmarks = set(self.microbenchmarks) - set(HARDWARE_MICROBENCHMARK_FIELDS)
        unknown_environment = set(self.environment) - set(_ENVIRONMENT_TO_COMPATIBILITY)
        if unknown_static or unknown_benchmarks or unknown_environment:
            raise ValueError(
                "hardware profile contains unknown fields: "
                + ", ".join(
                    sorted(unknown_static | unknown_benchmarks | unknown_environment)
                )
            )
        for name, value in (*self.static.items(), *self.microbenchmarks.items()):
            _optional_nonnegative(value, field_name=name)
        compute_capability = self.static.get("compute_capability")
        if compute_capability is not None and not 5.0 <= float(compute_capability) <= 20.0:
            raise ValueError("hardware compute_capability is outside supported NVIDIA bounds")
        if require_complete_signature:
            missing = [
                name
                for name in HARDWARE_MICROBENCHMARK_FIELDS
                if self.microbenchmarks.get(name) is None
            ]
            if missing:
                raise ValueError(
                    f"hardware profile is missing {len(missing)} required microbenchmarks"
                )
            missing_static = [
                name
                for name in ("memory_bytes", "sm_count", "compute_capability")
                if self.static.get(name) is None or float(self.static[name]) <= 0
            ]
            if missing_static:
                raise ValueError(
                    "complete hardware profile lacks core physical fields: "
                    + ", ".join(missing_static)
                )
            missing_environment = [
                name
                for name in (
                    "cuda_version",
                    "cudnn_version",
                    "pytorch_version",
                    "driver_version",
                )
                if _compatibility_number(self.environment.get(name)) is None
            ]
            if missing_environment:
                raise ValueError(
                    "complete hardware profile lacks environment identity: "
                    + ", ".join(missing_environment)
                )

    @property
    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "profile_version": self.profile_version,
            "hardware_id": canonical_hardware_id(self.hardware_id),
            "static": {
                name: _optional_nonnegative(self.static.get(name), field_name=name)
                for name in HARDWARE_STATIC_FIELDS
            },
            "microbenchmarks": {
                name: _optional_nonnegative(self.microbenchmarks.get(name), field_name=name)
                for name in HARDWARE_MICROBENCHMARK_FIELDS
            },
            "environment": {
                name: (None if self.environment.get(name) in (None, "") else str(self.environment[name]))
                for name in sorted(_ENVIRONMENT_TO_COMPATIBILITY)
            },
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical_payload).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardwareProfileV3":
        profile = cls(
            hardware_id=canonical_hardware_id(value.get("hardware_id")),
            static=dict(value.get("static") or {}),
            microbenchmarks=dict(value.get("microbenchmarks") or {}),
            environment=dict(value.get("environment") or {}),
            profile_version=str(value.get("profile_version", HARDWARE_PROFILE_VERSION)),
        )
        profile.validate()
        return profile


@dataclass(frozen=True)
class HardwareNormalizationResultV3:
    values: tuple[float, ...]
    missing_mask: tuple[float, ...]
    clipped_low: int
    clipped_high: int
    policy_version: str
    policy_sha256: str

    @property
    def clip_frequency(self) -> float:
        return (self.clipped_low + self.clipped_high) / max(1, len(self.values))


@dataclass(frozen=True)
class HardwareNormalizationPolicyV3:
    policy_version: str = HARDWARE_NORMALIZATION_VERSION

    @property
    def sha256(self) -> str:
        payload = {
            "policy_version": self.policy_version,
            "ordered_fields": HARDWARE_CONTINUOUS_FIELDS,
            "linear_bounds": _LINEAR_BOUNDS,
            "log_upper_bounds": _LOG_UPPER_BOUNDS,
            "microbenchmark_default_transform": "log1p_over_log1p_1e9",
            "missing_value": 0.0,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def normalize(self, profile: HardwareProfileV3) -> HardwareNormalizationResultV3:
        if self.policy_version != HARDWARE_NORMALIZATION_VERSION:
            raise ValueError("hardware normalization version mismatch")
        payload = profile.canonical_payload
        source = {**payload["static"], **payload["microbenchmarks"]}
        for environment_name, field_name in _ENVIRONMENT_TO_COMPATIBILITY.items():
            source[field_name] = _compatibility_number(payload["environment"][environment_name])
        values: list[float] = []
        missing: list[float] = []
        clipped_low = 0
        clipped_high = 0
        for name in HARDWARE_CONTINUOUS_FIELDS:
            raw = source.get(name)
            if raw is None:
                values.append(0.0)
                missing.append(1.0)
                continue
            number = float(raw)
            missing.append(0.0)
            if name in _LINEAR_BOUNDS:
                low, high = _LINEAR_BOUNDS[name]
                clipped_low += int(number < low)
                clipped_high += int(number > high)
                number = min(high, max(low, number))
                values.append((number - low) / (high - low))
            else:
                upper = _LOG_UPPER_BOUNDS.get(name, 1e9)
                clipped_high += int(number > upper)
                number = min(upper, max(0.0, number))
                values.append(math.log1p(number) / math.log1p(upper))
        return HardwareNormalizationResultV3(
            values=tuple(values),
            missing_mask=tuple(missing),
            clipped_low=clipped_low,
            clipped_high=clipped_high,
            policy_version=self.policy_version,
            policy_sha256=self.sha256,
        )


def hardware_profile_from_metadata(metadata: Mapping[str, Any]) -> HardwareProfileV3:
    raw_profile = metadata.get("hardware_profile")
    if isinstance(raw_profile, Mapping):
        profile = HardwareProfileV3.from_dict(raw_profile)
        expected = graph_hardware_id(metadata)
        if expected != "unknown" and profile.hardware_id != expected:
            raise ValueError("graph hardware ID and embedded hardware profile disagree")
        return profile
    legacy_static = metadata.get("hardware_features")
    static = dict(legacy_static) if isinstance(legacy_static, Mapping) else {}
    if "peak_flops" in static and "peak_fp32_flops" not in static:
        static["peak_fp32_flops"] = static.pop("peak_flops")
    environment_raw = metadata.get("hardware_environment")
    environment = dict(environment_raw) if isinstance(environment_raw, Mapping) else {}
    benchmark_raw = metadata.get("hardware_microbenchmarks")
    microbenchmarks = dict(benchmark_raw) if isinstance(benchmark_raw, Mapping) else {}
    profile = HardwareProfileV3(
        hardware_id=graph_hardware_id(metadata),
        static=static,
        microbenchmarks=microbenchmarks,
        environment=environment,
    )
    profile.validate()
    return profile


def hardware_profile_json_schema() -> dict[str, Any]:
    numeric_or_null = {"type": ["number", "null"], "minimum": 0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": HARDWARE_PROFILE_VERSION,
        "type": "object",
        "required": ["profile_version", "hardware_id", "static", "microbenchmarks", "environment"],
        "properties": {
            "profile_version": {"const": HARDWARE_PROFILE_VERSION},
            "hardware_id": {"type": "string", "minLength": 1},
            "static": {
                "type": "object",
                "properties": {name: numeric_or_null for name in HARDWARE_STATIC_FIELDS},
                "additionalProperties": False,
            },
            "microbenchmarks": {
                "type": "object",
                "properties": {name: numeric_or_null for name in HARDWARE_MICROBENCHMARK_FIELDS},
                "additionalProperties": False,
            },
            "environment": {
                "type": "object",
                "properties": {
                    name: {"type": ["string", "null"]}
                    for name in _ENVIRONMENT_TO_COMPATIBILITY
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


__all__ = [
    "HARDWARE_CONTINUOUS_FIELDS",
    "HARDWARE_MEMORY_BYTES_UPPER_BOUND",
    "HARDWARE_MICROBENCHMARK_FIELDS",
    "HARDWARE_MISSING_MASK_FIELDS",
    "HARDWARE_STATIC_FIELDS",
    "HardwareNormalizationPolicyV3",
    "HardwareNormalizationResultV3",
    "HardwareProfileV3",
    "assert_physical_hardware_identity",
    "canonical_hardware_id",
    "graph_hardware_id",
    "hardware_profile_from_metadata",
    "hardware_profile_json_schema",
    "require_specific_hardware_id",
]
