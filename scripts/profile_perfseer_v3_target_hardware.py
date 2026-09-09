#!/usr/bin/env python3
"""Produce the versioned 40-point PerfSeer target hardware signature."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.hardware import (
    HARDWARE_MICROBENCHMARK_FIELDS,
    HardwareNormalizationPolicyV3,
    HardwareProfileV3,
    assert_physical_hardware_identity,
    canonical_hardware_id,
)


def _driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cudnn_version() -> str | None:
    raw = torch.backends.cudnn.version()
    if raw is None or raw <= 0:
        return None
    if raw >= 90_000:
        major, remainder = divmod(raw, 10_000)
    else:
        major, remainder = divmod(raw, 1_000)
    minor, patch = divmod(remainder, 100)
    return f"{major}.{minor}.{patch}"


def _positive_or_none(value: object, *, scale: float = 1.0) -> float | None:
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _timed(operation: Callable[[], None], *, iterations: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return max(1e-9, start.elapsed_time(end) / 1000.0 / iterations)


def _pointwise(size: int, iterations: int) -> float:
    x = torch.randn(size, device="cuda")
    seconds = _timed(lambda: torch.sin(x).add_(x), iterations=iterations)
    return 3.0 * x.numel() * x.element_size() / seconds / 1e9


def _reduction(size: int, iterations: int) -> float:
    x = torch.randn(size, device="cuda")
    seconds = _timed(lambda: torch.sum(x), iterations=iterations)
    return x.numel() * x.element_size() / seconds / 1e9


def _gemm(size: int, dtype: torch.dtype, iterations: int) -> float:
    x = torch.randn((size, size), device="cuda", dtype=dtype)
    y = torch.randn((size, size), device="cuda", dtype=dtype)
    seconds = _timed(lambda: torch.mm(x, y), iterations=iterations)
    return 2.0 * size**3 / seconds / 1e12


def _convolution(size: int, dtype: torch.dtype, iterations: int) -> float:
    x = torch.randn((8, 32, size, size), device="cuda", dtype=dtype)
    weight = torch.randn((64, 32, 3, 3), device="cuda", dtype=dtype)
    seconds = _timed(lambda: F.conv2d(x, weight, padding=1), iterations=iterations)
    flops = 2.0 * 8 * size * size * 64 * 32 * 9
    return flops / seconds / 1e12


def _attention(length: int, dtype: torch.dtype, iterations: int) -> float:
    q = torch.randn((4, 8, length, 64), device="cuda", dtype=dtype)
    seconds = _timed(
        lambda: F.scaled_dot_product_attention(q, q, q), iterations=iterations
    )
    flops = 4.0 * 4 * 8 * length * length * 64
    return flops / seconds / 1e12


def _optimizer_throughput(kind: str, iterations: int) -> float:
    parameters = torch.randn(4_000_000, device="cuda")
    gradient = torch.randn_like(parameters)
    state = torch.zeros_like(parameters)
    if kind == "adamw":
        operation = lambda: (
            state.mul_(0.9).add_(gradient, alpha=0.1),
            parameters.addcdiv_(state, state.square().sqrt().add_(1e-8), value=-1e-3),
        )
    else:
        operation = lambda: parameters.add_(gradient, alpha=-1e-3)
    seconds = _timed(operation, iterations=iterations)
    return parameters.numel() / seconds / 1e6


def _layernorm_bandwidth(rows: int, width: int, iterations: int) -> float:
    value = torch.randn((rows, width), device="cuda")
    seconds = _timed(
        lambda: F.layer_norm(value, (width,)),
        iterations=iterations,
    )
    return 2.0 * value.numel() * value.element_size() / seconds / 1e9


def _softmax_bandwidth(rows: int, width: int, iterations: int) -> float:
    value = torch.randn((rows, width), device="cuda")
    seconds = _timed(lambda: torch.softmax(value, dim=-1), iterations=iterations)
    return 2.0 * value.numel() * value.element_size() / seconds / 1e9


def _collect_signature(iterations: int) -> dict[str, float]:
    signature: dict[str, float] = {}
    signature["launch_latency_us_1"] = _timed(
        lambda: torch.empty(1, device="cuda").add_(1), iterations=iterations
    ) * 1e6
    signature["launch_latency_us_32"] = _timed(
        lambda: torch.empty(32, device="cuda").add_(1), iterations=iterations
    ) * 1e6
    for label, size in (("small", 1 << 16), ("medium", 1 << 20), ("large", 1 << 24)):
        signature[f"pointwise_gbps_{label}"] = _pointwise(size, iterations)
        signature[f"reduction_gbps_{label}"] = _reduction(size, iterations)

    # Host-to-device is synchronized wall time; D2D uses the same CUDA timer as
    # kernels. Pinned host allocation is deliberately outside the timed region.
    host = torch.randn(1 << 22, pin_memory=True)
    device = torch.empty_like(host, device="cuda")
    seconds = _timed(lambda: device.copy_(host, non_blocking=True), iterations=iterations)
    signature["host_to_device_gbps"] = host.numel() * host.element_size() / seconds / 1e9
    device_2 = torch.empty_like(device)
    seconds = _timed(lambda: device_2.copy_(device), iterations=iterations)
    signature["device_to_device_gbps"] = device.numel() * device.element_size() / seconds / 1e9

    sizes = {"small": 128, "medium": 512, "large": 1024}
    for precision, dtype in (
        ("fp32", torch.float32),
        ("tf32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        try:
            for label, size in sizes.items():
                signature[f"gemm_{precision}_tflops_{label}"] = _gemm(
                    size, dtype, iterations
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    signature["conv_fp32_tflops_small"] = _convolution(16, torch.float32, iterations)
    signature["conv_fp32_tflops_large"] = _convolution(64, torch.float32, iterations)
    signature["conv_amp_tflops_small"] = _convolution(16, torch.float16, iterations)
    signature["conv_amp_tflops_large"] = _convolution(64, torch.float16, iterations)
    signature["sdpa_fp32_tflops_short"] = _attention(64, torch.float32, iterations)
    signature["sdpa_fp32_tflops_long"] = _attention(256, torch.float32, iterations)
    signature["sdpa_amp_tflops_short"] = _attention(64, torch.float16, iterations)
    signature["sdpa_amp_tflops_long"] = _attention(256, torch.float16, iterations)
    signature["adamw_mparams_per_second"] = _optimizer_throughput("adamw", iterations)
    signature["sgd_mparams_per_second"] = _optimizer_throughput("sgd", iterations)
    signature["layernorm_gbps"] = _layernorm_bandwidth(1 << 14, 256, iterations)
    signature["softmax_gbps"] = _softmax_bandwidth(1 << 14, 256, iterations)

    table = torch.randn((1 << 17, 64), device="cuda")
    indices = torch.randint(0, table.size(0), (1 << 18,), device="cuda")
    seconds = _timed(lambda: F.embedding(indices, table), iterations=iterations)
    signature["embedding_glookups_per_second"] = indices.numel() / seconds / 1e9
    seconds = _timed(
        lambda: torch.empty(1 << 20, device="cuda"), iterations=iterations
    )
    signature["allocator_allocs_per_second"] = 1.0 / seconds
    reserved = float(torch.cuda.memory_reserved())
    allocated = float(torch.cuda.memory_allocated())
    signature["allocator_fragmentation_ratio"] = max(0.0, (reserved - allocated) / max(1.0, reserved))
    torch.cuda.reset_peak_memory_stats()
    workspace = torch.empty(1 << 24, device="cuda")
    del workspace
    signature["workspace_peak_ratio"] = torch.cuda.max_memory_reserved() / max(
        1.0, torch.cuda.get_device_properties(0).total_memory
    )
    signature["mixed_precision_tflops"] = _gemm(1024, torch.float16, iterations)
    fp32 = max(1e-9, signature["gemm_fp32_tflops_large"])
    signature["tensor_core_utilization_ratio"] = signature["mixed_precision_tflops"] / fp32
    if set(signature) != set(HARDWARE_MICROBENCHMARK_FIELDS):
        missing = set(HARDWARE_MICROBENCHMARK_FIELDS) - set(signature)
        extra = set(signature) - set(HARDWARE_MICROBENCHMARK_FIELDS)
        raise RuntimeError(f"hardware signature field mismatch: missing={missing}, extra={extra}")
    return signature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("target hardware profiling requires a CUDA-capable PyTorch runtime")
    properties = torch.cuda.get_device_properties(0)
    hardware_id = canonical_hardware_id(args.hardware_id)
    assert_physical_hardware_identity(hardware_id, properties.name)
    static = {
        "memory_bytes": float(properties.total_memory),
        "sm_count": float(properties.multi_processor_count),
        "compute_capability": float(properties.major) + float(properties.minor) / 10.0,
        "l2_cache_bytes": _positive_or_none(
            getattr(properties, "L2_cache_size", None)
        ),
        "max_core_clock_hz": _positive_or_none(
            getattr(properties, "clock_rate", None), scale=1000.0
        ),
        "max_memory_clock_hz": _positive_or_none(
            getattr(properties, "memory_clock_rate", None), scale=1000.0
        ),
    }
    environment = {
        "cuda_version": torch.version.cuda,
        "cudnn_version": _cudnn_version(),
        "pytorch_version": torch.__version__,
        "triton_version": _package_version("triton"),
        "driver_version": _driver_version(),
    }
    profile = HardwareProfileV3(
        hardware_id=hardware_id,
        static=static,
        microbenchmarks=_collect_signature(args.iterations),
        environment=environment,
    )
    profile.validate(require_complete_signature=True)
    normalization = HardwareNormalizationPolicyV3().normalize(profile)
    payload = {
        **profile.canonical_payload,
        "hardware_profile_sha256": profile.sha256,
        "hardware_normalization_version": normalization.policy_version,
        "hardware_normalization_sha256": normalization.policy_sha256,
        "normalized_clip_frequency": normalization.clip_frequency,
        "physical_device_name": properties.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "hardware_profile_sha256": profile.sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
