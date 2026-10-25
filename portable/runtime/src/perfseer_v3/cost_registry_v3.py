"""Explicit-unit, confidence-bearing operation cost estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .graph_ir_v3 import Estimate
from .tensor_metadata import flatten_nodes, flatten_tensor_values, tensor_metadata


@dataclass(frozen=True)
class OperationCost:
    input_numel: int
    output_numel: int
    input_bytes: int
    output_bytes: int
    flops: Estimate
    macs: Estimate
    bytes_read: Estimate
    bytes_written: Estimate
    estimated_workspace_bytes: Estimate
    arithmetic_intensity_flops_per_byte: float


def _concrete_shape(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _tensor_values_for_inputs(node: torch.fx.Node) -> tuple[Any, ...]:
    values: list[Any] = []
    for source in (*flatten_nodes(node.args), *flatten_nodes(node.kwargs)):
        values.extend(flatten_tensor_values(source.meta.get("val")))
    return tuple(values)


def _totals(values: tuple[Any, ...]) -> tuple[int, int]:
    numel = 0
    byte_count = 0
    for value in values:
        metadata = tensor_metadata(value)
        numel += metadata.numel or 0
        byte_count += metadata.tensor_bytes or 0
    return numel, byte_count


def _formula(value: float, *, method: str = "shape_formula", confidence: float = 1.0) -> Estimate:
    return Estimate(max(0.0, float(value)), method, confidence)


def _matmul_macs(input_values: tuple[Any, ...], output_values: tuple[Any, ...]) -> int | None:
    if len(input_values) < 2 or not output_values:
        return None
    left = _concrete_shape(input_values[0])
    right = _concrete_shape(input_values[1])
    output = _concrete_shape(output_values[0])
    if left is None or right is None or output is None or not left or not right:
        return None
    if len(left) == 1 and len(right) == 1:
        return left[0]
    k = left[-1]
    if len(right) == 1:
        return math.prod(output) * k
    n = right[-1]
    output_elements = math.prod(output)
    return output_elements * k if n > 0 else None


def _linear_macs(input_values: tuple[Any, ...], output_values: tuple[Any, ...]) -> int | None:
    if len(input_values) < 2 or not output_values:
        return None
    weight = _concrete_shape(input_values[1])
    output = _concrete_shape(output_values[0])
    if weight is None or output is None or len(weight) != 2:
        return None
    return math.prod(output) * weight[1]


def _convolution_macs(
    node: torch.fx.Node,
    input_values: tuple[Any, ...],
    output_values: tuple[Any, ...],
    *,
    raw_target: str,
) -> int | None:
    if len(input_values) < 2 or not output_values:
        return None
    input_shape = _concrete_shape(input_values[0])
    weight = _concrete_shape(input_values[1])
    output = _concrete_shape(output_values[0])
    if input_shape is None or weight is None or output is None or len(weight) < 3:
        return None
    kernel_product = math.prod(weight[2:])
    transposed = "conv_transpose" in raw_target
    groups = 1
    if "conv_transpose" in raw_target and len(node.args) > 6 and isinstance(node.args[6], int):
        groups = max(1, int(node.args[6]))
    elif raw_target == "aten::convolution" and len(node.args) > 8:
        transposed = bool(node.args[6])
        if isinstance(node.args[8], int):
            groups = max(1, int(node.args[8]))
    input_channels_per_group = (
        input_shape[1] // groups if transposed else weight[1]
    )
    return math.prod(output) * input_channels_per_group * kernel_product


def _recurrent_macs(
    input_values: tuple[Any, ...],
    output_values: tuple[Any, ...],
) -> int | None:
    if not input_values or not output_values:
        return None
    output_shape = _concrete_shape(output_values[0])
    if output_shape is None or len(output_shape) < 3:
        return None
    batch_times_steps = output_shape[0] * output_shape[1]
    matrix_elements = 0
    for value in input_values:
        shape = _concrete_shape(value)
        if shape is not None and len(shape) == 2:
            matrix_elements += math.prod(shape)
    if matrix_elements <= 0:
        return None
    return batch_times_steps * matrix_elements


def _pool_kernel_product(node: torch.fx.Node) -> int | None:
    if len(node.args) < 2:
        return None
    kernel = node.args[1]
    if isinstance(kernel, int):
        return kernel
    if isinstance(kernel, (tuple, list)) and all(isinstance(value, int) for value in kernel):
        return math.prod(kernel)
    return None


def estimate_fx_node(
    node: torch.fx.Node,
    *,
    raw_target: str,
    cost_formula: str,
) -> OperationCost:
    input_values = _tensor_values_for_inputs(node)
    output_values = flatten_tensor_values(node.meta.get("val"))
    input_numel, input_bytes = _totals(input_values)
    output_numel, output_bytes = _totals(output_values)
    bytes_read = _formula(input_bytes, method="exact_formula")
    bytes_written = _formula(output_bytes, method="exact_formula")
    macs = Estimate()
    flops = Estimate()
    workspace = _formula(0, method="shape_formula", confidence=0.5)

    if cost_formula == "linear":
        value = _linear_macs(input_values, output_values)
        if value is not None:
            macs = _formula(value)
            bias = output_numel if len(input_values) >= 3 else 0
            flops = _formula(2 * value + bias)
    elif cost_formula in {"matmul", "einsum"}:
        value = _matmul_macs(input_values, output_values)
        if value is not None:
            macs = _formula(value, confidence=1.0 if cost_formula == "matmul" else 0.6)
            flops = _formula(2 * value, confidence=macs.confidence)
    elif cost_formula == "convolution":
        value = _convolution_macs(
            node,
            input_values,
            output_values,
            raw_target=raw_target,
        )
        if value is not None:
            macs = _formula(value)
            bias = output_numel if len(input_values) >= 3 else 0
            flops = _formula(2 * value + bias)
    elif cost_formula == "attention":
        if len(input_values) >= 3:
            q = _concrete_shape(input_values[0])
            k = _concrete_shape(input_values[1])
            v = _concrete_shape(input_values[2])
            if q and k and v and len(q) >= 3:
                batch_heads = math.prod(q[:-2])
                query_length, head_dim = q[-2:]
                key_length = k[-2]
                value_dim = v[-1]
                value = batch_heads * query_length * key_length * (head_dim + value_dim)
                macs = _formula(value, confidence=0.9)
                flops = _formula(2 * value + 5 * batch_heads * query_length * key_length, confidence=0.8)
    elif cost_formula == "recurrent":
        value = _recurrent_macs(input_values, output_values)
        if value is not None:
            macs = _formula(value, confidence=0.8)
            flops = _formula(2 * value + output_numel * 4, confidence=0.7)
    elif cost_formula in {"elementwise", "copy", "dropout"}:
        factor = 4 if "gelu" in raw_target else 1
        if cost_formula == "dropout":
            factor = 2
        flops = _formula(output_numel * factor, confidence=0.8)
    elif cost_formula in {"normalization", "softmax"}:
        flops = _formula(output_numel * (5 if cost_formula == "softmax" else 6), confidence=0.8)
    elif cost_formula in {"reduction", "loss"}:
        flops = _formula(input_numel, confidence=0.8)
    elif cost_formula == "pooling":
        kernel = _pool_kernel_product(node)
        if kernel is not None:
            flops = _formula(output_numel * kernel, confidence=0.9)
    elif cost_formula in {"indexing", "embedding", "view"}:
        flops = _formula(0, method="exact_formula")
    elif cost_formula == "view_or_copy":
        flops = _formula(0, method="shape_formula", confidence=0.5)
    elif cost_formula == "sort_select":
        output_shape = _concrete_shape(output_values[0]) if output_values else None
        if output_shape and output_shape[-1] > 0:
            n = output_shape[-1]
            flops = _formula(output_numel * math.log2(max(2, n)), confidence=0.4)
    elif cost_formula == "resample":
        flops = _formula(output_numel * 4, confidence=0.6)

    traffic = bytes_read.value + bytes_written.value
    intensity = flops.value / traffic if traffic > 0 else 0.0
    return OperationCost(
        input_numel=input_numel,
        output_numel=output_numel,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        flops=flops,
        macs=macs,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        estimated_workspace_bytes=workspace,
        arithmetic_intensity_flops_per_byte=intensity,
    )


__all__ = ["OperationCost", "estimate_fx_node"]
