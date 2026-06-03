"""Shared runtime for generated calibration models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _positive(value: Any, default: int = 1) -> int:
    out = _as_int(value, default)
    return out if out > 0 else default


class GraphModel(nn.Module):
    """Executable PyTorch module reconstructed from PerfSeer graph specs."""

    def __init__(self, node_specs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.node_specs = node_specs
        self.layers = nn.ModuleDict()
        self.flatten_before_gemm: set[int] = set()
        for spec in node_specs:
            if spec["type"] == "Gemm" and self._should_flatten_before_gemm(spec):
                self.flatten_before_gemm.add(int(spec["id"]))
            layer = self._make_layer(spec)
            if layer is not None:
                self.layers[str(spec["id"])] = layer

    def _make_layer(self, spec: dict[str, Any]) -> nn.Module | None:
        op = spec["type"]
        args = spec.get("args", {})
        mem = spec.get("memory_info", {})
        if op == "Conv":
            in_channels = _positive(mem.get("input_channels"))
            out_channels = _positive(mem.get("output_channels"))
            groups = _positive(args.get("conv_groups"), 1)
            if in_channels % groups != 0 or out_channels % groups != 0:
                groups = 1
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=_positive(args.get("conv_kernel_size")),
                stride=_positive(args.get("conv_stride")),
                padding=max(_as_int(args.get("conv_padding")), 0),
                dilation=_positive(args.get("conv_dilation")),
                groups=groups,
                bias=bool(_as_int(args.get("conv_bias"))),
            )
        if op == "Relu":
            return nn.ReLU()
        if op == "BatchNormalization":
            return nn.BatchNorm2d(_positive(mem.get("output_channels")))
        if op == "AveragePool":
            kernel = _positive(args.get("pool_kernel_size"))
            stride = _positive(args.get("pool_stride"), kernel)
            return nn.AvgPool2d(
                kernel_size=kernel,
                stride=stride,
                padding=max(_as_int(args.get("pool_padding")), 0),
                ceil_mode=bool(_as_int(args.get("pool_ceil_mode"))),
            )
        if op == "MaxPool":
            kernel = _positive(args.get("pool_kernel_size"))
            stride = _positive(args.get("pool_stride"), kernel)
            return nn.MaxPool2d(
                kernel_size=kernel,
                stride=stride,
                padding=max(_as_int(args.get("pool_padding")), 0),
                ceil_mode=bool(_as_int(args.get("pool_ceil_mode"))),
            )
        if op == "GlobalAveragePool":
            return nn.AdaptiveAvgPool2d((1, 1))
        if op == "Flatten":
            return nn.Flatten()
        if op == "Gemm":
            return nn.Linear(
                _positive(args.get("linear_in_features")),
                _positive(args.get("linear_out_features")),
                bias=bool(_as_int(args.get("linear_bias"))),
            )
        if op in {"Add", "Concat"}:
            return None
        raise ValueError(f"unsupported generated op {op!r}")

    def _should_flatten_before_gemm(self, spec: dict[str, Any]) -> bool:
        args = spec.get("args", {})
        mem = spec.get("memory_info", {})
        in_features = _positive(args.get("linear_in_features"))
        channels = _positive(mem.get("input_channels"))
        height = _positive(mem.get("input_h"))
        width = _positive(mem.get("input_w"))
        return channels * height * width == in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: dict[int, torch.Tensor] = {}
        for spec in self.node_specs:
            node_id = int(spec["id"])
            op = spec["type"]
            preds = [int(pred) for pred in spec.get("preds", [])]
            inputs = [values[pred] for pred in preds] if preds else [x]
            if op == "Add":
                out = inputs[0]
                for tensor in inputs[1:]:
                    out = out + tensor
            elif op == "Concat":
                out = torch.cat(inputs, dim=1)
            elif op == "Gemm":
                out = self._gemm(node_id, inputs[0])
            else:
                out = self.layers[str(node_id)](inputs[0])
            values[node_id] = out
        return values[int(self.node_specs[-1]["id"])]

    def _gemm(self, node_id: int, tensor: torch.Tensor) -> torch.Tensor:
        layer = self.layers[str(node_id)]
        if node_id in self.flatten_before_gemm:
            tensor = torch.flatten(tensor, 1)
        return layer(tensor)
