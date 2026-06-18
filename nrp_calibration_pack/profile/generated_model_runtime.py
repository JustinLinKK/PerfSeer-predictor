"""Shared runtime for generated calibration models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _last_dim(mem: dict[str, Any], default: int = 8) -> int:
    return _positive(mem.get("output_features"), _positive(mem.get("output_channels"), default))


def _input_last_dim(mem: dict[str, Any], default: int = 8) -> int:
    return _positive(mem.get("input_features"), _positive(mem.get("input_channels"), default))


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scale = max(q.size(-1), 1) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = self.out(torch.matmul(attn, v))
        return out.squeeze(1) if squeeze else out


class GraphMessageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        if adjacency is not None:
            x = torch.bmm(adjacency.float(), x.float())
        out = self.lin(x)
        return out.squeeze(1) if squeeze else out


class DetectorHead(nn.Module):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        elif x.dim() > 2:
            x = x.flatten(1)
        return self.fc(x)


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
        if op == "DepthwiseConv":
            in_channels = _positive(mem.get("input_channels"))
            out_channels = _positive(mem.get("output_channels"), in_channels)
            groups = in_channels if out_channels % in_channels == 0 else 1
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
        if op == "ConvTranspose":
            return nn.ConvTranspose2d(
                _positive(mem.get("input_channels")),
                _positive(mem.get("output_channels")),
                kernel_size=_positive(args.get("conv_kernel_size")),
                stride=_positive(args.get("conv_stride")),
                padding=max(_as_int(args.get("conv_padding")), 0),
                bias=bool(_as_int(args.get("conv_bias"))),
            )
        if op == "Relu":
            return nn.ReLU()
        if op == "Gelu":
            return nn.GELU()
        if op == "Silu":
            return nn.SiLU()
        if op == "Sigmoid":
            return nn.Sigmoid()
        if op == "Softmax":
            return nn.Softmax(dim=_as_int(args.get("softmax_dim"), -1))
        if op == "BatchNormalization":
            return nn.BatchNorm2d(_positive(mem.get("output_channels")))
        if op == "LayerNormalization":
            return nn.LayerNorm(_last_dim(mem))
        if op == "GroupNormalization":
            channels = _positive(mem.get("output_channels"), _last_dim(mem))
            groups = max(1, min(_positive(args.get("norm_groups"), 1), channels))
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)
        if op == "Embedding":
            return nn.Embedding(_positive(args.get("vocab_size"), 30522), _last_dim(mem))
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
        if op == "Upsample":
            return nn.Upsample(scale_factor=_positive(args.get("scale_factor"), 2), mode="nearest")
        if op == "Flatten":
            return nn.Flatten()
        if op == "Gemm":
            return nn.Linear(
                _positive(args.get("linear_in_features")),
                _positive(args.get("linear_out_features")),
                bias=bool(_as_int(args.get("linear_bias"))),
            )
        if op in {"MatMul", "Bmm", "TabularFeature"}:
            return nn.Linear(_input_last_dim(mem), _last_dim(mem))
        if op in {"Attention", "MultiHeadAttention"}:
            return SimpleSelfAttention(_last_dim(mem))
        if op == "RNN":
            return nn.RNN(_input_last_dim(mem), _last_dim(mem), batch_first=True)
        if op == "GRU":
            return nn.GRU(_input_last_dim(mem), _last_dim(mem), batch_first=True)
        if op == "LSTM":
            return nn.LSTM(_input_last_dim(mem), _last_dim(mem), batch_first=True)
        if op in {"GraphMessage", "GraphAttention"}:
            return GraphMessageLayer(_input_last_dim(mem), _last_dim(mem))
        if op == "DetectorHead":
            return DetectorHead(_positive(mem.get("input_channels"), _input_last_dim(mem)), _last_dim(mem))
        if op == "SegmentationHead":
            return nn.Conv2d(_positive(mem.get("input_channels")), _positive(mem.get("output_channels"), 1), kernel_size=1)
        if op in {"Add", "Concat", "Mul", "Reshape", "Transpose"}:
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

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = tuple(inputs[0])
        if not inputs:
            raise ValueError("GraphModel.forward requires at least one input tensor")
        values: dict[int, torch.Tensor] = {}
        for spec in self.node_specs:
            node_id = int(spec["id"])
            op = spec["type"]
            preds = [int(pred) for pred in spec.get("preds", [])]
            node_inputs = [values[pred] for pred in preds] if preds else [inputs[_as_int(spec.get("input_index"), 0)]]
            if op == "Add":
                out = node_inputs[0]
                for tensor in node_inputs[1:]:
                    out = out + tensor
            elif op == "Mul":
                out = node_inputs[0]
                for tensor in node_inputs[1:]:
                    out = out * tensor
            elif op == "Concat":
                dim = _as_int(spec.get("args", {}).get("concat_dim"), 1 if node_inputs[0].dim() == 4 else -1)
                out = torch.cat(node_inputs, dim=dim)
            elif op == "Gemm":
                out = self._gemm(node_id, node_inputs[0])
            elif op == "Embedding":
                out = self.layers[str(node_id)](node_inputs[0].long())
            elif op in {"RNN", "GRU", "LSTM"}:
                out, _hidden = self.layers[str(node_id)](node_inputs[0].float())
            elif op in {"GraphMessage", "GraphAttention"}:
                adjacency_index = _as_int(spec.get("args", {}).get("adjacency_input_index"), 1)
                adjacency = inputs[adjacency_index] if adjacency_index < len(inputs) else None
                out = self.layers[str(node_id)](node_inputs[0], adjacency)
            elif op == "Reshape":
                out = node_inputs[0].reshape(node_inputs[0].size(0), -1)
            elif op == "Transpose":
                tensor = node_inputs[0]
                out = tensor.transpose(1, 2) if tensor.dim() >= 3 else tensor
            elif op == "MatMul":
                out = self.layers[str(node_id)](node_inputs[0].float())
            elif op == "Bmm":
                out = self.layers[str(node_id)](node_inputs[0].float())
            else:
                out = self.layers[str(node_id)](node_inputs[0])
            values[node_id] = out
        return values[int(self.node_specs[-1]["id"])]

    def _gemm(self, node_id: int, tensor: torch.Tensor) -> torch.Tensor:
        layer = self.layers[str(node_id)]
        if node_id in self.flatten_before_gemm:
            tensor = torch.flatten(tensor, 1)
        return layer(tensor)
