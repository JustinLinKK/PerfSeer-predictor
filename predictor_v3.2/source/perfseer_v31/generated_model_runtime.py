"""Shared runtime for generated calibration models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TE_LOW_PRECISION_STRUCTURAL_OPS = {
    "Add",
    "Concat",
    "Flatten",
    "Gelu",
    "Mul",
    "Relu",
    "Reshape",
    "Silu",
    "Softmax",
    "Transpose",
}
TE_LOW_PRECISION_LAYER_OPS = {
    "Attention",
    "Bmm",
    "Gemm",
    "LayerNormalization",
    "MatMul",
    "MultiHeadAttention",
    "TabularFeature",
}
TE_LOW_PRECISION_OPS = TE_LOW_PRECISION_STRUCTURAL_OPS | TE_LOW_PRECISION_LAYER_OPS


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


def _leading_dim(mem: dict[str, Any], last_dim: int, *, field: str = "input_size") -> int:
    batch = _positive(mem.get("batch_size"), 1)
    rank = _as_int(mem.get("rank"), 0)
    if rank == 3:
        seq = _positive(mem.get("sequence_length"), 0)
        if seq > 0:
            return max(1, batch * seq)
    if rank == 4:
        spatial = _positive(mem.get("spatial_area"), 0)
        if spatial <= 0:
            spatial = _positive(mem.get("input_h"), 1) * _positive(mem.get("input_w"), 1)
        return max(1, batch * spatial)
    size = _positive(mem.get(field), last_dim)
    return max(1, size // max(last_dim, 1))


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


class TransformerEngineSelfAttention(nn.Module):
    def __init__(self, te: Any, dim: int, params_dtype: torch.dtype, device: torch.device | str) -> None:
        super().__init__()
        self.attn = te.MultiheadAttention(
            hidden_size=dim,
            num_attention_heads=1,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            qkv_format="bshd",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        out = self.attn(x, attn_mask_type="no_mask")
        tensor = out[0] if isinstance(out, tuple) else out
        return tensor.squeeze(1) if squeeze else tensor


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
        self.transformer_engine_enabled = False
        self.layers = nn.ModuleDict()
        self.flatten_before_gemm: set[int] = set()
        for spec in node_specs:
            if spec["type"] == "Gemm" and self._should_flatten_before_gemm(spec):
                self.flatten_before_gemm.add(int(spec["id"]))
            layer = self._make_layer(spec)
            if layer is not None:
                self.layers[str(spec["id"])] = layer

    def low_precision_unsupported_reasons(self, precision_config: str) -> list[str]:
        reasons: list[str] = []
        is_nvfp4 = precision_config == "nvfp4_te"
        feature_multiple = 32 if is_nvfp4 else 16
        leading_multiple = 32 if is_nvfp4 else 16
        min_leading = 32 if is_nvfp4 else 16
        for spec in self.node_specs:
            op = str(spec["type"])
            node_id = int(spec["id"])
            if op not in TE_LOW_PRECISION_OPS:
                reasons.append(f"node {node_id} {op} is not TE low-precision safe")
                continue
            if op in {"Gemm", "MatMul", "Bmm", "TabularFeature"}:
                in_features, out_features = self._linear_dims(spec)
                if in_features % feature_multiple != 0 or out_features % feature_multiple != 0:
                    reasons.append(
                        f"node {node_id} {op} dimensions {in_features}->{out_features} are not divisible by {feature_multiple}"
                    )
                leading = _leading_dim(spec.get("memory_info", {}), in_features)
                if leading % leading_multiple != 0 or leading < min_leading:
                    reasons.append(
                        f"node {node_id} {op} leading dimension product {leading} must be divisible by {leading_multiple} and >= {min_leading}"
                    )
            if op == "LayerNormalization":
                hidden = _last_dim(spec.get("memory_info", {}))
                leading = _leading_dim(spec.get("memory_info", {}), hidden)
                if hidden % feature_multiple != 0:
                    reasons.append(f"node {node_id} {op} hidden size {hidden} is not divisible by {feature_multiple}")
                if leading % leading_multiple != 0 or leading < min_leading:
                    reasons.append(
                        f"node {node_id} {op} leading dimension product {leading} must be divisible by {leading_multiple} and >= {min_leading}"
                    )
            if op in {"Attention", "MultiHeadAttention"}:
                mem = spec.get("memory_info", {})
                hidden = _last_dim(mem)
                if hidden % feature_multiple != 0:
                    reasons.append(f"node {node_id} {op} hidden size {hidden} is not divisible by {feature_multiple}")
                leading = _leading_dim(mem, hidden)
                if leading % leading_multiple != 0 or leading < min_leading:
                    reasons.append(
                        f"node {node_id} {op} leading dimension product {leading} must be divisible by {leading_multiple} and >= {min_leading}"
                    )
        return reasons

    def enable_transformer_engine(
        self,
        precision_config: str,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> list[str]:
        reasons = self.low_precision_unsupported_reasons(precision_config)
        if reasons:
            return reasons
        import transformer_engine.pytorch as te

        device_arg = str(device)
        for spec in self.node_specs:
            layer = self._make_te_layer(spec, te, params_dtype, device_arg)
            if layer is not None:
                self.layers[str(spec["id"])] = layer
        self.transformer_engine_enabled = True
        self.to(device)
        return []

    def _linear_dims(self, spec: dict[str, Any]) -> tuple[int, int]:
        op = spec["type"]
        args = spec.get("args", {})
        mem = spec.get("memory_info", {})
        if op == "Gemm":
            return _positive(args.get("linear_in_features")), _positive(args.get("linear_out_features"))
        return _input_last_dim(mem), _last_dim(mem)

    def _make_te_layer(self, spec: dict[str, Any], te: Any, params_dtype: torch.dtype, device: str) -> nn.Module | None:
        op = spec["type"]
        args = spec.get("args", {})
        mem = spec.get("memory_info", {})
        if op == "Gemm":
            return te.Linear(
                _positive(args.get("linear_in_features")),
                _positive(args.get("linear_out_features")),
                bias=bool(_as_int(args.get("linear_bias"))),
                params_dtype=params_dtype,
                device=device,
            )
        if op in {"MatMul", "Bmm", "TabularFeature"}:
            return te.Linear(_input_last_dim(mem), _last_dim(mem), params_dtype=params_dtype, device=device)
        if op == "LayerNormalization":
            try:
                return te.LayerNorm(_last_dim(mem), params_dtype=params_dtype, device=device)
            except TypeError:
                return te.LayerNorm(_last_dim(mem), params_dtype=params_dtype).to(device)
        if op in {"Attention", "MultiHeadAttention"}:
            return TransformerEngineSelfAttention(te, _last_dim(mem), params_dtype, device)
        return None

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
                layer = self.layers[str(node_id)]
                tokens = node_inputs[0].long()
                # Real-dataset seeded token tensors may use a larger tokenizer
                # id space than a compact generated embedding table.
                tokens = tokens.remainder(max(int(getattr(layer, "num_embeddings", 1)), 1))
                out = layer(tokens)
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
                tensor = node_inputs[0] if self.transformer_engine_enabled else node_inputs[0].float()
                out = self.layers[str(node_id)](tensor)
            elif op == "Bmm":
                tensor = node_inputs[0] if self.transformer_engine_enabled else node_inputs[0].float()
                out = self.layers[str(node_id)](tensor)
            else:
                out = self.layers[str(node_id)](node_inputs[0])
            values[node_id] = out
        return values[int(self.node_specs[-1]["id"])]

    def _gemm(self, node_id: int, tensor: torch.Tensor) -> torch.Tensor:
        layer = self.layers[str(node_id)]
        if node_id in self.flatten_before_gemm:
            tensor = torch.flatten(tensor, 1)
        return layer(tensor)
