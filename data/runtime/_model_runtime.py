"""Pure-PyTorch runtime for the generated multimodal model corpus.

Tensor symbols used throughout this module:

- ``B``: batch size.
- ``C``: channel or hidden width.
- ``H, W``: image height and width.
- ``T``: sequence length.
- ``N``: graph node count.
- ``F``: input feature count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from _model_catalog import FAMILY_SPECS


def _integer(config: Mapping[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def _head_count(width: int, requested: int) -> int:
    for heads in range(min(width, max(requested, 1)), 0, -1):
        if width % heads == 0:
            return heads
    return 1


def _activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "hardswish": nn.Hardswish(),
        "mish": nn.Mish(),
        "prelu": nn.PReLU(),
        "relu": nn.ReLU(inplace=False),
        "selu": nn.SELU(inplace=False),
        "silu": nn.SiLU(inplace=False),
    }.get(name, nn.GELU())


class SqueezeExcitation(nn.Module):
    """Channel gate with parameters ``C -> C/r -> C`` (Kaiming initialized)."""

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(4, int(channels * ratio))
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate source ``x`` with shape ``[B,C,H,W]``; return ``[B,C,H,W]``."""

        gate = x.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1], spatial mean of x.
        gate = F.silu(self.reduce(gate))  # [B,C/r,1,1], learned reduction.
        gate = torch.sigmoid(self.expand(gate))  # [B,C,1,1], learned channel gate.
        return x * gate  # [B,C,H,W], broadcast channel scaling.


class ConvNormAct(nn.Module):
    """Convolution-normalization-activation block with Kaiming defaults."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "relu",
        norm: str = "batch",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels) if norm == "group" else nn.BatchNorm2d(out_channels)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform source image ``x:[B,Cin,H,W]`` into ``[B,Cout,H/s,W/s]``."""

        y = self.conv(x)  # [B,Cout,H/s,W/s], learned spatial projection.
        y = self.norm(y)  # [B,Cout,H/s,W/s], batch/group normalized.
        return self.activation(y)  # [B,Cout,H/s,W/s], pointwise activation.


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V1/V2 block; all learned matrices use PyTorch defaults."""

    def __init__(self, channels: int, version: int = 1, activation: str = "gelu") -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, channels * 4)
        self.contract = nn.Linear(channels * 4, channels)
        self.activation = _activation(activation)
        self.version = version
        if version == 2:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a channel-preserving block to ``x:[B,C,H,W]``."""

        residual = x  # [B,C,H,W], identity branch from x.
        y = self.depthwise(x)  # [B,C,H,W], learned depthwise 7x7 convolution.
        y = y.permute(0, 2, 3, 1)  # [B,H,W,C], channels-last normalization layout.
        y = self.contract(self.activation(self.expand(self.norm(y))))  # [B,H,W,C], pointwise MLP.
        if self.version == 2:
            magnitude = torch.linalg.vector_norm(y, dim=(1, 2), keepdim=True)  # [B,1,1,C], spatial response norm.
            normalized = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)  # [B,1,1,C], global response norm.
            y = y + self.gamma * (y * normalized) + self.beta  # [B,H,W,C], ConvNeXt-V2 GRN.
        y = y.permute(0, 3, 1, 2)  # [B,C,H,W], restore image layout.
        return residual + y  # [B,C,H,W], residual update.


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block with optional ConvNeXt substitution."""

    def __init__(self, channels: int, base_module: str, activation: str = "relu", kernel_size: int = 3) -> None:
        super().__init__()
        self.convnext: nn.Module | None = None
        if "convnext_v2" in base_module:
            self.convnext = ConvNeXtBlock(channels, version=2, activation=activation)
        elif "convnext_v1" in base_module:
            self.convnext = ConvNeXtBlock(channels, version=1, activation=activation)
        self.first = ConvNormAct(channels, channels, kernel_size, activation=activation)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update source feature map ``x:[B,C,H,W]`` and preserve its shape."""

        if self.convnext is not None:
            return self.convnext(x)  # [B,C,H,W], substituted ConvNeXt path.
        y = self.second(self.first(x))  # [B,C,H,W], two learned convolutions.
        return self.activation(x + y)  # [B,C,H,W], residual activation.


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck with depthwise convolution and SE gating."""

    def __init__(self, channels: int, expand_ratio: int, fused: bool, se_ratio: float) -> None:
        super().__init__()
        hidden = max(channels, channels * expand_ratio)
        if fused:
            self.expand = ConvNormAct(channels, hidden, 3, activation="silu")
            self.depthwise = nn.Identity()
        else:
            self.expand = ConvNormAct(channels, hidden, 1, activation="silu")
            self.depthwise = ConvNormAct(hidden, hidden, 3, groups=hidden, activation="silu")
        self.se = SqueezeExcitation(hidden, se_ratio)
        self.project = nn.Sequential(nn.Conv2d(hidden, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform and gate ``x:[B,C,H,W]``; return a residual ``[B,C,H,W]``."""

        y = self.expand(x)  # [B,Cexp,H,W], learned expansion.
        y = self.depthwise(y)  # [B,Cexp,H,W], identity or depthwise convolution.
        y = self.project(self.se(y))  # [B,C,H,W], SE-gated projection.
        return x + y  # [B,C,H,W], inverted-bottleneck residual.


class InceptionBlock(nn.Module):
    """Multi-branch asymmetric convolution block."""

    def __init__(self, channels: int, asymmetric_kernel: int, residual: bool) -> None:
        super().__init__()
        branch = max(4, channels // 4)
        padding = asymmetric_kernel // 2
        self.branch1 = nn.Conv2d(channels, branch, 1)
        self.branch3 = nn.Sequential(nn.Conv2d(channels, branch, 1), nn.ReLU(), nn.Conv2d(branch, branch, 3, padding=1))
        self.branch_asym = nn.Sequential(
            nn.Conv2d(channels, branch, (1, asymmetric_kernel), padding=(0, padding)),
            nn.ReLU(),
            nn.Conv2d(branch, branch, (asymmetric_kernel, 1), padding=(padding, 0)),
        )
        self.branch_pool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(channels, branch, 1))
        self.project = nn.Conv2d(branch * 4, channels, 1)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate four branches from ``x:[B,C,H,W]`` and return ``[B,C,H,W]``."""

        branches = (self.branch1(x), self.branch3(x), self.branch_asym(x), self.branch_pool(x))  # Four [B,C/4,H,W] tensors.
        y = self.project(torch.cat(branches, dim=1))  # [B,C,H,W], concatenated branch projection.
        return F.relu(y + x if self.residual else y)  # [B,C,H,W], optional residual output.


class VisionClassifier(nn.Module):
    """Convolutional classifier used by compatible vision families."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        family = str(config["family_id"])
        width = _integer(config, "width", 32)
        depth = _integer(config, "depth", 4)
        base = str(config.get("base_module", "baseline"))
        activation = str(config.get("activation", "relu"))
        if family == "mish_resnet":
            activation = "mish"
        elif family == "prelu_elu_cnn":
            activation = "elu" if base == "elu" else "prelu"
        elif family == "mobilenet_v3_large":
            activation = "hardswish"
        kernel = _integer(config, "kernel_size", 3)
        self.stem = ConvNormAct(3, width, kernel, stride=2, activation=activation)
        blocks: list[nn.Module] = []
        for block_index in range(depth):
            if family == "efficientnet_b0_b4" and base != "convnext_v2":
                blocks.append(MBConvBlock(width, _integer(config, "expand_ratio", 4), base == "fused_mbconv", float(config.get("se_ratio", 0.25))))
            elif family == "mobilenet_v3_large" and base == "mobilenet_v3":
                blocks.append(nn.Sequential(ConvNormAct(width, width, kernel, groups=width, activation="hardswish"), SqueezeExcitation(width), nn.Conv2d(width, width, 1)))
            elif family == "inception_v3":
                blocks.append(InceptionBlock(width, _integer(config, "asymmetric_kernel", 7), base == "inception_residual"))
            elif family == "stn_cnn" and base == "stn_cnn":
                blocks.append(ConvNormAct(width, width, kernel, activation=activation))
            else:
                block_activation = "elu" if family == "prelu_elu_cnn" and base == "prelu_elu" and block_index % 2 else activation
                blocks.append(ResidualConvBlock(width, base, activation=block_activation, kernel_size=kernel))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Classify source image ``image:[B,3,H,W]`` into logits ``[B,10]``."""

        features = self.blocks(self.stem(image))  # [B,C,H/2,W/2], learned convolutional hierarchy.
        pooled = self.pool(features).flatten(1)  # [B,C], adaptive spatial mean.
        return self.head(pooled)  # [B,10], class logits.


class SpatialTransformerClassifier(nn.Module):
    """Affine spatial transformer followed by a convolutional classifier."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 24)
        localization_depth = _integer(config, "localization_depth", 2)
        localization: list[nn.Module] = [ConvNormAct(3, width, 5, stride=2)]
        localization.extend(ConvNormAct(width, width, 3) for _ in range(max(localization_depth - 1, 0)))
        self.localization = nn.Sequential(*localization, nn.AdaptiveAvgPool2d(1))
        self.affine = nn.Linear(width, 6)
        nn.init.zeros_(self.affine.weight)
        self.affine.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.classifier = VisionClassifier(config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Warp ``image:[B,3,H,W]`` with learned ``[B,2,3]`` affine matrices and classify to ``[B,10]``."""

        batch = image.shape[0]
        local = self.localization(image).flatten(1)  # [B,C], localization features.
        theta = self.affine(local).view(batch, 2, 3)  # [B,2,3], identity-initialized affine transform.
        grid = F.affine_grid(theta, image.shape, align_corners=False)  # [B,H,W,2], normalized sample grid.
        warped = F.grid_sample(image, grid, align_corners=False)  # [B,3,H,W], bilinearly warped image.
        return self.classifier(warped)  # [B,10], class logits.


class AttentionBlock(nn.Module):
    """MHA, GQA, MLA, or linear attention over ``[B,T,C]`` tokens."""

    def __init__(self, width: int, heads: int, mode: str, latent_ratio: float = 0.5) -> None:
        super().__init__()
        self.width = width
        self.heads = _head_count(width, heads)
        self.head_dim = width // self.heads
        self.mode = mode
        self.q_proj = nn.Linear(width, width, bias=False)
        if mode == "gqa":
            self.kv_heads = max(1, self.heads // 2)
            self.k_proj = nn.Linear(width, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(width, self.kv_heads * self.head_dim, bias=False)
            self.kv_down = None
            self.k_up = None
            self.v_up = None
        elif mode == "mla":
            latent = max(self.head_dim, int(width * latent_ratio))
            self.kv_heads = self.heads
            self.kv_down = nn.Linear(width, latent, bias=False)
            self.k_up = nn.Linear(latent, width, bias=False)
            self.v_up = nn.Linear(latent, width, bias=False)
            self.k_proj = None
            self.v_proj = None
        else:
            self.kv_heads = self.heads
            self.k_proj = nn.Linear(width, width, bias=False)
            self.v_proj = nn.Linear(width, width, bias=False)
            self.kv_down = None
            self.k_up = None
            self.v_up = None
        self.output = nn.Linear(width, width, bias=False)

    def _split(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        """Reshape projected ``x:[B,T,heads*D]`` into ``[B,heads,T,D]``."""

        batch, tokens, _channels = x.shape
        return x.view(batch, tokens, heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, causal: bool = False) -> torch.Tensor:
        """Attend from ``x:[B,Tq,C]`` to optional ``context:[B,Tk,C]``; return ``[B,Tq,C]``."""

        source = x if context is None else context  # [B,Tk,C], key/value source.
        query = self._split(self.q_proj(x), self.heads)  # [B,H,Tq,D], projected queries.
        if self.mode == "mla":
            assert self.kv_down is not None and self.k_up is not None and self.v_up is not None
            latent = self.kv_down(source)  # [B,Tk,L], low-rank KV latent.
            key = self._split(self.k_up(latent), self.heads)  # [B,H,Tk,D], reconstructed keys.
            value = self._split(self.v_up(latent), self.heads)  # [B,H,Tk,D], reconstructed values.
        else:
            assert self.k_proj is not None and self.v_proj is not None
            key = self._split(self.k_proj(source), self.kv_heads)  # [B,Hkv,Tk,D], projected keys.
            value = self._split(self.v_proj(source), self.kv_heads)  # [B,Hkv,Tk,D], projected values.
            if self.kv_heads != self.heads:
                repeats = self.heads // self.kv_heads
                key = key.repeat_interleave(repeats, dim=1)  # [B,H,Tk,D], grouped-query expansion.
                value = value.repeat_interleave(repeats, dim=1)  # [B,H,Tk,D], grouped value expansion.
        if self.mode == "delta_linear":
            query_features = F.normalize(F.elu(query) + 1.0, dim=-1)  # [B,H,Tq,D], bounded positive queries.
            key_features = F.normalize(F.elu(key) + 1.0, dim=-1)  # [B,H,Tk,D], bounded positive keys.
            state = torch.zeros(
                query.shape[0], query.shape[1], query.shape[-1], value.shape[-1],
                dtype=query.dtype, device=query.device,
            )  # [B,H,D,D], recurrent fast-weight state.
            outputs: list[torch.Tensor] = []
            for token_index in range(query.shape[2]):
                q_t = query_features[:, :, token_index]  # [B,H,D], current query.
                k_t = key_features[:, :, token_index]  # [B,H,D], current key.
                v_t = value[:, :, token_index]  # [B,H,D], current value.
                prediction = torch.einsum("bhd,bhde->bhe", k_t, state)  # [B,H,D], state prediction.
                beta = torch.sigmoid((q_t * k_t).sum(dim=-1, keepdim=True) / math.sqrt(self.head_dim))  # [B,H,1], update gate.
                delta = (v_t - prediction) * beta  # [B,H,D], gated delta-rule residual.
                state = state + torch.einsum("bhd,bhe->bhde", k_t, delta)  # [B,H,D,D], updated fast weights.
                outputs.append(torch.einsum("bhd,bhde->bhe", q_t, state))
            attended = torch.stack(outputs, dim=2)  # [B,H,Tq,D], causal delta-rule output.
        elif self.mode == "transposed_attention":
            channel_scores = torch.matmul(query.transpose(-2, -1), key) / math.sqrt(max(query.shape[2], 1))  # [B,H,D,D], channel attention.
            channel_weights = torch.softmax(channel_scores, dim=-1)  # [B,H,D,D], normalized channel mixing.
            attended = torch.matmul(value, channel_weights.transpose(-2, -1))  # [B,H,Tq,D], transposed-attention output.
        elif self.mode in {"linear_attention", "linear"}:
            query_features = F.elu(query) + 1.0  # [B,H,Tq,D], positive query feature map.
            key_features = F.elu(key) + 1.0  # [B,H,Tk,D], positive key feature map.
            kv_state = torch.einsum("bhtd,bhte->bhde", key_features, value)  # [B,H,D,D], linear-attention state.
            normalizer = torch.einsum("bhtd,bhd->bht", query_features, key_features.sum(dim=2)).clamp_min(1e-6)  # [B,H,Tq], denominator.
            attended = torch.einsum("bhtd,bhde->bhte", query_features, kv_state) / normalizer.unsqueeze(-1)  # [B,H,Tq,D], normalized output.
        else:
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=causal and context is None)  # [B,H,Tq,D], exact attention.
        batch, _heads, tokens, _dim = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, tokens, self.width)  # [B,Tq,C], merged heads.
        return self.output(merged)  # [B,Tq,C], output projection.


def _attention_mode(base_module: str) -> str:
    lowered = base_module.lower()
    if "transposed" in lowered:
        return "transposed_attention"
    if "mla" in lowered or "latent" in lowered:
        return "mla"
    if "gqa" in lowered:
        return "gqa"
    if "linear" in lowered or "delta" in lowered:
        return "delta_linear" if "delta" in lowered else "linear_attention"
    return "mha"


class FeedForward(nn.Module):
    """GELU or SwiGLU feed-forward network over ``[B,T,C]`` tokens."""

    def __init__(self, width: int, swiglu: bool = False, gate_kind: str | None = None) -> None:
        super().__init__()
        self.gate_kind = gate_kind or ("silu" if swiglu else "none")
        hidden = width * 4
        self.up = nn.Linear(width, hidden * (2 if self.gate_kind != "none" else 1))
        self.down = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map source tokens ``x:[B,T,C]`` through an MLP and return ``[B,T,C]``."""

        hidden = self.up(x)  # [B,T,4C] or [B,T,8C], learned expansion.
        if self.gate_kind != "none":
            value, gate = hidden.chunk(2, dim=-1)  # Two [B,T,4C] gated branches.
            gate = torch.sigmoid(gate) if self.gate_kind == "sigmoid" else F.silu(gate)  # [B,T,4C], selected gate.
            hidden = value * gate  # [B,T,4C], gated activation.
        else:
            hidden = F.gelu(hidden)  # [B,T,4C], GELU activation.
        return self.down(hidden)  # [B,T,C], learned contraction.


class TransformerLayer(nn.Module):
    """Pre-normalized attention and feed-forward residual layer."""

    def __init__(self, width: int, heads: int, mode: str, swiglu: bool, rms_norm: bool = False) -> None:
        super().__init__()
        norm = nn.RMSNorm if rms_norm else nn.LayerNorm
        self.norm1 = norm(width)
        self.attention = AttentionBlock(width, heads, mode)
        self.norm2 = norm(width)
        self.feed_forward = FeedForward(width, swiglu)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, causal: bool = False) -> torch.Tensor:
        """Update tokens ``x:[B,T,C]`` with optional ``context:[B,S,C]``; return ``[B,T,C]``."""

        attended = self.attention(self.norm1(x), context=context, causal=causal)  # [B,T,C], normalized attention update.
        x = x + attended  # [B,T,C], attention residual.
        return x + self.feed_forward(self.norm2(x))  # [B,T,C], feed-forward residual.


class VisionTokenModel(nn.Module):
    """ViT, Swin, or MLP-Mixer image-token classifier."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.family = str(config["family_id"])
        self.width = _integer(config, "width", 128)
        self.patch_size = _integer(config, "patch_size", 8)
        self.window_size = _integer(config, "window_size", 4)
        depth = _integer(config, "depth", 4)
        heads = _head_count(self.width, _integer(config, "heads", 4))
        self.base_module = str(config.get("base_module", "mha"))
        self.patch = nn.Conv2d(3, self.width, self.patch_size, stride=self.patch_size)
        if self.family == "mlp_mixer_s":
            self.blocks = nn.ModuleList(MixerBlock(self.width, self.base_module) for _ in range(depth))
        else:
            mode = _attention_mode(self.base_module)
            self.blocks = nn.ModuleList(TransformerLayer(self.width, heads, mode, swiglu=False) for _ in range(depth))
        self.norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, 10)

    def _window_attention(self, tokens: torch.Tensor, height: int, width: int, block: nn.Module, shifted: bool) -> torch.Tensor:
        """Apply block to windows from ``tokens:[B,H*W,C]`` and return the same shape."""

        batch, _tokens, channels = tokens.shape
        image = tokens.view(batch, height, width, channels)  # [B,H,W,C], spatial token grid.
        window = min(self.window_size, height, width)
        if shifted and window > 1:
            image = torch.roll(image, shifts=(-window // 2, -window // 2), dims=(1, 2))  # [B,H,W,C], shifted windows.
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        image = F.pad(image, (0, 0, 0, pad_w, 0, pad_h))  # [B,Hp,Wp,C], window-divisible grid.
        padded_h, padded_w = image.shape[1:3]
        windows = image.view(batch, padded_h // window, window, padded_w // window, window, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)  # [B*nwin,win^2,C], packed windows.
        windows = block(windows)  # [B*nwin,win^2,C], local attention update.
        image = windows.view(batch, padded_h // window, padded_w // window, window, window, channels)
        image = image.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, channels)  # [B,Hp,Wp,C], merged windows.
        image = image[:, :height, :width]
        if shifted and window > 1:
            image = torch.roll(image, shifts=(window // 2, window // 2), dims=(1, 2))  # [B,H,W,C], undo shift.
        return image.reshape(batch, height * width, channels)  # [B,H*W,C], flattened tokens.

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Patchify ``image:[B,3,H,W]`` and return class logits ``[B,10]``."""

        patches = self.patch(image)  # [B,C,Hp,Wp], learned non-overlapping patch embedding.
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)  # [B,Hp*Wp,C], token sequence.
        for index, block in enumerate(self.blocks):
            if self.family == "swin_t":
                tokens = self._window_attention(tokens, height, width, block, shifted=index % 2 == 1)  # [B,Hp*Wp,C], shifted-window update.
            else:
                tokens = block(tokens)  # [B,Hp*Wp,C], global attention or mixer update.
        pooled = self.norm(tokens).mean(dim=1)  # [B,C], normalized token mean.
        return self.head(pooled)  # [B,10], class logits.


class MixerBlock(nn.Module):
    """Token-independent channel mixer with transpose/reshape operations."""

    def __init__(self, width: int, base_module: str) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        gate_kind = "silu" if "swiglu" in base_module else "sigmoid" if "gated" in base_module else "none"
        self.channel_mlp = FeedForward(width, gate_kind=gate_kind)
        self.token_scale = nn.Parameter(torch.ones(1, 1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix source tokens ``x:[B,T,C]`` and preserve ``[B,T,C]``."""

        normalized = self.norm(x)  # [B,T,C], channel-normalized tokens.
        token_context = normalized.transpose(1, 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # [B,1,C], transpose-based token mixing.
        return x + self.channel_mlp(normalized) + token_context * self.token_scale  # [B,T,C], mixed residual.


class UpsampleBlock(nn.Module):
    """One of ConvTranspose, interpolate+Conv, PixelShuffle, or ConvNeXt upsampling."""

    def __init__(self, in_channels: int, out_channels: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "convtranspose":
            self.operation = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        elif mode == "pixelshuffle":
            self.operation = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, 3, padding=1), nn.PixelShuffle(2))
        else:
            layers: list[nn.Module] = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(in_channels, out_channels, 3, padding=1)]
            if mode == "convnext_v1":
                layers.append(ConvNeXtBlock(out_channels, version=1))
            elif mode == "convnext_v2":
                layers.append(ConvNeXtBlock(out_channels, version=2))
            self.operation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample source ``x:[B,Cin,H,W]`` to ``[B,Cout,2H,2W]``."""

        return self.operation(x)


class DenseVisionModel(nn.Module):
    """U-Net/pix2pix encoder-decoder with configurable upsampling."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 16)
        levels = min(max(_integer(config, "depth", 4) // 2, 2), 4)
        mode = str(config.get("base_module", "convtranspose"))
        norm = str(config.get("norm", "group"))
        channels = [width * (2**index) for index in range(levels)]
        self.stem = ConvNormAct(3, channels[0], norm=norm)
        self.down = nn.ModuleList(
            ConvNormAct(channels[index], channels[index + 1], stride=2, norm=norm)
            for index in range(levels - 1)
        )
        self.up = nn.ModuleList(
            UpsampleBlock(channels[index], channels[index - 1], mode)
            for index in range(levels - 1, 0, -1)
        )
        self.merge = nn.ModuleList(
            ConvNormAct(channels[index - 1] * 2, channels[index - 1], norm=norm)
            for index in range(levels - 1, 0, -1)
        )
        self.output = nn.Conv2d(channels[0], 3, 1)
        self.family = str(config["family_id"])
        if self.family == "pix2pix":
            discriminator_depth = _integer(config, "discriminator_depth", 2)
            discriminator: list[nn.Module] = [ConvNormAct(3, width, 4, stride=2, activation="elu")]
            discriminator.extend(ConvNormAct(width, width, 4, stride=2, activation="elu") for _ in range(discriminator_depth - 1))
            self.discriminator = nn.Sequential(*discriminator, nn.AdaptiveAvgPool2d(1))
            self.discriminator_head = nn.Linear(width, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map ``image:[B,3,H,W]`` to reconstruction features ``[B,K]``."""

        skips: list[torch.Tensor] = []
        x = self.stem(image)  # [B,C,H,W], encoder stem.
        skips.append(x)
        for down in self.down:
            x = down(x)  # [B,2C,H/2,W/2], downsampled encoder feature.
            skips.append(x)
        for index, (up, merge) in enumerate(zip(self.up, self.merge, strict=True)):
            x = up(x)  # [B,C,H*2,W*2], decoder upsampling.
            skip = skips[-index - 2]  # [B,C,H*2,W*2], matching encoder skip.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)  # [B,C,Hs,Ws], size alignment.
            x = merge(torch.cat((x, skip), dim=1))  # [B,C,Hs,Ws], skip-concatenation fusion.
        generated = torch.tanh(self.output(x))  # [B,3,H,W], bounded reconstruction.
        pooled = generated.mean(dim=(2, 3))  # [B,3], reconstruction summary.
        if self.family == "pix2pix":
            discriminator_features = self.discriminator(generated).flatten(1)  # [B,C], PatchGAN feature summary.
            discriminator_logit = self.discriminator_head(discriminator_features)  # [B,1], adversarial logit.
            pooled = torch.cat((pooled, discriminator_logit), dim=-1)  # [B,4], generator/discriminator joint output.
        return pooled


class RestormerModel(nn.Module):
    """Memory-bounded image restoration blocks with pooled attention."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 32)
        depth = _integer(config, "depth", 4)
        heads = _head_count(width, _integer(config, "heads", 4))
        mode = _attention_mode(str(config.get("base_module", "transposed_attention")))
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.local = nn.ModuleList(ConvNeXtBlock(width, version=2) for _ in range(depth))
        self.attention = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=True) for _ in range(depth))
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Restore ``image:[B,3,H,W]`` and return pooled RGB ``[B,3]``."""

        features = self.stem(image)  # [B,C,H,W], learned image embedding.
        for local, attention in zip(self.local, self.attention, strict=True):
            features = local(features)  # [B,C,H,W], local depthwise update.
            pooled = F.adaptive_avg_pool2d(features, (8, 8))  # [B,C,8,8], bounded attention grid.
            tokens = pooled.flatten(2).transpose(1, 2)  # [B,64,C], image tokens.
            tokens = attention(tokens)  # [B,64,C], efficient/global attention update.
            context = tokens.transpose(1, 2).reshape(features.shape[0], features.shape[1], 8, 8)  # [B,C,8,8], spatial context.
            features = features + F.interpolate(context, size=features.shape[-2:], mode="bilinear", align_corners=False)  # [B,C,H,W], restored context.
        restored = self.output(features) + image  # [B,3,H,W], image residual.
        return restored.mean(dim=(2, 3))  # [B,3], pooled reconstruction output.


class TokenTransformerModel(nn.Module):
    """Encoder or decoder transformer used by BERT/GPT/Llama/MLA/Kimi/DistilBERT."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        family = str(config["family_id"])
        width = _integer(config, "width", 128)
        depth = _integer(config, "depth", 2)
        heads = _head_count(width, _integer(config, "heads", 4))
        vocab_size = _integer(config, "vocab_size", 4096)
        self.family = family
        self.causal = family in {"gpt2_small", "llama_small", "kimi_linear"}
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.zeros(1, 512, width))
        mode = _attention_mode(str(config.get("base_module", "mha")))
        swiglu = family in {"llama_small", "mla_mini_transformer", "kimi_linear"}
        rms = family in {"llama_small", "kimi_linear"}
        self.layers = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=swiglu, rms_norm=rms) for _ in range(depth))
        self.norm = nn.RMSNorm(width) if rms else nn.LayerNorm(width)
        self.head = nn.Linear(width, 8)
        if family == "distilbert_distillation":
            self.teacher_head = nn.Linear(width, 8)
            for parameter in self.teacher_head.parameters():
                parameter.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode integer ``token_ids:[B,T]`` and return logits/features ``[B,K]``."""

        tokens = self.embedding(token_ids)  # [B,T,C], learned token embeddings.
        tokens = tokens + self.position[:, : tokens.shape[1]]  # [B,T,C], learned absolute positions.
        for layer in self.layers:
            tokens = layer(tokens, causal=self.causal)  # [B,T,C], transformer update.
        pooled = self.norm(tokens).mean(dim=1)  # [B,C], normalized sequence mean.
        logits = self.head(pooled)  # [B,8], task logits.
        if self.family == "distilbert_distillation":
            teacher = torch.softmax(self.teacher_head(pooled.detach()), dim=-1)  # [B,8], frozen teacher probabilities.
            student = F.log_softmax(logits, dim=-1)  # [B,8], student log probabilities.
            return torch.cat((student, teacher), dim=-1)  # [B,16], distillation pair.
        return logits


class T5Model(nn.Module):
    """Small encoder-decoder transformer with cross-attention."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 128)
        depth = _integer(config, "depth", 2)
        heads = _head_count(width, _integer(config, "heads", 4))
        vocab_size = _integer(config, "vocab_size", 4096)
        mode = _attention_mode(str(config.get("base_module", "mha")))
        self.embedding = nn.Embedding(vocab_size, width)
        self.encoder = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=False) for _ in range(depth))
        self.decoder = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=False) for _ in range(depth))
        self.head = nn.Linear(width, 16)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode/decode ``token_ids:[B,T]`` and return token logits ``[B,T,16]``."""

        encoded = self.embedding(token_ids)  # [B,T,C], shared token embeddings.
        for layer in self.encoder:
            encoded = layer(encoded)  # [B,T,C], encoder self-attention.
        decoded = self.embedding(torch.roll(token_ids, shifts=1, dims=1))  # [B,T,C], shifted decoder inputs.
        for layer in self.decoder:
            decoded = layer(decoded, context=encoded, causal=False)  # [B,T,C], decoder cross-attention.
        return self.head(decoded)  # [B,T,16], compact vocabulary logits.


class SwitchMoEModel(nn.Module):
    """Attention encoder with top-k routed feed-forward experts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 128)
        depth = _integer(config, "depth", 2)
        expert_count = _integer(config, "expert_count", 4)
        self.top_k = min(_integer(config, "top_k", 1), expert_count)
        heads = _head_count(width, _integer(config, "heads", 4))
        mode = _attention_mode(str(config.get("base_module", "mha_moe")))
        self.embedding = nn.Embedding(8192, width)
        self.layers = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=False) for _ in range(depth))
        self.router = nn.Linear(width, expert_count)
        self.experts = nn.ModuleList(FeedForward(width, swiglu=True) for _ in range(expert_count))
        self.head = nn.Linear(width, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Route ``token_ids:[B,T]`` through experts and return logits ``[B,8]``."""

        tokens = self.embedding(token_ids)  # [B,T,C], token embeddings.
        for layer in self.layers:
            tokens = layer(tokens)  # [B,T,C], attention context.
        router_logits = self.router(tokens)  # [B,T,E], expert routing logits.
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)  # [B,T,K] each, selected experts.
        top_weights = torch.softmax(top_values, dim=-1)  # [B,T,K], normalized routing weights.
        combined = torch.zeros_like(tokens)  # [B,T,C], initialized expert accumulator.
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(tokens)  # [B,T,C], expert-specific update.
            mask = (top_indices == expert_index).to(tokens.dtype) * top_weights  # [B,T,K], sparse route weights.
            combined = combined + expert_output * mask.sum(dim=-1, keepdim=True)  # [B,T,C], weighted expert sum.
        return self.head((tokens + combined).mean(dim=1))  # [B,8], pooled class logits.


class RecurrentTextModel(nn.Module):
    """BiLSTM-CRF-like or recurrent sequence-to-sequence model."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        family = str(config["family_id"])
        width = _integer(config, "width", 96)
        depth = _integer(config, "depth", 1)
        vocab_size = _integer(config, "vocab_size", 4096)
        base = str(config.get("base_module", "gru"))
        self.family = family
        self.base = base
        self.embedding = nn.Embedding(vocab_size, width)
        if family == "bilstm_crf":
            self.recurrent = nn.LSTM(width, width, num_layers=depth, batch_first=True, bidirectional=True)
            recurrent_width = width * 2
        else:
            recurrent_cls = {"gru": nn.GRU, "rnn": nn.RNN, "lstm": nn.LSTM}.get(base, nn.GRU)
            bidirectional = bool(config.get("bidirectional", False))
            self.recurrent = recurrent_cls(width, width, num_layers=depth, batch_first=True, bidirectional=bidirectional)
            recurrent_width = width * (2 if bidirectional else 1)
        self.head = nn.Linear(recurrent_width, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Process integer ``token_ids:[B,T]`` and return sequence features ``[B,T,K]``."""

        embeddings = self.embedding(token_ids)  # [B,T,C], learned token embeddings.
        sequence, _state = self.recurrent(embeddings)  # [B,T,Crnn], recurrent outputs and discarded final state.
        emissions = self.head(sequence)  # [B,T,8], token emissions.
        if self.family == "bilstm_crf" and self.base == "bilstm_crf":
            partition = torch.logsumexp(emissions, dim=-1, keepdim=True)  # [B,T,1], CRF-style local partition.
            return torch.cat((emissions, partition), dim=-1)  # [B,T,9], emissions plus partition feature.
        return emissions


class FastTextModel(nn.Module):
    """EmbeddingBag classifier with configurable reduction."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = _integer(config, "width", 64)
        depth = _integer(config, "depth", 1)
        vocab_size = _integer(config, "vocab_size", 4096)
        mode = "sum" if str(config.get("base_module")) == "embeddingbag_sum" else "mean"
        self.embedding = nn.EmbeddingBag(vocab_size, width, mode=mode)
        layers: list[nn.Module] = []
        for _ in range(max(depth - 1, 0)):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(width, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Reduce integer ``token_ids:[B,T]`` into class logits ``[B,8]``."""

        batch, tokens = token_ids.shape
        flat = token_ids.reshape(-1)  # [B*T], flattened token IDs.
        offsets = torch.arange(0, batch * tokens, tokens, device=token_ids.device)  # [B], bag start offsets.
        pooled = self.embedding(flat, offsets)  # [B,C], sparse bag reduction.
        return self.head(self.layers(pooled))  # [B,8], class logits.


class AudioConvModel(nn.Module):
    """PANNs-style 2D audio classifier or M5/TCN 1D waveform classifier."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.family = str(config["family_id"])
        width = _integer(config, "width", 32)
        depth = _integer(config, "depth", 4)
        base = str(config.get("base_module", "tcn"))
        if self.family == "panns_cnn14":
            blocks: list[nn.Module] = [ConvNormAct(1, width, 3)]
            blocks.extend(ResidualConvBlock(width, "baseline") if base == "residual_cnn" else ConvNormAct(width, width, 3) for _ in range(depth - 1))
            self.network = nn.Sequential(*blocks)
        else:
            kernel = _integer(config, "kernel_size", 3)
            layers: list[nn.Module] = []
            in_channels = 1
            for index in range(depth):
                dilation = 2 ** (index % _integer(config, "dilation_cycle", 4)) if self.family == "temporal_convolutional_network" else 1
                stride = _integer(config, "stride", 8) if self.family == "m5_waveform_cnn" and index == 0 else 1
                groups = in_channels if base == "depthwise_tcn" and in_channels == width else 1
                padding = dilation * (kernel // 2)
                layers.extend((nn.Conv1d(in_channels, width, kernel, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False), nn.BatchNorm1d(width), nn.ReLU()))
                if base == "m5_se":
                    layers.append(ChannelGate1D(width))
                in_channels = width
            self.network = nn.Sequential(*layers)
        self.head = nn.Linear(width, 8)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Classify spectrogram ``[B,1,M,T]`` or waveform ``[B,1,T]`` into ``[B,8]``."""

        features = self.network(audio)  # [B,C,...], learned temporal/frequency features.
        spatial_dims = tuple(range(2, features.ndim))
        pooled = features.mean(dim=spatial_dims)  # [B,C], global temporal/frequency mean.
        return self.head(pooled)  # [B,8], audio class logits.


class ChannelGate1D(nn.Module):
    """Squeeze-excitation gate for ``[B,C,T]`` waveforms."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 4)
        self.reduce = nn.Conv1d(channels, hidden, 1)
        self.expand = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate source ``x:[B,C,T]`` and preserve ``[B,C,T]``."""

        gate = torch.sigmoid(self.expand(F.relu(self.reduce(x.mean(dim=-1, keepdim=True)))))  # [B,C,1], learned channel gate.
        return x * gate  # [B,C,T], gated waveform features.


class SeluMLP(nn.Module):
    """Self-normalizing tabular MLP with AlphaDropout."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        features = _integer(config, "feature_count", 32)
        width = _integer(config, "width", 64)
        depth = _integer(config, "depth", 2)
        dropout = float(config.get("dropout", 0.1))
        self.input = nn.Linear(features, width)
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.dropout = nn.AlphaDropout(dropout)
        self.residual = str(config.get("base_module")) == "residual_selu"
        self.head = nn.Linear(width, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map tabular ``features:[B,F]`` to logits ``[B,2]``."""

        x = F.selu(self.input(features))  # [B,C], learned input projection.
        for layer in self.layers:
            update = self.dropout(F.selu(layer(x)))  # [B,C], self-normalizing hidden update.
            x = x + update if self.residual else update  # [B,C], optional residual path.
        return self.head(x)  # [B,2], binary logits.


class TabTransformerModel(nn.Module):
    """Feature-token transformer for mixed tabular inputs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        feature_count = _integer(config, "feature_count", 16)
        categorical = min(_integer(config, "categorical_features", 4), feature_count)
        width = _integer(config, "width", 64)
        depth = _integer(config, "depth", 2)
        heads = _head_count(width, _integer(config, "heads", 4))
        mode = _attention_mode(str(config.get("base_module", "mha")))
        self.feature_count = feature_count
        self.categorical = categorical
        self.category_embeddings = nn.ModuleList(nn.Embedding(32, width) for _ in range(categorical))
        self.numeric_projections = nn.ModuleList(nn.Linear(1, width) for _ in range(feature_count - categorical))
        self.layers = nn.ModuleList(TransformerLayer(width, heads, mode, swiglu=False) for _ in range(depth))
        self.head = nn.Linear(width, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Tokenize tabular ``features:[B,F]`` and return logits ``[B,2]``."""

        tokens: list[torch.Tensor] = []
        for index, embedding in enumerate(self.category_embeddings):
            category_ids = (features[:, index].abs() * 7).long().remainder(32)  # [B], deterministic categorical IDs.
            tokens.append(embedding(category_ids))  # [B,C], categorical token.
        for offset, projection in enumerate(self.numeric_projections, start=self.categorical):
            tokens.append(projection(features[:, offset : offset + 1]))  # [B,C], numeric feature token.
        x = torch.stack(tokens, dim=1)  # [B,F,C], stacked feature tokens.
        for layer in self.layers:
            x = layer(x)  # [B,F,C], feature attention update.
        return self.head(x.mean(dim=1))  # [B,2], class logits.


class MixtureDensityModel(nn.Module):
    """Gaussian or Laplace mixture-density regression head."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        features = _integer(config, "feature_count", 16)
        width = _integer(config, "width", 64)
        depth = _integer(config, "depth", 2)
        self.components = _integer(config, "component_count", 3)
        self.distribution = str(config.get("base_module", "gaussian_mixture"))
        layers: list[nn.Module] = [nn.Linear(features, width), nn.ELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ELU()))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(width, self.components * 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map regression features ``[B,F]`` to mixture parameters ``[B,3K]``."""

        raw = self.head(self.network(features))  # [B,3K], unconstrained mixture outputs.
        logits, location, raw_scale = raw.chunk(3, dim=-1)  # Three [B,K] parameter groups.
        weights = torch.softmax(logits, dim=-1)  # [B,K], mixture probabilities.
        scale = F.softplus(raw_scale) + 1e-4  # [B,K], positive scales.
        if self.distribution == "laplace_mixture":
            scale = torch.exp(torch.log(scale))  # [B,K], explicit Laplace exp/log path.
        return torch.cat((weights, location, scale), dim=-1)  # [B,3K], valid mixture parameters.


class DenseGraphModel(nn.Module):
    """Dense-batched CGCNN, GCN, GAT, or GraphSAGE approximation."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        feature_count = _integer(config, "feature_count", 32)
        width = _integer(config, "width", 64)
        depth = _integer(config, "depth", 2)
        self.base = str(config.get("base_module", "gcn"))
        self.input = nn.Linear(feature_count, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.neighbor_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.gates = nn.ModuleList(nn.Linear(width * 2, width) for _ in range(depth))
        self.head = nn.Linear(width, 1)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Process ``nodes:[B,N,F]`` and ``adjacency:[B,N,N]`` into ``[B,1]``."""

        x = self.input(nodes)  # [B,N,C], learned node embedding.
        eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)  # [1,N,N], self loops.
        graph = adjacency + eye  # [B,N,N], adjacency with self loops.
        degree = graph.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,N,1], node degree.
        normalized = graph / degree  # [B,N,N], row-normalized adjacency.
        for self_layer, neighbor_layer, gate_layer in zip(self.self_layers, self.neighbor_layers, self.gates, strict=True):
            if self.base == "gat":
                scores = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(x.shape[-1])  # [B,N,N], dot-product edge attention.
                scores = scores.masked_fill(graph <= 0, torch.finfo(scores.dtype).min)  # [B,N,N], adjacency mask.
                weights = torch.softmax(scores, dim=-1)  # [B,N,N], edge attention weights.
                neighbors = torch.matmul(weights, x)  # [B,N,C], attention aggregation.
            else:
                neighbors = torch.matmul(normalized, x)  # [B,N,C], normalized message aggregation.
            if self.base == "graphsage":
                x = F.relu(self_layer(x) + neighbor_layer(neighbors))  # [B,N,C], separate self/neighbor GraphSAGE update.
            elif self.base == "gated_cgcnn":
                gate = torch.sigmoid(gate_layer(torch.cat((x, neighbors), dim=-1)))  # [B,N,C], learned crystal-message gate.
                x = F.softplus(self_layer(x) + gate * neighbor_layer(neighbors))  # [B,N,C], gated crystal update.
            elif self.base == "cgcnn":
                x = F.softplus(self_layer(x) + neighbor_layer(neighbors))  # [B,N,C], ungated crystal update.
            else:
                x = F.relu(neighbor_layer(neighbors))  # [B,N,C], graph convolution update.
        return self.head(x.mean(dim=1))  # [B,1], graph-level regression output.


def input_specs_from_config(config: Mapping[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Build profiler input specifications with concrete leading batch size."""

    family = str(config.get("family_id", ""))
    if family not in FAMILY_SPECS:
        raise KeyError(f"unknown family {family!r}")
    batch = int(batch_size)
    modality = FAMILY_SPECS[family].modality
    if modality == "vision":
        resolution = _integer(config, "resolution", 128)
        return [{"name": "image", "shape": [batch, 3, resolution, resolution], "dtype": "float32", "kind": "image"}]
    if modality == "nlp":
        sequence_length = _integer(config, "sequence_length", 128)
        return [{"name": "token_ids", "shape": [batch, sequence_length], "dtype": "int64", "kind": "token_ids"}]
    if family == "panns_cnn14":
        return [{"name": "spectrogram", "shape": [batch, 1, _integer(config, "mel_bins", 64), _integer(config, "frames", 256)], "dtype": "float32", "kind": "audio"}]
    if modality == "audio":
        return [{"name": "waveform", "shape": [batch, 1, _integer(config, "waveform_length", 4096)], "dtype": "float32", "kind": "audio"}]
    if modality == "tabular":
        return [{"name": "features", "shape": [batch, _integer(config, "feature_count", 32)], "dtype": "float32", "kind": "tabular"}]
    if modality == "graph":
        nodes = _integer(config, "node_count", 32)
        features = _integer(config, "feature_count", 32)
        return [
            {"name": "nodes", "shape": [batch, nodes, features], "dtype": "float32", "kind": "graph_nodes"},
            {"name": "adjacency", "shape": [batch, nodes, nodes], "dtype": "float32", "kind": "adjacency"},
        ]
    raise AssertionError(f"unhandled modality {modality!r}")


def make_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate one requested family without external model libraries."""

    family = str(config.get("family_id", ""))
    if family not in FAMILY_SPECS:
        raise KeyError(f"unknown family {family!r}")
    if family == "stn_cnn":
        return SpatialTransformerClassifier(config)
    if family in {"vit_s16", "swin_t", "mlp_mixer_s"}:
        return VisionTokenModel(config)
    if family in {"unet", "pix2pix"}:
        return DenseVisionModel(config)
    if family == "restormer":
        return RestormerModel(config)
    if FAMILY_SPECS[family].modality == "vision":
        return VisionClassifier(config)
    if family == "t5_small":
        return T5Model(config)
    if family == "switch_moe_classifier":
        return SwitchMoEModel(config)
    if family in {"bilstm_crf", "rnn_seq2seq"}:
        return RecurrentTextModel(config)
    if family == "fasttext_embeddingbag":
        return FastTextModel(config)
    if FAMILY_SPECS[family].modality == "nlp":
        return TokenTransformerModel(config)
    if FAMILY_SPECS[family].modality == "audio":
        return AudioConvModel(config)
    if family == "selu_mlp":
        return SeluMLP(config)
    if family == "tabtransformer":
        return TabTransformerModel(config)
    if family == "mixture_density_network":
        return MixtureDensityModel(config)
    if FAMILY_SPECS[family].modality == "graph":
        return DenseGraphModel(config)
    raise AssertionError(f"unhandled family {family!r}")
