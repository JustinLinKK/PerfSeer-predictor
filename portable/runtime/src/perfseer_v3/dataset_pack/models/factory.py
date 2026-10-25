"""Small but semantically faithful implementations of all 35 model families."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.base import TaskAdapter
from ..fingerprints import canonical_sha256
from .architecture import resolve_architecture_parameters
from .base import FamilyBuildConfig, FamilyModel, ModelFactoryError


class ExactLinear(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_width, input_width))
        self.bias = nn.Parameter(torch.zeros(output_width))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.linear.default(value, self.weight, self.bias)


class ExactConv1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, *, padding: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels, kernel))
        self.bias = nn.Parameter(torch.zeros(output_channels))
        self.padding = padding
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.conv1d.default(value, self.weight, self.bias, [1], [self.padding], [1], 1)


class ExactConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(output_channels))
        self.stride = stride
        self.padding = padding
        self.groups = groups
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.conv2d.default(
            value,
            self.weight,
            self.bias,
            [self.stride, self.stride],
            [self.padding, self.padding],
            [1, 1],
            self.groups,
        )


class ExactConvTranspose2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, *, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_channels, output_channels, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(output_channels))
        self.stride = stride
        self.padding = padding
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.conv_transpose2d.input(
            value,
            self.weight,
            self.bias,
            [self.stride, self.stride],
            [self.padding, self.padding],
            [0, 0],
            1,
            [1, 1],
        )


class ExactBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.batch_norm.default(
            value,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            0.1,
            1e-5,
            False,
        )


def _layer_norm(value: torch.Tensor) -> torch.Tensor:
    width = value.shape[-1]
    return torch.ops.aten.layer_norm.default(value, [width], None, None, 1e-5, False)


def _attention(
    value: torch.Tensor,
    heads: int = 2,
    *,
    math_backend: bool = False,
) -> torch.Tensor:
    batch, length, width = value.shape
    if width % heads:
        raise ModelFactoryError("attention width must be divisible by its head count")
    # CUDA flash attention requires a contiguous last dimension. Window and
    # restoration partitioning can otherwise feed it a strided view that
    # works eagerly but fails after Inductor selects the fused kernel.
    shaped = value.reshape(batch, length, heads, width // heads).transpose(1, 2).contiguous()
    if math_backend:
        # Inductor can otherwise assign a non-unit last-dimension stride to a
        # fused window/restoration tensor before dispatching a CUDA kernel.
        # The math SDPA backend keeps the operation compiled and portable to
        # V100 while honoring the explicit contiguous layout above.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended = torch.ops.aten.scaled_dot_product_attention.default(
                shaped, shaped, shaped
            )
    else:
        attended = torch.ops.aten.scaled_dot_product_attention.default(
            shaped, shaped, shaped
        )
    return attended.transpose(1, 2).reshape(batch, length, width)


def _scatter_add_float32(
    output_shape: tuple[int, ...],
    index: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Deterministically segment-sum graph values in FP32."""

    destination = index[:, 0]
    order = torch.argsort(destination, stable=True)
    accumulated = torch.segment_reduce(
        source.float().index_select(0, order),
        reduce="sum",
        lengths=torch.bincount(destination, minlength=output_shape[0]),
    )
    return accumulated.to(source.dtype)


def _parameters(config: FamilyBuildConfig) -> dict[str, Any]:
    return dict(config.architecture_parameters or {})


def _use(
    model: FamilyModel,
    parameters: Mapping[str, Any],
    field: str,
    semantic_role: str,
) -> Any:
    model.architecture_semantic_roles[field] = semantic_role
    return parameters[field]


def _positive_int(value: Any, *, field: str, maximum: int = 64) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ModelFactoryError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _depth(value: Any, *, field: str, maximum: int = 16) -> int:
    if isinstance(value, list):
        if not value or any(type(item) is not int or item < 1 for item in value):
            raise ModelFactoryError(f"{field} must contain positive stage depths")
        result = sum(value)
    else:
        result = _positive_int(value, field=field, maximum=maximum)
    if result > maximum:
        raise ModelFactoryError(f"{field} total depth exceeds {maximum}")
    return result


def _stage_depths(value: Any, *, field: str, maximum: int = 16) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ModelFactoryError(f"{field} must be a non-empty stage-depth list")
    if any(type(item) is not int or item < 1 for item in value):
        raise ModelFactoryError(f"{field} must contain positive stage depths")
    if sum(value) > maximum:
        raise ModelFactoryError(f"{field} total depth exceeds {maximum}")
    return tuple(value)


def _scaled_width(value: Any, *, base: int, reference: float, field: str) -> int:
    if type(value) not in {int, float} or float(value) <= 0.0:
        raise ModelFactoryError(f"{field} must be positive")
    return max(2, min(64, int(round(base * float(value) / reference))))


def _resize_image(image: torch.Tensor, resolution: int) -> torch.Tensor:
    if image.shape[-2:] == (resolution, resolution):
        return image
    return F.interpolate(image, size=(resolution, resolution), mode="bilinear", align_corners=False)


def _resize_last_dimension(value: torch.Tensor, width: int) -> torch.Tensor:
    current = value.shape[-1]
    if current == width:
        return value
    if current > width:
        return value[..., :width]
    return F.pad(value, (0, width - current))


def _resize_sequence(value: torch.Tensor, length: int) -> torch.Tensor:
    current = value.shape[1]
    if current == length:
        return value
    if current > length:
        return value[:, :length]
    return F.pad(value, (0, length - current))


def _window_attention(
    value: torch.Tensor,
    *,
    heads: int,
    window_size: int,
    shifted: bool,
) -> torch.Tensor:
    tokens_per_window = window_size**2
    original_length = value.shape[1]
    grid_size = math.isqrt(original_length)
    if grid_size * grid_size != original_length:
        raise ModelFactoryError("Swin patch tokens must form a square spatial grid")
    shift = window_size // 2 if shifted else 0
    batch, _, width = value.shape
    spatial = value.reshape(batch, grid_size, grid_size, width)
    if shift:
        spatial = torch.ops.aten.roll.default(spatial, [-shift, -shift], [1, 2])
    padded_grid = math.ceil(grid_size / window_size) * window_size
    if padded_grid != grid_size:
        spatial = F.pad(
            spatial,
            (0, 0, 0, padded_grid - grid_size, 0, padded_grid - grid_size),
        )
    windows = spatial.reshape(
        batch,
        padded_grid // window_size,
        window_size,
        padded_grid // window_size,
        window_size,
        width,
    ).permute(0, 1, 3, 2, 4, 5)
    windows = windows.reshape(-1, tokens_per_window, width)
    attended = _attention(windows, heads=heads, math_backend=True)
    spatial = attended.reshape(
        batch,
        padded_grid // window_size,
        padded_grid // window_size,
        window_size,
        window_size,
        width,
    ).permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_grid, padded_grid, width)
    spatial = spatial[:, :grid_size, :grid_size]
    if shift:
        spatial = torch.ops.aten.roll.default(spatial, [shift, shift], [1, 2])
    return spatial.reshape(batch, original_length, width)


class VisionFamilyModel(FamilyModel):
    def __init__(self, family_id: str, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = family_id
        self.output_width = config.output_width
        self.task_kind = config.task_kind
        parameters = _parameters(config)
        if family_id == "resnet50":
            width = _scaled_width(
                _use(self, parameters, "width_multiplier", "residual stage channel width"),
                base=8,
                reference=0.25,
                field="width_multiplier",
            )
            stage_depths = _stage_depths(
                _use(self, parameters, "stage_depths", "residual blocks per stage"),
                field="stage_depths",
            )
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.conv1, self.norm1 = ExactConv2d(3, width, 3, padding=1), ExactBatchNorm2d(width)
            self.stages = nn.ModuleList(
                nn.ModuleList(
                    ExactConv2d(width, width, 3, padding=1) for _ in range(stage_depth)
                )
                for stage_depth in stage_depths
            )
            self.stage_norms = nn.ModuleList(
                nn.ModuleList(ExactBatchNorm2d(width) for _ in range(stage_depth))
                for stage_depth in stage_depths
            )
            self.stage_transitions = nn.ModuleList(
                ExactConv2d(width, width, 1) for _ in range(len(stage_depths) - 1)
            )
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "efficientnet_b0_b4":
            variant = str(_use(self, parameters, "variant", "EfficientNet compound scale variant"))
            variant_scale = 1 if variant.lower() == "b0" else 2
            width = _scaled_width(
                _use(self, parameters, "width_multiplier", "compound channel multiplier"),
                base=6 * variant_scale,
                reference=0.25,
                field="width_multiplier",
            )
            depth_multiplier = _use(self, parameters, "depth_multiplier", "MBConv repetition multiplier")
            if type(depth_multiplier) not in {int, float} or float(depth_multiplier) <= 0:
                raise ModelFactoryError("depth_multiplier must be positive")
            block_count = max(1, min(8, int(round(float(depth_multiplier) * 2 * variant_scale))))
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.conv1 = ExactConv2d(3, width, 3, padding=1)
            self.depthwise_blocks = nn.ModuleList(
                ExactConv2d(width, width, 3, padding=1, groups=width)
                for _ in range(block_count)
            )
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "mobilenet_v3_large":
            width = _scaled_width(
                _use(self, parameters, "width_multiplier", "MobileNet channel multiplier"),
                base=8,
                reference=0.25,
                field="width_multiplier",
            )
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.conv1 = ExactConv2d(3, width, 3, padding=1)
            self.depthwise = ExactConv2d(width, width, 3, padding=1, groups=width)
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "inception_v3":
            self.aux_logits = bool(
                _use(self, parameters, "aux_logits", "auxiliary classifier branch")
            )
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.branch1, self.branch3 = ExactConv2d(3, 4, 1), ExactConv2d(3, 4, 3, padding=1)
            self.head = ExactLinear(8, config.output_width)
            if self.aux_logits:
                self.aux_head = ExactLinear(4, config.output_width)
        elif family_id in {"vit_s16", "swin_t", "restormer"}:
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "attention head partition"),
                field="heads",
                maximum=16,
            )
            if family_id == "vit_s16":
                stage_depths = (
                    _depth(
                    _use(self, parameters, "depth", "transformer encoder block count"),
                    field="depth",
                    ),
                )
                self.patch_size = _positive_int(
                    _use(self, parameters, "patch_size", "patch embedding kernel and stride"),
                    field="patch_size",
                    maximum=self.input_resolution,
                )
                width = self.attention_heads * 4
            elif family_id == "swin_t":
                self.golden_learning_rate = 0.005
                stage_depths = _stage_depths(
                    _use(self, parameters, "depths", "shifted-window stage block counts"),
                    field="depths",
                )
                self.window_size = _positive_int(
                    _use(self, parameters, "window_size", "shifted attention window size"),
                    field="window_size",
                    maximum=self.input_resolution,
                )
                self.patch_size = 4
                width = self.attention_heads * 4
            else:
                stage_depths = _stage_depths(
                    _use(self, parameters, "depths", "restoration transformer block counts"),
                    field="depths",
                )
                width = _positive_int(
                    _use(self, parameters, "width", "restoration channel width"),
                    field="width",
                )
                width = max(self.attention_heads, math.ceil(width / self.attention_heads) * self.attention_heads)
                self.patch_size = 4
            self.patch = ExactConv2d(3, width, self.patch_size, stride=self.patch_size)
            self.attention_stages = nn.ModuleList(
                nn.ModuleList(ExactLinear(width, width) for _ in range(stage_depth))
                for stage_depth in stage_depths
            )
            self.attention_stage_transitions = nn.ModuleList(
                ExactLinear(width, width) for _ in range(len(stage_depths) - 1)
            )
            self.head = ExactLinear(width, config.output_width)
            if family_id == "restormer":
                self.restore = ExactConv2d(width, 3, 1)
        elif family_id == "mlp_mixer_s":
            block_count = _depth(
                _use(self, parameters, "depth", "mixer block count"), field="depth"
            )
            channel_dim = _positive_int(
                _use(self, parameters, "channel_dim", "patch embedding and channel width"),
                field="channel_dim",
            )
            token_dim = _positive_int(
                _use(self, parameters, "token_dim", "token-mixing hidden width"),
                field="token_dim",
            )
            self.patch_size = _positive_int(
                _use(self, parameters, "patch_size", "patch embedding kernel and stride"),
                field="patch_size",
                maximum=16,
            )
            self.mixer_resolution = 16
            token_count = (self.mixer_resolution // self.patch_size) ** 2
            if token_count < 1:
                raise ModelFactoryError("MLP-Mixer patch size exceeds its input resolution")
            self.patch = ExactConv2d(3, channel_dim, self.patch_size, stride=self.patch_size)
            self.token_in = nn.ModuleList(ExactLinear(token_count, token_dim) for _ in range(block_count))
            self.token_out = nn.ModuleList(ExactLinear(token_dim, token_count) for _ in range(block_count))
            self.channel_mixers = nn.ModuleList(ExactLinear(channel_dim, channel_dim) for _ in range(block_count))
            self.head = ExactLinear(channel_dim, config.output_width)
        elif family_id == "stn_cnn":
            localization_width = _positive_int(
                _use(self, parameters, "localization_width", "localization network channels"),
                field="localization_width",
            )
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.localization = ExactConv2d(3, localization_width, 3, padding=1)
            self.theta = ExactLinear(localization_width, 6)
            nn.init.zeros_(self.theta.weight)
            with torch.no_grad():
                self.theta.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
            self.conv1 = ExactConv2d(3, 6, 3, padding=1)
            self.head = ExactLinear(6, config.output_width)
        elif family_id == "mish_resnet":
            width = _scaled_width(
                _use(self, parameters, "width_multiplier", "Mish residual channel width"),
                base=8,
                reference=0.25,
                field="width_multiplier",
            )
            stage_depths = _stage_depths(
                _use(self, parameters, "stage_depths", "Mish residual block counts"),
                field="stage_depths",
            )
            self.conv1 = ExactConv2d(3, width, 3, padding=1)
            self.stages = nn.ModuleList(
                nn.ModuleList(
                    ExactConv2d(width, width, 3, padding=1) for _ in range(stage_depth)
                )
                for stage_depth in stage_depths
            )
            self.stage_transitions = nn.ModuleList(
                ExactConv2d(width, width, 1) for _ in range(len(stage_depths) - 1)
            )
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "prelu_elu_cnn":
            self.activation_name = str(
                _use(self, parameters, "activation", "per-layer nonlinear activation")
            ).lower()
            block_count = _depth(
                _use(self, parameters, "depth", "convolution layer count"), field="depth"
            )
            width = _positive_int(
                _use(self, parameters, "width", "convolution channel width"), field="width"
            )
            self.conv1 = ExactConv2d(3, width, 3, padding=1)
            self.blocks = nn.ModuleList(ExactConv2d(width, width, 3, padding=1) for _ in range(max(0, block_count - 1)))
            self.prelus = nn.ModuleList(nn.PReLU(width) for _ in range(block_count))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "unet_groupnorm":
            block_count = _depth(
                _use(self, parameters, "depth", "U-Net encoder/bottleneck depth"), field="depth"
            )
            width = _positive_int(
                _use(self, parameters, "base_channels", "U-Net base channel width"),
                field="base_channels",
            )
            if width % 2:
                width += 1
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.down = ExactConv2d(3, width, 3, padding=1)
            self.bottlenecks = nn.ModuleList(ExactConv2d(width, width, 3, padding=1) for _ in range(block_count))
            self.up = ExactConv2d(width, 3, 3, padding=1)
        elif family_id == "pix2pix":
            generator_width = _positive_int(
                _use(self, parameters, "generator_width", "generator encoder width"),
                field="generator_width",
            )
            discriminator_width = _positive_int(
                _use(self, parameters, "discriminator_width", "discriminator feature width"),
                field="discriminator_width",
            )
            self.input_resolution = _positive_int(
                _use(self, parameters, "input_resolution", "paired-image resize contract"),
                field="input_resolution",
                maximum=512,
            )
            self.generator_down = ExactConv2d(3, generator_width, 4, stride=2, padding=1)
            self.generator_up = ExactConvTranspose2d(generator_width, 3, 4, stride=2, padding=1)
            self.discriminator_features = ExactConv2d(6, discriminator_width, 3, padding=1)
            self.discriminator = ExactConv2d(discriminator_width, 1, 1)
        else:
            raise ModelFactoryError(f"unknown vision family {family_id!r}")

    def _classification_head(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(value.mean(dim=(-2, -1)))

    def forward(self, inputs: Mapping[str, Any]) -> Any:
        image = inputs["image"]
        original_size = image.shape[-2:]
        if hasattr(self, "input_resolution"):
            image = _resize_image(image, self.input_resolution)
        if self.family_id == "resnet50":
            residual = F.relu(self.norm1(self.conv1(image)))
            for stage_index, (stage, norms) in enumerate(zip(self.stages, self.stage_norms)):
                for block, norm in zip(stage, norms):
                    residual = F.relu(torch.ops.aten.add.Tensor(norm(block(residual)), residual))
                if stage_index < len(self.stage_transitions):
                    residual = F.avg_pool2d(
                        F.relu(self.stage_transitions[stage_index](residual)), 2
                    )
            return self._classification_head(residual)
        if self.family_id == "efficientnet_b0_b4":
            value = torch.ops.aten.silu.default(self.conv1(image))
            for block in self.depthwise_blocks:
                value = torch.ops.aten.silu.default(block(value))
            return self._classification_head(value)
        if self.family_id == "mobilenet_v3_large":
            return self._classification_head(torch.ops.aten.hardswish.default(self.depthwise(torch.ops.aten.hardswish.default(self.conv1(image)))))
        if self.family_id == "inception_v3":
            branch1, branch3 = F.relu(self.branch1(image)), F.relu(self.branch3(image))
            output = self._classification_head(torch.ops.aten.cat.default([branch1, branch3], 1))
            return output + 0.1 * self.aux_head(branch1.mean(dim=(-2, -1))) if self.aux_logits else output
        if self.family_id in {"vit_s16", "swin_t", "restormer"}:
            tokens = self.patch(image).flatten(2).transpose(1, 2)
            block_index = 0
            for stage_index, stage in enumerate(self.attention_stages):
                for projection in stage:
                    attended = (
                        _window_attention(
                            tokens,
                            heads=self.attention_heads,
                            window_size=self.window_size,
                            shifted=bool(block_index % 2),
                        )
                        if self.family_id == "swin_t"
                        else _attention(
                            tokens,
                            heads=self.attention_heads,
                            math_backend=self.family_id in {"vit_s16", "restormer"},
                        )
                    )
                    tokens = _layer_norm(
                        torch.ops.aten.add.Tensor(projection(attended), tokens)
                    )
                    block_index += 1
                if stage_index < len(self.attention_stage_transitions):
                    tokens = torch.ops.aten.gelu.default(
                        self.attention_stage_transitions[stage_index](tokens),
                        approximate="none",
                    )
            if self.family_id == "restormer":
                height = width = int(tokens.shape[1] ** 0.5)
                feature = tokens.transpose(1, 2).reshape(image.shape[0], tokens.shape[-1], height, width)
                restored = self.restore(F.interpolate(feature, size=image.shape[-2:], mode="bilinear", align_corners=False))
                return F.interpolate(restored, size=original_size, mode="bilinear", align_corners=False)
            return self.head(tokens.mean(dim=1))
        if self.family_id == "mlp_mixer_s":
            image = _resize_image(image, self.mixer_resolution)
            tokens = self.patch(image).flatten(2).transpose(1, 2)
            for token_in, token_out, channel in zip(self.token_in, self.token_out, self.channel_mixers):
                mixed = token_out(
                    torch.ops.aten.gelu.default(
                        token_in(tokens.transpose(1, 2)), approximate="none"
                    )
                ).transpose(1, 2)
                tokens = torch.ops.aten.add.Tensor(tokens, mixed)
                tokens = torch.ops.aten.gelu.default(channel(tokens), approximate="none")
            return self.head(tokens.mean(dim=1))
        if self.family_id == "stn_cnn":
            localized = self.localization(image).mean(dim=(-2, -1))
            theta = self.theta(localized).reshape(-1, 2, 3)
            grid = F.affine_grid(theta, image.shape, align_corners=False)
            transformed = torch.ops.aten.grid_sampler.default(image, grid, 0, 0, False)
            return self._classification_head(F.relu(self.conv1(transformed)))
        if self.family_id == "mish_resnet":
            residual = torch.ops.aten.mish.default(self.conv1(image))
            for stage_index, stage in enumerate(self.stages):
                for block in stage:
                    residual = torch.ops.aten.add.Tensor(
                        torch.ops.aten.mish.default(block(residual)), residual
                    )
                if stage_index < len(self.stage_transitions):
                    residual = F.avg_pool2d(
                        torch.ops.aten.mish.default(
                            self.stage_transitions[stage_index](residual)
                        ),
                        2,
                    )
            return self._classification_head(residual)
        if self.family_id == "prelu_elu_cnn":
            value = self.conv1(image)
            layers = [None, *self.blocks]
            for index, layer in enumerate(layers):
                if layer is not None:
                    value = layer(value)
                value = (
                    self.prelus[index](value)
                    if self.activation_name == "prelu"
                    else torch.ops.aten.elu.default(value, 1.0, 1.0, 1.0)
                )
            return self._classification_head(value)
        if self.family_id == "unet_groupnorm":
            value = torch.ops.aten.group_norm.default(self.down(image), 2, None, None, 1e-5, False)
            value = F.avg_pool2d(F.relu(value), 2)
            for block in self.bottlenecks:
                value = F.relu(block(value))
            value = F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
            return F.interpolate(self.up(value), size=original_size, mode="bilinear", align_corners=False)
        if self.family_id == "pix2pix":
            fake = torch.tanh(self.generator_up(F.leaky_relu(self.generator_down(image), 0.2)))
            real = inputs.get("real_target")
            real = _resize_image(real, self.input_resolution) if isinstance(real, torch.Tensor) else image.mul(0.8).add(0.1)
            real_score = self.discriminator(F.leaky_relu(self.discriminator_features(torch.cat((image, real), dim=1)), 0.2))
            fake_score = self.discriminator(F.leaky_relu(self.discriminator_features(torch.cat((image, fake), dim=1)), 0.2))
            fake_output = F.interpolate(fake, size=original_size, mode="bilinear", align_corners=False)
            return {"fake": fake_output, "real_score": real_score, "fake_score": fake_score}
        raise AssertionError("unreachable vision family")

    def compute_loss(self, output: Any, batch: Mapping[str, Any], adapter: TaskAdapter) -> torch.Tensor:
        target = adapter.build_targets(batch)
        if self.family_id == "pix2pix":
            real_score, fake_score = output["real_score"], output["fake_score"]
            discriminator = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
            discriminator = discriminator + F.binary_cross_entropy_with_logits(fake_score.detach(), torch.zeros_like(fake_score))
            generator = F.binary_cross_entropy_with_logits(fake_score, torch.ones_like(fake_score))
            reconstruction = F.l1_loss(output["fake"], target)
            return discriminator + generator + reconstruction
        return adapter.build_loss(output, target)

    def metric_output(self, output: Any) -> torch.Tensor:
        return output["fake"] if self.family_id == "pix2pix" else super().metric_output(output)


def _grouped_query_attention(
    query: torch.Tensor,
    key_value: torch.Tensor,
    *,
    heads: int,
    kv_heads: int,
) -> torch.Tensor:
    batch, length, width = query.shape
    if width % heads or heads % kv_heads:
        raise ModelFactoryError("GQA heads must divide model width and query heads")
    head_width = width // heads
    q = query.reshape(batch, length, heads, head_width).transpose(1, 2)
    kv = key_value.reshape(batch, length, kv_heads, head_width).transpose(1, 2)
    kv = kv.repeat_interleave(heads // kv_heads, dim=1)
    attended = torch.ops.aten.scaled_dot_product_attention.default(q, kv, kv)
    return attended.transpose(1, 2).reshape(batch, length, width)


class NLPFamilyModel(FamilyModel):
    def __init__(self, family_id: str, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = family_id
        self.output_width = config.output_width
        self.task_kind = config.task_kind
        parameters = _parameters(config)
        width = 8
        vocabulary = 64
        if family_id in {"bert_base", "gpt2_small"}:
            self.sequence_length = _positive_int(
                _use(self, parameters, "sequence_length", "token context length"),
                field="sequence_length",
                maximum=512,
            )
            layers = _depth(
                _use(self, parameters, "layer_count", "transformer encoder layer count"),
                field="layer_count",
            )
            width = _positive_int(
                _use(self, parameters, "hidden_size", "transformer hidden width"),
                field="hidden_size",
            )
            heads = 2 if width % 2 == 0 else 1
            self.attention_heads = heads
            self.embedding = nn.Embedding(vocabulary, width)
            self.projections = nn.ModuleList(ExactLinear(width, width) for _ in range(layers))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "t5_small":
            encoder_layers = _depth(
                _use(self, parameters, "encoder_layers", "encoder transformer layer count"),
                field="encoder_layers",
            )
            decoder_layers = _depth(
                _use(self, parameters, "decoder_layers", "decoder transformer layer count"),
                field="decoder_layers",
            )
            self.sequence_length = _positive_int(
                _use(self, parameters, "sequence_length", "encoder/decoder context length"),
                field="sequence_length",
                maximum=512,
            )
            self.embedding = nn.Embedding(vocabulary, width)
            self.decoder_embedding = nn.Embedding(vocabulary, width)
            self.encoder_blocks = nn.ModuleList(ExactLinear(width, width) for _ in range(encoder_layers))
            self.decoder_blocks = nn.ModuleList(ExactLinear(width, width) for _ in range(decoder_layers))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "llama_small":
            layers = _depth(_use(self, parameters, "layers", "decoder block count"), field="layers")
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "query attention head count"), field="heads", maximum=16
            )
            self.kv_heads = _positive_int(
                _use(self, parameters, "kv_heads", "grouped key/value head count"), field="kv_heads", maximum=16
            )
            if self.attention_heads % self.kv_heads:
                raise ModelFactoryError("kv_heads must divide heads")
            width = _positive_int(
                _use(self, parameters, "hidden_size", "decoder hidden width"), field="hidden_size"
            )
            width = max(self.attention_heads * 2, math.ceil(width / self.attention_heads) * self.attention_heads)
            self.embedding = nn.Embedding(vocabulary, width)
            self.query_blocks = nn.ModuleList(ExactLinear(width, width) for _ in range(layers))
            kv_width = self.kv_heads * (width // self.attention_heads)
            self.kv_blocks = nn.ModuleList(ExactLinear(width, kv_width) for _ in range(layers))
            self.output_blocks = nn.ModuleList(ExactLinear(width, width) for _ in range(layers))
            self.rms_weight = nn.Parameter(torch.ones(width))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "mla_mini_transformer":
            latent_rank = _positive_int(
                _use(self, parameters, "latent_rank", "latent key/value projection rank"), field="latent_rank"
            )
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "latent attention head count"), field="heads", maximum=16
            )
            layers = _depth(_use(self, parameters, "depth", "latent attention block count"), field="depth")
            width = self.attention_heads * 4
            self.embedding = nn.Embedding(vocabulary, width)
            self.down_blocks = nn.ModuleList(ExactLinear(width, latent_rank) for _ in range(layers))
            self.up_blocks = nn.ModuleList(ExactLinear(latent_rank, width) for _ in range(layers))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "kimi_delta_attention":
            state_size = _positive_int(
                _use(self, parameters, "state_size", "delta-attention recurrent state width"), field="state_size"
            )
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "delta-attention head count"), field="heads", maximum=16
            )
            layers = _depth(_use(self, parameters, "depth", "delta-attention block count"), field="depth")
            width = self.attention_heads * 4
            self.embedding = nn.Embedding(vocabulary, width)
            self.state_in = nn.ModuleList(ExactLinear(width, state_size) for _ in range(layers))
            self.state_out = nn.ModuleList(ExactLinear(state_size, width) for _ in range(layers))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "switch_moe":
            expert_count = _positive_int(
                _use(self, parameters, "expert_count", "number of routed feed-forward experts"), field="expert_count"
            )
            self.top_k = _positive_int(
                _use(self, parameters, "top_k", "experts selected per token"), field="top_k", maximum=expert_count
            )
            capacity = _use(
                self,
                parameters,
                "capacity_factor",
                "per-expert routed-token capacity and overflow dropping",
            )
            if type(capacity) not in {int, float} or float(capacity) <= 0:
                raise ModelFactoryError("capacity_factor must be positive")
            self.capacity_factor = float(capacity)
            self.embedding = nn.Embedding(vocabulary, width)
            self.router = ExactLinear(width, expert_count)
            self.experts = nn.ModuleList(ExactLinear(width, width) for _ in range(expert_count))
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "bilstm_crf":
            hidden_size = _positive_int(
                _use(self, parameters, "hidden_size", "per-direction recurrent hidden width"), field="hidden_size"
            )
            self.recurrent_layers = _depth(
                _use(self, parameters, "layers", "stacked bidirectional LSTM layer count"), field="layers"
            )
            tag_count = _positive_int(
                _use(self, parameters, "tag_count", "CRF emission and transition label count"), field="tag_count"
            )
            self.embedding = nn.Embedding(vocabulary, width)
            self.recurrent = nn.LSTM(
                width,
                hidden_size,
                num_layers=self.recurrent_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.head = ExactLinear(hidden_size * 2, tag_count)
            self.transitions = nn.Parameter(torch.zeros(tag_count, tag_count))
        elif family_id == "gru_rnn_seq2seq":
            width = _positive_int(
                _use(self, parameters, "hidden_size", "GRU hidden width"), field="hidden_size"
            )
            self.recurrent_layers = _depth(
                _use(self, parameters, "layers", "stacked GRU layer count"), field="layers"
            )
            self.sequence_length = _positive_int(
                _use(self, parameters, "sequence_length", "encoder/decoder context length"),
                field="sequence_length",
                maximum=512,
            )
            self.embedding = nn.Embedding(vocabulary, width)
            self.recurrent = nn.GRU(width, width, num_layers=self.recurrent_layers, batch_first=True)
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "fasttext_embeddingbag":
            width = _positive_int(
                _use(self, parameters, "embedding_dim", "subword embedding width"), field="embedding_dim"
            )
            vocabulary = _positive_int(
                _use(self, parameters, "ngram_buckets", "hashed subword bucket count"),
                field="ngram_buckets",
                maximum=1_000_000,
            )
            sparse_gradients = _use(
                self,
                parameters,
                "sparse_gradients",
                "SparseAdam-compatible sparse EmbeddingBag gradient route",
            )
            if type(sparse_gradients) is not bool:
                raise ModelFactoryError("sparse_gradients must be boolean")
            self.sparse_gradients = sparse_gradients
            self.register_buffer(
                "sparse_gradient_contract",
                torch.ones(1) if sparse_gradients else torch.empty(0),
            )
            self.embedding = nn.Embedding(vocabulary, width)
            self.embedding_bag = nn.EmbeddingBag(
                vocabulary, width, mode="mean", sparse=sparse_gradients
            )
            self.head = ExactLinear(width, config.output_width)
            if sparse_gradients:
                for parameter in self.embedding.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.head.parameters():
                    parameter.requires_grad_(False)
        elif family_id == "distilbert_distillation":
            layers = _depth(
                _use(self, parameters, "student_layers", "student transformer layer count"),
                field="student_layers",
            )
            self.sequence_length = _positive_int(
                _use(self, parameters, "sequence_length", "student/teacher context length"),
                field="sequence_length",
                maximum=512,
            )
            temperature = _use(self, parameters, "temperature", "KL distillation temperature")
            if type(temperature) not in {int, float} or float(temperature) <= 0:
                raise ModelFactoryError("temperature must be positive")
            self.distillation_temperature = float(temperature)
            self.embedding = nn.Embedding(vocabulary, width)
            self.student_blocks = nn.ModuleList(ExactLinear(width, width) for _ in range(layers))
            self.head = ExactLinear(width, config.output_width)
            self.teacher_projection = ExactLinear(width, width)
            self.teacher_head = ExactLinear(width, config.output_width)
        else:
            raise ModelFactoryError(f"unknown NLP family {family_id!r}")

    def _token_ids(self, inputs: Mapping[str, Any], name: str = "token_ids") -> torch.Tensor:
        value = inputs[name]
        # The architecture context constrains the encoder input only.  Real
        # normalization tasks may have a different decoder/target length, so
        # teacher-forcing inputs must retain their independently declared
        # decoder length.
        if name == "token_ids" and hasattr(self, "sequence_length"):
            value = _resize_sequence(value, self.sequence_length)
        return value.remainder(self.embedding.num_embeddings)

    def _tokens(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        return torch.ops.aten.embedding.default(
            self.embedding.weight, self._token_ids(inputs), -1, False, False
        )

    def forward(self, inputs: Mapping[str, Any]) -> Any:
        tokens = self._tokens(inputs)
        if self.family_id in {"bert_base", "gpt2_small"}:
            value = tokens
            for projection in self.projections:
                value = _layer_norm(
                    torch.ops.aten.add.Tensor(
                        _attention(projection(value), heads=self.attention_heads), value
                    )
                )
            return self.head(value.mean(dim=1))
        if self.family_id == "t5_small":
            encoded = tokens
            for block in self.encoder_blocks:
                encoded = _layer_norm(_attention(block(encoded)))
            decoded = torch.ops.aten.embedding.default(
                self.decoder_embedding.weight,
                self._token_ids(inputs, "decoder_ids"),
                -1,
                False,
                False,
            )
            for block in self.decoder_blocks:
                decoded = _layer_norm(
                    _attention(
                        torch.ops.aten.add.Tensor(
                            block(decoded), encoded.mean(dim=1, keepdim=True)
                        )
                    )
                )
            return self.head(decoded)
        if self.family_id == "llama_small":
            value = torch.ops.aten.rms_norm.default(
                tokens, [tokens.shape[-1]], self.rms_weight, None
            )
            for query, kv, output in zip(self.query_blocks, self.kv_blocks, self.output_blocks):
                attended = _grouped_query_attention(
                    query(value),
                    kv(value),
                    heads=self.attention_heads,
                    kv_heads=self.kv_heads,
                )
                value = torch.ops.aten.silu.default(
                    torch.ops.aten.add.Tensor(output(attended), value)
                )
            return self.head(value.mean(dim=1))
        if self.family_id == "mla_mini_transformer":
            value = tokens
            for down, up in zip(self.down_blocks, self.up_blocks):
                latent = down(value)
                score = torch.ops.aten.matmul.default(latent, latent.transpose(-1, -2))
                update = up(_attention(latent, heads=1))
                value = torch.ops.aten.add.Tensor(update, torch.matmul(score.softmax(-1), value))
            return self.head(value.mean(dim=1))
        if self.family_id == "kimi_delta_attention":
            if self.task_kind == "teacher_forced_seq2seq":
                decoder_tokens = torch.ops.aten.embedding.default(
                    self.embedding.weight,
                    self._token_ids(inputs, "decoder_ids"),
                    -1,
                    False,
                    False,
                )
                value = torch.ops.aten.add.Tensor(
                    decoder_tokens, tokens.mean(dim=1, keepdim=True)
                )
            else:
                value = tokens
            for state_in, state_out in zip(self.state_in, self.state_out):
                delta = torch.ops.aten.cumsum.default(state_in(value), 1)
                restored = state_out(delta)
                score = torch.ops.aten.matmul.default(restored, value.transpose(-1, -2)).softmax(-1)
                value = torch.ops.aten.silu.default(torch.ops.aten.matmul.default(score, value))
            return self.head(value.mean(dim=1)) if self.task_kind != "teacher_forced_seq2seq" else self.head(value)
        if self.family_id == "switch_moe":
            batch_size, sequence_length, hidden_width = tokens.shape
            flat_tokens = tokens.reshape(-1, hidden_width)
            # Routing probabilities and their capacity renormalization are
            # numerically sensitive in FP16; keep this control path in FP32.
            routes = self.router(flat_tokens).float()
            route_values, indices = torch.ops.aten.topk.default(
                routes, self.top_k, -1, True, True
            )
            expert_outputs = torch.stack(
                [expert(flat_tokens) for expert in self.experts], dim=1
            )
            selected = torch.ops.aten.gather.default(
                expert_outputs,
                1,
                indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]),
            )
            capacity = max(
                1,
                math.ceil(
                    self.capacity_factor
                    * flat_tokens.shape[0]
                    * self.top_k
                    / len(self.experts)
                ),
            )
            keep_masks = []
            for expert_index in range(len(self.experts)):
                assigned = indices.eq(expert_index)
                assignment_rank = assigned.reshape(-1).cumsum(0).reshape_as(assigned)
                keep_masks.append(assigned & assignment_rank.le(capacity))
            keep = torch.stack(keep_masks, dim=-1).any(dim=-1)
            weights = route_values.softmax(-1) * keep.to(route_values.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            routed = (selected.float() * weights.unsqueeze(-1)).sum(dim=1)
            routed = routed.reshape(batch_size, sequence_length, hidden_width).mean(dim=1)
            return self.head(routed.to(flat_tokens.dtype))
        if self.family_id == "bilstm_crf":
            batch_size = tokens.shape[0]
            hidden = torch.zeros(
                (self.recurrent_layers * 2, batch_size, self.recurrent.hidden_size),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            cell = torch.zeros_like(hidden)
            recurrent, _, _ = torch.ops.aten.lstm.input(
                tokens,
                [hidden, cell],
                [value for value in self.recurrent._flat_weights if value is not None],
                True,
                self.recurrent_layers,
                0.0,
                self.training,
                True,
                True,
            )
            return self.head(recurrent)
        if self.family_id == "gru_rnn_seq2seq":
            decoder = torch.ops.aten.embedding.default(
                self.embedding.weight,
                self._token_ids(inputs, "decoder_ids"),
                -1,
                False,
                False,
            )
            hidden = torch.zeros(
                (self.recurrent_layers, tokens.shape[0], tokens.shape[-1]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            recurrent_parameters = [
                value for value in self.recurrent._flat_weights if value is not None
            ]
            _, hidden = torch.ops.aten.gru.input(
                tokens, hidden, recurrent_parameters, True, self.recurrent_layers, 0.0, self.training, False, True
            )
            decoded, _ = torch.ops.aten.gru.input(
                decoder, hidden, recurrent_parameters, True, self.recurrent_layers, 0.0, self.training, False, True
            )
            return self.head(decoded)
        if self.family_id == "fasttext_embeddingbag":
            token_ids = self._token_ids(inputs)
            flat = token_ids.reshape(-1)
            offsets = torch.arange(0, flat.numel(), token_ids.shape[1], device=flat.device)
            pooled = torch.ops.aten.embedding_bag.padding_idx(
                self.embedding_bag.weight,
                flat,
                offsets,
                False,
                1,
                self.sparse_gradients,
                None,
                False,
                None,
            )[0]
            return self.head(pooled)
        if self.family_id == "distilbert_distillation":
            student_hidden = tokens
            for block in self.student_blocks:
                student_hidden = _layer_norm(_attention(block(student_hidden)))
            student = self.head(student_hidden.mean(dim=1))
            with torch.no_grad():
                teacher_hidden = _layer_norm(_attention(self.teacher_projection(tokens)))
                teacher = self.teacher_head(teacher_hidden.mean(dim=1))
            return {"student": student, "teacher": teacher}
        raise AssertionError("unreachable NLP family")

    def compute_loss(self, output: Any, batch: Mapping[str, Any], adapter: TaskAdapter) -> torch.Tensor:
        target = adapter.build_targets(batch)
        if self.family_id == "bilstm_crf":
            tags = target.long().unsqueeze(1).expand(-1, output.shape[1]).remainder(output.shape[-1])
            mask = _resize_sequence(batch["inputs"]["attention_mask"], output.shape[1]).to(output.dtype)
            gold = (torch.gather(output, -1, tags.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=1)
            partition = (torch.ops.aten.logsumexp.default(output, [-1], False) * mask).sum(dim=1)
            transition_score = (
                self.transitions[tags[:, :-1], tags[:, 1:]] * mask[:, 1:]
            ).sum(dim=1)
            return (partition - gold - transition_score).mean()
        if self.family_id == "distilbert_distillation":
            temperature = self.distillation_temperature
            supervised = adapter.build_loss(output["student"], target)
            distillation = torch.ops.aten.kl_div.default(
                F.log_softmax(output["student"] / temperature, dim=-1),
                F.softmax(output["teacher"] / temperature, dim=-1),
                2,
                log_target=False,
            ) * temperature**2
            return supervised + distillation
        return adapter.build_loss(output, target)

    def metric_output(self, output: Any) -> torch.Tensor:
        if self.family_id == "bilstm_crf":
            self.decode_crf(output)
            return output.mean(dim=1)
        if self.family_id == "distilbert_distillation":
            return output["student"]
        return super().metric_output(output)

    def decode_crf(self, emissions: torch.Tensor) -> torch.Tensor:
        if self.family_id != "bilstm_crf":
            raise ModelFactoryError("CRF decoding is only defined for bilstm_crf")
        score = emissions[:, 0]
        backpointers = []
        for timestep in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            score, previous = candidates.max(dim=1)
            score = score + emissions[:, timestep]
            backpointers.append(previous)
        current = score.argmax(dim=-1)
        path = [current]
        for previous in reversed(backpointers):
            current = torch.gather(previous, 1, current.unsqueeze(1)).squeeze(1)
            path.append(current)
        return torch.stack(list(reversed(path)), dim=1)


class AudioFamilyModel(FamilyModel):
    def __init__(self, family_id: str, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = family_id
        parameters = _parameters(config)
        if family_id == "panns_cnn14":
            sample_rate = _positive_int(
                _use(self, parameters, "sample_rate", "STFT time/frequency sampling rate"),
                field="sample_rate",
                maximum=192_000,
            )
            mel_bins = _positive_int(
                _use(self, parameters, "mel_bins", "log-mel filterbank channel count"),
                field="mel_bins",
                maximum=256,
            )
            self.clip_samples = _positive_int(
                _use(self, parameters, "clip_samples", "waveform clip length"),
                field="clip_samples",
                maximum=1_000_000,
            )
            self.n_fft = max(8, min(128, 4 * round(sample_rate / 2_000)))
            if self.n_fft % 2:
                self.n_fft += 1
            self.hop_length = max(2, self.n_fft // 2)
            self.conv = ExactConv2d(1, 8, 3, padding=1)
            self.norm = ExactBatchNorm2d(8)
            self.head = ExactLinear(8, config.output_width)
            self.register_buffer("window", torch.hann_window(self.n_fft))
            frequency = torch.linspace(0.0, 1.0, self.n_fft // 2 + 1)
            centers = torch.linspace(0.0, 1.0, mel_bins)
            filters = (1.0 - (frequency.unsqueeze(0) - centers.unsqueeze(1)).abs() * mel_bins).clamp_min(0.0)
            self.register_buffer("mel_filter", filters / filters.sum(dim=1, keepdim=True).clamp_min(1e-6))
        elif family_id == "temporal_convolutional_network":
            levels = _depth(
                _use(self, parameters, "levels", "temporal residual level count"), field="levels"
            )
            channels = _positive_int(
                _use(self, parameters, "channels", "temporal convolution channel width"), field="channels"
            )
            kernel = _positive_int(
                _use(self, parameters, "kernel_size", "temporal convolution receptive field"),
                field="kernel_size",
                maximum=31,
            )
            if kernel % 2 == 0:
                raise ModelFactoryError("temporal convolution kernel_size must be odd")
            self.conv1 = ExactConv1d(1, channels, kernel, padding=kernel // 2)
            self.temporal_blocks = nn.ModuleList(
                ExactConv1d(channels, channels, kernel, padding=kernel // 2)
                for _ in range(levels)
            )
            self.head = ExactLinear(channels, config.output_width)
        elif family_id == "m5_waveform_cnn":
            channels = _positive_int(
                _use(self, parameters, "channels", "M5 convolution channel width"), field="channels"
            )
            kernel = _positive_int(
                _use(self, parameters, "kernel_size", "raw-waveform convolution kernel"),
                field="kernel_size",
                maximum=31,
            )
            if kernel % 2 == 0:
                raise ModelFactoryError("M5 kernel_size must be odd")
            self.clip_samples = _positive_int(
                _use(self, parameters, "clip_samples", "raw-waveform clip length"),
                field="clip_samples",
                maximum=1_000_000,
            )
            self.conv1 = ExactConv1d(1, channels, kernel, padding=kernel // 2)
            self.conv2 = ExactConv1d(channels, channels, kernel, padding=kernel // 2)
            self.head = ExactLinear(channels, config.output_width)
        else:
            raise ModelFactoryError(f"unknown audio family {family_id!r}")

    def forward(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        waveform = inputs["waveform"]
        if hasattr(self, "clip_samples") and waveform.shape[-1] != self.clip_samples:
            waveform = F.interpolate(waveform, size=self.clip_samples, mode="linear", align_corners=False)
        if self.family_id == "panns_cnn14":
            spectrum = torch.stft(
                waveform.squeeze(1),
                self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
            )
            power = spectrum.abs().pow(2)
            mel = torch.matmul(self.mel_filter, power)
            value = mel.clamp_min(1e-6).log().unsqueeze(1)
            value = F.relu(self.norm(self.conv(value))).mean(dim=(-2, -1))
            return torch.ops.aten.log_softmax.int(self.head(value), -1)
        value = F.relu(self.conv1(waveform))
        if self.family_id == "temporal_convolutional_network":
            for block in self.temporal_blocks:
                value = F.relu(torch.ops.aten.add.Tensor(block(value), value))
        else:
            value = torch.ops.aten.max_pool1d.default(self.conv2(value), [2], [2], [0], [1], False)
        return self.head(value.mean(dim=-1))


class TabularFamilyModel(FamilyModel):
    def __init__(self, family_id: str, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = family_id
        self.output_width = config.output_width
        parameters = _parameters(config)
        if family_id == "selu_mlp":
            depth = _depth(_use(self, parameters, "depth", "SELU hidden layer count"), field="depth")
            width = _positive_int(
                _use(self, parameters, "width", "SELU hidden layer width"), field="width"
            )
            self.feature_count = _positive_int(
                _use(self, parameters, "feature_count", "dense input feature contract"),
                field="feature_count",
            )
            self.hidden_layers = nn.ModuleList(
                ExactLinear(self.feature_count if index == 0 else width, width)
                for index in range(depth)
            )
            self.head = ExactLinear(width, config.output_width)
        elif family_id == "tabtransformer":
            depth = _depth(
                _use(self, parameters, "depth", "categorical transformer block count"), field="depth"
            )
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "categorical attention head count"), field="heads", maximum=16
            )
            embedding_dim = _positive_int(
                _use(self, parameters, "embedding_dim", "categorical embedding width"), field="embedding_dim"
            )
            embedding_dim = max(
                self.attention_heads,
                math.ceil(embedding_dim / self.attention_heads) * self.attention_heads,
            )
            self.embeddings = nn.ModuleList([nn.Embedding(8, embedding_dim) for _ in range(3)])
            self.transformer_blocks = nn.ModuleList(
                ExactLinear(embedding_dim, embedding_dim) for _ in range(depth)
            )
            self.head = ExactLinear(3 * embedding_dim + 8, config.output_width)
        elif family_id == "mixture_density_network":
            self.components = _positive_int(
                _use(self, parameters, "components", "mixture component count"), field="components"
            )
            depth = _depth(
                _use(self, parameters, "depth", "density-network hidden layer count"), field="depth"
            )
            width = _positive_int(
                _use(self, parameters, "width", "density-network hidden width"), field="width"
            )
            self.hidden_layers = nn.ModuleList(
                ExactLinear(8 if index == 0 else width, width) for index in range(depth)
            )
            self.mixture = ExactLinear(width, self.components * 3)
        else:
            raise ModelFactoryError(f"unknown tabular family {family_id!r}")

    def forward(self, inputs: Mapping[str, Any]) -> Any:
        dense = inputs["dense"]
        if self.family_id == "selu_mlp":
            value = _resize_last_dimension(dense, self.feature_count)
            for layer in self.hidden_layers:
                value = torch.ops.aten.selu.default(layer(value))
            return self.head(value)
        if self.family_id == "tabtransformer":
            categorical = inputs["categorical"]
            embedded = torch.stack(
                [torch.ops.aten.embedding.default(layer.weight, categorical[:, index], -1, False, False) for index, layer in enumerate(self.embeddings)],
                dim=1,
            )
            attended = embedded
            for block in self.transformer_blocks:
                attended = _layer_norm(
                    torch.ops.aten.add.Tensor(
                        _attention(block(attended), heads=self.attention_heads), attended
                    )
                )
            attended = attended.flatten(1)
            return self.head(torch.cat((dense, attended), dim=-1))
        hidden = dense
        for layer in self.hidden_layers:
            hidden = F.silu(layer(hidden))
        raw = self.mixture(hidden).reshape(dense.shape[0], self.components, 3)
        return {"logits": raw[..., 0], "means": raw[..., 1], "log_scales": raw[..., 2].clamp(-4, 4)}

    def compute_loss(self, output: Any, batch: Mapping[str, Any], adapter: TaskAdapter) -> torch.Tensor:
        if self.family_id != "mixture_density_network":
            return super().compute_loss(output, batch, adapter)
        target = adapter.build_targets(batch).squeeze(-1).unsqueeze(-1)
        inverse_scale = torch.exp(-output["log_scales"])
        component = -0.5 * ((target - output["means"]) * inverse_scale).pow(2) - output["log_scales"]
        log_weights = F.log_softmax(output["logits"], dim=-1)
        return -torch.ops.aten.logsumexp.default(component + log_weights, [-1], False).mean()

    def metric_output(self, output: Any) -> torch.Tensor:
        if self.family_id != "mixture_density_network":
            return super().metric_output(output)
        weights = F.softmax(output["logits"], dim=-1)
        return (weights * output["means"]).sum(dim=-1, keepdim=True)


class GraphFamilyModel(FamilyModel):
    def __init__(self, family_id: str, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = family_id
        parameters = _parameters(config)
        if family_id == "cgcnn":
            layers = _depth(
                _use(self, parameters, "message_layers", "crystal message-passing layer count"),
                field="message_layers",
            )
            input_width = _positive_int(
                _use(self, parameters, "atom_features", "crystal atom feature width"),
                field="atom_features",
            )
            self.neighbor_limit = _positive_int(
                _use(self, parameters, "neighbor_limit", "maximum crystal edges consumed"),
                field="neighbor_limit",
                maximum=1_000_000,
            )
            hidden = 8
        elif family_id == "gcn":
            layers = _depth(_use(self, parameters, "layers", "GCN message layer count"), field="layers")
            hidden = _positive_int(
                _use(self, parameters, "hidden_size", "GCN hidden channel width"), field="hidden_size"
            )
            self.node_count = _positive_int(
                _use(self, parameters, "node_count", "graph node-count contract"), field="node_count", maximum=10_000
            )
            input_width = 6
        elif family_id == "gat":
            layers = _depth(_use(self, parameters, "layers", "GAT message layer count"), field="layers")
            self.attention_heads = _positive_int(
                _use(self, parameters, "heads", "graph attention head count"), field="heads", maximum=16
            )
            hidden = _positive_int(
                _use(self, parameters, "hidden_size", "GAT hidden channel width"), field="hidden_size"
            )
            input_width = 6
        elif family_id == "graphsage":
            layers = _depth(
                _use(self, parameters, "layers", "GraphSAGE aggregation layer count"), field="layers"
            )
            hidden = _positive_int(
                _use(self, parameters, "hidden_size", "GraphSAGE hidden channel width"), field="hidden_size"
            )
            self.neighbor_count = _positive_int(
                _use(self, parameters, "neighbor_count", "sampled neighbors per node"),
                field="neighbor_count",
                maximum=1_000,
            )
            input_width = 6
        else:
            raise ModelFactoryError(f"unknown graph family {family_id!r}")
        self.node = ExactLinear(input_width, hidden)
        self.messages = nn.ModuleList(ExactLinear(hidden, hidden) for _ in range(layers))
        self.head = ExactLinear(hidden, config.output_width)
        if family_id == "gat":
            self.attentions = nn.ModuleList(
                ExactLinear(hidden * 2, self.attention_heads) for _ in range(layers)
            )

    def forward(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        if self.family_id == "cgcnn":
            pyg_batch = inputs.get("pyg_batch")
            if pyg_batch is None:
                raise ModelFactoryError("CGCNN requires the adapter-provided PyG Batch")
            node_features = pyg_batch.x
            edge_index = pyg_batch.edge_index
            edge_features = pyg_batch.edge_attr
            graph_index = pyg_batch.batch
        else:
            node_features = inputs["node_features"]
            edge_index = inputs["edge_index"]
            edge_features = inputs["edge_features"]
            graph_index = inputs["graph_index"]
        if self.family_id == "cgcnn":
            node_features = _resize_last_dimension(node_features, self.node.weight.shape[1])
            edge_index = edge_index[:, : self.neighbor_limit]
            edge_features = edge_features[: self.neighbor_limit]
        elif self.family_id == "graphsage":
            edge_limit = min(edge_index.shape[1], node_features.shape[0] * self.neighbor_count)
            edge_index = edge_index[:, :edge_limit]
            edge_features = edge_features[:edge_limit]
        nodes = F.relu(self.node(node_features))
        source, destination = edge_index
        for layer_index, message_layer in enumerate(self.messages):
            source_values = torch.ops.aten.index_select.default(nodes, 0, source)
            destination_values = torch.ops.aten.index_select.default(nodes, 0, destination)
            if self.family_id == "gat":
                scores = self.attentions[layer_index](
                    torch.cat((source_values, destination_values), dim=-1)
                )
                weights = torch.ops.aten.softmax.int(scores, 0).mean(dim=-1, keepdim=True)
                messages = message_layer(source_values) * weights
            elif self.family_id == "cgcnn":
                edge_scale = edge_features.mean(dim=-1, keepdim=True)
                messages = message_layer(source_values) * torch.sigmoid(edge_scale)
            else:
                messages = message_layer(source_values)
            aggregate = _scatter_add_float32(
                tuple(nodes.shape),
                destination.unsqueeze(-1).expand_as(messages),
                messages,
            )
            if self.family_id == "graphsage":
                nodes = torch.cat((nodes, aggregate), dim=-1).reshape(nodes.shape[0], 2, -1).mean(dim=1)
            else:
                nodes = F.relu(torch.ops.aten.add.Tensor(nodes, aggregate))
        graph_count = inputs.get("graph_count")
        if type(graph_count) is not int or graph_count < 1:
            raise ModelFactoryError("graph adapter must provide a positive graph_count")
        pooled = _scatter_add_float32(
            (graph_count, nodes.shape[-1]),
            graph_index.unsqueeze(-1).expand_as(nodes),
            nodes,
        )
        counts = torch.bincount(graph_index, minlength=graph_count).clamp_min(1).unsqueeze(-1)
        return self.head(pooled / counts)


class GeneratedFamilyModel(FamilyModel):
    def __init__(self, config: FamilyBuildConfig) -> None:
        super().__init__()
        self.family_id = "independent_generated"
        parameters = _parameters(config)
        self.source_lineage = str(
            _use(self, parameters, "source_lineage", "independent topology lineage identity")
        )
        self.architecture_specification = str(
            _use(
                self,
                parameters,
                "architecture_specification",
                "generated topology and branch specification",
            )
        )
        self.generated_modality = str(
            _use(self, parameters, "modality", "generated input stem modality")
        )
        depth = _positive_int(
            _use(self, parameters, "depth", "generated block count"), field="depth", maximum=8
        )
        width = _positive_int(
            _use(self, parameters, "width", "generated channel width"), field="width", maximum=32
        )
        if not 2 <= depth <= 8 or not 4 <= width <= 32:
            raise ModelFactoryError("generated architecture depth/width is outside Phase 3 bounds")
        lineage_digest = canonical_sha256(self.source_lineage)
        self.kernel_size = (1, 3, 5)[int(lineage_digest[:2], 16) % 3]
        input_channels = 3 if self.generated_modality == "vision" else 1
        self.stem = ExactConv2d(
            input_channels,
            width,
            self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.layers = nn.ModuleList(
            ExactConv2d(width, width, self.kernel_size, padding=self.kernel_size // 2)
            for _ in range(depth - 1)
        )
        self.branch_layers = (
            nn.ModuleList(
                ExactConv2d(width, width, 1)
                for _ in range(depth - 1)
            )
            if "branch" in self.architecture_specification
            else nn.ModuleList()
        )
        self.head = ExactLinear(width, config.output_width)

    def forward(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        image = inputs["image"]
        if self.generated_modality != "vision":
            image = image.mean(dim=1, keepdim=True)
        value = F.silu(self.stem(image))
        for index, layer in enumerate(self.layers):
            update = F.gelu(layer(value)) if index % 2 else F.mish(layer(value))
            if self.branch_layers:
                update = torch.ops.aten.add.Tensor(update, F.silu(self.branch_layers[index](value)))
            value = (
                update
                if "sequential" in self.architecture_specification
                else torch.ops.aten.add.Tensor(value, update)
            )
        return self.head(value.mean(dim=(-2, -1)))


VISION_FAMILIES = {
    "resnet50", "efficientnet_b0_b4", "mobilenet_v3_large", "inception_v3",
    "vit_s16", "swin_t", "mlp_mixer_s", "stn_cnn", "mish_resnet",
    "prelu_elu_cnn", "unet_groupnorm", "pix2pix", "restormer",
}
NLP_FAMILIES = {
    "bert_base", "gpt2_small", "t5_small", "llama_small", "mla_mini_transformer",
    "kimi_delta_attention", "switch_moe", "bilstm_crf", "gru_rnn_seq2seq",
    "fasttext_embeddingbag", "distilbert_distillation",
}
AUDIO_FAMILIES = {"panns_cnn14", "temporal_convolutional_network", "m5_waveform_cnn"}
TABULAR_FAMILIES = {"selu_mlp", "tabtransformer", "mixture_density_network"}
GRAPH_FAMILIES = {"cgcnn", "gcn", "gat", "graphsage"}


def build_family_model(family_id: str, config: FamilyBuildConfig) -> FamilyModel:
    config.validate()
    architecture_parameters = resolve_architecture_parameters(
        family_id, config.architecture_parameters
    )
    resolved_config = FamilyBuildConfig(
        output_width=config.output_width,
        task_kind=config.task_kind,
        seed=config.seed,
        architecture_parameters=architecture_parameters,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        if family_id in VISION_FAMILIES:
            model = VisionFamilyModel(family_id, resolved_config)
        elif family_id in NLP_FAMILIES:
            model = NLPFamilyModel(family_id, resolved_config)
        elif family_id in AUDIO_FAMILIES:
            model = AudioFamilyModel(family_id, resolved_config)
        elif family_id in TABULAR_FAMILIES:
            model = TabularFamilyModel(family_id, resolved_config)
        elif family_id in GRAPH_FAMILIES:
            model = GraphFamilyModel(family_id, resolved_config)
        elif family_id == "independent_generated":
            model = GeneratedFamilyModel(resolved_config)
        else:
            raise ModelFactoryError(f"unknown model family {family_id!r}")
        model.bind_architecture(architecture_parameters)
        return model


def default_adapter_id(family_id: str, adapter_ids: tuple[str, ...]) -> str:
    preferred = {
        "unet_groupnorm": "denoising-dirty-documents",
        "pix2pix": "denoising-dirty-documents",
        "restormer": "denoising-dirty-documents",
        "t5_small": "text-normalization-english",
        "kimi_delta_attention": "text-normalization-english",
        "gru_rnn_seq2seq": "text-normalization-english",
        "mixture_density_network": "nyc-taxi-fare",
        "independent_generated": "histopathologic-cancer",
    }.get(family_id)
    if preferred is not None and preferred in adapter_ids:
        return preferred
    return adapter_ids[0]


__all__ = [
    "AUDIO_FAMILIES",
    "GRAPH_FAMILIES",
    "NLP_FAMILIES",
    "TABULAR_FAMILIES",
    "VISION_FAMILIES",
    "build_family_model",
    "default_adapter_id",
]
