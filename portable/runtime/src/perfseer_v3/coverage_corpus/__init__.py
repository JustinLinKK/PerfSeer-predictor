"""Small built-in corpus plus an adapter contract for larger external corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorTree = Any


@dataclass(frozen=True)
class CoverageCase:
    case_id: str
    family: str
    modality: str
    model_factory: Callable[[], nn.Module]
    input_factory: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]
    dynamic_shapes: Any | None = None
    loss_factory: Callable[[], Callable[[TensorTree, TensorTree], torch.Tensor]] | None = None
    target_factory: Callable[[], TensorTree] | None = None
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None
    training: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
        model = self.model_factory()
        args, kwargs = self.input_factory()
        return model, tuple(args), dict(kwargs)


class _ResidualConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, (3, 1), padding=(1, 0))
        self.norm = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x))) + x


class _SequenceBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.query = nn.Linear(16, 16)
        self.key = nn.Linear(16, 16)
        self.value = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / 4.0
        probs = torch.softmax(scores, dim=-1)
        return self.norm(torch.matmul(probs, v) + x), probs


class _IndexReduction(nn.Module):
    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        gathered = torch.gather(x, 1, index)
        values, _ = torch.topk(gathered, k=2, dim=-1)
        return torch.logsumexp(values.clamp_min(1e-4), dim=-1)


class _DenseVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 5)
        self.bilinear = nn.Bilinear(4, 3, 2)

    def forward(
        self,
        sequence: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        batch_left: torch.Tensor,
        batch_right: torch.Tensor,
        vector: torch.Tensor,
        bilinear_left: torch.Tensor,
        bilinear_right: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.linear(sequence),
            torch.mm(left, right),
            torch.matmul(batch_left, batch_right),
            torch.bmm(batch_left, batch_right),
            torch.einsum("bij,bjk->bik", batch_left, batch_right),
            torch.mv(left, vector),
            self.bilinear(bilinear_left, bilinear_right),
        )


class _ConvolutionVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 6, 3, padding=1)
        self.depthwise2 = nn.Conv2d(4, 4, (3, 5), padding=(1, 2), groups=4)
        self.grouped3 = nn.Conv3d(4, 8, 3, padding=1, groups=2)
        self.transpose1 = nn.ConvTranspose1d(6, 4, 3, padding=1)
        self.transpose2 = nn.ConvTranspose2d(4, 3, 3, padding=1)
        self.transpose3 = nn.ConvTranspose3d(8, 4, 3, padding=1)

    def forward(
        self,
        signal: torch.Tensor,
        image: torch.Tensor,
        volume: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        conv1 = self.conv1(signal)
        conv2 = self.depthwise2(image)
        conv3 = self.grouped3(volume)
        return (
            conv1,
            conv2,
            conv3,
            self.transpose1(conv1),
            self.transpose2(conv2),
            self.transpose3(conv3),
        )


class _AttentionVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(16, 4, batch_first=True)

    def forward(
        self,
        sequence: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        attended, weights = self.mha(sequence, sequence, sequence, need_weights=True)
        sdpa = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return attended, weights, sdpa


class _NormalizationVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch = nn.BatchNorm2d(8)
        self.layer = nn.LayerNorm(8)
        self.group = nn.GroupNorm(4, 8)
        self.instance = nn.InstanceNorm2d(8, affine=True, track_running_stats=True)
        self.rms = nn.RMSNorm(8)

    def forward(
        self,
        image: torch.Tensor,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.batch(image),
            self.layer(sequence),
            self.group(image),
            self.instance(image),
            self.rms(sequence),
        )


class _ElementwiseUnaryVariants(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        positive = x.abs() + 0.5
        return (
            x + y,
            x - y,
            x * y,
            x / positive,
            torch.pow(positive, y.abs() + 1.0),
            torch.remainder(x, positive),
            torch.minimum(x, y),
            torch.maximum(x, y),
            torch.clamp(x, -1.0, 1.0),
            torch.where(condition, x, y),
            torch.relu(x),
            F.gelu(x),
            F.silu(x),
            torch.sigmoid(x),
            torch.tanh(x),
            F.leaky_relu(x),
            F.elu(x),
            F.selu(x),
            F.softplus(x),
            F.mish(x),
            torch.exp(x.clamp(max=2.0)),
            torch.log(positive),
            torch.sqrt(positive),
            torch.rsqrt(positive),
            torch.erf(x),
            torch.abs(x),
            torch.sin(x),
            torch.cos(x),
        )


class _ReductionLossVariants(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        binary_target: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        return (
            x.sum(dim=(1, 2)),
            x.mean(dim=-1),
            x.prod(dim=-1),
            torch.amax(x, dim=-1),
            torch.amin(x, dim=-1),
            torch.argmax(x, dim=-1),
            torch.argmin(x, dim=-1),
            torch.linalg.vector_norm(x, dim=-1),
            torch.var(x, dim=-1),
            torch.std(x, dim=-1),
            torch.logsumexp(x, dim=-1),
            probabilities,
            log_probabilities,
            F.cross_entropy(logits, labels),
            F.nll_loss(log_probabilities, labels),
            F.binary_cross_entropy_with_logits(logits, binary_target),
            F.mse_loss(logits, binary_target),
            F.kl_div(log_probabilities, probabilities, reduction="batchmean"),
        )


class _PoolResampleVariants(nn.Module):
    def forward(
        self,
        signal: torch.Tensor,
        image: torch.Tensor,
        volume: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            F.max_pool1d(signal, 3, stride=2, padding=1),
            F.avg_pool1d(signal, 3, stride=2, padding=1),
            F.adaptive_avg_pool1d(signal, 5),
            F.max_pool2d(image, (3, 2), stride=2, padding=(1, 0)),
            F.avg_pool2d(image, (3, 2), stride=2, padding=(1, 0)),
            F.adaptive_avg_pool2d(image, (3, 4)),
            F.max_pool3d(volume, 3, stride=2, padding=1),
            F.avg_pool3d(volume, 3, stride=2, padding=1),
            F.adaptive_avg_pool3d(volume, (2, 3, 4)),
            F.interpolate(image, scale_factor=2.0, mode="bilinear", align_corners=False),
            F.pad(image, (1, 2, 2, 1), mode="constant", value=0.25),
            F.grid_sample(image, grid, mode="bilinear", align_corners=False),
        )


class _LayoutVariants(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[Any, ...]:
        split = torch.split(x, 2, dim=-1)
        chunk = torch.chunk(x, 2, dim=1)
        unbound = torch.unbind(x, dim=1)
        return (
            x.view(2, 3, 8),
            torch.reshape(x, (2, -1)),
            torch.flatten(x, 1),
            torch.transpose(x, 1, 2),
            torch.permute(x, (0, 2, 1)),
            torch.squeeze(x.unsqueeze(1), 1),
            x.transpose(1, 2).contiguous(),
            torch.clone(x),
            torch.cat((x, y), dim=-1),
            torch.stack((x, y), dim=0),
            split,
            chunk,
            unbound,
            x[:, :1, :].expand(-1, 3, -1),
            x.repeat(1, 2, 1),
            torch.tile(x, (1, 2, 1)),
        )


class _IndexScatterVariants(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        select_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        source = torch.ones_like(x)
        scatter_base = torch.zeros_like(x)
        advanced = x[(torch.arange(x.shape[0]), select_index)]
        index_put_base = torch.zeros_like(x)
        index_put = torch.index_put(
            index_put_base,
            (torch.arange(x.shape[0]), select_index),
            advanced,
        )
        return (
            x[:, 1:5],
            torch.select(x, 1, 2),
            torch.narrow(x, 1, 1, 3),
            torch.index_select(x, 1, select_index),
            torch.gather(x, 1, index),
            torch.scatter(scatter_base, 1, index, source),
            torch.scatter_reduce(scatter_base, 1, index, source, reduce="sum"),
            index_put,
            torch.masked_select(x, mask),
            torch.masked_fill(x, mask, 0.0),
            torch.topk(x, 3, dim=-1),
            torch.sort(x, dim=-1),
        )


class _EmbeddingSequenceVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.embedding_bag = nn.EmbeddingBag(64, 8, mode="mean")
        self.rnn = nn.RNN(8, 6, batch_first=True)
        self.gru = nn.GRU(8, 6, batch_first=True)
        self.lstm = nn.LSTM(8, 6, batch_first=True)

    def forward(
        self,
        tokens: torch.Tensor,
        flat_tokens: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[Any, ...]:
        embedded = self.embedding(tokens)
        return (
            embedded,
            self.embedding_bag(flat_tokens, offsets),
            self.rnn(embedded),
            self.gru(embedded),
            self.lstm(embedded),
        )


class _TrainingRandomVariants(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            F.dropout(x, p=0.25, training=True),
            torch.bernoulli(torch.sigmoid(x)),
            torch.rand_like(x),
        )


def _torchvision_resnet18() -> nn.Module:
    from torchvision.models import resnet18

    return resnet18(weights=None)


def _torchvision_mobilenet_v3() -> nn.Module:
    from torchvision.models import mobilenet_v3_small

    return mobilenet_v3_small(weights=None)


class _SmallBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import BertConfig, BertModel

        self.model = BertModel(
            BertConfig(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


class _SmallGraphConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.first = GCNConv(8, 16)
        self.second = GCNConv(16, 4)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.second(torch.relu(self.first(node_features, edge_index)), edge_index)


class _MelSpectrogram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=256,
            hop_length=128,
            n_mels=32,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.log1p(self.mel(waveform))


def smoke_cases() -> tuple[CoverageCase, ...]:
    """Return deterministic, dependency-free cases for PR and CI smoke runs."""

    return (
        CoverageCase(
            case_id="smoke_residual_conv",
            family="residual_cnn",
            modality="image",
            model_factory=_ResidualConv,
            input_factory=lambda: ((torch.randn(2, 3, 8, 6),), {}),
            loss_factory=lambda: torch.nn.MSELoss(),
            target_factory=lambda: torch.zeros(2, 3, 8, 6),
            optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=1e-3),
        ),
        CoverageCase(
            case_id="smoke_sequence_attention",
            family="transformer_block",
            modality="text",
            model_factory=_SequenceBlock,
            input_factory=lambda: ((torch.randint(0, 64, (2, 5)),), {}),
        ),
        CoverageCase(
            case_id="smoke_index_reduction",
            family="index_reduction",
            modality="tabular",
            model_factory=_IndexReduction,
            input_factory=lambda: (
                (
                    torch.randn(2, 6, 4),
                    torch.tensor(
                        [
                            [[0, 1, 2, 3]] * 3 + [[2, 1, 0, 3]] * 3,
                            [[1, 0, 3, 2]] * 6,
                        ],
                        dtype=torch.int64,
                    ),
                ),
                {},
            ),
        ),
    )


def p0_cases() -> tuple[CoverageCase, ...]:
    """Deterministic small-shape coverage cases spanning every P0 family."""

    return (
        *smoke_cases(),
        CoverageCase(
            "p0_dense_variants",
            "dense_matrix",
            "multimodal",
            _DenseVariants,
            lambda: (
                (
                    torch.randn(2, 3, 7),
                    torch.randn(4, 5),
                    torch.randn(5, 3),
                    torch.randn(2, 4, 5),
                    torch.randn(2, 5, 3),
                    torch.randn(5),
                    torch.randn(2, 4),
                    torch.randn(2, 3),
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_convolution_variants",
            "convolution",
            "vision_audio",
            _ConvolutionVariants,
            lambda: (
                (
                    torch.randn(2, 4, 9),
                    torch.randn(2, 4, 8, 10),
                    torch.randn(2, 4, 5, 6, 7),
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_attention_variants",
            "attention",
            "text",
            _AttentionVariants,
            lambda: (
                (
                    torch.randn(2, 5, 16),
                    torch.randn(2, 4, 5, 4),
                    torch.randn(2, 4, 5, 4),
                    torch.randn(2, 4, 5, 4),
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_normalization_variants",
            "normalization",
            "multimodal",
            _NormalizationVariants,
            lambda: ((torch.randn(2, 8, 6, 5), torch.randn(2, 5, 8)), {}),
        ),
        CoverageCase(
            "p0_elementwise_unary_variants",
            "elementwise_unary",
            "multimodal",
            _ElementwiseUnaryVariants,
            lambda: (
                (
                    torch.randn(2, 3, 4),
                    torch.randn(1, 3, 1),
                    torch.rand(2, 3, 4) > 0.5,
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_reduction_loss_variants",
            "reduction_loss",
            "multimodal",
            _ReductionLossVariants,
            lambda: (
                (
                    torch.rand(2, 3, 4) + 0.25,
                    torch.randn(4, 7),
                    torch.tensor([1, 3, 2, 5], dtype=torch.int64),
                    torch.rand(4, 7),
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_pool_resample_variants",
            "pool_resample",
            "vision_audio",
            _PoolResampleVariants,
            lambda: (
                (
                    torch.randn(2, 3, 12),
                    torch.randn(2, 3, 8, 10),
                    torch.randn(2, 3, 6, 8, 10),
                    torch.rand(2, 5, 6, 2) * 2 - 1,
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_layout_variants",
            "layout_shape",
            "multimodal",
            _LayoutVariants,
            lambda: ((torch.randn(2, 3, 8), torch.randn(2, 3, 8)), {}),
        ),
        CoverageCase(
            "p0_index_scatter_variants",
            "index_scatter",
            "multimodal",
            _IndexScatterVariants,
            lambda: (
                (
                    torch.randn(2, 6, 5),
                    torch.randint(0, 6, (2, 6, 5), dtype=torch.int64),
                    torch.tensor([1, 3], dtype=torch.int64),
                    torch.rand(2, 6, 5) > 0.6,
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_embedding_sequence_variants",
            "embedding_sequence",
            "text_timeseries",
            _EmbeddingSequenceVariants,
            lambda: (
                (
                    torch.randint(0, 64, (2, 5), dtype=torch.int64),
                    torch.randint(0, 64, (8,), dtype=torch.int64),
                    torch.tensor([0, 4], dtype=torch.int64),
                ),
                {},
            ),
        ),
        CoverageCase(
            "p0_training_random_variants",
            "random_regularization",
            "multimodal",
            _TrainingRandomVariants,
            lambda: ((torch.randn(2, 3, 4),), {}),
            training=True,
        ),
    )


def representative_source_cases() -> tuple[CoverageCase, ...]:
    """Small real-library models for the declared supported source corpus."""

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]],
        dtype=torch.int64,
    )
    return (
        CoverageCase(
            "source_torchvision_resnet18",
            "residual_cnn",
            "image",
            _torchvision_resnet18,
            lambda: ((torch.randn(1, 3, 64, 64),), {}),
            metadata={"source": "torchvision", "weights": "none"},
        ),
        CoverageCase(
            "source_torchvision_mobilenet_v3_small",
            "mobile_cnn",
            "image",
            _torchvision_mobilenet_v3,
            lambda: ((torch.randn(1, 3, 64, 64),), {}),
            metadata={"source": "torchvision", "weights": "none"},
        ),
        CoverageCase(
            "source_transformers_small_bert",
            "transformer_encoder",
            "text",
            _SmallBert,
            lambda: (
                (
                    torch.randint(0, 128, (2, 16), dtype=torch.int64),
                    torch.ones(2, 16, dtype=torch.int64),
                ),
                {},
            ),
            metadata={"source": "transformers", "weights": "random"},
        ),
        CoverageCase(
            "source_pyg_small_gcn",
            "graph_convolution",
            "graph",
            _SmallGraphConvolution,
            lambda: ((torch.randn(4, 8), edge_index.clone()), {}),
            metadata={"source": "torch_geometric", "weights": "random"},
        ),
    )


def frontier_source_cases() -> tuple[CoverageCase, ...]:
    """P1 diagnostic cases that do not define the strict supported boundary."""

    return (
        CoverageCase(
            "frontier_torchaudio_mel_spectrogram",
            "audio_spectral",
            "audio",
            _MelSpectrogram,
            lambda: ((torch.randn(2, 2048),), {}),
            metadata={
                "source": "torchaudio",
                "support": "diagnostic_frontier",
            },
        ),
    )


__all__ = [
    "CoverageCase",
    "frontier_source_cases",
    "p0_cases",
    "representative_source_cases",
    "smoke_cases",
]
