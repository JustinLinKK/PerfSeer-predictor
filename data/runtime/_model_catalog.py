"""Curated family definitions and deterministic architecture sampling."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FamilySpec:
    """Finite, family-compatible architecture search space."""

    family_id: str
    family_index: int
    modality: str
    task_kind: str
    quota: int
    widths: tuple[int, ...]
    depths: tuple[int, ...]
    base_modules: tuple[str, ...]
    batch_pairs: tuple[tuple[int, int], ...]
    extra_axes: tuple[tuple[str, tuple[Any, ...]], ...]

    @property
    def grid_size(self) -> int:
        axis_sizes = [
            len(self.widths),
            len(self.depths),
            len(self.base_modules),
            len(self.batch_pairs),
            *(len(values) for _name, values in self.extra_axes),
        ]
        return math.prod(axis_sizes)


def _spec(
    family_index: int,
    family_id: str,
    modality: str,
    task_kind: str,
    quota: int,
    widths: tuple[int, ...],
    depths: tuple[int, ...],
    base_modules: tuple[str, ...],
    batch_pairs: tuple[tuple[int, int], ...],
    **extra_axes: tuple[Any, ...],
) -> FamilySpec:
    return FamilySpec(
        family_id=family_id,
        family_index=family_index,
        modality=modality,
        task_kind=task_kind,
        quota=quota,
        widths=widths,
        depths=depths,
        base_modules=base_modules,
        batch_pairs=batch_pairs,
        extra_axes=tuple(extra_axes.items()),
    )


_VISION_BATCHES = ((1, 8), (4, 16), (8, 32))
_DENSE_VISION_BATCHES = ((1, 4), (2, 8), (4, 16))
_NLP_BATCHES = ((1, 8), (4, 16), (8, 32))
_AUDIO_BATCHES = ((2, 8), (4, 16), (8, 32))
_TABULAR_BATCHES = ((32, 128), (64, 256), (128, 512))
_GRAPH_BATCHES = ((2, 8), (4, 16), (8, 32))


_SPECS = (
    _spec(1, "resnet50", "vision", "classification", 260, (32, 48, 64, 80), (8, 12, 16, 24), ("baseline", "convnext_v1", "convnext_v2"), _VISION_BATCHES, resolution=(128, 160, 224), kernel_size=(3, 5), activation=("relu", "silu")),
    _spec(2, "efficientnet_b0_b4", "vision", "classification", 240, (24, 32, 40, 48), (8, 12, 16, 20), ("mbconv", "fused_mbconv", "convnext_v2"), _VISION_BATCHES, resolution=(128, 192, 224), expand_ratio=(2, 4, 6), se_ratio=(0.125, 0.25)),
    _spec(3, "mobilenet_v3_large", "vision", "classification", 180, (24, 32, 40, 48), (6, 10, 14, 18), ("mobilenet_v3", "convnext_v1", "convnext_v2"), _VISION_BATCHES, resolution=(128, 192, 224), kernel_size=(3, 5)),
    _spec(4, "inception_v3", "vision", "classification", 160, (24, 32, 48, 64), (4, 6, 8, 10), ("inception", "inception_residual"), _VISION_BATCHES, resolution=(128, 192, 224), branch_count=(3, 4), asymmetric_kernel=(5, 7)),
    _spec(5, "vit_s16", "vision", "classification", 240, (192, 256, 384, 512), (4, 6, 8, 12), ("mha", "gqa", "mla", "linear_attention"), _VISION_BATCHES, resolution=(128, 192, 224), patch_size=(8, 16), heads=(4, 8)),
    _spec(6, "swin_t", "vision", "classification", 200, (96, 128, 192, 256), (4, 6, 8, 12), ("window_mha", "window_gqa", "window_mla", "window_linear"), _VISION_BATCHES, resolution=(128, 192, 224), window_size=(4, 8), heads=(4, 8)),
    _spec(7, "mlp_mixer_s", "vision", "classification", 180, (128, 192, 256, 384), (4, 6, 8, 12), ("mixer", "gated_mixer", "swiglu_mixer"), _VISION_BATCHES, resolution=(128, 192, 224), patch_size=(8, 16), token_ratio=(0.25, 0.5)),
    _spec(8, "stn_cnn", "vision", "classification", 120, (24, 32, 48, 64), (3, 4, 6), ("stn_cnn", "stn_residual"), _VISION_BATCHES, resolution=(64, 96, 128), localization_depth=(2, 3)),
    _spec(9, "mish_resnet", "vision", "classification", 160, (32, 48, 64, 80), (8, 12, 16, 24), ("mish_residual", "mish_convnext_v1", "mish_convnext_v2"), _VISION_BATCHES, resolution=(128, 192, 224), kernel_size=(3, 5)),
    _spec(10, "prelu_elu_cnn", "vision", "classification", 160, (24, 32, 48, 64), (4, 6, 8, 12), ("prelu", "elu", "prelu_elu"), _VISION_BATCHES, resolution=(96, 128, 192), kernel_size=(3, 5)),
    _spec(11, "unet", "vision", "dense_prediction", 240, (16, 24, 32, 48), (3, 4, 5, 6), ("convtranspose", "upsample_conv", "pixelshuffle", "convnext_v1", "convnext_v2"), _DENSE_VISION_BATCHES, resolution=(64, 96, 128), norm=("group", "batch")),
    _spec(12, "pix2pix", "vision", "generation", 160, (16, 24, 32, 48), (3, 4, 5, 6), ("convtranspose", "upsample_conv", "pixelshuffle", "convnext_v2"), _DENSE_VISION_BATCHES, resolution=(64, 96, 128), discriminator_depth=(2, 3, 4)),
    _spec(13, "restormer", "vision", "dense_prediction", 200, (32, 48, 64, 96), (4, 6, 8, 12), ("transposed_attention", "gqa", "mla", "linear_attention"), _DENSE_VISION_BATCHES, resolution=(64, 96, 128), heads=(4, 8), expansion=(2, 3)),
    _spec(14, "bert_base_finetune", "nlp", "classification", 220, (128, 192, 256, 384), (2, 4, 6, 8), ("mha", "gqa", "mla", "linear_attention"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), vocab_size=(4096, 8192)),
    _spec(15, "gpt2_small", "nlp", "causal_classification", 220, (128, 192, 256, 384), (2, 4, 6, 8), ("mha", "gqa", "mla", "linear_attention"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), vocab_size=(4096, 8192)),
    _spec(16, "t5_small", "nlp", "seq2seq", 180, (128, 192, 256, 384), (2, 4, 6, 8), ("mha", "gqa", "mla", "linear_attention"), _NLP_BATCHES, sequence_length=(32, 64, 128), heads=(4, 8), vocab_size=(4096, 8192)),
    _spec(17, "llama_small", "nlp", "causal_classification", 220, (128, 192, 256, 384), (2, 4, 6, 8), ("mha", "gqa", "mla", "linear_attention"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), vocab_size=(4096, 8192)),
    _spec(18, "mla_mini_transformer", "nlp", "classification", 180, (128, 192, 256, 384), (2, 4, 6, 8), ("mla", "gqa", "linear_attention"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), latent_ratio=(0.25, 0.5)),
    _spec(19, "kimi_linear", "nlp", "classification", 160, (128, 192, 256, 384), (2, 4, 6, 8), ("delta_linear", "linear_attention", "gqa"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), state_ratio=(0.25, 0.5)),
    _spec(20, "switch_moe_classifier", "nlp", "classification", 180, (128, 192, 256, 384), (2, 4, 6, 8), ("mha_moe", "gqa_moe", "mla_moe", "linear_moe"), _NLP_BATCHES, sequence_length=(64, 128, 256), expert_count=(4, 8), top_k=(1, 2)),
    _spec(21, "bilstm_crf", "nlp", "token_labeling", 160, (96, 128, 192, 256), (1, 2, 3, 4), ("bilstm_crf", "bilstm_softmax"), _NLP_BATCHES, sequence_length=(64, 128, 256), vocab_size=(4096, 8192)),
    _spec(22, "rnn_seq2seq", "nlp", "seq2seq", 160, (96, 128, 192, 256), (1, 2, 3, 4), ("gru", "rnn", "lstm"), _NLP_BATCHES, sequence_length=(32, 64, 128), vocab_size=(4096, 8192), bidirectional=(False, True)),
    _spec(23, "fasttext_embeddingbag", "nlp", "classification", 120, (64, 96, 128, 192), (1, 2, 3), ("embeddingbag_mean", "embeddingbag_sum"), _NLP_BATCHES, sequence_length=(32, 64, 128), vocab_size=(4096, 8192), ngram_order=(1, 2, 3)),
    _spec(24, "distilbert_distillation", "nlp", "distillation", 200, (128, 192, 256, 384), (2, 3, 4, 6), ("mha", "gqa", "mla", "linear_attention"), _NLP_BATCHES, sequence_length=(64, 128, 256), heads=(4, 8), temperature=(1.0, 2.0, 4.0)),
    _spec(25, "panns_cnn14", "audio", "classification", 180, (32, 48, 64, 96), (6, 10, 14, 18), ("cnn14", "residual_cnn"), _AUDIO_BATCHES, mel_bins=(64, 96, 128), frames=(128, 256, 512), pooling=("avg", "max_avg")),
    _spec(26, "temporal_convolutional_network", "audio", "classification", 160, (32, 48, 64, 96), (4, 6, 8, 12), ("tcn", "depthwise_tcn"), _AUDIO_BATCHES, waveform_length=(2048, 4096, 8192), kernel_size=(3, 5, 7), dilation_cycle=(3, 4)),
    _spec(27, "m5_waveform_cnn", "audio", "classification", 160, (32, 48, 64, 96), (4, 6, 8, 10), ("m5", "m5_se"), _AUDIO_BATCHES, waveform_length=(2048, 4096, 8192), kernel_size=(40, 80), stride=(8, 16)),
    _spec(28, "selu_mlp", "tabular", "classification", 220, (64, 128, 256, 384), (2, 4, 6, 8), ("selu_mlp", "residual_selu"), _TABULAR_BATCHES, feature_count=(32, 64, 128), dropout=(0.0, 0.1, 0.2)),
    _spec(29, "tabtransformer", "tabular", "classification", 200, (64, 96, 128, 192), (2, 4, 6, 8), ("mha", "gqa", "mla", "linear_attention"), _TABULAR_BATCHES, feature_count=(16, 32, 64), categorical_features=(4, 8, 16), heads=(4, 8)),
    _spec(30, "mixture_density_network", "tabular", "regression", 180, (64, 128, 256, 384), (2, 4, 6, 8), ("gaussian_mixture", "laplace_mixture"), _TABULAR_BATCHES, feature_count=(16, 32, 64), component_count=(3, 5, 8), activation=("softplus", "elu")),
    _spec(31, "cgcnn", "graph", "regression", 200, (64, 96, 128, 192), (2, 4, 6, 8), ("cgcnn", "gated_cgcnn"), _GRAPH_BATCHES, node_count=(16, 32, 64), feature_count=(32, 64), neighbor_count=(4, 8, 12)),
    _spec(32, "gcn_gat_graphsage", "graph", "regression", 200, (64, 96, 128, 192), (2, 4, 6, 8), ("gcn", "gat", "graphsage"), _GRAPH_BATCHES, node_count=(16, 32, 64), feature_count=(32, 64), heads=(4, 8)),
)

FAMILY_SPECS: dict[str, FamilySpec] = {spec.family_id: spec for spec in _SPECS}
SOURCE_COUNT = sum(spec.quota for spec in _SPECS)
PROFILE_ROW_COUNT = SOURCE_COUNT * 2 * 2 * 2


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(":\u001f".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def sample_family_configs(family_id: str, count: int, seed: int = 42) -> list[dict[str, Any]]:
    """Return deterministic unique configurations with stratified base-module coverage."""

    if family_id not in FAMILY_SPECS:
        raise KeyError(f"unknown family {family_id!r}")
    spec = FAMILY_SPECS[family_id]
    if count < 0 or count > spec.grid_size:
        raise ValueError(f"requested {count} configurations from grid of {spec.grid_size} for {family_id}")
    if count == 0:
        return []

    common_axes: list[tuple[Any, ...]] = [
        spec.widths,
        spec.depths,
        spec.batch_pairs,
        *(values for _name, values in spec.extra_axes),
    ]
    queues: list[list[tuple[Any, ...]]] = []
    for base_module in spec.base_modules:
        combinations = list(itertools.product(*common_axes))
        random.Random(_stable_seed(seed, family_id, base_module)).shuffle(combinations)
        queues.append([(base_module, *combination) for combination in combinations])

    rows: list[dict[str, Any]] = []
    round_index = 0
    while len(rows) < count:
        made_progress = False
        for base_index, queue in enumerate(queues):
            if round_index >= len(queue) or len(rows) >= count:
                continue
            made_progress = True
            base_module, width, depth, batch_pair, *extra_values = queue[round_index]
            row: dict[str, Any] = {
                "family_id": spec.family_id,
                "family_index": spec.family_index,
                "modality": spec.modality,
                "task_kind": spec.task_kind,
                "width": int(width),
                "depth": int(depth),
                "base_module": str(base_module),
                "batch_sizes": [int(batch_pair[0]), int(batch_pair[1])],
            }
            for (axis_name, _axis_values), axis_value in zip(spec.extra_axes, extra_values, strict=True):
                row[axis_name] = _json_value(axis_value)
            row["sample_rank"] = round_index * len(queues) + base_index
            rows.append(row)
        if not made_progress:
            raise RuntimeError(f"exhausted configuration grid for {family_id} at {len(rows)} rows")
        round_index += 1
    return rows
