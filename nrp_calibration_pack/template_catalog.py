"""Deterministic expanded template catalog for PerfSeer calibration packs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from perfseer.architecture_schema import (
    ARCHITECTURE_FAMILIES,
    ARCHITECTURE_FAMILY_QUOTAS,
    FEATURE_SCHEMA_VERSION,
    VARIANT_KINDS,
    variant_counts_for_family,
)


ARG_DEFAULTS: dict[str, int] = {
    "conv_kernel_size": 0,
    "conv_stride": 0,
    "conv_padding": 0,
    "conv_dilation": 0,
    "conv_groups": 0,
    "conv_bias": 0,
    "linear_in_features": 0,
    "linear_out_features": 0,
    "linear_bias": 0,
    "pool_kernel_size": 0,
    "pool_stride": 0,
    "pool_padding": 0,
    "pool_ceil_mode": 0,
}


FAMILY_MODALITY: dict[str, str] = {
    "resnet_cnn": "image",
    "efficientnet_cnn": "image",
    "bert_encoder": "text",
    "vit_encoder": "image",
    "yolo_detector": "image",
    "unet_encoder_decoder": "image",
    "ast_audio_transformer": "audio",
    "gru_temporal": "temporal",
    "lstm_temporal": "temporal",
    "vgg_cnn": "image",
    "gat_graph": "graph",
    "mpnn_graph": "graph",
    "t5_encoder_decoder": "text",
    "wav2vec2_audio": "audio",
    "ft_transformer_tabular": "tabular",
}

TE_LOW_PRECISION_TRANSFORMER_FAMILIES: tuple[str, ...] = (
    "vit_encoder",
    "ast_audio_transformer",
    "wav2vec2_audio",
    "ft_transformer_tabular",
)

TEMPORAL_SEQUENCE_BUCKETS: tuple[int, ...] = (64, 96, 128, 160, 192)


@dataclass(frozen=True)
class TemplateSpec:
    model_index: int
    family: str
    variant_kind: str
    local_index: int
    family_index: int
    seed: int

    @property
    def model_stem(self) -> str:
        return f"template_{self.model_index:05d}_{self.family}_{self.variant_kind}_{self.local_index:04d}"

    @property
    def variant_signature(self) -> str:
        return f"{self.family}:{self.variant_kind}:{self.local_index}:seed{self.seed}"


def template_family_counts(subset_size: int, families: Iterable[str] | None = None) -> dict[str, int]:
    if families is not None:
        family_list = [family for family in families if family in ARCHITECTURE_FAMILIES]
        if not family_list:
            raise ValueError("template family filter selected no known families")
        base = subset_size // len(family_list)
        rem = subset_size % len(family_list)
        return {family: base + (1 if idx < rem else 0) for idx, family in enumerate(family_list)}
    full_total = sum(ARCHITECTURE_FAMILY_QUOTAS.values())
    if subset_size == full_total:
        return dict(ARCHITECTURE_FAMILY_QUOTAS)
    families = list(ARCHITECTURE_FAMILIES)
    base = subset_size // len(families)
    rem = subset_size % len(families)
    return {family: base + (1 if idx < rem else 0) for idx, family in enumerate(families)}


def iter_template_specs(subset_size: int, seed: int, families: Iterable[str] | None = None) -> Iterable[TemplateSpec]:
    model_index = 0
    for family, count in template_family_counts(subset_size, families=families).items():
        family_index = list(ARCHITECTURE_FAMILIES).index(family)
        variants = variant_plan(count)
        for local_index, variant_kind in enumerate(variants):
            yield TemplateSpec(model_index, family, variant_kind, local_index, family_index, seed)
            model_index += 1


def variant_plan(count: int) -> list[str]:
    if count <= len(VARIANT_KINDS):
        return [VARIANT_KINDS[idx % len(VARIANT_KINDS)] for idx in range(count)]
    counts = variant_counts_for_family(count)
    out: list[str] = []
    for kind in VARIANT_KINDS:
        out.extend([kind] * counts.get(kind, 0))
    return out[:count]


def build_template_graph(spec: TemplateSpec) -> nx.DiGraph:
    builders = {
        "resnet_cnn": _resnet_graph,
        "efficientnet_cnn": _efficientnet_graph,
        "bert_encoder": _bert_graph,
        "vit_encoder": _vit_graph,
        "yolo_detector": _yolo_graph,
        "unet_encoder_decoder": _unet_graph,
        "ast_audio_transformer": _ast_graph,
        "gru_temporal": _gru_graph,
        "lstm_temporal": _lstm_graph,
        "vgg_cnn": _vgg_graph,
        "gat_graph": _gat_graph,
        "mpnn_graph": _mpnn_graph,
        "t5_encoder_decoder": _t5_graph,
        "wav2vec2_audio": _wav2vec2_graph,
        "ft_transformer_tabular": _ft_transformer_graph,
    }
    graph = builders[spec.family](spec)
    graph.graph.update(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "architecture_family": spec.family,
            "modality": FAMILY_MODALITY[spec.family],
            "variant_kind": spec.variant_kind,
            "variant_signature": spec.variant_signature,
            "depth_bucket": _depth_bucket(spec),
            "width_bucket": _width_bucket(spec),
        }
    )
    return graph


def _depth_multiplier(spec: TemplateSpec) -> int:
    if spec.variant_kind == "added_depth":
        return 2 + (spec.local_index % 3)
    if spec.variant_kind == "dropped_depth":
        return 1
    if spec.variant_kind == "mixed_stress":
        return 3
    return 2


def _width(spec: TemplateSpec, base: int = 16) -> int:
    scale = 1 + ((spec.local_index + spec.family_index) % 3)
    if spec.variant_kind == "width_shape":
        scale += 2
    if spec.variant_kind == "mixed_stress":
        scale += 1
    return base * scale


def _align_up(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _te_transformer_width(spec: TemplateSpec) -> int:
    return _align_up(_width(spec), 32)


def _te_transformer_seq(spec: TemplateSpec, batch: int) -> int:
    base = max(1, 32 // max(batch, 1))
    stride = max(1, 32 // max(batch, 1))
    return base + stride * (spec.local_index % 2)


def _batch(spec: TemplateSpec) -> int:
    return (1, 2, 4, 8)[(spec.local_index + spec.family_index) % 4]


def _depth_bucket(spec: TemplateSpec) -> int:
    return min(5, _depth_multiplier(spec) + (1 if spec.variant_kind == "mixed_stress" else 0))


def _width_bucket(spec: TemplateSpec) -> int:
    return min(5, max(1, _width(spec) // 16))


def _new_graph(spec: TemplateSpec, input_specs: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph["input_specs"] = input_specs
    return graph


def _add_node(
    graph: nx.DiGraph,
    op_type: str,
    mem: dict[str, int | float],
    *,
    args: dict[str, Any] | None = None,
    preds: list[int] | None = None,
    input_index: int = 0,
) -> int:
    node_id = graph.number_of_nodes()
    merged_args = dict(ARG_DEFAULTS)
    if args:
        merged_args.update(args)
    flops = _flops(op_type, mem, merged_args)
    bytes_touched = max(float(mem.get("bytes", 0.0)), 1.0)
    graph.add_node(
        node_id,
        input_index=input_index,
        feature={
            "type": op_type,
            "args": merged_args,
            "memory_info": mem,
            "flops": flops,
            "arith_intensity": flops / bytes_touched,
        },
    )
    for pred in preds or []:
        graph.add_edge(pred, node_id)
    return node_id


def _image_input(spec: TemplateSpec, channels: int = 3, size: int = 16) -> list[dict[str, Any]]:
    return [{"name": "image", "shape": [_batch(spec), channels, size, size], "dtype": "float32", "kind": "image"}]


def _token_input(spec: TemplateSpec, seq: int) -> list[dict[str, Any]]:
    return [{"name": "tokens", "shape": [_batch(spec), seq], "dtype": "int64", "kind": "tokens"}]


def _seq_input(spec: TemplateSpec, seq: int, dim: int, kind: str = "sequence") -> list[dict[str, Any]]:
    return [{"name": kind, "shape": [_batch(spec), seq, dim], "dtype": "float32", "kind": kind}]


def _graph_inputs(spec: TemplateSpec, nodes: int, dim: int) -> list[dict[str, Any]]:
    batch = _batch(spec)
    return [
        {"name": "node_features", "shape": [batch, nodes, dim], "dtype": "float32", "kind": "graph_features"},
        {"name": "adjacency", "shape": [batch, nodes, nodes], "dtype": "float32", "kind": "adjacency"},
    ]


def _mem_image(batch: int, in_c: int, out_c: int, h: int, w: int, *, out_h: int | None = None, out_w: int | None = None) -> dict[str, int]:
    out_h = h if out_h is None else out_h
    out_w = w if out_w is None else out_w
    input_size = batch * in_c * h * w * 4
    output_size = batch * out_c * out_h * out_w * 4
    return {
        "bytes": input_size + output_size,
        "weight_size": in_c * out_c,
        "batch_size": batch,
        "rank": 4,
        "input_size_with_weight": input_size + in_c * out_c,
        "input_size": input_size,
        "input_channels": in_c,
        "input_w": w,
        "input_h": h,
        "input_features": in_c,
        "output_size": output_size,
        "output_channels": out_c,
        "output_w": out_w,
        "output_h": out_h,
        "output_features": out_c,
        "spatial_area": out_h * out_w,
    }


def _mem_seq(batch: int, seq: int, in_dim: int, out_dim: int, *, tokens: bool = False) -> dict[str, int]:
    input_size = batch * seq * (8 if tokens else in_dim * 4)
    output_size = batch * seq * out_dim * 4
    return {
        "bytes": input_size + output_size,
        "weight_size": in_dim * out_dim,
        "batch_size": batch,
        "rank": 3,
        "sequence_length": seq,
        "input_size_with_weight": input_size + in_dim * out_dim,
        "input_size": input_size,
        "input_channels": in_dim,
        "input_features": in_dim,
        "input_w": 0,
        "input_h": seq,
        "output_size": output_size,
        "output_channels": out_dim,
        "output_features": out_dim,
        "output_w": 0,
        "output_h": seq,
    }


def _mem_graph(batch: int, nodes: int, in_dim: int, out_dim: int) -> dict[str, int]:
    mem = _mem_seq(batch, nodes, in_dim, out_dim)
    mem["graph_nodes"] = nodes
    return mem


def _linear_args(in_features: int, out_features: int) -> dict[str, int]:
    return {"linear_in_features": in_features, "linear_out_features": out_features, "linear_bias": 1}


def _conv_args(kernel: int = 3, groups: int = 1, stride: int = 1, padding: int = 1) -> dict[str, int]:
    return {
        "conv_kernel_size": kernel,
        "conv_stride": stride,
        "conv_padding": padding,
        "conv_dilation": 1,
        "conv_groups": groups,
        "conv_bias": 1,
    }


def _pool_args(kernel: int = 2) -> dict[str, int]:
    return {"pool_kernel_size": kernel, "pool_stride": kernel, "pool_padding": 0, "pool_ceil_mode": 0}


def _flops(op_type: str, mem: dict[str, int | float], args: dict[str, Any]) -> int:
    output_size = max(int(float(mem.get("output_size", 1))) // 4, 1)
    if op_type in {"Conv", "DepthwiseConv", "ConvTranspose", "Gemm", "MatMul", "Bmm"}:
        return int(2 * output_size * max(int(float(mem.get("input_features", mem.get("input_channels", 1)))), 1))
    if op_type in {"Attention", "MultiHeadAttention", "GraphAttention"}:
        seq = max(int(float(mem.get("sequence_length", mem.get("graph_nodes", 1)))), 1)
        dim = max(int(float(mem.get("output_features", 1))), 1)
        batch = max(int(float(mem.get("batch_size", 1))), 1)
        return int(4 * batch * seq * seq * dim + 8 * batch * seq * dim * dim)
    if op_type in {"RNN", "GRU", "LSTM"}:
        gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[op_type]
        return int(gates * 2 * output_size * max(int(float(mem.get("input_features", 1))), 1))
    return output_size


def _resnet_graph(spec: TemplateSpec) -> nx.DiGraph:
    width = _width(spec)
    graph = _new_graph(spec, _image_input(spec))
    batch = _batch(spec)
    h = w = 16
    stem = _add_node(graph, "Conv", _mem_image(batch, 3, width, h, w), args=_conv_args())
    relu = _add_node(graph, "Relu", _mem_image(batch, width, width, h, w), preds=[stem])
    prev = relu
    for _ in range(_depth_multiplier(spec)):
        conv = _add_node(graph, "Conv", _mem_image(batch, width, width, h, w), args=_conv_args(), preds=[prev])
        add = _add_node(graph, "Add", _mem_image(batch, width, width, h, w), preds=[prev, conv])
        prev = _add_node(graph, "Relu", _mem_image(batch, width, width, h, w), preds=[add])
    pool = _add_node(graph, "GlobalAveragePool", _mem_image(batch, width, width, h, w, out_h=1, out_w=1), preds=[prev])
    flat = _add_node(graph, "Flatten", _mem_image(batch, width, width, 1, 1), preds=[pool])
    _add_node(graph, "Gemm", _mem_seq(batch, 1, width, 10), args=_linear_args(width, 10), preds=[flat])
    return graph


def _efficientnet_graph(spec: TemplateSpec) -> nx.DiGraph:
    width = _width(spec)
    graph = _new_graph(spec, _image_input(spec))
    batch = _batch(spec)
    h = w = 16
    prev = _add_node(graph, "Conv", _mem_image(batch, 3, width, h, w), args=_conv_args(kernel=1, padding=0))
    for _ in range(_depth_multiplier(spec)):
        depth = _add_node(graph, "DepthwiseConv", _mem_image(batch, width, width, h, w), args=_conv_args(groups=width), preds=[prev])
        act = _add_node(graph, "Silu", _mem_image(batch, width, width, h, w), preds=[depth])
        prev = _add_node(graph, "Conv", _mem_image(batch, width, width, h, w), args=_conv_args(kernel=1, padding=0), preds=[act])
    pool = _add_node(graph, "GlobalAveragePool", _mem_image(batch, width, width, h, w, out_h=1, out_w=1), preds=[prev])
    flat = _add_node(graph, "Flatten", _mem_image(batch, width, width, 1, 1), preds=[pool])
    _add_node(graph, "Gemm", _mem_seq(batch, 1, width, 10), args=_linear_args(width, 10), preds=[flat])
    return graph


def _vgg_graph(spec: TemplateSpec) -> nx.DiGraph:
    width = _width(spec)
    graph = _new_graph(spec, _image_input(spec))
    batch = _batch(spec)
    h = w = 16
    prev = _add_node(graph, "Conv", _mem_image(batch, 3, width, h, w), args=_conv_args())
    for _ in range(_depth_multiplier(spec)):
        prev = _add_node(graph, "Relu", _mem_image(batch, width, width, h, w), preds=[prev])
        prev = _add_node(graph, "Conv", _mem_image(batch, width, width, h, w), args=_conv_args(), preds=[prev])
    pool = _add_node(graph, "GlobalAveragePool", _mem_image(batch, width, width, h, w, out_h=1, out_w=1), preds=[prev])
    flat = _add_node(graph, "Flatten", _mem_image(batch, width, width, 1, 1), preds=[pool])
    _add_node(graph, "Gemm", _mem_seq(batch, 1, width, 10), args=_linear_args(width, 10), preds=[flat])
    return graph


def _yolo_graph(spec: TemplateSpec) -> nx.DiGraph:
    width = _width(spec)
    graph = _new_graph(spec, _image_input(spec))
    batch = _batch(spec)
    h = w = 16
    conv = _add_node(graph, "Conv", _mem_image(batch, 3, width, h, w), args=_conv_args())
    depth = _add_node(graph, "DepthwiseConv", _mem_image(batch, width, width, h, w), args=_conv_args(groups=width), preds=[conv])
    act = _add_node(graph, "Silu", _mem_image(batch, width, width, h, w), preds=[depth])
    _add_node(graph, "DetectorHead", _mem_image(batch, width, 14, h, w, out_h=1, out_w=1), preds=[act])
    return graph


def _unet_graph(spec: TemplateSpec) -> nx.DiGraph:
    width = _width(spec)
    graph = _new_graph(spec, _image_input(spec))
    batch = _batch(spec)
    h = w = 16
    skip = _add_node(graph, "Conv", _mem_image(batch, 3, width, h, w), args=_conv_args())
    pool = _add_node(graph, "MaxPool", _mem_image(batch, width, width, h, w, out_h=h // 2, out_w=w // 2), args=_pool_args(), preds=[skip])
    bottleneck = _add_node(graph, "Conv", _mem_image(batch, width, width, h // 2, w // 2), args=_conv_args(), preds=[pool])
    up = _add_node(graph, "Upsample", _mem_image(batch, width, width, h // 2, w // 2, out_h=h, out_w=w), args={"scale_factor": 2}, preds=[bottleneck])
    cat = _add_node(graph, "Concat", _mem_image(batch, width * 2, width * 2, h, w), args={"concat_dim": 1}, preds=[skip, up])
    _add_node(graph, "SegmentationHead", _mem_image(batch, width * 2, 2, h, w), preds=[cat])
    return graph


def _transformer_graph(spec: TemplateSpec, family: str, *, token_input: bool, kind: str) -> nx.DiGraph:
    batch = _batch(spec)
    width = _te_transformer_width(spec)
    seq = _te_transformer_seq(spec, batch)
    graph = _new_graph(spec, _token_input(spec, seq) if token_input else _seq_input(spec, seq, width, kind=kind))
    if token_input:
        prev = _add_node(graph, "Embedding", _mem_seq(batch, seq, width, width, tokens=True), args={"vocab_size": 1024}, input_index=0)
    else:
        prev = _add_node(graph, "TabularFeature" if family == "ft_transformer_tabular" else "Gemm", _mem_seq(batch, seq, width, width), args=_linear_args(width, width), input_index=0)
    for _ in range(_depth_multiplier(spec)):
        attn = _add_node(graph, "Attention", _mem_seq(batch, seq, width, width), preds=[prev])
        add = _add_node(graph, "Add", _mem_seq(batch, seq, width, width), preds=[prev, attn])
        norm = _add_node(graph, "LayerNormalization", _mem_seq(batch, seq, width, width), preds=[add])
        gelu = _add_node(graph, "Gelu", _mem_seq(batch, seq, width, width), preds=[norm])
        prev = _add_node(graph, "Gemm", _mem_seq(batch, seq, width, width), args=_linear_args(width, width), preds=[gelu])
    if family in {"bert_encoder", "t5_encoder_decoder"}:
        prev = _add_node(graph, "Softmax", _mem_seq(batch, seq, width, width), args={"softmax_dim": -1}, preds=[prev])
    _add_node(graph, "Gemm", _mem_seq(batch, seq, width, width), args=_linear_args(width, width), preds=[prev])
    return graph


def _bert_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "bert_encoder", token_input=True, kind="tokens")


def _vit_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "vit_encoder", token_input=False, kind="patch_tokens")


def _ast_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "ast_audio_transformer", token_input=False, kind="spectrogram_tokens")


def _t5_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "t5_encoder_decoder", token_input=True, kind="tokens")


def _wav2vec2_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "wav2vec2_audio", token_input=False, kind="audio_sequence")


def _ft_transformer_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _transformer_graph(spec, "ft_transformer_tabular", token_input=False, kind="tabular_tokens")


def _rnn_graph(spec: TemplateSpec, op_type: str) -> nx.DiGraph:
    width = _width(spec)
    seq = TEMPORAL_SEQUENCE_BUCKETS[spec.local_index % len(TEMPORAL_SEQUENCE_BUCKETS)]
    graph = _new_graph(spec, _seq_input(spec, seq, width, kind="temporal"))
    batch = _batch(spec)
    prev = _add_node(graph, op_type, _mem_seq(batch, seq, width, width), input_index=0)
    norm = _add_node(graph, "LayerNormalization", _mem_seq(batch, seq, width, width), preds=[prev])
    _add_node(graph, "Gemm", _mem_seq(batch, seq, width, max(4, width // 2)), args=_linear_args(width, max(4, width // 2)), preds=[norm])
    return graph


def _gru_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _rnn_graph(spec, "GRU")


def _lstm_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _rnn_graph(spec, "LSTM")


def _graph_model(spec: TemplateSpec, attention: bool) -> nx.DiGraph:
    width = _width(spec)
    nodes = 6 + (spec.local_index % 4)
    graph = _new_graph(spec, _graph_inputs(spec, nodes, width))
    batch = _batch(spec)
    op = "GraphAttention" if attention else "GraphMessage"
    prev = _add_node(graph, op, _mem_graph(batch, nodes, width, width), args={"adjacency_input_index": 1}, input_index=0)
    prev = _add_node(graph, "Relu", _mem_graph(batch, nodes, width, width), preds=[prev])
    prev = _add_node(graph, "GraphMessage", _mem_graph(batch, nodes, width, width), args={"adjacency_input_index": 1}, preds=[prev])
    _add_node(graph, "Gemm", _mem_graph(batch, nodes, width, max(4, width // 2)), args=_linear_args(width, max(4, width // 2)), preds=[prev])
    return graph


def _gat_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _graph_model(spec, attention=True)


def _mpnn_graph(spec: TemplateSpec) -> nx.DiGraph:
    return _graph_model(spec, attention=False)
