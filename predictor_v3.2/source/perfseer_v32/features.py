"""Versioned workload features; source and team metadata never enter tensors."""

from dataclasses import dataclass, replace

import torch
from perfseer_v3.op_registry import OperationRegistry

from perfseer_v31.features_core import build_graph_features as build_v3_features

from perfseer_v31.io import fingerprint
from perfseer_v3.features import apply_normalization

from .capture import graphs_from_design
from .version import FEATURE_VERSION, MODES


EXTRA_FIELDS = ("execution_cuda_eager", "execution_torch_compile", "execution_unknown",
                "scheduler_setting_unknown", "epoch_length_unknown", "optimizer_foreach_auto",
                "optimizer_fused_auto", "scheduler_progress", "learning_rate_trace_unavailable", "precision_tf32")


def _build_features(graph):
    features = build_v3_features(graph)
    def scalar_arguments(value):
        if isinstance(value, dict):
            return "tensor" if "tensor_ref" in value else {key: scalar_arguments(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scalar_arguments(item) for item in value]
        return value
    buckets = OperationRegistry.load().hash_buckets
    signatures = [int(fingerprint([node.raw_target, scalar_arguments(node.normalized_args)])[:16], 16) % buckets for node in graph.nodes]
    training = graph.training_config
    backend = str(training.get("backend", "unknown"))
    eager = backend in {"cuda_eager", "eager"}
    compiled = backend in {"torch_compile", "torch_compile_inductor", "inductor", "inductor_cuda"}
    values = [eager, compiled, not (eager or compiled),
              training.get("scheduler", {}).get("name", "unknown") == "unknown",
              training.get("steps_per_epoch") is None,
              training["optimizer"].get("foreach") is None,
              training["optimizer"].get("fused") is None,
              training["scheduler"].get("progress", 0.),
              "per_epoch_learning_rate_trace" in training.get("unavailable_settings", []),
              training["precision"] in {"tf32", "fp32_tf32"}]
    layout = replace(features.layout, feature_schema_version=FEATURE_VERSION,
                     feature_schema_sha256=fingerprint([features.layout.feature_schema_sha256, EXTRA_FIELDS,
                                                        "scalar_argument_signature_v1", "shared_parameter_relations_v1", FEATURE_VERSION]),
                     global_continuous_fields=features.layout.global_continuous_fields + EXTRA_FIELDS)
    result = replace(features, layout=layout, u_cont=torch.cat((features.u_cont, torch.tensor([values], dtype=torch.float32)), dim=1),
                     op_overload_hash_id=torch.tensor(signatures, dtype=torch.long))
    result.validate()
    return result


@dataclass(frozen=True)
class PairedFeatures:
    training: object
    inference: object

    @property
    def layout(self):
        return self.training.layout

    def validate(self):
        for mode in MODES:
            features = getattr(self, mode)
            features.validate()
            if features.layout.feature_schema_version != FEATURE_VERSION or features.layout != self.layout:
                raise ValueError("paired feature layout differs")


def build_features(design):
    graphs = graphs_from_design(design)
    training, inference = graphs["training"], graphs["inference"]
    if not training.training_mode or any(training.training_config.get(name) != inference.training_config.get(name)
                                         for name in ("microbatch_size", "precision", "backend")):
        raise ValueError("paired execution mode, batch, precision, or backend differs")
    if training.metadata.get("target_hardware_id") != inference.metadata.get("target_hardware_id"):
        raise ValueError("paired graph hardware differs")
    result = PairedFeatures(**{mode: _build_features(graphs[mode]) for mode in MODES})
    result.validate()
    return result


def normalize_features(features, normalization):
    return PairedFeatures(**{mode: apply_normalization(getattr(features, mode), getattr(normalization, mode)) for mode in MODES})
