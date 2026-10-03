"""Versioned workload features; source and team metadata never enter tensors."""

from dataclasses import replace

import torch
from perfseer_v3.op_registry import OperationRegistry

from .features_core import build_graph_features as build_v3_features

from .capture_training import graph_from_design
from .io import fingerprint


EXTRA_FIELDS = ("execution_cuda_eager", "execution_torch_compile", "execution_unknown",
                "scheduler_setting_unknown", "epoch_length_unknown", "optimizer_foreach_auto",
                "optimizer_fused_auto", "scheduler_progress", "learning_rate_trace_unavailable", "precision_tf32")


def build_features(design):
    graph = graph_from_design(design)
    features = build_v3_features(graph)
    def scalar_arguments(value):
        if isinstance(value, dict):
            return "tensor" if "tensor_ref" in value else {key: scalar_arguments(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scalar_arguments(item) for item in value]
        return value
    buckets = OperationRegistry.load().hash_buckets
    signatures = [int(fingerprint([node.raw_target, scalar_arguments(node.normalized_args)])[:16], 16) % buckets for node in graph.nodes]
    training = design["TRAINING_CONFIG"]
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
    layout = replace(features.layout, feature_schema_version="perfseer_v31_features_v1",
                     feature_schema_sha256=fingerprint([features.layout.feature_schema_sha256, EXTRA_FIELDS,
                                                        "scalar_argument_signature_v1", "shared_parameter_relations_v1"]),
                     global_continuous_fields=features.layout.global_continuous_fields + EXTRA_FIELDS)
    result = replace(features, layout=layout, u_cont=torch.cat((features.u_cont, torch.tensor([values], dtype=torch.float32)), dim=1),
                     op_overload_hash_id=torch.tensor(signatures, dtype=torch.long))
    result.validate()
    return result
