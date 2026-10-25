"""Static native dataflow through the established v3 typed feature contract."""

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from perfseer_v3.features import feature_layout, fit_normalization, apply_normalization
from perfseer_v3.graph_ir_v3 import GraphIRV3, OperationNodeV3, TensorEdgeV3, GraphGlobalFeatures, CoverageQuality, Estimate
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.schema import DTYPES
from perfseer_v31.io import atomic_write, file_sha256, fingerprint
from perfseer_v31.features_core import feature_layout as v31_feature_layout
from perfseer_v32.features import _build_features, EXTRA_FIELDS
from perfseer_v32.version import FEATURE_VERSION, NORMALIZATION_VERSION

from .native_data import LITERALS, TARGET_NAMES, static_description, native_rows, verify as verify_native
from .native_graph import build_graph, VERSION as GRAPH_VERSION

VERSION = "perfseer_v4_seerblock_static_dataset_v2"
LEGACY_VERSION = "perfseer_v4_seerblock_static_dataset_v1"
WIDTHS = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
          "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
ALIASES = {"Linear": "linear", "Conv1d": "conv1d", "Conv2d": "conv2d",
           "ConvTranspose2d": "conv_transpose2d.input", "BatchNorm1d": "batch_norm",
           "BatchNorm2d": "batch_norm", "LayerNorm": "layer_norm", "GroupNorm": "group_norm",
           "Embedding": "embedding", "EmbeddingBag": "embedding_bag.padding_idx",
           "AvgPool2d": "avg_pool2d", "AdaptiveAvgPool2d": "adaptive_avg_pool2d",
           "ReLU": "relu", "GELU": "gelu", "SiLU": "silu", "ELU": "elu",
           "SELU": "selu", "Mish": "mish", "Hardswish": "hardswish", "PReLU": "prelu",
           "Dropout": "dropout", "Add": "add.Tensor", "Sub": "sub.Tensor",
           "Mult": "mul.Tensor", "Div": "div.Tensor"}


def seerblock_feature_layout(registry=None):
    """The original v3.2 training-branch layout, including its ten extra fields."""
    layout = v31_feature_layout(registry or OperationRegistry.load())
    return replace(layout, feature_schema_version=FEATURE_VERSION,
                   feature_schema_sha256=fingerprint([layout.feature_schema_sha256, EXTRA_FIELDS,
                                                      "scalar_argument_signature_v1", "shared_parameter_relations_v1", FEATURE_VERSION]),
                   global_continuous_fields=layout.global_continuous_fields + EXTRA_FIELDS)


def graph_features(portable, training, source, registry=None):
    """Encode symbolic relations, marking unobserved execution details unknown."""
    registry = registry or OperationRegistry.load()
    layout = feature_layout(registry)
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for edge in portable["edges"]:
        incoming[edge["target"]].append(edge)
        outgoing[edge["source"]].append(edge)
    edges, nodes, depths = [], [], {}
    for i, edge in enumerate(portable["edges"]):
        parent = portable["nodes"][edge["source"]]
        shape = tuple(edge["shape"])
        if parent["op"] == "parameter" and not parent["attributes"].get("shape_known", True):
            shape = ("grouped_parameters",)
            numel = parent["parameters"]
        else:
            numel = math.prod(shape)
        width = WIDTHS[edge["dtype"]]
        role = "parameter" if edge["role"] == "parameter" else "model_input" if parent["op"] == "input" else "activation"
        parameter_identity = "static_parameter_" + str(edge["source"]) if role == "parameter" else None
        edges.append(TensorEdgeV3(str(i), str(edge["source"]), str(edge["target"]),
                                 edge["source_slot"], edge["target_slot"], role, shape, len(shape),
                                 edge["dtype"], width, numel, numel * width,
                                 source_name=parameter_identity or edge["role"],
                                 dynamic_shape_quality="estimated",
                                 is_materialized=False))
    for item in portable["nodes"]:
        index, op, attrs = item["id"], item["op"], item["attributes"]
        short = op.removeprefix("nn.")
        raw = "aten::" + ALIASES.get(short, short)
        resolved = registry.resolve(raw)
        if not resolved.is_known:
            raw = "perfseer_static::" + op
            resolved = registry.resolve(raw)
        input_edges = [e for e in incoming[index] if e["role"] != "loop_carried"]
        inputs = [portable["nodes"][e["source"]] for e in input_edges]
        in_numel = sum((n["parameters"] if n["op"] == "parameter" else math.prod(n["shape"])) * e["multiplicity"]
                       for n, e in zip(inputs, input_edges, strict=True))
        in_bytes = sum((n["parameters"] if n["op"] == "parameter" else math.prod(n["shape"])) * WIDTHS[n["dtype"]] * e["multiplicity"]
                       for n, e in zip(inputs, input_edges, strict=True))
        out_numel = item["parameters"] if op == "parameter" else math.prod(item["shape"])
        out_bytes = out_numel * WIDTHS[item["dtype"]]
        depths[index] = max((depths.get(e["source"], -1) + 1 for e in incoming[index] if e["role"] != "loop_carried"), default=0)
        macs = None
        args, kwargs = attrs.get("args", []), attrs.get("kwargs", {})
        if op == "nn.Linear":
            macs = out_numel * (args[0] if args else kwargs["in_features"])
        elif op in {"nn.Conv1d", "nn.Conv2d"}:
            kernel = args[2] if len(args) > 2 else kwargs["kernel_size"]
            volume = math.prod(kernel) if isinstance(kernel, (list, tuple)) else kernel ** (1 if op == "nn.Conv1d" else 2)
            channels = args[0] if args else kwargs["in_channels"]
            groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
            macs = out_numel * channels // groups * volume
        elif op == "matmul" and inputs:
            macs = out_numel * inputs[0]["shape"][-1]
        count = item["repeat"]
        estimate = Estimate() if macs is None else Estimate(float(macs * count), "shape_formula", 1.)
        nodes.append(OperationNodeV3(
            node_id=str(index), raw_target=raw, canonical_op_id=resolved.canonical_id,
            family_id=resolved.family_id, family=resolved.family, phase="forward",
            exact_op_id=resolved.exact_id, op_hash_bucket=resolved.hash_bucket,
            source_module_path=attrs.get("module"), flags={"estimated": True},
            normalized_args={**{key: value for key, value in attrs.items() if key != "module"}, "repeat": count,
                             "incoming_multiplicities": [e["multiplicity"] for e in incoming[index]],
                             "loop_carried_inputs": sum(e["role"] == "loop_carried" for e in incoming[index])},
            input_tensor_count=sum(e["multiplicity"] for e in input_edges), output_tensor_count=1,
            input_numel=in_numel, output_numel=out_numel,
            input_bytes=in_bytes, output_bytes=out_bytes, parameter_numel=item["parameters"],
            parameter_bytes=item["parameters"] * WIDTHS[item["dtype"]],
            flops=Estimate(2 * estimate.value, estimate.method, estimate.confidence), macs=estimate,
            bytes_read=Estimate(float(in_bytes * count), "shape_formula", .5),
            bytes_written=Estimate(float(out_bytes * count), "shape_formula", .5),
            topological_index=index, depth=depths[index], fan_in=len(incoming[index]), fan_out=len(outgoing[index])))
    n = len(nodes)
    global_features = GraphGlobalFeatures(
        operation_nodes=n, tensor_edges=len(edges), total_flops=sum(x.flops.value for x in nodes),
        total_macs=sum(x.macs.value for x in nodes),
        total_parameter_numel=portable["metadata"].get("allocated_parameter_count", sum(x.parameter_numel for x in nodes)),
        total_parameter_bytes=4 * portable["metadata"].get("allocated_parameter_count", sum(x.parameter_numel for x in nodes)),
        total_activation_bytes=sum(x.tensor_bytes or 0 for x in edges if x.tensor_role == "activation"),
        critical_path_length=max(depths.values(), default=0),
        unknown_operation_fraction=sum(x.canonical_op_id == "UNK" for x in nodes) / max(1, n),
        unknown_cost_fraction=sum(x.flops.method == "unknown" for x in nodes) / max(1, n),
        unknown_byte_fraction=1.)
    optimizer = {"name": training["optimizer"], "gradient_accumulation_steps": training["grad_accumulation_steps"]}
    workload_training = {"steps_per_epoch": training["steps_per_epoch"], "microbatch_size": training["batch_size"],
                         "gradient_accumulation_steps": training["grad_accumulation_steps"], "precision": training["precision"],
                         "optimizer": optimizer, "backend": "unknown", "scheduler": {"name": "unknown"},
                         "unavailable_settings": ["per_epoch_learning_rate_trace"]}
    unavailable = ["backend", "scheduler", "scheduler_progress", "optimizer_foreach", "optimizer_fused",
                   "optimizer_hyperparameters", "per_epoch_learning_rate_trace"]
    graph = GraphIRV3.create(
        operator_registry_sha256=registry.sha256, feature_schema_sha256=layout.feature_schema_sha256,
        capture_backend="static_ast", capture_mode="estimated", pytorch_version="not_executed",
        source_fingerprint=fingerprint(portable), model_fingerprint=fingerprint(portable), input_signature={},
        dynamic_constraints={}, training_mode=True, precision=training["precision"],
        optimizer_config=optimizer, training_config=workload_training, nodes=tuple(nodes), tensor_edges=tuple(edges),
        global_features=global_features,
        coverage=CoverageQuality(capture_quality="estimated", backward_capture_quality="failed",
                                 tensor_nodes_seen=n, tensor_nodes_encoded=n,
                                 unknown_operations=sum(x.canonical_op_id == "UNK" for x in nodes)),
        metadata={**portable["metadata"], "hardware_id": "nvidia_" + source.lower(), "batch_size": training["batch_size"]},
        warnings=("Static forward graph only; backward, optimizer kernels, runtime lifetimes and AMP operator dtypes unavailable.",))
    # Use the archived training feature semantics without its strict paired
    # capture loader: these inputs remain explicitly estimated and forward-only.
    feature = _build_features(graph)
    # This is the requested precision policy; per-operation dtypes are only nominal source types.
    precision = "float16" if training["precision"] == "fp16_amp" else "float32"
    return replace(feature, precision_id=torch.tensor([DTYPES.index(precision)]),
                   metadata={**feature.metadata, **portable["metadata"], "declared_precision": training["precision"],
                             "runtime_dtypes_available": False, "unavailable_training_settings": unavailable,
                             "optimizer_auto_flags_mean": "unspecified_in_input_not_verified_profiler_policy"})


def prepare(repo, native, output, source):
    repo, native, output = Path(repo).resolve(), Path(native).resolve(), Path(output).resolve()
    if (output / "dataset.pt").exists():
        raise ValueError("use a fresh output directory; retain previous static datasets for audit")
    manifest = verify_native(native / source, repo=repo)
    if manifest.get("split_policy") != "known-families-90-5-5":
        raise ValueError("expected frozen known-family configuration splits")
    existing = np.load(native / source / "dataset.npz", allow_pickle=False)
    configs = {r["profile_point_id"]: r for r in (json.loads(line) for line in (repo / "labels/A10/model_config.jsonl").open())}
    rows, _, _ = native_rows(repo, source, configs, {f"{i:02}": 0 for i in range(1, 33)})
    row_by_id = {r["sample_id"]: r for r in rows}
    if not np.array_equal(existing["targets"], np.asarray([row_by_id[x]["targets"] for x in existing["sample_ids"]], dtype=np.float32)):
        raise ValueError("labels differ from frozen native dataset")
    registry = OperationRegistry.load()
    templates, graphs, feature_cache, features = {}, OrderedDict(), {}, []
    for index, identifier in enumerate(existing["sample_ids"]):
        config = configs[identifier]
        path = (repo / "labels/A10" / config["model_file"]).resolve()
        if not path.is_relative_to(repo / "labels/A10") or file_sha256(path) != config["source_sha256"]:
            raise ValueError("native source integrity failure")
        text = path.read_text()
        description = static_description(text, config, templates)
        template = re.sub(r"^(?:" + "|".join(LITERALS) + r")\s*=.*$", "", text, flags=re.MULTILINE)
        template = re.sub(r'\A\s*"""[^\n]*"""\s*\n', "", template)
        key = fingerprint({"model": description["model"], "inputs": description["inputs"],
                           "template": hashlib.sha256(template.encode()).hexdigest()})
        if key not in graphs:
            graphs[key] = build_graph(text, description["model"], description["inputs"])
            if len(graphs) > 64:
                graphs.popitem(last=False)
        training = row_by_id[identifier]["training"]
        feature_key = fingerprint({"graph": key, "training": training})
        if feature_key not in feature_cache:
            feature_cache[feature_key] = graph_features(graphs[key], training, source, registry)
        features.append(feature_cache[feature_key])
        if index % 1000 == 0:
            print(json.dumps({"event": "static_graph_features", "source": source, "completed": index, "total": len(existing["sample_ids"])}), flush=True)
    del graphs
    split_sha = fingerprint({"ids": existing["sample_ids"].tolist(), "splits": existing["splits"].tolist()})
    print(json.dumps({"event": "fit_train_normalization", "source": source, "unique_features": len(feature_cache)}), flush=True)
    normalization = fit_normalization([f for f, s in zip(features, existing["splits"], strict=True) if s == 0],
                                      split_name="train", split_fingerprint=split_sha, quantiles=(0., 1.))
    normalization = replace(normalization, normalization_version=NORMALIZATION_VERSION)
    normalized = {id(f): apply_normalization(f, normalization) for f in feature_cache.values()}
    features = [normalized[id(f)] for f in features]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"version": VERSION, "graph_version": GRAPH_VERSION, "source": source,
                "native_fingerprint": manifest["fingerprint"], "target_names": list(TARGET_NAMES),
                "splits": manifest["splits"], "split_fingerprint": split_sha,
                "normalization_sha256": normalization.sha256, "workloads_executed": False,
                "feature_version": FEATURE_VERSION, "layout_sha256": seerblock_feature_layout(registry).layout_sha256,
                "static_forward_only": True, "unique_features": len(normalized),
                "preparation_code": {p: file_sha256(Path(__file__).with_name(p)) for p in
                                     ("seerblock_data.py", "native_graph.py", "native_static.py")}}
    metadata["fingerprint"] = fingerprint(metadata)
    data = {"features": features, "targets": torch.from_numpy(existing["targets"].copy()),
            "splits": torch.from_numpy(existing["splits"].astype(np.int64)),
            "sample_ids": existing["sample_ids"].tolist(), "configuration_groups": existing["configuration_groups"].tolist(),
            "families": existing["groups"].tolist(), "manifest": metadata, "normalization": normalization}
    verify(data)
    atomic_write(output / "dataset.pt", data, checkpoint=True)
    atomic_write(output / "normalization.json", asdict(normalization))
    atomic_write(output / "dataset_manifest.json", {**metadata, "dataset_sha256": file_sha256(output / "dataset.pt")})
    print(json.dumps({"event": "static_graph_dataset_verified", **metadata}), flush=True)


def verify(data):
    n = len(data["features"])
    if data["targets"].shape != (n, 6) or data["splits"].shape != (n,) or len(set(data["sample_ids"])) != n:
        raise ValueError("invalid graph dataset dimensions or identities")
    if not torch.isfinite(data["targets"]).all() or data["manifest"]["target_names"] != list(TARGET_NAMES):
        raise ValueError("invalid graph dataset targets")
    version = data["manifest"]["version"]
    if version not in {VERSION, LEGACY_VERSION}:
        raise ValueError("unsupported static graph dataset version")
    expected = seerblock_feature_layout() if version == VERSION else feature_layout(OperationRegistry.load())
    if data["normalization"].layout_sha256 != expected.layout_sha256:
        raise ValueError("normalization layout differs from static dataset version")
    if version == VERSION and data["manifest"].get("layout_sha256") != expected.layout_sha256:
        raise ValueError("manifest layout differs from static dataset version")
    if version == VERSION and (data["normalization"].normalization_version != NORMALIZATION_VERSION
                               or tuple(data["normalization"].quantiles) != (0., 1.)):
        raise ValueError("new static datasets require the original full-training-moments normalization")
    seen, groups = set(), {}
    for feature, split, group in zip(data["features"], data["splits"].tolist(), data["configuration_groups"], strict=True):
        if split not in (0, 1, 2) or groups.setdefault(group, split) != split:
            raise ValueError("configuration group crosses split")
        if id(feature) not in seen:
            feature.validate(); seen.add(id(feature))
            if feature.layout.layout_sha256 != expected.layout_sha256:
                raise ValueError("feature layout differs from static dataset version")
        if feature.metadata.get("normalization_sha256") != data["normalization"].sha256:
            raise ValueError("feature normalization differs")
    for split in range(3):
        if len({f for f, s in zip(data["families"], data["splits"].tolist(), strict=True) if s == split}) != 32:
            raise ValueError("each split must cover all 32 families")
    return {"status": "verified", "rows": n, "unique_features": len(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--native", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=("A100", "A10"), required=True)
    args = parser.parse_args()
    prepare(args.repo, args.native, args.output, args.source)


if __name__ == "__main__":
    main()
