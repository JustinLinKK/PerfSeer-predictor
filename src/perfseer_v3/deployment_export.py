"""Verified CPU export for a versioned PerfSeer v3 student artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifact import LoadedArtifactV3, load_checkpoint_artifact, sha256_file
from .coarsen_v3 import COARSENING_POLICY_ID, COARSENING_POLICY_SHA256, coarsen_graph
from .features import apply_normalization, batch_graph_features, build_graph_features
from .graph_ir_v3 import GraphIRV3
from .model import graph_batch_tensors
from .op_registry import OperationRegistry
from .version import STUDENT_MODEL_RELEASE


def _deployment_graph(graph: GraphIRV3, registry: OperationRegistry) -> GraphIRV3:
    record = graph.metadata.get("coarsening")
    if record is None:
        return coarsen_graph(graph, registry=registry)
    if record.get("policy") != COARSENING_POLICY_ID:
        raise ValueError("graph uses an incompatible coarsening policy")
    if record.get("policy_sha256") != COARSENING_POLICY_SHA256:
        raise ValueError("graph coarsening hash mismatch")
    return graph


def export_torchscript_student(
    *,
    artifact_path: str | Path,
    graph_path: str | Path,
    output_path: str | Path,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Script, reload, and numerically verify a student on a real v3 graph."""

    registry = OperationRegistry.load()
    loaded: LoadedArtifactV3 = load_checkpoint_artifact(artifact_path, registry=registry)
    if loaded.metadata.model_release != STUDENT_MODEL_RELEASE:
        raise ValueError("CPU deployment export requires a v3 student artifact")
    graph = _deployment_graph(GraphIRV3.load(graph_path), registry)
    features = build_graph_features(graph, registry=registry)
    if loaded.normalization is not None:
        features = apply_normalization(features, loaded.normalization)
    batch = batch_graph_features([features])
    inputs = graph_batch_tensors(batch)
    model = loaded.model.cpu().eval()
    with torch.no_grad():
        eager = model(*inputs)
    scripted = torch.jit.script(model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.no_grad():
        exported = reloaded(*inputs)
    output_names = (
        "prediction",
        "log_variance",
        "oom_logit",
        "confidence",
        "oom_stage_logits",
        "peak_live_bytes_log1p",
        "graph_embedding",
        "phase_embedding",
    )
    comparisons = {}
    for name, expected, actual in zip(output_names, eager, exported):
        maximum_error = float((expected - actual).abs().max()) if expected.numel() else 0.0
        matched = bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
        comparisons[name] = {"allclose": matched, "maximum_absolute_error": maximum_error}
        if not matched:
            raise RuntimeError(f"TorchScript output {name} differs from eager execution")
    report = {
        "format": "perfseer_v3_torchscript_export_v1",
        "source_artifact": str(Path(artifact_path).resolve()),
        "source_artifact_sha256": loaded.sha256,
        "torchscript_path": str(output.resolve()),
        "torchscript_sha256": sha256_file(output),
        "torchscript_bytes": output.stat().st_size,
        "output_contract_version": loaded.metadata.output_contract_version,
        "feature_schema_sha256": loaded.metadata.feature_schema_sha256,
        "operator_registry_sha256": loaded.metadata.operator_registry_sha256,
        "normalization_sha256": loaded.metadata.normalization_sha256,
        "coarsening_policy_sha256": loaded.metadata.coarsening_policy_sha256,
        "verification_graph_sha256": graph.graph_sha256,
        "atol": atol,
        "rtol": rtol,
        "comparisons": comparisons,
        "verified": True,
        "pytorch_version": torch.__version__,
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "sidecar_path": str(sidecar.resolve())}


__all__ = ["export_torchscript_student"]
