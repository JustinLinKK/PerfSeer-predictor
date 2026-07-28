"""Finite PyTorch-CUDA smoke verifier used locally and by the Nautilus Job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import capture_export
from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.features import batch_graph_features, build_graph_features
from perfseer_v3.model import SeerNetV3, SeerNetV3Config, graph_batch_tensors
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.profiling import ProfileOptions, ProfileWorkload, profile_workload


class CudaVerifierModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm(self.conv(x)))
        sequence = features.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(sequence).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
        ).squeeze(1)
        return self.proj(attended.mean(dim=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args_cli = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("PerfSeer v3 CUDA verifier requires an allocated NVIDIA GPU")
    torch.manual_seed(7)
    device = torch.device("cuda")
    source_model = CudaVerifierModel().to(device).eval()
    args = (torch.randn(2, 3, 8, 8, device=device),)
    capture = capture_export(source_model, args)
    if not capture.success or capture.graph is None:
        raise RuntimeError(f"strict CUDA capture failed: {capture.failures}")
    if capture.graph.coverage.capture_quality != "strict":
        raise RuntimeError("CUDA capture unexpectedly used non-strict fallback")
    if capture.graph.coverage.tensor_nodes_seen != capture.graph.coverage.tensor_nodes_encoded:
        raise RuntimeError("CUDA capture silently lost tensor-producing operations")

    registry = OperationRegistry.load()
    features = build_graph_features(capture.graph, registry=registry)
    batch = batch_graph_features([features])
    model_config = SeerNetV3Config.from_registry(
        registry,
        batch.layout,
        hidden=32,
        num_blocks=2,
        exact_embedding_dim=16,
        family_embedding_dim=8,
        hash_embedding_dim=8,
        phase_embedding_dim=4,
        dtype_embedding_dim=4,
        dropout=0.0,
    )
    predictor = SeerNetV3(model_config).to(device).train()
    cuda_tensors = tuple(tensor.to(device) for tensor in graph_batch_tensors(batch))
    output = predictor(*cuda_tensors)
    objective = (
        output.prediction.square().mean()
        + output.log_variance.square().mean()
        + output.oom_logit.square().mean()
        + output.confidence.square().mean()
        + output.oom_stage_logits.square().mean()
        + output.peak_live_bytes_log1p.square().mean()
        + output.graph_embedding.square().mean()
        + output.phase_embedding.square().mean()
    )
    objective.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in predictor.parameters()
    ):
        raise RuntimeError("v3 predictor produced a nonfinite CUDA gradient")
    scripted = torch.jit.script(predictor.eval())
    with torch.inference_mode():
        scripted_output = scripted(*cuda_tensors)
    for expected, actual in zip(predictor(*cuda_tensors), scripted_output):
        torch.testing.assert_close(actual, expected)

    source_model.train()
    target = torch.randn(2, 6, device=device)
    training_capture = capture_training_graph(
        source_model,
        args,
        target=target,
        loss_fn=F.mse_loss,
        optimizer_name="adamw",
    )
    if not training_capture.success or training_capture.graph is None:
        raise RuntimeError(f"training graph capture failed: {training_capture.failures}")
    phases = {node.phase for node in training_capture.graph.nodes}
    if phases != {"forward", "loss", "backward", "optimizer"}:
        raise RuntimeError(f"training graph phases are incomplete: {sorted(phases)}")

    source_model.eval()
    profile = profile_workload(
        ProfileWorkload(source_model, args, {}, capture),
        options=ProfileOptions(warmup_steps=1, measured_steps=2),
    )
    if profile.status != "ok" or profile.measured_steps_completed != 2:
        raise RuntimeError(f"CUDA source-first profile failed: {profile}")
    result = {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "strict_capture_nodes": capture.graph.coverage.tensor_nodes_encoded,
        "strict_capture_edges": len(capture.graph.tensor_edges),
        "feature_schema_sha256": features.layout.feature_schema_sha256,
        "operator_registry_sha256": registry.sha256,
        "training_backend": training_capture.backward_backend,
        "training_phases": sorted(phases),
        "prediction_shape": list(output.prediction.shape),
        "scripted_prediction_shape": list(scripted_output.prediction.shape),
        "oom_stage_shape": list(output.oom_stage_logits.shape),
        "phase_embedding_shape": list(output.phase_embedding.shape),
        "output_contract_version": "perfseer_v3_outputs_v2",
        "profile_step_ms": list(profile.measured_step_ms),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    output_path = args_cli.output
    if output_path is None and Path("/workspace").exists():
        output_path = Path("/workspace/perfseer_v3_cuda_verifier.json")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
