"""Deployment/runtime helpers for CPU PerfSeer evaluation."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


class _TensorData:
    pass


class TensorModelWrapper(nn.Module):
    """Tensor-only wrapper around a PyG-style model for tracing/export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        data = _TensorData()
        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.u = u
        data.batch = batch
        data.num_graphs = int(u.shape[0])
        return self.model(data)


@dataclass
class EvalProfile:
    name: str = "cpu_pytorch_fp32"
    track: str = "cpu_deploy"
    runtime_backend: str = "pytorch"
    device: str = "cpu"
    batch_size: int = 1
    bench_cpu: bool = True
    num_bench_graphs: int = 1000
    warmup: int = 20
    cpu_threads: int = 0
    cpu_interop_threads: int = 0
    quantize: bool = False
    export: bool = False
    artifact_dir: str | None = None
    onnx_opset: int = 17
    onnx_int8: bool = False
    openvino_int8: bool = False

    @classmethod
    def from_file(cls, path: str | None) -> "EvalProfile":
        if not path:
            return cls()
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class PreparedRuntime:
    backend: str
    models: list[Any]
    artifact_paths: list[str] = field(default_factory=list)
    statuses: list[dict[str, Any]] = field(default_factory=list)
    fallback: bool = False

    @property
    def artifact_size_mb(self) -> float:
        total = 0
        for path in self.artifact_paths:
            if path and os.path.exists(path):
                total += os.path.getsize(path)
        return total / (1024.0 * 1024.0)

    def predict_one_model(self, model_idx: int, batch) -> np.ndarray:
        model = self.models[model_idx]
        if self.backend in {"pytorch", "pytorch_dynamic_int8"}:
            with torch.no_grad():
                return model(batch.to(torch.device("cpu"))).detach().cpu().numpy()
        if self.backend == "torchscript":
            inputs = batch_to_tensors(batch, torch.device("cpu"))
            with torch.no_grad():
                return model(*inputs).detach().cpu().numpy()
        if self.backend.startswith("onnxruntime"):
            inputs = batch_to_numpy_inputs(batch)
            return model.run(None, inputs)[0]
        if self.backend.startswith("openvino"):
            inputs = batch_to_numpy_inputs(batch)
            result = model(inputs)
            first_key = next(iter(result))
            return np.asarray(result[first_key])
        raise ValueError(f"unknown backend {self.backend!r}")


def batch_to_tensors(batch, device: torch.device) -> tuple[torch.Tensor, ...]:
    batch = batch.to(device)
    node_batch = getattr(batch, "batch", None)
    if node_batch is None:
        node_batch = batch.x.new_zeros(batch.x.size(0), dtype=torch.long)
    return batch.x, batch.edge_index, batch.edge_attr, batch.u, node_batch


def batch_to_numpy_inputs(batch) -> dict[str, np.ndarray]:
    x, edge_index, edge_attr, u, node_batch = batch_to_tensors(batch, torch.device("cpu"))
    return {
        "x": x.detach().cpu().numpy(),
        "edge_index": edge_index.detach().cpu().numpy(),
        "edge_attr": edge_attr.detach().cpu().numpy(),
        "u": u.detach().cpu().numpy(),
        "batch": node_batch.detach().cpu().numpy(),
    }


def _save_model_artifact(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def _dynamic_quantize(model: nn.Module) -> nn.Module:
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized.eval()


def _trace_model(model: nn.Module, example_batch, path: str) -> torch.jit.ScriptModule:
    wrapper = TensorModelWrapper(copy.deepcopy(model).cpu().eval())
    wrapper.eval()
    inputs = batch_to_tensors(example_batch, torch.device("cpu"))
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            traced = torch.jit.trace(wrapper, inputs, strict=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    traced.save(path)
    return torch.jit.load(path).eval()


def _export_onnx(model: nn.Module, example_batch, path: str, opset: int) -> str:
    wrapper = TensorModelWrapper(copy.deepcopy(model).cpu().eval())
    wrapper.eval()
    inputs = batch_to_tensors(example_batch, torch.device("cpu"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            inputs,
            path,
            input_names=["x", "edge_index", "edge_attr", "u", "batch"],
            output_names=["pred"],
            opset_version=opset,
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_attr": {0: "num_edges"},
                "batch": {0: "num_nodes"},
                "pred": {0: "num_graphs"},
            },
            dynamo=False,
        )
    return path


def _quantize_onnx_dynamic(fp32_path: str, int8_path: str) -> str:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ort.set_default_logger_severity(3)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


def _load_onnx_session(path: str):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def _load_openvino_model(path: str, artifact_xml: str | None = None):
    import openvino as ov

    core = ov.Core()
    model = ov.convert_model(path)
    if artifact_xml:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_xml)), exist_ok=True)
        ov.save_model(model, artifact_xml)
    return core.compile_model(model, "CPU")


def prepare_runtime(
    models: list[nn.Module],
    profile: EvalProfile,
    example_batch,
    artifact_dir: str,
) -> PreparedRuntime:
    backend = profile.runtime_backend
    os.makedirs(artifact_dir, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    prepared: list[Any] = []
    artifact_paths: list[str] = []

    for idx, model in enumerate(models):
        try:
            if backend == "pytorch":
                runtime_model = copy.deepcopy(model).cpu().eval()
                artifact = os.path.join(artifact_dir, f"model{idx}_pytorch_fp32.pt")
                _save_model_artifact(runtime_model, artifact)
                prepared.append(runtime_model)
                artifact_paths.append(artifact)
            elif backend == "pytorch_dynamic_int8":
                runtime_model = _dynamic_quantize(model)
                artifact = os.path.join(artifact_dir, f"model{idx}_pytorch_dynamic_int8.pt")
                _save_model_artifact(runtime_model, artifact)
                prepared.append(runtime_model)
                artifact_paths.append(artifact)
            elif backend == "torchscript":
                artifact = os.path.join(artifact_dir, f"model{idx}_torchscript.pt")
                prepared.append(_trace_model(model, example_batch, artifact))
                artifact_paths.append(artifact)
            elif backend in {"onnxruntime", "onnxruntime_int8"}:
                fp32 = os.path.join(artifact_dir, f"model{idx}.onnx")
                _export_onnx(model, example_batch, fp32, profile.onnx_opset)
                artifact = fp32
                if backend == "onnxruntime_int8" or profile.onnx_int8:
                    artifact = os.path.join(artifact_dir, f"model{idx}.int8.onnx")
                    _quantize_onnx_dynamic(fp32, artifact)
                prepared.append(_load_onnx_session(artifact))
                artifact_paths.append(artifact)
            elif backend in {"openvino", "openvino_int8"}:
                fp32 = os.path.join(artifact_dir, f"model{idx}.onnx")
                _export_onnx(model, example_batch, fp32, profile.onnx_opset)
                source = fp32
                if backend == "openvino_int8" or profile.openvino_int8:
                    source = os.path.join(artifact_dir, f"model{idx}.int8.onnx")
                    _quantize_onnx_dynamic(fp32, source)
                artifact = os.path.join(artifact_dir, f"model{idx}_openvino.xml")
                prepared.append(_load_openvino_model(source, artifact))
                artifact_paths.append(artifact)
                bin_path = os.path.splitext(artifact)[0] + ".bin"
                if os.path.exists(bin_path):
                    artifact_paths.append(bin_path)
            else:
                raise ValueError(f"unknown runtime_backend {backend!r}")
            statuses.append({"model_idx": idx, "backend": backend, "status": "ok"})
        except Exception as exc:
            statuses.append(
                {
                    "model_idx": idx,
                    "backend": backend,
                    "status": "fallback_pytorch",
                    "error": repr(exc),
                }
            )
            fallback_models: list[Any] = []
            fallback_paths: list[str] = []
            for j, fallback_source in enumerate(models):
                fallback_model = copy.deepcopy(fallback_source).cpu().eval()
                artifact = os.path.join(artifact_dir, f"model{j}_fallback_pytorch_fp32.pt")
                _save_model_artifact(fallback_model, artifact)
                fallback_models.append(fallback_model)
                fallback_paths.append(artifact)
            return PreparedRuntime(
                backend="pytorch",
                models=fallback_models,
                artifact_paths=fallback_paths,
                statuses=statuses,
                fallback=True,
            )

    fallback = any(s["status"] != "ok" for s in statuses)
    return PreparedRuntime(backend=backend, models=prepared, artifact_paths=artifact_paths, statuses=statuses, fallback=fallback)


def benchmark_runtime(runtime: PreparedRuntime, loader, num_graphs: int, warmup: int) -> dict[str, float]:
    latencies: list[float] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            for _ in range(warmup if seen == 0 else 0):
                for idx in range(len(runtime.models)):
                    _ = runtime.predict_one_model(idx, batch)
            t0 = time.perf_counter()
            for idx in range(len(runtime.models)):
                _ = runtime.predict_one_model(idx, batch)
            elapsed = (time.perf_counter() - t0) * 1000.0
            n = int(getattr(batch, "num_graphs", 1))
            latencies.extend([elapsed / max(n, 1)] * n)
            seen += n
            if seen >= num_graphs:
                break
    if not latencies:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan"), "graphs_per_sec": float("nan")}
    arr = np.asarray(latencies[:num_graphs], dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "graphs_per_sec": float(1000.0 / mean_ms) if mean_ms > 0 else float("nan"),
    }


def current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return float("nan")
