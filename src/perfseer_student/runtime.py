"""Strict CPU-only runtime for a compatible 53/3/40 TorchScript artifact."""

from __future__ import annotations

from pathlib import Path

import torch

from .encoder import EncodedGraph
from .features import EDGE_DIM, GLOBAL_DIM, NODE_DIM, TARGET_NAMES


class PredictionError(RuntimeError):
    """Raised when encoded inputs or model outputs violate the student contract."""


class StudentRuntime:
    def __init__(self, artifact_path: str | Path, train_mem_index: int = 1) -> None:
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        self.train_mem_index = int(train_mem_index)
        try:
            self.model = torch.jit.load(str(self.artifact_path), map_location="cpu").eval()
        except Exception as exc:
            raise PredictionError(f"failed to load TorchScript artifact {self.artifact_path}: {exc}") from exc
        for parameter in self.model.parameters():
            if parameter.device.type != "cpu":
                raise PredictionError("predictor artifact contains a non-CPU parameter")
        for buffer in self.model.buffers():
            if buffer.device.type != "cpu":
                raise PredictionError("predictor artifact contains a non-CPU buffer")

    def predict(self, encoded: EncodedGraph) -> torch.Tensor:
        tensors = encoded.as_tuple()
        if tensors[0].ndim != 2 or tensors[0].shape[1] != NODE_DIM:
            raise PredictionError(f"x must have shape [nodes, {NODE_DIM}]")
        if tensors[1].ndim != 2 or tensors[1].shape[0] != 2:
            raise PredictionError("edge_index must have shape [2, edges]")
        if tensors[2].ndim != 2 or tensors[2].shape[1] != EDGE_DIM:
            raise PredictionError(f"edge_attr must have shape [edges, {EDGE_DIM}]")
        if tensors[3].shape != (1, GLOBAL_DIM):
            raise PredictionError(f"u must have shape [1, {GLOBAL_DIM}]")
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise PredictionError("all predictor inputs must be CPU tensors")
        with torch.inference_mode():
            output = self.model(*tensors)
        if output.shape != (1, len(TARGET_NAMES)):
            raise PredictionError(
                f"student output must have shape [1, {len(TARGET_NAMES)}], got {tuple(output.shape)}"
            )
        if output.device.type != "cpu" or not torch.isfinite(output).all():
            raise PredictionError("student output must be finite and CPU-resident")
        return output[0]

    def predict_train_mem_mb(self, encoded: EncodedGraph) -> float:
        value = float(self.predict(encoded)[self.train_mem_index].item())
        if value <= 0:
            raise PredictionError(f"student train_mem prediction must be positive, got {value}")
        return value
