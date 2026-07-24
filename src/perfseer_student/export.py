"""Export compatible student checkpoints as self-contained CPU TorchScript models."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from .features import EDGE_DIM, GLOBAL_DIM, NODE_DIM, OP_VOCAB, PRECISIONS, TARGET_NAMES
from .model import SeerNetConfig, SeerNetMulti


LEGACY_GLOBAL_DIM = 14
LEGACY_GRAPH_FEATURE_DIM = 10
SUPPORTED_CHECKPOINT_GLOBAL_DIMS = (LEGACY_GLOBAL_DIM, GLOBAL_DIM)


def _global_feature_indices(checkpoint_global_dim: int) -> tuple[int, ...]:
    """Map the raw 53/3/40 deployment layout to a checkpoint's global layout."""

    if checkpoint_global_dim == GLOBAL_DIM:
        return tuple(range(GLOBAL_DIM))
    if checkpoint_global_dim == LEGACY_GLOBAL_DIM:
        precision_start = GLOBAL_DIM - len(PRECISIONS)
        return (*range(LEGACY_GRAPH_FEATURE_DIM), *range(precision_start, GLOBAL_DIM))
    raise ValueError(
        f"unsupported checkpoint global_dim {checkpoint_global_dim}; "
        f"expected one of {SUPPORTED_CHECKPOINT_GLOBAL_DIMS}"
    )


class _RawFeatureStudent(nn.Module):
    """Normalize raw graph features and de-normalize all six predictions."""

    def __init__(self, model: SeerNetMulti, stats: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        checkpoint_global_dim = int(model.cfg.global_dim)
        global_feature_indices = _global_feature_indices(checkpoint_global_dim)
        self.register_buffer(
            "global_feature_indices",
            torch.as_tensor(global_feature_indices, dtype=torch.long),
        )
        self.register_buffer(
            "x_mean",
            torch.cat(
                [
                    torch.zeros(len(OP_VOCAB)),
                    torch.as_tensor(stats["x_mean"], dtype=torch.float32),
                ]
            ),
        )
        self.register_buffer(
            "x_std",
            torch.cat(
                [
                    torch.ones(len(OP_VOCAB)),
                    torch.as_tensor(stats["x_std"], dtype=torch.float32),
                ]
            ),
        )
        self.register_buffer("edge_mean", torch.as_tensor(stats["e_mean"], dtype=torch.float32))
        self.register_buffer("edge_std", torch.as_tensor(stats["e_std"], dtype=torch.float32))
        self.register_buffer(
            "u_mean",
            torch.cat(
                [
                    torch.as_tensor(stats["g_mean"], dtype=torch.float32),
                    torch.zeros(len(PRECISIONS)),
                ]
            ),
        )
        self.register_buffer(
            "u_std",
            torch.cat(
                [
                    torch.as_tensor(stats["g_std"], dtype=torch.float32),
                    torch.ones(len(PRECISIONS)),
                ]
            ),
        )
        self.register_buffer("y_mean", torch.as_tensor(stats["y_mean"], dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(stats["y_std"], dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        data = SimpleNamespace(
            x=(x - self.x_mean) / self.x_std,
            edge_index=edge_index,
            edge_attr=(edge_attr - self.edge_mean) / self.edge_std,
            u=(
                torch.index_select(u, 1, self.global_feature_indices) - self.u_mean
            )
            / self.u_std,
            batch=batch,
            num_graphs=1,
        )
        standardized = self.model(data)
        return torch.clamp_min(torch.expm1(standardized * self.y_std + self.y_mean), 0.0)


def _load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    # This is a one-time migration tool for the trusted repository checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"cfg", "model", "stats", "targets"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    cfg = checkpoint["cfg"]
    actual_schema = (cfg["node_dim"], cfg["edge_dim"], cfg["global_dim"])
    if actual_schema[:2] != (NODE_DIM, EDGE_DIM) or actual_schema[2] not in SUPPORTED_CHECKPOINT_GLOBAL_DIMS:
        raise ValueError(
            f"checkpoint schema is {actual_schema}, expected {NODE_DIM}/{EDGE_DIM}/"
            f"{SUPPORTED_CHECKPOINT_GLOBAL_DIMS}"
        )
    if tuple(checkpoint["targets"]) != TARGET_NAMES:
        raise ValueError(f"checkpoint targets are incompatible: {checkpoint['targets']!r}")
    stats = checkpoint["stats"]
    expected_stats = {
        "x_mean": NODE_DIM - len(OP_VOCAB),
        "x_std": NODE_DIM - len(OP_VOCAB),
        "e_mean": EDGE_DIM,
        "e_std": EDGE_DIM,
        "g_mean": actual_schema[2] - len(PRECISIONS),
        "g_std": actual_schema[2] - len(PRECISIONS),
        "y_mean": len(TARGET_NAMES),
        "y_std": len(TARGET_NAMES),
    }
    for key, expected_size in expected_stats.items():
        if key not in stats:
            raise ValueError(f"checkpoint stats are missing {key!r}")
        actual_size = int(np.asarray(stats[key]).size)
        if actual_size != expected_size:
            raise ValueError(
                f"checkpoint stat {key!r} has {actual_size} values, expected {expected_size}"
            )
    return checkpoint


def export_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.jit.ScriptModule:
    """Trace, freeze, verify, and save the CPU deployment artifact."""

    checkpoint = _load_checkpoint(checkpoint_path)
    eager_model = SeerNetMulti(SeerNetConfig.from_dict(checkpoint["cfg"])).cpu()
    eager_model.load_state_dict(checkpoint["model"])
    wrapped = _RawFeatureStudent(eager_model.eval(), checkpoint["stats"]).cpu().eval()
    cpu_inputs = tuple(tensor.detach().cpu() for tensor in example_inputs)
    with torch.inference_mode():
        eager_output = wrapped(*cpu_inputs)
        traced = torch.jit.trace(wrapped, cpu_inputs, strict=False)
        traced = torch.jit.freeze(traced.eval())
        traced_output = traced(*cpu_inputs)
    if traced_output.device.type != "cpu":
        raise RuntimeError("TorchScript verification produced a non-CPU tensor")
    if not torch.isfinite(traced_output).all():
        raise RuntimeError("TorchScript verification produced non-finite outputs")
    torch.testing.assert_close(traced_output, eager_output, rtol=1e-5, atol=1e-5)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(destination))
    loaded = torch.jit.load(str(destination), map_location="cpu").eval()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(*cpu_inputs), eager_output, rtol=1e-5, atol=1e-5)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    example = (
        torch.zeros((2, NODE_DIM)),
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.zeros((1, EDGE_DIM)),
        torch.zeros((1, GLOBAL_DIM)),
        torch.zeros(2, dtype=torch.long),
    )
    export_torchscript(args.checkpoint, args.output, example)


if __name__ == "__main__":
    main()
