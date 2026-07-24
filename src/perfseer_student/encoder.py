"""Convert a trusted PyTorch model source into raw student predictor tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from perfseer_source_converter import SourceModelSpec, convert_source_to_networkx

from .features import featurize_graph


@dataclass(frozen=True)
class EncodedGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    u: torch.Tensor
    batch: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return self.x, self.edge_index, self.edge_attr, self.u, self.batch


def encode_source(
    source_path: str | Path,
    entry: str,
    input_shapes: Sequence[Sequence[int]],
    precision: str = "fp32_ieee",
    *,
    constructor_args: Sequence[Any] = (),
    constructor_kwargs: dict[str, Any] | None = None,
    input_dtypes: Sequence[str] = ("float32",),
) -> EncodedGraph:
    """Trace and featurize one source model without allocating CUDA tensors."""

    spec = SourceModelSpec(
        source_path=source_path,
        entry=entry,
        input_shapes=input_shapes,
        constructor_args=tuple(constructor_args),
        constructor_kwargs=dict(constructor_kwargs or {}),
        input_dtypes=tuple(input_dtypes),
    )
    graph = convert_source_to_networkx(spec)
    x, edge_index, edge_attr, u = featurize_graph(graph, precision)
    x_tensor = torch.from_numpy(x)
    return EncodedGraph(
        x=x_tensor,
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        u=torch.from_numpy(u),
        batch=torch.zeros(x_tensor.shape[0], dtype=torch.long),
    )
