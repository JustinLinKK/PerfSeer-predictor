"""Executable factory for the fifty independent generated source lineages."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generated_lineages import GeneratedLineageError, lineage_for_id
from .architecture import resolve_architecture_parameters
from .base import FamilyBuildConfig, FamilyModel, ModelFactoryError


def _activation(value: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return F.relu(value)
    if name == "gelu":
        return F.gelu(value)
    if name == "silu":
        return F.silu(value)
    if name == "mish":
        return F.mish(value)
    raise ModelFactoryError(f"unknown generated activation {name!r}")


def _scatter_add_float32(
    output_shape: tuple[int, ...],
    index: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    destination = index[:, 0]
    order = torch.argsort(destination, stable=True)
    accumulated = torch.segment_reduce(
        source.float().index_select(0, order),
        reduce="sum",
        lengths=torch.bincount(destination, minlength=output_shape[0]),
    )
    return accumulated.to(source.dtype)


class IndependentGeneratedModel(FamilyModel):
    """Interpret one immutable generated-source DSL snapshot as a real model."""

    def __init__(self, config: FamilyBuildConfig) -> None:
        super().__init__()
        config.validate()
        self.family_id = "independent_generated"
        self.task_kind = config.task_kind
        self.output_width = config.output_width
        parameters = resolve_architecture_parameters(
            self.family_id, config.architecture_parameters
        )
        try:
            spec = lineage_for_id(str(parameters["source_lineage"]))
        except GeneratedLineageError as error:
            raise ModelFactoryError(str(error)) from error
        if (
            parameters["source_lineage"] != spec.lineage_id
            or parameters["architecture_specification"]
            != spec.architecture_specification
            or parameters["modality"] != spec.modality
            or type(parameters["depth"]) is not int
            or not 2 <= parameters["depth"] <= spec.depth
            or type(parameters["width"]) is not int
            or not 4 <= parameters["width"] <= spec.width
            or parameters["width"] % 4
        ):
            raise ModelFactoryError(
                "generated architecture parameters differ from their source snapshot"
            )
        self.source_lineage = spec.lineage_id
        self.generated_modality = spec.modality
        self.topology = spec.topology
        self.branch_period = spec.branch_period
        self.depth = int(parameters["depth"])
        self.width = int(parameters["width"])
        self.activation_pattern = spec.activation_pattern[: self.depth - 1]

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self._build_modules(spec.kernel_size)
        self.architecture_semantic_roles = {
            "source_lineage": "selects the immutable generated source snapshot and block program",
            "architecture_specification": "binds the topology, activation, branch, and kernel DSL",
            "modality": "selects the real modality-specific input stem and pooling path",
            "depth": "sets the number of executable generated blocks",
            "width": "sets generated embeddings, channels, projections, and head input width",
        }
        self.bind_architecture(parameters)

    def _block(self, kernel_size: int) -> nn.Module:
        if self.generated_modality == "vision":
            return nn.Conv2d(
                self.width,
                self.width,
                kernel_size,
                padding=kernel_size // 2,
            )
        if self.generated_modality == "audio":
            return nn.Conv1d(
                self.width,
                self.width,
                kernel_size,
                padding=kernel_size // 2,
            )
        return nn.Linear(self.width, self.width)

    def _build_modules(self, kernel_size: int) -> None:
        if self.topology == "sparse_embedding":
            if self.generated_modality != "nlp":
                raise ModelFactoryError("sparse generated lineages require NLP")
            self.sparse_tables = nn.ModuleList(
                nn.Embedding(64, self.width, sparse=True) for _ in range(self.depth)
            )
            self.blocks = nn.ModuleList()
            self.branch_blocks = nn.ModuleList()
            self.head = nn.Identity()
            return
        if self.generated_modality == "vision":
            self.stem = nn.Conv2d(3, self.width, kernel_size, padding=kernel_size // 2)
        elif self.generated_modality == "audio":
            self.stem = nn.Conv1d(1, self.width, kernel_size, padding=kernel_size // 2)
        elif self.generated_modality == "nlp":
            self.stem = nn.Embedding(64, self.width)
        elif self.generated_modality == "tabular":
            self.stem = nn.Linear(8, self.width)
            self.category_stem = nn.Embedding(8, self.width)
        elif self.generated_modality == "graph":
            self.stem = nn.Linear(6, self.width)
        else:
            raise ModelFactoryError(
                f"unsupported generated modality {self.generated_modality!r}"
            )
        self.blocks = nn.ModuleList(
            self._block(kernel_size) for _ in range(self.depth - 1)
        )
        self.branch_blocks = (
            nn.ModuleList(self._block(1) for _ in range(self.depth - 1))
            if self.topology == "branching"
            else nn.ModuleList()
        )
        self.head = nn.Linear(self.width, self.output_width)

    def _stem_value(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        if self.generated_modality == "vision":
            return self.stem(inputs["image"])
        if self.generated_modality == "audio":
            return self.stem(inputs["waveform"])
        if self.generated_modality == "nlp":
            return self.stem(inputs["token_ids"])
        if self.generated_modality == "tabular":
            dense = self.stem(inputs["dense"])
            categories = self.category_stem(inputs["categorical"]).mean(dim=1)
            return torch.ops.aten.add.Tensor(dense, categories)
        if self.generated_modality == "graph":
            return self.stem(inputs["node_features"])
        raise ModelFactoryError("generated modality has no input stem")

    def _graph_update(
        self,
        value: torch.Tensor,
        block: nn.Module,
        inputs: Mapping[str, Any],
    ) -> torch.Tensor:
        edge_index = inputs["edge_index"]
        source = value.index_select(0, edge_index[0])
        messages = block(source)
        return _scatter_add_float32(
            tuple(value.shape),
            edge_index[1].unsqueeze(-1).expand_as(messages),
            messages,
        )

    def _pool(self, value: torch.Tensor, inputs: Mapping[str, Any]) -> torch.Tensor:
        if self.generated_modality == "vision":
            return value.mean(dim=(-2, -1))
        if self.generated_modality == "audio":
            return value.mean(dim=-1)
        if self.generated_modality == "nlp":
            return value.mean(dim=1)
        if self.generated_modality == "tabular":
            return value
        graph_index = inputs["graph_index"]
        graph_count = inputs["graph_count"]
        if type(graph_count) is not int or graph_count < 1:
            raise ModelFactoryError("generated graph input requires graph_count")
        pooled = _scatter_add_float32(
            (graph_count, value.shape[-1]),
            graph_index.unsqueeze(-1).expand_as(value),
            value,
        )
        counts = torch.bincount(graph_index, minlength=graph_count).clamp_min(1)
        return pooled / counts.unsqueeze(-1)

    def forward(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        if self.topology == "sparse_embedding":
            token_ids = inputs["token_ids"]
            values = [table(token_ids).mean(dim=1) for table in self.sparse_tables]
            pooled = torch.stack(values, dim=0).mean(dim=0)
            output_indices = torch.arange(
                self.output_width,
                device=pooled.device,
            ).remainder(self.width)
            return pooled.index_select(1, output_indices)
        value = _activation(self._stem_value(inputs), "silu")
        for index, (block, activation) in enumerate(
            zip(self.blocks, self.activation_pattern)
        ):
            update = (
                self._graph_update(value, block, inputs)
                if self.generated_modality == "graph"
                else block(value)
            )
            update = _activation(update, activation)
            if self.topology == "residual":
                value = torch.ops.aten.add.Tensor(value, update)
            elif self.topology == "branching" and index % self.branch_period == 0:
                branch = self.branch_blocks[index](value)
                value = torch.ops.aten.add.Tensor(update, _activation(branch, "silu"))
            else:
                value = update
        return self.head(self._pool(value, inputs))


def build_model(
    *,
    output_width: int,
    task_kind: str,
    seed: int = 0,
    architecture_parameters: Mapping[str, Any] | None = None,
) -> IndependentGeneratedModel:
    if architecture_parameters is None:
        spec = lineage_for_id("generated_lineage_000")
        architecture_parameters = {
            "source_lineage": spec.lineage_id,
            "architecture_specification": spec.architecture_specification,
            "modality": spec.modality,
            "depth": spec.depth,
            "width": spec.width,
        }
    return IndependentGeneratedModel(
        FamilyBuildConfig(
            output_width=output_width,
            task_kind=task_kind,
            seed=seed,
            architecture_parameters=architecture_parameters,
        )
    )


__all__ = ["IndependentGeneratedModel", "build_model"]
