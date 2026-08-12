"""Fifty deterministic, executable, independently fingerprinted model lineages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from perfseer_v3.op_registry import OperationRegistry

from .fingerprints import canonical_sha256, canonical_value


GENERATED_LINEAGE_VERSION = "perfseer_v3_v100_generated_lineage_v2"
GENERATED_LINEAGE_REGISTRY_VERSION = (
    "perfseer_v3_v100_generated_lineage_registry_v2"
)
GENERATED_LINEAGE_COUNT = 50
GENERATED_HELD_OUT_COUNT = 10
_MODALITIES = ("vision", "nlp", "audio", "tabular", "graph")
_ACTIVATIONS = ("relu", "gelu", "silu", "mish")
_ACTIVATION_OPERATIONS = {
    "relu": "aten.relu",
    "gelu": "aten.gelu",
    "silu": "aten.silu",
    "mish": "aten.mish",
}
_STEM_OPERATIONS = {
    "vision": "aten.convolution.2d",
    "nlp": "aten.embedding",
    "audio": "aten.convolution.1d",
    "tabular": "aten.linear",
    "graph": "aten.linear",
}


class GeneratedLineageError(ValueError):
    """Raised when a generated source lineage is ambiguous or not executable."""


@dataclass(frozen=True)
class GeneratedLineageSpec:
    version: str
    lineage_id: str
    source_split: str
    modality: str
    topology: str
    depth: int
    width: int
    kernel_size: int
    branch_period: int
    activation_pattern: tuple[str, ...]
    architecture_specification: str
    expected_operation_ids: tuple[str, ...]
    structure_sha256: str
    source_sha256: str

    def source_snapshot(self) -> Mapping[str, Any]:
        return canonical_value(
            {
                "version": self.version,
                "lineage_id": self.lineage_id,
                "source_split": self.source_split,
                "modality": self.modality,
                "topology": self.topology,
                "depth": self.depth,
                "width": self.width,
                "kernel_size": self.kernel_size,
                "branch_period": self.branch_period,
                "activation_pattern": self.activation_pattern,
                "architecture_specification": self.architecture_specification,
                "expected_operation_ids": self.expected_operation_ids,
                "factory_id": "perfseer_v3.dataset_pack.models.independent_generated",
                "generator_version": GENERATED_LINEAGE_VERSION,
            }
        )

    def validate(self, *, registered_operations: set[str] | None = None) -> None:
        if self.version != GENERATED_LINEAGE_VERSION:
            raise GeneratedLineageError("generated lineage version mismatch")
        if not self.lineage_id.startswith("generated_lineage_"):
            raise GeneratedLineageError("generated lineage ID is malformed")
        if self.source_split not in {"development", "held_out"}:
            raise GeneratedLineageError("generated lineage split is invalid")
        if self.modality not in _MODALITIES:
            raise GeneratedLineageError("generated lineage modality is invalid")
        if self.topology not in {
            "sequential",
            "residual",
            "branching",
            "sparse_embedding",
        }:
            raise GeneratedLineageError("generated lineage topology is invalid")
        if not 2 <= self.depth <= 8 or not 4 <= self.width <= 32:
            raise GeneratedLineageError("generated lineage dimensions are outside bounds")
        if self.kernel_size not in {1, 3, 5} or not 1 <= self.branch_period <= 4:
            raise GeneratedLineageError("generated lineage structural controls are invalid")
        if len(self.activation_pattern) != self.depth - 1 or any(
            value not in _ACTIVATIONS for value in self.activation_pattern
        ):
            raise GeneratedLineageError("generated lineage activation program is invalid")
        expected_structure = canonical_sha256(
            {
                "modality": self.modality,
                "topology": self.topology,
                "depth": self.depth,
                "width": self.width,
                "kernel_size": self.kernel_size,
                "branch_period": self.branch_period,
                "activation_pattern": self.activation_pattern,
            }
        )
        if self.structure_sha256 != expected_structure:
            raise GeneratedLineageError("generated lineage structure fingerprint drifted")
        if self.source_sha256 != canonical_sha256(self.source_snapshot()):
            raise GeneratedLineageError("generated lineage source fingerprint drifted")
        if not self.expected_operation_ids or len(set(self.expected_operation_ids)) != len(
            self.expected_operation_ids
        ):
            raise GeneratedLineageError("generated lineage operation program is invalid")
        if registered_operations is not None and any(
            operation not in registered_operations
            and not operation.startswith("structural:")
            for operation in self.expected_operation_ids
        ):
            raise GeneratedLineageError("generated lineage references an unknown operation")


@dataclass(frozen=True)
class GeneratedLineageRegistry:
    version: str
    target_hardware_id: str
    generator_version: str
    lineages: tuple[GeneratedLineageSpec, ...]
    training_approved: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.version != GENERATED_LINEAGE_REGISTRY_VERSION:
            raise GeneratedLineageError("generated lineage registry version mismatch")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise GeneratedLineageError("generated lineages must target NRP V100")
        if self.generator_version != GENERATED_LINEAGE_VERSION:
            raise GeneratedLineageError("generated lineage generator version mismatch")
        if self.training_approved is not False:
            raise GeneratedLineageError("local generated lineages cannot approve training")
        if len(self.lineages) != GENERATED_LINEAGE_COUNT:
            raise GeneratedLineageError("generated registry must contain exactly 50 lineages")
        registered = {row.canonical_id for row in OperationRegistry.load().rules}
        for lineage in self.lineages:
            lineage.validate(registered_operations=registered)
        if len({row.lineage_id for row in self.lineages}) != len(self.lineages):
            raise GeneratedLineageError("generated lineage IDs are not unique")
        if len({row.source_sha256 for row in self.lineages}) != len(self.lineages):
            raise GeneratedLineageError("generated source snapshots are not independent")
        if len({row.structure_sha256 for row in self.lineages}) != len(self.lineages):
            raise GeneratedLineageError("generated architecture structures are not unique")
        if {row.modality for row in self.lineages} != set(_MODALITIES):
            raise GeneratedLineageError("generated lineages do not span every modality")
        if sum(row.source_split == "held_out" for row in self.lineages) < 10:
            raise GeneratedLineageError("generated held-out lineage minimum is not met")


def _lineage(index: int) -> GeneratedLineageSpec:
    modality = _MODALITIES[index % len(_MODALITIES)]
    topology = ("residual", "sequential", "branching")[(index // 5) % 3]
    if modality == "nlp" and index in {1, 6}:
        topology = "sparse_embedding"
    depth = 2 + index % 7
    width = 4 + 4 * ((index // 7) % 8)
    kernel_size = (1, 3, 5)[(index * 5 + index // 3) % 3]
    branch_period = 1 + (index * 7 + index // 5) % 4
    activation_pattern = tuple(
        _ACTIVATIONS[(index + layer * (1 + index % 3)) % len(_ACTIVATIONS)]
        for layer in range(depth - 1)
    )
    structure = {
        "modality": modality,
        "topology": topology,
        "depth": depth,
        "width": width,
        "kernel_size": kernel_size,
        "branch_period": branch_period,
        "activation_pattern": activation_pattern,
    }
    structure_sha256 = canonical_sha256(structure)
    specification = (
        f"{topology}:{modality}:d{depth}:w{width}:k{kernel_size}:"
        f"b{branch_period}:p{''.join(value[0] for value in activation_pattern)}:"
        f"{structure_sha256[:12]}"
    )
    operations = [_STEM_OPERATIONS[modality], "aten.mean"]
    if modality == "graph":
        operations.extend(("aten.index_select", "structural:aten.segment_reduce"))
    if topology in {"residual", "branching"}:
        operations.append("aten.add.Tensor")
    operations.extend(_ACTIVATION_OPERATIONS[value] for value in activation_pattern)
    if topology != "sparse_embedding":
        operations.append("aten.linear")
    base = {
        "version": GENERATED_LINEAGE_VERSION,
        "lineage_id": f"generated_lineage_{index:03d}",
        "source_split": "held_out" if index < GENERATED_HELD_OUT_COUNT else "development",
        **structure,
        "architecture_specification": specification,
        "expected_operation_ids": tuple(dict.fromkeys(operations)),
        "structure_sha256": structure_sha256,
    }
    temporary = GeneratedLineageSpec(source_sha256="0" * 64, **base)
    return GeneratedLineageSpec(
        source_sha256=canonical_sha256(temporary.source_snapshot()), **base
    )


def build_generated_lineage_registry() -> GeneratedLineageRegistry:
    result = GeneratedLineageRegistry(
        version=GENERATED_LINEAGE_REGISTRY_VERSION,
        target_hardware_id="nvidia_tesla_v100_sxm2_32gb_nrp",
        generator_version=GENERATED_LINEAGE_VERSION,
        lineages=tuple(_lineage(index) for index in range(GENERATED_LINEAGE_COUNT)),
        training_approved=False,
    )
    result.validate()
    return result


def lineage_for_id(lineage_id: str) -> GeneratedLineageSpec:
    try:
        return next(
            row
            for row in build_generated_lineage_registry().lineages
            if row.lineage_id == lineage_id
        )
    except StopIteration as error:
        raise GeneratedLineageError(f"unknown generated lineage {lineage_id!r}") from error


__all__ = [
    "GENERATED_HELD_OUT_COUNT",
    "GENERATED_LINEAGE_COUNT",
    "GENERATED_LINEAGE_REGISTRY_VERSION",
    "GENERATED_LINEAGE_VERSION",
    "GeneratedLineageError",
    "GeneratedLineageRegistry",
    "GeneratedLineageSpec",
    "build_generated_lineage_registry",
    "lineage_for_id",
]
