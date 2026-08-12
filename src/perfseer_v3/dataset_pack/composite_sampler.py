"""Deterministic composite-block registry and constrained candidate planner."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .coverage_config import OperationCoverageConfig, load_operation_coverage_config
from .fingerprints import canonical_sha256, canonical_value
from .operation_sampler import DEFAULT_BATCH_SIZES
from .operation_support import (
    FROZEN_COMPOSITE_CONTEXTS,
    OperationSupportContract,
    build_operation_support_contract,
)


COMPOSITE_REGISTRY_VERSION = "perfseer_v3_v100_composite_block_registry_v1"
COMPOSITE_PLAN_VERSION = "perfseer_v3_local_qa_composite_plan_v2"
LOCAL_QA_COMPOSITE_POLICY = ("local_qa_fixed", (2_000, 2_000), 1)
LOCAL_QA_COMPOSITE_POLICY_SHA256 = canonical_sha256(LOCAL_QA_COMPOSITE_POLICY)
_CONTEXT_INTERACTIONS = {
    "sequential": ("materialization", "forward_chain", "backward", "optimizer_interaction"),
    "residual_or_branch": ("branch", "join", "residual", "liveness", "backward"),
    "saved_activation_or_alias": (
        "alias",
        "saved_activation",
        "materialization",
        "liveness",
        "backward",
    ),
}


class CompositePlanningError(ValueError):
    """Raised when composite coverage declarations or candidates are incomplete."""


@dataclass(frozen=True)
class CompositeBlockSpec:
    block_id: str
    family: str
    context: str
    canonical_operation_ids: tuple[str, ...]
    required_phases: tuple[str, ...]
    required_backends: tuple[str, ...]
    interaction_requirements: tuple[str, ...]
    golden_test_id: str
    factory_id: str

    def validate(self) -> None:
        if self.context not in FROZEN_COMPOSITE_CONTEXTS:
            raise CompositePlanningError("unknown composite context")
        if not self.canonical_operation_ids:
            raise CompositePlanningError("composite block must cover operations")
        if type(self.canonical_operation_ids) is not tuple or any(
            type(value) is not str or not value for value in self.canonical_operation_ids
        ):
            raise CompositePlanningError("composite operations must be an ordered string tuple")
        for name in ("required_phases", "required_backends"):
            values = getattr(self, name)
            if type(values) is not tuple or not values or any(
                type(value) is not str or not value for value in values
            ):
                raise CompositePlanningError(f"composite {name} must be an ordered tuple")
        if type(self.interaction_requirements) is not tuple:
            raise CompositePlanningError("composite interactions must be an ordered tuple")
        if self.interaction_requirements != _CONTEXT_INTERACTIONS[self.context]:
            raise CompositePlanningError("composite interaction requirements drifted")
        if any(type(value) is not str or not value for value in asdict(self).values() if isinstance(value, str)):
            raise CompositePlanningError("composite identifiers must be non-empty strings")


@dataclass(frozen=True)
class CompositeBlockRegistry:
    version: str
    target_hardware_id: str
    support_contract_sha256: str
    blocks: tuple[CompositeBlockSpec, ...]

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self, support: OperationSupportContract | None = None) -> None:
        support = support or build_operation_support_contract()
        if self.version != COMPOSITE_REGISTRY_VERSION:
            raise CompositePlanningError("composite registry version mismatch")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise CompositePlanningError("composite registry must target NRP V100")
        if self.support_contract_sha256 != support.sha256:
            raise CompositePlanningError("composite registry targets a different support contract")
        if type(self.blocks) is not tuple:
            raise CompositePlanningError("composite blocks must be an ordered tuple")
        for block in self.blocks:
            block.validate()
        if self.blocks != _block_specs(support):
            raise CompositePlanningError(
                "composite block order or exact mapping drifted from the support contract"
            )

    def to_dict(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload["composite_registry_sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class CompositeCandidate:
    candidate_id: str
    block_id: str
    family: str
    context: str
    shape_regime: str
    dtype: str
    accumulation_dtype: str
    phase: str
    layout: str
    backend: str
    optimizer_context: str
    batch_size: int
    variant_seed: int
    coverage_cell_ids: tuple[str, ...]

    def unhashed_payload(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload.pop("candidate_id")
        return payload

    def validate(self, block: CompositeBlockSpec) -> None:
        if self.candidate_id != canonical_sha256(self.unhashed_payload()):
            raise CompositePlanningError("composite candidate ID mismatch")
        if (self.block_id, self.family, self.context) != (
            block.block_id,
            block.family,
            block.context,
        ):
            raise CompositePlanningError("composite candidate block identity mismatch")
        if self.shape_regime not in {"tiny", "small", "medium", "large", "boundary"}:
            raise CompositePlanningError("invalid composite shape regime")
        if self.dtype not in {"float32", "float16"}:
            raise CompositePlanningError("invalid composite dtype")
        if self.accumulation_dtype not in {self.dtype, "float32"}:
            raise CompositePlanningError("invalid composite accumulation dtype")
        if self.phase not in block.required_phases:
            raise CompositePlanningError("composite phase is outside its family contract")
        if (
            self.layout not in {"contiguous", "non_contiguous"}
            or self.backend not in block.required_backends
        ):
            raise CompositePlanningError("invalid composite layout/backend")
        if (
            self.optimizer_context not in {"sgd", "adamw"}
            or type(self.batch_size) is not int
            or self.batch_size not in DEFAULT_BATCH_SIZES
            or type(self.variant_seed) is not int
            or self.variant_seed < 0
        ):
            raise CompositePlanningError("invalid composite optimizer/batch")
        if type(self.coverage_cell_ids) is not tuple:
            raise CompositePlanningError("composite coverage-cell IDs must be an ordered tuple")
        expected_cells = tuple(
            canonical_sha256(
                {
                    "operation": operation,
                    "shape_regime": self.shape_regime,
                    "dtype": self.dtype,
                    "accumulation_dtype": self.accumulation_dtype,
                    "phase": self.phase,
                    "backend": self.backend,
                    "layout": self.layout,
                    "optimizer": self.optimizer_context,
                    "architecture_context": self.context,
                }
            )
            for operation in block.canonical_operation_ids
        )
        if self.coverage_cell_ids != expected_cells:
            raise CompositePlanningError("composite coverage-cell identities drifted")


@dataclass(frozen=True)
class CompositeCorpusPlan:
    version: str
    target_hardware_id: str
    composite_registry_sha256: str
    support_contract_sha256: str
    local_qa_policy_sha256: str
    coverage_config_sha256: str
    size_policy: str
    planning_envelope: tuple[int, int]
    normal_repetitions: int
    candidates: tuple[CompositeCandidate, ...]
    local_qa_only: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(
        self,
        registry: CompositeBlockRegistry | None = None,
        config: OperationCoverageConfig | None = None,
    ) -> None:
        registry = registry or build_composite_block_registry()
        config = config or load_operation_coverage_config()
        registry.validate()
        config.validate()
        if self.version != COMPOSITE_PLAN_VERSION:
            raise CompositePlanningError("composite plan version mismatch")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise CompositePlanningError("composite plan must target NRP V100")
        if self.composite_registry_sha256 != registry.sha256:
            raise CompositePlanningError("composite plan registry mismatch")
        if self.support_contract_sha256 != registry.support_contract_sha256:
            raise CompositePlanningError("composite plan support mismatch")
        if self.local_qa_policy_sha256 != LOCAL_QA_COMPOSITE_POLICY_SHA256:
            raise CompositePlanningError("composite plan local-QA policy fingerprint mismatch")
        if self.coverage_config_sha256 != config.sha256:
            raise CompositePlanningError("composite plan coverage-config fingerprint mismatch")
        if (
            self.size_policy,
            self.planning_envelope,
            self.normal_repetitions,
        ) != LOCAL_QA_COMPOSITE_POLICY:
            raise CompositePlanningError("local-QA composite policy drifted")
        if self.local_qa_only is not True:
            raise CompositePlanningError("composite plan must remain local QA only")
        if not self.planning_envelope[0] <= len(self.candidates) <= self.planning_envelope[1]:
            raise CompositePlanningError("composite plan size is outside its envelope")
        if len({row.candidate_id for row in self.candidates}) != len(self.candidates):
            raise CompositePlanningError("composite candidate IDs must be unique")
        if type(self.candidates) is not tuple:
            raise CompositePlanningError("composite candidates must be an ordered tuple")
        blocks = {block.block_id: block for block in registry.blocks}
        seen_blocks = set()
        shapes: dict[str, set[str]] = {block.block_id: set() for block in registry.blocks}
        counters = [0] * len(registry.blocks)
        for position, candidate in enumerate(self.candidates):
            block = blocks.get(candidate.block_id)
            if block is None:
                raise CompositePlanningError("composite candidate references an unknown block")
            candidate.validate(block)
            block_index = position % len(registry.blocks)
            expected = _candidate(
                registry.blocks[block_index],
                block_index,
                counters[block_index],
            )
            counters[block_index] += 1
            if candidate != expected:
                raise CompositePlanningError(
                    "composite candidate order or deterministic mapping drifted"
                )
            seen_blocks.add(candidate.block_id)
            shapes[candidate.block_id].add(candidate.shape_regime)
        if seen_blocks != set(blocks):
            raise CompositePlanningError("every composite block needs planned candidates")
        if any(values != {"tiny", "small", "medium", "large", "boundary"} for values in shapes.values()):
            raise CompositePlanningError("every composite block needs all five shapes")

    def to_dict(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload["plan_sha256"] = self.sha256
        return payload


def _block_specs(support: OperationSupportContract) -> tuple[CompositeBlockSpec, ...]:
    by_family: dict[str, list[str]] = {}
    ids_by_family: dict[str, tuple[str, ...]] = {}
    phases_by_family: dict[str, tuple[str, ...]] = {}
    backends_by_family: dict[str, tuple[str, ...]] = {}
    for row in support.operations:
        if row.support_state == "required_measured":
            by_family.setdefault(row.family, []).append(row.canonical_id)
            ids_by_family.setdefault(row.family, row.composite_block_ids)
            phases_by_family.setdefault(row.family, row.required_phases)
            backends_by_family.setdefault(row.family, row.required_backends)
    blocks: list[CompositeBlockSpec] = []
    for family in (policy.family for policy in support.families if policy.support_state == "required_measured"):
        for context, block_id in zip(FROZEN_COMPOSITE_CONTEXTS, ids_by_family[family]):
            blocks.append(
                CompositeBlockSpec(
                    block_id=block_id,
                    family=family,
                    context=context,
                    canonical_operation_ids=tuple(by_family[family]),
                    required_phases=phases_by_family[family],
                    required_backends=backends_by_family[family],
                    interaction_requirements=_CONTEXT_INTERACTIONS[context],
                    golden_test_id=block_id.replace("v100_composite:", "v100_composite_golden:"),
                    factory_id=f"perfseer_v3.dataset_pack.composite_blocks.{context}",
                )
            )
    return tuple(blocks)


def build_composite_block_registry(
    support: OperationSupportContract | None = None,
) -> CompositeBlockRegistry:
    support = support or build_operation_support_contract()
    result = CompositeBlockRegistry(
        version=COMPOSITE_REGISTRY_VERSION,
        target_hardware_id=support.target_hardware_id,
        support_contract_sha256=support.sha256,
        blocks=_block_specs(support),
    )
    result.validate(support)
    return result


def _candidate(block: CompositeBlockSpec, block_index: int, ordinal: int) -> CompositeCandidate:
    shapes = ("tiny", "small", "medium", "large", "boundary")
    dtypes = ("float32", "float16")
    layouts = ("contiguous", "non_contiguous")
    shape = shapes[ordinal % len(shapes)]
    dtype = dtypes[(block_index + ordinal * 2) % len(dtypes)]
    accumulation_dtype = "float32" if dtype != "float32" else dtype
    phase = block.required_phases[(block_index + ordinal) % len(block.required_phases)]
    backend = block.required_backends[(block_index + ordinal) % len(block.required_backends)]
    layout = layouts[(block_index + ordinal) % len(layouts)]
    optimizer = ("sgd", "adamw")[(block_index + ordinal) % 2]
    batch_size = DEFAULT_BATCH_SIZES[(block_index * 2 + ordinal) % len(DEFAULT_BATCH_SIZES)]
    variant_seed = block_index * 1_000_003 + ordinal
    cells = tuple(
        canonical_sha256(
            {
                "operation": operation,
                "shape_regime": shape,
                "dtype": dtype,
                "accumulation_dtype": accumulation_dtype,
                "phase": phase,
                "backend": backend,
                "layout": layout,
                "optimizer": optimizer,
                "architecture_context": block.context,
            }
        )
        for operation in block.canonical_operation_ids
    )
    payload = {
        "block_id": block.block_id,
        "family": block.family,
        "context": block.context,
        "shape_regime": shape,
        "dtype": dtype,
        "accumulation_dtype": accumulation_dtype,
        "phase": phase,
        "layout": layout,
        "backend": backend,
        "optimizer_context": optimizer,
        "batch_size": batch_size,
        "variant_seed": variant_seed,
        "coverage_cell_ids": cells,
    }
    result = CompositeCandidate(candidate_id=canonical_sha256(payload), **payload)
    result.validate(block)
    return result


def plan_composite_corpus(
    *,
    target_configurations: int = 2_000,
    registry: CompositeBlockRegistry | None = None,
    config: OperationCoverageConfig | None = None,
) -> CompositeCorpusPlan:
    registry = registry or build_composite_block_registry()
    config = config or load_operation_coverage_config()
    registry.validate()
    config.validate()
    size_policy, envelope, normal_repetitions = LOCAL_QA_COMPOSITE_POLICY
    low, high = envelope
    if type(target_configurations) is not int or not low <= target_configurations <= high:
        raise CompositePlanningError("target composite configurations must stay in the initial envelope")
    minimum = len(registry.blocks) * 5
    if target_configurations < minimum:
        raise CompositePlanningError("target cannot cover every block and shape regime")
    counters = [0] * len(registry.blocks)
    candidates = []
    while len(candidates) < target_configurations:
        index = len(candidates) % len(registry.blocks)
        candidates.append(_candidate(registry.blocks[index], index, counters[index]))
        counters[index] += 1
    result = CompositeCorpusPlan(
        version=COMPOSITE_PLAN_VERSION,
        target_hardware_id=registry.target_hardware_id,
        composite_registry_sha256=registry.sha256,
        support_contract_sha256=registry.support_contract_sha256,
        local_qa_policy_sha256=LOCAL_QA_COMPOSITE_POLICY_SHA256,
        coverage_config_sha256=config.sha256,
        size_policy=size_policy,
        planning_envelope=envelope,
        normal_repetitions=normal_repetitions,
        candidates=tuple(candidates),
        local_qa_only=True,
    )
    result.validate(registry, config)
    return result


def select_composite_gap_wave(
    candidates: Iterable[CompositeCandidate],
    observed_coverage_cell_ids: Iterable[str],
    *,
    limit: int,
    error_by_coverage_cell: Mapping[str, float] | None = None,
) -> tuple[CompositeCandidate, ...]:
    if type(limit) is not int or limit < 0:
        raise CompositePlanningError("gap-wave limit must be a nonnegative integer")
    observed = set(observed_coverage_cell_ids)
    errors = dict(error_by_coverage_cell or {})
    if any(
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in errors.values()
    ):
        raise CompositePlanningError("coverage-cell errors must be nonnegative numbers")
    candidate_rows = tuple(candidates)
    if len({row.candidate_id for row in candidate_rows}) != len(candidate_rows):
        raise CompositePlanningError("gap-wave candidates must not contain duplicate IDs")
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            all(cell in observed for cell in row.coverage_cell_ids),
            -max((float(errors.get(cell, 0.0)) for cell in row.coverage_cell_ids), default=0.0),
            row.block_id,
            row.candidate_id,
        ),
    )
    return tuple(ranked[:limit])


__all__ = [
    "COMPOSITE_PLAN_VERSION",
    "COMPOSITE_REGISTRY_VERSION",
    "CompositeBlockRegistry",
    "CompositeBlockSpec",
    "CompositeCandidate",
    "CompositeCorpusPlan",
    "CompositePlanningError",
    "build_composite_block_registry",
    "plan_composite_corpus",
    "select_composite_gap_wave",
]
