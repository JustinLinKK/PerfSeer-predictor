"""Deterministic constrained planning for the adaptive operation corpus."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from perfseer_v3.op_registry import OperationRegistry

from .coverage_config import OperationCoverageConfig, load_operation_coverage_config
from .fingerprints import canonical_sha256, canonical_value
from .operation_support import OperationSupportContract, build_operation_support_contract


OPERATION_PLAN_VERSION = "perfseer_v3_local_qa_operation_plan_v2"
GENERATOR_REGISTRY_VERSION = "perfseer_v3_a10g_operation_generator_registry_v1"
DEFAULT_BATCH_SIZES = (1, 2, 8)
DEFAULT_OPTIMIZER_CONTEXTS = ("sgd", "adamw", "none")
DEFAULT_ARCHITECTURE_CONTEXTS = (
    "sequential",
    "residual_or_branch",
    "saved_activation_or_alias",
)
LOCAL_QA_OPERATION_POLICY = ("local_qa_fixed", (8_000, 8_000), 1)
LOCAL_QA_OPERATION_POLICY_SHA256 = canonical_sha256(LOCAL_QA_OPERATION_POLICY)


class OperationPlanningError(ValueError):
    """Raised when an operation generator or candidate plan is incomplete."""


@dataclass(frozen=True)
class OperationGeneratorSpec:
    generator_id: str
    canonical_operation_id: str
    raw_target: str
    aliases: tuple[str, ...]
    family: str
    golden_test_id: str
    factory_id: str
    execution_route: str
    required_shape_regimes: tuple[str, ...]
    required_dtypes: tuple[str, ...]
    required_layouts: tuple[str, ...]
    required_phases: tuple[str, ...]
    required_backends: tuple[str, ...]
    composite_block_ids: tuple[str, ...]

    def validate(self) -> None:
        for name in (
            "generator_id",
            "canonical_operation_id",
            "raw_target",
            "family",
            "golden_test_id",
            "factory_id",
            "execution_route",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise OperationPlanningError(f"{name} must be a non-empty string")
        if type(self.aliases) is not tuple or any(
            type(item) is not str or not item for item in self.aliases
        ):
            raise OperationPlanningError("aliases must be a tuple of non-empty strings")
        for name in (
            "required_shape_regimes",
            "required_dtypes",
            "required_layouts",
            "required_phases",
            "required_backends",
            "composite_block_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or not values or any(
                type(item) is not str or not item for item in values
            ):
                raise OperationPlanningError(f"{name} must contain non-empty strings")


@dataclass(frozen=True)
class OperationGeneratorRegistry:
    version: str
    target_hardware_id: str
    operation_registry_sha256: str
    support_contract_sha256: str
    generators: tuple[OperationGeneratorSpec, ...]

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload["generator_registry_sha256"] = self.sha256
        return payload

    def validate(
        self,
        *,
        support: OperationSupportContract | None = None,
        registry: OperationRegistry | None = None,
    ) -> None:
        registry = registry or OperationRegistry.load()
        support = support or build_operation_support_contract(registry=registry)
        if self.version != GENERATOR_REGISTRY_VERSION:
            raise OperationPlanningError("operation generator registry version mismatch")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise OperationPlanningError("operation generators must target AWS A10G")
        if self.operation_registry_sha256 != registry.sha256:
            raise OperationPlanningError("operation generator registry is stale")
        if self.support_contract_sha256 != support.sha256:
            raise OperationPlanningError("operation generators target a different support contract")
        if type(self.generators) is not tuple:
            raise OperationPlanningError("operation generators must be an ordered tuple")
        expected = tuple(
            _generator_spec(row)
            for row in support.operations
            if row.support_state == "required_measured"
        )
        for spec in self.generators:
            spec.validate()
        if self.generators != expected:
            raise OperationPlanningError(
                "operation generator order or exact mapping drifted from the support contract"
            )


@dataclass(frozen=True)
class OperationCandidate:
    candidate_id: str
    generator_id: str
    canonical_operation_id: str
    raw_target: str
    family: str
    shape_regime: str
    dtype: str
    accumulation_dtype: str
    phase: str
    backend: str
    layout: str
    optimizer_context: str
    architecture_context: str
    batch_size: int
    variant_seed: int
    coverage_cell_id: str

    def unhashed_payload(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload.pop("candidate_id")
        return payload

    def validate(self, generator: OperationGeneratorSpec) -> None:
        if self.candidate_id != canonical_sha256(self.unhashed_payload()):
            raise OperationPlanningError("operation candidate ID does not match its payload")
        if self.generator_id != generator.generator_id:
            raise OperationPlanningError("operation candidate generator mismatch")
        if (
            self.canonical_operation_id != generator.canonical_operation_id
            or self.raw_target != generator.raw_target
            or self.family != generator.family
        ):
            raise OperationPlanningError("operation candidate identity mismatch")
        dimensions = {
            "shape_regime": (self.shape_regime, generator.required_shape_regimes),
            "dtype": (self.dtype, generator.required_dtypes),
            "phase": (self.phase, generator.required_phases),
            "backend": (self.backend, generator.required_backends),
            "layout": (self.layout, generator.required_layouts),
        }
        for name, (value, allowed) in dimensions.items():
            if value not in allowed:
                raise OperationPlanningError(f"operation candidate {name} is outside its generator")
        if self.accumulation_dtype not in {self.dtype, "float32"}:
            raise OperationPlanningError("unsupported accumulation dtype")
        if self.optimizer_context not in DEFAULT_OPTIMIZER_CONTEXTS:
            raise OperationPlanningError("unsupported optimizer context")
        if self.architecture_context not in DEFAULT_ARCHITECTURE_CONTEXTS:
            raise OperationPlanningError("unsupported architecture context")
        if (
            type(self.batch_size) is not int
            or self.batch_size not in DEFAULT_BATCH_SIZES
            or type(self.variant_seed) is not int
            or self.variant_seed < 0
        ):
            raise OperationPlanningError("invalid operation batch size or variant seed")
        expected_cell = canonical_sha256(
            {
                "operation": self.canonical_operation_id,
                "shape_regime": self.shape_regime,
                "dtype": self.dtype,
                "accumulation_dtype": self.accumulation_dtype,
                "phase": self.phase,
                "backend": self.backend,
                "layout": self.layout,
                "optimizer": self.optimizer_context,
                "architecture_context": self.architecture_context,
            }
        )
        if self.coverage_cell_id != expected_cell:
            raise OperationPlanningError("operation coverage-cell ID mismatch")


@dataclass(frozen=True)
class OperationCorpusPlan:
    version: str
    target_hardware_id: str
    generator_registry_sha256: str
    support_contract_sha256: str
    local_qa_policy_sha256: str
    coverage_config_sha256: str
    size_policy: str
    planning_envelope: tuple[int, int]
    normal_repetitions: int
    candidates: tuple[OperationCandidate, ...]
    local_qa_only: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(
        self,
        generator_registry: OperationGeneratorRegistry | None = None,
        config: OperationCoverageConfig | None = None,
    ) -> None:
        generator_registry = generator_registry or build_operation_generator_registry()
        config = config or load_operation_coverage_config()
        generator_registry.validate()
        config.validate()
        if self.version != OPERATION_PLAN_VERSION:
            raise OperationPlanningError("operation plan version mismatch")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise OperationPlanningError("operation plan must target AWS A10G")
        if self.generator_registry_sha256 != generator_registry.sha256:
            raise OperationPlanningError("operation plan generator registry mismatch")
        if self.support_contract_sha256 != generator_registry.support_contract_sha256:
            raise OperationPlanningError("operation plan support contract mismatch")
        if self.local_qa_policy_sha256 != LOCAL_QA_OPERATION_POLICY_SHA256:
            raise OperationPlanningError("operation plan local-QA policy fingerprint mismatch")
        if self.coverage_config_sha256 != config.sha256:
            raise OperationPlanningError("operation plan coverage-config fingerprint mismatch")
        if (
            self.size_policy,
            self.planning_envelope,
            self.normal_repetitions,
        ) != LOCAL_QA_OPERATION_POLICY:
            raise OperationPlanningError("local-QA operation policy drifted")
        if self.local_qa_only is not True:
            raise OperationPlanningError("operation plan must remain local QA only")
        if not self.planning_envelope[0] <= len(self.candidates) <= self.planning_envelope[1]:
            raise OperationPlanningError("operation plan size is outside the initial envelope")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise OperationPlanningError("operation candidate IDs must be unique")
        if type(self.candidates) is not tuple:
            raise OperationPlanningError("operation candidates must be an ordered tuple")
        by_generator = {spec.generator_id: spec for spec in generator_registry.generators}
        shapes: dict[str, set[str]] = {key: set() for key in by_generator}
        counters = [0] * len(generator_registry.generators)
        for position, candidate in enumerate(self.candidates):
            generator = by_generator.get(candidate.generator_id)
            if generator is None:
                raise OperationPlanningError("operation candidate references an unknown generator")
            candidate.validate(generator)
            generator_index = position % len(generator_registry.generators)
            expected = _candidate(
                generator_registry.generators[generator_index],
                operation_index=generator_index,
                ordinal=counters[generator_index],
            )
            counters[generator_index] += 1
            if candidate != expected:
                raise OperationPlanningError(
                    "operation candidate order or deterministic mapping drifted"
                )
            shapes[candidate.generator_id].add(candidate.shape_regime)
        for generator_id, generator in by_generator.items():
            if shapes[generator_id] != set(generator.required_shape_regimes):
                raise OperationPlanningError(f"generator {generator_id} lacks all five shape regimes")

    def to_dict(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload["plan_sha256"] = self.sha256
        return payload


def _execution_route(family: str, raw_target: str) -> str:
    if family == "optimizer":
        return "optimizer_step_annotation"
    if raw_target.startswith("perfseer::"):
        return "training_graph_annotation"
    if raw_target.startswith(("prim::", "operator::")) or raw_target == "aten::sym_size.int":
        return "capture_observed_python_semantics"
    if raw_target in {
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_cudnn_rnn",
    }:
        return "a10g_environment_gated_dispatcher"
    return "direct_or_higher_level_dispatcher"


def _generator_spec(row: object) -> OperationGeneratorSpec:
    return OperationGeneratorSpec(
        generator_id=row.microbenchmark_generator_ids[0],
        canonical_operation_id=row.canonical_id,
        raw_target=row.raw_target,
        aliases=row.aliases,
        family=row.family,
        golden_test_id=row.golden_test_ids[0],
        factory_id=f"perfseer_v3.dataset_pack.operation_benchmarks.{row.family}",
        execution_route=_execution_route(row.family, row.raw_target),
        required_shape_regimes=row.required_shape_regimes,
        required_dtypes=row.required_dtypes,
        required_layouts=row.required_layouts,
        required_phases=row.required_phases,
        required_backends=row.required_backends,
        composite_block_ids=row.composite_block_ids,
    )


def build_operation_generator_registry(
    *,
    support: OperationSupportContract | None = None,
    registry: OperationRegistry | None = None,
) -> OperationGeneratorRegistry:
    registry = registry or OperationRegistry.load()
    support = support or build_operation_support_contract(registry=registry)
    generators = []
    for row in support.operations:
        if row.support_state != "required_measured":
            continue
        generators.append(_generator_spec(row))
    result = OperationGeneratorRegistry(
        version=GENERATOR_REGISTRY_VERSION,
        target_hardware_id=support.target_hardware_id,
        operation_registry_sha256=registry.sha256,
        support_contract_sha256=support.sha256,
        generators=tuple(generators),
    )
    result.validate(support=support, registry=registry)
    return result


def _select(values: tuple[str, ...], operation_index: int, ordinal: int, stride: int) -> str:
    return values[(operation_index + ordinal * stride) % len(values)]


def _candidate(
    spec: OperationGeneratorSpec,
    *,
    operation_index: int,
    ordinal: int,
) -> OperationCandidate:
    shape = spec.required_shape_regimes[ordinal % len(spec.required_shape_regimes)]
    dtype = _select(spec.required_dtypes, operation_index, ordinal, 2)
    phase = _select(spec.required_phases, operation_index, ordinal, 1)
    backend = _select(spec.required_backends, operation_index, ordinal, 1)
    layout = _select(spec.required_layouts, operation_index, ordinal, 1)
    optimizer = DEFAULT_OPTIMIZER_CONTEXTS[(operation_index + ordinal) % 3]
    architecture_context = DEFAULT_ARCHITECTURE_CONTEXTS[(operation_index + ordinal * 2) % 3]
    accumulation_dtype = "float32" if dtype in {"float16", "bfloat16"} else dtype
    batch_size = DEFAULT_BATCH_SIZES[(operation_index * 2 + ordinal) % 3]
    variant_seed = operation_index * 1_000_003 + ordinal
    coverage_cell_id = canonical_sha256(
        {
            "operation": spec.canonical_operation_id,
            "shape_regime": shape,
            "dtype": dtype,
            "accumulation_dtype": accumulation_dtype,
            "phase": phase,
            "backend": backend,
            "layout": layout,
            "optimizer": optimizer,
            "architecture_context": architecture_context,
        }
    )
    payload = {
        "generator_id": spec.generator_id,
        "canonical_operation_id": spec.canonical_operation_id,
        "raw_target": spec.raw_target,
        "family": spec.family,
        "shape_regime": shape,
        "dtype": dtype,
        "accumulation_dtype": accumulation_dtype,
        "phase": phase,
        "backend": backend,
        "layout": layout,
        "optimizer_context": optimizer,
        "architecture_context": architecture_context,
        "batch_size": batch_size,
        "variant_seed": variant_seed,
        "coverage_cell_id": coverage_cell_id,
    }
    candidate = OperationCandidate(candidate_id=canonical_sha256(payload), **payload)
    candidate.validate(spec)
    return candidate


def plan_operation_corpus(
    *,
    target_configurations: int = 8_000,
    generator_registry: OperationGeneratorRegistry | None = None,
    config: OperationCoverageConfig | None = None,
) -> OperationCorpusPlan:
    generator_registry = generator_registry or build_operation_generator_registry()
    config = config or load_operation_coverage_config()
    generator_registry.validate()
    config.validate()
    size_policy, envelope, normal_repetitions = LOCAL_QA_OPERATION_POLICY
    low, high = envelope
    if type(target_configurations) is not int or not low <= target_configurations <= high:
        raise OperationPlanningError("target operation configurations must stay in the initial envelope")
    generators = generator_registry.generators
    minimum = sum(len(spec.required_shape_regimes) for spec in generators)
    if target_configurations < minimum:
        raise OperationPlanningError("target cannot cover five shapes for every generator")
    counters = [0] * len(generators)
    candidates: list[OperationCandidate] = []
    while len(candidates) < target_configurations:
        index = len(candidates) % len(generators)
        ordinal = counters[index]
        candidates.append(_candidate(generators[index], operation_index=index, ordinal=ordinal))
        counters[index] += 1
    plan = OperationCorpusPlan(
        version=OPERATION_PLAN_VERSION,
        target_hardware_id=generator_registry.target_hardware_id,
        generator_registry_sha256=generator_registry.sha256,
        support_contract_sha256=generator_registry.support_contract_sha256,
        local_qa_policy_sha256=LOCAL_QA_OPERATION_POLICY_SHA256,
        coverage_config_sha256=config.sha256,
        size_policy=size_policy,
        planning_envelope=envelope,
        normal_repetitions=normal_repetitions,
        candidates=tuple(candidates),
        local_qa_only=True,
    )
    plan.validate(generator_registry, config)
    return plan


def select_operation_gap_wave(
    candidates: Iterable[OperationCandidate],
    observed_coverage_cell_ids: Iterable[str],
    *,
    limit: int,
    error_by_coverage_cell: Mapping[str, float] | None = None,
) -> tuple[OperationCandidate, ...]:
    if type(limit) is not int or limit < 0:
        raise OperationPlanningError("gap-wave limit must be a nonnegative integer")
    observed = set(observed_coverage_cell_ids)
    errors = dict(error_by_coverage_cell or {})
    if any(
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in errors.values()
    ):
        raise OperationPlanningError("coverage-cell errors must be nonnegative numbers")
    candidate_rows = tuple(candidates)
    if len({candidate.candidate_id for candidate in candidate_rows}) != len(candidate_rows):
        raise OperationPlanningError("gap-wave candidates must not contain duplicate IDs")
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            row.coverage_cell_id in observed,
            -float(errors.get(row.coverage_cell_id, 0.0)),
            row.coverage_cell_id,
            row.candidate_id,
        ),
    )
    return tuple(ranked[:limit])


__all__ = [
    "GENERATOR_REGISTRY_VERSION",
    "OPERATION_PLAN_VERSION",
    "OperationCandidate",
    "OperationCorpusPlan",
    "OperationGeneratorRegistry",
    "OperationGeneratorSpec",
    "OperationPlanningError",
    "build_operation_generator_registry",
    "plan_operation_corpus",
    "select_operation_gap_wave",
]
