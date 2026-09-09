"""Exhaustive static gate and pairwise RTX-5090 representative planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import importlib
import heapq
import math
import sys
from functools import lru_cache
from typing import Any, Mapping

from .adapters import TaskAdapter, adapter_for_task
from .compatibility import CompatibilityRequest, evaluate_compatibility
from .fingerprints import canonical_sha256, canonical_value
from .generated_lineages import build_generated_lineage_registry
from .model_registry import load_model_registry
from .local_provenance import validation_harness_sha256
from .quota import load_quota_plan
from .sampler import (
    _FIELD_DOMAINS,
    TargetCandidate,
    TargetManifest,
    build_target_manifest,
    generated_candidate_source_sha256,
)
from .task_registry import load_task_registry


LOCAL_VALIDATION_PLAN_VERSION = "perfseer_v3_local_validation_plan_v9"
SIGNATURE_VERSION = "perfseer_v3_validation_signature_v8"
MAPPING_VERSION = "perfseer_v3_validation_mapping_v9"
PAIRWISE_DIMENSIONS = frozenset(
    {
        "resolution_bucket",
        "sequence_bucket",
        "audio_bucket",
        "graph_bucket",
        "width_bucket",
        "depth_bucket",
        "heads_bucket",
        "experts_bucket",
        "regime",
        "batch_tier",
        "batch_power",
        "precision",
        "accumulation",
        "optimizer",
        "parameter_groups",
        "scheduler",
        "checkpoint",
        "execution",
    }
)


class LocalValidationError(ValueError):
    """Raised when an 18K row is invalid or lacks representative coverage."""


def _value_bucket(field: str, value: Any) -> str:
    domain = _FIELD_DOMAINS.get(field)
    if isinstance(value, bool):
        return str(value).lower()
    if domain is None:
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric <= 8:
                return "tiny"
            if numeric <= 16:
                return "small"
            if numeric <= 32:
                return "medium"
            return "large"
        if isinstance(value, str):
            return value
        return canonical_sha256(value)[:12]
    try:
        index = next(index for index, item in enumerate(domain) if item == value)
    except StopIteration:
        return canonical_sha256(value)[:12]
    if index == 0:
        return "min"
    if index == len(domain) - 1:
        return "max"
    return "median_low" if index < len(domain) / 2 else "median_high"


def _architecture_profile(candidate: TargetCandidate) -> str:
    return f"{candidate.family_id}:{candidate.regime}"


def _shape_profile(candidate: TargetCandidate) -> str:
    return f"{candidate.source_modality}:{candidate.regime}"


def _axis_bucket(
    candidate: TargetCandidate,
    fields: tuple[str, ...],
    *,
    signature_field: str | None = None,
) -> str:
    for field in fields:
        if field in candidate.architecture_parameters:
            return _value_bucket(field, candidate.architecture_parameters[field])
    if signature_field is not None and signature_field in candidate.input_signature:
        domain_field = fields[0]
        return _value_bucket(domain_field, candidate.input_signature[signature_field])
    return "not_applicable"


def _parameter_group_structure(candidate: TargetCandidate) -> str:
    if candidate.optimizer["name"] == "sparse_adam":
        return "all_sparse_trainable"
    if candidate.optimizer["name"] == "muon":
        return "matrix_plus_auxiliary"
    if candidate.family_id == "distilbert_distillation":
        return "student_trainable_teacher_frozen"
    if candidate.family_id == "pix2pix":
        return "generator_and_discriminator"
    return "ordinary_dense"


@lru_cache(maxsize=None)
def _task_kind(task_id: str) -> str:
    return adapter_for_task(task_id).task_kind


def signature_factors(candidate: TargetCandidate) -> Mapping[str, str]:
    """Return implementation-path factors; scalar hyperparameters are deliberately absent."""

    lineage = (
        candidate.source_lineage
        if candidate.family_id == "independent_generated"
        else candidate.family_id
    )
    topology = (
        str(candidate.architecture_parameters.get("architecture_specification", "")).split(":", 1)[0]
        if candidate.family_id == "independent_generated"
        else str(candidate.architecture_parameters.get("variant", candidate.family_id))
    )
    return canonical_value(
        {
            "family": candidate.family_id,
            "source_route": lineage,
            "factory_source": f"{candidate.factory_id}:{candidate.source_sha256}",
            "task_adapter": candidate.task_id,
            "task_kind": _task_kind(candidate.task_id),
            "task_schema": f"{candidate.task_schema_sha256}:width={candidate.target_width}",
            "training_step": candidate.training_step_id,
            "topology": topology,
            "architecture_profile": _architecture_profile(candidate),
            "shape_profile": _shape_profile(candidate),
            "resolution_bucket": _axis_bucket(candidate, ("input_resolution",), signature_field="resolution"),
            "sequence_bucket": _axis_bucket(candidate, ("sequence_length",), signature_field="sequence_length"),
            "audio_bucket": _axis_bucket(candidate, ("clip_samples",), signature_field="clip_samples"),
            "graph_bucket": _axis_bucket(candidate, ("node_count",), signature_field="node_count"),
            "width_bucket": _axis_bucket(candidate, ("width", "hidden_size", "embedding_dim", "channel_dim")),
            "depth_bucket": _axis_bucket(candidate, ("depth", "layers", "layer_count", "encoder_layers", "decoder_layers")),
            "heads_bucket": _axis_bucket(candidate, ("heads",)),
            "experts_bucket": _axis_bucket(candidate, ("expert_count",)),
            "regime": candidate.regime,
            "batch_tier": candidate.batch_plan.effective_tier,
            "batch_power": str(candidate.microbatch_size),
            "precision": (
                f"{candidate.precision_policy['policy_id']}:"
                f"autocast={candidate.precision_policy['autocast']}:"
                f"scaler={candidate.precision_policy['gradient_scaler']}"
            ),
            "accumulation": str(candidate.gradient_accumulation_steps),
            "optimizer": str(candidate.optimizer["name"]),
            "parameter_groups": _parameter_group_structure(candidate),
            "scheduler": (
                f"{candidate.scheduler['name']}:{candidate.scheduler['step_unit']}:"
                f"minimum_lr_ratio={candidate.scheduler['minimum_lr_ratio']}"
            ),
            "scheduler_progress": str(candidate.scheduler["progress"]),
            "checkpoint": f"enabled={candidate.activation_checkpointing['enabled']}",
            "execution": (
                f"{candidate.execution['mode']}:{candidate.execution['backend_id']}:dynamic=false"
            ),
        }
    )


def _coverage_tokens(factors: Mapping[str, str]) -> frozenset[str]:
    items = tuple(sorted(factors.items()))
    singles = {sys.intern(f"value:{name}={value}") for name, value in items}
    pairwise_items = tuple(item for item in items if item[0] in PAIRWISE_DIMENSIONS)
    pairs = {
        sys.intern(f"pair:{left_name}={left_value}|{right_name}={right_value}")
        for index, (left_name, left_value) in enumerate(pairwise_items)
        for right_name, right_value in pairwise_items[index + 1 :]
    }
    # Schedule phase is hash-sensitive executable behavior, but making it a
    # global pairwise axis would multiply the quick local gate across unrelated
    # task and shape dimensions. Cover the interactions that can change whether
    # an optimizer update occurs: scheduler × phase and optimizer × phase.
    for left_name in ("optimizer", "scheduler"):
        pairs.add(
            sys.intern(
                f"pair:{left_name}={factors[left_name]}|"
                f"scheduler_progress={factors['scheduler_progress']}"
            )
        )
    return frozenset((*singles, *pairs))


def _mapping_route(factors: Mapping[str, str]) -> tuple[str, str]:
    return factors["source_route"], factors["task_adapter"]


@dataclass(frozen=True)
class ValidationSignature:
    version: str
    signature_id: str
    representative_candidate_id: str
    factors: Mapping[str, str]
    coverage_tokens: tuple[str, ...]
    forced_route_representative: bool

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(asdict(self))

    def validate(self) -> None:
        if self.version != SIGNATURE_VERSION:
            raise LocalValidationError("validation signature version mismatch")
        if self.signature_id != canonical_sha256(self.factors):
            raise LocalValidationError("validation signature ID differs from its factors")
        if len(self.representative_candidate_id) != 64:
            raise LocalValidationError("representative candidate ID is not SHA-256")
        expected = tuple(sorted(_coverage_tokens(self.factors)))
        if self.coverage_tokens != expected:
            raise LocalValidationError("validation signature coverage tokens drifted")
        if type(self.forced_route_representative) is not bool:
            raise LocalValidationError("forced-route flag must be boolean")


def validation_signature_from_dict(value: Mapping[str, Any]) -> ValidationSignature:
    if not isinstance(value, Mapping) or set(value) != set(ValidationSignature.__dataclass_fields__):
        raise LocalValidationError("serialized validation signature schema differs")
    result = ValidationSignature(
        **{
            **dict(value),
            "coverage_tokens": tuple(value["coverage_tokens"]),
        }
    )
    result.validate()
    return result


@dataclass(frozen=True)
class SignatureMapping:
    version: str
    configuration_id: str
    execution_signature_id: str
    validation_evidence_signature_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.version != MAPPING_VERSION:
            raise LocalValidationError("signature mapping version mismatch")
        for name in ("configuration_id", "execution_signature_id"):
            value = getattr(self, name)
            if len(value) != 64:
                raise LocalValidationError(f"{name} is not SHA-256")
        if not self.validation_evidence_signature_ids:
            raise LocalValidationError("signature mapping has no validation evidence")
        if len(set(self.validation_evidence_signature_ids)) != len(
            self.validation_evidence_signature_ids
        ):
            raise LocalValidationError("signature mapping evidence is duplicated")
        for value in self.validation_evidence_signature_ids:
            if len(value) != 64:
                raise LocalValidationError("validation evidence ID is not SHA-256")


@dataclass(frozen=True)
class LocalValidationPlan:
    version: str
    manifest_sha256: str
    target_hardware_id: str
    validation_hardware_id: str
    accepted_a10g_measurement: bool
    validation_harness_sha256: str
    static_row_count: int
    execution_signature_count: int
    representative_signatures: tuple[ValidationSignature, ...]
    mappings: tuple[SignatureMapping, ...]
    required_value_count: int
    required_pair_count: int
    covered_token_count: int

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self, manifest: TargetManifest | None = None) -> None:
        manifest = manifest or build_target_manifest()
        if self.version != LOCAL_VALIDATION_PLAN_VERSION:
            raise LocalValidationError("local validation plan version mismatch")
        if self.manifest_sha256 != manifest._sha256_unchecked():
            raise LocalValidationError("local validation plan targets another manifest")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise LocalValidationError("local plan target must remain AWS A10G")
        if self.validation_hardware_id != "nvidia_rtx_5090_32gb":
            raise LocalValidationError("local plan must identify the RTX 5090 verifier")
        if self.accepted_a10g_measurement is not False:
            raise LocalValidationError("local validation can never become A10G label evidence")
        if self.validation_harness_sha256 != validation_harness_sha256():
            raise LocalValidationError("local validation harness source identity drifted")
        if (
            len(manifest.candidates) != 18_000
            or len({row.candidate_id for row in manifest.candidates}) != 18_000
            or self.static_row_count != 18_000
            or len(self.mappings) != 18_000
        ):
            raise LocalValidationError("local validation must map all 18,000 rows")
        if len({row.configuration_id for row in self.mappings}) != 18_000:
            raise LocalValidationError("configuration mappings must be unique")
        signatures = {row.signature_id: row for row in self.representative_signatures}
        for row in self.representative_signatures:
            row.validate()
        mappings_by_id: dict[str, SignatureMapping] = {}
        for row in self.mappings:
            row.validate()
            if any(
                signature_id not in signatures
                for signature_id in row.validation_evidence_signature_ids
            ):
                raise LocalValidationError("configuration maps to unselected evidence")
            mappings_by_id[row.configuration_id] = row
        if set(mappings_by_id) != {row.candidate_id for row in manifest.candidates}:
            raise LocalValidationError("configuration mapping differs from the manifest")
        representative_routes = {
            _mapping_route(row.factors): row.signature_id
            for row in self.representative_signatures
            if row.forced_route_representative
        }
        required: set[str] = set()
        execution_signature_ids: set[str] = set()
        forced_values = {
            "family": set(),
            "source_route": set(),
            "task_adapter": set(),
            "training_step": set(),
        }
        for candidate in manifest.candidates:
            factors = signature_factors(candidate)
            mapping = mappings_by_id[candidate.candidate_id]
            execution_signature_id = canonical_sha256(factors)
            if mapping.execution_signature_id != execution_signature_id:
                raise LocalValidationError("row execution-signature identity drifted")
            route_witness = representative_routes.get(_mapping_route(factors))
            if route_witness not in mapping.validation_evidence_signature_ids:
                raise LocalValidationError("row evidence omits its exact source/task route")
            row_required = _coverage_tokens(factors)
            row_covered: set[str] = set()
            for signature_id in mapping.validation_evidence_signature_ids:
                row_covered.update(signatures[signature_id].coverage_tokens)
            if not row_required <= row_covered:
                raise LocalValidationError(
                    "row does not map to factor-complete pairwise validation evidence"
                )
            required.update(row_required)
            execution_signature_ids.add(execution_signature_id)
            forced_values["family"].add(candidate.family_id)
            if candidate.family_id == "independent_generated":
                forced_values["source_route"].add(candidate.source_lineage)
            forced_values["task_adapter"].add(candidate.task_id)
            forced_values["training_step"].add(candidate.training_step_id)
        covered: set[str] = set()
        for signature in self.representative_signatures:
            covered.update(signature.coverage_tokens)
        if not required <= covered:
            raise LocalValidationError("representative set does not cover every valid value and pair")
        value_count = sum(token.startswith("value:") for token in required)
        pair_count = len(required) - value_count
        if (
            self.required_value_count != value_count
            or self.required_pair_count != pair_count
            or self.covered_token_count != len(covered)
        ):
            raise LocalValidationError("pairwise coverage counts drifted")
        if self.execution_signature_count != len(execution_signature_ids):
            raise LocalValidationError("execution signature count drifted")
        selected_values = {
            key: {row.factors[key] for row in self.representative_signatures}
            for key in forced_values
        }
        if any(not values <= selected_values[key] for key, values in forced_values.items()):
            raise LocalValidationError("forced family/lineage/task/training-step coverage is incomplete")


def _static_validate(manifest: TargetManifest) -> tuple[Mapping[str, str], ...]:
    manifest.validate()
    quota = load_quota_plan()
    models = load_model_registry()
    tasks = load_task_registry()
    model_by_family = {row.family_id: row for row in models.entries}
    task_by_id = {row.task_id: row for row in tasks.entries}
    generated = {
        row.lineage_id: row for row in build_generated_lineage_registry().lineages
    }
    imported_factories: set[str] = set()
    factors: list[Mapping[str, str]] = []
    for row in manifest.candidates:
        row.validate()
        model = model_by_family[row.family_id]
        task = task_by_id[row.task_id]
        if row.factory_id not in imported_factories:
            module = importlib.import_module(row.factory_id)
            if not callable(getattr(module, "build_model", None)):
                raise LocalValidationError(f"factory {row.factory_id!r} has no build_model")
            imported_factories.add(row.factory_id)
        if row.factory_id != model.factory_id:
            raise LocalValidationError("candidate factory differs from its family registry")
        if row.family_id == "independent_generated":
            lineage = generated.get(row.source_lineage)
            if lineage is None or row.source_sha256 != generated_candidate_source_sha256(
                lineage.source_sha256,
                model.source_sha256,
            ):
                raise LocalValidationError("generated candidate source snapshot drifted")
        elif (
            row.source_sha256 != model.source_sha256
            or row.source_lineage != model.source_lineage
        ):
            raise LocalValidationError("hand-authored candidate source snapshot drifted")
        adapter = TaskAdapter(task)
        if adapter.task_id != task.task_id or row.task_id not in model.adapter_ids:
            raise LocalValidationError("candidate task adapter is not bound to its family")
        if (
            row.task_schema_sha256 != canonical_sha256(task.target_schema)
            or row.target_width != adapter.target_width
        ):
            raise LocalValidationError("candidate executable task schema binding drifted")
        if row.source_modality == "vision" and row.input_signature.get(
            "class_count"
        ) != row.target_width:
            raise LocalValidationError("vision input signature output width drifted")
        if row.batch_plan.expected_train_examples != task.expected_train_examples:
            raise LocalValidationError("candidate batch cap differs from task training count")
        if task.expected_train_examples // row.microbatch_size < 8:
            raise LocalValidationError("candidate supplies fewer than eight batches per epoch")
        if row.observation_protocol_sha256 != quota.protocol.sha256:
            raise LocalValidationError("candidate observation protocol binding drifted")
        request = CompatibilityRequest(
            family_id=row.family_id,
            modality=row.source_modality,
            architecture_parameters=row.architecture_parameters,
            input_signature=row.input_signature,
            precision_id=str(row.precision_policy["policy_id"]),
            optimizer_id=str(row.optimizer["name"]),
            scheduler_id=str(row.scheduler["name"]),
            execution_mode=str(row.execution["mode"]),
            backend_id=str(row.execution["backend_id"]),
        )
        if request.sha256 != row.compatibility_request_sha256 or not evaluate_compatibility(request).compatible:
            raise LocalValidationError("candidate compatibility binding failed static recheck")
        factors.append(signature_factors(row))
    return tuple(factors)


def build_local_validation_plan(
    manifest: TargetManifest | None = None,
) -> LocalValidationPlan:
    manifest = manifest or build_target_manifest()
    manifest_sha256 = manifest._sha256_unchecked()
    factors_by_candidate = _static_validate(manifest)
    candidates = manifest.candidates
    signature_members: dict[str, list[int]] = {}
    factors_by_signature: dict[str, Mapping[str, str]] = {}
    token_to_index: dict[str, int] = {}
    token_names: list[str] = []
    tokens_by_signature: dict[str, tuple[int, ...]] = {}
    for index, factors in enumerate(factors_by_candidate):
        signature_id = canonical_sha256(factors)
        signature_members.setdefault(signature_id, []).append(index)
        factors_by_signature[signature_id] = factors
        token_indices: list[int] = []
        for token in _coverage_tokens(factors):
            token_index = token_to_index.get(token)
            if token_index is None:
                token_index = len(token_names)
                token_to_index[token] = token_index
                token_names.append(token)
            token_indices.append(token_index)
        tokens_by_signature[signature_id] = tuple(sorted(token_indices))
    representative_index = {
        signature_id: min(indices, key=lambda index: candidates[index].candidate_id)
        for signature_id, indices in signature_members.items()
    }
    route_to_signatures: dict[tuple[str, str], list[str]] = {}
    for signature_id, factors in factors_by_signature.items():
        route_to_signatures.setdefault(_mapping_route(factors), []).append(signature_id)
    forced_signatures = {
        min(
            signature_ids,
            key=lambda signature_id: candidates[representative_index[signature_id]].candidate_id,
        )
        for signature_ids in route_to_signatures.values()
    }
    required_tokens = set(range(len(token_names)))
    selected = set(forced_signatures)
    covered: set[int] = set()
    for signature_id in selected:
        covered.update(tokens_by_signature[signature_id])
    uncovered = required_tokens - covered
    remaining = set(tokens_by_signature) - selected
    heap = [
        (
            -sum(token in uncovered for token in tokens_by_signature[signature_id]),
            candidates[representative_index[signature_id]].candidate_id,
            signature_id,
        )
        for signature_id in remaining
    ]
    heapq.heapify(heap)
    while uncovered:
        while heap:
            negative_score, candidate_id, best = heapq.heappop(heap)
            if best not in remaining:
                continue
            current_score = sum(token in uncovered for token in tokens_by_signature[best])
            if current_score != -negative_score:
                heapq.heappush(heap, (-current_score, candidate_id, best))
                continue
            break
        else:
            raise LocalValidationError("pairwise selector exhausted candidates")
        if current_score == 0:
            raise LocalValidationError("pairwise selector made no progress")
        newly_covered = set(tokens_by_signature[best]) & uncovered
        selected.add(best)
        remaining.remove(best)
        uncovered.difference_update(newly_covered)
    signatures = tuple(
        ValidationSignature(
            version=SIGNATURE_VERSION,
            signature_id=signature_id,
            representative_candidate_id=candidates[representative_index[signature_id]].candidate_id,
            factors=factors_by_signature[signature_id],
            coverage_tokens=tuple(sorted(_coverage_tokens(factors_by_signature[signature_id]))),
            forced_route_representative=signature_id in forced_signatures,
        )
        for signature_id in sorted(
            selected,
            key=lambda value: candidates[representative_index[value]].candidate_id,
        )
    )
    selected_order = tuple(row.signature_id for row in signatures)
    selected_rank = {
        signature_id: index for index, signature_id in enumerate(selected_order)
    }
    selected_tokens = {
        signature_id: _coverage_tokens(factors_by_signature[signature_id])
        for signature_id in selected_order
    }
    token_witness: dict[str, str] = {}
    for signature_id in selected_order:
        for token in selected_tokens[signature_id]:
            token_witness.setdefault(token, signature_id)
    route_representative = {
        _mapping_route(row.factors): row.signature_id
        for row in signatures
        if row.forced_route_representative
    }

    def evidence_for(factors: Mapping[str, str]) -> tuple[str, ...]:
        evidence = {
            route_representative[_mapping_route(factors)],
            *(token_witness[token] for token in _coverage_tokens(factors)),
        }
        return tuple(sorted(evidence, key=selected_rank.__getitem__))

    mappings = tuple(
        SignatureMapping(
            version=MAPPING_VERSION,
            configuration_id=candidate.candidate_id,
            execution_signature_id=canonical_sha256(factors_by_candidate[index]),
            validation_evidence_signature_ids=evidence_for(
                factors_by_candidate[index]
            ),
        )
        for index, candidate in enumerate(candidates)
    )
    value_count = sum(token.startswith("value:") for token in token_names)
    execution_signature_count = len(signature_members)
    required_pair_count = len(token_names) - value_count
    covered_token_count = len(token_names)
    target_hardware_id = manifest.target_hardware_id
    static_row_count = len(candidates)
    del (
        factors_by_candidate,
        signature_members,
        factors_by_signature,
        tokens_by_signature,
        token_to_index,
        token_names,
        required_tokens,
        selected,
        forced_signatures,
        route_to_signatures,
        route_representative,
        selected_order,
        selected_rank,
        selected_tokens,
        token_witness,
        representative_index,
        covered,
        uncovered,
        remaining,
        heap,
    )
    gc.collect()
    return LocalValidationPlan(
        version=LOCAL_VALIDATION_PLAN_VERSION,
        manifest_sha256=manifest_sha256,
        target_hardware_id=target_hardware_id,
        validation_hardware_id="nvidia_rtx_5090_32gb",
        accepted_a10g_measurement=False,
        validation_harness_sha256=validation_harness_sha256(),
        static_row_count=static_row_count,
        execution_signature_count=execution_signature_count,
        representative_signatures=signatures,
        mappings=mappings,
        required_value_count=value_count,
        required_pair_count=required_pair_count,
        covered_token_count=covered_token_count,
    )


__all__ = [
    "LOCAL_VALIDATION_PLAN_VERSION",
    "LocalValidationError",
    "LocalValidationPlan",
    "SignatureMapping",
    "ValidationSignature",
    "build_local_validation_plan",
    "signature_factors",
    "validation_signature_from_dict",
]
