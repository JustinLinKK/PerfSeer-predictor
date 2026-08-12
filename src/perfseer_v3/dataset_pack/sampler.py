"""Deterministic constrained planner for the exact 18K V100 target manifest."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

from perfseer_v3.op_registry import OperationRegistry

from .adapters import TaskAdapter, adapter_for_task
from .compatibility import (
    DEPLOYMENT_OPTIMIZERS,
    DEPLOYMENT_SCHEDULERS,
    MUON_MATRIX_FREE_FAMILIES,
    CompatibilityRequest,
    evaluate_compatibility,
)
from .fingerprints import canonical_sha256, canonical_value
from .generated_lineages import (
    GENERATED_LINEAGE_VERSION,
    GeneratedLineageSpec,
    build_generated_lineage_registry,
)
from .model_registry import ModelRegistry, load_model_registry
from .models.architecture import resolve_architecture_parameters
from .quota import QuotaPlan, load_quota_plan
from .task_registry import TaskRegistry, load_task_registry


TARGET_MANIFEST_VERSION = "perfseer_v3_v100_18k_target_manifest_v8"
CANDIDATE_VERSION = "perfseer_v3_v100_candidate_v8"
PLANNER_VERSION = "perfseer_v3_v100_constrained_sampler_v8"
BATCH_PLAN_VERSION = "perfseer_v3_v100_batch_plan_v2"
REGIMES = ("task_realistic", "shape_extrapolation", "memory_frontier")
PRECISION_PATTERN = (
    *("fp32_ieee",) * 5,
    *("fp16_grad_scaler",) * 11,
    *("mixed_structured",) * 4,
)
GRADIENT_ACCUMULATION = (1, 2, 4, 8)
SCHEDULE_PROGRESS = (0.0, 0.1, 0.5, 0.9, 1.0)
OPTIMIZER_STEP_SCHEDULERS = frozenset(
    {
        "constant",
        "constant_with_warmup",
        "linear",
        "linear_with_warmup",
        "cosine_with_warmup",
        "polynomial",
        "one_cycle",
        "cyclic",
        "inverse_sqrt",
        "warmup_stable_decay",
    }
)
EPOCH_STEP_SCHEDULERS = frozenset(
    {"step", "multi_step", "exponential", "cosine", "cosine_warm_restarts", "reduce_on_plateau"}
)
TERMINAL_DECAY_SCHEDULERS = frozenset(
    {"polynomial", "cosine", "cosine_with_warmup", "one_cycle"}
)
MINIMUM_TERMINAL_LEARNING_RATE_RATIO = 0.1


def generated_candidate_source_sha256(
    lineage_source_sha256: str,
    interpreter_source_sha256: str,
) -> str:
    """Bind generated DSL identity to the executable interpreter source."""

    return canonical_sha256(
        {
            "lineage_source_sha256": lineage_source_sha256,
            "interpreter_source_sha256": interpreter_source_sha256,
        }
    )


def scheduler_step_unit(name: str) -> str:
    if name == "none":
        return "none"
    if name in OPTIMIZER_STEP_SCHEDULERS:
        return "optimizer_step"
    if name in EPOCH_STEP_SCHEDULERS:
        return "epoch"
    raise CandidatePlanningError(f"scheduler {name!r} has no frozen step unit")


def scheduler_minimum_lr_ratio(name: str) -> float:
    """Return the explicit nonzero tail used by terminal-decay schedules."""

    if name not in DEPLOYMENT_SCHEDULERS:
        raise CandidatePlanningError(f"scheduler {name!r} has no frozen LR policy")
    return (
        MINIMUM_TERMINAL_LEARNING_RATE_RATIO
        if name in TERMINAL_DECAY_SCHEDULERS
        else 0.0
    )


class CandidatePlanningError(ValueError):
    """Raised when exact quotas or candidate identities do not validate."""


@lru_cache(maxsize=1)
def _registered_operation_ids() -> frozenset[str]:
    return frozenset(row.canonical_id for row in OperationRegistry.load().rules)


_FIELD_DOMAINS: Mapping[str, tuple[Any, ...]] = {
    "activation": ("elu", "prelu"),
    "atom_features": (4, 6, 8, 12),
    "aux_logits": (False, True),
    "base_channels": (4, 8, 12, 16),
    "capacity_factor": (0.75, 1.0, 1.25, 1.5),
    "channel_dim": (8, 12, 16, 24),
    "channels": (4, 8, 12, 16),
    "clip_samples": (128, 192, 256, 384, 512),
    "components": (2, 3, 4, 6),
    "decoder_layers": (1, 2, 3, 4),
    "depth": (2, 3, 4, 6),
    "depth_multiplier": (0.5, 0.75, 1.0, 1.5),
    "depths": ([1, 1], [2, 1], [1, 2], [2, 2]),
    "discriminator_width": (4, 8, 12, 16),
    "embedding_dim": (8, 12, 16, 24),
    "encoder_layers": (1, 2, 3, 4),
    "expert_count": (2, 3, 4, 8),
    "feature_count": (6, 8, 12, 16),
    "generator_width": (4, 8, 12, 16),
    "heads": (1, 2, 4),
    "hidden_size": (8, 16, 24, 32),
    "input_resolution": (16, 24, 32, 48),
    "kernel_size": (3, 5, 7),
    "kv_heads": (1, 2),
    "latent_rank": (2, 4, 6, 8),
    "layer_count": (1, 2, 3, 4),
    "layers": (1, 2, 3, 4),
    "levels": (1, 2, 3, 4),
    "localization_width": (4, 8, 12, 16),
    "mel_bins": (8, 16, 24, 32),
    "message_layers": (1, 2, 3, 4),
    "neighbor_count": (1, 2, 4, 8),
    "neighbor_limit": (4, 8, 12, 16),
    "ngram_buckets": (32, 64, 128, 256),
    "node_count": (8, 16, 32, 64, 128),
    "edge_count": (16, 32, 64, 128, 256, 512),
    "patch_size": (2, 4, 8),
    "sample_rate": (8_000, 16_000, 22_050, 32_000),
    "sequence_length": (8, 16, 32, 64, 128),
    "sparse_gradients": (False, True),
    "stage_depths": ([1, 1], [2, 1], [1, 2], [2, 2]),
    "state_size": (4, 8, 12, 16),
    "student_layers": (1, 2, 3, 4),
    "tag_count": (3, 4, 5, 6),
    "temperature": (1.0, 2.0, 3.0, 4.0),
    "token_dim": (8, 12, 16, 24),
    "top_k": (1, 2, 3),
    "variant": ("b0", "b4"),
    "width": (8, 12, 16, 24, 32),
    "width_multiplier": (0.25, 0.5, 0.75, 1.0),
    "window_size": (1, 2, 4),
}


LIGHT_BATCH_LADDER = (16, 32, 64, 128, 256, 512)
STANDARD_BATCH_LADDER = (8, 16, 32, 64, 128, 256)
HEAVY_BATCH_LADDER = (1, 2, 4, 8, 16, 32, 64)
BATCH_LADDERS = {
    "light": LIGHT_BATCH_LADDER,
    "standard": STANDARD_BATCH_LADDER,
    "heavy": HEAVY_BATCH_LADDER,
}
LIGHT_FAMILIES = {
    "mobilenet_v3_large",
    "prelu_elu_cnn",
    "fasttext_embeddingbag",
    "selu_mlp",
    "mixture_density_network",
}
HEAVY_FAMILIES = {
    "vit_s16",
    "swin_t",
    "unet_groupnorm",
    "pix2pix",
    "restormer",
    "t5_small",
    "llama_small",
    "switch_moe",
    "distilbert_distillation",
    "panns_cnn14",
    "cgcnn",
}

_DENSE_ATTENTION_FAMILIES = {
    "bert_base",
    "gpt2_small",
    "t5_small",
    "llama_small",
    "mla_mini_transformer",
    "kimi_delta_attention",
    "switch_moe",
    "distilbert_distillation",
}


@dataclass(frozen=True)
class BatchPlan:
    version: str
    base_tier: str
    effective_tier: str
    full_ladder: tuple[int, ...]
    eligible_ladder: tuple[int, ...]
    expected_train_examples: int
    maximum_batch_for_eight_batches: int
    promotion_reasons: tuple[str, ...]
    selected_microbatch: int

    def validate(self) -> None:
        if self.version != BATCH_PLAN_VERSION:
            raise CandidatePlanningError("batch-plan version mismatch")
        if self.base_tier not in BATCH_LADDERS or self.effective_tier not in BATCH_LADDERS:
            raise CandidatePlanningError("batch tier is invalid")
        if self.full_ladder != BATCH_LADDERS[self.effective_tier]:
            raise CandidatePlanningError("batch plan does not use the complete tier ladder")
        if type(self.expected_train_examples) is not int or self.expected_train_examples < 8:
            raise CandidatePlanningError("expected training example count is invalid")
        expected_cap = self.expected_train_examples // 8
        if self.maximum_batch_for_eight_batches != expected_cap:
            raise CandidatePlanningError("eight-batch cap differs from the prepared epoch view")
        expected_eligible = tuple(value for value in self.full_ladder if value <= expected_cap)
        if not expected_eligible or self.eligible_ladder != expected_eligible:
            raise CandidatePlanningError("eligible ladder differs from the eight-batch cap")
        if self.selected_microbatch not in self.eligible_ladder:
            raise CandidatePlanningError("selected microbatch is outside the eligible ladder")
        if len(set(self.promotion_reasons)) != len(self.promotion_reasons):
            raise CandidatePlanningError("batch promotion reasons must be unique")


@dataclass(frozen=True)
class TargetCandidate:
    version: str
    candidate_id: str
    ordinal: int
    quota_modality: str
    source_modality: str
    family_id: str
    regime: str
    source_lineage: str
    source_sha256: str
    source_split: str
    factory_id: str
    generator_version: str
    mutation_specification: Mapping[str, Any]
    architecture_parameters: Mapping[str, Any]
    task_id: str
    task_schema_sha256: str
    target_width: int
    dataset_revision: str
    input_signature: Mapping[str, Any]
    training_step_id: str
    batch_plan: BatchPlan
    microbatch_size: int
    gradient_accumulation_steps: int
    precision_policy: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    activation_checkpointing: Mapping[str, Any]
    execution: Mapping[str, Any]
    seed_policy: Mapping[str, Any]
    coverage_cell_specs: tuple[Mapping[str, Any], ...]
    coverage_cell_ids: tuple[str, ...]
    compatibility_request_sha256: str
    observation_protocol_sha256: str
    configuration_binding_state: str
    target_hardware_id: str
    training_approved: bool

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(asdict(self))

    def unhashed_payload(self) -> dict[str, Any]:
        payload = canonical_value(asdict(self))
        payload.pop("candidate_id")
        return payload

    def validate(self) -> None:
        if self.version != CANDIDATE_VERSION:
            raise CandidatePlanningError("candidate version mismatch")
        if self.candidate_id != canonical_sha256(self.unhashed_payload()):
            raise CandidatePlanningError("candidate ID differs from its canonical payload")
        if self.regime not in REGIMES or self.ordinal < 0:
            raise CandidatePlanningError("candidate regime or ordinal is invalid")
        if self.source_split not in {"development", "held_out"}:
            raise CandidatePlanningError("candidate source split is invalid")
        if not self.factory_id:
            raise CandidatePlanningError("candidate factory identity is empty")
        if (
            len(self.task_schema_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.task_schema_sha256
            )
            or type(self.target_width) is not int
            or self.target_width < 1
        ):
            raise CandidatePlanningError("candidate executable task schema is invalid")
        if self.configuration_binding_state != "task_materialization_required":
            raise CandidatePlanningError("local targets must await task materialization")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise CandidatePlanningError("candidate target hardware differs from NRP V100")
        if self.training_approved is not False:
            raise CandidatePlanningError("unmeasured target candidates cannot approve training")
        self.batch_plan.validate()
        expected_base_tier, expected_effective_tier, expected_promotion_reasons = (
            _batch_classification(
                self.family_id,
                self.architecture_parameters,
                self.input_signature,
                str(self.precision_policy.get("policy_id", "")),
                bool(self.activation_checkpointing.get("enabled", False)),
                str(self.optimizer.get("name", "")),
            )
        )
        if self.batch_plan.base_tier != expected_base_tier:
            raise CandidatePlanningError("candidate base batch tier differs from its classifier")
        if "oom_repair_history" not in self.mutation_specification and (
            self.batch_plan.effective_tier != expected_effective_tier
            or self.batch_plan.promotion_reasons != expected_promotion_reasons
        ):
            raise CandidatePlanningError("candidate batch promotion differs from its pressure score")
        if (
            type(self.microbatch_size) is not int
            or self.microbatch_size < 1
            or self.microbatch_size != self.batch_plan.selected_microbatch
        ):
            raise CandidatePlanningError("candidate microbatch differs from its batch plan")
        if len(self.observation_protocol_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.observation_protocol_sha256
        ):
            raise CandidatePlanningError("candidate observation protocol hash is invalid")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise CandidatePlanningError("gradient accumulation must be a positive integer")
        learning_rate = self.optimizer.get("learning_rate")
        weight_decay = self.optimizer.get("weight_decay")
        if (
            set(self.optimizer) != {"name", "learning_rate", "weight_decay"}
            or type(self.optimizer.get("name")) is not str
            or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0
            or isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not math.isfinite(float(weight_decay))
            or float(weight_decay) < 0
        ):
            raise CandidatePlanningError("optimizer hyperparameters are invalid")
        progress = self.scheduler.get("progress")
        minimum_lr_ratio = self.scheduler.get("minimum_lr_ratio")
        if (
            set(self.scheduler)
            != {"name", "progress", "step_unit", "minimum_lr_ratio"}
            or type(self.scheduler.get("name")) is not str
            or self.scheduler.get("step_unit")
            != scheduler_step_unit(str(self.scheduler.get("name")))
            or isinstance(minimum_lr_ratio, bool)
            or not isinstance(minimum_lr_ratio, (int, float))
            or not math.isfinite(float(minimum_lr_ratio))
            or float(minimum_lr_ratio)
            != scheduler_minimum_lr_ratio(str(self.scheduler.get("name")))
            or isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not math.isfinite(float(progress))
            or not 0.0 <= float(progress) <= 1.0
        ):
            raise CandidatePlanningError("scheduler progress is invalid")
        policy_id = self.precision_policy.get("policy_id")
        expected_autocast = policy_id != "fp32_ieee"
        expected_scaler = policy_id == "fp16_grad_scaler"
        if (
            set(self.precision_policy) != {"policy_id", "autocast", "gradient_scaler"}
            or type(policy_id) is not str
            or self.precision_policy.get("autocast") is not expected_autocast
            or self.precision_policy.get("gradient_scaler") is not expected_scaler
        ):
            raise CandidatePlanningError("precision policy fields are inconsistent")
        if (
            set(self.activation_checkpointing)
            not in ({"enabled"}, {"enabled", "segments"})
            or type(self.activation_checkpointing.get("enabled")) is not bool
            or (
                "segments" in self.activation_checkpointing
                and (
                    type(self.activation_checkpointing["segments"]) is not int
                    or self.activation_checkpointing["segments"] < 1
                )
            )
        ):
            raise CandidatePlanningError("activation-checkpointing policy is invalid")
        if (
            set(self.execution) != {"mode", "backend_id"}
            or type(self.execution.get("mode")) is not str
            or type(self.execution.get("backend_id")) is not str
        ):
            raise CandidatePlanningError("execution policy is invalid")
        if (
            set(self.seed_policy) != {"seed"}
            or type(self.seed_policy.get("seed")) is not int
            or self.seed_policy["seed"] < 0
        ):
            raise CandidatePlanningError("seed policy is invalid")
        if not self.coverage_cell_ids or len(set(self.coverage_cell_ids)) != len(
            self.coverage_cell_ids
        ):
            raise CandidatePlanningError("candidate coverage cells must be unique")
        expected_cells = tuple(
            canonical_sha256(specification)
            for specification in self.coverage_cell_specs
        )
        if self.coverage_cell_ids != expected_cells:
            raise CandidatePlanningError("candidate coverage-cell specifications drifted")
        coverage_keys = {
            "operation",
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer",
            "architecture_context",
        }
        registered_operations = _registered_operation_ids()
        dtype = {
            "fp32_ieee": "float32",
            "fp16_grad_scaler": "float16",
            "mixed_structured": "mixed_structured",
        }.get(policy_id)
        accumulation = (
            "float32"
            if dtype in {"float16", "mixed_structured"}
            else dtype
        )
        for specification in self.coverage_cell_specs:
            if set(specification) != coverage_keys or any(
                type(value) is not str or not value for value in specification.values()
            ):
                raise CandidatePlanningError("candidate coverage-cell schema is invalid")
            operation = specification["operation"]
            if operation not in registered_operations and not operation.startswith(
                "structural:"
            ):
                raise CandidatePlanningError(
                    "candidate coverage cell references an unknown operation"
                )
            if (
                specification["shape_regime"]
                not in {"tiny", "small", "medium", "large", "boundary"}
                or specification["dtype"] != dtype
                or specification["accumulation_dtype"] != accumulation
                or specification["phase"] not in {"forward", "backward"}
                or specification["backend"] != self.execution["backend_id"]
                or specification["layout"] not in {"contiguous", "non_contiguous"}
                or specification["optimizer"] != self.optimizer["name"]
                or specification["architecture_context"]
                not in {
                    "sequential",
                    "residual_or_branch",
                    "saved_activation_or_alias",
                    "hand_authored_family",
                }
            ):
                raise CandidatePlanningError(
                    "candidate coverage cell differs from its configuration"
                )
        try:
            request = CompatibilityRequest(
                family_id=self.family_id,
                modality=self.source_modality,
                architecture_parameters=self.architecture_parameters,
                input_signature=self.input_signature,
                precision_id=str(self.precision_policy["policy_id"]),
                optimizer_id=str(self.optimizer["name"]),
                scheduler_id=str(self.scheduler["name"]),
                execution_mode=str(self.execution["mode"]),
                backend_id=str(self.execution["backend_id"]),
            )
        except KeyError as error:
            raise CandidatePlanningError(
                "candidate is missing compatibility-defining configuration"
            ) from error
        decision = evaluate_compatibility(request)
        if request.sha256 != self.compatibility_request_sha256:
            raise CandidatePlanningError("candidate compatibility fingerprint drifted")
        if not decision.compatible:
            raise CandidatePlanningError(
                f"candidate violates compatibility policy: {decision.reason_codes}"
            )


def target_candidate_from_dict(value: Mapping[str, Any]) -> TargetCandidate:
    """Load one exact candidate for a fresh local/NRP worker process."""

    if not isinstance(value, Mapping) or set(value) != set(TargetCandidate.__dataclass_fields__):
        raise CandidatePlanningError("serialized candidate schema differs")
    batch_raw = value.get("batch_plan")
    if not isinstance(batch_raw, Mapping) or set(batch_raw) != set(BatchPlan.__dataclass_fields__):
        raise CandidatePlanningError("serialized batch plan schema differs")
    batch_plan = BatchPlan(
        **{
            **dict(batch_raw),
            "full_ladder": tuple(batch_raw["full_ladder"]),
            "eligible_ladder": tuple(batch_raw["eligible_ladder"]),
            "promotion_reasons": tuple(batch_raw["promotion_reasons"]),
        }
    )
    candidate = TargetCandidate(
        **{
            **dict(value),
            "batch_plan": batch_plan,
            "coverage_cell_specs": tuple(value["coverage_cell_specs"]),
            "coverage_cell_ids": tuple(value["coverage_cell_ids"]),
        }
    )
    candidate.validate()
    return candidate


@dataclass(frozen=True)
class TargetManifest:
    version: str
    planner_version: str
    target_hardware_id: str
    quota_plan_sha256: str
    model_registry_sha256: str
    task_registry_sha256: str
    generated_lineage_registry_sha256: str
    configuration_binding_state: str
    candidates: tuple[TargetCandidate, ...]
    training_approved: bool

    def _hash_payload(self) -> Mapping[str, Any]:
        """Compact commitment; every candidate ID already binds its full payload."""

        return {
            "version": self.version,
            "planner_version": self.planner_version,
            "target_hardware_id": self.target_hardware_id,
            "quota_plan_sha256": self.quota_plan_sha256,
            "model_registry_sha256": self.model_registry_sha256,
            "task_registry_sha256": self.task_registry_sha256,
            "generated_lineage_registry_sha256": self.generated_lineage_registry_sha256,
            "configuration_binding_state": self.configuration_binding_state,
            "candidate_ids": tuple(row.candidate_id for row in self.candidates),
            "training_approved": self.training_approved,
        }

    def _sha256_unchecked(self) -> str:
        return canonical_sha256(self._hash_payload())

    @property
    def sha256(self) -> str:
        self.validate()
        return self._sha256_unchecked()

    def to_summary(self) -> dict[str, Any]:
        self.validate()
        return self._summary_unchecked()

    def _summary_unchecked(self) -> dict[str, Any]:
        family_counts = Counter(row.family_id for row in self.candidates)
        modality_counts = Counter(row.quota_modality for row in self.candidates)
        regime_counts = Counter(row.regime for row in self.candidates)
        precision_counts = Counter(
            row.precision_policy["policy_id"] for row in self.candidates
        )
        batch_tier_counts = Counter(row.batch_plan.effective_tier for row in self.candidates)
        generated = [
            row for row in self.candidates if row.family_id == "independent_generated"
        ]
        lineage_counts = Counter(row.source_lineage for row in generated)
        return canonical_value(
            {
                "version": self.version,
                "manifest_sha256": self._sha256_unchecked(),
                "candidate_count": len(self.candidates),
                "family_counts": dict(sorted(family_counts.items())),
                "modality_counts": dict(sorted(modality_counts.items())),
                "regime_counts": dict(sorted(regime_counts.items())),
                "precision_counts": dict(sorted(precision_counts.items())),
                "batch_tier_counts": dict(sorted(batch_tier_counts.items())),
                "optimizer_ids": sorted({row.optimizer["name"] for row in self.candidates}),
                "scheduler_ids": sorted({row.scheduler["name"] for row in self.candidates}),
                "generated_lineage_count": len(lineage_counts),
                "generated_lineage_maximum": max(lineage_counts.values()),
                "generated_held_out_lineage_count": len(
                    {
                        row.source_lineage
                        for row in generated
                        if row.source_split == "held_out"
                    }
                ),
                "generated_lineage_registry_sha256": self.generated_lineage_registry_sha256,
                "configuration_binding_state": self.configuration_binding_state,
                "training_approved": self.training_approved,
            }
        )

    def validate(
        self,
        *,
        quota: QuotaPlan | None = None,
        models: ModelRegistry | None = None,
        tasks: TaskRegistry | None = None,
    ) -> None:
        quota = quota or load_quota_plan()
        tasks = tasks or load_task_registry()
        models = models or load_model_registry(quota=quota, tasks=tasks)
        generated_registry = build_generated_lineage_registry()
        if self.version != TARGET_MANIFEST_VERSION or self.planner_version != PLANNER_VERSION:
            raise CandidatePlanningError("target manifest version mismatch")
        if self.target_hardware_id != quota.target_hardware_id:
            raise CandidatePlanningError("target manifest hardware mismatch")
        if (
            self.quota_plan_sha256 != quota.sha256
            or self.model_registry_sha256 != models.sha256
            or self.task_registry_sha256 != tasks.sha256
            or self.generated_lineage_registry_sha256 != generated_registry.sha256
        ):
            raise CandidatePlanningError("target manifest registry binding mismatch")
        if self.configuration_binding_state != "task_materialization_required":
            raise CandidatePlanningError("target manifest must await task materialization")
        if self.training_approved is not False:
            raise CandidatePlanningError("target manifest cannot approve training")
        if len(self.candidates) != 18_000:
            raise CandidatePlanningError("target manifest must contain exactly 18,000 candidates")
        if len({row.candidate_id for row in self.candidates}) != len(self.candidates):
            raise CandidatePlanningError("target manifest candidate IDs are not unique")
        if tuple(row.ordinal for row in self.candidates) != tuple(range(18_000)):
            raise CandidatePlanningError("target manifest ordinals are not contiguous")
        models_by_family = {row.family_id: row for row in models.entries}
        tasks_by_id = {row.task_id: row for row in tasks.entries}
        task_kinds = {
            row.task_id: TaskAdapter(row).task_kind for row in tasks.entries
        }
        quota_by_family = {row.family_id: row for row in quota.cells}
        generated_by_id = {
            row.lineage_id: row for row in generated_registry.lineages
        }
        for row in self.candidates:
            row.validate()
            model = models_by_family.get(row.family_id)
            if model is None or row.factory_id != model.factory_id:
                raise CandidatePlanningError("candidate factory binding is invalid")
            task = tasks_by_id.get(row.task_id)
            quota_cell = quota_by_family.get(row.family_id)
            if (
                task is None
                or quota_cell is None
                or row.quota_modality != quota_cell.modality
                or row.task_id not in model.adapter_ids
                or row.source_modality != task.modality
                or row.task_schema_sha256 != canonical_sha256(task.target_schema)
                or row.target_width != task.target_schema.get("target_width")
                or row.dataset_revision != task.dataset_revision
                or row.training_step_id
                != _training_step(row.family_id, task_kinds[row.task_id])
            ):
                raise CandidatePlanningError(
                    "candidate quota/task/dataset/training-step binding is invalid"
                )
            if row.family_id == "independent_generated":
                lineage = generated_by_id.get(row.source_lineage)
                if lineage is None or (
                    row.source_sha256,
                    row.source_split,
                    row.source_modality,
                    row.architecture_parameters,
                ) != (
                    generated_candidate_source_sha256(
                        lineage.source_sha256,
                        model.source_sha256,
                    ),
                    lineage.source_split,
                    lineage.modality,
                    {
                        "source_lineage": lineage.lineage_id,
                        "architecture_specification": lineage.architecture_specification,
                        "modality": lineage.modality,
                        "depth": lineage.depth,
                        "width": lineage.width,
                    },
                ):
                    raise CandidatePlanningError(
                        "generated candidate differs from its immutable source lineage"
                    )
            elif (
                row.source_lineage != model.source_lineage
                or row.source_sha256 != model.source_sha256
                or row.source_modality != model.modality
            ):
                raise CandidatePlanningError("candidate source binding is invalid")
        family_counts = Counter(row.family_id for row in self.candidates)
        expected_family = {
            cell.family_id: cell.accepted_configurations for cell in quota.cells
        }
        if dict(family_counts) != expected_family:
            raise CandidatePlanningError("target manifest family quotas differ from frozen plan")
        modality_counts = Counter(row.quota_modality for row in self.candidates)
        if dict(modality_counts) != dict(quota.expected_modality_totals):
            raise CandidatePlanningError("target manifest modality quotas differ from frozen plan")
        generated = [
            row for row in self.candidates if row.family_id == "independent_generated"
        ]
        lineages = Counter(row.source_lineage for row in generated)
        if len(lineages) < quota.generated_lineage_minimum:
            raise CandidatePlanningError("generated lineage minimum is not met")
        if max(lineages.values()) > quota.generated_lineage_maximum_accepted:
            raise CandidatePlanningError("generated lineage maximum is exceeded")
        held_out = {
            row.source_lineage for row in generated if row.source_split == "held_out"
        }
        if len(held_out) < quota.generated_lineages_held_out_minimum:
            raise CandidatePlanningError("generated held-out lineage minimum is not met")
        if any(
            {row.source_split for row in generated if row.source_lineage == lineage}
            != ({"held_out"} if lineage in held_out else {"development"})
            for lineage in lineages
        ):
            raise CandidatePlanningError("generated lineage crosses source splits")
        if set(row.optimizer["name"] for row in self.candidates) != set(
            DEPLOYMENT_OPTIMIZERS
        ):
            raise CandidatePlanningError("optimizer identity coverage is incomplete")
        optimizer_families = {
            optimizer: {
                row.family_id
                for row in self.candidates
                if row.optimizer["name"] == optimizer
            }
            for optimizer in DEPLOYMENT_OPTIMIZERS
        }
        if any(len(families) < 2 for families in optimizer_families.values()):
            raise CandidatePlanningError(
                "every optimizer identity requires two compatible families"
            )
        if set(row.scheduler["name"] for row in self.candidates) != set(
            DEPLOYMENT_SCHEDULERS
        ):
            raise CandidatePlanningError("scheduler identity coverage is incomplete")


def _regime(local_ordinal: int, quota: int) -> str:
    realistic = int(quota * 0.70)
    extrapolation = int(quota * 0.10)
    if local_ordinal < realistic:
        return "task_realistic"
    if local_ordinal < realistic + extrapolation:
        return "shape_extrapolation"
    return "memory_frontier"


def _architecture_parameters(
    family_id: str,
    local_ordinal: int,
    regime: str,
    *,
    defaults: Mapping[str, Any] | None = None,
    generated_lineage: GeneratedLineageSpec | None = None,
) -> dict[str, Any]:
    values = deepcopy(
        dict(defaults)
        if defaults is not None
        else resolve_architecture_parameters(family_id, None)
    )
    regime_offset = REGIMES.index(regime)
    for field in tuple(values):
        domain = _FIELD_DOMAINS.get(field)
        if domain is not None:
            values[field] = domain[(local_ordinal + regime_offset) % len(domain)]
    if "heads" in values and "hidden_size" in values:
        heads = int(values["heads"])
        hidden = int(values["hidden_size"])
        values["hidden_size"] = max(heads, ((hidden + heads - 1) // heads) * heads)
    if "heads" in values and "width" in values:
        heads = int(values["heads"])
        width = int(values["width"])
        values["width"] = max(heads, ((width + heads - 1) // heads) * heads)
    if "heads" in values and "embedding_dim" in values:
        heads = int(values["heads"])
        width = int(values["embedding_dim"])
        values["embedding_dim"] = max(heads, ((width + heads - 1) // heads) * heads)
    if "kv_heads" in values and "heads" in values:
        allowed = [item for item in (1, 2, 4) if int(values["heads"]) % item == 0]
        values["kv_heads"] = allowed[local_ordinal % len(allowed)]
    if "window_size" in values and "input_resolution" in values:
        patch_grid = int(values["input_resolution"]) // int(
            values.get("patch_size", 4)
        )
        allowed_windows = [
            item
            for item in _FIELD_DOMAINS["window_size"]
            if patch_grid >= item and patch_grid % item == 0
        ]
        values["window_size"] = allowed_windows[local_ordinal % len(allowed_windows)]
    if "top_k" in values and "expert_count" in values:
        values["top_k"] = min(int(values["top_k"]), int(values["expert_count"]))
    if family_id == "independent_generated":
        if generated_lineage is None:
            raise CandidatePlanningError("generated family requires a source lineage")
        values.update(
            {
                "source_lineage": generated_lineage.lineage_id,
                "architecture_specification": generated_lineage.architecture_specification,
                "modality": generated_lineage.modality,
                "depth": generated_lineage.depth,
                "width": generated_lineage.width,
            }
        )
    return canonical_value(values)


def _input_signature(
    modality: str,
    parameters: Mapping[str, Any],
    local_ordinal: int,
    regime: str,
    *,
    target_width: int,
) -> dict[str, Any]:
    scale = (1, 2, 4)[REGIMES.index(regime)]
    if modality == "vision":
        resolution = int(parameters.get("input_resolution", 16))
        return {
            "resolution": resolution,
            "height": resolution,
            "width": resolution + (local_ordinal % 3) * scale,
            "channels": 3,
            "class_count": target_width,
            "crop_size": resolution,
        }
    if modality == "nlp":
        length = int(parameters.get("sequence_length", 8 * scale))
        return {
            "sequence_length": length,
            "decoder_length": max(2, length - local_ordinal % 3),
            "vocabulary_size": 64 * scale,
        }
    if modality == "audio":
        return {
            "sample_rate": int(parameters.get("sample_rate", 16_000)),
            "clip_samples": int(parameters.get("clip_samples", 256 * scale)),
            "mel_bins": int(parameters.get("mel_bins", 16)),
            "hop_size": 8 * scale,
        }
    if modality == "tabular":
        return {
            "feature_count": int(parameters.get("feature_count", 8)),
            "categorical_fields": 3,
            "cardinality": 8 * scale,
        }
    if modality == "graph":
        nodes = int(parameters.get("node_count", 8 * scale))
        return {
            "node_count": nodes,
            "edge_count": nodes * (2 + local_ordinal % 4),
            "node_feature_width": int(parameters.get("atom_features", 6)),
            "edge_feature_width": 3,
            "graph_batch_size": 1,
        }
    raise CandidatePlanningError(f"unsupported source modality {modality!r}")


def _top_quartile(field: str, value: Any) -> bool:
    domain = _FIELD_DOMAINS.get(field)
    if domain is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = sorted(float(item) for item in domain if isinstance(item, (int, float)))
    threshold = numeric[max(0, math.ceil(0.75 * len(numeric)) - 1)]
    return float(value) >= threshold


def _batch_classification(
    family_id: str,
    architecture_parameters: Mapping[str, Any],
    input_signature: Mapping[str, Any],
    precision_id: str,
    checkpoint_enabled: bool,
    optimizer_id: str,
) -> tuple[str, str, tuple[str, ...]]:
    if family_id in LIGHT_FAMILIES:
        base_tier = "light"
    elif family_id in HEAVY_FAMILIES or (
        family_id == "efficientnet_b0_b4"
        and architecture_parameters.get("variant") == "b4"
    ):
        base_tier = "heavy"
    elif family_id == "independent_generated":
        topology = str(architecture_parameters.get("architecture_specification", "")).split(":", 1)[0]
        modality = str(architecture_parameters.get("modality", ""))
        generated_high_footprint = (
            int(architecture_parameters.get("depth", 3)) >= 7
            or int(architecture_parameters.get("width", 24)) >= 32
        )
        if generated_high_footprint:
            base_tier = "heavy"
        elif (
            topology in {"sequential", "sparse_embedding"}
            and modality in {"nlp", "audio", "tabular"}
            and int(architecture_parameters.get("depth", 3)) <= 3
            and int(architecture_parameters.get("width", 24)) <= 16
        ):
            base_tier = "light"
        else:
            base_tier = "standard"
    else:
        base_tier = "standard"

    axes = {
        "image_resolution": _top_quartile(
            "input_resolution",
            architecture_parameters.get("input_resolution", input_signature.get("resolution")),
        ),
        "sequence_length": _top_quartile(
            "sequence_length",
            architecture_parameters.get("sequence_length", input_signature.get("sequence_length")),
        ),
        "audio_length": _top_quartile(
            "clip_samples",
            architecture_parameters.get("clip_samples", input_signature.get("clip_samples")),
        ),
        "model_width": any(
            _top_quartile(name, architecture_parameters.get(name))
            for name in ("width", "hidden_size", "embedding_dim", "channel_dim")
        ),
        "model_depth": any(
            _top_quartile(name, architecture_parameters.get(name))
            for name in ("depth", "layers", "layer_count", "encoder_layers", "decoder_layers")
        ),
        "graph_nodes": _top_quartile(
            "node_count",
            architecture_parameters.get("node_count", input_signature.get("node_count")),
        ),
        "graph_edges": _top_quartile(
            "edge_count",
            input_signature.get("edge_count"),
        ),
        "attention_heads": _top_quartile("heads", architecture_parameters.get("heads")),
        "moe_experts": _top_quartile("expert_count", architecture_parameters.get("expert_count")),
    }
    top_axes = tuple(sorted(name for name, active in axes.items() if active))
    quadratic_axis = (
        family_id in _DENSE_ATTENTION_FAMILIES and axes["sequence_length"]
    ) or (family_id in {"vit_s16", "swin_t"} and axes["image_resolution"])
    top_model_axis = axes["model_width"] or axes["model_depth"]
    top_shape_axis = any(
        axes[name]
        for name in (
            "image_resolution",
            "sequence_length",
            "audio_length",
            "graph_nodes",
            "graph_edges",
        )
    )
    reasons: list[str] = []
    if quadratic_axis:
        reasons.append("top_quartile_quadratic_axis")
    if len(top_axes) >= 2:
        reasons.append("multiple_top_quartile_activation_axes")
    if precision_id == "fp32_ieee" and top_shape_axis:
        reasons.append("fp32_with_top_quartile_shape")
    if not checkpoint_enabled and len(top_axes) >= 2:
        reasons.append("checkpoint_disabled_with_multiple_top_axes")
    if optimizer_id == "lbfgs" and top_model_axis:
        reasons.append("lbfgs_with_top_quartile_model")
    should_promote = bool(reasons)
    if base_tier == "heavy":
        effective_tier = "heavy"
    elif base_tier == "standard" and should_promote:
        effective_tier = "heavy"
    elif base_tier == "light" and should_promote:
        effective_tier = "standard"
    else:
        effective_tier = base_tier
    return base_tier, effective_tier, tuple(reasons)


def _batch_plan(
    family_id: str,
    architecture_parameters: Mapping[str, Any],
    input_signature: Mapping[str, Any],
    selection_ordinal: int,
    precision_id: str,
    checkpoint_enabled: bool,
    optimizer_id: str,
    expected_train_examples: int,
) -> BatchPlan:
    base_tier, effective_tier, reasons = _batch_classification(
        family_id,
        architecture_parameters,
        input_signature,
        precision_id,
        checkpoint_enabled,
        optimizer_id,
    )
    ladder = BATCH_LADDERS[effective_tier]
    cap = expected_train_examples // 8
    eligible = tuple(value for value in ladder if value <= cap)
    if not eligible:
        raise CandidatePlanningError("prepared epoch view cannot supply eight batches")
    result = BatchPlan(
        version=BATCH_PLAN_VERSION,
        base_tier=base_tier,
        effective_tier=effective_tier,
        full_ladder=ladder,
        eligible_ladder=eligible,
        expected_train_examples=expected_train_examples,
        maximum_batch_for_eight_batches=cap,
        promotion_reasons=reasons,
        selected_microbatch=eligible[selection_ordinal % len(eligible)],
    )
    result.validate()
    return result


def _training_step(family_id: str, task_kind: str) -> str:
    specialized = {
        "pix2pix": "pix2pix_joint_generator_discriminator_v1",
        "distilbert_distillation": "teacher_student_distillation_v1",
        "bilstm_crf": "masked_bilstm_crf_v1",
        "switch_moe": "switch_moe_capacity_routing_v1",
        "cgcnn": "cgcnn_pyg_crystal_v1",
        "panns_cnn14": "waveform_stft_logmel_v1",
        "m5_waveform_cnn": "raw_waveform_v1",
        "mixture_density_network": "mixture_density_nll_v1",
    }
    return specialized.get(family_id, f"{task_kind}_v1")


def _coverage_specs(
    regime: str,
    assertions: tuple[str, ...],
    precision: str,
    backend: str,
    optimizer: str,
    architecture_context: str,
    ordinal: int,
) -> tuple[Mapping[str, Any], ...]:
    dtype = {
        "fp32_ieee": "float32",
        "fp16_grad_scaler": "float16",
        "mixed_structured": "mixed_structured",
    }[precision]
    accumulation = "float32" if dtype in {"float16", "mixed_structured"} else dtype
    shape_regime = {
        "task_realistic": ("small", "medium")[ordinal % 2],
        "shape_extrapolation": "large",
        "memory_frontier": "boundary",
    }[regime]
    return tuple(
        canonical_value(
            {
                "operation": operation,
                "shape_regime": shape_regime,
                "dtype": dtype,
                "accumulation_dtype": accumulation,
                "phase": ("forward", "backward")[(ordinal + index) % 2],
                "backend": backend,
                "layout": ("contiguous", "non_contiguous")[(ordinal + index) % 2],
                "optimizer": optimizer,
                "architecture_context": architecture_context,
            }
        )
        for index, operation in enumerate(assertions)
    )


@lru_cache(maxsize=1)
def build_target_manifest() -> TargetManifest:
    quota = load_quota_plan()
    tasks = load_task_registry()
    models = load_model_registry(quota=quota, tasks=tasks)
    generated_registry = build_generated_lineage_registry()
    task_by_id = {row.task_id: row for row in tasks.entries}
    task_kind_by_id = {
        task_id: adapter_for_task(task_id).task_kind for task_id in task_by_id
    }
    model_by_family = {row.family_id: row for row in models.entries}
    architecture_defaults = {
        family_id: resolve_architecture_parameters(family_id, None)
        for family_id in model_by_family
    }
    candidates: list[TargetCandidate] = []
    batch_selection_counters: Counter[tuple[str, str]] = Counter()
    global_ordinal = 0
    for cell in quota.cells:
        model = model_by_family[cell.family_id]
        for local_ordinal in range(cell.accepted_configurations):
            regime = _regime(local_ordinal, cell.accepted_configurations)
            lineage_index = (
                local_ordinal % quota.generated_lineage_minimum
                if cell.family_id == "independent_generated"
                else None
            )
            generated_lineage = (
                generated_registry.lineages[lineage_index]
                if lineage_index is not None
                else None
            )
            architecture = _architecture_parameters(
                cell.family_id,
                local_ordinal,
                regime,
                defaults=architecture_defaults[cell.family_id],
                generated_lineage=generated_lineage,
            )
            source_modality = (
                str(architecture["modality"])
                if cell.family_id == "independent_generated"
                else cell.modality
            )
            compatible_tasks = [
                task_id
                for task_id in model.adapter_ids
                if task_by_id[task_id].modality == source_modality
            ]
            if not compatible_tasks:
                raise CandidatePlanningError("family has no task for sampled source modality")
            task_id = compatible_tasks[local_ordinal % len(compatible_tasks)]
            task = task_by_id[task_id]
            signature = _input_signature(
                source_modality,
                architecture,
                local_ordinal,
                regime,
                target_width=int(task.target_schema["target_width"]),
            )
            precision = PRECISION_PATTERN[global_ordinal % len(PRECISION_PATTERN)]
            optimizer = DEPLOYMENT_OPTIMIZERS[
                global_ordinal % len(DEPLOYMENT_OPTIMIZERS)
            ]
            if cell.family_id == "fasttext_embeddingbag" and local_ordinal == 0:
                optimizer = "sparse_adam"
            sparse_generated = (
                cell.family_id == "independent_generated"
                and source_modality == "nlp"
                and "sparse" in str(architecture["architecture_specification"])
            )
            if sparse_generated:
                optimizer = "sparse_adam"
            if optimizer == "sparse_adam" and not (
                cell.family_id == "fasttext_embeddingbag" or sparse_generated
            ):
                optimizer = "adamw"
            if optimizer == "muon" and cell.family_id in MUON_MATRIX_FREE_FAMILIES:
                optimizer = "adamw"
            if cell.family_id == "fasttext_embeddingbag":
                architecture = dict(architecture)
                architecture["sparse_gradients"] = optimizer == "sparse_adam"
            if optimizer == "lbfgs":
                precision = "fp32_ieee"
            if cell.family_id == "switch_moe" and precision == "fp16_grad_scaler":
                precision = "fp32_ieee"
            checkpoint_enabled = global_ordinal % 5 == 0
            scheduler = DEPLOYMENT_SCHEDULERS[
                global_ordinal % len(DEPLOYMENT_SCHEDULERS)
            ]
            execution_mode = (
                "compiled"
                if global_ordinal % 3 == 0
                and cell.family_id
                not in {"pix2pix", "bilstm_crf", "gru_rnn_seq2seq", "cgcnn"}
                else "eager"
            )
            backend = "inductor_cuda" if execution_mode == "compiled" else "cuda_eager"
            request = CompatibilityRequest(
                family_id=cell.family_id,
                modality=source_modality,
                architecture_parameters=architecture,
                input_signature=signature,
                precision_id=precision,
                optimizer_id=optimizer,
                scheduler_id=scheduler,
                execution_mode=execution_mode,
                backend_id=backend,
            )
            decision = evaluate_compatibility(request)
            if not decision.compatible:
                raise CandidatePlanningError(
                    f"planner emitted incompatible {cell.family_id}: {decision.reason_codes}"
                )
            if lineage_index is None:
                source_lineage = model.source_lineage
                source_sha256 = model.source_sha256
                source_split = "development"
                generator_version = PLANNER_VERSION
                coverage_assertions = model.faithful_operation_assertions
            else:
                assert generated_lineage is not None
                source_lineage = generated_lineage.lineage_id
                source_sha256 = generated_candidate_source_sha256(
                    generated_lineage.source_sha256,
                    model.source_sha256,
                )
                source_split = generated_lineage.source_split
                generator_version = GENERATED_LINEAGE_VERSION
                coverage_assertions = generated_lineage.expected_operation_ids
            mutation = {
                "local_ordinal": local_ordinal,
                "regime": regime,
                "architecture_parameters": architecture,
            }
            provisional_batch_plan = _batch_plan(
                cell.family_id,
                architecture,
                signature,
                local_ordinal,
                precision,
                checkpoint_enabled,
                optimizer,
                task.expected_train_examples,
            )
            batch_counter_key = (
                cell.family_id,
                provisional_batch_plan.effective_tier,
            )
            batch_plan = _batch_plan(
                cell.family_id,
                architecture,
                signature,
                batch_selection_counters[batch_counter_key],
                precision,
                checkpoint_enabled,
                optimizer,
                task.expected_train_examples,
            )
            batch_selection_counters[batch_counter_key] += 1
            architecture_context = (
                {
                    "sequential": "sequential",
                    "residual": "residual_or_branch",
                    "branching": "residual_or_branch",
                    "sparse_embedding": "saved_activation_or_alias",
                }[generated_lineage.topology]
                if generated_lineage is not None
                else "hand_authored_family"
            )
            coverage_specs = _coverage_specs(
                regime,
                coverage_assertions,
                precision,
                backend,
                optimizer,
                architecture_context,
                global_ordinal,
            )
            payload = {
                "version": CANDIDATE_VERSION,
                "ordinal": global_ordinal,
                "quota_modality": cell.modality,
                "source_modality": source_modality,
                "family_id": cell.family_id,
                "regime": regime,
                "source_lineage": source_lineage,
                "source_sha256": source_sha256,
                "source_split": source_split,
                "factory_id": model.factory_id,
                "generator_version": generator_version,
                "mutation_specification": mutation,
                "architecture_parameters": architecture,
                "task_id": task_id,
                "task_schema_sha256": canonical_sha256(task.target_schema),
                "target_width": int(task.target_schema["target_width"]),
                "dataset_revision": task.dataset_revision,
                "input_signature": signature,
                "training_step_id": _training_step(
                    cell.family_id, task_kind_by_id[task_id]
                ),
                "batch_plan": batch_plan,
                "microbatch_size": batch_plan.selected_microbatch,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION[
                    global_ordinal % len(GRADIENT_ACCUMULATION)
                ],
                "precision_policy": {
                    "policy_id": precision,
                    "autocast": precision != "fp32_ieee",
                    "gradient_scaler": precision == "fp16_grad_scaler",
                },
                "optimizer": {
                    "name": optimizer,
                    "learning_rate": (1e-4, 3e-4, 1e-3)[global_ordinal % 3],
                    "weight_decay": (0.0, 0.01)[global_ordinal % 2],
                },
                "scheduler": {
                    "name": scheduler,
                    "step_unit": scheduler_step_unit(scheduler),
                    "minimum_lr_ratio": scheduler_minimum_lr_ratio(scheduler),
                    "progress": SCHEDULE_PROGRESS[
                        global_ordinal % len(SCHEDULE_PROGRESS)
                    ],
                },
                "activation_checkpointing": {
                    "enabled": checkpoint_enabled
                },
                "execution": {"mode": execution_mode, "backend_id": backend},
                "seed_policy": {
                    "seed": int(canonical_sha256({"ordinal": global_ordinal})[:8], 16)
                },
                "coverage_cell_specs": coverage_specs,
                "coverage_cell_ids": tuple(
                    canonical_sha256(specification)
                    for specification in coverage_specs
                ),
                "compatibility_request_sha256": request.sha256,
                "observation_protocol_sha256": quota.protocol.sha256,
                "configuration_binding_state": "task_materialization_required",
                "target_hardware_id": quota.target_hardware_id,
                "training_approved": False,
            }
            candidate = TargetCandidate(
                candidate_id=canonical_sha256(payload), **payload
            )
            candidate.validate()
            candidates.append(candidate)
            global_ordinal += 1
    manifest = TargetManifest(
        version=TARGET_MANIFEST_VERSION,
        planner_version=PLANNER_VERSION,
        target_hardware_id=quota.target_hardware_id,
        quota_plan_sha256=quota.sha256,
        model_registry_sha256=models.sha256,
        task_registry_sha256=tasks.sha256,
        generated_lineage_registry_sha256=generated_registry.sha256,
        configuration_binding_state="task_materialization_required",
        candidates=tuple(candidates),
        training_approved=False,
    )
    manifest.validate(quota=quota, models=models, tasks=tasks)
    return manifest


def select_generated_gap_wave(
    candidates: tuple[TargetCandidate, ...],
    observed_coverage_cell_ids: set[str] | tuple[str, ...],
    *,
    limit: int,
    error_by_coverage_cell: Mapping[str, float] | None = None,
) -> tuple[TargetCandidate, ...]:
    """Select a deterministic, lineage-diverse wave aimed at measured gaps."""

    if type(limit) is not int or limit < 1:
        raise CandidatePlanningError("generated gap-wave limit must be positive")
    if len({row.candidate_id for row in candidates}) != len(candidates):
        raise CandidatePlanningError("generated gap-wave candidates contain duplicate IDs")
    observed = set(observed_coverage_cell_ids)
    errors = dict(error_by_coverage_cell or {})
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in errors.values()
    ):
        raise CandidatePlanningError(
            "generated coverage errors must be finite nonnegative numbers"
        )
    generated = [
        row for row in candidates if row.family_id == "independent_generated"
    ]
    if not generated:
        raise CandidatePlanningError("generated gap wave has no generated candidates")

    def priority(row: TargetCandidate) -> tuple[float, int, str]:
        missing = sum(cell not in observed for cell in row.coverage_cell_ids)
        error = max((float(errors.get(cell, 0.0)) for cell in row.coverage_cell_ids), default=0.0)
        return (-error, -missing, row.candidate_id)

    ordered = sorted(generated, key=priority)
    selected: list[TargetCandidate] = []
    selected_ids: set[str] = set()
    seen_lineages: set[str] = set()
    for row in ordered:
        if row.source_lineage in seen_lineages:
            continue
        selected.append(row)
        selected_ids.add(row.candidate_id)
        seen_lineages.add(row.source_lineage)
        if len(selected) == min(limit, len(generated)):
            return tuple(selected)
    for row in ordered:
        if row.candidate_id not in selected_ids:
            selected.append(row)
            if len(selected) == min(limit, len(generated)):
                break
    return tuple(selected)


__all__ = [
    "BATCH_PLAN_VERSION",
    "BATCH_LADDERS",
    "BatchPlan",
    "CANDIDATE_VERSION",
    "CandidatePlanningError",
    "PLANNER_VERSION",
    "REGIMES",
    "TARGET_MANIFEST_VERSION",
    "TargetCandidate",
    "TargetManifest",
    "MINIMUM_TERMINAL_LEARNING_RATE_RATIO",
    "TERMINAL_DECAY_SCHEDULERS",
    "build_target_manifest",
    "generated_candidate_source_sha256",
    "scheduler_minimum_lr_ratio",
    "select_generated_gap_wave",
    "target_candidate_from_dict",
    "scheduler_step_unit",
]
