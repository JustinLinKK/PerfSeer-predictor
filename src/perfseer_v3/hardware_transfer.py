"""Few-shot NVIDIA transfer contracts, residual transforms, and freeze audits."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .baseline import canonical_json
from .hardware import require_specific_hardware_id
from .version import TRANSFER_MANIFEST_VERSION


TRANSFER_TARGET_NAMES: tuple[str, ...] = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_p95_sm_util_percent",
    "train_peak_vram_used_mib",
    "train_peak_torch_reserved_mib",
    "train_peak_memory_controller_util_percent",
)
TRANSFER_LABEL_BUDGETS = (128, 256, 512, 1024)
TRANSFER_SPLIT_COUNTS: dict[int, tuple[int, int, int]] = {
    128: (96, 16, 16),
    256: (192, 32, 32),
    512: (384, 64, 64),
    1024: (768, 128, 128),
}
TRANSFER_ADAPTER_POLICIES = (
    "linear_only",
    "film_only",
    "low_rank_only",
    "film_low_rank",
    "film_low_rank_heads",
    "film_low_rank_last_block",
)


@dataclass(frozen=True)
class BaseTransferLineageV3:
    """Exact frozen A10 inputs used to select and train a target subset."""

    base_training_manifest_sha256: str
    base_dataset_fingerprint: str
    base_split_fingerprint: str
    base_teacher_artifact_sha256: str
    base_student_artifact_sha256: str
    base_teacher_embeddings_sha256: str
    workload_normalization_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            _require_sha256(value, context=f"base transfer lineage {name}")
        if self.base_teacher_artifact_sha256 == self.base_student_artifact_sha256:
            raise ValueError("base teacher and student artifacts must be distinct")

    @property
    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["base_lineage_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaseTransferLineageV3":
        if not isinstance(value, Mapping):
            raise ValueError("base transfer lineage must be an object")
        raw = dict(value)
        declared_sha256 = str(raw.pop("base_lineage_sha256", ""))
        expected_fields = {field.name for field in cls.__dataclass_fields__.values()}
        if set(raw) != expected_fields:
            raise ValueError("base transfer lineage fields do not match the frozen contract")
        lineage = cls(**{name: str(raw[name]) for name in expected_fields})
        lineage.validate()
        if declared_sha256 != lineage.sha256:
            raise ValueError("base transfer lineage content hash mismatch")
        return lineage


def base_transfer_lineage_json_schema() -> dict[str, Any]:
    fields = (
        "base_training_manifest_sha256",
        "base_dataset_fingerprint",
        "base_split_fingerprint",
        "base_teacher_artifact_sha256",
        "base_student_artifact_sha256",
        "base_teacher_embeddings_sha256",
        "workload_normalization_sha256",
        "base_lineage_sha256",
    )
    return {
        "type": "object",
        "required": list(fields),
        "properties": {
            name: {"type": "string", "pattern": "^[0-9a-f]{64}$"}
            for name in fields
        },
        "additionalProperties": False,
    }


def _require_sha256(value: str | None, *, context: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class TransferConfigV3:
    base_hardware_id: str
    target_hardware_id: str
    label_budget: int
    adapter_policy: str
    adapter_rank: int
    freeze_workload_backbone: bool = True
    l2_sp_weight: float = 1e-4
    adapter_regularization_weight: float = 1e-4
    utilization_clip: float = 0.005
    epsilon: float = 1e-6

    def validate(self) -> None:
        base = require_specific_hardware_id(
            self.base_hardware_id, context="transfer base_hardware_id"
        )
        target = require_specific_hardware_id(
            self.target_hardware_id, context="transfer target_hardware_id"
        )
        if base == target:
            raise ValueError("target adaptation must use a GPU different from the base GPU")
        if self.label_budget not in TRANSFER_LABEL_BUDGETS:
            raise ValueError(
                f"target label budget must be one of {TRANSFER_LABEL_BUDGETS}; "
                f"got {self.label_budget}"
            )
        if self.adapter_policy not in TRANSFER_ADAPTER_POLICIES:
            raise ValueError(f"unsupported transfer adapter policy {self.adapter_policy!r}")
        if self.adapter_rank <= 0:
            raise ValueError("adapter rank must be positive")
        if not self.freeze_workload_backbone:
            raise ValueError("full-backbone target training is not an allowed default")
        if self.l2_sp_weight < 0 or self.adapter_regularization_weight < 0:
            raise ValueError("transfer regularization weights must be nonnegative")
        if not 0 < self.utilization_clip < 0.5 or self.epsilon <= 0:
            raise ValueError("invalid residual transform numerical bounds")

    @property
    def split_counts(self) -> tuple[int, int, int]:
        self.validate()
        return TRANSFER_SPLIT_COUNTS[self.label_budget]

    @property
    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


class PairedResidualTransformV3:
    """Output-specific invertible A10-to-target residual transforms."""

    positive_indices = (0, 3, 4)
    utilization_indices = (1, 2, 5)

    def __init__(self, *, epsilon: float = 1e-6, utilization_clip: float = 0.005) -> None:
        if epsilon <= 0 or not 0 < utilization_clip < 0.5:
            raise ValueError("invalid paired residual transform bounds")
        self.epsilon = float(epsilon)
        self.utilization_clip = float(utilization_clip)

    def _validate(self, value: torch.Tensor, *, context: str) -> None:
        if value.size(-1) != len(TRANSFER_TARGET_NAMES):
            raise ValueError(f"{context} must use the six-target PerfSeer order")
        if not torch.isfinite(value).all():
            raise ValueError(f"{context} must be finite")

    def encode(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._validate(base, context="base target tensor")
        self._validate(target, context="target tensor")
        if base.shape != target.shape:
            raise ValueError("paired base and target tensors must have identical shapes")
        residual = torch.empty_like(target)
        for index in self.positive_indices:
            residual[..., index] = torch.log(target[..., index].clamp_min(0) + self.epsilon) - torch.log(
                base[..., index].clamp_min(0) + self.epsilon
            )
        for index in self.utilization_indices:
            base_probability = (base[..., index] / 100.0).clamp(
                self.utilization_clip, 1.0 - self.utilization_clip
            )
            target_probability = (target[..., index] / 100.0).clamp(
                self.utilization_clip, 1.0 - self.utilization_clip
            )
            residual[..., index] = torch.logit(target_probability) - torch.logit(
                base_probability
            )
        return residual

    def decode(self, base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self._validate(base, context="base target tensor")
        self._validate(residual, context="paired residual tensor")
        if base.shape != residual.shape:
            raise ValueError("base and residual tensors must have identical shapes")
        decoded = torch.empty_like(residual)
        for index in self.positive_indices:
            scale = base[..., index].clamp_min(0) + self.epsilon
            decoded[..., index] = (
                scale * torch.exp(residual[..., index]) - self.epsilon
            ).clamp_min(0)
            decoded[..., index] = torch.where(
                residual[..., index] == 0,
                base[..., index],
                decoded[..., index],
            )
        for index in self.utilization_indices:
            base_probability = (base[..., index] / 100.0).clamp(
                self.utilization_clip, 1.0 - self.utilization_clip
            )
            decoded[..., index] = 100.0 * torch.sigmoid(
                torch.logit(base_probability) + residual[..., index]
            )
            decoded[..., index] = torch.where(
                residual[..., index] == 0,
                base[..., index],
                decoded[..., index],
            )
        return decoded

    @property
    def policy_sha256(self) -> str:
        payload = {
            "target_names": TRANSFER_TARGET_NAMES,
            "positive_indices": self.positive_indices,
            "utilization_indices": self.utilization_indices,
            "positive_transform": "log_ratio",
            "utilization_transform": "logit_residual_percent",
            "epsilon": self.epsilon,
            "utilization_clip": self.utilization_clip,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransferLineageV3:
    base_artifact_sha256: str
    base_hardware_id: str
    target_hardware_id: str
    target_subset_sha256: str
    hardware_profile_sha256: str
    workload_normalization_sha256: str
    hardware_normalization_sha256: str
    calibration_sha256: str
    paired_residual_policy_sha256: str
    adapter_policy: str
    adapter_rank: int
    trainable_parameter_count: int

    def validate(self) -> None:
        for name in (
            "base_artifact_sha256",
            "target_subset_sha256",
            "hardware_profile_sha256",
            "workload_normalization_sha256",
            "hardware_normalization_sha256",
            "calibration_sha256",
            "paired_residual_policy_sha256",
        ):
            _require_sha256(getattr(self, name), context=f"transfer lineage {name}")
        base = require_specific_hardware_id(
            self.base_hardware_id, context="transfer lineage base_hardware_id"
        )
        target = require_specific_hardware_id(
            self.target_hardware_id, context="transfer lineage target_hardware_id"
        )
        if base == target:
            raise ValueError("adapted artifact base and target GPUs must differ")
        if self.adapter_policy not in TRANSFER_ADAPTER_POLICIES:
            raise ValueError("transfer lineage adapter policy is unsupported")
        if self.adapter_rank <= 0 or self.trainable_parameter_count <= 0:
            raise ValueError("transfer lineage parameter counts/rank must be positive")

    @property
    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


def configure_transfer_trainable_parameters(model: Any, policy: str) -> dict[str, Any]:
    audit = model.configure_trainable_parameters(policy)
    if not audit["trainable_names"]:
        raise ValueError("transfer freeze policy selected no parameters")
    forbidden_prefixes = ("node_encoder.", "edge_encoder.", "global_encoder.")
    if any(name.startswith(forbidden_prefixes) for name in audit["trainable_names"]):
        raise ValueError("transfer freeze policy unexpectedly unfreezes the workload backbone")
    return audit


def parameter_checksum(model: Any, names: Sequence[str]) -> str:
    wanted = set(names)
    digest = hashlib.sha256()
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        if name not in wanted:
            continue
        seen.add(name)
        digest.update(name.encode("utf-8"))
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        # A byte view works for bfloat16 and other dtypes that NumPy cannot
        # represent on every supported runtime.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    if seen != wanted:
        raise ValueError("parameter checksum requested unknown names")
    return digest.hexdigest()


def adapter_regularization_loss(
    model: Any,
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if reference is not None and name in reference:
            terms.append((parameter - reference[name].to(parameter.device)).square().mean())
        else:
            terms.append(parameter.square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


def adapter_identity_regularization_loss(model: Any) -> torch.Tensor:
    """Penalize only parameters whose zero/one state defines identity transfer."""

    zero_prefixes = (
        "hardware_adapter.film.",
        "hardware_adapter.up.",
        "target_residual_head.",
    )
    terms: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(zero_prefixes):
            terms.append(parameter.square().mean())
        elif name == "hardware_adapter.scale":
            terms.append((parameter - 1.0).square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


def transfer_manifest_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TRANSFER_MANIFEST_VERSION,
        "type": "object",
        "required": [
            "manifest_version",
            "base_lineage",
            "base_hardware_id",
            "target_hardware_id",
            "label_budget",
            "split_counts",
            "selection_method",
            "test_selection_uses_model_errors",
            "selection",
            "memory_boundary_probes",
            "subset_sha256",
        ],
        "properties": {
            "manifest_version": {"const": TRANSFER_MANIFEST_VERSION},
            "base_lineage": base_transfer_lineage_json_schema(),
            "base_hardware_id": {"type": "string", "minLength": 1},
            "target_hardware_id": {"type": "string", "minLength": 1},
            "label_budget": {"enum": list(TRANSFER_LABEL_BUDGETS)},
            "split_counts": {
                "type": "object",
                "required": ["train", "validation", "test"],
                "properties": {
                    name: {"type": "integer", "minimum": 1}
                    for name in ("train", "validation", "test")
                },
                "additionalProperties": False,
            },
            "selection_method": {"type": "string", "minLength": 1},
            "test_selection_uses_model_errors": {"const": False},
            "embedding_dimension": {"type": "integer", "minimum": 1},
            "selection": {
                "type": "array",
                "minItems": 128,
                "maxItems": 1024,
                "items": {
                    "type": "object",
                    "required": [
                        "configuration_id",
                        "split",
                        "source_group",
                        "graph_signature",
                        "selection_reason",
                        "graph_path",
                    ],
                    "properties": {
                        "configuration_id": {"type": "string", "minLength": 1},
                        "split": {"enum": ["train", "validation", "test"]},
                        "source_group": {"type": "string", "minLength": 1},
                        "graph_signature": {"type": "string", "minLength": 1},
                        "selection_reason": {"type": "string", "minLength": 1},
                        "graph_path": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
            },
            "memory_boundary_probes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "configuration_id",
                        "grouped_split",
                        "paired_batch_first",
                        "batch_ladder",
                        "retain_oom_and_repair",
                    ],
                    "properties": {
                        "configuration_id": {"type": "string", "minLength": 1},
                        "grouped_split": {
                            "enum": ["train", "validation", "test"]
                        },
                        "paired_batch_first": {"const": True},
                        "batch_ladder": {"type": "string", "minLength": 1},
                        "retain_oom_and_repair": {"const": True},
                    },
                    "additionalProperties": False,
                },
            },
            "subset_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": True,
    }


__all__ = [
    "BaseTransferLineageV3",
    "PairedResidualTransformV3",
    "TRANSFER_ADAPTER_POLICIES",
    "TRANSFER_LABEL_BUDGETS",
    "TRANSFER_SPLIT_COUNTS",
    "TRANSFER_TARGET_NAMES",
    "TransferConfigV3",
    "TransferLineageV3",
    "adapter_identity_regularization_loss",
    "adapter_regularization_loss",
    "base_transfer_lineage_json_schema",
    "configure_transfer_trainable_parameters",
    "parameter_checksum",
    "transfer_manifest_json_schema",
]
