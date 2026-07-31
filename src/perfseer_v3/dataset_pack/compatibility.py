"""Declarative fail-closed compatibility checks for A10G target candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from perfseer_v3.training_semantics import OPTIMIZERS, SCHEDULERS

from .fingerprints import canonical_sha256, canonical_value
from .models.architecture import (
    ArchitectureBindingError,
    declared_architecture_fields,
)
from .quota import FROZEN_QUOTA_CELLS


COMPATIBILITY_VERSION = "perfseer_v3_a10g_compatibility_v3"
SUPPORTED_PRECISIONS = (
    "fp32_tf32",
    "bf16",
    "fp16_grad_scaler",
    "mixed_structured",
)
DEPLOYMENT_OPTIMIZERS = tuple(
    value for value in OPTIMIZERS if value not in {"none", "other"}
)
DEPLOYMENT_SCHEDULERS = tuple(
    value for value in SCHEDULERS if value not in {"other"}
)
EXECUTION_MODES = ("eager", "compiled")
MUON_MATRIX_FREE_FAMILIES = frozenset({"unet_groupnorm", "pix2pix"})
FAMILY_MODALITIES = {
    family_id: modality for modality, family_id, _, _ in FROZEN_QUOTA_CELLS
}
REASON_CODES = (
    "family_not_registered",
    "family_modality_mismatch",
    "precision_not_supported_on_a10g",
    "fp16_moe_routing_unstable",
    "head_hidden_not_divisible",
    "optimizer_not_pinned",
    "optimizer_parameter_contract_invalid",
    "scheduler_not_pinned",
    "compile_path_unsupported",
    "patch_does_not_tile_resolution",
    "window_does_not_tile_patch_grid",
    "sequence_length_invalid",
    "graph_batch_invalid",
    "architecture_safety_limit_exceeded",
    "backend_identity_unresolved",
    "distributed_only_not_allowed",
    "architecture_value_invalid",
    "architecture_field_contract_invalid",
    "input_signature_invalid",
)


class CompatibilityError(ValueError):
    """Raised when the compatibility policy or decision is malformed."""


@dataclass(frozen=True)
class CompatibilityRequest:
    family_id: str
    modality: str
    architecture_parameters: Mapping[str, Any]
    input_signature: Mapping[str, Any]
    precision_id: str
    optimizer_id: str
    scheduler_id: str
    execution_mode: str
    backend_id: str
    single_process: bool = True

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class CompatibilityDecision:
    version: str
    request_sha256: str
    compatible: bool
    reason_codes: tuple[str, ...]

    def validate(self) -> None:
        if self.version != COMPATIBILITY_VERSION:
            raise CompatibilityError("compatibility decision version mismatch")
        if len(self.request_sha256) != 64:
            raise CompatibilityError("compatibility request fingerprint must be SHA-256")
        if self.compatible != (not self.reason_codes):
            raise CompatibilityError("compatibility boolean differs from reason codes")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise CompatibilityError("compatibility reason codes must be sorted and unique")
        if any(code not in REASON_CODES for code in self.reason_codes):
            raise CompatibilityError("compatibility decision has an unknown reason code")


def _integer(parameters: Mapping[str, Any], name: str) -> int | None:
    value = parameters.get(name)
    return value if type(value) is int else None


def evaluate_compatibility(request: CompatibilityRequest) -> CompatibilityDecision:
    canonical_value(asdict(request))
    reasons: set[str] = set()
    parameters = request.architecture_parameters
    signature = request.input_signature
    registered_modality = FAMILY_MODALITIES.get(request.family_id)
    if registered_modality is None:
        reasons.add("family_not_registered")
    elif registered_modality not in {request.modality, "generated"}:
        reasons.add("family_modality_mismatch")
    else:
        try:
            declared_fields = set(declared_architecture_fields(request.family_id))
        except ArchitectureBindingError:
            reasons.add("architecture_field_contract_invalid")
        else:
            if set(parameters) != declared_fields:
                reasons.add("architecture_field_contract_invalid")
    if request.precision_id not in SUPPORTED_PRECISIONS:
        reasons.add("precision_not_supported_on_a10g")
    if request.family_id == "switch_moe" and request.precision_id == "fp16_grad_scaler":
        reasons.add("fp16_moe_routing_unstable")
    if request.optimizer_id not in DEPLOYMENT_OPTIMIZERS:
        reasons.add("optimizer_not_pinned")
    if request.scheduler_id not in DEPLOYMENT_SCHEDULERS:
        reasons.add("scheduler_not_pinned")
    if request.execution_mode not in EXECUTION_MODES:
        reasons.add("compile_path_unsupported")
    compile_blocked = {
        "pix2pix",
        "bilstm_crf",
        "gru_rnn_seq2seq",
        "cgcnn",
    }
    if request.execution_mode == "compiled" and request.family_id in compile_blocked:
        reasons.add("compile_path_unsupported")
    if request.backend_id != (
        "inductor_cuda" if request.execution_mode == "compiled" else "cuda_eager"
    ):
        reasons.add("backend_identity_unresolved")
    if not request.single_process:
        reasons.add("distributed_only_not_allowed")

    heads = _integer(parameters, "heads")
    hidden = next(
        (
            _integer(parameters, name)
            for name in ("hidden_size", "width", "embedding_dim", "channel_dim")
            if _integer(parameters, name) is not None
        ),
        None,
    )
    if heads is not None and hidden is not None and (heads < 1 or hidden % heads):
        reasons.add("head_hidden_not_divisible")
    kv_heads = _integer(parameters, "kv_heads")
    if heads is not None and kv_heads is not None and (
        kv_heads < 1 or heads % kv_heads
    ):
        reasons.add("head_hidden_not_divisible")

    resolution = _integer(parameters, "input_resolution") or signature.get(
        "resolution"
    )
    patch = _integer(parameters, "patch_size")
    if type(resolution) is int and type(patch) is int and (
        resolution < 1 or patch < 1 or resolution % patch
    ):
        reasons.add("patch_does_not_tile_resolution")
    window = _integer(parameters, "window_size")
    if type(resolution) is int and type(window) is int:
        patch_grid = resolution // max(1, patch or 4)
        if window < 1 or patch_grid < window or patch_grid % window:
            reasons.add("window_does_not_tile_patch_grid")
    sequence_length = _integer(parameters, "sequence_length") or signature.get(
        "sequence_length"
    )
    if sequence_length is not None and (
        type(sequence_length) is not int or not 2 <= sequence_length <= 4096
    ):
        reasons.add("sequence_length_invalid")
    if request.modality == "graph" and (
        type(signature.get("node_count")) is not int
        or signature["node_count"] < 1
        or type(signature.get("edge_count")) is not int
        or signature["edge_count"] < 1
        or type(signature.get("graph_batch_size")) is not int
        or signature["graph_batch_size"] < 1
    ):
        reasons.add("graph_batch_invalid")
    if (
        request.family_id == "gcn"
        and _integer(parameters, "node_count") != signature.get("node_count")
    ):
        reasons.add("input_signature_invalid")
    for value in parameters.values():
        numeric_values = value if isinstance(value, (list, tuple)) else (value,)
        if any(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and item <= 0
            for item in numeric_values
        ):
            reasons.add("architecture_value_invalid")
            break
    positive_signature_fields = {
        "resolution",
        "height",
        "width",
        "channels",
        "class_count",
        "crop_size",
        "sequence_length",
        "decoder_length",
        "vocabulary_size",
        "sample_rate",
        "clip_samples",
        "mel_bins",
        "hop_size",
        "feature_count",
        "cardinality",
        "node_count",
        "edge_count",
        "node_feature_width",
        "edge_feature_width",
        "graph_batch_size",
    }
    if any(
        field in signature
        and (
            isinstance(signature[field], bool)
            or not isinstance(signature[field], (int, float))
            or signature[field] <= 0
        )
        for field in positive_signature_fields
    ):
        reasons.add("input_signature_invalid")
    safety_values = {
        "input_resolution": 1024,
        "sequence_length": 4096,
        "hidden_size": 4096,
        "width": 1024,
        "node_count": 100_000,
        "clip_samples": 2_000_000,
        "expert_count": 128,
    }
    for field, maximum in safety_values.items():
        value = parameters.get(field, signature.get(field))
        if type(value) is int and value > maximum:
            reasons.add("architecture_safety_limit_exceeded")
    sparse_fasttext = request.family_id == "fasttext_embeddingbag" and bool(
        parameters.get("sparse_gradients", False)
    )
    sparse_generated = (
        request.family_id == "independent_generated"
        and request.modality == "nlp"
        and "sparse" in str(parameters.get("architecture_specification", ""))
    )
    sparse_capable = sparse_fasttext or sparse_generated
    if request.optimizer_id == "sparse_adam" and not sparse_capable:
        reasons.add("optimizer_parameter_contract_invalid")
    if sparse_capable and request.optimizer_id != "sparse_adam":
        reasons.add("optimizer_parameter_contract_invalid")
    if request.optimizer_id == "lbfgs" and request.precision_id != "fp32_tf32":
        reasons.add("optimizer_parameter_contract_invalid")
    if (
        request.optimizer_id == "muon"
        and request.family_id in MUON_MATRIX_FREE_FAMILIES
    ):
        reasons.add("optimizer_parameter_contract_invalid")

    decision = CompatibilityDecision(
        version=COMPATIBILITY_VERSION,
        request_sha256=request.sha256,
        compatible=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )
    decision.validate()
    return decision


__all__ = [
    "COMPATIBILITY_VERSION",
    "CompatibilityDecision",
    "CompatibilityError",
    "CompatibilityRequest",
    "DEPLOYMENT_OPTIMIZERS",
    "DEPLOYMENT_SCHEDULERS",
    "EXECUTION_MODES",
    "FAMILY_MODALITIES",
    "MUON_MATRIX_FREE_FAMILIES",
    "REASON_CODES",
    "SUPPORTED_PRECISIONS",
    "evaluate_compatibility",
]
