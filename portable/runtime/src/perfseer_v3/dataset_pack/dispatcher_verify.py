"""Fail-closed requested-versus-observed dispatcher and backend verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from perfseer_v3.op_registry import OperationRegistry

from .fingerprints import canonical_sha256
from .operation_sampler import OperationGeneratorRegistry, build_operation_generator_registry


DISPATCH_REQUEST_VERSION = "perfseer_v3_v100_dispatch_request_v1"
DISPATCH_EVIDENCE_VERSION = "perfseer_v3_v100_dispatch_evidence_v1"
IDENTITY_SOURCES = {
    "dispatcher_trace",
    "capture_semantic_summary",
    "perfseer_phase_annotation",
}
MEASUREMENT_SCOPES = {"local_smoke", "v100_measurement"}
TARGET_HARDWARE_ID = "nvidia_tesla_v100_sxm2_32gb_nrp"
CAPTURE_SEMANTIC_TARGETS = {
    "prim.getitem": ("prim::TupleIndex",),
}


class DispatchVerificationError(ValueError):
    """Raised when a requested operation or backend was substituted or unverified."""


def _text(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise DispatchVerificationError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, *, context: str) -> str:
    text = _text(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DispatchVerificationError(f"{context} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class DispatchRequest:
    version: str
    canonical_operation_id: str
    raw_target: str
    aliases: tuple[str, ...]
    generator_registry_sha256: str
    measurement_scope: str
    requested_backend_id: str
    identity_source: str
    environment_gated: bool
    specialized_backend: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(
        self,
        registry: OperationRegistry | None = None,
        generators: OperationGeneratorRegistry | None = None,
    ) -> None:
        registry = registry or OperationRegistry.load()
        generators = generators or build_operation_generator_registry(registry=registry)
        generators.validate(registry=registry)
        if self.version != DISPATCH_REQUEST_VERSION:
            raise DispatchVerificationError("dispatch request version mismatch")
        for name in ("canonical_operation_id", "raw_target", "requested_backend_id"):
            _text(getattr(self, name), context=name)
        if self.identity_source not in IDENTITY_SOURCES:
            raise DispatchVerificationError("unknown dispatch identity source")
        if self.measurement_scope not in MEASUREMENT_SCOPES:
            raise DispatchVerificationError("unknown dispatch measurement scope")
        _sha256(self.generator_registry_sha256, context="generator_registry_sha256")
        if self.generator_registry_sha256 != generators.sha256:
            raise DispatchVerificationError("dispatch request generator registry is stale")
        if type(self.environment_gated) is not bool or type(self.specialized_backend) is not bool:
            raise DispatchVerificationError("dispatch request flags must be booleans")
        if type(self.aliases) is not tuple or any(
            type(value) is not str or not value for value in self.aliases
        ):
            raise DispatchVerificationError("dispatch request aliases must be a tuple")
        resolved = registry.resolve(self.raw_target)
        if not resolved.is_known or resolved.canonical_id != self.canonical_operation_id:
            raise DispatchVerificationError("dispatch request is not bound to the operation registry")
        generator = next(
            (
                row
                for row in generators.generators
                if row.canonical_operation_id == self.canonical_operation_id
            ),
            None,
        )
        if generator is None:
            raise DispatchVerificationError("dispatch request has no required generator")
        if (self.raw_target, self.aliases) != (generator.raw_target, generator.aliases):
            raise DispatchVerificationError("dispatch request identity differs from its generator")
        expected_source = {
            "optimizer_step_annotation": "perfseer_phase_annotation",
            "training_graph_annotation": "perfseer_phase_annotation",
            "capture_observed_python_semantics": "capture_semantic_summary",
        }.get(generator.execution_route, "dispatcher_trace")
        expected_gated = generator.execution_route == "v100_environment_gated_dispatcher"
        if self.identity_source != expected_source:
            raise DispatchVerificationError("dispatch identity source differs from generator route")
        if self.environment_gated != expected_gated or self.specialized_backend != expected_gated:
            raise DispatchVerificationError("dispatch gating flags differ from generator route")
        if self.measurement_scope == "local_smoke":
            if self.requested_backend_id != "cpu_eager":
                raise DispatchVerificationError("local smoke must request cpu_eager")
        elif self.requested_backend_id not in generator.required_backends:
            raise DispatchVerificationError("V100 backend differs from generator requirements")


@dataclass(frozen=True)
class DispatchEvidence:
    version: str
    request_sha256: str
    target_hardware_id: str
    measurement_occurrence_id: str
    accepted_measurement: bool
    supported: bool
    unsupported_reason: str | None
    capture_workload_sha256: str
    profile_workload_sha256: str
    identity_source: str
    observed_raw_targets: tuple[str, ...]
    observed_backend_ids: tuple[str, ...]

    @property
    def sha256(self) -> str:
        self.validate_shape()
        return canonical_sha256(asdict(self))

    def validate_shape(self) -> None:
        if self.version != DISPATCH_EVIDENCE_VERSION:
            raise DispatchVerificationError("dispatch evidence version mismatch")
        _sha256(self.request_sha256, context="request_sha256")
        _sha256(self.capture_workload_sha256, context="capture_workload_sha256")
        _sha256(self.profile_workload_sha256, context="profile_workload_sha256")
        _text(self.target_hardware_id, context="target_hardware_id")
        _text(self.measurement_occurrence_id, context="measurement_occurrence_id")
        if type(self.accepted_measurement) is not bool or type(self.supported) is not bool:
            raise DispatchVerificationError("dispatch evidence flags must be booleans")
        if self.identity_source not in IDENTITY_SOURCES:
            raise DispatchVerificationError("unknown evidence identity source")
        if type(self.observed_raw_targets) is not tuple:
            raise DispatchVerificationError("observed raw targets must be a tuple")
        if type(self.observed_backend_ids) is not tuple:
            raise DispatchVerificationError("observed backend IDs must be a tuple")
        if any(type(value) is not str or not value for value in self.observed_raw_targets):
            raise DispatchVerificationError("observed raw targets must be non-empty strings")
        if any(type(value) is not str or not value for value in self.observed_backend_ids):
            raise DispatchVerificationError("observed backend IDs must be non-empty strings")
        if self.supported:
            if self.unsupported_reason is not None:
                raise DispatchVerificationError("supported evidence cannot retain an unsupported reason")
            if not self.observed_raw_targets or not self.observed_backend_ids:
                raise DispatchVerificationError("supported evidence requires observed identities")
        elif type(self.unsupported_reason) is not str or not self.unsupported_reason:
            raise DispatchVerificationError("unsupported evidence requires a written reason")
        if self.accepted_measurement and self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise DispatchVerificationError("accepted dispatcher evidence must come from NRP V100")


@dataclass(frozen=True)
class DispatchVerification:
    request_sha256: str
    evidence_sha256: str
    measurement_scope: str
    canonical_operation_id: str
    raw_target: str
    requested_backend_id: str
    target_hardware_id: str
    status: str
    matched_raw_targets: tuple[str, ...]
    matched_backend_ids: tuple[str, ...]
    accepted_measurement: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        _sha256(self.request_sha256, context="request_sha256")
        _sha256(self.evidence_sha256, context="evidence_sha256")
        for name in (
            "canonical_operation_id",
            "raw_target",
            "requested_backend_id",
            "target_hardware_id",
        ):
            _text(getattr(self, name), context=name)
        if self.measurement_scope not in MEASUREMENT_SCOPES:
            raise DispatchVerificationError("unknown verification measurement scope")
        if self.status not in {"verified", "environment_unsupported"}:
            raise DispatchVerificationError("unknown dispatch verification status")
        if type(self.matched_raw_targets) is not tuple or type(self.matched_backend_ids) is not tuple:
            raise DispatchVerificationError("matched dispatch identities must be tuples")
        if any(type(value) is not str or not value for value in self.matched_raw_targets):
            raise DispatchVerificationError("matched raw targets must be non-empty strings")
        if any(type(value) is not str or not value for value in self.matched_backend_ids):
            raise DispatchVerificationError("matched backend IDs must be non-empty strings")
        if type(self.accepted_measurement) is not bool:
            raise DispatchVerificationError("verification accepted flag must be boolean")
        if self.status == "verified":
            if not self.matched_raw_targets or self.matched_backend_ids != (
                self.requested_backend_id,
            ):
                raise DispatchVerificationError("verified dispatch must retain exact matches")
            if self.measurement_scope == "v100_measurement":
                if not self.accepted_measurement or self.target_hardware_id != TARGET_HARDWARE_ID:
                    raise DispatchVerificationError("V100 verification requires accepted V100 evidence")
            elif self.accepted_measurement:
                raise DispatchVerificationError("local smoke cannot be accepted measurement evidence")
        elif self.matched_raw_targets or self.matched_backend_ids or self.accepted_measurement:
            raise DispatchVerificationError("unsupported verification cannot retain matches or acceptance")


def verify_dispatch(
    request: DispatchRequest,
    evidence: DispatchEvidence,
    *,
    registry: OperationRegistry | None = None,
    generators: OperationGeneratorRegistry | None = None,
) -> DispatchVerification:
    registry = registry or OperationRegistry.load()
    generators = generators or build_operation_generator_registry(registry=registry)
    request.validate(registry, generators)
    request_sha256 = canonical_sha256(asdict(request))
    evidence.validate_shape()
    if evidence.request_sha256 != request_sha256:
        raise DispatchVerificationError("dispatch evidence targets a different request")
    if evidence.capture_workload_sha256 != evidence.profile_workload_sha256:
        raise DispatchVerificationError("capture/profile workload fingerprints do not match")
    if evidence.identity_source != request.identity_source:
        raise DispatchVerificationError("requested and observed identity sources differ")
    if request.measurement_scope == "local_smoke":
        if evidence.accepted_measurement or evidence.target_hardware_id == TARGET_HARDWARE_ID:
            raise DispatchVerificationError("local smoke evidence cannot claim NRP V100 acceptance")
    elif evidence.target_hardware_id != TARGET_HARDWARE_ID:
        raise DispatchVerificationError("V100 request requires NRP V100 evidence")
    if not evidence.supported:
        if not request.environment_gated:
            raise DispatchVerificationError("non-gated operation cannot be accepted as unsupported")
        if evidence.accepted_measurement:
            raise DispatchVerificationError("unsupported evidence cannot be an accepted measurement")
        result = DispatchVerification(
            request_sha256=request_sha256,
            evidence_sha256=evidence.sha256,
            measurement_scope=request.measurement_scope,
            canonical_operation_id=request.canonical_operation_id,
            raw_target=request.raw_target,
            requested_backend_id=request.requested_backend_id,
            target_hardware_id=evidence.target_hardware_id,
            status="environment_unsupported",
            matched_raw_targets=(),
            matched_backend_ids=(),
            accepted_measurement=False,
        )
        result.validate()
        return result
    if request.measurement_scope == "v100_measurement" and not evidence.accepted_measurement:
        raise DispatchVerificationError("supported V100 evidence must be accepted")
    semantic_targets = CAPTURE_SEMANTIC_TARGETS.get(request.canonical_operation_id, ())
    matched_raw = tuple(
        sorted(
            target
            for target in set(evidence.observed_raw_targets)
            if registry.resolve(target).canonical_id == request.canonical_operation_id
            or (
                request.identity_source == "capture_semantic_summary"
                and target in semantic_targets
            )
        )
    )
    if not matched_raw:
        raise DispatchVerificationError(
            f"requested operation {request.canonical_operation_id} was not observed"
        )
    if evidence.observed_backend_ids != (request.requested_backend_id,):
        raise DispatchVerificationError(
            f"requested backend {request.requested_backend_id!r} was substituted or ambiguous"
        )
    result = DispatchVerification(
        request_sha256=request_sha256,
        evidence_sha256=evidence.sha256,
        measurement_scope=request.measurement_scope,
        canonical_operation_id=request.canonical_operation_id,
        raw_target=request.raw_target,
        requested_backend_id=request.requested_backend_id,
        target_hardware_id=evidence.target_hardware_id,
        status="verified",
        matched_raw_targets=matched_raw,
        matched_backend_ids=evidence.observed_backend_ids,
        accepted_measurement=evidence.accepted_measurement,
    )
    result.validate()
    return result


__all__ = [
    "DISPATCH_EVIDENCE_VERSION",
    "DISPATCH_REQUEST_VERSION",
    "CAPTURE_SEMANTIC_TARGETS",
    "MEASUREMENT_SCOPES",
    "DispatchEvidence",
    "DispatchRequest",
    "DispatchVerification",
    "DispatchVerificationError",
    "verify_dispatch",
]
