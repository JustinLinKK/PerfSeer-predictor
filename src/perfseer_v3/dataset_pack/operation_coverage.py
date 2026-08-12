"""Measured V100 operation coverage, vocabulary, and runtime-allowlist gates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

from .coverage_config import OperationCoverageConfig, load_operation_coverage_config
from .dispatcher_verify import (
    DispatchEvidence,
    DispatchRequest,
    DispatchVerification,
    verify_dispatch,
)
from .fingerprints import canonical_sha256, canonical_value
from .operation_sampler import OperationGeneratorRegistry, build_operation_generator_registry
from .operation_support import OperationSupportContract, build_operation_support_contract


COVERAGE_OBSERVATION_VERSION = "perfseer_v3_v100_operation_coverage_observation_v1"
COVERAGE_REPORT_VERSION = "perfseer_v3_v100_operation_coverage_report_v1"
STRUCTURAL_EVIDENCE_VERSION = "perfseer_v3_v100_structural_coverage_evidence_v1"
MEASUREMENT_EVIDENCE_VERSION = "perfseer_v3_v100_coverage_measurement_evidence_v1"
STRUCTURAL_GENERATOR_ID = "structural:family_hash_custom"
RUNTIME_ALLOWLIST_VERSION = "perfseer_v3_v100_runtime_allowlist_v1"
TARGET_HARDWARE_ID = "nvidia_tesla_v100_sxm2_32gb_nrp"


class OperationCoverageError(ValueError):
    """Raised when coverage or allowlist evidence is incomplete or non-V100."""


def _text(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise OperationCoverageError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, *, context: str) -> str:
    text = _text(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise OperationCoverageError(f"{context} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class StructuralCoverageEvidence:
    version: str
    evidence_id: str
    raw_target: str
    target_hardware_id: str
    accepted_measurement: bool
    capture_workload_sha256: str
    profile_workload_sha256: str
    observed_backend_id: str
    structural_path: str

    def unhashed_payload(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload.pop("evidence_id")
        return payload

    def validate(self) -> None:
        if self.version != STRUCTURAL_EVIDENCE_VERSION:
            raise OperationCoverageError("structural evidence version mismatch")
        if self.evidence_id != canonical_sha256(self.unhashed_payload()):
            raise OperationCoverageError("structural evidence ID mismatch")
        _text(self.raw_target, context="structural raw target")
        _sha256(self.capture_workload_sha256, context="capture_workload_sha256")
        _sha256(self.profile_workload_sha256, context="profile_workload_sha256")
        if self.capture_workload_sha256 != self.profile_workload_sha256:
            raise OperationCoverageError("structural capture/profile fingerprints do not match")
        if self.target_hardware_id != TARGET_HARDWARE_ID:
            raise OperationCoverageError("structural timing evidence must come from NRP V100")
        if type(self.accepted_measurement) is not bool or not self.accepted_measurement:
            raise OperationCoverageError("structural evidence must be an accepted measurement")
        if self.observed_backend_id != "cuda_eager":
            raise OperationCoverageError("structural evidence must observe cuda_eager")
        if self.structural_path != "family_hash_custom":
            raise OperationCoverageError("unknown/custom evidence must use family_hash_custom")


@dataclass(frozen=True)
class CoverageMeasurementEvidence:
    version: str
    measurement_id: str
    measurement_occurrence_id: str
    source_evidence_sha256: str
    configuration_sha256: str
    capture_artifact_sha256: str
    profiler_artifact_sha256: str
    artifact_record_index: int
    artifact_record_sha256: str
    artifact_manifest_record_sha256s: tuple[str, ...]
    artifact_manifest_sha256: str
    canonical_operation_id: str
    coverage_cell_id: str
    generator_id: str
    family: str
    shape_regime: str
    dtype: str
    accumulation_dtype: str
    phase: str
    backend: str
    layout: str
    optimizer_context: str
    architecture_context: str
    target_hardware_id: str
    accepted: bool
    complete_capture: bool
    strict_capture: bool
    tensor_producing_nodes: int
    structurally_encoded_nodes: int
    silently_dropped_tensor_nodes: int
    capture_profile_fingerprint_match: bool
    gpu_time_us: float

    def unhashed_payload(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload.pop("measurement_id")
        return payload

    def configuration_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "canonical_operation_id",
                "coverage_cell_id",
                "generator_id",
                "family",
                "shape_regime",
                "dtype",
                "accumulation_dtype",
                "phase",
                "backend",
                "layout",
                "optimizer_context",
                "architecture_context",
                "target_hardware_id",
            )
        }

    def observation_binding(self) -> dict[str, object]:
        result = self.configuration_payload()
        for name in (
            "accepted",
            "complete_capture",
            "strict_capture",
            "tensor_producing_nodes",
            "structurally_encoded_nodes",
            "silently_dropped_tensor_nodes",
            "capture_profile_fingerprint_match",
            "gpu_time_us",
        ):
            result[name] = getattr(self, name)
        return result

    def validate(self) -> None:
        if self.version != MEASUREMENT_EVIDENCE_VERSION:
            raise OperationCoverageError("measurement evidence version mismatch")
        if self.measurement_id != canonical_sha256(self.unhashed_payload()):
            raise OperationCoverageError("measurement evidence ID mismatch")
        for name in (
            "measurement_occurrence_id",
            "canonical_operation_id",
            "generator_id",
            "family",
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer_context",
            "architecture_context",
            "target_hardware_id",
        ):
            _text(getattr(self, name), context=f"measurement {name}")
        for name in (
            "source_evidence_sha256",
            "configuration_sha256",
            "capture_artifact_sha256",
            "profiler_artifact_sha256",
            "artifact_record_sha256",
            "artifact_manifest_sha256",
            "coverage_cell_id",
        ):
            _sha256(getattr(self, name), context=f"measurement {name}")
        if self.configuration_sha256 != canonical_sha256(self.configuration_payload()):
            raise OperationCoverageError("measurement configuration fingerprint mismatch")
        if type(self.artifact_record_index) is not int or not (
            0 <= self.artifact_record_index < len(self.artifact_manifest_record_sha256s)
        ):
            raise OperationCoverageError("measurement artifact record index is invalid")
        if type(self.artifact_manifest_record_sha256s) is not tuple or any(
            _sha256(value, context="artifact manifest record") != value
            for value in self.artifact_manifest_record_sha256s
        ):
            raise OperationCoverageError("artifact manifest records must be a SHA-256 tuple")
        expected_record_sha256 = canonical_sha256(self.observation_binding())
        if (
            self.artifact_record_sha256 != expected_record_sha256
            or self.profiler_artifact_sha256 != expected_record_sha256
        ):
            raise OperationCoverageError(
                "measurement differs from its content-addressed profiler record"
            )
        if (
            self.artifact_record_index != 0
            or self.artifact_manifest_record_sha256s != (expected_record_sha256,)
        ):
            raise OperationCoverageError(
                "each observation requires one immutable profiler artifact record"
            )
        expected_manifest_sha256 = canonical_sha256(
            {
                "capture_artifact_sha256": self.capture_artifact_sha256,
                "profiler_artifact_sha256": self.profiler_artifact_sha256,
                "record_sha256s": self.artifact_manifest_record_sha256s,
            }
        )
        if self.artifact_manifest_sha256 != expected_manifest_sha256:
            raise OperationCoverageError("measurement artifact manifest fingerprint mismatch")
        expected_occurrence_id = canonical_sha256(
            {
                "artifact_manifest_sha256": self.artifact_manifest_sha256,
                "artifact_record_index": self.artifact_record_index,
            }
        )
        if self.measurement_occurrence_id != expected_occurrence_id:
            raise OperationCoverageError("measurement occurrence is not artifact-derived")
        for name in (
            "accepted",
            "complete_capture",
            "strict_capture",
            "capture_profile_fingerprint_match",
        ):
            if type(getattr(self, name)) is not bool:
                raise OperationCoverageError(f"measurement {name} must be boolean")
        for name in (
            "tensor_producing_nodes",
            "structurally_encoded_nodes",
            "silently_dropped_tensor_nodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise OperationCoverageError(f"measurement {name} must be nonnegative")
        if isinstance(self.gpu_time_us, bool) or not isinstance(self.gpu_time_us, (int, float)):
            raise OperationCoverageError("measurement gpu time must be numeric")
        if not math.isfinite(float(self.gpu_time_us)) or self.gpu_time_us < 0:
            raise OperationCoverageError("measurement gpu time must be finite and nonnegative")


@dataclass(frozen=True)
class CoverageObservation:
    version: str
    observation_id: str
    canonical_operation_id: str
    coverage_cell_id: str
    generator_id: str
    family: str
    shape_regime: str
    dtype: str
    accumulation_dtype: str
    phase: str
    backend: str
    layout: str
    optimizer_context: str
    architecture_context: str
    target_hardware_id: str
    accepted: bool
    measurement_evidence: CoverageMeasurementEvidence
    dispatch_request: DispatchRequest | None
    dispatch_evidence: DispatchEvidence | None
    dispatch_verification: DispatchVerification | None
    structural_evidence: StructuralCoverageEvidence | None
    complete_capture: bool
    strict_capture: bool
    tensor_producing_nodes: int
    structurally_encoded_nodes: int
    silently_dropped_tensor_nodes: int
    capture_profile_fingerprint_match: bool
    gpu_time_us: float

    def unhashed_payload(self) -> dict[str, object]:
        payload = canonical_value(asdict(self))
        payload.pop("observation_id")
        return payload

    def validate(self, generators: OperationGeneratorRegistry | None = None) -> None:
        if self.version != COVERAGE_OBSERVATION_VERSION:
            raise OperationCoverageError("coverage observation version mismatch")
        if self.observation_id != canonical_sha256(self.unhashed_payload()):
            raise OperationCoverageError("coverage observation ID mismatch")
        for name in (
            "canonical_operation_id",
            "coverage_cell_id",
            "generator_id",
            "family",
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer_context",
            "architecture_context",
            "target_hardware_id",
        ):
            _text(getattr(self, name), context=name)
        _sha256(self.coverage_cell_id, context="coverage_cell_id")
        for name in (
            "accepted",
            "complete_capture",
            "strict_capture",
            "capture_profile_fingerprint_match",
        ):
            if type(getattr(self, name)) is not bool:
                raise OperationCoverageError(f"{name} must be boolean")
        for name in (
            "tensor_producing_nodes",
            "structurally_encoded_nodes",
            "silently_dropped_tensor_nodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise OperationCoverageError(f"{name} must be a nonnegative integer")
        if self.structurally_encoded_nodes > self.tensor_producing_nodes:
            raise OperationCoverageError("encoded node count exceeds tensor-producing nodes")
        if self.silently_dropped_tensor_nodes > self.tensor_producing_nodes:
            raise OperationCoverageError("dropped node count exceeds tensor-producing nodes")
        if self.strict_capture and not self.complete_capture:
            raise OperationCoverageError("strict capture requires complete capture")
        if isinstance(self.gpu_time_us, bool) or not isinstance(self.gpu_time_us, (int, float)):
            raise OperationCoverageError("gpu time must be numeric")
        if not math.isfinite(float(self.gpu_time_us)) or self.gpu_time_us < 0:
            raise OperationCoverageError("gpu time must be finite and nonnegative")

        dispatch_values = (
            self.dispatch_request,
            self.dispatch_evidence,
            self.dispatch_verification,
        )
        if not isinstance(self.measurement_evidence, CoverageMeasurementEvidence):
            raise OperationCoverageError("coverage requires typed measurement evidence")
        self.measurement_evidence.validate()
        if self.structural_evidence is None:
            if not all(dispatch_values) or not isinstance(self.dispatch_request, DispatchRequest) or not isinstance(self.dispatch_evidence, DispatchEvidence) or not isinstance(self.dispatch_verification, DispatchVerification):
                raise OperationCoverageError("registered coverage requires complete dispatcher evidence")
            verified = verify_dispatch(
                self.dispatch_request,
                self.dispatch_evidence,
                generators=generators,
            )
            if verified != self.dispatch_verification:
                raise OperationCoverageError("stored dispatcher verification was not reproduced")
            if verified.status != "verified":
                raise OperationCoverageError("coverage requires a verified dispatcher result")
            if (
                self.canonical_operation_id != verified.canonical_operation_id
                or self.backend != verified.requested_backend_id
                or self.target_hardware_id != verified.target_hardware_id
            ):
                raise OperationCoverageError("coverage dimensions differ from dispatcher evidence")
            if self.accepted and not verified.accepted_measurement:
                raise OperationCoverageError("accepted coverage requires accepted dispatcher evidence")
            source_evidence_sha256 = self.dispatch_evidence.sha256
        else:
            if any(value is not None for value in dispatch_values):
                raise OperationCoverageError("structural coverage cannot also claim dispatcher evidence")
            self.structural_evidence.validate()
            if (
                self.canonical_operation_id != "UNK"
                or self.generator_id != STRUCTURAL_GENERATOR_ID
                or self.family != "unknown_or_custom"
                or self.backend != self.structural_evidence.observed_backend_id
                or self.target_hardware_id != self.structural_evidence.target_hardware_id
            ):
                raise OperationCoverageError("unknown/custom coverage identity is not structural")
            source_evidence_sha256 = self.structural_evidence.evidence_id

        if self.measurement_evidence.source_evidence_sha256 != source_evidence_sha256:
            raise OperationCoverageError("measurement evidence targets a different source record")
        observation_binding = {
            name: getattr(self, name)
            for name in self.measurement_evidence.observation_binding()
        }
        if self.measurement_evidence.observation_binding() != observation_binding:
            raise OperationCoverageError(
                "coverage dimensions, node counts, or timing differ from measurement evidence"
            )

        if self.accepted:
            if self.target_hardware_id != TARGET_HARDWARE_ID:
                raise OperationCoverageError("accepted coverage must be measured on NRP V100")
            if not self.capture_profile_fingerprint_match:
                raise OperationCoverageError("accepted capture/profile fingerprints must match")
            if not self.complete_capture or self.gpu_time_us <= 0:
                raise OperationCoverageError("accepted coverage requires complete capture and positive time")
            if self.silently_dropped_tensor_nodes != 0:
                raise OperationCoverageError("accepted coverage cannot silently drop tensor nodes")
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
            raise OperationCoverageError("coverage-cell ID does not match its dimensions")


def make_coverage_observation(
    *,
    dispatch_request: DispatchRequest,
    dispatch_evidence: DispatchEvidence,
    canonical_operation_id: str,
    coverage_cell_id: str,
    generator_id: str,
    family: str,
    shape_regime: str,
    dtype: str,
    accumulation_dtype: str,
    phase: str,
    backend: str,
    layout: str,
    optimizer_context: str,
    architecture_context: str,
    target_hardware_id: str,
    accepted: bool,
    complete_capture: bool,
    strict_capture: bool,
    tensor_producing_nodes: int,
    structurally_encoded_nodes: int,
    silently_dropped_tensor_nodes: int,
    capture_profile_fingerprint_match: bool,
    gpu_time_us: float,
    capture_artifact_sha256: str,
    artifact_record_index: int = 0,
    artifact_manifest_record_sha256s: tuple[str, ...] | None = None,
    generators: OperationGeneratorRegistry | None = None,
) -> CoverageObservation:
    generators = generators or build_operation_generator_registry()
    dispatch = verify_dispatch(
        dispatch_request,
        dispatch_evidence,
        generators=generators,
    )
    binding = {
        "canonical_operation_id": canonical_operation_id,
        "coverage_cell_id": coverage_cell_id,
        "generator_id": generator_id,
        "family": family,
        "shape_regime": shape_regime,
        "dtype": dtype,
        "accumulation_dtype": accumulation_dtype,
        "phase": phase,
        "backend": backend,
        "layout": layout,
        "optimizer_context": optimizer_context,
        "architecture_context": architecture_context,
        "target_hardware_id": target_hardware_id,
        "accepted": accepted,
        "complete_capture": complete_capture,
        "strict_capture": strict_capture,
        "tensor_producing_nodes": tensor_producing_nodes,
        "structurally_encoded_nodes": structurally_encoded_nodes,
        "silently_dropped_tensor_nodes": silently_dropped_tensor_nodes,
        "capture_profile_fingerprint_match": capture_profile_fingerprint_match,
        "gpu_time_us": gpu_time_us,
    }
    configuration = {
        name: binding[name]
        for name in (
            "canonical_operation_id",
            "coverage_cell_id",
            "generator_id",
            "family",
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer_context",
            "architecture_context",
            "target_hardware_id",
        )
    }
    artifact_record_sha256 = canonical_sha256(binding)
    profiler_artifact_sha256 = artifact_record_sha256
    artifact_manifest_record_sha256s = artifact_manifest_record_sha256s or (
        artifact_record_sha256,
    )
    artifact_manifest_sha256 = canonical_sha256(
        {
            "capture_artifact_sha256": capture_artifact_sha256,
            "profiler_artifact_sha256": profiler_artifact_sha256,
            "record_sha256s": artifact_manifest_record_sha256s,
        }
    )
    measurement_occurrence_id = canonical_sha256(
        {
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "artifact_record_index": artifact_record_index,
        }
    )
    measurement_payload = {
        "version": MEASUREMENT_EVIDENCE_VERSION,
        "measurement_occurrence_id": measurement_occurrence_id,
        "source_evidence_sha256": dispatch_evidence.sha256,
        "configuration_sha256": canonical_sha256(configuration),
        "capture_artifact_sha256": capture_artifact_sha256,
        "profiler_artifact_sha256": profiler_artifact_sha256,
        "artifact_record_index": artifact_record_index,
        "artifact_record_sha256": artifact_record_sha256,
        "artifact_manifest_record_sha256s": artifact_manifest_record_sha256s,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        **binding,
    }
    measurement = CoverageMeasurementEvidence(
        measurement_id=canonical_sha256(measurement_payload),
        **measurement_payload,
    )
    payload = {
        "version": COVERAGE_OBSERVATION_VERSION,
        **binding,
        "measurement_evidence": measurement,
        "dispatch_request": dispatch_request,
        "dispatch_evidence": dispatch_evidence,
        "dispatch_verification": dispatch,
        "structural_evidence": None,
    }
    result = CoverageObservation(observation_id=canonical_sha256(payload), **payload)
    result.validate(generators)
    return result


def make_structural_coverage_observation(
    *,
    raw_target: str,
    workload_sha256: str,
    shape_regime: str,
    dtype: str,
    accumulation_dtype: str,
    phase: str,
    layout: str,
    optimizer_context: str,
    architecture_context: str,
    accepted: bool,
    complete_capture: bool,
    strict_capture: bool,
    tensor_producing_nodes: int,
    structurally_encoded_nodes: int,
    silently_dropped_tensor_nodes: int,
    capture_profile_fingerprint_match: bool,
    gpu_time_us: float,
    capture_artifact_sha256: str,
    artifact_record_index: int = 0,
    artifact_manifest_record_sha256s: tuple[str, ...] | None = None,
) -> CoverageObservation:
    evidence_payload = {
        "version": STRUCTURAL_EVIDENCE_VERSION,
        "raw_target": raw_target,
        "target_hardware_id": TARGET_HARDWARE_ID,
        "accepted_measurement": True,
        "capture_workload_sha256": workload_sha256,
        "profile_workload_sha256": workload_sha256,
        "observed_backend_id": "cuda_eager",
        "structural_path": "family_hash_custom",
    }
    structural = StructuralCoverageEvidence(
        evidence_id=canonical_sha256(evidence_payload), **evidence_payload
    )
    coverage_cell_id = canonical_sha256(
        {
            "operation": "UNK",
            "shape_regime": shape_regime,
            "dtype": dtype,
            "accumulation_dtype": accumulation_dtype,
            "phase": phase,
            "backend": "cuda_eager",
            "layout": layout,
            "optimizer": optimizer_context,
            "architecture_context": architecture_context,
        }
    )
    binding = {
        "version": COVERAGE_OBSERVATION_VERSION,
        "canonical_operation_id": "UNK",
        "coverage_cell_id": coverage_cell_id,
        "generator_id": STRUCTURAL_GENERATOR_ID,
        "family": "unknown_or_custom",
        "shape_regime": shape_regime,
        "dtype": dtype,
        "accumulation_dtype": accumulation_dtype,
        "phase": phase,
        "backend": "cuda_eager",
        "layout": layout,
        "optimizer_context": optimizer_context,
        "architecture_context": architecture_context,
        "target_hardware_id": TARGET_HARDWARE_ID,
        "accepted": accepted,
        "complete_capture": complete_capture,
        "strict_capture": strict_capture,
        "tensor_producing_nodes": tensor_producing_nodes,
        "structurally_encoded_nodes": structurally_encoded_nodes,
        "silently_dropped_tensor_nodes": silently_dropped_tensor_nodes,
        "capture_profile_fingerprint_match": capture_profile_fingerprint_match,
        "gpu_time_us": gpu_time_us,
    }
    configuration = {
        name: binding[name]
        for name in (
            "canonical_operation_id",
            "coverage_cell_id",
            "generator_id",
            "family",
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer_context",
            "architecture_context",
            "target_hardware_id",
        )
    }
    record_binding = {key: value for key, value in binding.items() if key != "version"}
    artifact_record_sha256 = canonical_sha256(record_binding)
    profiler_artifact_sha256 = artifact_record_sha256
    artifact_manifest_record_sha256s = artifact_manifest_record_sha256s or (
        artifact_record_sha256,
    )
    artifact_manifest_sha256 = canonical_sha256(
        {
            "capture_artifact_sha256": capture_artifact_sha256,
            "profiler_artifact_sha256": profiler_artifact_sha256,
            "record_sha256s": artifact_manifest_record_sha256s,
        }
    )
    measurement_occurrence_id = canonical_sha256(
        {
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "artifact_record_index": artifact_record_index,
        }
    )
    measurement_payload = {
        "version": MEASUREMENT_EVIDENCE_VERSION,
        "measurement_occurrence_id": measurement_occurrence_id,
        "source_evidence_sha256": structural.evidence_id,
        "configuration_sha256": canonical_sha256(configuration),
        "capture_artifact_sha256": capture_artifact_sha256,
        "profiler_artifact_sha256": profiler_artifact_sha256,
        "artifact_record_index": artifact_record_index,
        "artifact_record_sha256": artifact_record_sha256,
        "artifact_manifest_record_sha256s": artifact_manifest_record_sha256s,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        **record_binding,
    }
    measurement = CoverageMeasurementEvidence(
        measurement_id=canonical_sha256(measurement_payload), **measurement_payload
    )
    payload = {
        **binding,
        "measurement_evidence": measurement,
        "dispatch_request": None,
        "dispatch_evidence": None,
        "dispatch_verification": None,
        "structural_evidence": structural,
    }
    result = CoverageObservation(observation_id=canonical_sha256(payload), **payload)
    result.validate()
    return result


def _smallest_time_vocabulary(
    gpu_time_by_operation: Mapping[str, float],
    threshold: float,
    total_gpu_time: float,
) -> tuple[str, ...]:
    positive = sorted(
        ((operation, float(value)) for operation, value in gpu_time_by_operation.items() if value > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if total_gpu_time <= 0:
        return ()
    selected: list[str] = []
    cumulative = 0.0
    for operation, value in positive:
        selected.append(operation)
        cumulative += value
        if cumulative / total_gpu_time >= threshold:
            break
    return tuple(selected)


def build_operation_coverage_report(
    observations: Iterable[CoverageObservation],
    *,
    support: OperationSupportContract | None = None,
    generators: OperationGeneratorRegistry | None = None,
    config: OperationCoverageConfig | None = None,
) -> dict[str, object]:
    support = support or build_operation_support_contract()
    generators = generators or build_operation_generator_registry(support=support)
    config = config or load_operation_coverage_config()
    generators.validate(support=support)
    config.validate()
    rows = tuple(sorted(observations, key=lambda row: row.observation_id))
    if len({row.observation_id for row in rows}) != len(rows):
        raise OperationCoverageError("coverage observations must be unique")
    if len({row.measurement_evidence.measurement_id for row in rows}) != len(rows):
        raise OperationCoverageError("measurement evidence records must be unique")
    if len({row.measurement_evidence.measurement_occurrence_id for row in rows}) != len(rows):
        raise OperationCoverageError("measurement occurrence IDs must not be replayed")
    if len({row.measurement_evidence.source_evidence_sha256 for row in rows}) != len(rows):
        raise OperationCoverageError("dispatcher/structural evidence must not be replayed")
    manifests_by_artifact_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        evidence = row.measurement_evidence
        manifests_by_artifact_pair[
            (evidence.capture_artifact_sha256, evidence.profiler_artifact_sha256)
        ].add(evidence.artifact_manifest_sha256)
    if any(len(manifests) != 1 for manifests in manifests_by_artifact_pair.values()):
        raise OperationCoverageError(
            "one artifact pair cannot be relabeled with conflicting manifests"
        )
    generator_by_id = {row.generator_id: row for row in generators.generators}
    for row in rows:
        row.validate(generators)
        if row.generator_id == STRUCTURAL_GENERATOR_ID:
            continue
        generator = generator_by_id.get(row.generator_id)
        if generator is None:
            raise OperationCoverageError("coverage observation references an unknown generator")
        if (row.canonical_operation_id, row.family) != (
            generator.canonical_operation_id,
            generator.family,
        ):
            raise OperationCoverageError("coverage observation generator identity mismatch")

    v100_rows = tuple(row for row in rows if row.target_hardware_id == TARGET_HARDWARE_ID)
    accepted = tuple(row for row in v100_rows if row.accepted)
    tensor_nodes = sum(row.tensor_producing_nodes for row in v100_rows)
    encoded_nodes = sum(row.structurally_encoded_nodes for row in v100_rows)
    dropped_nodes = sum(row.silently_dropped_tensor_nodes for row in v100_rows)
    strict_rate = (
        sum(row.strict_capture and row.complete_capture for row in v100_rows) / len(v100_rows)
        if v100_rows
        else 0.0
    )
    encoding_rate = encoded_nodes / tensor_nodes if tensor_nodes else 0.0
    fingerprint_mismatches = sum(
        not row.capture_profile_fingerprint_match for row in v100_rows
    )
    time_by_operation: Counter[str] = Counter()
    time_by_family: Counter[str] = Counter()
    dimensions: dict[str, dict[str, int]] = {
        name: defaultdict(int)
        for name in (
            "phase",
            "dtype",
            "accumulation_dtype",
            "backend",
            "layout",
            "optimizer_context",
            "family",
            "architecture_context",
        )
    }
    coverage_by_operation: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in accepted:
        time_by_operation[row.canonical_operation_id] += row.gpu_time_us
        time_by_family[row.family] += row.gpu_time_us
        for name in dimensions:
            dimensions[name][str(getattr(row, name))] += 1
        for name in (
            "shape_regime",
            "dtype",
            "accumulation_dtype",
            "phase",
            "backend",
            "layout",
            "optimizer_context",
            "architecture_context",
        ):
            coverage_by_operation[row.canonical_operation_id][name].add(str(getattr(row, name)))
    required = {
        row.canonical_id: row for row in support.operations if row.support_state == "required_measured"
    }
    gaps: dict[str, list[str]] = {}
    for operation_id, requirement in required.items():
        observed = coverage_by_operation.get(operation_id, {})
        missing: list[str] = []
        expected = {
            "shape_regime": requirement.required_shape_regimes,
            "dtype": requirement.required_dtypes,
            "accumulation_dtype": ("float32",),
            "phase": requirement.required_phases,
            "backend": requirement.required_backends,
            "layout": requirement.required_layouts,
            "optimizer_context": ("sgd", "adamw", "none"),
            "architecture_context": (
                "sequential",
                "residual_or_branch",
                "saved_activation_or_alias",
            ),
        }
        for name, values in expected.items():
            for value in values:
                if value not in observed.get(name, set()):
                    missing.append(f"{name}:{value}")
        if missing:
            gaps[operation_id] = missing
    total_time = sum(time_by_operation.values())
    unknown_time = time_by_operation.get("UNK", 0.0)
    unknown_fraction = unknown_time / total_time if total_time else 1.0
    known_time = {key: value for key, value in time_by_operation.items() if key in required}
    vocabulary = _smallest_time_vocabulary(
        known_time,
        config.gates.minimum_exact_vocabulary_gpu_time_fraction,
        total_time,
    )
    vocabulary_time = sum(time_by_operation[operation] for operation in vocabulary)
    vocabulary_fraction = vocabulary_time / total_time if total_time else 0.0
    gate_values = {
        "silently_dropped_tensor_producing_operations": dropped_nodes,
        "structurally_encoded_node_fraction": encoding_rate,
        "strict_complete_capture_rate": strict_rate,
        "complete_tensor_node_encoding_rate": encoding_rate,
        "measured_unknown_or_custom_gpu_time_fraction": unknown_fraction,
        "exact_vocabulary_cumulative_v100_gpu_time_coverage": vocabulary_fraction,
        "capture_profile_workload_fingerprint_mismatches": fingerprint_mismatches,
        "required_operation_gap_count": len(gaps),
    }
    gates_passed = bool(accepted) and (
        dropped_nodes == config.gates.silently_dropped_tensor_producing_operations
        and encoding_rate >= config.gates.structurally_encoded_node_fraction
        and strict_rate >= config.gates.minimum_strict_complete_capture_rate
        and encoding_rate >= config.gates.minimum_complete_tensor_node_encoding_rate
        and unknown_fraction <= config.gates.maximum_unknown_or_custom_gpu_time_fraction
        and vocabulary_fraction >= config.gates.minimum_exact_vocabulary_gpu_time_fraction
        and fingerprint_mismatches == config.gates.capture_profile_workload_fingerprint_mismatches
        and not gaps
    )
    payload: dict[str, object] = {
        "version": COVERAGE_REPORT_VERSION,
        "target_hardware_id": TARGET_HARDWARE_ID,
        "measurement_source": "accepted_v100" if accepted else "no_accepted_measurements",
        "training_approved": gates_passed,
        "support_contract_sha256": support.sha256,
        "generator_registry_sha256": generators.sha256,
        "coverage_config_sha256": config.sha256,
        "observation_count": len(rows),
        "accepted_observation_count": len(accepted),
        "observations": [canonical_value(asdict(row)) for row in rows],
        "gate_values": gate_values,
        "required_operation_gaps": dict(sorted(gaps.items())),
        "gpu_time_us_by_operation": dict(sorted(time_by_operation.items())),
        "gpu_time_us_by_family": dict(sorted(time_by_family.items())),
        "coverage_slices": {
            name: dict(sorted(values.items())) for name, values in dimensions.items()
        },
        "provisional_exact_vocabulary": list(vocabulary),
        "gates_passed": gates_passed,
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return canonical_value(payload)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, context: str) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        raise OperationCoverageError(f"{context} differs from the frozen evidence schema")


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OperationCoverageError(f"{context} must be a mapping")
    return value


def _tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise OperationCoverageError(f"{context} must be a JSON string list")
    return tuple(value)


def _nested_dataclass(value: object, cls: type[Any], tuple_names: tuple[str, ...]) -> Any:
    raw = dict(_mapping(value, context=cls.__name__))
    _exact_keys(raw, {field.name for field in fields(cls)}, context=cls.__name__)
    for name in tuple_names:
        raw[name] = _tuple(raw[name], context=f"{cls.__name__}.{name}")
    return cls(**raw)


def coverage_observation_from_dict(
    value: object,
    generators: OperationGeneratorRegistry | None = None,
) -> CoverageObservation:
    raw = dict(_mapping(value, context="coverage observation"))
    _exact_keys(
        raw,
        {field.name for field in fields(CoverageObservation)},
        context="coverage observation",
    )
    raw["measurement_evidence"] = _nested_dataclass(
        raw["measurement_evidence"],
        CoverageMeasurementEvidence,
        ("artifact_manifest_record_sha256s",),
    )
    if raw["dispatch_request"] is not None:
        raw["dispatch_request"] = _nested_dataclass(
            raw["dispatch_request"], DispatchRequest, ("aliases",)
        )
    if raw["dispatch_evidence"] is not None:
        raw["dispatch_evidence"] = _nested_dataclass(
            raw["dispatch_evidence"],
            DispatchEvidence,
            ("observed_raw_targets", "observed_backend_ids"),
        )
    if raw["dispatch_verification"] is not None:
        raw["dispatch_verification"] = _nested_dataclass(
            raw["dispatch_verification"],
            DispatchVerification,
            ("matched_raw_targets", "matched_backend_ids"),
        )
    if raw["structural_evidence"] is not None:
        raw["structural_evidence"] = _nested_dataclass(
            raw["structural_evidence"], StructuralCoverageEvidence, ()
        )
    result = CoverageObservation(**raw)
    result.validate(generators)
    return result


def freeze_runtime_allowlist(
    report: Mapping[str, object],
    *,
    support: OperationSupportContract | None = None,
    generators: OperationGeneratorRegistry | None = None,
    config: OperationCoverageConfig | None = None,
) -> dict[str, object]:
    support = support or build_operation_support_contract()
    generators = generators or build_operation_generator_registry(support=support)
    config = config or load_operation_coverage_config()
    expected_report_keys = {
        "version",
        "target_hardware_id",
        "measurement_source",
        "training_approved",
        "support_contract_sha256",
        "generator_registry_sha256",
        "coverage_config_sha256",
        "observation_count",
        "accepted_observation_count",
        "observations",
        "gate_values",
        "required_operation_gaps",
        "gpu_time_us_by_operation",
        "gpu_time_us_by_family",
        "coverage_slices",
        "provisional_exact_vocabulary",
        "gates_passed",
        "report_sha256",
    }
    _exact_keys(report, expected_report_keys, context="coverage report")
    serialized = report.get("observations")
    if not isinstance(serialized, list):
        raise OperationCoverageError("coverage report observations must be a JSON list")
    observations = tuple(
        coverage_observation_from_dict(value, generators) for value in serialized
    )
    rebuilt = build_operation_coverage_report(
        observations,
        support=support,
        generators=generators,
        config=config,
    )
    if canonical_value(dict(report)) != rebuilt:
        raise OperationCoverageError("coverage report does not reproduce from current evidence")
    if (
        rebuilt["measurement_source"] != "accepted_v100"
        or rebuilt["training_approved"] is not True
        or rebuilt["gates_passed"] is not True
    ):
        raise OperationCoverageError(
            "runtime allowlist requires passing accepted NRP V100 coverage evidence"
        )
    vocabulary = rebuilt["provisional_exact_vocabulary"]
    required_ids = {
        row.canonical_id for row in support.operations if row.support_state == "required_measured"
    }
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or any(type(value) is not str or value not in required_ids for value in vocabulary)
    ):
        raise OperationCoverageError("passing coverage report requires a current exact vocabulary")
    payload: dict[str, object] = {
        "version": RUNTIME_ALLOWLIST_VERSION,
        "target_hardware_id": TARGET_HARDWARE_ID,
        "coverage_report_sha256": rebuilt["report_sha256"],
        "allowed_canonical_operation_ids": sorted(set(vocabulary)),
        "fallback_policy": "structured_fallback_on_nonallowlisted_or_ood",
        "training_approved": True,
    }
    payload["allowlist_sha256"] = canonical_sha256(payload)
    return canonical_value(payload)


__all__ = [
    "COVERAGE_OBSERVATION_VERSION",
    "COVERAGE_REPORT_VERSION",
    "MEASUREMENT_EVIDENCE_VERSION",
    "RUNTIME_ALLOWLIST_VERSION",
    "STRUCTURAL_EVIDENCE_VERSION",
    "STRUCTURAL_GENERATOR_ID",
    "CoverageObservation",
    "CoverageMeasurementEvidence",
    "OperationCoverageError",
    "StructuralCoverageEvidence",
    "build_operation_coverage_report",
    "coverage_observation_from_dict",
    "freeze_runtime_allowlist",
    "make_coverage_observation",
    "make_structural_coverage_observation",
]
