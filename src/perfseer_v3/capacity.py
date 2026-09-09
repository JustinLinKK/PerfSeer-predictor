"""Validated teacher/student capacity-study definitions for PerfSeer v3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .baseline import canonical_json
from .features import FeatureLayoutV3
from .model import SeerNetV3, SeerNetV3Config
from .op_registry import OperationRegistry
from .version import FEATURE_SCHEMA_VERSION, OP_REGISTRY_VERSION


DEFAULT_CAPACITY_STUDY_PATH = (
    Path(__file__).with_name("configs")
    / "capacity_sweep"
    / "capacity_candidates.yaml"
)


@dataclass(frozen=True)
class CapacityCandidateV3:
    candidate_id: str
    role: str
    purpose: str
    model_overrides: dict[str, Any]

    def model_config(
        self,
        registry: OperationRegistry,
        layout: FeatureLayoutV3,
        **overrides: Any,
    ) -> SeerNetV3Config:
        values = dict(self.model_overrides)
        values.update(overrides)
        return SeerNetV3Config.from_registry(registry, layout, **values)


@dataclass(frozen=True)
class CapacityStudyV3:
    version: str
    feature_schema_version: str
    operator_registry_version: str
    controls: dict[str, Any]
    candidates: tuple[CapacityCandidateV3, ...]
    deployment_gates: dict[str, float]
    selection: dict[str, str]

    @property
    def sha256(self) -> str:
        payload = {
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "operator_registry_version": self.operator_registry_version,
            "controls": self.controls,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "role": candidate.role,
                    "purpose": candidate.purpose,
                    "model_overrides": candidate.model_overrides,
                }
                for candidate in self.candidates
            ],
            "deployment_gates": self.deployment_gates,
            "selection": self.selection,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def candidate(self, candidate_id: str) -> CapacityCandidateV3:
        matches = [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == candidate_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown capacity candidate {candidate_id!r}")
        return matches[0]


def _candidate_rows(
    role: str,
    rows: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> tuple[CapacityCandidateV3, ...]:
    result = []
    for candidate_id, raw in rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"capacity candidate {candidate_id!r} must be a mapping")
        purpose = str(raw.get("purpose", "")).strip()
        if not purpose:
            raise ValueError(f"capacity candidate {candidate_id!r} needs a purpose")
        model_overrides = {
            **dict(controls),
            **{key: value for key, value in raw.items() if key != "purpose"},
        }
        result.append(
            CapacityCandidateV3(
                candidate_id=str(candidate_id),
                role=role,
                purpose=purpose,
                model_overrides=model_overrides,
            )
        )
    return tuple(result)


def load_capacity_study(
    path: str | Path = DEFAULT_CAPACITY_STUDY_PATH,
) -> CapacityStudyV3:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("capacity study root must be a mapping")
    controls = dict(raw.get("controls", {}))
    study = CapacityStudyV3(
        version=str(raw.get("capacity_study_version", "")),
        feature_schema_version=str(raw.get("feature_schema_version", "")),
        operator_registry_version=str(raw.get("operator_registry_version", "")),
        controls=controls,
        candidates=(
            *_candidate_rows(
                "teacher",
                raw.get("teacher_candidates", {}),
                controls,
            ),
            *_candidate_rows(
                "student",
                raw.get("student_candidates", {}),
                controls,
            ),
        ),
        deployment_gates={
            str(key): float(value)
            for key, value in dict(raw.get("deployment_gates", {})).items()
        },
        selection={
            str(key): str(value)
            for key, value in dict(raw.get("selection", {})).items()
        },
    )
    validate_capacity_study(study)
    return study


def validate_capacity_study(study: CapacityStudyV3) -> None:
    if study.version != "perfseer_v3_capacity_study_v1":
        raise ValueError("unsupported capacity study version")
    if study.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("capacity study feature schema version mismatch")
    if study.operator_registry_version != OP_REGISTRY_VERSION:
        raise ValueError("capacity study registry version mismatch")
    ids = [candidate.candidate_id for candidate in study.candidates]
    required = {"T0", "T1", "T2", "S0", "S1", "S2", "S3"}
    if set(ids) != required or len(ids) != len(set(ids)):
        raise ValueError("capacity study must define exactly T0/T1/T2/S0/S1/S2/S3")
    expected_dimensions = {
        "T0": (1024, 8),
        "T1": (1280, 10),
        "T2": (1536, 10),
        "S0": (192, 2),
        "S1": (224, 2),
        "S2": (256, 2),
        "S3": (256, 3),
    }
    for candidate in study.candidates:
        expected_hidden, expected_blocks = expected_dimensions[candidate.candidate_id]
        if int(candidate.model_overrides["hidden"]) != expected_hidden:
            raise ValueError(f"{candidate.candidate_id} hidden size drifted")
        if int(candidate.model_overrides["num_blocks"]) != expected_blocks:
            raise ValueError(f"{candidate.candidate_id} block count drifted")
        if int(candidate.model_overrides.get("num_outputs", 0)) != 6:
            raise ValueError(f"{candidate.candidate_id} changed the six-output contract")
        if candidate.model_overrides.get("pooling_mode") not in {
            "existing",
            "phase_aware",
        }:
            raise ValueError(f"{candidate.candidate_id} has invalid pooling mode")
    for key in (
        "maximum_cpu_p95_latency_ratio_vs_v2",
        "maximum_artifact_size_ratio_vs_v2",
    ):
        if study.deployment_gates.get(key, 0.0) <= 0:
            raise ValueError(f"capacity study is missing deployment gate {key!r}")


def exact_parameter_count(config: SeerNetV3Config) -> int:
    """Instantiate on the meta device and count the configured trainable tensors."""

    import torch

    with torch.device("meta"):
        model = SeerNetV3(config)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "CapacityCandidateV3",
    "CapacityStudyV3",
    "DEFAULT_CAPACITY_STUDY_PATH",
    "exact_parameter_count",
    "load_capacity_study",
    "validate_capacity_study",
]
