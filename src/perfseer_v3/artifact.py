"""Self-describing v3 checkpoint artifacts and integrity-checked registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .coarsen_v3 import COARSENING_POLICY_SHA256
from .features import (
    FeatureLayoutV3,
    NormalizationBlock,
    NormalizationStatsV3,
)
from .hardware import canonical_hardware_id, require_specific_hardware_id
from .model import SeerNetV3, SeerNetV3Config
from .op_registry import OperationRegistry
from .schema import build_feature_schema
from .training import TARGET_NAMES
from .training_semantics import canonical_optimizer_name, canonical_scheduler_name
from .version import (
    FEATURE_SCHEMA_VERSION,
    GRAPH_IR_VERSION,
    LABEL_SCHEMA_VERSION,
    OP_REGISTRY_VERSION,
    STUDENT_MODEL_RELEASE,
    TEACHER_MODEL_RELEASE,
)


class ArtifactIntegrityError(RuntimeError):
    """Raised if an artifact, registry record, or schema contract is corrupt."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetTransformV3:
    mean: tuple[float, ...] = (0.0,) * 6
    std: tuple[float, ...] = (1.0,) * 6
    transform: str = "identity"

    def validate(self) -> None:
        if len(self.mean) != len(TARGET_NAMES) or len(self.std) != len(TARGET_NAMES):
            raise ArtifactIntegrityError("target transform must match six-target order")
        if any(value <= 0 for value in self.std):
            raise ArtifactIntegrityError("target transform std must be positive")
        if self.transform not in {"identity", "log1p_standardized"}:
            raise ArtifactIntegrityError(f"unsupported target transform {self.transform!r}")


@dataclass(frozen=True)
class ArtifactMetadataV3:
    model_release: str
    graph_ir_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    operator_registry_version: str
    operator_registry_sha256: str
    ordered_feature_layout: dict[str, Any]
    normalization_sha256: str | None
    coarsening_policy_sha256: str
    target_names: tuple[str, ...]
    target_transform: TargetTransformV3
    label_schema_version: str
    target_hardware_id: str
    hardware_allowlist: tuple[str, ...]
    precision_allowlist: tuple[str, ...]
    capture_quality_allowlist: tuple[str, ...]
    optimizer_allowlist: tuple[str, ...]
    scheduler_allowlist: tuple[str, ...]
    training_mode_allowlist: tuple[str, ...]
    dataset_fingerprint: str
    split_fingerprint: str
    pytorch_version: str
    cuda_build_version: str | None
    model_config: dict[str, Any]
    minimum_confidence: float = 0.2
    allow_ok_with_unknowns: bool = False
    output_contract_version: str = "perfseer_v3_outputs_v2"
    optional_output_names: tuple[str, ...] = (
        "log_variance",
        "oom_probability",
        "oom_failure_stage",
        "confidence",
        "peak_live_bytes_log1p",
    )

    def validate(
        self,
        *,
        registry: OperationRegistry,
        layout: FeatureLayoutV3,
    ) -> None:
        expected_versions = {
            "graph_ir_version": GRAPH_IR_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "operator_registry_version": OP_REGISTRY_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                raise ArtifactIntegrityError(
                    f"artifact {name}={getattr(self, name)!r}, expected {expected!r}"
                )
        if self.model_release not in {TEACHER_MODEL_RELEASE, STUDENT_MODEL_RELEASE}:
            raise ArtifactIntegrityError(f"unknown model release {self.model_release!r}")
        if self.feature_schema_sha256 != layout.feature_schema_sha256:
            raise ArtifactIntegrityError("artifact feature schema hash mismatch")
        if self.operator_registry_sha256 != registry.sha256:
            raise ArtifactIntegrityError("artifact operation registry hash mismatch")
        if self.coarsening_policy_sha256 != COARSENING_POLICY_SHA256:
            raise ArtifactIntegrityError("artifact coarsening policy hash mismatch")
        if tuple(self.target_names) != TARGET_NAMES:
            raise ArtifactIntegrityError("artifact target order mismatch")
        if self.ordered_feature_layout != asdict(layout):
            raise ArtifactIntegrityError("artifact ordered feature layout mismatch")
        if any(
            not values
            for values in (
                self.hardware_allowlist,
                self.precision_allowlist,
                self.capture_quality_allowlist,
                self.optimizer_allowlist,
                self.scheduler_allowlist,
                self.training_mode_allowlist,
            )
        ):
            raise ArtifactIntegrityError("artifact deployment allowlists cannot be empty")
        try:
            target_hardware_id = require_specific_hardware_id(
                self.target_hardware_id,
                context="artifact target_hardware_id",
            )
        except ValueError as exc:
            raise ArtifactIntegrityError(str(exc)) from exc
        hardware_allowlist = tuple(
            canonical_hardware_id(value) for value in self.hardware_allowlist
        )
        if hardware_allowlist != (target_hardware_id,):
            raise ArtifactIntegrityError(
                "a v3 artifact must target exactly one GPU and its hardware_allowlist "
                "must contain only target_hardware_id"
            )
        if tuple(canonical_optimizer_name(value) for value in self.optimizer_allowlist) != (
            self.optimizer_allowlist
        ):
            raise ArtifactIntegrityError("artifact optimizer allowlist must be canonical")
        if tuple(canonical_scheduler_name(value) for value in self.scheduler_allowlist) != (
            self.scheduler_allowlist
        ):
            raise ArtifactIntegrityError("artifact scheduler allowlist must be canonical")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ArtifactIntegrityError("minimum confidence must be in [0, 1]")
        if self.output_contract_version != "perfseer_v3_outputs_v2":
            raise ArtifactIntegrityError("unsupported v3 output contract")
        required_optional = {
            "log_variance",
            "oom_probability",
            "oom_failure_stage",
            "confidence",
            "peak_live_bytes_log1p",
        }
        if set(self.optional_output_names) != required_optional:
            raise ArtifactIntegrityError("artifact optional output contract mismatch")
        self.target_transform.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactMetadataV3":
        raw = dict(value)
        raw["target_names"] = tuple(raw["target_names"])
        raw["optional_output_names"] = tuple(raw.get("optional_output_names", ()))
        for name in (
            "hardware_allowlist",
            "precision_allowlist",
            "capture_quality_allowlist",
            "optimizer_allowlist",
            "scheduler_allowlist",
            "training_mode_allowlist",
        ):
            raw[name] = tuple(raw[name])
        raw["target_transform"] = TargetTransformV3(**raw["target_transform"])
        return cls(**raw)


@dataclass(frozen=True)
class ArtifactRecordV3:
    artifact_id: str
    path: str
    sha256: str
    model_release: str
    target_hardware_id: str
    hardware_allowlist: tuple[str, ...]
    precision_allowlist: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecordV3":
        raw = dict(value)
        raw["hardware_allowlist"] = tuple(raw["hardware_allowlist"])
        raw["precision_allowlist"] = tuple(raw["precision_allowlist"])
        record = cls(**raw)
        try:
            target = require_specific_hardware_id(
                record.target_hardware_id,
                context="artifact registry target_hardware_id",
            )
        except ValueError as exc:
            raise ArtifactIntegrityError(str(exc)) from exc
        if tuple(canonical_hardware_id(value) for value in record.hardware_allowlist) != (
            target,
        ):
            raise ArtifactIntegrityError(
                "artifact registry records must contain exactly one matching GPU"
            )
        return record


class ArtifactRegistryV3:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("registry_version") != "perfseer_v3_artifacts_v2":
            raise ArtifactIntegrityError("unsupported artifact registry version")
        records = tuple(
            ArtifactRecordV3.from_dict(record)
            for record in payload.get("artifacts", ())
        )
        ids = [record.artifact_id for record in records]
        if len(ids) != len(set(ids)):
            raise ArtifactIntegrityError("duplicate artifact IDs")
        self.records = records

    def select(
        self,
        *,
        hardware_id: str,
        precision: str,
        model_release: str = STUDENT_MODEL_RELEASE,
    ) -> tuple[ArtifactRecordV3, Path]:
        try:
            requested_hardware_id = require_specific_hardware_id(
                hardware_id,
                context="artifact selection hardware_id",
            )
        except ValueError as exc:
            raise ArtifactIntegrityError(str(exc)) from exc
        matches = [
            record
            for record in self.records
            if record.model_release == model_release
            and requested_hardware_id == canonical_hardware_id(record.target_hardware_id)
            and precision in record.precision_allowlist
        ]
        if len(matches) != 1:
            raise ArtifactIntegrityError(
                f"expected exactly one artifact for {hardware_id}/{precision}/{model_release}, "
                f"found {len(matches)}"
            )
        record = matches[0]
        artifact_path = (self.path.parent / record.path).resolve()
        if not artifact_path.is_file():
            raise ArtifactIntegrityError(f"artifact is missing: {artifact_path}")
        actual = sha256_file(artifact_path)
        if actual != record.sha256:
            raise ArtifactIntegrityError(
                f"artifact hash mismatch for {record.artifact_id}: {actual} != {record.sha256}"
            )
        return record, artifact_path


def save_checkpoint_artifact(
    path: str | Path,
    *,
    model: SeerNetV3,
    metadata: ArtifactMetadataV3,
    normalization: NormalizationStatsV3 | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "perfseer_v3_state_dict_v2",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "metadata": metadata.to_dict(),
        "normalization": asdict(normalization) if normalization is not None else None,
        "calibration": dict(calibration or {}),
    }
    torch.save(payload, output)
    return output


def _normalization_from_dict(value: Mapping[str, Any] | None) -> NormalizationStatsV3 | None:
    if value is None:
        return None
    raw = dict(value)
    for name in ("node", "edge", "global_features"):
        raw[name] = NormalizationBlock(**raw[name])
    raw["quantiles"] = tuple(raw["quantiles"])
    return NormalizationStatsV3(**raw)


@dataclass(frozen=True)
class LoadedArtifactV3:
    model: SeerNetV3
    metadata: ArtifactMetadataV3
    normalization: NormalizationStatsV3 | None
    calibration: dict[str, Any]
    path: Path
    sha256: str


def load_checkpoint_artifact(
    path: str | Path,
    *,
    registry: OperationRegistry | None = None,
) -> LoadedArtifactV3:
    artifact_path = Path(path).resolve()
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(artifact_path, map_location="cpu")
    if payload.get("format") != "perfseer_v3_state_dict_v2":
        raise ArtifactIntegrityError("unsupported PerfSeer artifact format")
    registry = registry or OperationRegistry.load()
    schema = build_feature_schema(registry)
    layout_data = payload["metadata"]["ordered_feature_layout"]
    layout = FeatureLayoutV3(
        feature_schema_version=layout_data["feature_schema_version"],
        feature_schema_sha256=layout_data["feature_schema_sha256"],
        operator_registry_sha256=layout_data["operator_registry_sha256"],
        node_continuous_fields=tuple(layout_data["node_continuous_fields"]),
        edge_continuous_fields=tuple(layout_data["edge_continuous_fields"]),
        global_continuous_fields=tuple(layout_data["global_continuous_fields"]),
        node_flag_fields=tuple(layout_data["node_flag_fields"]),
        edge_flag_fields=tuple(layout_data["edge_flag_fields"]),
        quality_fields=tuple(layout_data["quality_fields"]),
    )
    if layout.feature_schema_sha256 != schema["feature_schema_sha256"]:
        raise ArtifactIntegrityError("runtime feature schema differs from artifact layout")
    metadata = ArtifactMetadataV3.from_dict(payload["metadata"])
    metadata.validate(registry=registry, layout=layout)
    if payload["model_config"] != metadata.model_config:
        raise ArtifactIntegrityError("artifact model config metadata mismatch")
    model_config = SeerNetV3Config(**payload["model_config"])
    model = SeerNetV3(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    normalization = _normalization_from_dict(payload.get("normalization"))
    if normalization is not None:
        if metadata.normalization_sha256 != normalization.sha256:
            raise ArtifactIntegrityError("artifact normalization hash mismatch")
    elif metadata.normalization_sha256 is not None:
        raise ArtifactIntegrityError("artifact declares normalization but embeds none")
    return LoadedArtifactV3(
        model=model,
        metadata=metadata,
        normalization=normalization,
        calibration=dict(payload.get("calibration") or {}),
        path=artifact_path,
        sha256=sha256_file(artifact_path),
    )


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactMetadataV3",
    "ArtifactRecordV3",
    "ArtifactRegistryV3",
    "LoadedArtifactV3",
    "TargetTransformV3",
    "load_checkpoint_artifact",
    "save_checkpoint_artifact",
    "sha256_file",
]
