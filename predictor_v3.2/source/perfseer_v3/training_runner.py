"""Fail-closed v3 teacher training and teacher-to-student distillation runner."""

from __future__ import annotations

import json
import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifact import (
    ArtifactMetadataV3,
    TargetTransformV3,
    load_checkpoint_artifact,
    save_checkpoint_artifact,
    sha256_file,
)
from .baseline import canonical_json
from .capture_export import CaptureOptions, capture_export
from .coarsen_v3 import COARSENING_POLICY_ID, COARSENING_POLICY_SHA256, coarsen_graph
from .features import (
    GraphFeaturesV3,
    apply_normalization,
    batch_graph_features,
    build_graph_features,
    fit_normalization,
    graph_precision_category,
)
from .graph_ir_v3 import GraphIRV3
from .hardware import (
    HardwareNormalizationPolicyV3,
    canonical_hardware_id,
    graph_hardware_id,
    require_specific_hardware_id,
)
from .hardware_transfer import (
    BaseTransferLineageV3,
    PairedResidualTransformV3,
    TransferLineageV3,
    configure_transfer_trainable_parameters,
    parameter_checksum,
)
from .model import OOM_FAILURE_STAGES, SeerNetV3, graph_batch_tensors
from .op_registry import OperationRegistry
from .training import (
    DatasetGateReport,
    EncoderPretrainer,
    PRETRAINING_OBJECTIVES,
    TARGET_NAMES,
    TrainingConfigV3,
    TrainingGateError,
    TrainingSampleV3,
    PairedTransferSampleV3,
    _batch_to_device,
    assert_training_ready,
    encoder_pretrain_step,
    fit_binary_temperature_calibration,
    fit_linear_calibration,
    fit_uncertainty_calibration,
    run_tiny_training_smoke,
    student_distill_step,
    target_student_adapter_step,
    target_teacher_adapter_step,
    teacher_train_step,
)
from .training_semantics import canonical_optimizer_name, canonical_scheduler_name
from .dataset_pack.transfer_labeling import (
    TARGET_TRAINING_MANIFEST_VERSION,
    TransferLabelAttemptV3,
)
from .version import WORKLOAD_NORMALIZATION_VERSION


TRAINING_MANIFEST_VERSION = "perfseer_v3_training_manifest_v2"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingManifestRowV3:
    sample_id: str
    graph_path: str
    split: str
    source_group: str
    graph_signature: str
    hardware_id: str
    target: tuple[float, ...]
    oom: float = 0.0
    oom_stage: str = "none"
    peak_live_bytes: float | None = None
    domain_weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingManifestRowV3":
        raw = dict(value)
        raw["target"] = tuple(float(item) for item in raw["target"])
        raw["hardware_id"] = require_specific_hardware_id(
            raw.get("hardware_id"),
            context="training manifest row hardware_id",
        )
        row = cls(**raw)
        row.validate()
        return row

    def validate(self) -> None:
        if not self.sample_id or not self.graph_path:
            raise ValueError("training manifest rows require sample_id and graph_path")
        if self.split not in SPLITS:
            raise ValueError(f"invalid training split {self.split!r}")
        if not self.source_group or not self.graph_signature:
            raise ValueError("source_group and graph_signature are required for leakage checks")
        require_specific_hardware_id(
            self.hardware_id,
            context="training manifest row hardware_id",
        )
        if len(self.target) != len(TARGET_NAMES):
            raise ValueError("training manifest targets must follow the six-target contract")
        if any(not math.isfinite(value) for value in self.target):
            raise ValueError("training targets must be finite")
        if self.oom not in {0.0, 1.0}:
            raise ValueError("OOM target must be binary")
        if self.oom_stage not in OOM_FAILURE_STAGES:
            raise ValueError(f"invalid OOM stage {self.oom_stage!r}")
        if (self.oom == 0.0) != (self.oom_stage == "none"):
            raise ValueError("OOM status and failure stage disagree")
        if self.peak_live_bytes is not None and self.peak_live_bytes < 0:
            raise ValueError("peak live bytes must be nonnegative")
        if self.domain_weight <= 0:
            raise ValueError("domain weight must be positive")


@dataclass(frozen=True)
class TrainingManifestV3:
    path: Path
    dataset_gate: DatasetGateReport
    deployment: dict[str, Any]
    rows: tuple[TrainingManifestRowV3, ...]
    campaign_evidence: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "TrainingManifestV3":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("manifest_version") != TRAINING_MANIFEST_VERSION:
            raise ValueError("unsupported PerfSeer v3 training manifest")
        gate = DatasetGateReport(**payload["dataset_gate"])
        rows = tuple(
            TrainingManifestRowV3.from_dict(row) for row in payload.get("samples", ())
        )
        if not rows:
            raise ValueError("training manifest contains no samples")
        sample_ids = [row.sample_id for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("training manifest sample IDs must be unique")
        deployment = dict(payload.get("deployment", {}))
        target_hardware_id = require_specific_hardware_id(
            deployment.get("target_hardware_id"),
            context="deployment.target_hardware_id",
        )
        for name in (
            "hardware_allowlist",
            "precision_allowlist",
            "capture_quality_allowlist",
            "optimizer_allowlist",
            "scheduler_allowlist",
            "training_mode_allowlist",
        ):
            values = deployment.get(name)
            if not isinstance(values, list) or not values:
                raise ValueError(f"deployment.{name} must be a nonempty list")
        hardware_allowlist = tuple(
            canonical_hardware_id(value)
            for value in deployment["hardware_allowlist"]
        )
        if hardware_allowlist != (target_hardware_id,):
            raise TrainingGateError(
                "each v3 manifest must target exactly one GPU; hardware_allowlist "
                "must contain only deployment.target_hardware_id"
            )
        if any(row.hardware_id != target_hardware_id for row in rows):
            raise TrainingGateError(
                "every training row must use deployment.target_hardware_id"
            )
        deployment["target_hardware_id"] = target_hardware_id
        deployment["hardware_allowlist"] = [target_hardware_id]
        deployment["optimizer_allowlist"] = [
            canonical_optimizer_name(value)
            for value in deployment["optimizer_allowlist"]
        ]
        deployment["scheduler_allowlist"] = [
            canonical_scheduler_name(value)
            for value in deployment["scheduler_allowlist"]
        ]
        campaign_evidence = dict(payload.get("campaign_evidence") or {})
        manifest = cls(manifest_path, gate, deployment, rows, campaign_evidence)
        manifest.validate_splits()
        return manifest

    @property
    def target_hardware_id(self) -> str:
        return str(self.deployment["target_hardware_id"])

    def validate_splits(self) -> None:
        counts = {split: 0 for split in SPLITS}
        source_split: dict[str, str] = {}
        signature_split: dict[str, str] = {}
        for row in self.rows:
            counts[row.split] += 1
            for value, seen, label in (
                (row.source_group, source_split, "source group"),
                (row.graph_signature, signature_split, "graph signature"),
            ):
                prior = seen.setdefault(value, row.split)
                if prior != row.split:
                    raise TrainingGateError(
                        f"{label} {value!r} leaks across {prior!r} and {row.split!r}"
                    )
        if any(counts[split] == 0 for split in SPLITS):
            raise TrainingGateError("train, validation, and test splits must all be nonempty")
        if not self.dataset_gate.source_group_isolated:
            raise TrainingGateError("dataset gate does not attest source-group isolation")

    @property
    def split_fingerprint(self) -> str:
        payload = [
            {
                "sample_id": row.sample_id,
                "source_group": row.source_group,
                "graph_signature": row.graph_signature,
                "split": row.split,
            }
            for row in sorted(self.rows, key=lambda item: item.sample_id)
        ]
        import hashlib

        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def dataset_fingerprint(self) -> str:
        import hashlib

        payload = [
            {
                "sample_id": row.sample_id,
                "graph_signature": row.graph_signature,
                "hardware_id": row.hardware_id,
                "target": row.target,
                "oom": row.oom,
                "oom_stage": row.oom_stage,
                "peak_live_bytes": row.peak_live_bytes,
                "domain_weight": row.domain_weight,
            }
            for row in sorted(self.rows, key=lambda item: item.sample_id)
        ]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def assert_base_campaign_ready(
    manifest: TrainingManifestV3,
    registry: OperationRegistry,
    *,
    feature_schema_sha256: str | None = None,
) -> None:
    """Require complete, frozen A10 campaign evidence before base training."""

    failures: list[str] = []
    evidence = dict(manifest.campaign_evidence)
    evidence_sha256 = str(evidence.pop("campaign_evidence_sha256", ""))
    expected_evidence_sha256 = hashlib.sha256(
        canonical_json(evidence).encode("utf-8")
    ).hexdigest()
    if evidence_sha256 != expected_evidence_sha256:
        failures.append("campaign evidence content hash is missing or invalid")
    if manifest.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
        failures.append("base corpus is not bound to nvidia_a10g_24gb_aws_g5")
    if len(manifest.rows) != 18_000:
        failures.append("base corpus does not contain exactly 18,000 accepted configurations")
    if evidence.get("accepted_configuration_count") != 18_000:
        failures.append("campaign evidence does not attest exactly 18,000 accepted rows")
    if evidence.get("measured_epoch_record_count") != 54_000:
        failures.append("campaign evidence does not attest exactly 54,000 measured epochs")
    for name in (
        "failed_attempts_retained",
        "oom_attempts_retained",
        "unstable_attempts_retained",
        "quarantined_attempts_retained",
        "all_graphs_materialized_and_verified",
        "source_group_isolated",
    ):
        if evidence.get(name) is not True:
            failures.append(f"campaign evidence lacks {name}")
    required_hashes = (
        "accepted_labels_sha256",
        "audit_report_sha256",
        "completion_receipt_sha256",
        "dataset_manifest_sha256",
        "failure_index_sha256",
        "graph_corpus_sha256",
        "operation_coverage_report_sha256",
        "operator_registry_sha256",
        "source_manifest_sha256",
        "target_manifest_sha256",
        "feature_schema_sha256",
    )
    for name in required_hashes:
        value = str(evidence.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            failures.append(f"campaign evidence has invalid {name}")
    if evidence.get("operator_registry_sha256") != registry.sha256:
        failures.append("campaign evidence operator registry differs from the loaded registry")
    if evidence.get("dataset_fingerprint") != manifest.dataset_fingerprint:
        failures.append("campaign evidence dataset fingerprint differs from manifest rows")
    if evidence.get("split_fingerprint") != manifest.split_fingerprint:
        failures.append("campaign evidence split fingerprint differs from manifest rows")
    if feature_schema_sha256 is not None and evidence.get("feature_schema_sha256") != feature_schema_sha256:
        failures.append("campaign evidence feature schema differs from materialized graphs")
    if failures:
        raise TrainingGateError("; ".join(failures))


@dataclass(frozen=True)
class TargetTrainingManifestRowV3:
    sample_id: str
    base_configuration_id: str
    graph_path: str
    split: str
    source_group: str
    graph_signature: str
    base_target: tuple[float, ...]
    target: tuple[float, ...]
    oom: float = 0.0
    oom_stage: str = "none"
    peak_live_bytes: float | None = None
    domain_weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetTrainingManifestRowV3":
        raw = dict(value)
        raw["base_target"] = tuple(float(item) for item in raw["base_target"])
        raw["target"] = tuple(float(item) for item in raw["target"])
        for extra in (
            "base_hardware_id",
            "target_hardware_id",
            "attempt_sha256",
        ):
            raw.pop(extra, None)
        row = cls(**raw)
        row.validate()
        return row

    def validate(self) -> None:
        if not all(
            (self.sample_id, self.base_configuration_id, self.graph_path, self.source_group, self.graph_signature)
        ):
            raise ValueError("target training rows require IDs, graph, group, and signature")
        if self.split not in SPLITS:
            raise ValueError("target training row has an invalid split")
        for name, values in (("base_target", self.base_target), ("target", self.target)):
            if len(values) != len(TARGET_NAMES) or any(
                not math.isfinite(value) or value < 0 for value in values
            ):
                raise ValueError(f"target training {name} violates the six-target contract")
        if self.oom not in {0.0, 1.0} or self.oom_stage not in OOM_FAILURE_STAGES:
            raise ValueError("target training OOM contract is invalid")
        if (self.oom == 0.0) != (self.oom_stage == "none"):
            raise ValueError("target training OOM status and stage disagree")
        if self.peak_live_bytes is not None and self.peak_live_bytes < 0:
            raise ValueError("target peak live bytes must be nonnegative")
        if self.domain_weight <= 0:
            raise ValueError("target domain weight must be positive")


@dataclass(frozen=True)
class TargetTrainingManifestV3:
    path: Path
    source_subset_sha256: str
    base_lineage: BaseTransferLineageV3
    label_budget: int
    base_hardware_id: str
    target_hardware_id: str
    hardware_profile_sha256: str
    deployment: dict[str, Any]
    rows: tuple[TargetTrainingManifestRowV3, ...]
    oom_attempts: tuple[TransferLabelAttemptV3, ...]
    memory_probe_attempts: tuple[TransferLabelAttemptV3, ...]
    manifest_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "TargetTrainingManifestV3":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("manifest_version") != TARGET_TRAINING_MANIFEST_VERSION:
            raise ValueError("unsupported target training manifest")
        declared_hash = str(payload.get("manifest_sha256", ""))
        unhashed = dict(payload)
        unhashed.pop("manifest_sha256", None)
        actual_hash = hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()
        if declared_hash != actual_hash:
            raise ValueError("target training manifest content hash mismatch")
        try:
            base_lineage = BaseTransferLineageV3.from_dict(
                payload.get("base_lineage", {})
            )
        except ValueError as exc:
            raise TrainingGateError(f"target manifest base lineage is invalid: {exc}") from exc
        budget = int(payload.get("label_budget", 0))
        expected_counts = {128: (96, 16, 16), 256: (192, 32, 32), 512: (384, 64, 64), 1024: (768, 128, 128)}
        if budget not in expected_counts:
            raise TrainingGateError("target manifest label budget is outside configured gates")
        base_hardware_id = require_specific_hardware_id(
            payload.get("base_hardware_id"), context="target manifest base_hardware_id"
        )
        if base_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise TrainingGateError("target manifest must derive from the frozen A10G corpus")
        target_hardware_id = require_specific_hardware_id(
            payload.get("target_hardware_id"), context="target manifest target_hardware_id"
        )
        if base_hardware_id == target_hardware_id:
            raise TrainingGateError("target manifest cannot adapt a GPU to itself")
        rows = tuple(
            TargetTrainingManifestRowV3.from_dict(row)
            for row in payload.get("samples", ())
        )
        if len(rows) != budget:
            raise TrainingGateError("target manifest successful label count differs from budget")
        counts = tuple(sum(row.split == split for row in rows) for split in SPLITS)
        if counts != expected_counts[budget]:
            raise TrainingGateError(
                f"target manifest split counts {counts} differ from {expected_counts[budget]}"
            )
        ids = [row.sample_id for row in rows]
        base_ids = [row.base_configuration_id for row in rows]
        if len(ids) != len(set(ids)) or len(base_ids) != len(set(base_ids)):
            raise TrainingGateError("target manifest paired IDs must be unique")
        source_splits: dict[str, str] = {}
        signature_splits: dict[str, str] = {}
        for row in rows:
            for value, seen, label in (
                (row.source_group, source_splits, "source group"),
                (row.graph_signature, signature_splits, "graph signature"),
            ):
                previous = seen.setdefault(value, row.split)
                if previous != row.split:
                    raise TrainingGateError(
                        f"target {label} {value!r} leaks across {previous!r} and {row.split!r}"
                    )
        deployment = dict(payload.get("deployment") or {})
        hardware_allowlist = tuple(
            canonical_hardware_id(value)
            for value in deployment.get("hardware_allowlist", [target_hardware_id])
        )
        if hardware_allowlist != (target_hardware_id,):
            raise TrainingGateError("target manifest hardware allowlist must be exact")
        deployment["hardware_allowlist"] = [target_hardware_id]
        deployment["target_hardware_id"] = target_hardware_id
        for name, default in (
            ("precision_allowlist", ["float32", "float16", "bfloat16", "mixed"]),
            ("capture_quality_allowlist", ["strict"]),
            ("optimizer_allowlist", ["adamw", "sgd"]),
            ("scheduler_allowlist", ["none", "cosine", "linear"]),
            ("training_mode_allowlist", ["training"]),
        ):
            values = deployment.get(name, default)
            if not isinstance(values, list) or not values:
                raise TrainingGateError(f"target deployment {name} must be nonempty")
            deployment[name] = values
        oom_attempts = tuple(
            TransferLabelAttemptV3.from_dict(item)
            for item in payload.get("oom_attempts", ())
        )
        memory_probe_attempts = tuple(
            TransferLabelAttemptV3.from_dict(item)
            for item in payload.get("memory_probe_attempts", ())
        )
        row_by_base_id = {row.base_configuration_id: row for row in rows}
        for collection_name, attempts in (
            ("OOM", oom_attempts),
            ("memory probe", memory_probe_attempts),
        ):
            hashes = [attempt.sha256 for attempt in attempts]
            if len(hashes) != len(set(hashes)):
                raise TrainingGateError(
                    f"target manifest repeats a retained {collection_name} attempt"
                )
            for attempt in attempts:
                if collection_name == "OOM" and attempt.status != "oom":
                    raise TrainingGateError(
                        "target manifest OOM evidence contains a non-OOM attempt"
                    )
                if collection_name == "memory probe" and attempt.split != "memory_probe":
                    raise TrainingGateError(
                        "target manifest memory-probe evidence has another split"
                    )
                if attempt.base_hardware_id != base_hardware_id:
                    raise TrainingGateError(
                        "retained target attempt names another base GPU"
                    )
                if attempt.target_hardware_id != target_hardware_id:
                    raise TrainingGateError(
                        "retained target attempt names another target GPU"
                    )
                if attempt.hardware_profile_sha256 != str(
                    payload.get("hardware_profile_sha256", "")
                ):
                    raise TrainingGateError(
                        "retained target attempt uses another hardware profile"
                    )
                row = row_by_base_id.get(attempt.base_configuration_id)
                if row is None:
                    raise TrainingGateError(
                        "retained target attempt is outside the frozen subset"
                    )
                if attempt.split != "memory_probe" and attempt.split != row.split:
                    raise TrainingGateError(
                        "retained target attempt split differs from the frozen subset"
                    )
                if tuple(attempt.base_targets) != row.base_target:
                    raise TrainingGateError(
                        "retained target attempt A10 targets differ from its paired row"
                    )
                if (
                    attempt.status == "oom"
                    and attempt.failure_stage not in OOM_FAILURE_STAGES
                ):
                    raise TrainingGateError(
                        "retained target OOM attempt has an unsupported failure stage"
                    )
                if not attempt.graph_path:
                    raise TrainingGateError(
                        "retained target attempt has no target-conditioned graph"
                    )
        manifest = cls(
            path=manifest_path,
            source_subset_sha256=str(payload.get("source_subset_sha256", "")),
            base_lineage=base_lineage,
            label_budget=budget,
            base_hardware_id=base_hardware_id,
            target_hardware_id=target_hardware_id,
            hardware_profile_sha256=str(payload.get("hardware_profile_sha256", "")),
            deployment=deployment,
            rows=rows,
            oom_attempts=oom_attempts,
            memory_probe_attempts=memory_probe_attempts,
            manifest_sha256=declared_hash,
        )
        for name, digest in (
            ("source subset", manifest.source_subset_sha256),
            ("hardware profile", manifest.hardware_profile_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise TrainingGateError(f"target manifest {name} hash is invalid")
        return manifest

    @property
    def split_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(
                [
                    {
                        "sample_id": row.sample_id,
                        "base_configuration_id": row.base_configuration_id,
                        "source_group": row.source_group,
                        "graph_signature": row.graph_signature,
                        "split": row.split,
                    }
                    for row in sorted(self.rows, key=lambda item: item.sample_id)
                ]
            ).encode("utf-8")
        ).hexdigest()


def _coarsen_once(graph: GraphIRV3, registry: OperationRegistry) -> GraphIRV3:
    record = graph.metadata.get("coarsening")
    if record is None:
        return coarsen_graph(graph, registry=registry)
    if record.get("policy") != COARSENING_POLICY_ID:
        raise ValueError("training graph uses an incompatible coarsening policy")
    if record.get("policy_sha256") != COARSENING_POLICY_SHA256:
        raise ValueError("training graph coarsening hash mismatch")
    return graph


def _classify_workload_regime(features: GraphFeaturesV3) -> int:
    """Create a hardware-independent Stage A graph-regime target."""

    global_fields = features.layout.global_continuous_fields
    node_fields = features.layout.node_continuous_fields

    def global_value(name: str) -> float:
        return float(features.u_cont[0, global_fields.index(name)])

    peak_live_bytes = global_value("peak_live_activation_bytes")
    if peak_live_bytes >= 16 * 1024**3:
        return 3  # capacity_bound on the frozen 24 GiB A10 corpus
    total_flops = global_value("total_flops")
    operation_nodes = max(1.0, global_value("operation_nodes"))
    if total_flops / operation_nodes < 1_000_000.0:
        return 0  # launch_bound
    bytes_read = float(features.x_cont[:, node_fields.index("bytes_read")].sum())
    bytes_written = float(features.x_cont[:, node_fields.index("bytes_written")].sum())
    arithmetic_intensity = total_flops / max(1.0, bytes_read + bytes_written)
    if arithmetic_intensity >= 32.0:
        return 2  # compute_bound
    return 1  # memory_bound


def materialize_training_samples(
    manifest: TrainingManifestV3,
    *,
    registry: OperationRegistry,
) -> tuple[dict[str, list[TrainingSampleV3]], Any]:
    """Load, validate, coarsen, encode, and train-only normalize a manifest."""

    raw: dict[str, list[tuple[TrainingManifestRowV3, GraphFeaturesV3]]] = {
        split: [] for split in SPLITS
    }
    for row in manifest.rows:
        graph_path = (manifest.path.parent / row.graph_path).resolve()
        graph = _coarsen_once(GraphIRV3.load(graph_path), registry)
        captured_hardware_id = graph_hardware_id(graph.metadata)
        if captured_hardware_id != manifest.target_hardware_id:
            raise TrainingGateError(
                f"graph {row.sample_id!r} targets {captured_hardware_id!r}, expected "
                f"the pair GPU {manifest.target_hardware_id!r}"
            )
        if row.hardware_id != captured_hardware_id:
            raise TrainingGateError(
                f"manifest/graph hardware mismatch for {row.sample_id!r}"
            )
        precision = graph_precision_category(graph)
        allowed_precisions = {
            str(value).removeprefix("torch.").lower()
            for value in manifest.deployment["precision_allowlist"]
        }
        if precision not in allowed_precisions:
            raise TrainingGateError(
                f"graph {row.sample_id!r} precision {precision!r} is not in the "
                "pair precision allowlist"
            )
        observed_paired_signature = str(
            graph.metadata.get("paired_base_graph_signature", graph.graph_sha256)
        )
        if row.graph_signature != observed_paired_signature:
            raise TrainingGateError(
                f"graph signature mismatch for {row.sample_id!r}: "
                f"{row.graph_signature} != {graph.graph_sha256}"
            )
        raw[row.split].append((row, build_graph_features(graph, registry=registry)))
    normalization = fit_normalization(
        [features for _, features in raw["train"]],
        split_name="train",
        split_fingerprint=manifest.split_fingerprint,
    )
    if manifest.dataset_gate.split_fingerprint != manifest.split_fingerprint:
        raise TrainingGateError("dataset gate split fingerprint does not match the manifest")
    if manifest.dataset_gate.dataset_fingerprint != manifest.dataset_fingerprint:
        raise TrainingGateError("dataset gate fingerprint does not match the manifest rows")
    materialized: dict[str, list[TrainingSampleV3]] = {split: [] for split in SPLITS}
    for split, rows in raw.items():
        for row, features in rows:
            sample = TrainingSampleV3(
                features=apply_normalization(features, normalization),
                target=torch.tensor(row.target, dtype=torch.float32),
                oom=row.oom,
                oom_stage=OOM_FAILURE_STAGES.index(row.oom_stage),
                peak_live_bytes=row.peak_live_bytes,
                domain_weight=row.domain_weight,
                source_group=row.source_group,
                graph_signature=row.graph_signature,
                workload_regime=_classify_workload_regime(features),
            )
            sample.validate()
            materialized[split].append(sample)
    return materialized, normalization


def materialize_target_training_samples(
    manifest: TargetTrainingManifestV3,
    *,
    registry: OperationRegistry,
    base_normalization: Any,
) -> dict[str, list[PairedTransferSampleV3]]:
    """Encode target graphs with the frozen base workload normalizer only."""

    if base_normalization is None:
        raise TrainingGateError("target adaptation requires embedded base workload normalization")
    if base_normalization.normalization_version != WORKLOAD_NORMALIZATION_VERSION:
        raise TrainingGateError("target adaptation workload normalization version mismatch")
    materialized: dict[str, list[PairedTransferSampleV3]] = {
        split: [] for split in SPLITS
    }
    def normalized_features(
        graph_path_value: str,
        *,
        sample_id: str,
        expected_signature: str,
    ) -> tuple[GraphFeaturesV3, GraphIRV3]:
        graph_path = (manifest.path.parent / graph_path_value).resolve()
        graph = _coarsen_once(GraphIRV3.load(graph_path), registry)
        if graph_hardware_id(graph.metadata) != manifest.target_hardware_id:
            raise TrainingGateError(
                f"target graph {sample_id!r} was captured for another GPU"
            )
        observed_paired_signature = str(
            graph.metadata.get("paired_base_graph_signature", "")
        )
        if expected_signature != observed_paired_signature:
            raise TrainingGateError(
                f"target graph signature mismatch for {sample_id!r}"
            )
        features = build_graph_features(graph, registry=registry)
        if features.metadata.get("hardware_profile_sha256") != manifest.hardware_profile_sha256:
            raise TrainingGateError(
                f"target graph {sample_id!r} hardware profile differs from manifest"
            )
        if base_normalization.split_name != "train":
            raise TrainingGateError("base workload normalization was not fit on train")
        normalized = apply_normalization(features, base_normalization)
        if normalized.metadata["normalization_sha256"] != base_normalization.sha256:
            raise TrainingGateError("target adaptation refit or changed workload normalization")
        return normalized, graph

    row_by_base_id = {
        row.base_configuration_id: row for row in manifest.rows
    }
    for row in manifest.rows:
        normalized, _ = normalized_features(
            row.graph_path,
            sample_id=row.sample_id,
            expected_signature=row.graph_signature,
        )
        sample = PairedTransferSampleV3(
            features=normalized,
            base_target=torch.tensor(row.base_target, dtype=torch.float32),
            target=torch.tensor(row.target, dtype=torch.float32),
            oom=row.oom,
            oom_stage=OOM_FAILURE_STAGES.index(row.oom_stage),
            peak_live_bytes=row.peak_live_bytes,
            domain_weight=row.domain_weight,
            regression_available=True,
            evidence_kind="paired_label",
        )
        sample.validate()
        materialized[row.split].append(sample)

    # OOM attempts and deliberate memory probes are useful classification
    # evidence but are not exact paired residual labels. A memory-probe OOM is
    # intentionally present in both retained collections, so learn from each
    # content-hashed attempt exactly once.
    retained_by_sha256 = {
        attempt.sha256: attempt
        for attempt in (*manifest.oom_attempts, *manifest.memory_probe_attempts)
    }
    for attempt_sha256, attempt in sorted(retained_by_sha256.items()):
        if attempt.status == "quarantined":
            continue
        row = row_by_base_id[attempt.base_configuration_id]
        assert attempt.graph_path is not None
        normalized, graph = normalized_features(
            attempt.graph_path,
            sample_id=attempt.paired_configuration_id,
            expected_signature=row.graph_signature,
        )
        measured_configuration_id = (
            attempt.measured_configuration_id or attempt.base_configuration_id
        )
        if measured_configuration_id != attempt.base_configuration_id:
            if graph.metadata.get("measured_configuration_id") != measured_configuration_id:
                raise TrainingGateError(
                    "changed-batch target attempt graph has another measured configuration"
                )
            measured_signature = str(
                graph.metadata.get("measured_workload_graph_signature", "")
            )
            if len(measured_signature) != 64:
                raise TrainingGateError(
                    "changed-batch target attempt lacks a measured workload graph signature"
                )
        target_values = attempt.target_targets or attempt.base_targets
        sample = PairedTransferSampleV3(
            features=normalized,
            base_target=torch.tensor(attempt.base_targets, dtype=torch.float32),
            target=torch.tensor(target_values, dtype=torch.float32),
            oom=float(attempt.status == "oom"),
            oom_stage=OOM_FAILURE_STAGES.index(attempt.failure_stage),
            peak_live_bytes=float(graph.global_features.peak_live_activation_bytes),
            domain_weight=1.0,
            regression_available=False,
            evidence_kind=(
                "memory_probe"
                if attempt.split == "memory_probe"
                else "oom_attempt"
            ),
        )
        sample.validate()
        materialized[row.split].append(sample)
    return materialized


def _batches(
    samples: Sequence[TrainingSampleV3],
    *,
    batch_size: int,
    seed: int,
) -> list[list[TrainingSampleV3]]:
    order = list(samples)
    random.Random(seed).shuffle(order)
    return [order[index : index + batch_size] for index in range(0, len(order), batch_size)]


def _optimizer(model: torch.nn.Module, config: TrainingConfigV3) -> torch.optim.Optimizer:
    name = str(config.training["optimizer"]).lower()
    if name != "adamw":
        raise ValueError(f"unsupported v3 predictor optimizer {name!r}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )


def _amp_settings(
    device: torch.device,
    amp: str,
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None]:
    if amp == "none":
        return None, None
    if device.type != "cuda":
        raise ValueError("predictor AMP is only enabled for CUDA training")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[amp]
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return dtype, scaler


def _predict(
    model: SeerNetV3,
    samples: Sequence[TrainingSampleV3],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    if not samples:
        raise TrainingGateError("prediction split cannot be empty")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    model.eval()
    predictions: list[torch.Tensor] = []
    log_variances: list[torch.Tensor] = []
    oom_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    oom_targets: list[float] = []
    regression_available: list[bool] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch = _batch_to_device(
                batch_graph_features([sample.features for sample in batch_samples]),
                device,
            )
            output = model(*graph_batch_tensors(batch))
            predictions.append(output.prediction.cpu())
            log_variances.append(output.log_variance.cpu())
            oom_logits.append(output.oom_logit.flatten().cpu())
            targets.append(torch.stack([sample.target for sample in batch_samples]))
            oom_targets.extend(sample.oom for sample in batch_samples)
            regression_available.extend(
                sample.regression_available for sample in batch_samples
            )
    return {
        "prediction": torch.cat(predictions),
        "log_variance": torch.cat(log_variances),
        "oom_logit": torch.cat(oom_logits),
        "target": torch.cat(targets),
        "oom_target": torch.tensor(oom_targets, dtype=torch.float32),
        "regression_available": torch.tensor(
            regression_available, dtype=torch.bool
        ),
    }


def _regression_validation_score(
    predictions: Mapping[str, torch.Tensor],
) -> float:
    mask = predictions["regression_available"]
    if not bool(mask.any()):
        raise TrainingGateError("validation split has no regression labels")
    target = predictions["target"][mask]
    prediction = predictions["prediction"][mask]
    positive_indexes = (0, 3, 4)
    utilization_indexes = (1, 2, 5)
    positive_mape = (
        (prediction[:, positive_indexes] - target[:, positive_indexes]).abs()
        / target[:, positive_indexes].abs().clamp_min(1e-6)
    ).mean()
    utilization_mae_fraction = (
        prediction[:, utilization_indexes] - target[:, utilization_indexes]
    ).abs().mean() / 100.0
    return float(0.5 * (positive_mape + utilization_mae_fraction))


@dataclass
class _EarlyStoppingV3:
    patience_checks: int
    minimum_delta: float
    best_metric: float = math.inf
    best_epoch: int = 0
    checks_without_improvement: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.patience_checks < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.minimum_delta < 0:
            raise ValueError("early-stopping minimum delta must be nonnegative")

    def observe(self, model: torch.nn.Module, *, metric: float, epoch: int) -> bool:
        if not math.isfinite(metric) or epoch < 1:
            raise TrainingGateError("early-stopping validation metric is invalid")
        if metric < self.best_metric - self.minimum_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.checks_without_improvement = 0
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.checks_without_improvement += 1
        return self.checks_without_improvement >= self.patience_checks

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is None:
            raise TrainingGateError("early stopping captured no validation checkpoint")
        model.load_state_dict(self.best_state, strict=True)


def _early_stopping_settings(
    config: TrainingConfigV3,
) -> tuple[_EarlyStoppingV3, int]:
    patience = int(config.training.get("early_stopping_patience_checks", 10))
    interval = int(config.training.get("validation_interval_epochs", 1))
    minimum_delta = float(config.training.get("early_stopping_min_delta", 0.0))
    if interval < 1:
        raise ValueError("validation interval must be positive")
    return _EarlyStoppingV3(patience, minimum_delta), interval


def _artifact_metadata(
    *,
    config: TrainingConfigV3,
    model: SeerNetV3,
    sample: TrainingSampleV3,
    registry: OperationRegistry,
    manifest: TrainingManifestV3,
    normalization: Any,
    calibration: Mapping[str, Any],
) -> ArtifactMetadataV3:
    deployment = manifest.deployment
    return ArtifactMetadataV3(
        model_release=config.run["model_release"],
        graph_ir_version=config.features["graph_ir_version"],
        feature_schema_version=config.features["feature_schema_version"],
        feature_schema_sha256=sample.features.layout.feature_schema_sha256,
        operator_registry_version=config.features["op_registry_version"],
        operator_registry_sha256=registry.sha256,
        ordered_feature_layout=asdict(sample.features.layout),
        normalization_sha256=normalization.sha256,
        coarsening_policy_sha256=COARSENING_POLICY_SHA256,
        target_names=TARGET_NAMES,
        target_transform=TargetTransformV3(),
        label_schema_version=config.run["label_schema_version"],
        target_hardware_id=manifest.target_hardware_id,
        hardware_allowlist=tuple(deployment["hardware_allowlist"]),
        precision_allowlist=tuple(deployment["precision_allowlist"]),
        capture_quality_allowlist=tuple(deployment["capture_quality_allowlist"]),
        optimizer_allowlist=tuple(deployment["optimizer_allowlist"]),
        scheduler_allowlist=tuple(deployment["scheduler_allowlist"]),
        training_mode_allowlist=tuple(deployment["training_mode_allowlist"]),
        dataset_fingerprint=manifest.dataset_gate.dataset_fingerprint,
        split_fingerprint=manifest.dataset_gate.split_fingerprint,
        pytorch_version=torch.__version__,
        cuda_build_version=torch.version.cuda,
        model_config=model.config.to_dict(),
        minimum_confidence=float(deployment.get("minimum_confidence", 0.2)),
        allow_ok_with_unknowns=bool(deployment.get("allow_ok_with_unknowns", False)),
        adapter_policy=model.config.adapter_policy,
        adapter_rank=model.config.adapter_rank,
        trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        base_hardware_id=manifest.target_hardware_id,
        hardware_profile_sha256=sample.features.metadata.get("hardware_profile_sha256"),
        workload_normalization_sha256=normalization.sha256,
        hardware_normalization_sha256=sample.features.metadata.get(
            "hardware_normalization_sha256"
        ),
        calibration_sha256=hashlib.sha256(
            canonical_json(dict(calibration)).encode("utf-8")
        ).hexdigest(),
    )


def assert_distillation_compatible(
    teacher_metadata: ArtifactMetadataV3,
    manifest: TrainingManifestV3,
    *,
    normalization_sha256: str,
) -> None:
    """Require a teacher modeling the same target GPU, dataset, split, and features."""

    failures: list[str] = []
    if teacher_metadata.model_release != "perfseer_v3_teacher":
        failures.append("distillation artifact is not a v3 teacher")
    if teacher_metadata.target_hardware_id != manifest.target_hardware_id:
        failures.append(
            "teacher and student target different GPU types: "
            f"{teacher_metadata.target_hardware_id!r} != {manifest.target_hardware_id!r}"
        )
    if teacher_metadata.dataset_fingerprint != manifest.dataset_gate.dataset_fingerprint:
        failures.append("teacher and student dataset fingerprints differ")
    if teacher_metadata.split_fingerprint != manifest.dataset_gate.split_fingerprint:
        failures.append("teacher and student split fingerprints differ")
    if teacher_metadata.normalization_sha256 != normalization_sha256:
        failures.append("teacher and student normalization hashes differ")
    if failures:
        raise TrainingGateError("; ".join(failures))


def _assert_base_model_compatible(
    base_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
) -> None:
    ignored = {"adapter_policy"}
    mismatches = [
        name
        for name in sorted(set(base_config) | set(target_config))
        if name not in ignored and base_config.get(name) != target_config.get(name)
    ]
    if mismatches:
        raise TrainingGateError(
            "target adapter architecture differs from its base artifact: "
            + ", ".join(mismatches)
        )


def assert_target_base_artifact_compatible(
    loaded_base: Any,
    manifest: TargetTrainingManifestV3,
    *,
    expected_release: str,
) -> None:
    """Require the exact frozen A10 artifact and normalizer selected upstream."""

    lineage = manifest.base_lineage
    expected_sha256 = (
        lineage.base_teacher_artifact_sha256
        if expected_release == "perfseer_v3_teacher"
        else lineage.base_student_artifact_sha256
    )
    metadata = loaded_base.metadata
    normalization = loaded_base.normalization
    normalization_sha256 = normalization.sha256 if normalization is not None else None
    checks = (
        (metadata.model_release == expected_release, "target adapter base artifact has the wrong model role"),
        (loaded_base.sha256 == expected_sha256, "target adapter uses a different frozen base artifact"),
        (metadata.target_hardware_id == manifest.base_hardware_id, "target manifest base GPU differs from base artifact"),
        (metadata.adapter_policy == "base", "target adapter base artifact is already adapted"),
        (metadata.dataset_fingerprint == lineage.base_dataset_fingerprint, "target adapter base artifact uses another dataset"),
        (metadata.split_fingerprint == lineage.base_split_fingerprint, "target adapter base artifact uses another grouped split"),
        (normalization_sha256 is not None, "target adaptation base artifact embeds no workload normalization"),
        (normalization_sha256 == lineage.workload_normalization_sha256, "target adapter base artifact uses another workload normalizer"),
        (metadata.normalization_sha256 == lineage.workload_normalization_sha256, "base artifact normalization metadata differs from target lineage"),
        (metadata.workload_normalization_sha256 == lineage.workload_normalization_sha256, "base artifact workload-normalization lineage differs"),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise TrainingGateError("; ".join(failures))


def assert_target_resume_compatible(
    resume_metadata: ArtifactMetadataV3,
    manifest: TargetTrainingManifestV3,
    *,
    expected_release: str,
    base_artifact_sha256: str,
    workload_normalization_sha256: str,
    adapter_policy: str,
    adapter_rank: int,
) -> None:
    """Reject a resume artifact from another role, GPU, subset, or lineage."""

    checks = (
        (resume_metadata.model_release == expected_release, "resume artifact has the wrong model role"),
        (
            resume_metadata.target_hardware_id == manifest.target_hardware_id,
            "resume artifact targets another GPU",
        ),
        (
            resume_metadata.base_hardware_id == manifest.base_hardware_id,
            "resume artifact names another base GPU",
        ),
        (
            resume_metadata.base_artifact_sha256 == base_artifact_sha256,
            "resume artifact derives from another base artifact",
        ),
        (
            resume_metadata.target_subset_sha256 == manifest.source_subset_sha256,
            "resume artifact uses another target subset",
        ),
        (
            resume_metadata.hardware_profile_sha256 == manifest.hardware_profile_sha256,
            "resume artifact uses another hardware profile",
        ),
        (
            resume_metadata.workload_normalization_sha256
            == workload_normalization_sha256,
            "resume artifact uses another workload normalizer",
        ),
        (
            resume_metadata.adapter_policy == adapter_policy,
            "resume artifact uses another adapter policy",
        ),
        (
            resume_metadata.adapter_rank == adapter_rank,
            "resume artifact uses another adapter rank",
        ),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise TrainingGateError("; ".join(failures))


def _run_target_training(
    *,
    stage: str,
    config: TrainingConfigV3,
    manifest_path: str | Path,
    output_path: str | Path,
    device_name: str,
    amp: str,
    epochs: int | None,
    base_artifact: str | Path | None,
    teacher_artifact: str | Path | None,
    resume_artifact: str | Path | None,
) -> dict[str, Any]:
    if base_artifact is None:
        raise ValueError("target adaptation requires --base-artifact")
    registry = OperationRegistry.load()
    if config.training.get("require_training_approved_registry", True) and not registry.training_approved:
        raise TrainingGateError(
            "operator registry is not training-approved from measured GPU time"
        )
    manifest = TargetTrainingManifestV3.load(manifest_path)
    loaded_base = load_checkpoint_artifact(base_artifact, registry=registry)
    expected_release = (
        "perfseer_v3_teacher"
        if stage == "target_teacher_adapter"
        else "perfseer_v3_student"
    )
    assert_target_base_artifact_compatible(
        loaded_base,
        manifest,
        expected_release=expected_release,
    )
    assert loaded_base.normalization is not None
    samples = materialize_target_training_samples(
        manifest,
        registry=registry,
        base_normalization=loaded_base.normalization,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    model_config = config.model_config(
        registry,
        samples["train"][0].features.layout,
    )
    if model_config.adapter_policy == "base":
        raise TrainingGateError("target stage cannot use the base adapter policy")
    _assert_base_model_compatible(loaded_base.model.config.to_dict(), model_config.to_dict())
    model = SeerNetV3(model_config)
    model.load_state_dict(loaded_base.model.state_dict(), strict=True)
    parameter_audit = configure_transfer_trainable_parameters(
        model, model_config.adapter_policy
    )
    reference_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    base_frozen_sha256 = parameter_checksum(model, parameter_audit["frozen_names"])
    resumed_from_sha256 = None
    if resume_artifact is not None:
        loaded_resume = load_checkpoint_artifact(resume_artifact, registry=registry)
        assert_target_resume_compatible(
            loaded_resume.metadata,
            manifest,
            expected_release=expected_release,
            base_artifact_sha256=loaded_base.sha256,
            workload_normalization_sha256=loaded_base.normalization.sha256,
            adapter_policy=model_config.adapter_policy,
            adapter_rank=model_config.adapter_rank,
        )
        _assert_base_model_compatible(
            loaded_resume.model.config.to_dict(), model_config.to_dict()
        )
        model.load_state_dict(loaded_resume.model.state_dict(), strict=True)
        if parameter_checksum(model, parameter_audit["frozen_names"]) != base_frozen_sha256:
            raise TrainingGateError("resume artifact changed frozen base parameters")
        resumed_from_sha256 = loaded_resume.sha256
    model = model.to(device)
    frozen_before = parameter_checksum(model, parameter_audit["frozen_names"])
    optimizer = _optimizer(model, config)
    autocast_dtype, scaler = _amp_settings(device, amp)
    total_epochs = int(config.training["epochs"] if epochs is None else epochs)
    if total_epochs <= 0:
        raise ValueError("target adaptation epochs must be positive")
    batch_size = int(config.training["batch_size"])
    seed = int(config.run["seed"])
    residual_transform = PairedResidualTransformV3(
        epsilon=float(config.training.get("residual_epsilon", 1e-6)),
        utilization_clip=float(config.training.get("utilization_clip", 0.005)),
    )
    teacher = None
    if stage == "target_student_adapter":
        if teacher_artifact is None:
            raise ValueError("target student adaptation requires --teacher-artifact")
        loaded_teacher = load_checkpoint_artifact(teacher_artifact, registry=registry)
        teacher_metadata = loaded_teacher.metadata
        failures = []
        if teacher_metadata.model_release != "perfseer_v3_teacher":
            failures.append("target student teacher artifact has the wrong role")
        if teacher_metadata.target_hardware_id != manifest.target_hardware_id:
            failures.append("target teacher and student GPU IDs differ")
        if teacher_metadata.target_subset_sha256 != manifest.source_subset_sha256:
            failures.append("target teacher and student subsets differ")
        if teacher_metadata.hardware_profile_sha256 != manifest.hardware_profile_sha256:
            failures.append("target teacher and student hardware profiles differ")
        if teacher_metadata.workload_normalization_sha256 != loaded_base.normalization.sha256:
            failures.append("target teacher and student workload normalizers differ")
        if (
            teacher_metadata.base_artifact_sha256
            != manifest.base_lineage.base_teacher_artifact_sha256
        ):
            failures.append("target teacher derives from another frozen base teacher")
        if failures:
            raise TrainingGateError("; ".join(failures))
        teacher = loaded_teacher.model.to(device).eval()

    started = time.perf_counter()
    losses: list[float] = []
    early_stopping, validation_interval = _early_stopping_settings(config)
    epochs_completed = 0
    early_stopped = False
    for epoch in range(total_epochs):
        model.train()
        for batch in _batches(
            samples["train"], batch_size=batch_size, seed=seed + epoch
        ):
            if stage == "target_teacher_adapter":
                losses.append(
                    target_teacher_adapter_step(
                        model,
                        batch,
                        optimizer,
                        residual_transform=residual_transform,
                        adapter_regularization_weight=float(
                            config.training.get("adapter_regularization_weight", 1e-4)
                        ),
                        l2_sp_weight=float(config.training.get("l2_sp_weight", 1e-4)),
                        reference_parameters=reference_parameters,
                        autocast_dtype=autocast_dtype,
                        scaler=scaler,
                    )
                )
            else:
                assert teacher is not None
                losses.append(
                    target_student_adapter_step(
                        model,
                        teacher,
                        batch,
                        optimizer,
                        hard_label_weight=float(
                            config.training.get("hard_label_weight", 0.6)
                        ),
                        representation_weight=float(
                            config.training.get(
                                "representation_distillation_weight", 0.05
                            )
                        ),
                        residual_transform=residual_transform,
                        adapter_regularization_weight=float(
                            config.training.get("adapter_regularization_weight", 1e-4)
                        ),
                        l2_sp_weight=float(
                            config.training.get("l2_sp_weight", 1e-4)
                        ),
                        reference_parameters=reference_parameters,
                        autocast_dtype=autocast_dtype,
                        scaler=scaler,
                    )
                )
        epochs_completed = epoch + 1
        if epochs_completed % validation_interval == 0 or epochs_completed == total_epochs:
            validation_metric = _regression_validation_score(
                _predict(
                    model,
                    samples["validation"],
                    device=device,
                    batch_size=batch_size,
                )
            )
            if early_stopping.observe(
                model,
                metric=validation_metric,
                epoch=epochs_completed,
            ):
                early_stopped = True
                break
    early_stopping.restore(model)
    frozen_after = parameter_checksum(model, parameter_audit["frozen_names"])
    if frozen_after != frozen_before:
        raise TrainingGateError("frozen workload/base parameters changed during target adaptation")

    validation = _predict(model, samples["validation"], device=device)
    regression_mask = validation["regression_available"]
    if not bool(regression_mask.any()):
        raise TrainingGateError(
            "target validation has no exact paired labels for regression calibration"
        )
    regression_prediction = validation["prediction"][regression_mask]
    regression_target = validation["target"][regression_mask]
    regression_log_variance = validation["log_variance"][regression_mask]
    linear = fit_linear_calibration(regression_prediction, regression_target)
    oom = fit_binary_temperature_calibration(
        validation["oom_logit"], validation["oom_target"]
    )
    uncertainty = fit_uncertainty_calibration(
        regression_prediction,
        regression_target,
        regression_log_variance,
    )
    calibration_payload = {
        "slope": linear.slope,
        "intercept": linear.intercept,
        "oom_temperature": oom.temperature,
        "uncertainty_log_variance_offset": uncertainty.log_variance_offset,
        "fit_split": "validation",
        "target_specific": True,
    }
    calibration_sha256 = hashlib.sha256(
        canonical_json(calibration_payload).encode("utf-8")
    ).hexdigest()
    lineage = TransferLineageV3(
        base_artifact_sha256=loaded_base.sha256,
        base_hardware_id=manifest.base_hardware_id,
        target_hardware_id=manifest.target_hardware_id,
        target_subset_sha256=manifest.source_subset_sha256,
        hardware_profile_sha256=manifest.hardware_profile_sha256,
        workload_normalization_sha256=loaded_base.normalization.sha256,
        hardware_normalization_sha256=HardwareNormalizationPolicyV3().sha256,
        calibration_sha256=calibration_sha256,
        paired_residual_policy_sha256=residual_transform.policy_sha256,
        adapter_policy=model_config.adapter_policy,
        adapter_rank=model_config.adapter_rank,
        trainable_parameter_count=int(parameter_audit["trainable_parameter_count"]),
    )
    lineage.validate()
    deployment = manifest.deployment
    metadata = ArtifactMetadataV3(
        model_release=expected_release,
        graph_ir_version=config.features["graph_ir_version"],
        feature_schema_version=config.features["feature_schema_version"],
        feature_schema_sha256=samples["train"][0].features.layout.feature_schema_sha256,
        operator_registry_version=config.features["op_registry_version"],
        operator_registry_sha256=registry.sha256,
        ordered_feature_layout=asdict(samples["train"][0].features.layout),
        normalization_sha256=loaded_base.normalization.sha256,
        coarsening_policy_sha256=COARSENING_POLICY_SHA256,
        target_names=TARGET_NAMES,
        target_transform=TargetTransformV3(),
        label_schema_version=config.run["label_schema_version"],
        target_hardware_id=manifest.target_hardware_id,
        hardware_allowlist=(manifest.target_hardware_id,),
        precision_allowlist=tuple(deployment["precision_allowlist"]),
        capture_quality_allowlist=tuple(deployment["capture_quality_allowlist"]),
        optimizer_allowlist=tuple(
            canonical_optimizer_name(value)
            for value in deployment["optimizer_allowlist"]
        ),
        scheduler_allowlist=tuple(
            canonical_scheduler_name(value)
            for value in deployment["scheduler_allowlist"]
        ),
        training_mode_allowlist=tuple(deployment["training_mode_allowlist"]),
        dataset_fingerprint=manifest.manifest_sha256,
        split_fingerprint=manifest.split_fingerprint,
        pytorch_version=torch.__version__,
        cuda_build_version=torch.version.cuda,
        model_config=model_config.to_dict(),
        minimum_confidence=float(deployment.get("minimum_confidence", 0.2)),
        allow_ok_with_unknowns=bool(deployment.get("allow_ok_with_unknowns", False)),
        adapter_policy=lineage.adapter_policy,
        adapter_rank=lineage.adapter_rank,
        trainable_parameter_count=lineage.trainable_parameter_count,
        base_artifact_sha256=lineage.base_artifact_sha256,
        base_hardware_id=lineage.base_hardware_id,
        target_subset_sha256=lineage.target_subset_sha256,
        hardware_profile_sha256=lineage.hardware_profile_sha256,
        workload_normalization_sha256=lineage.workload_normalization_sha256,
        hardware_normalization_sha256=lineage.hardware_normalization_sha256,
        calibration_sha256=lineage.calibration_sha256,
        paired_residual_policy_sha256=lineage.paired_residual_policy_sha256,
        transfer_lineage_sha256=lineage.sha256,
    )
    artifact_path = save_checkpoint_artifact(
        output_path,
        model=model.cpu().eval(),
        metadata=metadata,
        normalization=loaded_base.normalization,
        calibration=calibration_payload,
    )
    denominator = regression_target.abs().clamp_min(1e-6)
    report = {
        "report_version": "perfseer_v3_transfer_training_run_v1",
        "stage": stage,
        "status": "completed",
        "target_hardware_id": manifest.target_hardware_id,
        "base_hardware_id": manifest.base_hardware_id,
        "label_budget": manifest.label_budget,
        "target_subset_sha256": manifest.source_subset_sha256,
        "base_lineage": manifest.base_lineage.to_dict(),
        "hardware_profile_sha256": manifest.hardware_profile_sha256,
        "workload_normalization_sha256": loaded_base.normalization.sha256,
        "hardware_normalization_sha256": lineage.hardware_normalization_sha256,
        "parameter_freeze_audit": parameter_audit,
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "transfer_lineage_sha256": lineage.sha256,
        "resumed_from_artifact_sha256": resumed_from_sha256,
        "epochs": total_epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "best_validation_epoch": early_stopping.best_epoch,
        "best_validation_composite_error": early_stopping.best_metric,
        "validation_interval_epochs": validation_interval,
        "early_stopping_patience_checks": early_stopping.patience_checks,
        "batch_size": batch_size,
        "device": str(device),
        "amp": amp,
        "elapsed_seconds": time.perf_counter() - started,
        "train_loss_last": losses[-1] if losses else None,
        "validation_mape_near_zero_floor_1e-6": float(
            ((regression_prediction - regression_target).abs() / denominator).mean()
        ),
        "sample_counts": {split: len(values) for split, values in samples.items()},
        "paired_regression_sample_counts": {
            split: sum(sample.regression_available for sample in values)
            for split, values in samples.items()
        },
        "oom_positive_sample_counts": {
            split: sum(sample.oom == 1.0 for sample in values)
            for split, values in samples.items()
        },
        "evidence_kind_counts": {
            split: {
                kind: sum(sample.evidence_kind == kind for sample in values)
                for kind in ("paired_label", "oom_attempt", "memory_probe")
            }
            for split, values in samples.items()
        },
        "retained_oom_attempt_count": len(manifest.oom_attempts),
        "retained_memory_probe_attempt_count": len(manifest.memory_probe_attempts),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "production_accuracy_gates_evaluated": False,
    }
    report_path = artifact_path.with_suffix(artifact_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report


def run_training(
    *,
    stage: str,
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    device_name: str,
    amp: str = "none",
    epochs: int | None = None,
    pretrain_epochs: int | None = None,
    teacher_artifact: str | Path | None = None,
    base_artifact: str | Path | None = None,
    resume_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Run a gated teacher or student job and save a self-describing artifact."""

    aliases = {"teacher": "base_teacher", "student": "base_student"}
    stage = aliases.get(stage, stage)
    if stage not in {
        "base_teacher",
        "base_student",
        "target_teacher_adapter",
        "target_student_adapter",
    }:
        raise ValueError("unsupported PerfSeer v3 training stage")
    config = TrainingConfigV3.load(config_path)
    if config.training.get("enabled", True) is not True:
        raise TrainingGateError("requested training ablation is disabled by default")
    expected_stage = {
        "base_teacher": "base_teacher",
        "base_student": "base_student",
        "target_teacher_adapter": "target_teacher_adapter",
        "target_student_adapter": "target_student_adapter",
    }[stage]
    legacy_expected = {
        "base_teacher": "teacher",
        "base_student": "student_distillation",
    }.get(stage)
    if config.training.get("stage") not in {expected_stage, legacy_expected}:
        raise ValueError("requested stage does not match the training config")
    if stage.startswith("target_"):
        if pretrain_epochs not in (None, 0):
            raise ValueError("target adaptation cannot pretrain the workload backbone")
        return _run_target_training(
            stage=stage,
            config=config,
            manifest_path=manifest_path,
            output_path=output_path,
            device_name=device_name,
            amp=amp,
            epochs=epochs,
            base_artifact=base_artifact,
            teacher_artifact=teacher_artifact,
            resume_artifact=resume_artifact,
        )
    if resume_artifact is not None:
        raise ValueError("--resume-artifact is supported only for target adaptation")
    configured_pretrain_epochs = int(config.training.get("pretrain_epochs", 0))
    effective_pretrain_epochs = (
        configured_pretrain_epochs if pretrain_epochs is None else int(pretrain_epochs)
    )
    if stage != "base_teacher" and effective_pretrain_epochs:
        raise ValueError("only base_teacher may run workload-backbone pretraining")
    registry = OperationRegistry.load()
    manifest = TrainingManifestV3.load(manifest_path)
    assert_training_ready(config, manifest.dataset_gate, registry)
    assert_base_campaign_ready(manifest, registry)
    samples, normalization = materialize_training_samples(manifest, registry=registry)
    assert_base_campaign_ready(
        manifest,
        registry,
        feature_schema_sha256=samples["train"][0].features.layout.feature_schema_sha256,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    model_config = config.model_config(registry, samples["train"][0].features.layout)
    model = SeerNetV3(model_config).to(device)
    parameter_audit = model.configure_trainable_parameters("base")
    autocast_dtype, scaler = _amp_settings(device, amp)
    total_epochs = int(config.training["epochs"] if epochs is None else epochs)
    if total_epochs <= 0 or effective_pretrain_epochs < 0:
        raise ValueError("training epochs must be positive and pretraining nonnegative")
    if stage == "base_teacher" and effective_pretrain_epochs == 0:
        raise TrainingGateError("base teacher requires positive Stage A pretraining epochs")
    batch_size = int(config.training["batch_size"])
    seed = int(config.run["seed"])
    started = time.perf_counter()
    losses: list[float] = []
    pretrain_losses: list[float] = []
    early_stopping, validation_interval = _early_stopping_settings(config)
    epochs_completed = 0
    early_stopped = False

    if stage == "base_teacher":
        if effective_pretrain_epochs:
            pretrainer = EncoderPretrainer(model).to(device)
            pretrain_optimizer = _optimizer(pretrainer, config)
            # Stage A is label-free and intentionally uses all 18K frozen A10
            # workload graphs. Target GPU labels and hardware tensors never
            # enter EncoderPretrainer.
            pretraining_samples = [
                sample
                for split in SPLITS
                for sample in samples[split]
            ]
            for epoch in range(effective_pretrain_epochs):
                for batch in _batches(
                    pretraining_samples,
                    batch_size=batch_size,
                    seed=seed + epoch,
                ):
                    pretrain_losses.append(
                        encoder_pretrain_step(
                            pretrainer,
                            batch,
                            pretrain_optimizer,
                            autocast_dtype=autocast_dtype,
                            scaler=scaler,
                        )
                    )
            # Release Stage A Adam moments before allocating the full Stage B
            # optimizer; otherwise T1 briefly holds both optimizer states.
            del pretrain_optimizer
            del pretrainer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        optimizer = _optimizer(model, config)
        for epoch in range(total_epochs):
            model.train()
            for batch in _batches(samples["train"], batch_size=batch_size, seed=seed + epoch):
                losses.append(
                    teacher_train_step(
                        model,
                        batch,
                        optimizer,
                        autocast_dtype=autocast_dtype,
                        scaler=scaler,
                    )
                )
            epochs_completed = epoch + 1
            if (
                epochs_completed % validation_interval == 0
                or epochs_completed == total_epochs
            ):
                validation_metric = _regression_validation_score(
                    _predict(
                        model,
                        samples["validation"],
                        device=device,
                        batch_size=batch_size,
                    )
                )
                if early_stopping.observe(
                    model,
                    metric=validation_metric,
                    epoch=epochs_completed,
                ):
                    early_stopped = True
                    break
    else:
        if teacher_artifact is None:
            raise ValueError("student distillation requires --teacher-artifact")
        loaded_teacher = load_checkpoint_artifact(teacher_artifact, registry=registry)
        assert_distillation_compatible(
            loaded_teacher.metadata,
            manifest,
            normalization_sha256=normalization.sha256,
        )
        teacher = loaded_teacher.model.to(device).eval()
        optimizer = _optimizer(model, config)
        for epoch in range(total_epochs):
            model.train()
            for batch in _batches(samples["train"], batch_size=batch_size, seed=seed + epoch):
                losses.append(
                    student_distill_step(
                        model,
                        teacher,
                        batch,
                        optimizer,
                        hard_label_weight=float(config.training["hard_label_weight"]),
                        representation_weight=float(
                            config.training.get("representation_distillation_weight", 0.0)
                        ),
                        autocast_dtype=autocast_dtype,
                        scaler=scaler,
                    )
                )
            epochs_completed = epoch + 1
            if (
                epochs_completed % validation_interval == 0
                or epochs_completed == total_epochs
            ):
                validation_metric = _regression_validation_score(
                    _predict(
                        model,
                        samples["validation"],
                        device=device,
                        batch_size=batch_size,
                    )
                )
                if early_stopping.observe(
                    model,
                    metric=validation_metric,
                    epoch=epochs_completed,
                ):
                    early_stopped = True
                    break

    early_stopping.restore(model)
    validation = _predict(
        model,
        samples["validation"],
        device=device,
        batch_size=batch_size,
    )
    calibration = fit_linear_calibration(
        validation["prediction"], validation["target"]
    )
    oom_calibration = fit_binary_temperature_calibration(
        validation["oom_logit"], validation["oom_target"]
    )
    uncertainty_calibration = fit_uncertainty_calibration(
        validation["prediction"],
        validation["target"],
        validation["log_variance"],
    )
    denominator = validation["target"].abs().clamp_min(1e-6)
    validation_mape = float(
        ((validation["prediction"] - validation["target"]).abs() / denominator).mean()
    )
    calibration_payload = {
        "slope": calibration.slope,
        "intercept": calibration.intercept,
        "oom_temperature": oom_calibration.temperature,
        "uncertainty_log_variance_offset": (
            uncertainty_calibration.log_variance_offset
        ),
        "fit_split": "validation",
    }
    metadata = _artifact_metadata(
        config=config,
        model=model,
        sample=samples["train"][0],
        registry=registry,
        manifest=manifest,
        normalization=normalization,
        calibration=calibration_payload,
    )
    artifact_path = save_checkpoint_artifact(
        output_path,
        model=model.cpu().eval(),
        metadata=metadata,
        normalization=normalization,
        calibration=calibration_payload,
    )
    elapsed = time.perf_counter() - started
    report = {
        "report_version": "perfseer_v3_training_run_v2",
        "stage": stage,
        "status": "completed",
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config.sha256,
        "manifest_path": str(manifest.path),
        "dataset_fingerprint": manifest.dataset_gate.dataset_fingerprint,
        "target_hardware_id": manifest.target_hardware_id,
        "split_fingerprint": manifest.dataset_gate.split_fingerprint,
        "feature_schema_sha256": metadata.feature_schema_sha256,
        "operator_registry_sha256": registry.sha256,
        "normalization_sha256": normalization.sha256,
        "epochs": total_epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "best_validation_epoch": early_stopping.best_epoch,
        "best_validation_composite_error": early_stopping.best_metric,
        "validation_interval_epochs": validation_interval,
        "early_stopping_patience_checks": early_stopping.patience_checks,
        "pretrain_epochs": effective_pretrain_epochs,
        "pretraining_sample_count": (
            sum(len(samples[split]) for split in SPLITS)
            if effective_pretrain_epochs
            else 0
        ),
        "pretraining_objectives": (
            list(PRETRAINING_OBJECTIVES) if effective_pretrain_epochs else []
        ),
        "pretraining_hardware_inputs_consumed": False,
        "batch_size": batch_size,
        "device": str(device),
        "amp": amp,
        "elapsed_seconds": elapsed,
        "train_loss_last": losses[-1] if losses else None,
        "pretrain_loss_last": pretrain_losses[-1] if pretrain_losses else None,
        "validation_mape_near_zero_floor_1e-6": validation_mape,
        "sample_counts": {split: len(values) for split, values in samples.items()},
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "pytorch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "production_accuracy_gates_evaluated": False,
        "parameter_freeze_audit": parameter_audit,
    }
    report_path = artifact_path.with_suffix(artifact_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report


def run_smoke(*, output_path: str | Path, seed: int = 42) -> dict[str, Any]:
    """Run a local non-production capture→encode→teacher→student smoke test."""

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(value))

    capture = capture_export(
        Tiny(),
        (torch.randn(2, 4),),
        options=CaptureOptions(target_hardware_id="local_smoke_gpu"),
    )
    if not capture.success or capture.graph is None:
        raise RuntimeError(f"smoke capture failed: {capture.failures}")
    registry = OperationRegistry.load()
    features = build_graph_features(coarsen_graph(capture.graph, registry=registry), registry=registry)
    sample = TrainingSampleV3(features=features, target=torch.ones(len(TARGET_NAMES)))
    result = run_tiny_training_smoke([sample], seed=seed)
    report = {
        "report_version": "perfseer_v3_training_smoke_v1",
        "status": "smoke_only_not_production",
        "pretrain_loss": result.pretrain_loss,
        "teacher_loss": result.teacher_loss,
        "student_loss": result.student_loss,
        "feature_schema_sha256": features.layout.feature_schema_sha256,
        "operator_registry_sha256": registry.sha256,
        "pytorch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_path": str(output.resolve())}


__all__ = [
    "SPLITS",
    "TRAINING_MANIFEST_VERSION",
    "TrainingManifestRowV3",
    "TrainingManifestV3",
    "assert_base_campaign_ready",
    "assert_distillation_compatible",
    "assert_target_base_artifact_compatible",
    "assert_target_resume_compatible",
    "materialize_training_samples",
    "run_smoke",
    "run_training",
]
