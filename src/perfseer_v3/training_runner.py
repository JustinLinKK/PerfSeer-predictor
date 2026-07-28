"""Fail-closed v3 teacher training and teacher-to-student distillation runner."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
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
    canonical_hardware_id,
    graph_hardware_id,
    require_specific_hardware_id,
)
from .model import OOM_FAILURE_STAGES, SeerNetV3, graph_batch_tensors
from .op_registry import OperationRegistry
from .training import (
    DatasetGateReport,
    EncoderPretrainer,
    TARGET_NAMES,
    TrainingConfigV3,
    TrainingGateError,
    TrainingSampleV3,
    _batch_to_device,
    assert_training_ready,
    encoder_pretrain_step,
    fit_binary_temperature_calibration,
    fit_linear_calibration,
    fit_uncertainty_calibration,
    run_tiny_training_smoke,
    student_distill_step,
    teacher_train_step,
)
from .training_semantics import canonical_optimizer_name, canonical_scheduler_name


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
        manifest = cls(manifest_path, gate, deployment, rows)
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


def _coarsen_once(graph: GraphIRV3, registry: OperationRegistry) -> GraphIRV3:
    record = graph.metadata.get("coarsening")
    if record is None:
        return coarsen_graph(graph, registry=registry)
    if record.get("policy") != COARSENING_POLICY_ID:
        raise ValueError("training graph uses an incompatible coarsening policy")
    if record.get("policy_sha256") != COARSENING_POLICY_SHA256:
        raise ValueError("training graph coarsening hash mismatch")
    return graph


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
        if row.graph_signature != graph.graph_sha256:
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
            )
            sample.validate()
            materialized[split].append(sample)
    return materialized, normalization


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
) -> dict[str, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    log_variances: list[torch.Tensor] = []
    oom_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    oom_targets: list[float] = []
    with torch.no_grad():
        for sample in samples:
            batch = _batch_to_device(batch_graph_features([sample.features]), device)
            output = model(*graph_batch_tensors(batch))
            predictions.append(output.prediction.cpu())
            log_variances.append(output.log_variance.cpu())
            oom_logits.append(output.oom_logit.flatten().cpu())
            targets.append(sample.target.unsqueeze(0))
            oom_targets.append(sample.oom)
    return {
        "prediction": torch.cat(predictions),
        "log_variance": torch.cat(log_variances),
        "oom_logit": torch.cat(oom_logits),
        "target": torch.cat(targets),
        "oom_target": torch.tensor(oom_targets, dtype=torch.float32),
    }


def _artifact_metadata(
    *,
    config: TrainingConfigV3,
    model: SeerNetV3,
    sample: TrainingSampleV3,
    registry: OperationRegistry,
    manifest: TrainingManifestV3,
    normalization: Any,
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


def run_training(
    *,
    stage: str,
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    device_name: str,
    amp: str = "none",
    epochs: int | None = None,
    pretrain_epochs: int = 0,
    teacher_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Run a gated teacher or student job and save a self-describing artifact."""

    if stage not in {"teacher", "student"}:
        raise ValueError("stage must be teacher or student")
    config = TrainingConfigV3.load(config_path)
    expected_stage = "teacher" if stage == "teacher" else "student_distillation"
    if config.training.get("stage") != expected_stage:
        raise ValueError("requested stage does not match the training config")
    registry = OperationRegistry.load()
    manifest = TrainingManifestV3.load(manifest_path)
    assert_training_ready(config, manifest.dataset_gate, registry)
    samples, normalization = materialize_training_samples(manifest, registry=registry)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    model_config = config.model_config(registry, samples["train"][0].features.layout)
    model = SeerNetV3(model_config).to(device)
    autocast_dtype, scaler = _amp_settings(device, amp)
    total_epochs = int(config.training["epochs"] if epochs is None else epochs)
    if total_epochs <= 0 or pretrain_epochs < 0:
        raise ValueError("training epochs must be positive and pretraining nonnegative")
    batch_size = int(config.training["batch_size"])
    seed = int(config.run["seed"])
    started = time.perf_counter()
    losses: list[float] = []
    pretrain_losses: list[float] = []

    if stage == "teacher":
        if pretrain_epochs:
            pretrainer = EncoderPretrainer(model).to(device)
            pretrain_optimizer = _optimizer(pretrainer, config)
            for epoch in range(pretrain_epochs):
                for batch in _batches(samples["train"], batch_size=batch_size, seed=seed + epoch):
                    pretrain_losses.append(
                        encoder_pretrain_step(
                            pretrainer,
                            batch,
                            pretrain_optimizer,
                            autocast_dtype=autocast_dtype,
                            scaler=scaler,
                        )
                    )
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

    validation = _predict(model, samples["validation"], device=device)
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
    metadata = _artifact_metadata(
        config=config,
        model=model,
        sample=samples["train"][0],
        registry=registry,
        manifest=manifest,
        normalization=normalization,
    )
    artifact_path = save_checkpoint_artifact(
        output_path,
        model=model.cpu().eval(),
        metadata=metadata,
        normalization=normalization,
        calibration={
            "slope": calibration.slope,
            "intercept": calibration.intercept,
            "oom_temperature": oom_calibration.temperature,
            "uncertainty_log_variance_offset": (
                uncertainty_calibration.log_variance_offset
            ),
            "fit_split": "validation",
        },
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
        "pretrain_epochs": pretrain_epochs,
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
    "assert_distillation_compatible",
    "materialize_training_samples",
    "run_smoke",
    "run_training",
]
