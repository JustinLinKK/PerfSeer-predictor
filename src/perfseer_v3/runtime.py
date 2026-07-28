"""Fail-closed CPU runtime and scheduler result contract for PerfSeer v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .artifact import LoadedArtifactV3, load_checkpoint_artifact
from .coarsen_v3 import COARSENING_POLICY_SHA256, coarsen_graph
from .features import (
    apply_normalization,
    batch_graph_features,
    build_graph_features,
    graph_precision_category,
)
from .graph_ir_v3 import GraphIRV3
from .hardware import graph_hardware_id
from .model import OOM_FAILURE_STAGES, graph_batch_tensors
from .op_registry import OperationRegistry
from .training_semantics import (
    canonical_optimizer_name,
    canonical_scheduler_name,
    scheduler_config,
)


RESULT_STATUSES = (
    "ok",
    "ok_with_unknowns",
    "ood_low_confidence",
    "unsupported_capture",
    "hardware_mismatch",
    "unsupported_precision",
    "unsupported_training_mode",
    "unsupported_optimizer",
    "unsupported_scheduler",
    "schema_mismatch",
    "encoder_error",
)


@dataclass(frozen=True)
class SchedulerPredictionV3:
    status: str
    prediction: tuple[float, ...] | None
    uncertainty: tuple[float, ...] | None
    oom_probability: float | None
    oom_failure_stage: str | None
    peak_live_bytes: float | None
    confidence: float | None
    unknown_gpu_cost_proxy_fraction: float
    capture_mode: str
    capture_quality: str
    graph_ir_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    operator_registry_version: str
    operator_registry_sha256: str
    output_contract_version: str
    recommended_fallback: str | None
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"invalid scheduler result status {self.status!r}")
        if self.status == "ok" and self.recommended_fallback is not None:
            raise ValueError("ok result cannot recommend fallback")
        if self.status != "ok" and self.recommended_fallback is None:
            raise ValueError("every non-ok result must recommend fallback")


def _failure(
    graph: GraphIRV3,
    status: str,
    message: str,
) -> SchedulerPredictionV3:
    return SchedulerPredictionV3(
        status=status,
        prediction=None,
        uncertainty=None,
        oom_probability=None,
        oom_failure_stage=None,
        peak_live_bytes=None,
        confidence=None,
        unknown_gpu_cost_proxy_fraction=graph.global_features.unknown_cost_fraction,
        capture_mode=graph.capture_mode,
        capture_quality=graph.coverage.capture_quality,
        graph_ir_version=graph.graph_ir_version,
        feature_schema_version=graph.feature_schema_version,
        feature_schema_sha256=graph.feature_schema_sha256,
        operator_registry_version=graph.operator_registry_version,
        operator_registry_sha256=graph.operator_registry_sha256,
        output_contract_version="perfseer_v3_outputs_v2",
        recommended_fallback="branch_profile",
        message=message,
    )


class PerfSeerV3Runtime:
    def __init__(
        self,
        artifact: str | Any,
        *,
        registry: OperationRegistry | None = None,
    ) -> None:
        self.registry = registry or OperationRegistry.load()
        self.artifact = (
            artifact
            if isinstance(artifact, LoadedArtifactV3)
            else load_checkpoint_artifact(artifact, registry=self.registry)
        )

    def _decode_target(
        self,
        prediction: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transform = self.artifact.metadata.target_transform
        mean = prediction.new_tensor(transform.mean)
        std = prediction.new_tensor(transform.std)
        decoded = prediction * std + mean
        uncertainty = torch.exp(0.5 * log_variance) * std
        if transform.transform == "log1p_standardized":
            decoded = torch.expm1(decoded).clamp_min(0.0)
            uncertainty = uncertainty * torch.exp((prediction * std + mean).clamp(max=30.0))
        calibration = self.artifact.calibration
        if calibration:
            uncertainty_offset = decoded.new_tensor(
                calibration.get(
                    "uncertainty_log_variance_offset",
                    [0.0] * decoded.size(-1),
                )
            )
            uncertainty = uncertainty * torch.exp(0.5 * uncertainty_offset)
            slope = decoded.new_tensor(calibration.get("slope", [1.0] * decoded.size(-1)))
            intercept = decoded.new_tensor(calibration.get("intercept", [0.0] * decoded.size(-1)))
            decoded = decoded * slope + intercept
            uncertainty = uncertainty * slope.abs()
        return decoded, uncertainty

    def predict_graph(self, graph: GraphIRV3) -> SchedulerPredictionV3:
        metadata = self.artifact.metadata
        if (
            graph.graph_ir_version != metadata.graph_ir_version
            or graph.feature_schema_version != metadata.feature_schema_version
            or graph.feature_schema_sha256 != metadata.feature_schema_sha256
            or graph.operator_registry_version != metadata.operator_registry_version
            or graph.operator_registry_sha256 != metadata.operator_registry_sha256
        ):
            return _failure(graph, "schema_mismatch", "graph/artifact schema or registry mismatch")
        if graph.coverage.capture_quality not in metadata.capture_quality_allowlist:
            return _failure(
                graph,
                "unsupported_capture",
                f"capture quality {graph.coverage.capture_quality!r} is not allowed",
            )
        captured_hardware_id = graph_hardware_id(graph.metadata)
        if captured_hardware_id != metadata.target_hardware_id:
            return _failure(
                graph,
                "hardware_mismatch",
                f"graph targets GPU {captured_hardware_id!r}, but this model pair targets "
                f"{metadata.target_hardware_id!r}",
            )
        precision = graph_precision_category(graph)
        allowed_precisions = {
            str(value).removeprefix("torch.").lower()
            for value in metadata.precision_allowlist
        }
        if precision not in allowed_precisions:
            return _failure(
                graph,
                "unsupported_precision",
                f"graph precision policy {precision!r} is not allowed for this GPU pair",
            )
        requested_mode = "training" if graph.training_mode else "inference"
        if requested_mode not in metadata.training_mode_allowlist:
            return _failure(
                graph,
                "unsupported_training_mode",
                f"training mode {requested_mode!r} is not allowed",
            )
        optimizer = canonical_optimizer_name(graph.optimizer_config.get("name", "none"))
        if graph.training_mode and optimizer not in metadata.optimizer_allowlist:
            return _failure(
                graph,
                "unsupported_optimizer",
                f"optimizer {optimizer!r} is not allowed",
            )
        scheduler = canonical_scheduler_name(
            scheduler_config(graph.training_config).get("name", "none")
        )
        if graph.training_mode and scheduler not in metadata.scheduler_allowlist:
            return _failure(
                graph,
                "unsupported_scheduler",
                f"learning-rate scheduler {scheduler!r} is not allowed",
            )
        try:
            view = graph
            if "coarsening" not in graph.metadata:
                view = coarsen_graph(graph, registry=self.registry)
            features = build_graph_features(view, registry=self.registry)
            if self.artifact.normalization is not None:
                features = apply_normalization(features, self.artifact.normalization)
            batch = batch_graph_features([features])
            with torch.inference_mode():
                output = self.artifact.model(*graph_batch_tensors(batch))
            prediction, uncertainty = self._decode_target(
                output.prediction[0],
                output.log_variance[0],
            )
            confidence = float(output.confidence[0, 0])
            oom_temperature = float(
                self.artifact.calibration.get("oom_temperature", 1.0)
            )
            if oom_temperature <= 0:
                raise RuntimeError("artifact OOM calibration temperature must be positive")
            oom_probability = float(
                torch.sigmoid(output.oom_logit[0, 0] / oom_temperature)
            )
            oom_stage_index = int(output.oom_stage_logits[0].argmax())
            oom_failure_stage = OOM_FAILURE_STAGES[oom_stage_index]
            peak_live_bytes = float(
                torch.expm1(output.peak_live_bytes_log1p[0, 0].clamp(max=50.0))
            )
            if not torch.isfinite(prediction).all() or not torch.isfinite(uncertainty).all():
                raise RuntimeError("model produced nonfinite prediction/uncertainty")
        except Exception as exc:
            return _failure(graph, "encoder_error", f"{type(exc).__name__}: {exc}")
        unknown_fraction = graph.global_features.unknown_cost_fraction
        if confidence < metadata.minimum_confidence:
            status = "ood_low_confidence"
        elif unknown_fraction > 0:
            status = "ok_with_unknowns"
        else:
            status = "ok"
        if status == "ok":
            fallback = None
        elif status == "ok_with_unknowns" and metadata.allow_ok_with_unknowns:
            # The result state remains non-OK for observability; policy may
            # consume it, but the contract still carries an explicit fallback.
            fallback = "branch_profile"
        else:
            fallback = "branch_profile"
        return SchedulerPredictionV3(
            status=status,
            prediction=tuple(float(value) for value in prediction),
            uncertainty=tuple(float(value) for value in uncertainty),
            oom_probability=oom_probability,
            oom_failure_stage=oom_failure_stage,
            peak_live_bytes=peak_live_bytes,
            confidence=confidence,
            unknown_gpu_cost_proxy_fraction=unknown_fraction,
            capture_mode=graph.capture_mode,
            capture_quality=graph.coverage.capture_quality,
            graph_ir_version=graph.graph_ir_version,
            feature_schema_version=graph.feature_schema_version,
            feature_schema_sha256=graph.feature_schema_sha256,
            operator_registry_version=graph.operator_registry_version,
            operator_registry_sha256=graph.operator_registry_sha256,
            output_contract_version=metadata.output_contract_version,
            recommended_fallback=fallback,
        )


__all__ = [
    "PerfSeerV3Runtime",
    "RESULT_STATUSES",
    "SchedulerPredictionV3",
]
