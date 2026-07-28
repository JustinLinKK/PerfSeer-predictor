"""Six-target evaluation, required ablations, and fail-closed safety gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .baseline import canonical_json
from .training import TARGET_NAMES


REQUIRED_ABLATIONS = (
    "v2_onehot_vs_v3_hierarchical",
    "forward_only_vs_training_graph",
    "exact_only_vs_family_exact_hash",
    "topology_only_vs_cost_liveness",
    "raw_vs_coarsened_graph",
    "existing_vs_phase_aware_pooling",
    "teacher_t0_vs_t1_t2",
    "student_s0_vs_s1_s2",
    "random_init_vs_trunk_reuse",
    "hard_label_vs_distillation",
)


@dataclass(frozen=True)
class TargetMetrics:
    count: int
    mape_count: int
    mae: float
    mape_percent: float
    rmse_raw: float
    rmse_log1p: float
    r2: float
    p50_absolute_percentage_error: float
    p90_absolute_percentage_error: float
    p95_absolute_percentage_error: float
    interval_coverage: float | None = None


@dataclass(frozen=True)
class PredictionRecord:
    prediction: tuple[float, ...]
    target: tuple[float, ...]
    log_variance: tuple[float, ...] | None = None
    architecture_family: str = "unknown"
    operation_family: str = "unknown"
    modality: str = "unknown"
    phase: str = "unknown"
    batch_size_bucket: str = "unknown"
    precision: str = "unknown"
    optimizer: str = "unknown"
    capture_quality: str = "unknown"
    graph_size_bucket: str = "unknown"
    resource_regime: str = "unknown"
    unknown_fraction_bucket: str = "unknown"
    evaluation_slice: str = "default"
    oom_probability: float | None = None
    oom_target: int | None = None
    oom_failure_stage_prediction: str | None = None
    oom_failure_stage_target: str | None = None

    def validate(self) -> None:
        if len(self.prediction) != len(TARGET_NAMES) or len(self.target) != len(TARGET_NAMES):
            raise ValueError("prediction records must follow the six-target contract")
        if self.log_variance is not None and len(self.log_variance) != len(TARGET_NAMES):
            raise ValueError("log variance must follow the six-target contract")
        if not np.isfinite(self.prediction).all() or not np.isfinite(self.target).all():
            raise ValueError("prediction and target values must be finite")
        if self.oom_probability is not None and not 0.0 <= self.oom_probability <= 1.0:
            raise ValueError("OOM probability must be in [0, 1]")
        if self.oom_target is not None and self.oom_target not in {0, 1}:
            raise ValueError("OOM target must be binary")


def _target_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    log_variance: np.ndarray | None,
    *,
    near_zero_epsilon: float,
) -> TargetMetrics:
    error = prediction - target
    absolute = np.abs(error)
    mask = np.abs(target) >= near_zero_epsilon
    percentages = absolute[mask] / np.abs(target[mask]) * 100.0
    target_mean = target.mean()
    ss_total = ((target - target_mean) ** 2).sum()
    ss_residual = (error**2).sum()
    r2 = 1.0 - ss_residual / ss_total if ss_total > 1e-12 else (1.0 if ss_residual <= 1e-12 else 0.0)
    interval_coverage = None
    if log_variance is not None:
        standard_deviation = np.exp(0.5 * np.clip(log_variance, -20.0, 20.0))
        interval_coverage = float(np.mean(absolute <= 1.96 * standard_deviation))
    if percentages.size:
        p50, p90, p95 = np.percentile(percentages, (50, 90, 95))
        mape = float(percentages.mean())
    else:
        p50 = p90 = p95 = mape = 0.0
    return TargetMetrics(
        count=int(target.size),
        mape_count=int(mask.sum()),
        mae=float(absolute.mean()),
        mape_percent=mape,
        rmse_raw=float(np.sqrt(np.mean(error**2))),
        rmse_log1p=float(
            np.sqrt(
                np.mean(
                    (
                        np.log1p(np.maximum(prediction, 0.0))
                        - np.log1p(np.maximum(target, 0.0))
                    )
                    ** 2
                )
            )
        ),
        r2=float(r2),
        p50_absolute_percentage_error=float(p50),
        p90_absolute_percentage_error=float(p90),
        p95_absolute_percentage_error=float(p95),
        interval_coverage=interval_coverage,
    )


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    *,
    near_zero_epsilon: float = 1e-6,
) -> dict[str, TargetMetrics]:
    if not records:
        raise ValueError("evaluation requires prediction records")
    for record in records:
        record.validate()
    prediction = np.asarray([record.prediction for record in records], dtype=np.float64)
    target = np.asarray([record.target for record in records], dtype=np.float64)
    if all(record.log_variance is not None for record in records):
        log_variance = np.asarray([record.log_variance for record in records], dtype=np.float64)
    else:
        log_variance = None
    return {
        name: _target_metrics(
            prediction[:, index],
            target[:, index],
            None if log_variance is None else log_variance[:, index],
            near_zero_epsilon=near_zero_epsilon,
        )
        for index, name in enumerate(TARGET_NAMES)
    }


_SLICE_FIELDS = (
    "architecture_family",
    "operation_family",
    "modality",
    "phase",
    "batch_size_bucket",
    "precision",
    "optimizer",
    "capture_quality",
    "graph_size_bucket",
    "resource_regime",
    "unknown_fraction_bucket",
    "evaluation_slice",
)


def evaluate_oom_calibration(
    records: Sequence[PredictionRecord],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("OOM threshold must be in (0, 1)")
    usable = [
        record
        for record in records
        if record.oom_probability is not None and record.oom_target is not None
    ]
    if not usable:
        return {"available": False, "count": 0, "threshold": threshold}
    probability = np.asarray(
        [record.oom_probability for record in usable],
        dtype=np.float64,
    )
    target = np.asarray([record.oom_target for record in usable], dtype=np.int64)
    predicted = probability >= threshold
    positive = target == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    stages = [
        record
        for record in usable
        if record.oom_target == 1
        and record.oom_failure_stage_prediction is not None
        and record.oom_failure_stage_target is not None
    ]
    stage_correct = sum(
        record.oom_failure_stage_prediction == record.oom_failure_stage_target
        for record in stages
    )
    return {
        "available": True,
        "count": len(usable),
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "failure_stage_count": len(stages),
        "failure_stage_accuracy": stage_correct / max(1, len(stages)),
    }


def evaluate_slices(
    records: Sequence[PredictionRecord],
    *,
    slice_fields: Iterable[str] = _SLICE_FIELDS,
    near_zero_epsilon: float = 1e-6,
) -> dict[str, dict[str, dict[str, TargetMetrics]]]:
    result: dict[str, dict[str, dict[str, TargetMetrics]]] = {}
    for field_name in slice_fields:
        if field_name not in _SLICE_FIELDS:
            raise ValueError(f"unsupported evaluation slice field {field_name!r}")
        grouped: dict[str, list[PredictionRecord]] = {}
        for record in records:
            grouped.setdefault(str(getattr(record, field_name)), []).append(record)
        result[field_name] = {
            name: evaluate_predictions(rows, near_zero_epsilon=near_zero_epsilon)
            for name, rows in sorted(grouped.items())
        }
    return result


@dataclass(frozen=True)
class AblationResult:
    name: str
    baseline_mean_mape: float
    candidate_mean_mape: float
    candidate_student_latency_ms: float | None = None
    notes: str = ""


def validate_ablation_matrix(results: Sequence[AblationResult]) -> None:
    names = [result.name for result in results]
    missing = set(REQUIRED_ABLATIONS) - set(names)
    duplicates = {name for name in names if names.count(name) > 1}
    if missing or duplicates:
        messages = []
        if missing:
            messages.append("missing=" + ",".join(sorted(missing)))
        if duplicates:
            messages.append("duplicates=" + ",".join(sorted(duplicates)))
        raise ValueError("invalid required ablation matrix: " + "; ".join(messages))
    for result in results:
        if not np.isfinite((result.baseline_mean_mape, result.candidate_mean_mape)).all():
            raise ValueError(f"ablation {result.name!r} contains nonfinite metrics")


@dataclass(frozen=True)
class AcceptanceEvidence:
    no_silent_operation_drops: bool
    strict_complete_capture_rate: float
    complete_encoding_rate: float
    unknown_gpu_time_fraction: float | None
    v2_matched_mean_mape: float
    v3_teacher_matched_mean_mape: float
    v3_student_matched_mean_mape: float
    v2_new_operations_mean_mape: float
    v3_teacher_new_operations_mean_mape: float
    v3_student_new_operations_mean_mape: float
    student_latency_ratio_vs_v2: float
    artifact_size_ratio_vs_v2: float
    source_group_leakage: bool
    schema_mismatch_fails_closed: bool
    ablations: tuple[AblationResult, ...]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: Any
    requirement: str


@dataclass(frozen=True)
class AcceptanceReport:
    report_version: str
    overall_passed: bool
    gates: tuple[GateResult, ...]
    blockers: tuple[str, ...]
    evidence: AcceptanceEvidence

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_sha256"] = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
        return data

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output


def evaluate_acceptance(
    evidence: AcceptanceEvidence,
    *,
    matched_tolerance_relative: float = 0.05,
    minimum_teacher_new_operation_improvement_relative: float = 0.15,
    minimum_student_new_operation_improvement_relative: float = 0.10,
    maximum_student_latency_ratio: float = 1.25,
    maximum_artifact_size_ratio: float = 1.5,
) -> AcceptanceReport:
    validate_ablation_matrix(evidence.ablations)
    matched_limit = evidence.v2_matched_mean_mape * (1.0 + matched_tolerance_relative)
    teacher_new_operation_limit = evidence.v2_new_operations_mean_mape * (
        1.0 - minimum_teacher_new_operation_improvement_relative
    )
    student_new_operation_limit = evidence.v2_new_operations_mean_mape * (
        1.0 - minimum_student_new_operation_improvement_relative
    )
    gates = (
        GateResult(
            "no_silent_operation_drops",
            evidence.no_silent_operation_drops,
            evidence.no_silent_operation_drops,
            "true",
        ),
        GateResult(
            "strict_complete_capture_rate",
            evidence.strict_complete_capture_rate >= 0.95,
            evidence.strict_complete_capture_rate,
            ">= 0.95",
        ),
        GateResult(
            "complete_encoding_rate",
            evidence.complete_encoding_rate >= 0.99,
            evidence.complete_encoding_rate,
            ">= 0.99",
        ),
        GateResult(
            "unknown_gpu_time_fraction",
            evidence.unknown_gpu_time_fraction is not None
            and evidence.unknown_gpu_time_fraction <= 0.02,
            evidence.unknown_gpu_time_fraction,
            "measured and <= 0.02",
        ),
        GateResult(
            "teacher_v2_matched_non_regression",
            evidence.v3_teacher_matched_mean_mape <= matched_limit,
            evidence.v3_teacher_matched_mean_mape,
            f"<= {matched_limit}",
        ),
        GateResult(
            "student_v2_matched_non_regression",
            evidence.v3_student_matched_mean_mape <= matched_limit,
            evidence.v3_student_matched_mean_mape,
            f"<= {matched_limit}",
        ),
        GateResult(
            "teacher_new_operation_improvement",
            evidence.v3_teacher_new_operations_mean_mape
            <= teacher_new_operation_limit,
            evidence.v3_teacher_new_operations_mean_mape,
            f"<= {teacher_new_operation_limit}",
        ),
        GateResult(
            "student_new_operation_improvement",
            evidence.v3_student_new_operations_mean_mape
            <= student_new_operation_limit,
            evidence.v3_student_new_operations_mean_mape,
            f"<= {student_new_operation_limit}",
        ),
        GateResult(
            "student_latency",
            evidence.student_latency_ratio_vs_v2 <= maximum_student_latency_ratio,
            evidence.student_latency_ratio_vs_v2,
            f"<= {maximum_student_latency_ratio}",
        ),
        GateResult(
            "artifact_size",
            evidence.artifact_size_ratio_vs_v2 <= maximum_artifact_size_ratio,
            evidence.artifact_size_ratio_vs_v2,
            f"<= {maximum_artifact_size_ratio}",
        ),
        GateResult(
            "source_group_isolation",
            not evidence.source_group_leakage,
            evidence.source_group_leakage,
            "false",
        ),
        GateResult(
            "schema_mismatch_fails_closed",
            evidence.schema_mismatch_fails_closed,
            evidence.schema_mismatch_fails_closed,
            "true",
        ),
    )
    blockers = tuple(gate.name for gate in gates if not gate.passed)
    return AcceptanceReport(
        report_version="perfseer_v3_acceptance_v1",
        overall_passed=not blockers,
        gates=gates,
        blockers=blockers,
        evidence=evidence,
    )


def assert_accepted(report: AcceptanceReport) -> None:
    if not report.overall_passed:
        raise RuntimeError(
            "PerfSeer v3 acceptance gates failed; keep scheduler fallback enabled: "
            + ", ".join(report.blockers)
        )


def evaluation_report(
    records: Sequence[PredictionRecord],
    evidence: AcceptanceEvidence,
    *,
    near_zero_epsilon: float = 1e-6,
) -> dict[str, Any]:
    metrics = evaluate_predictions(records, near_zero_epsilon=near_zero_epsilon)
    slices = evaluate_slices(records, near_zero_epsilon=near_zero_epsilon)
    acceptance = evaluate_acceptance(evidence)
    payload: dict[str, Any] = {
        "report_version": "perfseer_v3_evaluation_v1",
        "near_zero_policy": {
            "method": "exclude_from_percentage_metrics_only",
            "absolute_target_epsilon": near_zero_epsilon,
        },
        "target_order": list(TARGET_NAMES),
        "metrics": {
            name: asdict(value) for name, value in metrics.items()
        },
        "oom_calibration": evaluate_oom_calibration(records),
        "slices": {
            field_name: {
                slice_name: {
                    target_name: asdict(value)
                    for target_name, value in target_metrics.items()
                }
                for slice_name, target_metrics in field_slices.items()
            }
            for field_name, field_slices in slices.items()
        },
        "acceptance": acceptance.to_dict(),
    }
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


__all__ = [
    "AblationResult",
    "AcceptanceEvidence",
    "AcceptanceReport",
    "GateResult",
    "PredictionRecord",
    "REQUIRED_ABLATIONS",
    "TargetMetrics",
    "assert_accepted",
    "evaluate_acceptance",
    "evaluate_oom_calibration",
    "evaluate_predictions",
    "evaluate_slices",
    "evaluation_report",
    "validate_ablation_matrix",
]
