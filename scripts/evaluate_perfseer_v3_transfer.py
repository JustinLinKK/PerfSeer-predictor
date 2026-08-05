#!/usr/bin/env python3
"""Evaluate grouped transfer ablations and 128/256/512/1024 label curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.evaluation import (
    PredictionRecord,
    evaluate_oom_calibration,
    evaluate_predictions,
    evaluate_slices,
)
from perfseer_v3.hardware_transfer import TRANSFER_LABEL_BUDGETS


REQUIRED_TRANSFER_EXPERIMENTS = {
    "no_adaptation",
    "linear_calibration",
    "hardware_specs_microbenchmarks",
    "low_rank_adapter",
    "film_low_rank_adapter",
    "adapter_heads",
    "adapter_last_block",
    "absolute_target_labels",
    "paired_residual_labels",
    "random_subset",
    "categorical_stratification",
    "stratification_latent_diversity",
    "fixed_256_subset",
    "active_128_plus_128",
    "student_hard_labels",
    "student_target_teacher_distillation",
    "teacher_t0",
    "teacher_t1",
    "student_s0",
    "student_s1",
    "static_hardware_specs",
    "static_specs_plus_microbenchmarks",
}

REQUIRED_RECORD_SLICES = (
    "architecture_family",
    "operation_family",
    "modality",
    "precision",
    "optimizer",
    "scheduler",
    "execution_mode",
    "resource_regime",
    "unknown_fraction_bucket",
    "evaluation_slice",
)

REQUIRED_SCHEDULER_OUTCOMES = (
    "incorrect_pack_admit_decisions",
    "missed_ooms",
    "unnecessary_fallbacks",
    "selected_batch_size_regret",
    "packed_job_throughput_regret",
)


def _record(value: dict[str, Any]) -> PredictionRecord:
    raw = dict(value)
    raw["prediction"] = tuple(float(item) for item in raw["prediction"])
    raw["target"] = tuple(float(item) for item in raw["target"])
    if raw.get("log_variance") is not None:
        raw["log_variance"] = tuple(float(item) for item in raw["log_variance"])
    for name in (
        "configuration_id",
        "source_group",
        "graph_signature",
        "split",
    ):
        raw.pop(name, None)
    return PredictionRecord(**raw)


def _serialize_slices(records: list[PredictionRecord]) -> dict[str, Any]:
    return {
        field_name: {
            slice_name: {
                target_name: asdict(metrics)
                for target_name, metrics in target_metrics.items()
            }
            for slice_name, target_metrics in field_slices.items()
        }
        for field_name, field_slices in evaluate_slices(records).items()
    }


def _oom_auroc(records: list[PredictionRecord]) -> dict[str, Any]:
    positives = [
        float(record.oom_probability)
        for record in records
        if record.oom_target == 1 and record.oom_probability is not None
    ]
    negatives = [
        float(record.oom_probability)
        for record in records
        if record.oom_target == 0 and record.oom_probability is not None
    ]
    if not positives or not negatives:
        return {
            "available": False,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "auroc": None,
        }
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return {
        "available": True,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "auroc": wins / (len(positives) * len(negatives)),
    }


def _quality_gates(
    overall: dict[str, Any],
    oom: dict[str, Any],
    auroc: dict[str, Any],
) -> list[dict[str, Any]]:
    definitions = (
        ("epoch_time_mape", overall["train_epoch_ms"]["mape_percent"], 10.0, "<="),
        ("average_sm_util_mae", overall["train_avg_sm_util_percent"]["mae"], 8.0, "<="),
        ("p95_sm_util_mae", overall["train_p95_sm_util_percent"]["mae"], 8.0, "<="),
        ("peak_vram_mape", overall["train_peak_vram_used_mib"]["mape_percent"], 8.0, "<="),
        (
            "peak_reserved_mape",
            overall["train_peak_torch_reserved_mib"]["mape_percent"],
            8.0,
            "<=",
        ),
        (
            "memory_controller_util_mae",
            overall["train_peak_memory_controller_util_percent"]["mae"],
            10.0,
            "<=",
        ),
    )
    gates = [
        {
            "name": name,
            "observed": observed,
            "requirement": f"{operator} {limit}",
            "passed": observed <= limit,
        }
        for name, observed, limit, operator in definitions
    ]
    gates.extend(
        (
            {
                "name": "oom_recall",
                "observed": oom.get("recall") if oom.get("available") else None,
                "requirement": ">= 0.98 with measured OOM labels",
                "passed": bool(oom.get("available") and oom.get("recall", 0.0) >= 0.98),
            },
            {
                "name": "oom_auroc",
                "observed": auroc.get("auroc"),
                "requirement": ">= 0.95 with both OOM classes represented",
                "passed": bool(auroc.get("available") and auroc.get("auroc", 0.0) >= 0.95),
            },
        )
    )
    for target_name, metrics in overall.items():
        for nominal, field_name in (
            (0.80, "interval_coverage_80"),
            (0.95, "interval_coverage_95"),
        ):
            observed = metrics.get(field_name)
            gates.append(
                {
                    "name": f"{target_name}_{field_name}",
                    "observed": observed,
                    "requirement": f"within 0.05 of nominal {nominal:.2f}",
                    "passed": bool(
                        observed is not None
                        and abs(float(observed) - nominal) <= 0.05
                    ),
                }
            )
    return gates


def _scheduler_target_error(overall: dict[str, Any], target_name: str) -> float:
    if target_name in {
        "train_epoch_ms",
        "train_peak_vram_used_mib",
        "train_peak_torch_reserved_mib",
    }:
        return float(overall[target_name]["mape_percent"])
    return float(overall[target_name]["mae"])


def _critical_family_gate(
    overall: dict[str, Any],
    slices: dict[str, Any],
    raw_records: list[dict[str, Any]],
) -> dict[str, Any]:
    regressions: list[dict[str, Any]] = []
    for family, family_metrics in slices["architecture_family"].items():
        affected_targets = []
        for target_name in overall:
            overall_error = _scheduler_target_error(overall, target_name)
            family_error = _scheduler_target_error(family_metrics, target_name)
            if family_error > 1.2 * overall_error and family_error > 0.0:
                affected_targets.append(
                    {
                        "target": target_name,
                        "overall_error": overall_error,
                        "family_error": family_error,
                    }
                )
        if not affected_targets:
            continue
        family_rows = [
            row for row in raw_records if str(row["architecture_family"]) == family
        ]
        fallback_attested = bool(family_rows) and all(
            row.get("fallback_or_low_confidence") is True for row in family_rows
        )
        regressions.append(
            {
                "architecture_family": family,
                "affected_targets": affected_targets,
                "fallback_or_low_confidence_attested": fallback_attested,
            }
        )
    return {
        "name": "critical_family_regression_or_fallback",
        "observed": regressions,
        "requirement": (
            "no architecture-family error above 1.2x overall, unless every row "
            "in that family forces fallback or low confidence"
        ),
        "passed": all(
            item["fallback_or_low_confidence_attested"] for item in regressions
        ),
    }


def _comparison_gates(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(report["experiment"]), int(report["label_budget"])): report
        for report in reports
    }
    adapted = indexed[("film_low_rank_adapter", 256)]
    adapted_mean = sum(
        _scheduler_target_error(adapted["overall"], target_name)
        for target_name in adapted["overall"]
    ) / len(adapted["overall"])
    gates: list[dict[str, Any]] = []
    for control_name in ("no_adaptation", "linear_calibration"):
        control = indexed[(control_name, 256)]
        control_mean = sum(
            _scheduler_target_error(control["overall"], target_name)
            for target_name in control["overall"]
        ) / len(control["overall"])
        relative_improvement = (
            (control_mean - adapted_mean) / control_mean if control_mean > 0 else 0.0
        )
        gates.append(
            {
                "name": f"adapted_teacher_materially_beats_{control_name}",
                "observed": {
                    "adapted_mean_scheduler_error": adapted_mean,
                    "control_mean_scheduler_error": control_mean,
                    "relative_improvement": relative_improvement,
                },
                "requirement": ">= 0.05 relative improvement",
                "passed": control_mean > 0 and relative_improvement >= 0.05,
            }
        )
    student = indexed[("student_target_teacher_distillation", 256)]
    per_target = {
        target_name: {
            "teacher_error": _scheduler_target_error(adapted["overall"], target_name),
            "student_error": _scheduler_target_error(student["overall"], target_name),
        }
        for target_name in adapted["overall"]
    }
    gates.append(
        {
            "name": "adapted_student_within_10_percent_of_teacher",
            "observed": per_target,
            "requirement": "student error <= 1.10x teacher error for every scheduler target",
            "passed": all(
                values["student_error"] <= 1.10 * values["teacher_error"]
                if values["teacher_error"] > 0
                else values["student_error"] == 0
                for values in per_target.values()
            ),
        }
    )
    student_s1 = indexed[("student_s1", 256)]
    deployment = student_s1["deployment"]
    for field_name, limit in (
        ("cpu_p95_latency_ratio_vs_v2", 1.25),
        ("artifact_size_ratio_vs_v2", 1.5),
    ):
        observed = float(deployment[field_name])
        gates.append(
            {
                "name": f"student_s1_{field_name}",
                "observed": observed,
                "requirement": f"<= {limit}",
                "passed": observed <= limit,
            }
        )
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if payload.get("group_isolation_attested") is not True:
        raise ValueError("transfer evidence lacks grouped split-isolation attestation")
    frozen_test_sha256 = str(payload.get("frozen_test_sha256", ""))
    if len(frozen_test_sha256) != 64:
        raise ValueError("transfer evidence lacks a frozen grouped-test fingerprint")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("transfer evidence contains no experiment runs")
    names = {str(run.get("experiment")) for run in runs}
    missing = REQUIRED_TRANSFER_EXPERIMENTS - names
    if missing:
        raise ValueError("transfer ablation matrix is incomplete: " + ", ".join(sorted(missing)))
    curve_budgets = {
        int(run["label_budget"])
        for run in runs
        if run.get("experiment") == "film_low_rank_adapter"
    }
    if curve_budgets != set(TRANSFER_LABEL_BUDGETS):
        raise ValueError("adapter label curve must contain exactly 128/256/512/1024")
    keys = [(str(run.get("experiment")), int(run.get("label_budget", 0))) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("transfer evidence repeats an experiment/label-budget pair")
    if any(budget not in TRANSFER_LABEL_BUDGETS for _, budget in keys):
        raise ValueError("transfer experiment label budget is outside the frozen gates")
    if any(
        budget != 256
        for experiment, budget in keys
        if experiment != "film_low_rank_adapter"
    ):
        raise ValueError("non-curve transfer ablations must use the comparable 256-label budget")

    reports = []
    for run in runs:
        if run.get("test_sha256") != frozen_test_sha256:
            raise ValueError("an experiment evaluated a different grouped test set")
        raw_records = run.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("every transfer experiment needs held-out prediction records")
        if any(str(row.get("split")) != "test" for row in raw_records):
            raise ValueError("transfer evaluation may consume only the frozen test split")
        for row in raw_records:
            missing_slices = [
                name for name in REQUIRED_RECORD_SLICES if row.get(name) in (None, "")
            ]
            if missing_slices:
                raise ValueError(
                    "held-out records lack required slices: " + ", ".join(missing_slices)
                )
            if "fallback_or_low_confidence" in row and not isinstance(
                row["fallback_or_low_confidence"], bool
            ):
                raise ValueError("fallback_or_low_confidence must be a boolean attestation")
        records = [_record(dict(row)) for row in raw_records]
        scheduler_outcomes = dict(run.get("scheduler_outcomes") or {})
        missing_outcomes = set(REQUIRED_SCHEDULER_OUTCOMES) - set(scheduler_outcomes)
        if missing_outcomes:
            raise ValueError(
                "scheduler outcome report is incomplete: "
                + ", ".join(sorted(missing_outcomes))
            )
        if any(
            not isinstance(scheduler_outcomes[name], (int, float))
            or not math.isfinite(float(scheduler_outcomes[name]))
            or float(scheduler_outcomes[name]) < 0
            for name in REQUIRED_SCHEDULER_OUTCOMES
        ):
            raise ValueError("scheduler outcomes must be finite and nonnegative")
        overall = {
            name: asdict(metrics)
            for name, metrics in evaluate_predictions(records).items()
        }
        slices = _serialize_slices(records)
        oom = evaluate_oom_calibration(records)
        auroc = _oom_auroc(records)
        deployment = dict(run.get("deployment") or {})
        if run.get("experiment") == "student_s1":
            for name in (
                "cpu_p95_latency_ratio_vs_v2",
                "artifact_size_ratio_vs_v2",
            ):
                value = deployment.get(name)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(
                        f"student_s1 deployment evidence requires finite nonnegative {name}"
                    )
        reports.append(
            {
                "experiment": str(run["experiment"]),
                "label_budget": int(run["label_budget"]),
                "record_count": len(records),
                "overall": overall,
                "slices": slices,
                "oom": oom,
                "oom_auroc": auroc,
                "quality_gates": _quality_gates(overall, oom, auroc),
                "critical_family_gate": _critical_family_gate(
                    overall,
                    slices,
                    raw_records,
                ),
                "scheduler_outcomes": scheduler_outcomes,
                "deployment": deployment,
            }
        )
    comparison_gates = _comparison_gates(reports)
    gate_failures = [
        f"{report['experiment']}@{report['label_budget']}:{gate['name']}"
        for report in reports
        for gate in (
            *report["quality_gates"],
            report["critical_family_gate"],
        )
        if not gate["passed"]
    ]
    gate_failures.extend(
        f"comparison:{gate['name']}"
        for gate in comparison_gates
        if not gate["passed"]
    )
    report: dict[str, Any] = {
        "report_version": "perfseer_v3_transfer_evaluation_v1",
        "source_evidence": str(args.evidence.resolve()),
        "source_evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "group_isolation_attested": True,
        "frozen_test_sha256": frozen_test_sha256,
        "label_curve_budgets": list(TRANSFER_LABEL_BUDGETS),
        "experiments": reports,
        "comparison_gates": comparison_gates,
        "evidence_acceptance_passed": not gate_failures,
        "evidence_acceptance_blockers": gate_failures,
        "broad_any_nvidia_claim_authorized": False,
        "claim_note": (
            "Architecture-generation coverage must be audited separately before any "
            "broad NVIDIA claim."
        ),
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json(report).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report_sha256": report["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
