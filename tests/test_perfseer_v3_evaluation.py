from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.evaluation import (
    AblationResult,
    AcceptanceEvidence,
    PredictionRecord,
    REQUIRED_ABLATIONS,
    assert_accepted,
    evaluate_acceptance,
    evaluate_oom_calibration,
    evaluate_predictions,
    evaluation_report,
    validate_ablation_matrix,
)


def _ablations() -> tuple[AblationResult, ...]:
    return tuple(
        AblationResult(name, 10.0, 9.0)
        for name in REQUIRED_ABLATIONS
    )


def _passing_evidence() -> AcceptanceEvidence:
    return AcceptanceEvidence(
        no_silent_operation_drops=True,
        strict_complete_capture_rate=0.96,
        complete_encoding_rate=1.0,
        unknown_gpu_time_fraction=0.01,
        v2_matched_mean_mape=10.0,
        v3_teacher_matched_mean_mape=10.1,
        v3_student_matched_mean_mape=10.2,
        v2_new_operations_mean_mape=30.0,
        v3_teacher_new_operations_mean_mape=24.0,
        v3_student_new_operations_mean_mape=26.0,
        student_latency_ratio_vs_v2=1.2,
        artifact_size_ratio_vs_v2=1.4,
        source_group_leakage=False,
        schema_mismatch_fails_closed=True,
        ablations=_ablations(),
    )


class EvaluationTests(unittest.TestCase):
    def test_six_target_metrics_and_near_zero_policy(self) -> None:
        records = [
            PredictionRecord(
                prediction=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                target=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                log_variance=(0.0,) * 6,
            ),
            PredictionRecord(
                prediction=(1e-9, 2.0, 3.0, 4.0, 5.0, 6.0),
                target=(1e-9, 2.0, 3.0, 4.0, 5.0, 6.0),
                log_variance=(0.0,) * 6,
            ),
        ]
        metrics = evaluate_predictions(records, near_zero_epsilon=1e-6)
        self.assertEqual(len(metrics), 6)
        first = metrics["train_epoch_ms"]
        self.assertEqual(first.count, 2)
        self.assertEqual(first.mape_count, 0)
        self.assertEqual(first.mae, 0)
        self.assertEqual(first.mape_percent, 0)
        self.assertEqual(first.interval_coverage, 1.0)

    def test_missing_ablation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_ablation_matrix(_ablations()[:-1])

    def test_oom_calibration_and_failure_stage_metrics(self) -> None:
        records = [
            PredictionRecord(
                prediction=(1.0,) * 6,
                target=(1.0,) * 6,
                oom_probability=0.9,
                oom_target=1,
                oom_failure_stage_prediction="backward",
                oom_failure_stage_target="backward",
            ),
            PredictionRecord(
                prediction=(1.0,) * 6,
                target=(1.0,) * 6,
                oom_probability=0.1,
                oom_target=0,
            ),
            PredictionRecord(
                prediction=(1.0,) * 6,
                target=(1.0,) * 6,
                oom_probability=0.8,
                oom_target=0,
            ),
        ]
        metrics = evaluate_oom_calibration(records)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["failure_stage_accuracy"], 1.0)

    def test_deliberate_gate_failures_keep_fallback_enabled(self) -> None:
        passing = _passing_evidence()
        failing = AcceptanceEvidence(
            **{
                **passing.__dict__,
                "strict_complete_capture_rate": 0.5,
                "unknown_gpu_time_fraction": None,
                "schema_mismatch_fails_closed": False,
            }
        )
        report = evaluate_acceptance(failing)
        self.assertFalse(report.overall_passed)
        self.assertIn("strict_complete_capture_rate", report.blockers)
        self.assertIn("unknown_gpu_time_fraction", report.blockers)
        self.assertIn("schema_mismatch_fails_closed", report.blockers)
        with self.assertRaisesRegex(RuntimeError, "keep scheduler fallback enabled"):
            assert_accepted(report)

    def test_passing_report_is_deterministic_and_serializable(self) -> None:
        evidence = _passing_evidence()
        acceptance = evaluate_acceptance(evidence)
        self.assertTrue(acceptance.overall_passed)
        assert_accepted(acceptance)
        records = [
            PredictionRecord(
                prediction=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                target=(1.1, 2.1, 3.1, 4.1, 5.1, 6.1),
                architecture_family="cnn",
                operation_family="dense_matrix",
                phase="forward",
                batch_size_bucket="small",
                precision="float32",
                evaluation_slice="v2_compatible",
            )
        ]
        first = evaluation_report(records, evidence)
        second = evaluation_report(records, evidence)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            acceptance.save(path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(loaded["overall_passed"])
        self.assertEqual(len(loaded["report_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
