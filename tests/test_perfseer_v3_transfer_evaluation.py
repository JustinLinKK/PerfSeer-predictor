from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_perfseer_v3_transfer.py"
SPEC = importlib.util.spec_from_file_location("evaluate_perfseer_v3_transfer", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load transfer evaluator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record() -> dict[str, object]:
    return {
        "configuration_id": "cfg-test",
        "source_group": "held-out-family",
        "graph_signature": "held-out-graph",
        "split": "test",
        "prediction": [10.0, 50.0, 60.0, 4000.0, 4500.0, 40.0],
        "target": [10.0, 50.0, 60.0, 4000.0, 4500.0, 40.0],
        "log_variance": [0.0] * 6,
        "architecture_family": "transformer",
        "operation_family": "attention",
        "modality": "nlp",
        "phase": "training",
        "batch_size_bucket": "medium",
        "precision": "bfloat16",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "execution_mode": "compiled",
        "capture_quality": "strict",
        "graph_size_bucket": "medium",
        "resource_regime": "memory_bound",
        "unknown_fraction_bucket": "none",
        "evaluation_slice": "new_operation_slice",
        "oom_probability": 0.9,
        "oom_target": 1,
        "oom_failure_stage_prediction": "allocator",
        "oom_failure_stage_target": "allocator",
    }


def _scheduler_outcomes() -> dict[str, float]:
    return {
        "incorrect_pack_admit_decisions": 0.0,
        "missed_ooms": 0.0,
        "unnecessary_fallbacks": 0.0,
        "selected_batch_size_regret": 0.0,
        "packed_job_throughput_regret": 0.0,
    }


class TransferEvaluationTests(unittest.TestCase):
    def test_complete_matrix_emits_slices_gates_and_denies_broad_claim(self) -> None:
        runs = []
        for experiment in sorted(MODULE.REQUIRED_TRANSFER_EXPERIMENTS):
            budgets = (128, 256, 512, 1024) if experiment == "film_low_rank_adapter" else (256,)
            for budget in budgets:
                run = {
                    "experiment": experiment,
                    "label_budget": budget,
                    "test_sha256": "a" * 64,
                    "records": [_record()],
                    "scheduler_outcomes": _scheduler_outcomes(),
                }
                if experiment == "student_s1":
                    run["deployment"] = {
                        "cpu_p95_latency_ratio_vs_v2": 1.0,
                        "artifact_size_ratio_vs_v2": 1.0,
                    }
                runs.append(run)
        payload = {
            "group_isolation_attested": True,
            "frozen_test_sha256": "a" * 64,
            "runs": runs,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            output = root / "report.json"
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                MODULE.main(["--evidence", str(evidence), "--output", str(output)]),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["label_curve_budgets"], [128, 256, 512, 1024])
        self.assertFalse(report["broad_any_nvidia_claim_authorized"])
        self.assertIn("comparison_gates", report)
        self.assertIn("evidence_acceptance_passed", report)
        first = report["experiments"][0]
        self.assertIn("execution_mode", first["slices"])
        self.assertIn("quality_gates", first)
        self.assertFalse(first["oom_auroc"]["available"])
        self.assertEqual(first["scheduler_outcomes"]["missed_ooms"], 0.0)
        self.assertIn("interval_coverage_80", first["overall"]["train_epoch_ms"])
        self.assertTrue(first["critical_family_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
