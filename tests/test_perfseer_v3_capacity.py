from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capacity import exact_parameter_count, load_capacity_study
from perfseer_v3.capture_export import capture_export
from perfseer_v3.features import batch_graph_features, build_graph_features
from perfseer_v3.model import SeerNetV3, graph_batch_tensors
from perfseer_v3.op_registry import OperationRegistry
from scripts.benchmark_perfseer_v3_capacity import main as capacity_main


class _Tiny(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + 1)


class CapacityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        capture = capture_export(_Tiny(), (torch.randn(2, 4),))
        if not capture.success or capture.graph is None:
            raise AssertionError(capture.failures)
        cls.batch = batch_graph_features([build_graph_features(capture.graph)])
        cls.registry = OperationRegistry.load()

    def test_required_candidates_and_parameter_guardrails(self) -> None:
        study = load_capacity_study()
        self.assertEqual(
            {candidate.candidate_id for candidate in study.candidates},
            {"T0", "T1", "T2", "S0", "S1", "S2", "S3"},
        )
        counts = {
            candidate.candidate_id: exact_parameter_count(
                candidate.model_config(self.registry, self.batch.layout)
            )
            for candidate in study.candidates
        }
        self.assertLess(counts["T0"], counts["T1"])
        self.assertLess(counts["T1"], counts["T2"])
        self.assertLessEqual(counts["T2"], 3 * counts["T0"])
        self.assertLess(counts["S0"], counts["S1"])
        self.assertLess(counts["S1"], counts["S2"])
        self.assertLess(counts["S2"], counts["S3"])

    def test_student_candidate_preserves_six_output_contract(self) -> None:
        candidate = load_capacity_study().candidate("S1")
        config = candidate.model_config(self.registry, self.batch.layout, dropout=0.0)
        model = SeerNetV3(config).eval()
        with torch.no_grad():
            output = model(*graph_batch_tensors(self.batch))
        self.assertEqual(output.prediction.shape, (1, 6))
        self.assertEqual(output.oom_stage_logits.shape, (1, 7))
        self.assertTrue(torch.isfinite(output.prediction).all())

    def test_parameter_report_is_hash_verified_and_explicitly_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capacity.json"
            self.assertEqual(capacity_main(["--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["candidates"]), 7)
        self.assertFalse(report["production_measurement_status"]["available"])
        self.assertEqual(len(report["report_sha256"]), 64)
        self.assertTrue(
            all(row["trainable_parameter_count"] > 0 for row in report["candidates"])
        )


if __name__ == "__main__":
    unittest.main()
