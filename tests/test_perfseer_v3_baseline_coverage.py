from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import (
    BaselineSnapshot,
    canonical_json,
    sha256_file,
    tree_fingerprint,
    write_baseline,
)
from perfseer_v3.coverage import audit_corpus, smallest_time_vocabulary, write_coverage_reports
from perfseer_v3.coverage_corpus import (
    frontier_source_cases,
    p0_cases,
    representative_source_cases,
    smoke_cases,
)


class BaselineCoverageTests(unittest.TestCase):
    def test_tree_fingerprint_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.yaml"
            path.write_text("value: 1\n", encoding="utf-8")
            first = tree_fingerprint(root, sample_large_files=False)
            second = tree_fingerprint(root, sample_large_files=False)
            self.assertEqual(first, second)
            path.write_text("value: 2\n", encoding="utf-8")
            self.assertNotEqual(first["sha256"], tree_fingerprint(root, sample_large_files=False)["sha256"])
            self.assertTrue(sha256_file(path, sample_large_files=False).startswith("full:"))

    def test_baseline_snapshot_hash_and_writer_are_stable(self) -> None:
        empty = {"status": "missing", "sha256": None}
        snapshot = BaselineSnapshot(
            baseline_version="test",
            commit_sha="abc",
            branch="v2",
            python_version="3",
            pytorch_version="2",
            cuda_version=None,
            cudnn_version=None,
            config_fingerprint=empty,
            dataset_fingerprint=empty,
            checkpoint_fingerprint=empty,
            evaluation_fingerprint=empty,
        )
        self.assertEqual(snapshot.to_dict(), snapshot.to_dict())
        with tempfile.TemporaryDirectory() as temporary:
            output = write_baseline(snapshot, Path(temporary) / "baseline.json")
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["snapshot_sha256"], snapshot.to_dict()["snapshot_sha256"])
        self.assertEqual(canonical_json(loaded), canonical_json(snapshot.to_dict()))

    def test_collected_baseline_records_current_35_entry_v2_schema(self) -> None:
        schema_path = SRC / "perfseer" / "architecture_schema.py"
        spec = importlib.util.spec_from_file_location("_perfseer_v2_schema_test", schema_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.FEATURE_SCHEMA_VERSION, "perfseer_graph_v1")
        self.assertEqual(len(module.NODE_TYPES), 35)
        self.assertNotEqual(len(module.NODE_TYPES), 23)
        baseline = json.loads((ROOT / "reports" / "v2_baseline.json").read_text())
        self.assertEqual(baseline["v2_feature_schema_version"], module.FEATURE_SCHEMA_VERSION)
        self.assertEqual(tuple(baseline["v2_node_types"]), module.NODE_TYPES)

    def test_smoke_corpus_retains_every_exported_tensor_operation(self) -> None:
        report, failures = audit_corpus(smoke_cases())
        self.assertEqual(report["strict_export_success_rate"], 1.0, failures)
        self.assertEqual(report["complete_graph_success_rate"], 1.0, failures)
        self.assertEqual(report["tensor_nodes"], report["encoded_tensor_nodes"])
        self.assertGreater(report["unique_raw_operations"], 5)
        self.assertIn("convolution", report["family_counts"])
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_coverage_reports(report, failures, temporary)
            self.assertEqual(set(paths), {"json", "markdown", "failures"})
            self.assertEqual(
                json.loads(paths["json"].read_text(encoding="utf-8"))["report_sha256"],
                report["report_sha256"],
            )
            failure_text = paths["failures"].read_text(encoding="utf-8")
            self.assertNotRegex(failure_text, r"0x[0-9a-fA-F]{6,}")
            self.assertNotIn(str(ROOT), failure_text)

    def test_time_vocabulary_uses_minimal_deterministic_prefix(self) -> None:
        selected = smallest_time_vocabulary({"aten.slow": 90.0, "aten.fast": 5.0, "aten.tail": 5.0})
        self.assertEqual(selected, ("aten.slow", "aten.fast"))
        self.assertEqual(smallest_time_vocabulary({}), ())
        with self.assertRaises(ValueError):
            smallest_time_vocabulary({"x": 1}, coverage=0)

    def test_p0_corpus_has_complete_strict_capture_and_weighted_metrics(self) -> None:
        report, failures = audit_corpus(
            p0_cases(),
            gpu_time_by_operation={
                "aten::conv2d": 90.0,
                "aten::linear": 5.0,
                "custom_namespace::opaque": 5.0,
            },
        )
        export_failures = [failure for failure in failures if failure.backend != "legacy_fx"]
        self.assertEqual(export_failures, [])
        self.assertEqual(report["strict_export_success_rate"], 1.0)
        self.assertEqual(report["complete_graph_success_rate"], 1.0)
        self.assertEqual(report["tensor_nodes"], report["encoded_tensor_nodes"])
        self.assertGreaterEqual(report["models"], 14)
        self.assertGreaterEqual(report["unique_raw_operations"], 100)
        self.assertGreater(report["coverage"]["exact_known"], 0)
        self.assertGreater(report["flop_weighted_coverage"]["total"], 0)
        self.assertGreater(report["tensor_byte_weighted_coverage"]["total"], 0)
        self.assertAlmostEqual(
            report["profiler_time_weighted_coverage"]["custom_fraction"],
            0.05,
        )
        self.assertEqual(
            report["recommended_exact_vocabulary_95pct_gpu_time"],
            ["aten::conv2d", "aten::linear"],
        )
        self.assertIn("attention", report["coverage_by_architecture_family"])
        self.assertIn("text", report["coverage_by_modality"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch_geometric") is not None,
        "torch_geometric is not installed",
    )
    def test_legacy_diagnostics_are_stable_after_pyg_function_patches(self) -> None:
        import torch
        import torch_geometric.hash_tensor  # noqa: F401

        case = next(
            case
            for case in p0_cases()
            if case.case_id == "p0_index_scatter_variants"
        )
        patched_index_select = torch.index_select
        patched_select = torch.select
        first_report, first_failures = audit_corpus((case,))
        second_report, second_failures = audit_corpus((case,))
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_failures, second_failures)
        self.assertIs(torch.index_select, patched_index_select)
        self.assertIs(torch.select, patched_select)

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name) is not None
            for name in ("torchvision", "transformers", "torch_geometric")
        ),
        "representative coverage dependencies are not installed",
    )
    def test_representative_library_corpus_meets_strict_supported_boundary(self) -> None:
        report, failures = audit_corpus(representative_source_cases())
        export_failures = [failure for failure in failures if failure.backend != "legacy_fx"]
        self.assertEqual(export_failures, [])
        self.assertEqual(report["models"], 4)
        self.assertEqual(report["strict_export_success_rate"], 1.0)
        self.assertEqual(report["complete_graph_success_rate"], 1.0)
        self.assertEqual(report["tensor_nodes"], report["encoded_tensor_nodes"])
        self.assertGreater(report["tensor_nodes"], 300)
        self.assertIn("graph", report["coverage_by_modality"])
        self.assertIn("image", report["coverage_by_modality"])
        self.assertIn("text", report["coverage_by_modality"])

    @unittest.skipUnless(
        importlib.util.find_spec("torchaudio") is not None,
        "torchaudio is not installed",
    )
    def test_frontier_audio_uses_validated_non_strict_capture(self) -> None:
        report, failures = audit_corpus(frontier_source_cases())
        self.assertEqual(report["strict_export_success_rate"], 0.0)
        self.assertEqual(report["validated_non_strict_success_rate"], 1.0)
        self.assertEqual(report["complete_graph_success_rate"], 1.0)
        self.assertEqual(report["capture_quality_counts"], {"non_strict_validated": 1})
        self.assertEqual(report["tensor_nodes"], report["encoded_tensor_nodes"])
        self.assertTrue(
            any(
                failure.backend == "torch_export_strict"
                for failure in failures
            )
        )
        self.assertNotIn(
            "/tmp/torch_geometric",
            "\n".join(
                text
                for failure in failures
                for text in (failure.message, *failure.traceback_tail)
            ),
        )


if __name__ == "__main__":
    unittest.main()
