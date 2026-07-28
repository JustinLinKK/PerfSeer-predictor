from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.artifact import (
    ArtifactIntegrityError,
    ArtifactMetadataV3,
    ArtifactRegistryV3,
    TargetTransformV3,
    load_checkpoint_artifact,
    save_checkpoint_artifact,
    sha256_file,
)
from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.coarsen_v3 import COARSENING_POLICY_SHA256
from perfseer_v3.deployment_export import export_torchscript_student
from perfseer_v3.features import build_graph_features
from perfseer_v3.migration import CanaryPolicy, canary_decision, compare_shadow
from perfseer_v3.model import SeerNetV3, SeerNetV3Config
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.runtime import PerfSeerV3Runtime, RESULT_STATUSES
from perfseer_v3.training import TARGET_NAMES


class _Known(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class _Unknown(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.special.i0(x)


class _MixedPrecision(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).relu().float()


class ArtifactRuntimeTests(unittest.TestCase):
    def make_graph(self, model: nn.Module):
        result = capture_export(
            model,
            (torch.randn(2, 4),),
            options=CaptureOptions(target_hardware_id="test_gpu"),
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        return result.graph

    def make_artifact(self, directory: Path):
        registry = OperationRegistry.load()
        graph = self.make_graph(_Known())
        sample = build_graph_features(graph, registry=registry)
        config = SeerNetV3Config.from_registry(
            registry,
            sample.layout,
            hidden=16,
            num_blocks=1,
            exact_embedding_dim=8,
            family_embedding_dim=8,
            hash_embedding_dim=8,
            phase_embedding_dim=4,
            dtype_embedding_dim=4,
            dropout=0.0,
        )
        model = SeerNetV3(config).eval()
        metadata = ArtifactMetadataV3(
            model_release="perfseer_v3_student",
            graph_ir_version="perfseer_ir_v3",
            feature_schema_version="perfseer_graph_v3",
            feature_schema_sha256=sample.layout.feature_schema_sha256,
            operator_registry_version="perfseer_aten_ops_v3",
            operator_registry_sha256=registry.sha256,
            ordered_feature_layout=asdict(sample.layout),
            normalization_sha256=None,
            coarsening_policy_sha256=COARSENING_POLICY_SHA256,
            target_names=TARGET_NAMES,
            target_transform=TargetTransformV3(),
            label_schema_version="scheduler_resource_label_v3",
            target_hardware_id="test_gpu",
            hardware_allowlist=("test_gpu",),
            precision_allowlist=("float32",),
            capture_quality_allowlist=("strict",),
            optimizer_allowlist=("sgd", "adam", "adamw"),
            scheduler_allowlist=("none", "cosine_with_warmup"),
            training_mode_allowlist=("inference",),
            dataset_fingerprint="dataset",
            split_fingerprint="split",
            pytorch_version=torch.__version__,
            cuda_build_version=torch.version.cuda,
            model_config=config.to_dict(),
            minimum_confidence=0.01,
        )
        artifact_path = save_checkpoint_artifact(
            directory / "student.pt",
            model=model,
            metadata=metadata,
        )
        registry_path = directory / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "registry_version": "perfseer_v3_artifacts_v2",
                    "artifacts": [
                        {
                            "artifact_id": "test_student",
                            "path": artifact_path.name,
                            "sha256": sha256_file(artifact_path),
                            "model_release": "perfseer_v3_student",
                            "target_hardware_id": "test_gpu",
                            "hardware_allowlist": ["test_gpu"],
                            "precision_allowlist": ["float32"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return graph, artifact_path, registry_path

    def test_registry_integrity_load_and_ok_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, registry_path = self.make_artifact(Path(temporary))
            record, selected_path = ArtifactRegistryV3(registry_path).select(
                hardware_id="test_gpu",
                precision="float32",
            )
            self.assertEqual(selected_path, artifact_path.resolve())
            loaded = load_checkpoint_artifact(selected_path)
            self.assertEqual(loaded.metadata.model_release, "perfseer_v3_student")
            runtime = PerfSeerV3Runtime(loaded)
            result = runtime.predict_graph(graph)
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.recommended_fallback)
        self.assertEqual(len(result.prediction), 6)
        self.assertTrue(all(torch.isfinite(torch.tensor(result.prediction))))

    def test_runtime_applies_validation_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, _ = self.make_artifact(Path(temporary))
            loaded = load_checkpoint_artifact(artifact_path)
            runtime = PerfSeerV3Runtime(loaded)
            baseline = runtime.predict_graph(graph)
            self.assertEqual(baseline.status, "ok")
            assert baseline.uncertainty is not None
            assert baseline.oom_probability is not None

            loaded.calibration.update(
                {
                    "oom_temperature": 2.0,
                    "uncertainty_log_variance_offset": [math.log(4.0)] * 6,
                }
            )
            calibrated = runtime.predict_graph(graph)

        self.assertEqual(calibrated.status, "ok")
        assert calibrated.uncertainty is not None
        assert calibrated.oom_probability is not None
        torch.testing.assert_close(
            torch.tensor(calibrated.uncertainty),
            torch.tensor(baseline.uncertainty) * 2.0,
        )
        baseline_logit = math.log(
            baseline.oom_probability / (1.0 - baseline.oom_probability)
        )
        expected_oom_probability = torch.sigmoid(torch.tensor(baseline_logit / 2.0))
        self.assertAlmostEqual(
            calibrated.oom_probability,
            float(expected_oom_probability),
            places=6,
        )

    def test_verified_torchscript_student_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph, artifact_path, _ = self.make_artifact(root)
            graph_path = graph.save(root / "graph.json")
            report = export_torchscript_student(
                artifact_path=artifact_path,
                graph_path=graph_path,
                output_path=root / "student.ts",
            )
            self.assertTrue(report["verified"])
            self.assertTrue(Path(report["torchscript_path"]).is_file())
            self.assertTrue(Path(report["sidecar_path"]).is_file())
            self.assertTrue(
                all(row["allclose"] for row in report["comparisons"].values())
            )

    def test_artifact_corruption_and_schema_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, registry_path = self.make_artifact(Path(temporary))
            artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ArtifactIntegrityError, "hash mismatch"):
                ArtifactRegistryV3(registry_path).select(
                    hardware_id="test_gpu",
                    precision="float32",
                )
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, _ = self.make_artifact(Path(temporary))
            runtime = PerfSeerV3Runtime(artifact_path)
            mismatched = replace(graph, feature_schema_sha256="0" * 64)
            result = runtime.predict_graph(mismatched)
        self.assertEqual(result.status, "schema_mismatch")
        self.assertEqual(result.recommended_fallback, "branch_profile")

    def test_runtime_rejects_a_graph_from_another_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, _ = self.make_artifact(Path(temporary))
            runtime = PerfSeerV3Runtime(artifact_path)
            mismatched = replace(
                graph,
                metadata={**graph.metadata, "target_hardware_id": "h100"},
            )
            result = runtime.predict_graph(mismatched)
        self.assertEqual(result.status, "hardware_mismatch")
        self.assertEqual(result.recommended_fallback, "branch_profile")

    def test_runtime_checks_actual_mixed_precision_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, artifact_path, _ = self.make_artifact(Path(temporary))
            runtime = PerfSeerV3Runtime(artifact_path)
            mixed = self.make_graph(_MixedPrecision())
            result = runtime.predict_graph(mixed)
        self.assertEqual(result.status, "unsupported_precision")
        self.assertEqual(result.recommended_fallback, "branch_profile")

    def test_runtime_optimizer_and_scheduler_policies_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, _ = self.make_artifact(Path(temporary))
            loaded = load_checkpoint_artifact(artifact_path)
            graph = replace(
                graph,
                training_mode=True,
                optimizer_config={"name": "muon", "lr": 0.02},
                training_config={
                    "scheduler": {"name": "cosine_with_warmup"},
                    "epochs": 20,
                },
            )
            optimizer_blocked = PerfSeerV3Runtime(
                replace(
                    loaded,
                    metadata=replace(
                        loaded.metadata,
                        training_mode_allowlist=("training",),
                        optimizer_allowlist=("adamw",),
                    ),
                )
            ).predict_graph(graph)
            scheduler_blocked = PerfSeerV3Runtime(
                replace(
                    loaded,
                    metadata=replace(
                        loaded.metadata,
                        training_mode_allowlist=("training",),
                        optimizer_allowlist=("muon",),
                        scheduler_allowlist=("none",),
                    ),
                )
            ).predict_graph(graph)
        self.assertEqual(optimizer_blocked.status, "unsupported_optimizer")
        self.assertEqual(scheduler_blocked.status, "unsupported_scheduler")

    def test_unknown_capture_and_training_mode_states_recommend_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, artifact_path, _ = self.make_artifact(Path(temporary))
            runtime = PerfSeerV3Runtime(artifact_path)
            unknown = self.make_graph(_Unknown())
            unknown_result = runtime.predict_graph(unknown)
            self.assertEqual(unknown_result.status, "ood_low_confidence")
            unsupported_capture = replace(
                unknown,
                coverage=replace(unknown.coverage, capture_quality="estimated"),
            )
            capture_result = runtime.predict_graph(unsupported_capture)
            self.assertEqual(capture_result.status, "unsupported_capture")
            training = replace(
                unknown,
                training_mode=True,
                optimizer_config={"name": "adamw"},
            )
            training_result = runtime.predict_graph(training)
            self.assertEqual(training_result.status, "unsupported_training_mode")
        for result in (unknown_result, capture_result, training_result):
            self.assertEqual(result.recommended_fallback, "branch_profile")

    def test_shadow_canary_and_result_state_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph, artifact_path, _ = self.make_artifact(Path(temporary))
            runtime = PerfSeerV3Runtime(artifact_path)
            result = runtime.predict_graph(graph)
        comparison = compare_shadow(
            v2_prediction=(0.0,) * 6,
            v3_result=result,
            fallback_prediction=(1.0,) * 6,
        )
        self.assertEqual(len(comparison.v2_v3_absolute_difference), 6)
        conservative = canary_decision(result, CanaryPolicy(minimum_confidence=1.0))
        self.assertEqual(conservative.route, "fallback")
        permissive = canary_decision(result, CanaryPolicy(minimum_confidence=0.0))
        self.assertEqual(permissive.route, "perfseer_v3")
        self.assertEqual(permissive.rollback_target, "perfseer_v2")
        self.assertEqual(
            set(RESULT_STATUSES),
            {
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
            },
        )


if __name__ == "__main__":
    unittest.main()
