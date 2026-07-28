from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.coarsen_v3 import coarsen_graph
from perfseer_v3.features import build_graph_features
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.training import (
    DatasetGateReport,
    EncoderPretrainer,
    TARGET_NAMES,
    TrainingConfigV3,
    TrainingGateError,
    TrainingSampleV3,
    assert_training_ready,
    checkpoint_metadata,
    encoder_pretrain_step,
    fit_binary_temperature_calibration,
    fit_linear_calibration,
    fit_uncertainty_calibration,
    run_tiny_training_smoke,
)
from perfseer_v3.training_runner import (
    TRAINING_MANIFEST_VERSION,
    TrainingManifestV3,
    assert_distillation_compatible,
    materialize_training_samples,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


TEACHER_CONFIG = (
    SRC / "perfseer_v3" / "configs" / "train_hardware_teacher" / "v3_teacher.yaml"
)
STUDENT_CONFIG = (
    SRC / "perfseer_v3" / "configs" / "train_deploy_model" / "v3_student.yaml"
)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = capture_export(_Tiny(), (torch.randn(2, 4),))
        if not result.success or result.graph is None:
            raise AssertionError(result.failures)
        cls.features = build_graph_features(result.graph)

    def test_teacher_and_student_configs_use_independent_v3_versions(self) -> None:
        teacher = TrainingConfigV3.load(TEACHER_CONFIG)
        student = TrainingConfigV3.load(STUDENT_CONFIG)
        self.assertEqual(teacher.run["model_release"], "perfseer_v3_teacher")
        self.assertEqual(student.run["model_release"], "perfseer_v3_student")
        self.assertEqual(teacher.features["graph_ir_version"], "perfseer_ir_v3")
        self.assertEqual(teacher.features["feature_schema_version"], "perfseer_graph_v3")
        self.assertEqual(teacher.features["op_registry_version"], "perfseer_aten_ops_v3")
        self.assertEqual(teacher.training["initialization"], "random_v3")
        self.assertEqual(student.training["teacher_model_release"], "perfseer_v3_teacher")
        self.assertGreater(teacher.model["hidden"], student.model["hidden"])
        teacher_model_config = teacher.model_config(
            OperationRegistry.load(),
            self.features.layout,
            hidden=32,
            num_blocks=1,
            dropout=0.0,
        )
        self.assertEqual(teacher_model_config.num_outputs, 6)
        self.assertEqual(teacher_model_config.node_identity_fusion, "additive")

    def test_production_training_gate_blocks_unmeasured_bootstrap_registry(self) -> None:
        config = TrainingConfigV3.load(TEACHER_CONFIG)
        report = DatasetGateReport(
            strict_capture_rate=1.0,
            complete_encoding_rate=1.0,
            unknown_gpu_time_fraction=None,
            source_group_isolated=True,
            measured_gpu_time=False,
            dataset_fingerprint="dataset",
            split_fingerprint="split",
        )
        with self.assertRaisesRegex(TrainingGateError, "not training-approved"):
            assert_training_ready(config, report, OperationRegistry.load())

    def test_tiny_pretrain_teacher_and_distillation_smoke(self) -> None:
        samples = [
            TrainingSampleV3(
                self.features,
                torch.tensor([10.0, 20.0, 25.0, 100.0, 110.0, 30.0]),
            ),
            TrainingSampleV3(
                self.features,
                torch.tensor([12.0, 22.0, 27.0, 120.0, 130.0, 35.0]),
                oom=1.0,
                oom_stage=5,
                peak_live_bytes=4096.0,
                domain_weight=0.5,
            ),
        ]
        result = run_tiny_training_smoke(samples)
        self.assertGreaterEqual(result.pretrain_loss, 0)
        self.assertGreaterEqual(result.teacher_loss, 0)
        self.assertGreaterEqual(result.student_loss, 0)
        self.assertEqual(result.teacher.config.num_oom_stages, 7)
        self.assertNotEqual(
            result.teacher.config.hidden,
            result.student.config.hidden,
        )

    def test_encoder_pretraining_heads_persist_across_steps(self) -> None:
        samples = [
            TrainingSampleV3(self.features, torch.ones(6)),
        ]
        smoke = run_tiny_training_smoke(samples)
        pretrainer = EncoderPretrainer(smoke.teacher)
        optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)
        head_id = id(pretrainer.family_head.weight)
        parameter_groups = len(optimizer.param_groups)
        first = encoder_pretrain_step(pretrainer, samples, optimizer)
        second = encoder_pretrain_step(pretrainer, samples, optimizer)
        self.assertGreaterEqual(first, 0)
        self.assertGreaterEqual(second, 0)
        self.assertEqual(id(pretrainer.family_head.weight), head_id)
        self.assertEqual(len(optimizer.param_groups), parameter_groups)

    def test_calibration_and_checkpoint_metadata_contract(self) -> None:
        prediction = torch.tensor([[1.0] * 6, [2.0] * 6, [3.0] * 6])
        target = prediction * 2 + 3
        calibration = fit_linear_calibration(prediction, target)
        torch.testing.assert_close(calibration.apply(prediction), target)
        oom_logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        oom_target = torch.tensor([0.0, 0.0, 1.0, 1.0])
        oom_calibration = fit_binary_temperature_calibration(
            oom_logits, oom_target
        )
        self.assertGreater(oom_calibration.temperature, 0)
        calibrated_probability = oom_calibration.apply_probability(oom_logits)
        self.assertTrue(((calibrated_probability >= 0) & (calibrated_probability <= 1)).all())
        self.assertTrue(torch.isfinite(calibrated_probability).all())
        uncertainty_calibration = fit_uncertainty_calibration(
            prediction,
            target,
            torch.zeros_like(prediction),
        )
        calibrated_log_variance = uncertainty_calibration.apply_log_variance(
            torch.zeros_like(prediction)
        )
        self.assertEqual(calibrated_log_variance.shape, prediction.shape)
        self.assertTrue(torch.isfinite(calibrated_log_variance).all())
        config = TrainingConfigV3.load(TEACHER_CONFIG)
        report = DatasetGateReport(1.0, 1.0, 0.0, True, True, "dataset", "split")
        smoke = run_tiny_training_smoke(
            [TrainingSampleV3(self.features, torch.ones(6))]
        )
        metadata = checkpoint_metadata(
            config=config,
            model_config=smoke.teacher.config,
            sample=self.features,
            registry=OperationRegistry.load(),
            dataset_gate=report,
            coarsening_sha256="coarsening",
        )
        self.assertEqual(metadata["target_names"], list(TARGET_NAMES))
        self.assertFalse(metadata["v2_checkpoint_loaded"])
        self.assertEqual(metadata["initialization"], "random_v3")
        self.assertEqual(len(metadata["feature_schema_sha256"]), 64)
        self.assertEqual(len(metadata["operator_registry_sha256"]), 64)

    def test_training_manifest_enforces_grouping_and_train_only_normalization(self) -> None:
        registry = OperationRegistry.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, batch in enumerate((2, 3, 4)):
                capture = capture_export(
                    _Tiny(),
                    (torch.randn(batch, 4),),
                    options=CaptureOptions(target_hardware_id="test_gpu"),
                )
                self.assertTrue(capture.success, capture.failures)
                assert capture.graph is not None
                graph = coarsen_graph(capture.graph, registry=registry)
                graph_path = graph.save(root / f"graph-{index}.json")
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "graph_path": graph_path.name,
                        "split": ("train", "validation", "test")[index],
                        "source_group": f"family-{index}",
                        "graph_signature": graph.graph_sha256,
                        "hardware_id": "test_gpu",
                        "target": [float(index + 1)] * 6,
                    }
                )
            payload = {
                "manifest_version": TRAINING_MANIFEST_VERSION,
                "dataset_gate": {
                    "strict_capture_rate": 1.0,
                    "complete_encoding_rate": 1.0,
                    "unknown_gpu_time_fraction": None,
                    "source_group_isolated": True,
                    "measured_gpu_time": False,
                    "dataset_fingerprint": "pending",
                    "split_fingerprint": "pending",
                },
                "deployment": {
                    "target_hardware_id": "test_gpu",
                    "hardware_allowlist": ["test_gpu"],
                    "precision_allowlist": ["float32"],
                    "capture_quality_allowlist": ["strict"],
                    "optimizer_allowlist": ["adamw"],
                    "scheduler_allowlist": ["none", "cosine_with_warmup"],
                    "training_mode_allowlist": ["training"],
                },
                "samples": rows,
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            initial = TrainingManifestV3.load(path)
            payload["dataset_gate"]["split_fingerprint"] = initial.split_fingerprint
            payload["dataset_gate"]["dataset_fingerprint"] = initial.dataset_fingerprint
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = TrainingManifestV3.load(path)
            samples, normalization = materialize_training_samples(
                manifest, registry=registry
            )
            self.assertEqual({name: len(value) for name, value in samples.items()}, {
                "train": 1,
                "validation": 1,
                "test": 1,
            })
            self.assertEqual(normalization.split_name, "train")
            self.assertEqual(normalization.split_fingerprint, manifest.split_fingerprint)
            self.assertEqual(
                samples["validation"][0].features.metadata["normalization_sha256"],
                normalization.sha256,
            )

            payload["samples"][1]["source_group"] = "family-0"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TrainingGateError, "leaks across"):
                TrainingManifestV3.load(path)

    def test_manifest_and_distillation_enforce_one_gpu_pair(self) -> None:
        rows = [
            {
                "sample_id": f"sample-{index}",
                "graph_path": f"graph-{index}.json",
                "split": split,
                "source_group": f"family-{index}",
                "graph_signature": f"signature-{index}",
                "hardware_id": "rtx_5090",
                "target": [1.0] * 6,
            }
            for index, split in enumerate(("train", "validation", "test"))
        ]
        payload = {
            "manifest_version": TRAINING_MANIFEST_VERSION,
            "dataset_gate": {
                "strict_capture_rate": 1.0,
                "complete_encoding_rate": 1.0,
                "unknown_gpu_time_fraction": 0.0,
                "source_group_isolated": True,
                "measured_gpu_time": True,
                "dataset_fingerprint": "dataset",
                "split_fingerprint": "split",
            },
            "deployment": {
                "target_hardware_id": "rtx_5090",
                "hardware_allowlist": ["rtx_5090"],
                "precision_allowlist": ["float32", "mixed"],
                "capture_quality_allowlist": ["strict"],
                "optimizer_allowlist": ["meuon", "AdamW"],
                "scheduler_allowlist": ["none", "Cosine-With-Warmup"],
                "training_mode_allowlist": ["training"],
            },
            "samples": rows,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = TrainingManifestV3.load(path)
            self.assertEqual(manifest.target_hardware_id, "rtx_5090")
            self.assertEqual(manifest.deployment["optimizer_allowlist"], ["muon", "adamw"])
            self.assertEqual(
                manifest.deployment["scheduler_allowlist"],
                ["none", "cosine_with_warmup"],
            )

            payload["deployment"]["hardware_allowlist"].append("h100")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TrainingGateError, "exactly one GPU"):
                TrainingManifestV3.load(path)
            payload["deployment"]["hardware_allowlist"] = ["rtx_5090"]

            payload["samples"][2]["hardware_id"] = "h100"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TrainingGateError, "every training row"):
                TrainingManifestV3.load(path)

        teacher_metadata = SimpleNamespace(
            model_release="perfseer_v3_teacher",
            target_hardware_id="h100",
            dataset_fingerprint="dataset",
            split_fingerprint="split",
            normalization_sha256="normalization",
        )
        with self.assertRaisesRegex(TrainingGateError, "different GPU types"):
            assert_distillation_compatible(
                teacher_metadata,
                manifest,
                normalization_sha256="normalization",
            )


if __name__ == "__main__":
    unittest.main()
