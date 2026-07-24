from __future__ import annotations

import hashlib
import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

from perfseer_student import (
    HardwareInfo,
    ModelRegistry,
    ModelUnavailableError,
    StudentRuntime,
    UnsupportedStudentOperationError,
    encode_source,
)
from perfseer_student.features import OP_VOCAB
from perfseer_student.export import export_torchscript
from perfseer_student.model import SeerNetConfig, SeerNetMulti
from perfseer_source_converter import converter


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "registry.json"
A10_ARTIFACT_PATH = ROOT / "models" / "nvidia_a10" / "student_a10_cpu.torchscript.pt"
BLACKWELL_ARTIFACT_PATH = (
    ROOT
    / "models"
    / "nvidia_rtx_pro_6000_blackwell"
    / "student_rtx_pro_6000_blackwell_cpu.torchscript.pt"
)
SOURCE_PATH = ROOT / "tests" / "fixtures" / "tiny_conv.py"
BATCH_NORM_SOURCE_PATH = ROOT / "tests" / "fixtures" / "tiny_batch_norm.py"
OPERATION_REPORT_PATH = ROOT / "docs" / "student_operation_coverage_and_dataset_redesign.md"


class StudentPredictorTest(unittest.TestCase):
    def test_source_to_cpu_torchscript_inference(self) -> None:
        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        encoded = encode_source(
            SOURCE_PATH,
            "build_model",
            [[2, 3, 32, 32]],
            constructor_kwargs={"channels": 4},
        )
        self.assertEqual(encoded.x.shape[1], 53)
        self.assertEqual(encoded.edge_attr.shape[1], 3)
        self.assertEqual(encoded.u.shape, (1, 40))
        for artifact_path in (A10_ARTIFACT_PATH, BLACKWELL_ARTIFACT_PATH):
            with self.subTest(artifact=artifact_path.name):
                runtime = StudentRuntime(artifact_path)
                output = runtime.predict(encoded)
                self.assertEqual(output.shape, (6,))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(runtime.predict_train_mem_mb(encoded), 0)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in encoded.as_tuple()))
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.assertEqual(after, before)

    def test_registry_selects_matching_a10_and_blackwell(self) -> None:
        registry = ModelRegistry(REGISTRY_PATH)
        a10 = registry.select(HardwareInfo("NVIDIA A10", "8.6", 23028))
        self.assertEqual(a10.artifact_path, A10_ARTIFACT_PATH)
        for name in (
            "NVIDIA RTX PRO 6000 Blackwell",
            "NVIDIA RTX 6000 Blackwell",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        ):
            with self.subTest(name=name):
                blackwell = registry.select(HardwareInfo(name, "12.0", 98304))
                self.assertEqual(blackwell.artifact_path, BLACKWELL_ARTIFACT_PATH)
        with self.assertRaises(ModelUnavailableError):
            registry.select(HardwareInfo("NVIDIA GeForce RTX 5090", "12.0", 32607))
        with self.assertRaises(ModelUnavailableError):
            registry.select(HardwareInfo("NVIDIA RTX PRO 6000 Blackwell", "12.1", 98304))
        with self.assertRaises(ModelUnavailableError):
            registry.select(HardwareInfo("NVIDIA RTX PRO 6000 Blackwell", "12.0", 80000))

    def test_converter_label_without_student_slot_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedStudentOperationError,
            "student operation vocabulary does not cover: BatchNormalization",
        ):
            encode_source(BATCH_NORM_SOURCE_PATH, "build_model", [[2, 3, 16, 16]])

    def test_operation_report_matches_converter_and_student_vocabularies(self) -> None:
        tree = ast.parse(inspect.getsource(converter._classify_node))
        converter_operations = {
            value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Return) and node.value is not None
            for value in ast.walk(node.value)
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        }
        converter_only = converter_operations - set(OP_VOCAB)
        student_only = set(OP_VOCAB) - converter_operations
        self.assertEqual(
            converter_only,
            {
                "AveragePool",
                "BatchNormalization",
                "Bmm",
                "ConvTranspose",
                "Div",
                "GroupNormalization",
                "HardSigmoid",
                "HardSwish",
                "MatMul",
                "Mul",
                "MultiHeadAttention",
                "RNN",
                "Reduce",
                "Reshape",
                "Sigmoid",
                "Sub",
                "Tanh",
                "Transpose",
            },
        )
        self.assertEqual(
            student_only,
            {
                "Attention",
                "DetectorHead",
                "GraphAttention",
                "GraphMessage",
                "SegmentationHead",
                "TabularFeature",
            },
        )
        report = OPERATION_REPORT_PATH.read_text(encoding="utf-8")
        for operation in converter_only | student_only:
            self.assertIn(f"`{operation}`", report)

    def test_registry_hashes_match_artifacts(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        for record in payload["models"]:
            with self.subTest(model_id=record["id"]):
                artifact_path = (REGISTRY_PATH.parent / record["artifact"]["path"]).resolve()
                expected = record["artifact"]["sha256"]
                actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_registry_rejects_corrupt_artifact_hash(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            registry_path = Path(temporary) / "registry.json"
            artifact_path = Path(temporary) / "student.pt"
            artifact_path.write_bytes(A10_ARTIFACT_PATH.read_bytes())
            payload["models"][0]["artifact"]["path"] = "student.pt"
            payload["models"][0]["artifact"]["sha256"] = "0" * 64
            registry_path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ModelRegistry(registry_path)
            with self.assertRaisesRegex(ModelUnavailableError, "hash mismatch"):
                registry.select(HardwareInfo("NVIDIA A10", "8.6", 23028))

    def test_legacy_global_checkpoint_exports_to_current_raw_schema(self) -> None:
        cfg = SeerNetConfig(
            node_dim=53,
            edge_dim=3,
            global_dim=14,
            hidden=8,
            num_blocks=1,
            num_outputs=6,
            head_hidden=8,
        )
        model = SeerNetMulti(cfg)
        stats = {
            "x_mean": [0.0] * 30,
            "x_std": [1.0] * 30,
            "e_mean": [0.0] * 3,
            "e_std": [1.0] * 3,
            "g_mean": [0.0] * 10,
            "g_std": [1.0] * 10,
            "y_mean": [0.0] * 6,
            "y_std": [1.0] * 6,
        }
        example = (
            torch.zeros((2, 53)),
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.zeros((1, 3)),
            torch.zeros((1, 40)),
            torch.zeros(2, dtype=torch.long),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "legacy.pt"
            artifact_path = Path(temporary) / "legacy.torchscript.pt"
            torch.save(
                {
                    "cfg": cfg.to_dict(),
                    "model": model.state_dict(),
                    "stats": stats,
                    "targets": [
                        "train_util",
                        "train_mem",
                        "train_time",
                        "infer_util",
                        "infer_mem",
                        "infer_time",
                    ],
                },
                checkpoint_path,
            )
            exported = export_torchscript(checkpoint_path, artifact_path, example)
            with torch.inference_mode():
                baseline = exported(*example)
                changed = list(example)
                changed[3] = changed[3].clone()
                changed[3][:, 10:36] = 999.0
                ignored_globals = exported(*tuple(changed))
            torch.testing.assert_close(ignored_globals, baseline, rtol=0, atol=0)

    def test_only_registered_torchscript_artifacts_are_retained(self) -> None:
        artifacts = [
            path
            for path in (ROOT / "models").rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".pth", ".ckpt"}
        ]
        self.assertEqual(
            set(artifacts),
            {A10_ARTIFACT_PATH, BLACKWELL_ARTIFACT_PATH},
        )


if __name__ == "__main__":
    unittest.main()
