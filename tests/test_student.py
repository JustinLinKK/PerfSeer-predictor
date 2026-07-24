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
from perfseer_source_converter import converter


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "registry.json"
ARTIFACT_PATH = ROOT / "models" / "nvidia_a10" / "student_a10_cpu.torchscript.pt"
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
        runtime = StudentRuntime(ARTIFACT_PATH)
        output = runtime.predict(encoded)
        self.assertEqual(output.shape, (6,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(runtime.predict_train_mem_mb(encoded), 0)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in encoded.as_tuple()))
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.assertEqual(after, before)

    def test_registry_selects_only_matching_a10(self) -> None:
        registry = ModelRegistry(REGISTRY_PATH)
        selected = registry.select(HardwareInfo("NVIDIA A10", "8.6", 23028))
        self.assertEqual(selected.artifact_path, ARTIFACT_PATH)
        with self.assertRaises(ModelUnavailableError):
            registry.select(HardwareInfo("NVIDIA GeForce RTX 5090", "12.0", 32607))

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

    def test_registry_hash_matches_artifact(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        expected = payload["models"][0]["artifact"]["sha256"]
        actual = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

    def test_registry_rejects_corrupt_artifact_hash(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            registry_path = Path(temporary) / "registry.json"
            artifact_path = Path(temporary) / "student.pt"
            artifact_path.write_bytes(ARTIFACT_PATH.read_bytes())
            payload["models"][0]["artifact"]["path"] = "student.pt"
            payload["models"][0]["artifact"]["sha256"] = "0" * 64
            registry_path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ModelRegistry(registry_path)
            with self.assertRaisesRegex(ModelUnavailableError, "hash mismatch"):
                registry.select(HardwareInfo("NVIDIA A10", "8.6", 23028))

    def test_exactly_one_neural_artifact_is_retained(self) -> None:
        artifacts = [
            path
            for path in (ROOT / "models").rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".pth", ".ckpt"}
        ]
        self.assertEqual(artifacts, [ARTIFACT_PATH])


if __name__ == "__main__":
    unittest.main()
