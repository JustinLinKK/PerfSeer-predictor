from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.graph_ir_v3 import (
    CoverageQuality,
    Estimate,
    GraphGlobalFeatures,
    GraphIRV3,
    GraphValidationError,
    OperationNodeV3,
    TensorEdgeV3,
)
from perfseer_v3.op_registry import DEFAULT_REGISTRY_PATH, OperationRegistry, RegistryValidationError
from perfseer_v3.schema import NODE_CONTINUOUS_FIELDS, build_feature_schema, validate_feature_schema


def _sample_graph(registry: OperationRegistry) -> GraphIRV3:
    schema = build_feature_schema(registry)
    add = registry.resolve("aten::add.Tensor")
    node = OperationNodeV3(
        node_id="n0",
        raw_target=add.raw_target,
        canonical_op_id=add.canonical_id,
        family_id=add.family_id,
        family=add.family,
        phase="forward",
        exact_op_id=add.exact_id,
        op_hash_bucket=add.hash_bucket,
        flags={"broadcast": True},
        input_tensor_count=2,
        output_tensor_count=1,
        flops=Estimate(8, "shape_formula", 1.0),
        bytes_read=Estimate(64, "exact_formula", 1.0),
        bytes_written=Estimate(32, "exact_formula", 1.0),
    )
    edges = (
        TensorEdgeV3(
            edge_id="e0",
            producer_node_id=None,
            consumer_node_id="n0",
            producer_output_index=0,
            consumer_input_index=0,
            tensor_role="model_input",
            shape=(2, 4),
            rank=2,
            dtype="float32",
            element_width_bytes=4,
            numel=8,
            tensor_bytes=32,
            stride=(4, 1),
            memory_format="contiguous",
        ),
        TensorEdgeV3(
            edge_id="e1",
            producer_node_id=None,
            consumer_node_id="n0",
            producer_output_index=0,
            consumer_input_index=1,
            tensor_role="model_input",
            shape=(2, 4),
            rank=2,
            dtype="float32",
            element_width_bytes=4,
            numel=8,
            tensor_bytes=32,
            stride=(4, 1),
            memory_format="contiguous",
        ),
        TensorEdgeV3(
            edge_id="e2",
            producer_node_id="n0",
            consumer_node_id=None,
            producer_output_index=0,
            consumer_input_index=0,
            tensor_role="model_output",
            shape=(2, 4),
            rank=2,
            dtype="float32",
            element_width_bytes=4,
            numel=8,
            tensor_bytes=32,
            stride=(4, 1),
            memory_format="contiguous",
        ),
    )
    return GraphIRV3.create(
        operator_registry_sha256=registry.sha256,
        feature_schema_sha256=schema["feature_schema_sha256"],
        capture_backend="torch_export",
        capture_mode="strict",
        pytorch_version="test",
        source_fingerprint="source",
        model_fingerprint="model",
        input_signature={"args": [{"shape": [2, 4], "dtype": "float32"}]},
        dynamic_constraints={},
        training_mode=False,
        precision="float32",
        optimizer_config={},
        nodes=(node,),
        tensor_edges=edges,
        global_features=GraphGlobalFeatures(
            operation_nodes=1,
            tensor_edges=3,
            total_flops=8,
            total_activation_bytes=96,
        ),
        coverage=CoverageQuality(tensor_nodes_seen=1, tensor_nodes_encoded=1),
    )


class RegistryIRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = OperationRegistry.load()

    def test_registry_alias_and_unknown_identity(self) -> None:
        add = self.registry.resolve("aten::add_.Tensor")
        self.assertEqual(add.canonical_id, "aten.add.Tensor")
        self.assertEqual(add.exact_id, 4)
        self.assertIn("in_place", add.flags)
        self.assertNotIn("in_place", self.registry.resolve("aten::add.Tensor").flags)
        custom = self.registry.resolve("my_extension::fused_magic")
        self.assertEqual(custom.exact_id, 0)
        self.assertEqual(custom.family_id, 0)
        self.assertGreater(custom.hash_bucket, 0)
        self.assertTrue(custom.is_custom)
        self.assertEqual(custom.hash_bucket, self.registry.resolve(custom.raw_target).hash_bucket)
        self.assertFalse(self.registry.training_approved)

    def test_registry_rejects_duplicate_and_unmeasured_training_approval(self) -> None:
        payload = yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(payload)
        duplicate["operations"][1]["raw"] = duplicate["operations"][0]["raw"]
        with self.assertRaisesRegex(RegistryValidationError, "duplicate operation target"):
            OperationRegistry(duplicate)
        unmeasured = copy.deepcopy(payload)
        unmeasured["selection"]["training_approved"] = True
        with self.assertRaisesRegex(RegistryValidationError, "GPU-time report"):
            OperationRegistry(unmeasured)

    def test_registry_loader_rejects_duplicate_yaml_mapping_keys(self) -> None:
        payload = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8").replace(
            "hash_buckets: 1024",
            "hash_buckets: 1024\nhash_buckets: 2048",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.yaml"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(
                RegistryValidationError,
                "duplicate YAML mapping key 'hash_buckets'",
            ):
                OperationRegistry.load(path)

    def test_graph_round_trip_preserves_multiedges_and_slots(self) -> None:
        graph = _sample_graph(self.registry)
        encoded = graph.to_json()
        decoded = GraphIRV3.from_dict(json.loads(encoded))
        self.assertEqual(decoded, graph)
        self.assertEqual(decoded.graph_sha256, graph.graph_sha256)
        input_slots = [edge.consumer_input_index for edge in decoded.tensor_edges[:2]]
        self.assertEqual(input_slots, [0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = graph.save(Path(temporary) / "graph.json")
            self.assertEqual(GraphIRV3.load(path), graph)

    def test_invalid_units_or_silent_node_loss_fail_closed(self) -> None:
        graph = _sample_graph(self.registry)
        raw = graph.to_dict()
        raw["tensor_edges"][0]["tensor_bytes"] = 31
        with self.assertRaisesRegex(GraphValidationError, "byte units"):
            GraphIRV3.from_dict(raw)
        raw = graph.to_dict()
        raw["coverage"]["tensor_nodes_seen"] = 2
        with self.assertRaisesRegex(GraphValidationError, "not completely encoded"):
            GraphIRV3.from_dict(raw)

    def test_feature_schema_hash_covers_layout_and_registry(self) -> None:
        schema = build_feature_schema(self.registry)
        validate_feature_schema(schema, self.registry)
        changed = build_feature_schema(
            self.registry,
            node_continuous_fields=(*NODE_CONTINUOUS_FIELDS, "new_cost"),
        )
        self.assertNotEqual(schema["feature_schema_sha256"], changed["feature_schema_sha256"])
        corrupted = copy.deepcopy(schema)
        corrupted["ordered_layout"]["node_continuous"].append("unhashed_change")
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_feature_schema(corrupted, self.registry)


if __name__ == "__main__":
    unittest.main()
