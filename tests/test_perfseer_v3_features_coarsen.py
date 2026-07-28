from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.coarsen_v3 import coarsen_graph
from perfseer_v3.features import (
    apply_normalization,
    batch_graph_features,
    build_graph_features,
    fit_normalization,
    sample_cache_key,
    validate_checkpoint_layout,
)
from perfseer_v3.model import SeerNetV3, SeerNetV3Config
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.schema import DTYPES


class _PointwiseChain(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.sigmoid(torch.relu(x)))


class _CriticalAndBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.relu(self.conv(x))
        left = base + 1
        right = base * 2
        return torch.sigmoid(left + right)


class _LowPrecisionReduction(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y).sum(dim=-1)


class _MixedLayerDtypes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fp32 = nn.Linear(8, 8).float()
        self.bf16 = nn.Linear(8, 4).bfloat16()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fp32(x.float()))
        x = self.bf16(x.to(torch.bfloat16))
        return x.float()


class FeatureCoarsenTests(unittest.TestCase):
    def capture(self, model: nn.Module, args: tuple[torch.Tensor, ...]):
        result = capture_export(model, args)
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        return result.graph

    def test_coarsener_conserves_costs_and_operation_histogram(self) -> None:
        graph = self.capture(_PointwiseChain(), (torch.randn(2, 4),))
        coarsened = coarsen_graph(graph)
        self.assertLess(len(coarsened.nodes), len(graph.nodes))
        self.assertEqual(
            sum(node.flops.value for node in coarsened.nodes),
            sum(node.flops.value for node in graph.nodes),
        )
        self.assertEqual(
            sum(node.bytes_read.value for node in coarsened.nodes),
            sum(node.bytes_read.value for node in graph.nodes),
        )
        region = next(node for node in coarsened.nodes if node.raw_target == "perfseer::coarsened_region")
        histogram = region.normalized_args["exact_operation_histogram"]
        self.assertEqual(sum(histogram.values()), 3)
        self.assertEqual(coarsened.global_features.coarsening_ratio, 1 / 3)
        self.assertGreaterEqual(
            coarsened.global_features.peak_live_activation_bytes,
            graph.global_features.peak_live_activation_bytes,
        )

    def test_coarsener_preserves_critical_and_branch_boundaries(self) -> None:
        graph = self.capture(_CriticalAndBranch(), (torch.randn(2, 3, 8, 8),))
        coarsened = coarsen_graph(graph)
        self.assertTrue(any(node.family == "convolution" for node in coarsened.nodes))
        captured_targets = [node.raw_target for node in coarsened.nodes]
        self.assertIn("aten::conv2d", captured_targets)
        # The fan-out branch prevents the ReLU from merging across either arm.
        relu = next(node for node in coarsened.nodes if node.raw_target == "aten::relu")
        self.assertGreaterEqual(relu.fan_out, 2)

    def test_features_keep_categorical_ids_separate_and_hash_unknowns(self) -> None:
        class Unknown(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.special.i0(x)

        graph = self.capture(Unknown(), (torch.randn(2, 4),))
        sample = build_graph_features(graph)
        sample.validate()
        self.assertEqual(sample.x_cont.dtype, torch.float32)
        self.assertEqual(sample.op_exact_id.dtype, torch.long)
        self.assertEqual(sample.op_exact_id.item(), 0)
        self.assertGreater(sample.op_hash_id.item(), 0)
        self.assertGreater(sample.op_overload_hash_id.item(), 0)
        self.assertEqual(sample.input_dtype_id.dtype, torch.long)
        self.assertEqual(sample.backend_id.dtype, torch.long)
        self.assertEqual(sample.feature_quality_id.dtype, torch.long)
        self.assertEqual(sample.x_cont.shape[1], len(sample.layout.node_continuous_fields))
        self.assertEqual(sample.node_flags.shape[1], len(sample.layout.node_flag_fields))

    def test_external_tensor_roles_and_edge_semantics_are_not_dropped(self) -> None:
        graph = self.capture(_CriticalAndBranch(), (torch.randn(2, 3, 8, 8),))
        sample = build_graph_features(graph)
        external_count = sum(
            edge.producer_node_id is None or edge.consumer_node_id is None
            for edge in graph.tensor_edges
            if edge.producer_node_id is not None or edge.consumer_node_id is not None
        )
        self.assertEqual(
            sample.metadata["external_edges_as_typed_self_loops"],
            external_count,
        )
        self.assertGreater(external_count, 0)
        self.assertEqual(sample.edge_source_slot_id.shape, sample.edge_role_id.shape)
        self.assertEqual(sample.edge_destination_slot_id.shape, sample.edge_role_id.shape)
        self.assertEqual(sample.edge_dtype_id.shape, sample.edge_role_id.shape)
        self.assertEqual(sample.edge_layout_id.shape, sample.edge_role_id.shape)
        self.assertEqual(sample.edge_alias_id.shape, sample.edge_role_id.shape)
        self.assertEqual(
            sample.edge_flags.shape[1],
            len(sample.layout.edge_flag_fields),
        )

    def test_accumulation_dtype_is_distinct_from_tensor_dtype(self) -> None:
        graph = self.capture(
            _LowPrecisionReduction(),
            (
                torch.randn(2, 4, 5, dtype=torch.float16),
                torch.randn(2, 5, 3, dtype=torch.float16),
            ),
        )
        sample = build_graph_features(graph)
        self.assertTrue(all(node.accumulation_dtype == "float32" for node in graph.nodes))
        self.assertTrue(torch.all(sample.dtype_id != sample.accumulation_dtype_id))
        self.assertEqual(sample.accumulation_dtype_id.dtype, torch.long)

    def test_layer_level_fp32_to_bf16_transition_reaches_model_backward(self) -> None:
        result = capture_export(
            _MixedLayerDtypes(),
            (torch.randn(2, 8),),
            options=CaptureOptions(
                precision="mixed",
                target_hardware_id="test_gpu",
            ),
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        graph = result.graph
        self.assertEqual(
            {edge.dtype for edge in graph.tensor_edges},
            {"float32", "bfloat16"},
        )
        sample = build_graph_features(graph)
        self.assertEqual(sample.precision_id.item(), DTYPES.index("mixed"))
        global_values = dict(
            zip(sample.layout.global_continuous_fields, sample.u_cont[0].tolist())
        )
        self.assertEqual(global_values["distinct_floating_dtype_count"], 2.0)
        self.assertGreater(global_values["float32_tensor_fraction"], 0.0)
        self.assertGreater(global_values["bfloat16_tensor_fraction"], 0.0)
        changes_dtype = sample.node_flags[
            :, sample.layout.node_flag_fields.index("changes_dtype")
        ]
        self.assertGreaterEqual(int(changes_dtype.sum()), 2)

        registry = OperationRegistry.load()
        config = SeerNetV3Config.from_registry(
            registry,
            sample.layout,
            hidden=16,
            num_blocks=1,
            dropout=0.0,
        )
        model = SeerNetV3(config)
        output = model.forward_batch(batch_graph_features([sample]))
        self.assertEqual(output.prediction.shape, (1, 6))
        output.prediction.sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_unseen_optimizer_and_scheduler_keep_distinct_hash_identity(self) -> None:
        graph = self.capture(_PointwiseChain(), (torch.randn(2, 4),))
        first = build_graph_features(
            replace(
                graph,
                optimizer_config={"name": "future_optimizer_alpha", "lr": 1e-3},
                training_config={"scheduler": {"name": "future_decay_alpha"}},
            )
        )
        second = build_graph_features(
            replace(
                graph,
                optimizer_config={"name": "future_optimizer_beta", "lr": 1e-3},
                training_config={"scheduler": {"name": "future_decay_beta"}},
            )
        )
        self.assertEqual(first.metadata["optimizer"], "other")
        self.assertEqual(first.metadata["optimizer_family"], "custom")
        self.assertEqual(first.metadata["scheduler"], "other")
        self.assertEqual(first.metadata["scheduler_family"], "custom")
        self.assertNotEqual(first.optimizer_hash_id.item(), second.optimizer_hash_id.item())
        self.assertNotEqual(first.scheduler_hash_id.item(), second.scheduler_hash_id.item())

    def test_normalization_is_train_only_hash_checked_and_reports_clipping(self) -> None:
        graph = self.capture(_PointwiseChain(), (torch.randn(2, 4),))
        sample = build_graph_features(graph)
        with self.assertRaisesRegex(ValueError, "training split"):
            fit_normalization([sample], split_name="validation", split_fingerprint="v")
        stats = fit_normalization(
            [sample],
            split_name="train",
            split_fingerprint="train-fingerprint",
            quantiles=(0.0, 0.5),
        )
        normalized = apply_normalization(sample, stats)
        self.assertTrue(torch.isfinite(normalized.x_cont).all())
        self.assertGreaterEqual(normalized.metadata["clip_frequency"], 0.0)
        self.assertEqual(normalized.metadata["normalization_sha256"], stats.sha256)
        bad = copy.deepcopy(stats)
        object.__setattr__(bad, "feature_schema_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "schema hash"):
            apply_normalization(sample, bad)

    def test_layout_validation_cache_invalidation_and_batching(self) -> None:
        graph = self.capture(_PointwiseChain(), (torch.randn(2, 4),))
        sample = build_graph_features(graph)
        metadata = {
            "feature_schema_sha256": sample.layout.feature_schema_sha256,
            "operator_registry_sha256": sample.layout.operator_registry_sha256,
            "layout_sha256": sample.layout.layout_sha256,
            "node_continuous_fields": list(sample.layout.node_continuous_fields),
            "edge_continuous_fields": list(sample.layout.edge_continuous_fields),
            "global_continuous_fields": list(sample.layout.global_continuous_fields),
            "node_flag_fields": list(sample.layout.node_flag_fields),
            "edge_flag_fields": list(sample.layout.edge_flag_fields),
            "quality_fields": list(sample.layout.quality_fields),
        }
        validate_checkpoint_layout(sample, metadata)
        corrupted = dict(metadata)
        corrupted["node_continuous_fields"] = list(reversed(metadata["node_continuous_fields"]))
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            validate_checkpoint_layout(sample, corrupted)
        first = sample_cache_key(
            sample,
            coarsening_sha256="a",
            split_fingerprint="b",
            normalization_sha256="c",
        )
        changed = sample_cache_key(
            sample,
            coarsening_sha256="changed",
            split_fingerprint="b",
            normalization_sha256="c",
        )
        self.assertNotEqual(first, changed)
        batch = batch_graph_features([sample, sample])
        self.assertEqual(batch.u_cont.shape[0], 2)
        self.assertEqual(batch.x_cont.shape[0], 2 * sample.x_cont.shape[0])
        self.assertEqual(set(batch.batch.tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
