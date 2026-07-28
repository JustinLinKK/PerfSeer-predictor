from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.features import build_graph_features


class _AsymmetricConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=(3, 5), padding=(1, 2), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SequenceLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Broadcast(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _BranchAndView(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transposed = x.transpose(0, 1)
        left = transposed + 1
        right = transposed * 2
        return left + right


class _TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _BufferMutation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("counter", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counter.add_(1)
        return x + self.counter


class _RankGeneralConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 6, 3, padding=1)
        self.depthwise2 = nn.Conv2d(4, 4, (3, 5), padding=(1, 2), groups=4)
        self.grouped3 = nn.Conv3d(4, 8, 3, padding=1, groups=2)
        self.transpose2 = nn.ConvTranspose2d(4, 3, 3, padding=1)

    def forward(self, x1, x2, x3):
        return self.conv1(x1), self.depthwise2(x2), self.grouped3(x3), self.transpose2(x2)


class _Recurrent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(5, 7, batch_first=True)

    def forward(self, x: torch.Tensor):
        return self.gru(x)


class CostLivenessTrainingTests(unittest.TestCase):
    def test_asymmetric_convolution_and_sequence_linear_formulas(self) -> None:
        conv_result = capture_export(_AsymmetricConv(), (torch.randn(2, 3, 6, 8),))
        self.assertTrue(conv_result.success, conv_result.failures)
        assert conv_result.graph is not None
        conv = next(node for node in conv_result.graph.nodes if node.raw_target == "aten::conv2d")
        expected_macs = conv.output_numel * 3 * 3 * 5
        self.assertEqual(conv.macs.value, expected_macs)
        self.assertEqual(conv.flops.value, 2 * expected_macs + conv.output_numel)
        self.assertEqual(conv.parameter_bytes, (4 * 3 * 3 * 5 + 4) * 4)
        self.assertGreater(conv.arithmetic_intensity_flops_per_byte, 0)

        linear_result = capture_export(_SequenceLinear(), (torch.randn(2, 11, 7),))
        self.assertTrue(linear_result.success, linear_result.failures)
        assert linear_result.graph is not None
        linear = next(node for node in linear_result.graph.nodes if node.raw_target == "aten::linear")
        self.assertEqual(linear.output_numel, 2 * 11 * 5)
        self.assertEqual(linear.macs.value, linear.output_numel * 7)
        self.assertEqual(linear.flops.value, 2 * linear.macs.value + linear.output_numel)

    def test_broadcast_bytes_use_actual_input_and_output_dtypes(self) -> None:
        x = torch.randn(2, 3, dtype=torch.float64)
        y = torch.randn(1, 3, dtype=torch.float64)
        result = capture_export(_Broadcast(), (x, y))
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        add = next(node for node in result.graph.nodes if node.raw_target == "aten::add.Tensor")
        self.assertEqual(add.input_numel, 9)
        self.assertEqual(add.output_numel, 6)
        self.assertEqual(add.input_bytes, 9 * 8)
        self.assertEqual(add.output_bytes, 6 * 8)
        self.assertEqual(add.bytes_read.value, add.input_bytes)
        self.assertEqual(add.bytes_written.value, add.output_bytes)

    def test_rank_general_grouped_and_transposed_convolution_formulas(self) -> None:
        result = capture_export(
            _RankGeneralConvolution(),
            (
                torch.randn(2, 4, 9),
                torch.randn(2, 4, 7, 9),
                torch.randn(2, 4, 5, 6, 7),
            ),
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        nodes = {node.raw_target: node for node in result.graph.nodes}
        expected_channels_and_kernel = {
            "aten::conv1d": 4 * 3,
            "aten::conv2d": 1 * 3 * 5,
            "aten::conv3d": 2 * 3 * 3 * 3,
            "aten::conv_transpose2d.input": 4 * 3 * 3,
        }
        for raw_target, per_output_macs in expected_channels_and_kernel.items():
            with self.subTest(raw_target=raw_target):
                node = nodes[raw_target]
                self.assertEqual(node.macs.value, node.output_numel * per_output_macs)
                self.assertEqual(
                    node.flops.value,
                    2 * node.macs.value + node.output_numel,
                )
                self.assertEqual(node.bytes_read.value, node.input_bytes)
                self.assertEqual(node.bytes_written.value, node.output_bytes)

    def test_recurrent_formula_is_nonzero_and_confidence_bearing(self) -> None:
        result = capture_export(_Recurrent(), (torch.randn(2, 4, 5),))
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        recurrent = next(node for node in result.graph.nodes if node.raw_target == "aten::gru.input")
        self.assertGreater(recurrent.macs.value, 0)
        self.assertGreater(recurrent.flops.value, recurrent.macs.value)
        self.assertEqual(recurrent.macs.method, "shape_formula")
        self.assertGreater(recurrent.macs.confidence, 0)

    def test_alias_aware_liveness_tracks_view_reuse(self) -> None:
        result = capture_export(_BranchAndView(), (torch.randn(2, 3),))
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        graph = result.graph
        transpose = next(
            node
            for node in graph.nodes
            if node.raw_target in {"aten::transpose.int", "aten::permute"}
        )
        transpose_edges = [
            edge for edge in graph.tensor_edges if edge.producer_node_id == transpose.node_id
        ]
        self.assertGreaterEqual(len(transpose_edges), 2)
        self.assertEqual(len({edge.alias_group for edge in transpose_edges}), 1)
        self.assertGreater(max(edge.last_use_distance for edge in transpose_edges), 0)
        self.assertGreater(graph.global_features.peak_live_activation_bytes, 0)
        self.assertTrue(any(node.live_bytes_after > 0 for node in graph.nodes))

    def test_aot_training_capture_tags_loss_backward_and_optimizer(self) -> None:
        model = _TrainModel()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        result = capture_training_graph(
            model,
            (x,),
            target=target,
            loss_fn=torch.nn.functional.mse_loss,
            optimizer_name="adamw",
        )
        self.assertTrue(result.success, result.failures)
        self.assertEqual(result.backward_backend, "aot_autograd_joint")
        assert result.graph is not None
        phases = {node.phase for node in result.graph.nodes}
        self.assertEqual(phases, {"forward", "loss", "backward", "optimizer"})
        self.assertEqual(result.graph.coverage.backward_capture_quality, "strict")
        self.assertGreater(
            sum(node.phase == "backward" for node in result.graph.nodes),
            1,
        )
        self.assertEqual(
            result.graph.global_features.total_optimizer_state_bytes,
            2 * result.graph.global_features.total_parameter_bytes,
        )
        self.assertTrue(any(edge.tensor_role == "gradient" for edge in result.graph.tensor_edges))
        self.assertTrue(any(edge.tensor_role == "optimizer_state" for edge in result.graph.tensor_edges))

    def test_optimizer_variants_and_training_controls_remain_explicit(self) -> None:
        result = capture_training_graph(
            _TrainModel(),
            (torch.randn(2, 4),),
            target=torch.randn(2, 3),
            loss_fn=torch.nn.functional.mse_loss,
            optimizer_name="adamw",
            optimizer_config={
                "amsgrad": True,
                "foreach": True,
                "fused": False,
                "gradient_accumulation_steps": 4,
                "gradient_clip_norm": 1.0,
                "loss_scale": 1024.0,
                "activation_checkpointing": True,
            },
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        optimizer = next(
            node for node in result.graph.nodes if node.phase == "optimizer"
        )
        self.assertTrue(optimizer.flags["foreach"])
        self.assertFalse(optimizer.flags["fused"])
        self.assertGreater(optimizer.estimated_workspace_bytes.value, 0)
        self.assertEqual(
            result.graph.global_features.total_optimizer_state_bytes,
            3 * result.graph.global_features.total_parameter_bytes,
        )
        features = build_graph_features(result.graph)
        globals_by_name = dict(
            zip(features.layout.global_continuous_fields, features.u_cont[0].tolist())
        )
        self.assertEqual(globals_by_name["gradient_accumulation_steps"], 4.0)
        self.assertEqual(globals_by_name["gradient_clip_norm"], 1.0)
        self.assertEqual(globals_by_name["loss_scale"], 1024.0)
        self.assertEqual(globals_by_name["activation_checkpointing"], 1.0)
        self.assertEqual(globals_by_name["optimizer_foreach"], 1.0)
        self.assertEqual(globals_by_name["optimizer_fused"], 0.0)

    def test_muon_composite_and_learning_rate_schedule_are_encoded(self) -> None:
        result = capture_training_graph(
            _TrainModel(),
            (torch.randn(2, 4),),
            target=torch.randn(2, 3),
            loss_fn=torch.nn.functional.mse_loss,
            optimizer_name="meuon",
            optimizer_config={
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 0.1,
                "nesterov": True,
                "ns_steps": 5,
                "components": [
                    {"name": "muon", "parameter_fraction": 0.8, "lr": 0.02},
                    {"name": "adamw", "parameter_fraction": 0.2, "lr": 0.0003},
                ],
                "parameter_groups": [
                    {"lr": 0.02, "weight_decay": 0.1},
                    {"lr": 0.0003, "weight_decay": 0.01},
                ],
            },
            scheduler_name="cosine_with_warmup",
            scheduler_config={
                "warmup_steps": 100,
                "total_steps": 1000,
                "eta_min": 1e-5,
                "num_cycles": 0.5,
            },
            training_config={
                "epochs": 20,
                "current_epoch": 3,
                "steps_per_epoch": 50,
                "current_step": 150,
            },
            target_hardware_id="test_gpu",
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        optimizer_node = next(
            node for node in result.graph.nodes if node.phase == "optimizer"
        )
        self.assertEqual(optimizer_node.canonical_op_id, "perfseer.optimizer.muon")
        self.assertEqual(result.graph.optimizer_config["name"], "muon")
        features = build_graph_features(result.graph)
        self.assertEqual(features.metadata["optimizer"], "muon")
        self.assertEqual(features.metadata["optimizer_family"], "composite")
        self.assertEqual(features.metadata["scheduler"], "cosine_with_warmup")
        self.assertGreater(features.optimizer_hash_id.item(), 0)
        self.assertGreater(features.scheduler_hash_id.item(), 0)
        globals_by_name = dict(
            zip(features.layout.global_continuous_fields, features.u_cont[0].tolist())
        )
        self.assertEqual(globals_by_name["total_epochs"], 20.0)
        self.assertEqual(globals_by_name["current_epoch"], 3.0)
        self.assertEqual(globals_by_name["total_training_steps"], 1000.0)
        self.assertEqual(globals_by_name["current_training_step"], 150.0)
        self.assertAlmostEqual(globals_by_name["learning_rate_current"], 0.02)
        self.assertAlmostEqual(globals_by_name["learning_rate_min"], 1e-5)
        self.assertEqual(globals_by_name["scheduler_warmup_steps"], 100.0)
        self.assertEqual(globals_by_name["optimizer_ns_steps"], 5.0)
        self.assertEqual(globals_by_name["optimizer_parameter_group_count"], 2.0)
        self.assertEqual(globals_by_name["optimizer_component_count"], 2.0)
        self.assertEqual(globals_by_name["optimizer_nesterov"], 1.0)

    def test_backward_failure_uses_explicit_analytical_nodes(self) -> None:
        model = _TrainModel()
        with mock.patch(
            "perfseer_v3.capture_training._capture_aot_joint",
            side_effect=RuntimeError("unsupported custom autograd"),
        ):
            result = capture_training_graph(
                model,
                (torch.randn(2, 4),),
                target=torch.randn(2, 3),
                loss_fn=torch.nn.functional.mse_loss,
                optimizer_name="sgd",
                optimizer_config={"momentum": 0.9},
            )
        self.assertTrue(result.success, result.failures)
        self.assertEqual(result.backward_backend, "analytical_fallback")
        assert result.graph is not None
        self.assertEqual(result.graph.coverage.backward_capture_quality, "estimated")
        backward = next(node for node in result.graph.nodes if node.phase == "backward")
        self.assertTrue(backward.flags["estimated"])
        self.assertGreater(backward.saved_for_backward_bytes, 0)
        self.assertGreater(result.graph.global_features.total_saved_for_backward_bytes, 0)
        self.assertEqual(result.failures[-1].stage, "backward_capture")

    def test_export_normalizes_buffer_mutation_instead_of_dropping_it(self) -> None:
        result = capture_export(
            _BufferMutation(),
            (torch.ones(2),),
            options=CaptureOptions(training_mode=True),
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        self.assertTrue(any(edge.tensor_role == "buffer" for edge in result.graph.tensor_edges))
        self.assertEqual(
            result.graph.coverage.tensor_nodes_seen,
            result.graph.coverage.tensor_nodes_encoded,
        )


if __name__ == "__main__":
    unittest.main()
