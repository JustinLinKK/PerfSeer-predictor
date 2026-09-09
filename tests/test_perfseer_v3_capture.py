from __future__ import annotations

import operator
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.coverage_corpus import smoke_cases
from perfseer_v3.source import SourceModelSpecV3, capture_source


@torch.library.custom_op("perfseer_v3_tests::triple", mutates_args=())
def _triple(x: torch.Tensor) -> torch.Tensor:
    return x * 3


@_triple.register_fake
def _triple_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _MultiOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.topk(x, 2, dim=-1)
        return values + 1, indices


class _KeywordNested(nn.Module):
    def forward(
        self,
        values: dict[str, torch.Tensor],
        *,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output = values["left"] + values["right"] * scale
        return output, [output.transpose(0, 1)]


class _Dynamic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.special.i0(x).sum(dim=1)


class _Custom(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.perfseer_v3_tests.triple(x)


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Stochastic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.dropout(x, p=0.25, training=True)
            + torch.rand_like(x)
        )


class _SelectiveDecomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return torch.nn.functional.scaled_dot_product_attention(
            projected.unsqueeze(1),
            projected.unsqueeze(1),
            projected.unsqueeze(1),
        ).squeeze(1).transpose(-2, -1)


class CaptureTests(unittest.TestCase):
    def test_custom_operation_fake_registration_passes_opcheck(self) -> None:
        result = torch.library.opcheck(_triple, (torch.randn(2, 3),))
        self.assertTrue(all(status == "SUCCESS" for status in result.values()), result)

    def test_smoke_cases_capture_all_tensor_nodes_and_roles(self) -> None:
        for case in smoke_cases():
            with self.subTest(case=case.case_id):
                model, args, kwargs = case.build()
                result = capture_export(model, args, kwargs)
                self.assertTrue(result.success, result.failures)
                graph = result.graph
                assert graph is not None
                self.assertEqual(
                    graph.coverage.tensor_nodes_seen,
                    graph.coverage.tensor_nodes_encoded,
                )
                self.assertTrue(all(node.output_tensor_count > 0 for node in graph.nodes))
                self.assertTrue(any(edge.tensor_role == "model_input" for edge in graph.tensor_edges))
                self.assertTrue(any(edge.tensor_role == "model_output" for edge in graph.tensor_edges))
                if case.case_id == "smoke_residual_conv":
                    self.assertTrue(any(edge.tensor_role == "parameter" for edge in graph.tensor_edges))
                    self.assertGreater(graph.global_features.total_parameter_bytes, 0)

    def test_multi_output_slots_and_repeated_edges_are_preserved(self) -> None:
        result = capture_export(_MultiOutput(), (torch.randn(2, 5),))
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        topk = next(node for node in graph.nodes if node.raw_target == "aten::topk")
        self.assertEqual(topk.output_tensor_count, 2)
        topk_edges = [
            edge for edge in graph.tensor_edges if edge.producer_node_id == topk.node_id
        ]
        self.assertEqual({edge.producer_output_index for edge in topk_edges}, {0, 1})
        self.assertGreaterEqual(len(topk_edges), 2)
        self.assertEqual(len({edge.edge_id for edge in graph.tensor_edges}), len(graph.tensor_edges))

    def test_nested_positional_and_keyword_inputs_capture(self) -> None:
        args = ({"left": torch.randn(2, 3), "right": torch.randn(2, 3)},)
        kwargs = {"scale": torch.tensor(2.0)}
        result = capture_export(_KeywordNested(), args, kwargs)
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        self.assertEqual(graph.input_signature["args"]["kind"], "tuple")
        self.assertEqual(graph.input_signature["kwargs"]["kind"], "mapping")
        self.assertEqual(
            sum(edge.tensor_role == "model_output" for edge in graph.tensor_edges),
            2,
        )

    def test_dynamic_constraints_and_unknown_aten_are_retained(self) -> None:
        batch = torch.export.Dim("batch", min=1, max=8)
        result = capture_export(
            _Dynamic(),
            (torch.randn(3, 4),),
            dynamic_shapes={"x": {0: batch}},
        )
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        self.assertTrue(graph.dynamic_constraints)
        long_tail = next(node for node in graph.nodes if "i0" in node.raw_target)
        self.assertEqual(long_tail.exact_op_id, 0)
        self.assertGreater(long_tail.op_hash_bucket, 0)
        self.assertGreater(graph.coverage.unknown_operations, 0)

    def test_registered_custom_operation_uses_nonzero_generic_identity(self) -> None:
        result = capture_export(_Custom(), (torch.randn(2, 3),))
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        custom = next(node for node in graph.nodes if "perfseer_v3_tests" in node.raw_target)
        self.assertEqual(custom.exact_op_id, 0)
        self.assertEqual(custom.family, "unknown_or_custom")
        self.assertGreater(custom.op_hash_bucket, 0)
        self.assertTrue(custom.flags["custom"])
        self.assertEqual(graph.coverage.custom_operations, 1)
        self.assertGreater(graph.global_features.unknown_byte_fraction, 0)
        self.assertGreater(graph.global_features.unknown_cost_fraction, 0)

    def test_selective_decomposition_preserves_expensive_semantics(self) -> None:
        result = capture_export(
            _SelectiveDecomposition(),
            (torch.randn(2, 4, 8),),
        )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        targets = {node.raw_target for node in result.graph.nodes}
        self.assertIn("aten::linear", targets)
        self.assertIn("aten::scaled_dot_product_attention", targets)
        metadata = result.graph.metadata
        summary = metadata["semantic_summary_pre_decomposition"]
        report = metadata["selective_decomposition"]
        self.assertGreater(summary["tensor_nodes"], 0)
        self.assertEqual(
            report["functionalization"],
            "torch_export_run_decompositions",
        )
        self.assertIn(
            "aten::scaled_dot_product_attention",
            report["preserved_semantic_targets"],
        )
        self.assertIn("aten::transpose.int", report["requested_targets"])
        self.assertIn("aten::permute", targets)

    def test_zero_node_identity_graph_is_valid(self) -> None:
        result = capture_export(_Identity(), (torch.randn(2, 3),))
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        self.assertEqual(graph.nodes, ())
        self.assertEqual(len(graph.tensor_edges), 1)
        edge = graph.tensor_edges[0]
        self.assertIsNone(edge.producer_node_id)
        self.assertIsNone(edge.consumer_node_id)
        self.assertEqual(edge.tensor_role, "model_output")
        self.assertEqual(edge.source_name, "x")

    def test_non_strict_requires_three_successful_replays(self) -> None:
        model = _Dynamic()
        args = (torch.randn(2, 4),)
        real_export = torch.export.export

        def strict_failure_then_export(*call_args, **call_kwargs):
            if call_kwargs.get("strict", True):
                raise RuntimeError("forced strict failure")
            return real_export(*call_args, **call_kwargs)

        with mock.patch("torch.export.export", side_effect=strict_failure_then_export):
            result = capture_export(model, args, options=CaptureOptions(replay_samples=3))
        self.assertTrue(result.success, result.failures)
        graph = result.graph
        assert graph is not None
        self.assertEqual(graph.coverage.capture_quality, "non_strict_validated")
        self.assertEqual(graph.coverage.replay_samples, 3)
        self.assertTrue(graph.coverage.replay_validated)
        self.assertEqual(result.failures[0].mode, "strict")

    def test_non_strict_stochastic_replay_uses_identical_rng_state(self) -> None:
        model = _Stochastic()
        args = (torch.randn(2, 4),)
        real_export = torch.export.export

        def strict_failure_then_export(*call_args, **call_kwargs):
            if call_kwargs.get("strict", True):
                raise RuntimeError("forced strict failure")
            return real_export(*call_args, **call_kwargs)

        with mock.patch(
            "torch.export.export",
            side_effect=strict_failure_then_export,
        ):
            result = capture_export(
                model,
                args,
                options=CaptureOptions(replay_samples=3),
            )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        self.assertEqual(
            result.graph.coverage.capture_quality,
            "non_strict_validated",
        )

    def test_replay_mismatch_returns_structured_failure(self) -> None:
        class Stateful(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return x + self.calls

        model = Stateful()
        args = (torch.ones(2),)
        real_export = torch.export.export

        def strict_failure_then_export(*call_args, **call_kwargs):
            if call_kwargs.get("strict", True):
                raise RuntimeError("forced strict failure")
            return real_export(*call_args, **call_kwargs)

        with mock.patch("torch.export.export", side_effect=strict_failure_then_export):
            result = capture_export(model, args, options=CaptureOptions(replay_samples=3))
        self.assertFalse(result.success)
        self.assertEqual(result.failures[-1].stage, "replay_validation")

    def test_source_entrypoint_defaults_to_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.py"
            source.write_text(
                textwrap.dedent(
                    """
                    import torch
                    import torch.nn as nn

                    class Model(nn.Module):
                        def forward(self, x, *, bias):
                            return torch.sigmoid(x + bias)
                    """
                ),
                encoding="utf-8",
            )
            spec = SourceModelSpecV3(
                source_path=source,
                entry="Model",
                positional_inputs=(torch.randn(2, 3),),
                keyword_inputs={"bias": torch.randn(2, 3)},
            )
            result = capture_source(spec)
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        self.assertEqual(result.graph.capture_backend, "torch_export")
        self.assertEqual(result.graph.capture_mode, "strict")

    def test_source_entrypoint_supports_dotted_module_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.py"
            source.write_text(
                textwrap.dedent(
                    """
                    import torch
                    import torch.nn as nn

                    class Model(nn.Module):
                        def forward(self, x):
                            return torch.relu(x)

                    class Namespace:
                        pass

                    namespace = Namespace()
                    namespace.instance = Model()
                    """
                ),
                encoding="utf-8",
            )
            result = capture_source(
                SourceModelSpecV3(
                    source_path=source,
                    entry="namespace.instance",
                    positional_inputs=(torch.randn(2, 3),),
                )
            )
        self.assertTrue(result.success, result.failures)
        assert result.graph is not None
        self.assertEqual(
            [node.raw_target for node in result.graph.nodes],
            ["aten::relu"],
        )


if __name__ == "__main__":
    unittest.main()
