from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import capture_export
from perfseer_v3.features import batch_graph_features, build_graph_features
from perfseer_v3.model import SeerNetV3, SeerNetV3Config, graph_batch_tensors
from perfseer_v3.op_registry import OperationRegistry


class _UnknownOnly(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.special.i0(x)


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Isolated(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.relu(x), torch.sigmoid(y)


def _features(model: nn.Module, args: tuple[torch.Tensor, ...]):
    result = capture_export(model, args)
    if not result.success or result.graph is None:
        raise AssertionError(result.failures)
    return build_graph_features(result.graph)


class ModelTests(unittest.TestCase):
    def make_model(self, batch) -> SeerNetV3:
        config = SeerNetV3Config.from_registry(
            OperationRegistry.load(),
            batch.layout,
            hidden=32,
            num_blocks=2,
            exact_embedding_dim=16,
            family_embedding_dim=8,
            hash_embedding_dim=8,
            phase_embedding_dim=4,
            dtype_embedding_dim=4,
            dropout=0.0,
        )
        return SeerNetV3(config).eval()

    def test_unknown_only_graph_has_finite_prediction_and_low_confidence(self) -> None:
        sample = _features(_UnknownOnly(), (torch.randn(2, 4),))
        batch = batch_graph_features([sample])
        model = self.make_model(batch)
        with torch.no_grad():
            output = model(*graph_batch_tensors(batch))
        self.assertEqual(output.prediction.shape, (1, 6))
        self.assertEqual(output.log_variance.shape, (1, 6))
        self.assertEqual(output.oom_logit.shape, (1, 1))
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertTrue(torch.isfinite(output.log_variance).all())
        self.assertEqual(output.confidence.item(), 0.0)

    def test_empty_and_isolated_graphs_batch(self) -> None:
        empty = _features(_Identity(), (torch.randn(2, 4),))
        isolated = _features(_Isolated(), (torch.randn(2, 4), torch.randn(2, 4)))
        self.assertEqual(empty.x_cont.size(0), 0)
        self.assertGreater(isolated.edge_index.size(1), 0)
        self.assertTrue(torch.equal(isolated.edge_index[0], isolated.edge_index[1]))
        batch = batch_graph_features([empty, isolated])
        model = self.make_model(batch)
        with torch.no_grad():
            output = model(*graph_batch_tensors(batch))
        self.assertEqual(output.prediction.shape, (2, 6))
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertEqual(set(batch.batch.tolist()), {1})

    def test_torchscript_and_export_round_trip(self) -> None:
        sample = _features(_Isolated(), (torch.randn(2, 4), torch.randn(2, 4)))
        batch = batch_graph_features([sample, sample])
        tensors = graph_batch_tensors(batch)
        model = self.make_model(batch)
        with torch.no_grad():
            baseline = model(*tensors)
        scripted = torch.jit.script(model)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            scripted.save(str(path))
            reloaded = torch.jit.load(str(path))
            with torch.no_grad():
                actual = reloaded(*tensors)
        for expected_tensor, actual_tensor in zip(baseline, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        exported = torch.export.export(model, tensors, strict=True)
        with torch.no_grad():
            exported_output = exported.module()(*tensors)
        for expected_tensor, actual_tensor in zip(baseline, exported_output):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_hierarchical_identity_changes_for_unknown_hash_bucket(self) -> None:
        sample = _features(_UnknownOnly(), (torch.randn(2, 4),))
        batch = batch_graph_features([sample])
        model = self.make_model(batch)
        tensors = list(graph_batch_tensors(batch))
        changed = list(tensors)
        changed[3] = (changed[3] + 1) % model.config.num_hash_buckets
        with torch.no_grad():
            baseline = model(*tensors).prediction
            altered = model(*changed).prediction
        self.assertFalse(torch.equal(baseline, altered))

    def test_overload_edge_slot_hardware_optimizer_scheduler_embeddings_affect_identity(self) -> None:
        sample = _features(_Isolated(), (torch.randn(2, 4), torch.randn(2, 4)))
        batch = batch_graph_features([sample])
        model = self.make_model(batch)
        tensors = list(graph_batch_tensors(batch))
        changes = {
            4: model.config.num_hash_buckets,
            17: model.config.num_slot_buckets,
            27: model.config.num_hardware_buckets,
            30: model.config.num_optimizer_families,
            31: model.config.num_optimizer_hash_buckets,
            32: model.config.num_schedulers,
            33: model.config.num_scheduler_families,
            34: model.config.num_scheduler_hash_buckets,
        }
        with torch.no_grad():
            baseline = model(*tensors).prediction
            for index, cardinality in changes.items():
                changed = list(tensors)
                changed[index] = (changed[index] + 1) % cardinality
                altered = model(*changed).prediction
                self.assertFalse(torch.equal(baseline, altered), index)

    def test_accumulation_dtype_embedding_affects_identity(self) -> None:
        sample = _features(_UnknownOnly(), (torch.randn(2, 4),))
        batch = batch_graph_features([sample])
        model = self.make_model(batch)
        tensors = list(graph_batch_tensors(batch))
        changed = list(tensors)
        accumulation_index = 8
        changed[accumulation_index] = (
            changed[accumulation_index] + 1
        ) % model.config.num_dtypes
        with torch.no_grad():
            baseline = model(*tensors).prediction
            altered = model(*changed).prediction
        self.assertFalse(torch.equal(baseline, altered))

    def test_phase_aware_pooling_preserves_contract_and_changes_representation(self) -> None:
        sample = _features(_Isolated(), (torch.randn(2, 4), torch.randn(2, 4)))
        batch = batch_graph_features([sample])
        registry = OperationRegistry.load()
        base_config = SeerNetV3Config.from_registry(
            registry,
            batch.layout,
            hidden=32,
            num_blocks=1,
            dropout=0.0,
            pooling_mode="existing",
        )
        phase_config = SeerNetV3Config.from_registry(
            registry,
            batch.layout,
            hidden=32,
            num_blocks=1,
            dropout=0.0,
            pooling_mode="phase_aware",
        )
        torch.manual_seed(9)
        existing = SeerNetV3(base_config).eval()
        torch.manual_seed(9)
        phase_aware = SeerNetV3(phase_config).eval()
        with torch.no_grad():
            existing_output = existing(*graph_batch_tensors(batch))
            phase_output = phase_aware(*graph_batch_tensors(batch))
        self.assertEqual(existing_output.prediction.shape, (1, 6))
        self.assertEqual(phase_output.prediction.shape, (1, 6))
        self.assertEqual(phase_output.oom_stage_logits.shape, (1, 7))
        self.assertEqual(phase_output.phase_embedding.shape, (1, 4, 32))
        self.assertFalse(
            torch.equal(existing_output.graph_embedding, phase_output.graph_embedding)
        )


if __name__ == "__main__":
    unittest.main()
