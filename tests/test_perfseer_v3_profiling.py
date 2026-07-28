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
from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.profiling import (
    ProfileOptions,
    ProfileRecord,
    ProfileWorkload,
    WorkloadIdentityError,
    input_value_fingerprint,
    profile_workload,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class _NoParameters(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + x


class _RandomOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.dropout(
            x,
            p=0.5,
            training=True,
        ) + torch.rand_like(x)


class ProfilingTests(unittest.TestCase):
    def test_source_first_profile_retains_raw_boundaries_and_round_trips(self) -> None:
        model = _Tiny().eval()
        args = (torch.randn(2, 4),)
        capture = capture_export(model, args)
        workload = ProfileWorkload(model, args, {}, capture)
        sample_counter = {"value": 0}

        def sampler() -> dict[str, float]:
            sample_counter["value"] += 1
            return {
                "sm_util_percent": float(sample_counter["value"]),
                "other_process_memory_bytes": 0.0,
            }

        record = profile_workload(
            workload,
            options=ProfileOptions(
                warmup_steps=1,
                measured_steps=3,
                nvml_sample_interval_s=1e-6,
            ),
            nvml_sampler=sampler,
            metadata={"seed": 7},
        )
        self.assertEqual(record.status, "ok")
        self.assertTrue(record.correctness_validated)
        self.assertEqual(record.measured_steps_completed, 3)
        self.assertEqual(len(record.measured_step_ms), 3)
        self.assertTrue(all(value > 0 for value in record.measured_step_ms))
        boundaries = [sample.boundary for sample in record.raw_samples]
        self.assertIn("warmup_start", boundaries)
        self.assertIn("measured_start", boundaries)
        self.assertIn("profile_end", boundaries)
        self.assertIn("nvml_poll", boundaries)
        self.assertTrue(all(sample.nvml for sample in record.raw_samples))
        self.assertIn("optional_versions", record.environment)
        self.assertEqual(record.metadata["nvml_sample_interval_s"], 1e-6)
        self.assertEqual(record.identity.graph_sha256, capture.graph.graph_sha256)
        with tempfile.TemporaryDirectory() as temporary:
            path = record.save(Path(temporary) / "profile.json")
            reloaded = ProfileRecord.load(path)
        self.assertEqual(reloaded, record)
        self.assertEqual(reloaded.record_sha256, record.record_sha256)

    def test_substitute_model_instance_is_rejected_before_execution(self) -> None:
        model = _Tiny().eval()
        args = (torch.randn(2, 4),)
        capture = capture_export(model, args)
        substitute = _Tiny().eval()
        substitute.load_state_dict(model.state_dict())
        workload = ProfileWorkload(substitute, args, {}, capture)
        with self.assertRaisesRegex(WorkloadIdentityError, "exact model instance"):
            profile_workload(workload, options=ProfileOptions(warmup_steps=0, measured_steps=1))

    def test_parameter_change_is_rejected_before_execution(self) -> None:
        model = _Tiny().eval()
        args = (torch.randn(2, 4),)
        capture = capture_export(model, args)
        with torch.no_grad():
            model.linear.weight.add_(1)
        workload = ProfileWorkload(model, args, {}, capture)
        with self.assertRaisesRegex(WorkloadIdentityError, "parameter fingerprint"):
            profile_workload(workload, options=ProfileOptions(warmup_steps=0, measured_steps=1))

    def test_value_fingerprint_is_structure_and_content_sensitive(self) -> None:
        args = ({"x": torch.ones(2)},)
        first = input_value_fingerprint(args, {"flag": True})
        second = input_value_fingerprint(args, {"flag": True})
        changed = input_value_fingerprint(({"x": torch.zeros(2)},), {"flag": True})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_changed_input_signature_is_rejected_before_execution(self) -> None:
        model = _Tiny().eval()
        capture = capture_export(model, (torch.randn(2, 4),))
        workload = ProfileWorkload(model, (torch.randn(3, 4),), {}, capture)
        with self.assertRaisesRegex(WorkloadIdentityError, "input signature"):
            profile_workload(workload, options=ProfileOptions(warmup_steps=0, measured_steps=1))

    def test_training_profile_uses_matching_loss_backward_optimizer_graph(self) -> None:
        model = _Tiny().train()
        args = (torch.randn(2, 4),)
        target = torch.randn(2, 3)
        capture = capture_training_graph(
            model,
            args,
            target=target,
            loss_fn=torch.nn.functional.mse_loss,
            optimizer_name="adamw",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        workload = ProfileWorkload(
            model,
            args,
            {},
            capture,
            mode="training",
            loss_fn=torch.nn.functional.mse_loss,
            target=target,
            optimizer=optimizer,
        )
        record = profile_workload(
            workload,
            options=ProfileOptions(warmup_steps=0, measured_steps=1),
        )
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.identity.mode, "training")
        self.assertEqual(record.identity.optimizer_config["name"], "adamw")

    def test_gradient_only_training_profile_needs_no_fake_optimizer(self) -> None:
        model = _NoParameters().train()
        args = (torch.randn(2, 4, requires_grad=True),)

        def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (output - target).square().mean()

        target = torch.zeros(2, 4)
        capture = capture_training_graph(
            model,
            args,
            target=target,
            loss_fn=loss_fn,
            optimizer_name="none",
        )
        workload = ProfileWorkload(
            model,
            args,
            {},
            capture,
            mode="training",
            loss_fn=loss_fn,
            target=target,
        )
        record = profile_workload(
            workload,
            options=ProfileOptions(warmup_steps=1, measured_steps=2),
        )
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.identity.optimizer_config["name"], "none")
        self.assertNotIn("optimizer", {node.phase for node in capture.graph.nodes})
        self.assertTrue(torch.isfinite(args[0].grad).all())

    def test_random_callable_correctness_replays_identical_rng_state(self) -> None:
        model = _RandomOutput().eval()
        args = (torch.randn(2, 4),)
        capture = capture_export(model, args)
        before = torch.random.get_rng_state().clone()
        record = profile_workload(
            ProfileWorkload(model, args, {}, capture),
            options=ProfileOptions(warmup_steps=0, measured_steps=1),
        )
        self.assertEqual(record.status, "ok")
        after_profile = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(before)
        with torch.no_grad():
            model(*args)
        expected_after_one_measured_step = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(after_profile, expected_after_one_measured_step))

    def test_compiled_execution_is_explicit_and_excludes_warmup(self) -> None:
        model = _Tiny().eval()
        args = (torch.randn(2, 4),)
        capture = capture_export(model, args)
        record = profile_workload(
            ProfileWorkload(model, args, {}, capture),
            options=ProfileOptions(
                warmup_steps=1,
                measured_steps=2,
                execution_mode="compile",
                compile_backend="eager",
            ),
        )
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.identity.execution_mode, "compile")
        self.assertEqual(record.metadata["compile_backend"], "eager")
        self.assertTrue(record.metadata["compile_warmup_excluded"])
        with self.assertRaisesRegex(ValueError, "warmup step"):
            ProfileOptions(
                warmup_steps=0,
                measured_steps=1,
                execution_mode="compile",
                compile_warmup_excluded=True,
            )


if __name__ == "__main__":
    unittest.main()
