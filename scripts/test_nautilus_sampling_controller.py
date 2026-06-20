#!/usr/bin/env python3
"""Tests for Nautilus sampling controller scheduling logic."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_nautilus_label_sampling_e2e.py")


def load_module():
    spec = importlib.util.spec_from_file_location("run_nautilus_label_sampling_e2e", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NautilusSamplingControllerTests(unittest.TestCase):
    def test_readme_gpu_presets_are_supported(self) -> None:
        module = load_module()

        self.assertEqual(set(module.README_GPU_KEYS), set(module.GPU_PRESETS))

    def test_pending_timeout_switches_to_unused_gpu(self) -> None:
        module = load_module()
        state = module.ControllerState(
            candidate_gpus=["a10", "a40", "a100", "l4", "l40"],
            active_limit=4,
            pending_timeout_seconds=300,
            blacklist_ttl_seconds=21600,
        )
        for key in ["a10", "a40", "a100", "l4"]:
            state.mark_submitted(key, f"job-{key}", now=0.0)

        decision = state.plan_pending_switch("a10", now=301.0, pod_phase="Pending", unschedulable=True)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.old_gpu, "a10")
        self.assertEqual(decision.new_gpu, "l40")
        self.assertEqual(state.gpu_states["a10"].status, "failed")
        self.assertEqual(state.gpu_states["l40"].status, "pending")

    def test_pending_timeout_without_replacement_marks_failed(self) -> None:
        module = load_module()
        state = module.ControllerState(
            candidate_gpus=["a10"],
            active_limit=4,
            pending_timeout_seconds=300,
            blacklist_ttl_seconds=21600,
        )
        state.mark_submitted("a10", "job-a10", now=0.0)

        decision = state.plan_pending_switch("a10", now=301.0, pod_phase="Pending", unschedulable=True)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.old_gpu, "a10")
        self.assertEqual(decision.new_gpu, "")
        self.assertEqual(state.gpu_states["a10"].status, "failed")
        self.assertEqual(state.terminal_count(), 1)

    def test_verified_outputs_are_not_resubmitted(self) -> None:
        module = load_module()
        state = module.ControllerState(
            candidate_gpus=["a10", "a40", "a100", "l4"],
            active_limit=4,
            pending_timeout_seconds=300,
            blacklist_ttl_seconds=21600,
            verified_gpus={"a40"},
        )

        self.assertEqual(state.next_replacement_gpu(), "a10")
        self.assertEqual(state.gpu_states["a40"].status, "succeeded")

    def test_blocked_node_is_excluded_from_gpu_affinity(self) -> None:
        module = load_module()

        yaml_text = module.gpu_job_yaml(
            "ecepxie",
            "run1",
            "test-pvc",
            module.IMAGE,
            "l4",
            "/workspace/out",
            blocked_nodes={"bad-node.example.edu"},
        )

        self.assertIn("operator: NotIn", yaml_text)
        self.assertIn("bad-node.example.edu", yaml_text)

    def test_distinct_gpu_verifier_accepts_extra_valid_labels(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            labels_dir = root / "labels"
            module.prepare_configmap_dir(config_dir)
            for gpu_name in ["NVIDIA A100 80GB PCIe", "NVIDIA RTX A4000"]:
                shard = labels_dir / gpu_name.replace(" ", "_") / "results_shard0.jsonl"
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_text(
                    json.dumps({"status": "ok", "hardware": {"gpu_name": gpu_name}}) + "\n",
                    encoding="utf-8",
                )

            proc = subprocess.run(
                [sys.executable, str(config_dir / "verify_distinct_gpus.py"), str(labels_dir), "1"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(proc.returncode, 0, proc.stdout)


if __name__ == "__main__":
    unittest.main()
