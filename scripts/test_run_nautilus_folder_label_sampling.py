#!/usr/bin/env python3
"""Tests for Nautilus folder label sampling script."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_nautilus_folder_label_sampling as sampler


class FolderLabelSamplingTests(unittest.TestCase):
    def test_discover_models_reads_shape_and_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "with_shape.py").write_text(
                "MODEL_ID = 'custom'\nINPUT_SHAPE = (4, 8)\ndef make_model():\n    return None\n",
                encoding="utf-8",
            )
            (root / "without_shape.py").write_text("def make_model():\n    return None\n", encoding="utf-8")
            entries = sampler.discover_models(root, [16, 32])
        self.assertEqual([entry.model_id for entry in entries], ["custom", "without_shape"])
        self.assertEqual(entries[0].input_shape, [4, 8])
        self.assertEqual(entries[1].input_shape, [16, 32])

    def test_manifest_contains_required_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            sampler.write_manifest(manifest, [sampler.ModelEntry("m", "m.py", [2, 3])], "fp32_ieee")
            row = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(row["model_id"], "m")
        self.assertEqual(row["model_file"], "m.py")
        self.assertEqual(row["input_shape"], [2, 3])
        self.assertEqual(row["label_file"], "label/label/m_fp32_ieee.txt")

    def test_gpu_job_yaml_profiles_and_verifies(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            run_id="run",
            image="img",
            pvc="pvc",
            backoff_limit=0,
            warmup_epochs=1,
            profile_epochs=1,
            batches_per_epoch=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            precision_config="fp32_ieee",
            cpu="1",
            cpu_limit="2",
            memory="2Gi",
            memory_limit="4Gi",
        )
        text = sampler.gpu_job_yaml(args, "a100", "/workspace/run")
        self.assertIn("--warmup-epochs 1", text)
        self.assertIn("--profile-epochs 1", text)
        self.assertIn("verify_sampled_labels.py", text)
        self.assertIn("nvidia.com/a100", text)
        self.assertIn("/workspace/run/models", text)

    def test_gpu_job_yaml_can_block_bad_node(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            run_id="run",
            image="img",
            pvc="pvc",
            backoff_limit=0,
            warmup_epochs=1,
            profile_epochs=1,
            batches_per_epoch=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            precision_config="fp32_ieee",
            cpu="1",
            cpu_limit="2",
            memory="2Gi",
            memory_limit="4Gi",
        )
        text = sampler.gpu_job_yaml(args, "l4", "/workspace/run", {"bad-node.example.edu"})
        self.assertIn("kubernetes.io/hostname", text)
        self.assertIn("operator: NotIn", text)
        self.assertIn("bad-node.example.edu", text)

    def test_dry_run_writes_artifacts(self) -> None:
        run_id = "dry-test"
        artifacts = [
            Path(f"record/nautilus_folder_labels_{run_id}.md"),
            Path(f"record/nautilus_folder_labels_{run_id}.yaml"),
            Path(f"record/nautilus_folder_labels_{run_id}.manifest.jsonl"),
        ]
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            (models / "m.py").write_text(
                "INPUT_SHAPE = (2, 3)\ndef make_model():\n    return None\n",
                encoding="utf-8",
            )
            labels = root / "labels"
            sampler.main(
                [
                    "--models-dir",
                    str(models),
                    "--local-labels-dir",
                    str(labels),
                    "--namespace",
                    "ns",
                    "--pvc",
                    "pvc",
                    "--gpus",
                    "a100,a40,l4,rtx_a4000",
                    "--run-id",
                    run_id,
                    "--dry-run",
                ]
            )
        try:
            for artifact in artifacts:
                self.assertTrue(artifact.exists())
        finally:
            for artifact in artifacts:
                artifact.unlink(missing_ok=True)

    def test_live_flow_copies_remote_labels_back_to_local_dir(self) -> None:
        run_id = "mock-run"
        artifacts = [
            Path(f"record/nautilus_folder_labels_{run_id}.md"),
            Path(f"record/nautilus_folder_labels_{run_id}.yaml"),
            Path(f"record/nautilus_folder_labels_{run_id}.manifest.jsonl"),
        ]
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            (models / "m.py").write_text(
                "INPUT_SHAPE = (2, 3)\ndef make_model():\n    return None\n",
                encoding="utf-8",
            )
            labels = root / "labels"
            copied: list[tuple[str, str]] = []

            def fake_cp(src: str, dst: str) -> None:
                copied.append((src, dst))
                if src.endswith(":/workspace/perfseer-folder-runs/mock-run/labels"):
                    Path(dst).mkdir(parents=True, exist_ok=True)
                    (Path(dst) / "results_shard0.jsonl").write_text("{}\n", encoding="utf-8")

            with (
                mock.patch.object(sampler, "kubectl"),
                mock.patch.object(sampler, "wait_pod_running"),
                mock.patch.object(sampler, "wait_jobs_with_switching"),
                mock.patch.object(sampler, "wait_verifier", return_value='{"bad_rows": 0}\n'),
                mock.patch.object(sampler, "start_monitor") as start_monitor,
                mock.patch.object(sampler, "kubectl_cp", side_effect=fake_cp),
                mock.patch.object(sampler, "cleanup"),
            ):
                start_monitor.return_value = mock.Mock(terminate=mock.Mock())
                sampler.main(
                    [
                        "--models-dir",
                        str(models),
                        "--local-labels-dir",
                        str(labels),
                        "--namespace",
                        "ns",
                        "--pvc",
                        "pvc",
                        "--gpus",
                        "a100",
                        "--run-id",
                        run_id,
                    ]
                )
        try:
            self.assertIn(
                (
                    "ns/perfseer-label-stage-mock-run:/workspace/perfseer-folder-runs/mock-run/labels",
                    str((labels / run_id).resolve()),
                ),
                copied,
            )
        finally:
            for artifact in artifacts:
                artifact.unlink(missing_ok=True)

    def test_switching_deletes_pending_job_and_submits_replacement(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            run_id="switch-test",
            image="img",
            pvc="pvc",
            backoff_limit=0,
            warmup_epochs=1,
            profile_epochs=1,
            batches_per_epoch=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            precision_config="fp32_ieee",
            cpu="1",
            cpu_limit="2",
            memory="2Gi",
            memory_limit="4Gi",
            timeout_seconds=5,
            active_gpus=1,
            pending_timeout_seconds=0,
            blacklist_ttl_seconds=1,
            max_retries_per_gpu=0,
            min_successful_gpus=1,
        )
        record = Path("record/nautilus_folder_labels_switch-test.md")
        record.parent.mkdir(exist_ok=True)
        record.write_text("", encoding="utf-8")
        state = sampler.ControllerState(
            candidate_gpus=["a100", "a40"],
            active_limit=1,
            pending_timeout_seconds=0,
            blacklist_ttl_seconds=1,
        )
        calls: list[list[str]] = []

        def fake_kubectl(command, input_text=None, check=True):
            calls.append(list(command))
            if command[:2] == ["get", "jobs"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {"name": "perfseer-label-switch-test-a100"},
                                    "status": {},
                                }
                            ]
                        }
                    ),
                )
            if command[:2] == ["get", "pods"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {"labels": {"perfseer-gpu-key": "a100"}},
                                    "status": {
                                        "phase": "Pending",
                                        "conditions": [
                                            {
                                                "type": "PodScheduled",
                                                "status": "False",
                                                "reason": "Unschedulable",
                                            }
                                        ],
                                    },
                                }
                            ]
                        }
                    ),
                )
            return SimpleNamespace(returncode=0, stdout="")

        with (
            mock.patch.object(sampler, "kubectl", side_effect=fake_kubectl),
            mock.patch.object(sampler.time, "sleep", side_effect=SystemExit("stop")),
        ):
            with self.assertRaisesRegex(SystemExit, "stop"):
                sampler.wait_jobs_with_switching(args, "/remote", record, state)
        try:
            self.assertEqual(state.gpu_states["a100"].status, "failed")
            self.assertEqual(state.gpu_states["a40"].status, "pending")
            self.assertTrue(any(call[:2] == ["delete", "job"] for call in calls))
            self.assertIn("GPU Switch", record.read_text(encoding="utf-8"))
        finally:
            record.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
