#!/usr/bin/env python3
"""Tests for the local source-first Nautilus workflow runner."""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_nrp_source_workflow_local as runner


class SourceWorkflowLocalRunnerTests(unittest.TestCase):
    def test_runner_rejects_invalid_kubernetes_job_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "Kubernetes name prefix"):
            runner.main(
                [
                    "--namespace",
                    "ns",
                    "--image",
                    "img",
                    "--pvc",
                    "pvc",
                    "--job-prefix",
                    "bad_prefix",
                    "--no-monitor",
                    "--skip-prepare",
                    "--skip-profile",
                    "--skip-package",
                ]
            )

    def test_runner_can_stage_local_repo_before_submit(self) -> None:
        run_calls: list[list[str]] = []
        kubectl_calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool = True):
            run_calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="ok\n")

        def fake_kubectl(cmd: list[str], input_text: str | None = None, check: bool = True):
            kubectl_calls.append(list(cmd))
            if cmd[:2] == ["get", "pod"] and "-o" in cmd and "json" in cmd:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"status": {"phase": "Running"}}))
            return SimpleNamespace(returncode=0, stdout="ok\n")

        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            (fake_repo / "nrp_calibration_pack").mkdir(parents=True)
            (fake_repo / "scripts").mkdir()
            (fake_repo / "src").mkdir()
            (fake_repo / "README.md").write_text("readme\n", encoding="utf-8")
            (fake_repo / "requirements.txt").write_text("torch\n", encoding="utf-8")
            (fake_repo / "pyproject.toml").write_text("[project]\nname = \"fake\"\n", encoding="utf-8")
            (fake_repo / "nrp_calibration_pack" / "submit_nrp_source_workflow.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (fake_repo / "scripts" / "rebuild_source_tar_dataset.py").write_text("print('ok')\n", encoding="utf-8")

            args = SimpleNamespace(
                namespace="ns",
                image="img",
                utility_image="python:3.11-slim",
                pvc="pvc",
                job_prefix="pref",
                workflow_dir="/mnt/output/run",
                repo_dir="/workspace/old",
                local_repo_dir=str(fake_repo),
                staged_repo_dir="",
                staging_pod_name="",
                stage_timeout_seconds=5,
                poll_seconds=1,
                keep_staging_pod=False,
            )
            with mock.patch.object(runner, "run", side_effect=fake_run), mock.patch.object(
                runner, "kubectl", side_effect=fake_kubectl
            ):
                runner.stage_local_repo(args)

        self.assertEqual(args.repo_dir, "/mnt/output/run/repo")
        self.assertTrue(any(call[:2] == ["apply", "-f"] for call in kubectl_calls))
        self.assertTrue(any(call[:2] == ["kubectl", "cp"] for call in run_calls))
        self.assertTrue(any(call[:4] == ["kubectl", "exec", "-n", "ns"] for call in run_calls))

    def test_runner_submits_stages_and_downloads_source_and_dataset_tarballs(self) -> None:
        run_calls: list[list[str]] = []
        kubectl_calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool = True):
            run_calls.append(list(cmd))
            if cmd[:2] == ["kubectl", "cp"]:
                dst = Path(cmd[3])
                dst.parent.mkdir(parents=True, exist_ok=True)
                if cmd[2].endswith("source_labels.tar.gz"):
                    with tarfile.open(dst, "w:gz") as tar:
                        model = root / "model.py"
                        model.write_text("def make_model():\n    return None\n", encoding="utf-8")
                        tar.add(model, arcname="pack/models/calib_0000.py")
                        results = root / "results_shard0.jsonl"
                        results.write_text('{"status":"ok"}\n', encoding="utf-8")
                        tar.add(results, arcname="results/rtx5090/results_shard0.jsonl")
                elif cmd[2].endswith("dataset.tar.gz"):
                    with tarfile.open(dst, "w:gz") as tar:
                        graph = root / "calib_0000.pkl"
                        graph.write_bytes(b"pickle")
                        tar.add(graph, arcname="cg/cg/calib_0000.pkl")
                        label = root / "calib_0000_rtx5090_fp32_ieee.txt"
                        label.write_text(
                            repr({"train": "1|2|3|4|5|6|7", "infer": "1|2|3|4|5|6|7"}) + "\n",
                            encoding="utf-8",
                        )
                        tar.add(label, arcname="label/label/calib_0000_rtx5090_fp32_ieee.txt")
                        metadata = root / "precision_metadata.jsonl"
                        metadata.write_text("{}\n", encoding="utf-8")
                        tar.add(metadata, arcname="label/precision_metadata.jsonl")
            return SimpleNamespace(returncode=0, stdout="ok\n")

        def fake_kubectl(cmd: list[str], input_text: str | None = None, check: bool = True):
            kubectl_calls.append(list(cmd))
            if cmd[:2] == ["get", "jobs"] and f"app=pref" in cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {"name": "pref-prepare-sources", "labels": {"stage": "prepare-sources"}},
                                    "spec": {"template": {"spec": {"containers": [{"image": "img"}]}}},
                                },
                                {
                                    "metadata": {"name": "pref-package-results", "labels": {"stage": "package-results"}},
                                    "spec": {"template": {"spec": {"containers": [{"image": "img"}]}}},
                                },
                            ]
                        }
                    ),
                )
            if cmd[:2] == ["get", "job"]:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"spec": {"completions": 1}, "status": {"succeeded": 1}}))
            if cmd[:2] == ["get", "pod"] and "-o" in cmd and "json" in cmd:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"status": {"phase": "Running"}}))
            if cmd[:2] == ["get", "pods"] and "-o" in cmd and "json" in cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "name": "package-pod",
                                        "creationTimestamp": "2026-06-21T00:00:00Z",
                                    }
                                }
                            ]
                        }
                    ),
                )
            return SimpleNamespace(returncode=0, stdout="ok\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(runner, "run", side_effect=fake_run), mock.patch.object(
                runner, "kubectl", side_effect=fake_kubectl
            ), mock.patch.object(runner, "run_partitioned_profile_jobs", return_value={"completed_jobs": 4}) as partition_runner:
                runner.main(
                    [
                        "--namespace",
                        "ns",
                        "--image",
                        "img",
                        "--allow-mutable-image-tag",
                        "--pvc",
                        "pvc",
                        "--job-prefix",
                        "pref",
                        "--hardware-id",
                        "rtx5090",
                        "--subset-size",
                        "2",
                        "--completions",
                        "1",
                        "--parallelism",
                        "1",
                        "--local-output-dir",
                        tmp,
                        "--poll-seconds",
                        "1",
                        "--skip-image-warmup",
                        "--no-monitor",
                    ]
                )

            submit_stages = [
                call[call.index("--stage") + 1]
                for call in run_calls
                if call and call[0].endswith("submit_nrp_source_workflow.sh")
            ]
            cp_calls = [call for call in run_calls if call[:2] == ["kubectl", "cp"]]

        self.assertEqual(submit_stages, ["prepare", "package"])
        partition_runner.assert_called_once()
        self.assertIn(
            [
                "kubectl",
                "cp",
                "ns/pref-download:/mnt/output/perfseer_nrp_source_workflow/perfseer_rtx5090_source_labels.tar.gz",
                str(Path(tmp).resolve() / "perfseer_rtx5090_source_labels.tar.gz"),
            ],
            cp_calls,
        )
        self.assertIn(
            [
                "kubectl",
                "cp",
                "ns/pref-download:/mnt/output/perfseer_nrp_source_workflow/perfseer_rtx5090_dataset.tar.gz",
                str(Path(tmp).resolve() / "perfseer_rtx5090_dataset.tar.gz"),
            ],
            cp_calls,
        )
        self.assertTrue(any(call[:2] == ["get", "job"] for call in kubectl_calls))
        self.assertTrue(any(call[:2] == ["describe", "pod"] for call in kubectl_calls))

    def test_partition_profile_shards_distributes_all_shards_across_four_gpus(self) -> None:
        partitions = runner.partition_profile_shards(10, ["a100", "a40", "l4", "rtx_a4000"])
        self.assertEqual(
            partitions,
            [
                runner.ProfileGpuPartition("a100", 0, 3),
                runner.ProfileGpuPartition("a40", 3, 3),
                runner.ProfileGpuPartition("l4", 6, 2),
                runner.ProfileGpuPartition("rtx_a4000", 8, 2),
            ],
        )
        covered = []
        for partition in partitions:
            covered.extend(range(partition.shard_start, partition.shard_start + partition.shard_count))
        self.assertEqual(covered, list(range(10)))

    def test_partition_profile_requires_exactly_four_gpu_presets(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            runner.partition_profile_shards(8, ["a100", "a40", "l4"])
        with self.assertRaisesRegex(ValueError, "at least 4"):
            runner.partition_profile_shards(3, ["a100", "a40", "l4", "rtx_a4000"])

    def test_profile_partition_yaml_uses_one_gpu_kind_and_local_index_mapping(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="same/profile:image",
            pvc="pvc",
            job_prefix="pref",
            workflow_dir="/mnt/output/run",
            repo_dir="/mnt/output/run/repo",
            hardware_id="mixed",
            warmup=1,
            infer_repeats=1,
            train_repeats=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            profile_precision_sweep="auto",
            completions=10,
            bootstrap_command="",
        )
        yaml_text = runner.profile_partition_job_yaml(args, runner.ProfileGpuPartition("a40", 3, 3))
        self.assertIn("name: pref-profile-a40", yaml_text)
        self.assertIn("perfseer-gpu-key: a40", yaml_text)
        self.assertIn("completionMode: Indexed", yaml_text)
        self.assertIn("completions: 3", yaml_text)
        self.assertIn("parallelism: 1", yaml_text)
        self.assertIn("NVIDIA-A40", yaml_text)
        self.assertNotIn("NVIDIA-L4", yaml_text)
        self.assertIn("GLOBAL_SHARD_INDEX=$((3 + ${JOB_COMPLETION_INDEX:-0}))", yaml_text)
        self.assertIn("--hardware-id mixed_a40", yaml_text)
        self.assertIn("--num-shards 10", yaml_text)
        self.assertIn("--shard-index ${GLOBAL_SHARD_INDEX}", yaml_text)
        self.assertIn("nvidia.com/a40: \"1\"", yaml_text)

    def test_profile_shard_yaml_uses_distinct_gpu_preset_and_single_shard(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="img",
            pvc="pvc",
            job_prefix="pref",
            workflow_dir="/mnt/output/run",
            repo_dir="/mnt/output/run/repo",
            hardware_id="l4_speed",
            warmup=1,
            infer_repeats=1,
            train_repeats=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            profile_precision_sweep="fp32_ieee",
            completions=16,
            bootstrap_command="",
        )
        yaml_text = runner.profile_shard_job_yaml(args, shard_index=3, gpu_key="l4", attempt_number=2)
        self.assertIn("name: pref-profile-s0003-l4-a2", yaml_text)
        self.assertIn("perfseer-shard-index: \"3\"", yaml_text)
        self.assertIn("NVIDIA-L4", yaml_text)
        self.assertIn("--precision-sweep fp32_ieee", yaml_text)
        self.assertIn("--num-shards 16", yaml_text)
        self.assertIn("--shard-index 3", yaml_text)
        self.assertIn("nvidia.com/gpu: \"1\"", yaml_text)

    def test_all_profile_gpu_presets_use_the_same_container_image_argument(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="same/profile:image",
            pvc="pvc",
            job_prefix="pref",
            workflow_dir="/mnt/output/run",
            repo_dir="/mnt/output/run/repo",
            hardware_id="mixed",
            warmup=1,
            infer_repeats=1,
            train_repeats=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            profile_precision_sweep="auto",
            completions=16,
            bootstrap_command="",
        )
        for shard_index, gpu_key in enumerate(["a100", "a40", "l4", "rtx_a4000"]):
            yaml_text = runner.profile_shard_job_yaml(args, shard_index, gpu_key, attempt_number=0)
            self.assertIn("image: same/profile:image", yaml_text)
            self.assertEqual(yaml_text.count("image: same/profile:image"), 1)

    def test_profile_shard_yaml_can_reuse_warmed_image_node(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="img",
            pvc="pvc",
            job_prefix="pref",
            workflow_dir="/mnt/output/run",
            repo_dir="/mnt/output/run/repo",
            hardware_id="mixed",
            warmup=1,
            infer_repeats=1,
            train_repeats=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            profile_precision_sweep="auto",
            completions=16,
            bootstrap_command="",
        )
        yaml_text = runner.profile_shard_job_yaml(
            args,
            shard_index=0,
            gpu_key="a100",
            attempt_number=0,
            warmed_nodes={"warmed-node"},
        )
        self.assertIn("kubernetes.io/hostname", yaml_text)
        self.assertIn("operator: In", yaml_text)
        self.assertIn("warmed-node", yaml_text)

    def test_image_warmup_job_uses_same_profile_image_and_gpu_affinity(self) -> None:
        args = SimpleNamespace(namespace="ns", image="same/profile:image", job_prefix="pref")
        yaml_text = runner.image_warmup_job_yaml(args, "l4", 0)
        self.assertIn("stage: image-warmup", yaml_text)
        self.assertIn("image: same/profile:image", yaml_text)
        self.assertIn("imagePullPolicy: IfNotPresent", yaml_text)
        self.assertIn("NVIDIA-L4", yaml_text)

    def test_verify_same_workflow_image_rejects_mismatch(self) -> None:
        args = SimpleNamespace(namespace="ns", job_prefix="pref", image="expected:image")

        def fake_kubectl(cmd: list[str], input_text: str | None = None, check: bool = True):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "good", "labels": {"stage": "profile-labels"}},
                                "spec": {"template": {"spec": {"containers": [{"image": "expected:image"}]}}},
                            },
                            {
                                "metadata": {"name": "bad", "labels": {"stage": "package-results"}},
                                "spec": {"template": {"spec": {"containers": [{"image": "other:image"}]}}},
                            },
                        ]
                    }
                ),
            )

        with mock.patch.object(runner, "kubectl", side_effect=fake_kubectl):
            with self.assertRaisesRegex(RuntimeError, "workflow image mismatch"):
                runner.verify_same_workflow_image(args)

    def test_kubectl_uses_request_timeout_and_hard_timeout(self) -> None:
        calls: list[tuple[list[str], float | None]] = []

        def fake_run(cmd: list[str], input_text: str | None = None, check: bool = True, timeout_seconds: float | None = None):
            calls.append((list(cmd), timeout_seconds))
            return SimpleNamespace(returncode=0, stdout="ok\n")

        with mock.patch.object(runner, "run", side_effect=fake_run), mock.patch.dict(
            runner.os.environ,
            {
                "PERFSEER_KUBECTL_REQUEST_TIMEOUT": "9s",
                "PERFSEER_KUBECTL_HARD_TIMEOUT_SECONDS": "12",
            },
        ):
            runner.kubectl(["get", "pods", "-n", "ns"])

        self.assertEqual(calls, [(["kubectl", "--request-timeout=9s", "get", "pods", "-n", "ns"], 12.0)])

    def test_profile_image_reference_requires_digest_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned by digest"):
            runner.validate_profile_image_reference("repo/image:latest", allow_mutable_tag=False)
        runner.validate_profile_image_reference("repo/image:latest@sha256:" + "a" * 64, allow_mutable_tag=False)
        runner.validate_profile_image_reference("repo/image:latest", allow_mutable_tag=True)

    def test_run_profile_source_contains_resume_checkpoint_and_synchronized_row_writes(self) -> None:
        source = (runner.ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py").read_text(encoding="utf-8")
        self.assertIn("default=True", source)
        self.assertIn("load_resume_checkpoint(output_dir, results_path)", source)
        self.assertIn("point_id in resume_checkpoint.completed_profile_points", source)
        self.assertIn("skip_completed", source)
        self.assertIn("results_fh.flush()", source)
        self.assertIn("os.fsync(results_fh.fileno())", source)
        self.assertIn("tmp_path.replace(path)", source)

    def test_utility_pod_uses_small_image_not_cuda_profile_image(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="nvcr.io/nvidia/pytorch:26.03-py3",
            utility_image="python:3.11-slim",
            pvc="pvc",
            job_prefix="pref",
        )
        yaml_text = runner.pvc_utility_pod_manifest(args, "pref-repo-stage", "repo-stage")
        self.assertIn("image: python:3.11-slim", yaml_text)
        self.assertNotIn("image: nvcr.io/nvidia/pytorch:26.03-py3", yaml_text)

    def test_profile_retry_policy_matches_switcher_semantics(self) -> None:
        state = runner.ProfileSwitchState(
            gpu_keys=["a100", "a40", "l4"],
            active_limit=2,
            pending_timeout_seconds=300,
            max_retries_per_shard=3,
        )
        attempt = runner.ProfileShardAttempt(
            shard_index=0,
            gpu_key="a100",
            job_name="job",
            attempt_number=0,
            submitted_at=0.0,
        )
        self.assertEqual(
            runner.retry_gpu_for_profile_failure(state, attempt, "cuda_initialization_failure", "bad-node"),
            "a100",
        )
        self.assertEqual(state.blocked_nodes_by_gpu["a100"], {"bad-node"})
        self.assertEqual(runner.retry_gpu_for_profile_failure(state, attempt, "unschedulable", ""), "a40")
        self.assertIsNone(runner.retry_gpu_for_profile_failure(state, attempt, "model_or_code_failure", ""))

    def test_profile_shard_yaml_excludes_blocked_node_for_same_gpu_retry(self) -> None:
        args = SimpleNamespace(
            namespace="ns",
            image="img",
            pvc="pvc",
            job_prefix="pref",
            workflow_dir="/mnt/output/run",
            repo_dir="/mnt/output/run/repo",
            hardware_id="mixed",
            warmup=1,
            infer_repeats=1,
            train_repeats=1,
            sample_interval=0.01,
            optimizer="sgd",
            sm_occupancy_source="nvml_proxy",
            profile_precision_sweep="auto",
            completions=16,
            bootstrap_command="",
        )
        yaml_text = runner.profile_shard_job_yaml(
            args,
            shard_index=0,
            gpu_key="a100",
            attempt_number=1,
            blocked_nodes={"bad-node"},
        )
        self.assertIn("operator: NotIn", yaml_text)
        self.assertIn("bad-node", yaml_text)


if __name__ == "__main__":
    unittest.main()
