from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.a10g_runner import _allows_a10_family
from perfseer_v3.dataset_pack.modality_sharding import (
    EXPECTED_MODALITY_COUNTS,
    MODALITIES,
    ModalityCompletion,
    ModalityShardError,
    analysis_summary,
    build_modality_contract,
    merge_modality_workspaces,
    verify_modality_workspace,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("PERFSEER_TEST_PYTHON", os.sys.executable)).resolve()
IMAGE = (
    "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@"
    "sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89"
)
IMAGE_DIGEST = IMAGE.rsplit("@", 1)[1]
REVISION = "0123456789abcdef0123456789abcdef01234567"
COMMON = (
    "--namespace", "perfseer",
    "--pvc", "perfseer-rwx",
    "--repository-url", "https://github.com/example/perfseer.git",
    "--revision", REVISION,
    "--kaggle-secret", "kaggle-api",
    "--image", IMAGE,
    "--run-id", "run001",
)


def wrapper(modality: str) -> Path:
    return ROOT / f"submit_nautilus_a10_{modality}.sh"


def render(modality: str, action: str) -> tuple[str, dict]:
    result = subprocess.run(
        [str(wrapper(modality)), action, *COMMON],
        cwd=ROOT,
        env={**os.environ, "PYTHON_BIN": str(PYTHON)},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout, yaml.safe_load(result.stdout)


class ModalityPartitionTests(unittest.TestCase):
    def test_exact_disjoint_union_and_epoch_count(self) -> None:
        summary = analysis_summary()
        self.assertEqual(summary["rest_candidate_count"], 5_500)
        self.assertEqual(summary["rest_measured_epoch_count"], 16_500)
        contracts = [build_modality_contract(value) for value in MODALITIES]
        ids = [candidate for contract in contracts for candidate in contract.candidate_ids]
        self.assertEqual(len(ids), 5_500)
        self.assertEqual(len(set(ids)), 5_500)
        self.assertEqual(
            {row.modality: row.candidate_count for row in contracts},
            EXPECTED_MODALITY_COUNTS,
        )

    def test_a10_family_is_explicit_opt_in_and_legacy_default_is_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_allows_a10_family())
        with patch.dict(os.environ, {"PERFSEER_ALLOW_A10_FAMILY": "1"}, clear=True):
            self.assertTrue(_allows_a10_family())
        with patch.dict(os.environ, {"PERFSEER_ALLOW_A10_FAMILY": "true"}, clear=True):
            self.assertFalse(_allows_a10_family())


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = {
            row["modality"]: row
            for row in yaml.safe_load_all(
                (ROOT / "tests/fixtures/nautilus_a10_manifest_golden.yaml").read_text()
            )
        }

    def test_all_modalities_match_pod_and_job_goldens(self) -> None:
        for modality in MODALITIES:
            with self.subTest(modality=modality):
                pod_text, pod = render(modality, "render-pod")
                second_text, _ = render(modality, "render-pod")
                job_text, job = render(modality, "render-production-job")
                self.assertEqual(pod_text, second_text)
                golden = self.golden[modality]
                self.assertEqual(pod["kind"], "Pod")
                self.assertEqual(pod["metadata"]["name"], golden["pod_name"])
                self.assertEqual(job["kind"], "Job")
                self.assertEqual(job["metadata"]["name"], golden["job_name"])
                pod_container = pod["spec"]["containers"][0]
                job_spec = job["spec"]["template"]["spec"]
                job_container = job_spec["containers"][0]
                pod_env = {row["name"]: row["value"] for row in pod_container["env"]}
                job_env = {row["name"]: row["value"] for row in job_container["env"]}
                self.assertEqual(pod_env["PERFSEER_RUN_ROOT"], golden["debug_workspace"])
                self.assertEqual(job_env["PERFSEER_RUN_ROOT"], golden["workspace"])
                self.assertEqual(job["spec"]["backoffLimit"], 0)
                self.assertEqual(job["spec"]["parallelism"], 1)
                self.assertEqual(job_spec["restartPolicy"], "Never")
                self.assertEqual(job_container["resources"]["limits"]["nvidia.com/gpu"], 1)
                self.assertEqual(job_container["resources"]["requests"]["nvidia.com/gpu"], 1)
                self.assertEqual(
                    job_spec["affinity"]["nodeAffinity"]
                    ["requiredDuringSchedulingIgnoredDuringExecution"]
                    ["nodeSelectorTerms"][0]["matchExpressions"][0],
                    {"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A10"]},
                )
                self.assertEqual(
                    job_spec["volumes"][0]["persistentVolumeClaim"]["claimName"],
                    "perfseer-rwx",
                )
                self.assertEqual(job_container["envFrom"], [{"secretRef": {"name": "kaggle-api"}}])
                self.assertNotIn("TOP_SECRET_FIXTURE", pod_text + job_text)

    def test_pilot_is_bounded_and_production_is_not(self) -> None:
        pilot_text, pilot = render("audio", "render-pilot-job")
        production_text, _ = render("audio", "render-production-job")
        self.assertIn("--max-new-accepted 8", pilot["spec"]["template"]["spec"]["containers"][0]["args"][0])
        self.assertNotIn("--max-new-accepted", production_text)
        self.assertNotEqual(pilot_text, production_text)

    def test_invalid_revision_and_mutable_image_fail_before_output(self) -> None:
        for replacement in (
            ("--revision", "main"),
            ("--image", "pytorch/pytorch:latest"),
        ):
            args = list(COMMON)
            index = args.index(replacement[0])
            args[index + 1] = replacement[1]
            result = subprocess.run(
                [str(wrapper("audio")), "render-pod", *args],
                cwd=ROOT,
                env={**os.environ, "PYTHON_BIN": str(PYTHON), "KAGGLE_API_TOKEN": "TOP_SECRET_FIXTURE"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("TOP_SECRET_FIXTURE", result.stderr)


class FakeKubernetesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.log = Path(self.temporary.name) / "calls.tsv"
        self.fake = ROOT / "tests/fixtures/fake_kubectl.sh"
        self.environment = {
            **os.environ,
            "PYTHON_BIN": str(PYTHON),
            "KUBECTL_BIN": str(self.fake),
            "FAKE_KUBECTL_LOG": str(self.log),
            "NAUTILUS_DISABLE_AUTO_MONITOR": "1",
            "NAUTILUS_MONITOR_ITERATIONS": "1",
            "NAUTILUS_MONITOR_INITIAL_SECONDS": "0",
            "NAUTILUS_MONITOR_STEADY_SECONDS": "0",
            "KAGGLE_API_TOKEN": "TOP_SECRET_FIXTURE",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, action: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(wrapper("audio")), action, *COMMON, *extra],
            cwd=ROOT,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
        )

    def _calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [line.split("\t") for line in self.log.read_text().splitlines()]

    def test_render_never_invokes_kubernetes_executable(self) -> None:
        self._run("render-pod")
        self._run("render-pilot-job")
        self._run("render-production-job")
        self.assertEqual(self._calls(), [])

    def test_submit_collects_immediate_feedback_in_order(self) -> None:
        result = self._run("submit-production-job")
        calls = self._calls()
        verbs = [(row[1], row[2] if len(row) > 2 else "") for row in calls]
        self.assertEqual(
            verbs[:7],
            [
                ("apply", "--namespace"),
                ("get", "job"),
                ("get", "pods"),
                ("get", "pods"),
                ("describe", "pod"),
                ("logs", "fake-perfseer-pod"),
                ("get", "events"),
            ],
        )
        self.assertNotIn("TOP_SECRET_FIXTURE", result.stdout + result.stderr + self.log.read_text())

    def test_status_and_monitor_use_only_fake_commands(self) -> None:
        self._run("status")
        status_calls = self._calls()
        self.assertEqual(
            [(row[1], row[2] if len(row) > 2 else "") for row in status_calls],
            [
                ("get", "job"),
                ("get", "pods"),
                ("get", "pods"),
                ("describe", "pod"),
                ("logs", "fake-perfseer-pod"),
                ("get", "events"),
            ],
        )
        status_count = len(status_calls)
        self._run("monitor")
        monitor_calls = self._calls()[status_count:]
        self.assertEqual(
            [(row[1], row[2] if len(row) > 2 else "") for row in monitor_calls],
            [
                ("get", "job"),
                ("get", "pods"),
                ("get", "pods"),
                ("get", "events"),
                ("logs", "fake-perfseer-pod"),
                ("exec", "fake-perfseer-pod"),
            ],
        )


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        provenance = {
            "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
            "name": "NVIDIA A10",
            "uuid": "GPU-fixture",
            "total_memory_bytes": 24 * 1024**3,
            "compute_capability": [8, 6],
        }
        cls.provenance_sha = canonical_sha256(provenance)
        cls.workspaces = {}
        for modality in MODALITIES:
            workspace = cls.root / modality
            cls.workspaces[modality] = workspace
            hardware = workspace / "provenance/hardware"
            accepted = workspace / "attempts/accepted"
            hardware.mkdir(parents=True)
            accepted.mkdir(parents=True)
            (hardware / f"{cls.provenance_sha}.json").write_text(json.dumps(provenance))
            for candidate_id in build_modality_contract(modality).candidate_ids:
                record = {
                    "configuration_id": candidate_id,
                    "status": "accepted",
                    "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
                    "measured_epochs": [3, 4, 5],
                    "fingerprints": {"hardware_sha256": cls.provenance_sha},
                }
                (accepted / f"{candidate_id}.json").write_text(json.dumps(record))
            verify_modality_workspace(
                workspace,
                modality,
                repository_revision=REVISION,
                image_digest=IMAGE_DIGEST,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_merge_publishes_5500_rows(self) -> None:
        output = self.root / "merged-rest"
        if output.exists():
            shutil.rmtree(output)
        receipt = merge_modality_workspaces(self.workspaces, output)
        self.assertEqual(receipt["candidate_count"], 5_500)
        self.assertEqual(receipt["measured_epoch_count"], 16_500)
        self.assertEqual(len(list((output / "attempts/accepted").glob("*.json"))), 5_500)

    def test_missing_duplicate_and_overlap_contracts_fail(self) -> None:
        completion_path = self.workspaces["audio"] / "state/modality_completion.json"
        original = completion_path.read_text()
        try:
            value = json.loads(original)
            for mutation in ("missing", "duplicate", "overlap"):
                changed = copy.deepcopy(value)
                if mutation == "missing":
                    changed["candidate_ids"] = changed["candidate_ids"][:-1]
                elif mutation == "duplicate":
                    changed["resolved_candidate_ids"][1] = changed["resolved_candidate_ids"][0]
                elif mutation == "overlap":
                    changed["candidate_ids"][0] = build_modality_contract("tabular").candidate_ids[0]
                changed["completion_sha256"] = canonical_sha256(
                    {key: item for key, item in changed.items() if key != "completion_sha256"}
                )
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ModalityShardError):
                        ModalityCompletion.from_dict(changed)
        finally:
            completion_path.write_text(original)

    def test_differently_pinned_workspace_fails_merge(self) -> None:
        completion_path = self.workspaces["generated"] / "state/modality_completion.json"
        original = completion_path.read_text()
        try:
            changed = json.loads(original)
            changed["repository_revision"] = "f" * 40
            changed["completion_sha256"] = canonical_sha256(
                {key: item for key, item in changed.items() if key != "completion_sha256"}
            )
            completion_path.write_text(json.dumps(changed))
            with self.assertRaises(ModalityShardError):
                merge_modality_workspaces(self.workspaces, self.root / "different-pin")
        finally:
            completion_path.write_text(original)

    def test_record_tampering_after_completion_fails_merge(self) -> None:
        candidate = build_modality_contract("graph").candidate_ids[0]
        path = self.workspaces["graph"] / "attempts/accepted" / f"{candidate}.json"
        original = path.read_text()
        try:
            value = json.loads(original)
            value["measured_epochs"] = [2, 3, 4]
            path.write_text(json.dumps(value))
            with self.assertRaises(ModalityShardError):
                merge_modality_workspaces(self.workspaces, self.root / "tampered")
        finally:
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()
