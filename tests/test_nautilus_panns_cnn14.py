from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from perfseer_v3.dataset_pack.family_sharding import (
    FamilyShardError,
    build_family_contract,
    verify_family_workspace,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "submit_nautilus_a10_panns_cnn14.sh"
PYTHON = Path(os.environ.get("PERFSEER_TEST_PYTHON", os.sys.executable)).resolve()
IMAGE = (
    "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@"
    "sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89"
)
REVISION = "0123456789abcdef0123456789abcdef01234567"
COMMON = (
    "--namespace", "perfseer",
    "--pvc", "perfseer-rwx",
    "--repository-url", "https://github.com/example/perfseer.git",
    "--revision", REVISION,
    "--kaggle-secret", "kaggle-api",
    "--image", IMAGE,
    "--run-id", "panns001",
)
IMAGE_DIGEST = IMAGE.rsplit("@", 1)[1]


class PannsContractTests(unittest.TestCase):
    def test_exact_frozen_family_projection(self) -> None:
        contract = build_family_contract("panns_cnn14")
        self.assertEqual(contract.modality, "audio")
        self.assertEqual(contract.display_name, "PANNs CNN14")
        self.assertEqual(contract.accepted_configurations, 550)
        self.assertEqual(contract.candidate_count, 550)
        self.assertEqual(contract.measured_epoch_count, 1_650)
        self.assertEqual(
            contract.task_ids,
            ("mlsp-2013-birds", "icml-2013-whale"),
        )
        self.assertEqual(len(set(contract.candidate_ids)), 550)


class PannsCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.accepted = cls.root / "attempts/accepted"
        hardware = cls.root / "provenance/hardware"
        cls.accepted.mkdir(parents=True)
        hardware.mkdir(parents=True)
        provenance = {
            "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
            "name": "NVIDIA A10",
            "uuid": "GPU-panns-fixture",
            "total_memory_bytes": 24 * 1024**3,
            "compute_capability": [8, 6],
        }
        cls.provenance_sha = canonical_sha256(provenance)
        (hardware / f"{cls.provenance_sha}.json").write_text(json.dumps(provenance))
        for candidate_id in build_family_contract("panns_cnn14").candidate_ids:
            record = {
                "configuration_id": candidate_id,
                "status": "accepted",
                "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
                "measured_epochs": [3, 4, 5],
                "fingerprints": {"hardware_sha256": cls.provenance_sha},
            }
            (cls.accepted / f"{candidate_id}.json").write_text(json.dumps(record))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_completion_publishes_550_rows_and_1650_epochs(self) -> None:
        completion = verify_family_workspace(
            self.root,
            "panns_cnn14",
            repository_revision=REVISION,
            image_digest=IMAGE_DIGEST,
        )
        self.assertEqual(completion.candidate_count, 550)
        self.assertEqual(completion.measured_epoch_count, 1_650)
        self.assertTrue((self.root / "state/family_completion.json").is_file())

    def test_extra_accepted_record_fails_closed(self) -> None:
        extra = self.accepted / f"{'f' * 64}.json"
        extra.write_text("{}")
        try:
            with self.assertRaises(FamilyShardError):
                verify_family_workspace(
                    self.root,
                    "panns_cnn14",
                    repository_revision=REVISION,
                    image_digest=IMAGE_DIGEST,
                )
        finally:
            extra.unlink()


class PannsManifestTests(unittest.TestCase):
    def _run(self, action: str, *, environment: dict[str, str] | None = None):
        return subprocess.run(
            [str(WRAPPER), action, *COMMON],
            cwd=ROOT,
            env={**os.environ, "PYTHON_BIN": str(PYTHON), **(environment or {})},
            check=True,
            capture_output=True,
            text=True,
        )

    def test_rendered_job_selects_only_panns(self) -> None:
        first = self._run("render-production-job")
        second = self._run("render-production-job")
        self.assertEqual(first.stdout, second.stdout)
        job = yaml.safe_load(first.stdout)
        self.assertEqual(job["kind"], "Job")
        self.assertEqual(
            job["metadata"]["name"],
            "perfseer-a10-panns-cnn14-production-panns001",
        )
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        environment = {row["name"]: row["value"] for row in container["env"]}
        self.assertEqual(environment["PERFSEER_MODALITY"], "audio")
        self.assertEqual(environment["PERFSEER_FAMILY_ID"], "panns_cnn14")
        self.assertEqual(
            environment["PERFSEER_RUN_ROOT"],
            "/pvc/perfseer-v3/families/panns_cnn14/panns001",
        )
        self.assertEqual(
            environment["KAGGLE_CONFIG_DIR"],
            "/var/run/secrets/perfseer-kaggle",
        )
        self.assertNotIn("envFrom", container)
        self.assertEqual(
            container["volumeMounts"][1],
            {
                "name": "kaggle-credentials",
                "mountPath": "/var/run/secrets/perfseer-kaggle",
                "readOnly": True,
            },
        )
        secret = pod["volumes"][1]["secret"]
        self.assertEqual(secret["secretName"], "kaggle-api")
        self.assertEqual(secret["defaultMode"], 0o400)
        self.assertEqual(
            secret["items"],
            [{"key": "kaggle.json", "path": "kaggle.json", "mode": 0o400}],
        )
        self.assertIn("--family-id panns_cnn14", container["args"][0])
        self.assertNotIn("--max-new-accepted", container["args"][0])
        self.assertEqual(container["image"], IMAGE)
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], 1)
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], 1)
        self.assertEqual(pod["restartPolicy"], "Never")

    def test_pilot_is_bounded(self) -> None:
        pilot = yaml.safe_load(self._run("render-pilot-job").stdout)
        script = pilot["spec"]["template"]["spec"]["containers"][0]["args"][0]
        self.assertIn("--family-id panns_cnn14", script)
        self.assertIn("--max-new-accepted 8", script)

    def test_submit_collects_required_immediate_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "kubectl.tsv"
            result = self._run(
                "submit-production-job",
                environment={
                    "KUBECTL_BIN": str(ROOT / "tests/fixtures/fake_kubectl.sh"),
                    "FAKE_KUBECTL_LOG": str(log),
                    "NAUTILUS_DISABLE_AUTO_MONITOR": "1",
                },
            )
            calls = [line.split("\t") for line in log.read_text().splitlines()]
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
            self.assertNotIn("KAGGLE", result.stdout + result.stderr + log.read_text())

    def test_analyze_is_machine_readable_and_does_not_call_kubectl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "kubectl.tsv"
            result = self._run(
                "analyze",
                environment={
                    "KUBECTL_BIN": str(ROOT / "tests/fixtures/fake_kubectl.sh"),
                    "FAKE_KUBECTL_LOG": str(log),
                },
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["family_id"], "panns_cnn14")
            self.assertEqual(payload["candidate_count"], 550)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
