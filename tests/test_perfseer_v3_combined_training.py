import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINER = runpy.run_path(str(ROOT / "scripts" / "train_perfseer_v3_combined_dev.py"))


def test_regression_accuracy_uses_existing_composite_error() -> None:
    predictions = {
        "prediction": torch.tensor([[90.0, 40.0, 40.0, 90.0, 90.0, 40.0]]),
        "target": torch.tensor([[100.0, 50.0, 50.0, 100.0, 100.0, 50.0]]),
        "regression_available": torch.tensor([True]),
    }

    assert TRAINER["_regression_accuracy"](predictions) == pytest.approx(0.9)


def test_teacher_quality_gate_is_strictly_greater_than_threshold() -> None:
    gate = TRAINER["_passes_teacher_quality_gate"]

    assert not gate(0.9, 0.9)
    assert gate(0.900001, 0.9)


def test_nautilus_job_requests_gated_teacher_student_schedule() -> None:
    documents = list(
        yaml.safe_load_all(
            (ROOT / "k8s" / "perfseer-v3-combined-l40s-training.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    job = next(document for document in documents if document["kind"] == "Job")
    args = job["spec"]["template"]["spec"]["containers"][0]["args"]

    assert args[args.index("--epochs") + 1] == "200"
    assert args[args.index("--distillation-epochs") + 1] == "100"
    assert args[args.index("--teacher-quality-gate") + 1] == "0.90"
    assert args[args.index("--target-accuracy") + 1] == "0.98"


def test_container_manifest_binds_current_training_script() -> None:
    manifest = json.loads(
        (
            ROOT
            / "containers"
            / "perfseer-v3-combined-training"
            / "build-manifest.json"
        ).read_text(encoding="utf-8")
    )
    script_sha256 = hashlib.sha256(
        (ROOT / "scripts" / "train_perfseer_v3_combined_dev.py").read_bytes()
    ).hexdigest()

    assert manifest["train_script_sha256"] == script_sha256
