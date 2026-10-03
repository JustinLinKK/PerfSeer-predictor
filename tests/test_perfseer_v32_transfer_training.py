from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_perfseer_v32 import sample
from test_perfseer_v32_transfer_labeling import anchor
from perfseer_v31.io import atomic_write, file_sha256, read_json
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v32 import runner, transfer_labeling, transfer_training
from perfseer_v32.features import build_features
from perfseer_v32.inference import export_model, predict
from perfseer_v32.model import SeerNetV32, SeerNetV32Config
from perfseer_v32.training import checkpoint_payload, restore_model
from perfseer_v32.version import TARGET_NAMES


def base_checkpoint(sample):
    normalization = runner.fit_training_normalization([sample], "base")
    config = SeerNetV32Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                          hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    model = SeerNetV32(config, sample["target"])
    return checkpoint_payload(model, role="teacher", epoch=26, normalization=asdict(normalization),
                              dataset_fingerprint="base")


def test_transfer_starts_from_teacher_weights_with_fresh_optimizer_and_resumes(sample, tmp_path, monkeypatch):
    base = base_checkpoint(sample)
    path = tmp_path / "base.pt"
    atomic_write(path, base, checkpoint=True)
    norm = transfer_training.transfer_normalization(base, "target")
    for mode in ("training", "inference"):
        expected = dict(base["normalization"][mode], split_fingerprint="target")
        assert asdict(getattr(norm, mode)) == expected
    manifest = {"fingerprint": "target", "prediction_hardware": transfer_training.HARDWARE_ID,
                "transfer": {"base_checkpoint_sha256": file_sha256(path), "target_hardware_id": transfer_training.HARDWARE_ID}}

    class Samples:
        rows = [{"sample_id": str(i), "targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))} for i in range(3)]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self[i] for i in range(*index.indices(len(self)))]
            return {**sample, "sample_id": str(index)}

    def probe(model, *args, **kwargs):
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, base["model_state_dict"][key], rtol=0, atol=0)
        assert model.config.checkpoint_blocks
        return 1

    monkeypatch.setattr(runner, "memory_probe", probe)
    output = tmp_path / "run"
    kwargs = dict(epochs=1, effective_batch=2, maximum_microbatch=1, initial_checkpoint=path, learning_rate=1e-4)
    model, selected = runner.run_stage("teacher", Samples(), Samples(), output, manifest, norm, "cpu", **kwargs)
    assert selected["epoch"] == 1 and selected["next_epoch"] == 2
    assert selected["transfer"] == manifest["transfer"] and selected["learning_rate"] == 1e-4
    assert selected["teacher_checkpoint_sha256"] is None
    assert any(not torch.equal(tensor, selected["model_state_dict"][key]) for key, tensor in base["model_state_dict"].items())
    restored, _ = runner.run_stage("teacher", Samples(), Samples(), output, manifest, norm, "cpu", resume=True, **kwargs)
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for changed in ({**kwargs, "learning_rate": 2e-4}, {**kwargs, "epochs": 2}):
        with pytest.raises(ValueError, match="resume"):
            runner.run_stage("teacher", Samples(), Samples(), output, manifest, norm, "cpu", resume=True, **changed)
    changed = {**manifest, "transfer": {**manifest["transfer"], "base_checkpoint_sha256": "changed"}}
    with pytest.raises(ValueError, match="lineage"):
        runner.run_stage("teacher", Samples(), Samples(), output, changed, norm, "cpu", resume=True, **kwargs)
    export_model(selected, tmp_path / "export.pt")
    exported = torch.load(tmp_path / "export.pt", weights_only=False)
    assert exported["prediction_hardware"] == transfer_training.HARDWARE_ID
    with pytest.raises(ValueError, match="design hardware"):
        predict(exported, [sample["design"]])
    model.eval().requires_grad_(False)
    teacher_weights = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(runner, "S1", dict(hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware"))
    monkeypatch.setattr(runner, "memory_probe", lambda *args, **kwargs: 1)
    _, student = runner.run_stage("student", Samples(), Samples(), output, manifest, norm, "cpu", teacher=model,
                                  epochs=1, effective_batch=2, maximum_microbatch=1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, teacher_weights[key], rtol=0, atol=0)
    export_model(student, tmp_path / "student-export.pt")
    student_export = torch.load(tmp_path / "student-export.pt", weights_only=False)
    assert student_export["teacher_checkpoint_sha256"] == file_sha256(output / "teacher-best.pt")


def test_pair_capture_preserves_original_training_graph_and_hardware(anchor, tmp_path):
    torch.set_num_threads(1)
    row = transfer_labeling.configuration(anchor, 2, "fp32_ieee", accumulation=2)
    material = {"seed": 123, "sample_keys": ["a", "b", "c"]}
    hardware = {"static": {"memory_bytes": 32 * 1024**3, "sm_count": 170, "compute_capability": 12}, "environment": {}}
    training, _ = transfer_labeling.capture_workload(anchor, row, material, hardware)
    source, output = tmp_path / "native", tmp_path / "dataset"
    relative = "graphs/test.json.gz"
    atomic_write(source / relative, training, compress=True)
    row.update(profile_point_id=row["configuration_id"], graph_path=relative, graph_sha256=file_sha256(source / relative))
    assert transfer_training._prepare_group(source, output, [row], anchor, material) == 1
    pair_path = output / "pairs" / (row["profile_point_id"] + ".json.gz")
    pair = read_json(pair_path)
    assert pair["training"] == read_json(source / relative)
    assert pair["inference"]["graph"]["metadata"]["hardware_profile"] == training["graph"]["metadata"]["hardware_profile"]
    features = build_features(pair)
    assert features.inference.metadata["hardware_id"] == transfer_training.HARDWARE_ID
    transfer_training._verify_prepared_row((output, None, "train", {
        "input_path": str(pair_path.relative_to(output)), "input_sha256": file_sha256(pair_path)}, None))
    assert transfer_training._prepare_group(source, output, [row], anchor, material) == 1
    pair_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cached transfer capture"):
        transfer_training._prepare_group(source, output, [row], anchor, material)


@pytest.mark.parametrize("bad", ["hardware", "leak", "hash", "target", "duplicate", "path"])
def test_native_label_verifier_rejects_bad_inputs(tmp_path, monkeypatch, bad):
    splits = {"train": 1, "validation": 1}
    monkeypatch.setattr(transfer_training, "SPLITS", splits)
    rows = []
    for split in splits:
        path = f"graphs/{split}.json.gz"
        atomic_write(tmp_path / path, {}, compress=True)
        rows.append({"profile_point_id": split, "split": split, "status": "ok", "hardware_id": transfer_training.HARDWARE_ID,
                     "group_id": split, "target_names": list(TARGET_NAMES), "targets": dict.fromkeys(TARGET_NAMES, 1.),
                     "graph_path": path, "graph_sha256": file_sha256(tmp_path / path)})
    if bad == "hardware": rows[0]["hardware_id"] = "a100"
    if bad == "leak": rows[0]["group_id"] = "validation"
    if bad == "hash": rows[0]["graph_sha256"] = "wrong"
    if bad == "target": rows[0]["targets"][TARGET_NAMES[0]] = 0
    if bad == "duplicate": rows[0]["profile_point_id"] = "validation"
    if bad == "path": rows[0]["graph_path"] = "../escaped"
    atomic_write(tmp_path / "labels-24h.json.gz", rows, compress=True)
    atomic_write(tmp_path / "campaign-scope-24h.json.gz", {"configuration_ids": [r["profile_point_id"] for r in rows]}, compress=True)
    with pytest.raises(ValueError):
        transfer_training.load_labels(tmp_path)


@pytest.mark.parametrize("passed", [False, True])
def test_transfer_teacher_gate_controls_student_start(sample, tmp_path, monkeypatch, passed):
    base = base_checkpoint(sample)
    atomic_write(tmp_path / "base.pt", base, checkpoint=True)
    atomic_write(tmp_path / "pair.json.gz", sample["design"], compress=True)
    row = {"sample_id": "one", "input_path": "pair.json.gz", "input_sha256": file_sha256(tmp_path / "pair.json.gz"),
           "target_names": list(TARGET_NAMES), "targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))}
    atomic_write(tmp_path / "rows.json.gz", [row], compress=True)
    manifest = {"fingerprint": "target", "split_files": {s: {"path": "rows.json.gz"} for s in ("train", "validation", "test")}}
    atomic_write(tmp_path / "dataset_manifest.json", manifest)
    monkeypatch.setattr(transfer_training, "verify", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "NVIDIA A100 80GB")
    from perfseer_v32 import training, inference
    monkeypatch.setattr(training, "restore_model", lambda payload, **kw: restore_model(payload))
    monkeypatch.setattr(training, "evaluate", lambda *args, **kw: {"baseline": True})
    monkeypatch.setattr(inference, "export_model", lambda *args: None)
    calls = []

    def stage(role, *args, **kwargs):
        calls.append(role)
        model = restore_model(base)
        if role == "teacher":
            assert kwargs["initial_checkpoint"] == tmp_path / "base.pt"
        else:
            assert "initial_checkpoint" not in kwargs
            assert not kwargs["teacher"].training
            assert all(not parameter.requires_grad for parameter in kwargs["teacher"].parameters())
        return model, base

    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner, "run_gate", lambda *args: {"passed": passed})
    args = SimpleNamespace(dataset=tmp_path, base_checkpoint=tmp_path / "base.pt", output=tmp_path / "run",
                           teacher_epochs=1, student_epochs=1, learning_rate=1e-4, effective_batch=2, microbatch=1, resume=False)
    transfer_training.train(args)
    assert calls == (["teacher", "student"] if passed else ["teacher"])
    report = read_json(args.output / "transfer-report.json")
    assert report["student_started"] == passed
    assert report["status"] == ("accepted" if passed else "teacher_gate_failed")
    if passed:
        (args.output / "transfer-report.json").unlink()
        atomic_write(args.output / "teacher-gate.json", {"passed": True})
        atomic_write(args.output / "teacher-best.pt", {**base, "transfer": report["transfer"]}, checkpoint=True)
        calls.clear()
        args.resume = True
        transfer_training.train(args)
        assert calls == ["student"]


def test_submission_dry_run_has_no_external_actions_and_requests_a100(tmp_path):
    root = Path(__file__).resolve().parents[1]
    template = root / "src/perfseer_v3.2/containers/transfer/submit.sh"
    shutil.copyfile(template if template.exists() else root / "submit.sh", tmp_path / "submit.sh")
    (tmp_path / "verify_package.py").write_text("print('fixture verified')\n")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset/dataset_manifest.json").write_text(json.dumps({"fingerprint": "test", "prediction_hardware": transfer_training.HARDWARE_ID}))
    (tmp_path / "PACKAGE-MANIFEST.json").write_text(json.dumps({"base_teacher_sha256": "teacher"}))
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("kubectl", "docker"):
        path = commands / name
        path.write_text("#!/bin/sh\necho unexpected-external-action >&2\nexit 99\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(tmp_path / "submit.sh"), "--dry-run", "--job", "transfer-test",
                             "--teacher-epochs", "2", "--student-epochs", "3"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job = read_json(tmp_path / "record/transfer-test/job.json")
    trainer = job["spec"]["template"]["spec"]["containers"][0]
    assert trainer["resources"]["requests"]["nvidia.com/a100"] == "1"
    assert trainer["resources"]["requests"] == trainer["resources"]["limits"]
    assert "nvidia.com/gpu" not in trainer["resources"]["requests"]
    assert trainer["args"][:4] == ["--teacher-epochs", "2", "--student-epochs", "3"]
    assert job["metadata"]["annotations"]["perfseer.ai/prediction-hardware"] == transfer_training.HARDWARE_ID
    assert job["spec"]["activeDeadlineSeconds"] == 172800
    result = subprocess.run(["bash", str(tmp_path / "submit.sh"), "--dry-run", "--image", "mutable:latest"],
                             env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "immutable image digest" in result.stderr
