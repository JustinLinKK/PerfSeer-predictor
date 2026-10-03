import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from perfseer_v31.io import atomic_write, read_json
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v4.capture import capture
from perfseer_v4.features import build_features, normalize_features
from perfseer_v4.model import SeerNetV4, SeerNetV4Config
from perfseer_v4.runner import fit_training_normalization
from perfseer_v4.training import (
    build_optimizers, checkpoint_payload, distillation_loss, evaluate,
    metrics_from_predictions, normalized_errors, quality_gate, regression_loss,
    selection_key, to_batch, train_batch,
)
from perfseer_v4.version import HARDWARE_ID, HEAD_GROUPS, TARGET_NAMES


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="test")
    return {"features": build_features(design), "target": torch.tensor([1., .9, 12., 20., 200., 210., 50.]),
            "sample_id": "test", "design": design}


def small_model(sample, hidden=16):
    config = SeerNetV4Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                         hidden=hidden, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    scales = sample["target"].clone()
    scales[3] = 100
    return SeerNetV4(config, scales, sample["target"])


def test_seven_target_loss_balances_heads_and_zero_sm(sample):
    target = sample["target"][None]
    for indices in HEAD_GROUPS:
        prediction = target.clone()
        prediction[:, indices] *= 2
        torch.testing.assert_close(regression_loss(prediction, target), torch.tensor(1 / 3))
    target[:, 3] = 0
    prediction = target.clone()
    prediction[:, 3] = .5
    assert normalized_errors(prediction, target).max() == .5
    torch.testing.assert_close(regression_loss(prediction, target), torch.tensor(1 / 6))


def test_exact_seven_target_metric_boundaries_selection_and_gate():
    target = torch.full((20, 7), 100.)
    prediction = target.clone()
    prediction[:, 0] = 105
    prediction[0, 0] = torch.nextafter(torch.tensor(105.), torch.tensor(float("inf")))
    metrics = metrics_from_predictions(prediction, target)
    assert metrics["within_5pct_count"][0] == 19 and quality_gate(metrics)
    prediction[1, 0] = 106
    failed = metrics_from_predictions(prediction, target)
    assert not quality_gate(failed)
    assert selection_key(metrics, 8) < selection_key(failed, 1)
    assert selection_key(metrics, 7) < selection_key(metrics, 8)
    prediction[:, 1] = 110
    prediction[0, 1] = torch.nextafter(torch.tensor(110.), torch.tensor(float("inf")))
    assert metrics_from_predictions(prediction, target)["within_10pct_count"][1] == 19
    target[:, 3] = 0
    prediction[:, 3] = 0
    prediction[0, 3], prediction[1, 3] = 1e-8, 1e-6
    metrics = metrics_from_predictions(prediction, target)
    assert metrics["zero_sm"][TARGET_NAMES[3]]["rows"] == 20
    assert metrics["within_5pct_count"][3] == 19
    for bad in (float("nan"), float("inf")):
        prediction[0, 0] = bad
        with pytest.raises(ValueError, match="nonfinite"):
            metrics_from_predictions(prediction, target)


def test_training_only_distillation_preserves_teacher_and_updates_three_heads(sample):
    teacher, student = small_model(sample, 24).eval(), small_model(sample, 16)
    teacher.requires_grad_(False)
    teacher_state = copy.deepcopy(teacher.state_dict())
    batch = to_batch([sample, sample], "cpu")
    target = torch.stack([sample["target"]] * 2)
    teacher_output = teacher.predict_batch(batch)
    output = student.predict_batch(batch)
    assert output.graph_embedding.shape == (2, 1, 16)
    loss = distillation_loss(output, teacher_output, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is None for value in teacher.parameters())
    with pytest.raises(ValueError, match="phase presence"):
        distillation_loss(output, teacher_output._replace(phase_presence=~teacher_output.phase_presence), target)
    optimizers, _ = build_optimizers(student, .001)
    changed = {**sample, "target": sample["target"] * 1.1}
    for _ in range(3):
        assert train_batch(student, [changed, changed], optimizers, 1, teacher=teacher, amp=False)[1] == 1
    assert len(student.prediction_heads) == 3
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
               for head in student.prediction_heads for parameter in head.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_state[name], rtol=0, atol=0)


def test_accumulation_and_oom_retry_preserve_entire_logical_batch(sample, monkeypatch):
    model = small_model(sample)
    warmup, _ = build_optimizers(model, .001)
    train_batch(model, [{**sample, "target": sample["target"] * 1.1}], warmup, 1, amp=False)
    reference, retried = copy.deepcopy(model), copy.deepcopy(model)
    samples = [{**sample, "target": sample["target"] * (1.1 + i * .05)} for i in range(4)]
    optimizers, _ = build_optimizers(model, .001)
    reference_optimizers, _ = build_optimizers(reference, .001)
    retry_optimizers, _ = build_optimizers(retried, .001)
    train_batch(model, samples, optimizers, 4, amp=False)
    expected_loss, _ = train_batch(reference, samples, reference_optimizers, 1, amp=False)
    original, calls = retried.predict_batch, []

    def fail_second_batch(batch):
        calls.append(batch.training.u_cont.shape[0])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected after partial accumulation")
        return original(batch)

    monkeypatch.setattr(retried, "predict_batch", fail_second_batch)
    loss, microbatch = train_batch(retried, samples, retry_optimizers, 2, amp=False)
    assert calls == [2, 2, 1, 1, 1, 1] and microbatch == 1
    assert loss == pytest.approx(expected_loss)
    for one, many, retry in zip(model.parameters(), reference.parameters(), retried.parameters(), strict=True):
        torch.testing.assert_close(one.grad, many.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(retry, many, rtol=0, atol=0)


def test_full_cpu_epoch_visits_every_row_and_rejects_resume_schedule_change(sample, tmp_path, monkeypatch):
    from perfseer_v4 import runner

    class Samples:
        rows = [{"sample_id": str(i), "targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))} for i in range(5)]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self[i] for i in range(*index.indices(len(self)))]
            return {**sample, "sample_id": self.rows[index]["sample_id"]}

    normalization = fit_training_normalization([sample], "test")
    visited = []
    monkeypatch.setattr(runner, "T1", dict(hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware"))
    monkeypatch.setattr(runner, "memory_probe", lambda *a, **kw: 1)
    original = runner.train_batch

    def record(model, samples, *args, **kwargs):
        visited.extend(row["sample_id"] for row in samples)
        return original(model, samples, *args, **kwargs)

    monkeypatch.setattr(runner, "train_batch", record)
    _, selected = runner.run_stage("teacher", Samples(), Samples(), tmp_path, {"fingerprint": "test"},
                                   normalization, "cpu", epochs=1, effective_batch=2, maximum_microbatch=1)
    assert sorted(visited) == ["0", "1", "2", "3", "4"]
    report = read_json(tmp_path / "teacher-epoch-0001.json")
    assert report["train_rows"] == report["validation_rows"] == 5 and report["optimizer_steps"] == 3
    assert selected["next_epoch"] == 2
    with pytest.raises(ValueError, match="schedule or effective batch"):
        runner.run_stage("teacher", Samples(), Samples(), tmp_path, {"fingerprint": "test"}, normalization,
                         "cpu", epochs=2, effective_batch=2, maximum_microbatch=1, resume=True)


def test_normalized_export_prediction_parity_and_independent_metrics(sample, tmp_path):
    from perfseer_v4.inference import export_model, predict
    from perfseer_v4.verification import verify_predictions

    normalization = fit_training_normalization([sample], "test")
    assert set(asdict(normalization)) == {"training", "version"}
    normalized = {**sample, "features": normalize_features(sample["features"], normalization)}
    model = small_model(sample).eval()
    path = tmp_path / "predictions.json.gz"
    metrics = evaluate(model, [normalized], 1, amp=False, prediction_path=path)
    expected = [{"sample_id": sample["sample_id"], "targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))}]
    assert verify_predictions(path, expected)["rows"] == 1
    checkpoint = checkpoint_payload(model, role="teacher", epoch=1, normalization=asdict(normalization),
                                    dataset_fingerprint="test", validation=metrics, prediction_hardware=HARDWARE_ID)
    export_model(checkpoint, tmp_path / "export.pt")
    artifact = torch.load(tmp_path / "export.pt", weights_only=False)
    result = predict(artifact, [sample["design"]])[0]
    assert tuple(result) == TARGET_NAMES
    torch.testing.assert_close(torch.tensor(list(result.values())),
                               torch.tensor(read_json(path)["rows"][0]["prediction"]), rtol=0, atol=0)
    with pytest.raises(ValueError, match="normalization"):
        predict({**artifact, "normalization_sha256": "wrong"}, [sample["design"]])


@pytest.mark.parametrize("tamper", ["count", "loss", "target", "duplicate"])
def test_independent_export_verifier_rejects_tampering(sample, tmp_path, tamper):
    from perfseer_v4.verification import verify_predictions

    path = tmp_path / "predictions.json.gz"
    evaluate(small_model(sample), [sample], 1, amp=False, prediction_path=path)
    exported = read_json(path)
    expected = [{"sample_id": sample["sample_id"], "targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))}]
    if tamper == "count":
        exported["metrics"]["within_5pct_count"][0] = 0
    elif tamper == "loss":
        exported["metrics"]["group_balanced_loss"] += 1
    elif tamper == "target":
        exported["rows"][0]["target"][0] += 1
    else:
        exported["rows"].append(exported["rows"][0])
    atomic_write(path, exported, compress=True)
    with pytest.raises(ValueError):
        verify_predictions(path, expected)


def test_teacher_validation_gate_blocks_test_and_rejects_changed_checkpoint(sample, tmp_path, monkeypatch):
    from perfseer_v4 import runner

    model = small_model(sample).eval()
    normalization = fit_training_normalization([sample], "test")
    target = sample["target"][None]
    selected = {"epoch": 1, "microbatch": 1, "validation": metrics_from_predictions(target * 2, target)}
    (tmp_path / "teacher-best.pt").write_bytes(b"checkpoint-one")
    monkeypatch.setattr(runner, "Samples", lambda *args: pytest.fail("failed validation must not access test"))
    manifest = {"fingerprint": "test"}
    gate = runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization)
    assert not gate["passed"] and gate["test"] is None
    assert runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization) == gate
    (tmp_path / "teacher-best.pt").write_bytes(b"checkpoint-two")
    with pytest.raises(ValueError, match="cached gate"):
        runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization)


def test_runner_blocks_distillation_after_teacher_gate_failure(sample, tmp_path, monkeypatch):
    from perfseer_v4 import inference, runner

    normalization = fit_training_normalization([sample], "test")
    model = small_model(sample)
    manifest = {"fingerprint": "test", "split_files": {"train": {"path": "train.json"}, "validation": {"path": "validation.json"}}}
    atomic_write(tmp_path / "dataset_manifest.json", manifest)
    atomic_write(tmp_path / "train.json", [{"input_path": "graph"}])
    atomic_write(tmp_path / "validation.json", [{"input_path": "graph"}])
    monkeypatch.setattr(runner, "verify", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner, "fit_training_normalization", lambda *args: normalization)
    monkeypatch.setattr(inference, "export_model", lambda *args: None)
    monkeypatch.setattr(runner, "run_gate", lambda *args: {"passed": False})
    stages = []

    def stage(role, *args, **kwargs):
        stages.append(role)
        assert role == "teacher"
        return model, {}

    monkeypatch.setattr(runner, "run_stage", stage)
    args = SimpleNamespace(dataset=tmp_path, output=tmp_path / "output", resume=False,
                           teacher_early_stopping_patience=6, teacher_early_stopping_min_epochs=30,
                           teacher_stop_on_validation_gate=False, compatible_resume_code_fingerprint=[],
                           teacher_epochs=600, student_epochs=100, effective_batch=256, microbatch=1,
                           local_validation=False)
    runner.run(args)
    assert stages == ["teacher"]


@pytest.mark.parametrize("checkpoint", ["teacher-best.pt", "teacher-latest.pt", "student-best.pt", "student-latest.pt"])
def test_existing_nonresume_checkpoint_rejected_before_normalization_write(tmp_path, monkeypatch, checkpoint):
    from perfseer_v4 import runner

    output = tmp_path / "output"
    output.mkdir()
    (output / checkpoint).write_bytes(b"existing-checkpoint")
    normalization = output / "normalization.json"
    normalization.write_bytes(b"existing-normalization")
    monkeypatch.setattr(runner, "verify", lambda *args: None)
    monkeypatch.setattr(runner, "fit_training_normalization", lambda *args: pytest.fail("must preserve normalization"))
    args = SimpleNamespace(dataset=tmp_path, output=output, resume=False)
    with pytest.raises(ValueError, match="output contains checkpoints"):
        runner.run(args)
    assert normalization.read_bytes() == b"existing-normalization"
    assert (output / checkpoint).read_bytes() == b"existing-checkpoint"


@pytest.mark.parametrize("variant", ["v4.2", "v4.3"])
@pytest.mark.parametrize("resume", [False, True])
def test_graph_training_rejects_variant_checkpoint_before_restore(tmp_path, monkeypatch, variant, resume):
    from perfseer_v4 import runner

    checkpoint = tmp_path / ("teacher-latest.pt" if resume else "initial.pt")
    atomic_write(checkpoint, {"model_variant": variant}, checkpoint=True)
    monkeypatch.setattr(runner, "restore_model", lambda *args, **kwargs: pytest.fail("wrong variant reached model restore"))
    monkeypatch.setattr(runner, "memory_probe", lambda *args, **kwargs: pytest.fail("wrong variant reached fitting"))
    with pytest.raises(ValueError, match="requires a v4.0 checkpoint"):
        runner.run_stage("teacher", [], [], tmp_path, {}, None, "cpu", epochs=1, resume=resume,
                         initial_checkpoint=None if resume else checkpoint)


@pytest.mark.parametrize("teacher_passed", [False, True])
def test_resume_completed_gates_restore_selected_models_without_training(sample, tmp_path, monkeypatch, teacher_passed):
    from perfseer_v31.io import file_sha256
    from perfseer_v4 import runner

    output = tmp_path / "output"
    output.mkdir()
    normalization = fit_training_normalization([sample], "test")
    atomic_write(output / "normalization.json", asdict(normalization))
    atomic_write(tmp_path / "training.json.gz", sample["design"], compress=True)
    row = {"sample_id": sample["sample_id"], "target_names": list(TARGET_NAMES),
           "targets": dict(zip(TARGET_NAMES, sample["target"].tolist())), "input_path": "training.json.gz",
           "input_sha256": file_sha256(tmp_path / "training.json.gz")}
    manifest = {"fingerprint": "test", "split_files": {name: {"path": f"{name}.json"} for name in ("train", "validation", "test")}}
    for split in manifest["split_files"]:
        atomic_write(tmp_path / f"{split}.json", [row])
    atomic_write(tmp_path / "dataset_manifest.json", manifest)
    args = SimpleNamespace(dataset=tmp_path, output=output, resume=True,
                           teacher_early_stopping_patience=6, teacher_early_stopping_min_epochs=30,
                           teacher_stop_on_validation_gate=False, compatible_resume_code_fingerprint=[],
                           teacher_epochs=600, student_epochs=100, effective_batch=256, microbatch=1,
                           local_validation=False)
    expected_roles = ["teacher", "student"] if teacher_passed else ["teacher"]
    for role in expected_roles:
        model = small_model(sample).eval()
        target = sample["target"][None]
        metrics = metrics_from_predictions(target if teacher_passed else target * 2, target)
        selected = checkpoint_payload(model, role=role, epoch=1, normalization=asdict(normalization),
                                      dataset_fingerprint="test", microbatch=1, validation=metrics,
                                      planned_epochs=getattr(args, f"{role}_epochs"), effective_batch=256,
                                      code_fingerprint=runner.code_fingerprint(),
                                      teacher_checkpoint_sha256=file_sha256(output / "teacher-best.pt") if role == "student" else None)
        atomic_write(output / f"{role}-best.pt", selected, checkpoint=True)
        gate = runner.run_gate(role, model, selected, tmp_path, output, manifest, normalization)
        assert gate["passed"] is teacher_passed

    monkeypatch.setattr(runner, "verify", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner, "fit_training_normalization", lambda *args: pytest.fail("resume must reuse normalization"))
    monkeypatch.setattr(runner, "run_stage", lambda *args, **kwargs: pytest.fail("completed stage must not train again"))
    original_restore, original_gate = runner.restore_model, runner.run_gate
    restored, checked = [], []

    def restore(payload, *, dataset_fingerprint, device):
        restored.append(payload["role"])
        assert device == "cuda"
        return original_restore(payload, dataset_fingerprint=dataset_fingerprint, device="cpu")

    def gate(role, *args):
        checked.append(role)
        return original_gate(role, *args)

    monkeypatch.setattr(runner, "restore_model", restore)
    monkeypatch.setattr(runner, "run_gate", gate)
    runner.run(args)
    assert restored == checked == expected_roles
