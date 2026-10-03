import json
import zipfile

import pytest
import torch

from perfseer_v32.capture import capture
from perfseer_v31.io import read_json
from perfseer_v32.runner import fit_training_normalization
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v32 import dataset
from perfseer_v32.features import build_features
from perfseer_v32.model import SeerNetV32, SeerNetV32Config
from perfseer_v32.training import checkpoint_payload, restore_model, build_optimizers, train_batch
from perfseer_v32.version import TARGET_NAMES


def write_archive(path, rows, model=b"model", member="models/calib_0000.py"):
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, model)
        bundle.writestr("labels/nlp/shard0.jsonl", "\n".join(json.dumps(row) for row in rows))


def source_row():
    return {"profile_point_id": "one", "model_id": "calib_0000", "status": "ok", "hardware_id": "a10",
            "dataset": {"modality": "text"}, "target_names": list(TARGET_NAMES),
            "targets": dict(zip(TARGET_NAMES, [1., .9, 12., 20., 200., 210., 50., .5, .4, 15., 100., 110.]))}


def test_merge_exact_duplicates_preserves_native_labels_and_provenance(tmp_path, monkeypatch):
    (tmp_path / "raw_source").mkdir()
    row = source_row()
    row["targets"]["infer_step_gpu_ms"] = .5
    for name in dataset.ARCHIVES:
        write_archive(tmp_path / "raw_source" / name, [row])
    monkeypatch.setattr(dataset, "SOURCE_ROWS", 2)
    monkeypatch.setattr(dataset, "UNIQUE_ROWS", 1)
    result = dataset.merge_sources(tmp_path)
    assert result["exact_duplicates"] == 1 and result["unique_rows"] == 1
    merged = read_json(tmp_path / "merged_samples.json.gz")
    assert merged[0]["native_label"] == row
    assert len(merged[0]["sources"]) == 2
    assert dataset.merge_sources(tmp_path) == result


@pytest.mark.parametrize("conflict", ["label", "model"])
def test_conflicting_duplicate_fails_without_merged_artifact(tmp_path, conflict):
    (tmp_path / "raw_source").mkdir()
    row, changed = source_row(), source_row()
    if conflict == "label":
        changed["targets"][TARGET_NAMES[0]] += 1
    write_archive(tmp_path / "raw_source" / dataset.ARCHIVES[0], [row])
    write_archive(tmp_path / "raw_source" / dataset.ARCHIVES[1], [changed], b"changed" if conflict == "model" else b"model")
    with pytest.raises(ValueError, match="conflicting"):
        dataset.merge_sources(tmp_path)
    assert not (tmp_path / "merged_samples.json.gz").exists()


@pytest.mark.parametrize("member", ["../escaped", "/absolute"])
def test_archive_path_traversal_rejected(tmp_path, member):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "invalid")
    with pytest.raises(ValueError, match="unsafe"):
        dataset.extract_archive(archive, tmp_path / "extracted")


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="test")
    features = build_features(design)
    return {"features": features, "target": torch.tensor(list(source_row()["targets"].values())), "sample_id": "test", "design": design}


def test_native_features_training_and_checkpoint_reject_v31(sample):
    from perfseer_v31.training import restore_model as restore_v31
    config = SeerNetV32Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                          hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    model = SeerNetV32(config, sample["target"])
    assert sample["features"].layout.feature_schema_version == "perfseer_v32_features_v2"
    optimizers, _ = build_optimizers(model, .001)
    loss, microbatch = train_batch(model, [sample] * 3, optimizers, 1, amp=False)
    assert loss > 0 and microbatch == 1
    payload = checkpoint_payload(model, role="teacher", epoch=1, normalization={}, dataset_fingerprint="v32")
    restored = restore_model(payload, dataset_fingerprint="v32")
    assert isinstance(restored, SeerNetV32)
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="v3.1"):
        restore_v31(payload)
    with pytest.raises(ValueError, match="v3.2"):
        restore_model({**payload, "version": "perfseer_v31_checkpoint_v1"})


def test_full_epoch_visits_all_rows_and_resume_rejects_schedule_change(sample, tmp_path, monkeypatch):
    import perfseer_v32.runner as runner

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
        visited.extend(s["sample_id"] for s in samples)
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


def small_model(sample, hidden=16):
    config = SeerNetV32Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                          hidden=hidden, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    scales = sample["target"].clone()
    scales[[3, 9]] = 100
    return SeerNetV32(config, scales, sample["target"])


def test_native_order_and_six_heads_train_and_infer_independently(sample):
    from dataclasses import replace
    from perfseer_v32.training import to_batch
    from perfseer_v32.transfer_labeling import TARGET_NAMES as measured
    from perfseer_v32.version import HEAD_GROUPS

    assert TARGET_NAMES == measured and sum(map(len, HEAD_GROUPS)) == 12
    model = small_model(sample)
    batch = to_batch([sample, sample], "cpu")
    torch.testing.assert_close(model.predict_batch(batch).prediction, sample["target"].expand(2, -1))
    optimizers, partition = build_optimizers(model, .001)
    assert len(set(sum(partition.values(), []))) == len(list(model.parameters()))
    assert not any(name.startswith("prediction_heads") for name in partition["muon"])
    changed = {**sample, "target": sample["target"] * 1.1}
    for _ in range(3):
        train_batch(model, [changed], optimizers, 1, amp=False)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
               for head in model.prediction_heads for parameter in head.parameters())
    model.eval()
    original = model.predict_batch(batch)
    altered = batch._replace(training=replace(batch.training, u_cont=batch.training.u_cont + 3))
    result = model.predict_batch(altered)
    torch.testing.assert_close(result.prediction[:, 7:], original.prediction[:, 7:], rtol=0, atol=0)
    assert not torch.equal(result.prediction[:, :7], original.prediction[:, :7])
    assert original.graph_embedding.shape == (2, 2, 16)
    assert original.phase_presence[:, 1].sum().item() == 2


def test_capture_preserves_state_rng_and_inference_ignores_training_settings():
    from perfseer_v32.capture import capture_inference, graphs_from_design
    from perfseer_v31.io import fingerprint

    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5), torch.nn.Linear(4, 2))
    model.train()
    model[1].eval()
    inputs = (torch.randn(3, 4),)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, audit = capture(model, inputs, None, None, training, architecture_key="test")
    assert audit["training"]["gradient_parameters_checked"] == 6
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    graph = graphs_from_design(design)["inference"]
    assert all(node.phase == "forward" for node in graph.nodes)
    assert not any("dropout" in node.raw_target for node in graph.nodes)
    other, _ = capture_inference(model, inputs, {**training, "optimizer": {"name": "sgd", "learning_rate": 10.},
                                               "steps_per_epoch": 999, "gradient_accumulation_steps": 8}, architecture_key="test")
    assert fingerprint(other) == fingerprint(design["inference"])


def test_group_balanced_loss_and_zero_sm_denominators(sample):
    from perfseer_v32.training import regression_loss, normalized_errors
    from perfseer_v32.version import HEAD_GROUPS

    target = sample["target"][None]
    for indices in HEAD_GROUPS:
        predicted = target.clone()
        predicted[:, indices] += target[:, indices]
        torch.testing.assert_close(regression_loss(predicted, target), torch.tensor(1 / 6))
    zeros = target.clone()
    zeros[:, [3, 9]] = 0
    predicted = zeros.clone()
    predicted[:, [3, 9]] = .5
    assert normalized_errors(predicted, zeros).max() == .5
    assert torch.isfinite(regression_loss(predicted, zeros))


def test_exact_metric_boundaries_zero_sm_selection_and_gate():
    from perfseer_v32.training import metrics_from_predictions, quality_gate, selection_key

    target = torch.full((20, 12), 100.)
    predicted = target.clone()
    predicted[:, 0] = 105
    predicted[0, 0] = torch.nextafter(torch.tensor(105.), torch.tensor(float("inf")))
    result = metrics_from_predictions(predicted, target)
    assert result["within_5pct_count"][0] == 19 and quality_gate(result)
    predicted[1, 0] = 106
    failed = metrics_from_predictions(predicted, target)
    assert not quality_gate(failed)
    assert selection_key(result, 8) < selection_key(failed, 1)
    assert selection_key(result, 7) < selection_key(result, 8)
    target[:, 3] = 0
    predicted[:, 3] = 0
    predicted[0, 3], predicted[1, 3] = 1e-8, 1e-6
    result = metrics_from_predictions(predicted, target)
    assert result["zero_sm"][TARGET_NAMES[3]]["rows"] == 20
    assert result["within_5pct_count"][3] == 19
    assert not quality_gate({**result, "metric_version": "old"})
    for corrupt in (float("nan"), float("inf")):
        predicted[0, 0] = corrupt
        with pytest.raises(ValueError, match="nonfinite"):
            metrics_from_predictions(predicted, target)


def test_paired_distillation_and_microbatch_gradients(sample):
    from perfseer_v32.training import distillation_loss, to_batch

    teacher, student = small_model(sample, 24).eval(), small_model(sample, 16)
    teacher.requires_grad_(False)
    batch = to_batch([sample, sample], "cpu")
    target = torch.stack([sample["target"]] * 2)
    teacher_output = teacher.predict_batch(batch)
    teacher_prediction = teacher_output.prediction.clone()
    teacher_prediction[:, [3, 9]] = 0
    output = student.predict_batch(batch)
    loss = distillation_loss(output, teacher_output._replace(prediction=teacher_prediction), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is None for value in teacher.parameters())
    optimizers, _ = build_optimizers(student, .001)
    assert train_batch(student, [sample, sample], optimizers, 1, teacher=teacher, amp=False)[1] == 1


def test_export_and_metric_reconstruction(sample, tmp_path):
    from dataclasses import asdict
    from perfseer_v32.inference import export_model, predict
    from perfseer_v32.training import evaluate
    from perfseer_v32.features import normalize_features
    from perfseer_v32.verification import verify_predictions

    normalization = fit_training_normalization([sample], "test")
    normalized = {**sample, "features": normalize_features(sample["features"], normalization)}
    model = small_model(sample).eval()
    path = tmp_path / "predictions.json.gz"
    metrics = evaluate(model, [normalized] * 2, 1, amp=False, prediction_path=path)
    # Give exported rows distinct identities for independent coverage verification.
    from perfseer_v31.io import atomic_write
    data = read_json(path)
    data["rows"][1]["sample_id"] = "second"
    atomic_write(path, data, compress=True)
    assert verify_predictions(path)["rows"] == 2
    policy = {"version": "perfseer_v32_shorter_timing_v1", "reference_semantics": "user_selected_shorter_time_not_remeasured_truth"}
    checkpoint = checkpoint_payload(model, role="teacher", epoch=1, normalization=asdict(normalization), dataset_fingerprint="test", validation=metrics, label_policy=policy)
    export_model(checkpoint, tmp_path / "export.pt")
    artifact = torch.load(tmp_path / "export.pt", weights_only=False)
    assert artifact["label_policy"] == policy
    with pytest.raises(ValueError, match="dataset fingerprint"):
        restore_model(checkpoint, dataset_fingerprint="revised-labels")
    result = predict(artifact, [sample["design"]])[0]
    assert tuple(result) == TARGET_NAMES
    torch.testing.assert_close(torch.tensor(list(result.values())), torch.tensor(data["rows"][0]["prediction"]), rtol=0, atol=0)
    with pytest.raises(ValueError, match="fresh training"):
        restore_model({**checkpoint, "selection_version": "old"})
    with pytest.raises(ValueError, match="normalization"):
        predict({**artifact, "normalization_sha256": "wrong"}, [sample["design"]])


def test_teacher_gate_blocks_test_and_rejects_stale_identity(sample, tmp_path, monkeypatch):
    from perfseer_v32 import runner
    from perfseer_v32.training import metrics_from_predictions

    model = small_model(sample).eval()
    normalization = fit_training_normalization([sample], "test")
    target = sample["target"][None]
    metrics = metrics_from_predictions(target * 2, target)
    selected = {"epoch": 1, "microbatch": 1, "validation": metrics}
    (tmp_path / "teacher-best.pt").write_bytes(b"checkpoint-one")
    monkeypatch.setattr(runner, "Samples", lambda *args: pytest.fail("failed validation must not access test"))
    manifest = {"fingerprint": "test"}
    gate = runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization)
    assert not gate["passed"] and gate["test"] is None
    assert runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization) == gate
    (tmp_path / "teacher-best.pt").write_bytes(b"checkpoint-two")
    with pytest.raises(ValueError, match="cached gate"):
        runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization)


def test_all_splits_load_native_targets_without_supplemental_map(sample, tmp_path):
    from perfseer_v31.io import atomic_write, file_sha256
    from perfseer_v32.runner import Samples

    atomic_write(tmp_path / "pair.json.gz", sample["design"], compress=True)
    for split in ("train", "validation", "test"):
        row = {"sample_id": split, "target_names": list(TARGET_NAMES), "targets": dict(zip(TARGET_NAMES, sample["target"].tolist())),
               "input_path": "pair.json.gz", "input_sha256": file_sha256(tmp_path / "pair.json.gz")}
        value = Samples(tmp_path, [row])[0]
        assert value["target"].shape == (12,)
        torch.testing.assert_close(value["target"], sample["target"])
        assert value["features"].training.training_mode_id.item() != value["features"].inference.training_mode_id.item()
        row["target_names"] = list(reversed(TARGET_NAMES))
        with pytest.raises(ValueError, match="target contract"):
            Samples(tmp_path, [row])[0]


def test_accumulation_and_full_batch_oom_retry_preserve_updates(sample, monkeypatch):
    import copy

    model = small_model(sample)
    warmup, _ = build_optimizers(model, .001)
    train_batch(model, [{**sample, "target": sample["target"] * 1.1}], warmup, 1, amp=False)
    reference, retried = copy.deepcopy(model), copy.deepcopy(model)
    samples = [{**sample, "target": sample["target"] * (1.1 + i * .05)} for i in range(4)]
    optimizers, _ = build_optimizers(model, .001)
    ref_optimizers, _ = build_optimizers(reference, .001)
    retry_optimizers, _ = build_optimizers(retried, .001)
    train_batch(model, samples, optimizers, 4, amp=False)
    expected_loss, _ = train_batch(reference, samples, ref_optimizers, 1, amp=False)
    original, calls = retried.predict_batch, []

    def fail_second_batch(batch):
        calls.append(batch.training.u_cont.shape[0])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected after partial accumulation")
        return original(batch)

    monkeypatch.setattr(retried, "predict_batch", fail_second_batch)
    actual_loss, actual_microbatch = train_batch(retried, samples, retry_optimizers, 2, amp=False)
    assert calls == [2, 2, 1, 1, 1, 1] and actual_microbatch == 1
    assert actual_loss == pytest.approx(expected_loss)
    for one, many, retry in zip(model.parameters(), reference.parameters(), retried.parameters(), strict=True):
        # Muon orthogonalizes in BF16; compare the accumulated FP32 gradients.
        torch.testing.assert_close(one.grad, many.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(retry, many, rtol=0, atol=0)


def test_successful_gate_reuses_test_export_and_rejects_changed_execution(sample, tmp_path, monkeypatch):
    from perfseer_v31.io import atomic_write, file_sha256
    from perfseer_v32 import runner
    from perfseer_v32.training import metrics_from_predictions

    model = small_model(sample).eval()
    normalization = fit_training_normalization([sample], "test")
    selected = {"epoch": 1, "microbatch": 1, "validation": metrics_from_predictions(sample["target"][None], sample["target"][None])}
    atomic_write(tmp_path / "pair.json.gz", sample["design"], compress=True)
    row = {"sample_id": "test", "target_names": list(TARGET_NAMES), "targets": dict(zip(TARGET_NAMES, sample["target"].tolist())),
           "input_path": "pair.json.gz", "input_sha256": file_sha256(tmp_path / "pair.json.gz")}
    atomic_write(tmp_path / "test.json.gz", [row], compress=True)
    manifest = {"fingerprint": "test", "split_files": {"test": {"path": "test.json.gz"}}}
    (tmp_path / "teacher-best.pt").write_bytes(b"selected-checkpoint")
    gate = runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization)
    assert gate["passed"]
    monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: pytest.fail("test must be reused"))
    assert runner.run_gate("teacher", model, selected, tmp_path, tmp_path, manifest, normalization) == gate
    with pytest.raises(ValueError, match="configuration"):
        runner.run_gate("teacher", model, {**selected, "microbatch": 2}, tmp_path, tmp_path, manifest, normalization)


def test_independent_verifier_rejects_changed_counts(sample, tmp_path):
    from perfseer_v31.io import atomic_write
    from perfseer_v32.training import evaluate
    from perfseer_v32.verification import verify_predictions

    path = tmp_path / "predictions.json.gz"
    evaluate(small_model(sample), [sample], 1, amp=False, prediction_path=path)
    exported = read_json(path)
    exported["metrics"]["within_5pct_count"][0] = 0
    atomic_write(path, exported, compress=True)
    with pytest.raises(ValueError, match="hit counts"):
        verify_predictions(path)


def test_heads_remain_fp32_under_backbone_autocast(sample):
    from perfseer_v32.training import to_batch

    model = small_model(sample).eval()
    dtypes = []
    handles = [head[0].register_forward_pre_hook(lambda module, args: dtypes.append(args[0].dtype)) for head in model.prediction_heads]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.predict_batch(to_batch([sample], "cpu"))
    for handle in handles:
        handle.remove()
    assert result.prediction.dtype == torch.float32 and dtypes == [torch.float32] * 6
    assert torch.isfinite(result.prediction).all()


def test_runner_never_starts_student_when_teacher_gate_fails(sample, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from perfseer_v31.io import atomic_write
    from perfseer_v32 import runner, inference

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


def test_cpu_capture_and_rng_snapshot_do_not_initialize_cuda(monkeypatch):
    from perfseer_v32.training import rng_state, restore_rng

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU work initialized CUDA RNG"))
    state = rng_state()
    assert state["cuda"] == []
    restore_rng(state)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="cpu")
