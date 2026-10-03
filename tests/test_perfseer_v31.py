import copy
from dataclasses import asdict, fields

import pytest
import torch

from perfseer_v3.features import fit_normalization, apply_normalization
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v31.capture_training import _gradient_tolerances, capture, graph_from_design
from perfseer_v31.features import build_features
from perfseer_v31.inference import export_model, predict
from perfseer_v31.io import fingerprint
from perfseer_v31.model import SeerNetV31, SeerNetV31Config
from perfseer_v31.training import (build_optimizers, build_schedulers, checkpoint_payload,
                                  lr_multiplier, normalized_errors, quality_gate, regression_loss,
                                  restore_model, restore_rng, selection_key, to_batch, train_batch)
from perfseer_v31.version import TARGET_NAMES


@pytest.fixture(scope="module")
def design():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=2, optimizer={"name": "adamw", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    result, audit = capture(model, (torch.randn(3, 4),), None, None, training, architecture_key="test")
    assert audit["gradient_parameters_checked"] == 4
    return result


def tiny_model(design, hidden=16):
    features = build_features(design)
    config = SeerNetV31Config.from_registry(OperationRegistry.load(), features.layout, hidden=hidden,
                                          num_blocks=1, dropout=0., pooling_mode="phase_aware")
    return SeerNetV31(config, torch.tensor([100., 100., 200.])), features


def test_exact_connected_capture_and_configuration(design):
    graph = graph_from_design(design)
    assert graph.coverage.backward_capture_quality == "strict"
    assert {node.phase for node in graph.nodes} == {"forward", "loss", "backward", "optimizer"}
    assert design["TRAINING_CONFIG"]["microbatch_size"] == 3
    assert design["TRAINING_CONFIG"]["steps_per_epoch"] == 12
    assert any(edge.tensor_role == "parameter" for edge in graph.tensor_edges)


def test_linear_time_features_and_liveness_preserve_v3_results(design):
    from perfseer_v3.features import build_graph_features as legacy_features
    from perfseer_v3.liveness_v3 import apply_liveness as legacy_liveness
    from perfseer_v31.features_core import build_graph_features
    from perfseer_v31.liveness import apply_liveness

    graph = graph_from_design(design)
    assert asdict(apply_liveness(graph)) == asdict(legacy_liveness(graph))
    first, second = legacy_features(graph), build_graph_features(graph)
    for field in fields(first):
        if isinstance(getattr(first, field.name), torch.Tensor):
            torch.testing.assert_close(getattr(first, field.name), getattr(second, field.name), rtol=0, atol=0)


def test_shared_and_unused_parameter_accounting(design):
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self.unused = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.layer(self.layer(value))

    model = Shared()
    converted, audit = capture(model, (torch.randn(3, 4),), None, None, design["TRAINING_CONFIG"], architecture_key="shared")
    assert audit["gradient_parameters_checked"] == 2
    assert audit["forward_outputs_checked"] == 1
    assert graph_from_design(converted).global_features.total_parameter_numel == 40
    assert build_features(converted).edge_index.size(1) > len(converted["EDGE_SPECS"])


def test_metadata_cannot_change_features(design):
    changed = copy.deepcopy(design)
    changed["provenance"] = {"modality": "nlp", "dataset": "secret-team", "model_id": "arbitrary"}
    changed["graph"]["metadata"].update(changed["provenance"])
    changed["graph"]["model_fingerprint"] = "another-model-name"
    changed["graph"]["source_fingerprint"] = "another-source-name"
    first, second = build_features(design), build_features(changed)
    for field in fields(first):
        value = getattr(first, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(second, field.name), rtol=0, atol=0)


def test_three_output_forward_backward_and_optimizer_partition(design):
    model, features = tiny_model(design)
    sample = {"features": features, "target": torch.tensor([100., 40., 200.])}
    output = model.predict_batch(to_batch([sample], "cpu"))
    assert output.prediction.shape == (1, 3)
    assert (output.prediction[:, (0, 2)] > 0).all()
    assert 0 <= output.prediction[0, 1] <= 100
    optimizers, partition = build_optimizers(model, .001)
    assert all("prediction_head" not in name and "embedding" not in name for name in partition["muon"])
    names = sum(partition.values(), [])
    assert len(names) == len(set(names)) == len(list(model.parameters()))
    loss, microbatch = train_batch(model, [sample] * 4, optimizers, 2, amp=False)
    assert loss > 0 and microbatch == 2


def test_tf32_precision_aliases_are_explicit(design):
    for precision in ("tf32", "fp32_tf32", "fp32", "bf16_amp"):
        changed = copy.deepcopy(design)
        changed["TRAINING_CONFIG"]["precision"] = precision
        features = build_features(changed)
        index = features.layout.global_continuous_fields.index("precision_tf32")
        assert features.u_cont[0, index].item() == (precision in {"tf32", "fp32_tf32"})


def test_bf16_capture_tolerance_covers_one_quantization_step_only():
    actual, expected = torch.tensor([.13184929]), torch.tensor([.13964844])
    torch.testing.assert_close(actual, expected, rtol=_gradient_tolerances("bf16_amp")[0],
                               atol=_gradient_tolerances("bf16_amp")[1])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual, expected, rtol=_gradient_tolerances("fp32")[0],
                                   atol=_gradient_tolerances("fp32")[1])
    assert _gradient_tolerances("fp16_amp") == (2e-2, 2e-3)


@pytest.mark.parametrize("precision,dtype", [("fp16_amp", "float16"), ("fp16_grad_scaler", "float16"),
                                             ("bf16_amp", "bfloat16"), ("mixed_structured", "bfloat16")])
def test_precision_capture_matches_recorded_policy(design, precision, dtype, monkeypatch):
    original_grad = torch.autograd.grad
    def checked_grad(*args, **kwargs):
        assert not torch.backends.mkldnn.enabled
        return original_grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", checked_grad)
    training = {**design["TRAINING_CONFIG"], "precision": precision, "backend": "inductor_cuda"}
    converted, audit = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None,
                               training, architecture_key="precision")
    assert dtype in {edge["dtype"] for edge in converted["EDGE_SPECS"]}
    assert audit["gradient_parameters_checked"] == 2
    features = build_features(converted)
    index = features.layout.global_continuous_fields.index("execution_torch_compile")
    assert features.u_cont[0, index] == 1


def test_recovered_optimizer_defaults_preserve_adamw_decay():
    from perfseer_v31.dataset import PACKAGE, training_settings
    from perfseer_v31.io import read_json
    defaults = read_json(PACKAGE / "optimizer_defaults.json")["defaults"]
    for name, decay in (("adamw", .01), ("adam", 0.), ("sgd", 0.)):
        training = training_settings({"optimizer": {"name": name, "learning_rate": .001, "weight_decay": decay, "momentum": 0.},
                                      "scheduler": {"name": "none"}}, defaults)
        assert training["optimizer"]["weight_decay"] == decay
        assert training["optimizer"]["momentum"] == 0.


def test_failed_conversion_cannot_leave_active_manifest(tmp_path, monkeypatch):
    import perfseer_v31.dataset as dataset
    from perfseer_v31.io import atomic_write
    manifest = tmp_path / "dataset_manifest.json"
    atomic_write(manifest, {"fingerprint": "previous"})
    def fail():
        raise ValueError("injected source conversion failure")
    monkeypatch.setattr(dataset, "source_rows", fail)
    with pytest.raises(ValueError, match="injected"):
        dataset.prepare(tmp_path)
    assert not manifest.exists()
    assert (tmp_path / "dataset_manifest.previous.previous.json").exists()


def test_normalized_loss_and_strict_gate_boundaries():
    target = torch.tensor([[100., 50., 1000.]])
    prediction = torch.tensor([[90., 40., 900.]], requires_grad=True)
    torch.testing.assert_close(normalized_errors(prediction, target), torch.full((1, 3), .1))
    regression_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert quality_gate([.099, .099, .099])
    for invalid in ([.1, 0., 0.], [0., .1, 0.], [0., 0., .1], [float("nan"), 0., 0.]):
        assert not quality_gate(invalid)
    with pytest.raises(ValueError, match="physical"):
        normalized_errors(prediction, torch.tensor([[0., 50., 1000.]]))
    with pytest.raises(ValueError, match="nonfinite"):
        normalized_errors(prediction, torch.tensor([[float("nan"), 50., 1000.]]))


def test_selection_and_schedule():
    from perfseer_v31.runner import early_stopping_reason

    assert selection_key([.09, .01, .01], 9) < selection_key([.1, 0., 0.], 1)
    assert selection_key([.05] * 3, 2) < selection_key([.05] * 3, 3)
    assert lr_multiplier(0, total_steps=600, warmup_steps=10) == .1
    assert lr_multiplier(9, total_steps=600, warmup_steps=10) == 1.
    assert lr_multiplier(599, total_steps=600, warmup_steps=10) == .01
    assert early_stopping_reason({"gate_passed": True}, 2, (.05, .05, 2), stop_on_gate=True) == "validation_gate_passed"
    assert early_stopping_reason({"gate_passed": False}, 29, (.1, .05, 23), patience=6, min_epochs=30) is None
    assert early_stopping_reason({"gate_passed": False}, 30, (.1, .05, 24), patience=6, min_epochs=30) == "no_validation_improvement_for_6_epochs"


@pytest.mark.parametrize(
    "metrics,options,expected_epochs",
    [
        ({"errors": [.05] * 3, "gate_passed": True}, {"stop_on_gate": True}, [1]),
        ({"errors": [.2] * 3, "gate_passed": False},
         {"early_stopping_patience": 2, "early_stopping_min_epochs": 1}, [1, 2, 3]),
    ],
)
def test_teacher_early_stopping_restores_best(design, tmp_path, monkeypatch, metrics, options, expected_epochs):
    import perfseer_v31.runner as runner
    from pathlib import Path

    _, features = tiny_model(design)
    normalization = fit_normalization([features], split_name="train", split_fingerprint="test")
    sample = {"features": features, "target": torch.tensor([100., 40., 200.])}

    class Fixture:
        rows = [{"targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))}]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample

    saved, seen = {}, []
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in saved or original_exists(path))
    monkeypatch.setattr(runner, "T1", dict(hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware"))
    monkeypatch.setattr(runner, "memory_probe", lambda *a, **k: 1)
    monkeypatch.setattr(runner, "atomic_write", lambda path, value, **kwargs: saved.__setitem__(str(path), copy.deepcopy(value)) if str(path).endswith(".pt") else None)
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: saved[str(path)])
    monkeypatch.setattr(runner, "train_batch", lambda *a, **k: (.1, 1))

    def evaluate(*args, **kwargs):
        seen.append(len(seen) + 1)
        return metrics

    monkeypatch.setattr(runner, "evaluate", evaluate)
    _, selected = runner.run_stage(
        "teacher", Fixture(), Fixture(), tmp_path, {"fingerprint": "test"}, normalization, "cpu", **options
    )
    assert seen == expected_epochs
    assert selected["epoch"] == 1


def test_probe_finds_largest_microbatch_and_discards_updates(design, tmp_path, monkeypatch):
    import perfseer_v31.runner as runner
    model, features = tiny_model(design)
    original = copy.deepcopy(model.state_dict())
    artifact = tmp_path / "graph"
    artifact.touch()

    class Fixture:
        root = tmp_path
        rows = [{"input_path": "graph"}] * 64

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {"features": features}

    attempted = []
    def probe(model, samples, optimizers, microbatch, **kwargs):
        attempted.append(microbatch)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        while microbatch > 37:
            microbatch //= 2
        return .1, microbatch

    monkeypatch.setattr(runner, "train_batch", probe)
    assert runner.memory_probe(model, Fixture(), .001) == 37
    assert 38 in attempted and 37 in attempted
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("distill", [False, True])
def test_accumulation_matches_full_batch(design, distill):
    full, features = tiny_model(design)
    split = copy.deepcopy(full)
    teacher = tiny_model(design, hidden=24)[0].eval().requires_grad_(False) if distill else None
    samples = [{"features": features, "target": torch.tensor([100. + i, 40. + i, 200. + i])} for i in range(5)]
    full_optimizers, _ = build_optimizers(full, .001)
    split_optimizers, _ = build_optimizers(split, .001)
    train_batch(full, samples, full_optimizers, 5, teacher=teacher, amp=False)
    train_batch(split, samples, split_optimizers, 2, teacher=teacher, amp=False)
    for first, second in zip(full.parameters(), split.parameters(), strict=True):
        # Compare accumulated gradients before Muon's BF16 orthogonalization,
        # whose rounded update can amplify tiny FP32 summation differences.
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=5e-4, atol=5e-6)
        else:
            assert second.grad is None


def test_oom_retries_whole_batch_without_dropping_samples(design, monkeypatch):
    full, features = tiny_model(design)
    retried = copy.deepcopy(full)
    samples = [{"features": features, "target": torch.tensor([100. + i, 40. + i, 200. + i])} for i in range(6)]
    original = retried.predict_batch
    calls = []
    def fail_second_batch(batch):
        calls.append(int(batch.u_cont.size(0)))
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected allocation failure")
        return original(batch)
    monkeypatch.setattr(retried, "predict_batch", fail_second_batch)
    train_batch(full, samples, build_optimizers(full, .001)[0], 6, amp=False)
    _, microbatch = train_batch(retried, samples, build_optimizers(retried, .001)[0], 3, amp=False)
    assert microbatch == 1 and calls == [3, 3, 1, 1, 1, 1, 1, 1]
    for first, second in zip(full.parameters(), retried.parameters(), strict=True):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=5e-4, atol=5e-6)


def test_checkpoint_resume_export_and_reject_legacy(design, tmp_path):
    import json
    from perfseer_v31.runner import normalization_from_dict
    model, features = tiny_model(design)
    normalization = fit_normalization([features], split_name="train", split_fingerprint="test")
    assert asdict(normalization_from_dict(json.loads(json.dumps(asdict(normalization))))) == asdict(normalization)
    features = apply_normalization(features, normalization)
    sample = {"features": features, "target": torch.tensor([100., 40., 200.])}
    optimizers, _ = build_optimizers(model, .001)
    schedulers = build_schedulers(optimizers, epochs=100, warmup_epochs=5, steps_per_epoch=1)
    train_batch(model, [sample], optimizers, 1, amp=False)
    for scheduler in schedulers:
        scheduler.step()
    saved = copy.deepcopy(checkpoint_payload(model, role="student", epoch=1, normalization=asdict(normalization),
                          dataset_fingerprint="test", optimizers=optimizers, schedulers=schedulers))
    restored = restore_model(saved, dataset_fingerprint="test")
    resumed_optimizers, _ = build_optimizers(restored, .001)
    resumed_schedulers = build_schedulers(resumed_optimizers, epochs=100, warmup_epochs=5, steps_per_epoch=1)
    for optimizer, state in zip(resumed_optimizers, saved["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(resumed_schedulers, saved["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    restore_rng(saved["rng"])
    train_batch(model, [sample], optimizers, 1, amp=False)
    restore_rng(saved["rng"])
    train_batch(restored, [sample], resumed_optimizers, 1, amp=False)
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    export_model(saved, tmp_path / "student.pt")
    artifact = torch.load(tmp_path / "student.pt", weights_only=False)
    predictions = predict(artifact, [design])
    assert tuple(predictions[0]) == TARGET_NAMES
    assert not set(artifact) & {"optimizers", "schedulers"}
    with pytest.raises(ValueError, match="v3.1"):
        restore_model({"version": "v3", "target_names": ["old"] * 6})
    with pytest.raises(ValueError, match="fingerprint"):
        restore_model(saved, dataset_fingerprint="different")


def test_all_600_epochs_restore_best_not_final(design, tmp_path, monkeypatch):
    import perfseer_v31.runner as runner
    from pathlib import Path
    import signal

    _, features = tiny_model(design)
    normalization = fit_normalization([features], split_name="train", split_fingerprint="test")
    sample = {"features": features, "target": torch.tensor([100., 40., 200.])}

    class Fixture:
        rows = [{"targets": dict(zip(TARGET_NAMES, sample["target"].tolist()))}]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample

    saved, seen = {}, []
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in saved or original_exists(path))
    monkeypatch.setattr(runner, "T1", dict(hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware"))
    monkeypatch.setattr(runner, "memory_probe", lambda *a, **k: 1)
    monkeypatch.setattr(runner, "atomic_write", lambda path, value, **kwargs: saved.__setitem__(str(path), copy.deepcopy(value)) if str(path).endswith(".pt") else None)
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: saved[str(path)])

    def train(model, *args, **kwargs):
        with torch.no_grad():
            next(model.parameters()).fill_(len(seen) + 1)
        return .1, 1

    def evaluate(model, *args, **kwargs):
        epoch = len(seen) + 1
        seen.append(epoch)
        error = .2 - epoch / 6000 if epoch < 600 else .9
        return {"errors": [error] * 3, "gate_passed": False}

    monkeypatch.setattr(runner, "train_batch", train)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    def interrupt(model, *args, **kwargs):
        result = train(model, *args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(runner, "train_batch", interrupt)
    with pytest.raises(InterruptedError):
        runner.run_stage("teacher", Fixture(), Fixture(), tmp_path, {"fingerprint": "test"}, normalization, "cpu")
    assert saved[str(tmp_path / "teacher-latest.pt")]["next_batch"] == 1
    saved[str(tmp_path / "teacher-latest.pt")]["code_fingerprint"] = "approved-operational-change"
    monkeypatch.setattr(runner, "train_batch", train)
    with pytest.raises(ValueError, match="code fingerprint"):
        runner.run_stage("teacher", Fixture(), Fixture(), tmp_path, {"fingerprint": "test"}, normalization, "cpu", resume=True)
    restored, selected = runner.run_stage(
        "teacher", Fixture(), Fixture(), tmp_path, {"fingerprint": "test"}, normalization, "cpu", resume=True,
        compatible_resume_code_fingerprints=("approved-operational-change",),
    )
    assert seen == list(range(1, 601))
    assert selected["epoch"] == 599
    assert saved[str(tmp_path / "teacher-latest.pt")]["epoch"] == 600
    assert torch.all(next(restored.parameters()) == 599)
