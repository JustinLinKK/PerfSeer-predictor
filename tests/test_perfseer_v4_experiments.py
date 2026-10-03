import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32.calibration_contracts import ENVIRONMENT_VERSION
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v4.capture import capture
from perfseer_v4.features import build_features, normalize_features
from perfseer_v4.model import SeerNetV4, SeerNetV4Config
from perfseer_v4.runner import fit_training_normalization
from perfseer_v4.training import build_optimizers, checkpoint_payload, restore_model, to_batch, train_batch
from perfseer_v4.version import HARDWARE_ID, TARGET_NAMES


def environment():
    return dict(version=ENVIRONMENT_VERSION, hardware_id=HARDWARE_ID, gpu_model="test", capacity_bytes=24 * 1024**3,
                partition="none", driver="test", cuda="test", framework="test", libraries={}, allocator="default",
                backend_policy="eager", host_context={"threads": 1})


@pytest.fixture
def trained_source():
    torch.set_num_threads(1)
    torch.manual_seed(47)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="test")
    sample = {"features": build_features(design), "target": torch.tensor([1., .9, 12., 20., 200., 210., 50.]),
              "sample_id": "source", "design": design}
    normalization = fit_training_normalization([sample], "source-dataset")
    sample["features"] = normalize_features(sample["features"], normalization)
    config = SeerNetV4Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                         hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    scales = sample["target"].clone()
    scales[3] = 100
    model = SeerNetV4(config, scales, sample["target"])
    optimizers, _ = build_optimizers(model, .001)
    for _ in range(3):
        train_batch(model, [{**sample, "target": sample["target"] * 1.1}], optimizers, 1, amp=False)
    payload = checkpoint_payload(model, role="student", epoch=3, normalization=asdict(normalization),
                                 dataset_fingerprint="source-dataset", prediction_hardware=HARDWARE_ID)
    return payload, sample


@pytest.mark.parametrize("variant", ["v4.0-finetune", "v4.2"])
def test_bounded_neural_adaptation_lineage_source_preservation_and_reload(trained_source, variant):
    from perfseer_v4.inference import predict
    from perfseer_v4.transfer import fit_neural, source_identity

    source, sample = trained_source
    original = copy.deepcopy(source)
    train = [{**sample, "sample_id": f"train-{i}", "group_id": f"train-group-{i}", "workload_id": f"train-work-{i}",
              "split": "train", "target": sample["target"] * (1.1 + i * .05)} for i in range(2)]
    validation = [{**sample, "sample_id": "validation", "group_id": "validation-group", "workload_id": "validation-work",
                   "split": "validation", "target": sample["target"] * 1.2}]
    adapted, result = fit_neural(source, train, validation, variant=variant, dataset_fingerprint="target-dataset",
                                 hardware=HARDWARE_ID, environment=environment(), epochs=2, effective_batch=2, microbatch=1)
    assert result["epochs_completed"] == 2 and result["epoch"] in (1, 2)
    assert adapted["transfer"]["variant"] == variant and adapted["transfer"]["experimental"]
    assert adapted["transfer"]["fit_ids"] == ["train-0", "train-1"]
    assert adapted["transfer"]["validation_ids"] == ["validation"]
    assert adapted["transfer"]["source_identity"] == source_identity(original)
    assert adapted["normalization"]["training"]["split_fingerprint"] == "target-dataset"
    assert adapted["normalization_sha256"] == fingerprint(adapted["normalization"])
    for name, value in source["model_state_dict"].items():
        torch.testing.assert_close(value, original["model_state_dict"][name], rtol=0, atol=0)
        if variant == "v4.2":
            torch.testing.assert_close(adapted["model_state_dict"]["base." + name], value, rtol=0, atol=0)
    model = restore_model(adapted, dataset_fingerprint="target-dataset").eval()
    result = model.predict_batch(to_batch(validation, "cpu")).prediction
    assert result.shape == (1, 7) and torch.isfinite(result).all()
    with pytest.raises(ValueError, match="bound target environment"):
        predict(adapted, [sample["design"]])
    with pytest.raises(ValueError, match="bound target environment"):
        predict(adapted, [sample["design"]], environment={**environment(), "driver": "changed"})
    prediction = predict(adapted, [sample["design"]], environment=environment())
    torch.testing.assert_close(torch.tensor([list(prediction[0].values())]), result, rtol=0, atol=0)
    if variant == "v4.2":
        assert all(not parameter.requires_grad for parameter in model.base.parameters())
        assert model.displacement_penalty() > 0


def test_neural_adaptation_rejects_wrong_environment_before_fitting(trained_source):
    from perfseer_v4.transfer import fit_neural

    source, sample = trained_source
    with pytest.raises(ValueError, match="hardware differ"):
        fit_neural(source, [sample], [sample], variant="v4.2", dataset_fingerprint="target",
                   hardware="another-gpu", environment=environment(), epochs=1)


@pytest.mark.parametrize("mismatch", ["sample_id", "group_id", "workload_id", "split", "hardware"])
def test_neural_transfer_rejects_leaking_or_wrong_hardware_samples(trained_source, monkeypatch, mismatch):
    from perfseer_v4 import transfer

    source, sample = trained_source
    train = {**sample, "sample_id": "fit", "group_id": "fit-group", "workload_id": "fit-work", "split": "train"}
    validation = {**sample, "sample_id": "validation", "group_id": "validation-group", "workload_id": "validation-work",
                  "split": "validation"}
    if mismatch == "hardware":
        features = validation["features"]
        validation["features"] = replace(features, training=replace(features.training,
            metadata={**features.training.metadata, "hardware_id": "another-gpu"}))
    else:
        validation[mismatch] = train[mismatch]
    monkeypatch.setattr(transfer, "_fit", lambda *args, **kwargs: pytest.fail("invalid samples reached fitting"))
    with pytest.raises(ValueError, match="identities|leakage|hardware differs"):
        transfer.fit_neural(source, [train], [validation], variant="v4.2", dataset_fingerprint="target",
                            hardware=HARDWARE_ID, environment=environment(), epochs=1)


def test_fit_budgets_are_nested_target_blind_and_cover_groups():
    from perfseer_v4.experiments import choose_fit_ids

    rows = [{"sample_id": f"{group}-{index}", "group_id": group, "split": "train", "targets": {"unused": index}}
            for group, count in (("a", 6), ("b", 3), ("c", 2)) for index in range(count)]
    previous = []
    for budget in (1, 3, 5, 11):
        actual = choose_fit_ids(rows, budget, 29)
        assert len(actual) == len(set(actual)) == budget and actual[:len(previous)] == previous
        assert actual == choose_fit_ids(list(reversed(rows)), budget, 29)
        assert actual == choose_fit_ids([{**row, "targets": None} for row in rows], budget, 29)
        previous = actual
    assert {key.split("-")[0] for key in choose_fit_ids(rows, 3, 29)} == {"a", "b", "c"}
    for budget in (0, 12, True):
        with pytest.raises(ValueError, match="budget"):
            choose_fit_ids(rows, budget, 29)
    with pytest.raises(ValueError, match="training"):
        choose_fit_ids([{**rows[0], "split": "test"}], 1, 29)
    with pytest.raises(ValueError, match="duplicate"):
        choose_fit_ids([rows[0], rows[0]], 1, 29)


@pytest.fixture
def comparison(tmp_path, monkeypatch, trained_source):
    from perfseer_v4 import experiments
    from perfseer_v4.resource_model import ResourceMLP, ResourceMLPConfig

    source, sample = trained_source
    dataset = tmp_path / "dataset"
    splits, metadata = {}, {}
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        rows = []
        for index in range(count):
            key = f"{split}-{index}"
            relative = f"designs/{key}.json"
            atomic_write(dataset / relative, sample["design"])
            rows.append({"sample_id": key, "group_id": key, "split": split, "input_path": relative,
                         "input_sha256": fingerprint(key), "target_names": list(TARGET_NAMES),
                         "targets": dict(zip(TARGET_NAMES, sample["target"].tolist())),
                         "native_targets": dict(zip(TARGET_NAMES, (sample["target"] * 1.5).tolist()))})
        relative = f"{split}.json"
        atomic_write(dataset / relative, rows)
        metadata[split] = {"path": relative, "sha256": file_sha256(dataset / relative), "rows": count}
        splits[split] = rows
    manifest = {"fingerprint": "target-dataset", "prediction_hardware": HARDWARE_ID, "split_files": metadata,
                "label_policy": {"version": "synthetic-revised-labels"}}
    source_path = tmp_path / "source.pt"
    atomic_write(source_path, source, checkpoint=True)
    config = ResourceMLPConfig()
    resource_model = ResourceMLP(config, sample["target"], torch.zeros(config.resource_dim),
                                 torch.ones(config.resource_dim), sample["target"])
    resource = checkpoint_payload(resource_model, role="resource", epoch=1, normalization=source["normalization"],
                                  dataset_fingerprint=source["dataset_fingerprint"], prediction_hardware=HARDWARE_ID)
    resource_path = tmp_path / "resource.pt"
    atomic_write(resource_path, resource, checkpoint=True)
    monkeypatch.setattr(experiments, "verify", lambda path: manifest)
    monkeypatch.setattr(experiments, "_samples", lambda root, rows, payload: rows)
    predictions = dict(zip(TARGET_NAMES, sample["target"].tolist()))
    monkeypatch.setattr(experiments, "_predict", lambda payload, samples, microbatch, device="cpu":
                        ([dict(predictions) for _ in samples], {"scope": "mocked forward", "microbatch": microbatch}))
    fit_calls = []

    def neural(payload, train, validation, **kwargs):
        fit_calls.append((kwargs["variant"], [row["sample_id"] for row in train], [row["sample_id"] for row in validation]))
        return copy.deepcopy(payload), {"epoch": 1, "trainable_parameters": 1, "epochs_completed": 1}

    monkeypatch.setattr(experiments, "fit_neural", neural)
    reads = []
    permit_test = [False]

    def guarded_read(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == (dataset / "test.json").resolve() and not permit_test[0]:
            pytest.fail("test labels were accessed before sealed evaluation")
        return read_json(path)

    monkeypatch.setattr(experiments, "read_json", guarded_read)
    return {"dataset": dataset, "source": source_path, "resource": resource_path, "output": tmp_path / "selection",
            "splits": splits, "manifest": manifest, "reads": reads, "permit_test": permit_test, "fit_calls": fit_calls}


def test_all_candidate_selection_is_test_blind_and_evaluation_keeps_native_labels_separate(comparison, tmp_path):
    from perfseer_v4 import experiments

    data = comparison
    selected = experiments.select(data["dataset"], data["source"], data["resource"], environment(), data["output"],
                                  budgets=(2, 4), seeds=(11,), epochs=1, microbatch=1)
    assert len(selected["trials"]) == 2 and selected["deployment_approved"] is False
    assert selected["cost"]["fit_budget_unit"] == "distinct recorded sample IDs"
    assert selected["cost"]["validation_samples"] == 2
    assert selected["cost"]["validation_distinct_training_inputs"] == 2
    assert selected["cost"]["validation_independent_groups"] == 2
    assert selected["cost"]["new_gpu_measurements"] == 0
    assert "not available" in selected["cost"]["repeat_measurements"]
    for trial in selected["trials"]:
        assert trial["fit_cost"] == {"recorded_samples": trial["budget"], "distinct_training_inputs": trial["budget"],
                                     "independent_groups": trial["budget"]}
        assert set(trial["candidates"]) == set(experiments.VARIANTS)
        for candidate in trial["candidates"].values():
            assert candidate["fit_ids"] == trial["fit_ids"]
            assert candidate["validation_ids"] == ["validation-0", "validation-1"]
            assert candidate["validation"]["mae"] == [0.] * 7
            assert all(value > 0 for value in candidate["validation_native"]["mae"])
    assert set(selected["trials"][0]["fit_ids"]) < set(selected["trials"][1]["fit_ids"])
    assert len(data["fit_calls"]) == 4
    assert (data["dataset"] / "test.json").resolve() not in data["reads"]
    data["permit_test"][0] = True
    report = experiments.evaluate_selection(data["output"] / "selection.json", tmp_path / "evaluation.json", microbatch=1)
    assert report["status"] == "experimental_not_promoted" and report["deployment_approved"] is False
    assert report["test_ids"] == ["test-0", "test-1"]
    for trial in report["trials"]:
        assert trial["fit_cost"] == {"recorded_samples": trial["budget"], "distinct_training_inputs": trial["budget"],
                                     "independent_groups": trial["budget"]}
        for candidate in trial["candidates"].values():
            assert candidate["evidence_status"] == "inconclusive"
            assert candidate["test"]["mae"] == [0.] * 7
            assert all(value > 0 for value in candidate["test_native"]["mae"])
            assert set(candidate["primary_paired_intervals"]) == {TARGET_NAMES[0], TARGET_NAMES[5]}


def test_repeated_inputs_report_sample_budget_and_independent_counts_separately(comparison, tmp_path):
    from perfseer_v4 import experiments

    data = comparison
    for split in ("train", "validation"):
        rows = data["splits"][split]
        rows[1]["input_sha256"] = rows[0]["input_sha256"]
        rows[1]["group_id"] = rows[0]["group_id"]
        path = data["dataset"] / data["manifest"]["split_files"][split]["path"]
        atomic_write(path, rows)
        data["manifest"]["split_files"][split]["sha256"] = file_sha256(path)
    selected = experiments.select(data["dataset"], data["source"], None, environment(), data["output"],
                                  budgets=(4,), seeds=(11,), variants=("source",))
    trial = selected["trials"][0]
    assert len(trial["fit_ids"]) == trial["budget"] == 4
    assert trial["fit_cost"] == {"recorded_samples": 4, "distinct_training_inputs": 3, "independent_groups": 3}
    assert selected["cost"]["fit_budget_unit"] == "distinct recorded sample IDs"
    assert selected["cost"]["validation_samples"] == 2
    assert selected["cost"]["validation_distinct_training_inputs"] == selected["cost"]["validation_independent_groups"] == 1
    data["permit_test"][0] = True
    report = experiments.evaluate_selection(data["output"] / "selection.json", tmp_path / "evaluation.json")
    assert report["trials"][0]["fit_cost"] == trial["fit_cost"]
    assert report["cost"] == selected["cost"]


@pytest.mark.parametrize("tamper", ["source", "adapter", "environment", "selection", "dataset"])
def test_evaluation_rejects_tampering_before_test_label_access(comparison, tmp_path, tamper):
    from perfseer_v4 import experiments

    data = comparison
    selected = experiments.select(data["dataset"], data["source"], None, environment(), data["output"],
                                  budgets=(2,), seeds=(11,), variants=("source", "v4.1"))
    selection_path = data["output"] / "selection.json"
    if tamper == "source":
        data["source"].write_bytes(data["source"].read_bytes() + b"tampered")
    elif tamper == "adapter":
        candidate = selected["trials"][0]["candidates"]["v4.1"]
        adapter = data["output"] / candidate["path"]
        adapter.write_bytes(adapter.read_bytes() + b" ")
    elif tamper == "environment":
        selected["environment"]["driver"] = "changed"
        selected["fingerprint"] = fingerprint({key: value for key, value in selected.items() if key != "fingerprint"})
        atomic_write(selection_path, selected)
    elif tamper == "selection":
        selected["trials"][0]["fit_ids"] = []
        atomic_write(selection_path, selected)
    else:
        data["manifest"]["fingerprint"] = "changed-dataset"
    with pytest.raises(ValueError):
        experiments.evaluate_selection(selection_path, tmp_path / "evaluation.json")
    assert not (tmp_path / "evaluation.json").exists()
    assert (data["dataset"] / "test.json").resolve() not in data["reads"]


def test_resource_source_one_epoch_uses_training_statistics_without_test_access(comparison, trained_source, tmp_path, monkeypatch):
    from perfseer_v4 import dataset, transfer

    data = comparison
    _, sample = trained_source
    features = build_features(sample["design"], include_resources=True)
    monkeypatch.setattr(dataset, "verify", lambda path: data["manifest"])

    def samples(root, rows, normalization=None, *, include_resources=False):
        assert include_resources
        result = []
        for row in rows:
            current = features if row["split"] == "train" else replace(features, resources=features.resources + 100)
            if normalization is not None:
                current = normalize_features(current, normalization)
            result.append({"sample_id": row["sample_id"], "features": current,
                           "target": torch.tensor([row["targets"][name] for name in TARGET_NAMES])})
        return result

    def guarded_read(path):
        assert Path(path).resolve() != (data["dataset"] / "test.json").resolve()
        return read_json(path)

    monkeypatch.setattr(transfer, "Samples", samples)
    monkeypatch.setattr(transfer, "read_json", guarded_read)
    args = SimpleNamespace(dataset=data["dataset"], output=tmp_path / "resource-training", seed=11,
                           device="cpu", epochs=1, microbatch=1)
    result = transfer.train_resource(args)
    assert result["epochs_completed"] == result["epoch"] == 1
    artifact = torch.load(args.output / "resource-best.pt", weights_only=False)
    model = restore_model(artifact, dataset_fingerprint=data["manifest"]["fingerprint"])
    assert artifact["model_variant"] == "v4.3"
    torch.testing.assert_close(model.resource_mean, features.resources[0], rtol=0, atol=0)
    torch.testing.assert_close(model.resource_scale, torch.ones_like(model.resource_scale), rtol=0, atol=0)
    report = read_json(args.output / "training-report.json")
    assert report["model_variant"] == "v4.3"
    assert report["status"] == "source_trained_not_promoted" and report["test_evaluated"] is False
