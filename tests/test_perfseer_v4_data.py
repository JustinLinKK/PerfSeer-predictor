from dataclasses import fields

import pytest
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32.features import _build_features
from perfseer_v32.version import TARGET_NAMES as SOURCE_NAMES
from perfseer_v4 import dataset
from perfseer_v4.capture import capture, graph_from_design
from perfseer_v4.features import build_features, normalize_features
from perfseer_v4.version import INPUT_SCHEMA_VERSION, TARGET_NAMES


def make_design(width=4):
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    return capture(torch.nn.Linear(width, 2), (torch.randn(3, width),), None, None, training,
                   architecture_key=f"linear-{width}")[0]


def test_training_only_capture_preserves_state_rng_and_exact_numeric_features(monkeypatch):
    import perfseer_v32.capture as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("inference capture is forbidden")

    monkeypatch.setattr(legacy, "capture_inference", forbidden)
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
    model.eval()
    model[1].train()
    inputs = (torch.randn(3, 4),)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, audit = capture(model, inputs, None, None, training, architecture_key="state")
    assert set(design) == {"version", "training"} and set(audit) == {"training"}
    assert audit["training"]["gradient_parameters_checked"] == 6
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    actual, expected = build_features(design).training, _build_features(graph_from_design(design))
    assert actual.layout == expected.layout
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="training-only"):
        build_features({**design, "inference": {}})
    with pytest.raises(ValueError, match="training-only"):
        graph_from_design({**design, "version": dataset.SOURCE_INPUT})


def test_normalization_uses_training_only():
    from perfseer_v3.features import apply_normalization
    from perfseer_v4.runner import fit_training_normalization

    feature = build_features(make_design())
    normal = fit_training_normalization([{"features": feature}], "train-identity")
    actual = normalize_features(feature, normal)
    expected = apply_normalization(feature.training, normal.training)
    for name in ("x_cont", "edge_cont", "u_cont"):
        torch.testing.assert_close(getattr(actual.training, name), getattr(expected, name), rtol=0, atol=0)
    assert not hasattr(normal, "inference") and not hasattr(actual, "inference")


@pytest.fixture
def source_dataset(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "source"
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        design = make_design(4 + index)
        hardware = graph_from_design(design).metadata["target_hardware_id"]
        paired = {"version": dataset.SOURCE_INPUT, "training": design["training"],
                  "inference": {"deliberately_unused": True}}
        path = f"models/{index}.json.gz"
        atomic_write(source / path, paired, compress=True)
        targets = dict(zip(SOURCE_NAMES, [1., .9, 12., 20., 200., 210., 50., .5, .4, 15., 100., 110.]))
        row = {"version": dataset.SOURCE_VERSION, "sample_id": f"sample-{index}", "group_id": f"group-{index}",
               "split": split, "hardware_id": hardware, "input_path": path, "input_sha256": file_sha256(source / path),
               "target_names": list(reversed(SOURCE_NAMES)), "native_targets": targets.copy(),
               "targets": {**targets, "train_step_wall_ms": 1.2},
               "provenance": {"model_id": f"linear-{index}", "source": "original"}, "target_policy": "revised"}
        relative = f"{split}/samples.json.gz"
        atomic_write(source / relative, [row], compress=True)
        splits[split] = {"path": relative, "sha256": file_sha256(source / relative), "rows": 1}
    atomic_write(source / "policy" / "changes.json", {"note": "native measurements were not altered"})
    policy = {"version": "test", "changes": {"path": "policy/changes.json", "sha256": file_sha256(source / "policy/changes.json")},
              "historical_validation": {"path": str((source / "validation/samples.json.gz").resolve()),
                                        "sha256": splits["validation"]["sha256"]}}
    atomic_write(source / "policy/policy.json", policy)
    manifest = {"version": dataset.SOURCE_VERSION, "target_names": list(SOURCE_NAMES), "hardware_id": hardware,
                "split_files": splits, "total_rows": 3,
                "label_policy": {"path": "policy/policy.json", "sha256": file_sha256(source / "policy/policy.json"),
                                 "version": "test", "reference_semantics": "revised_reference_not_native"}}
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(source / "dataset_manifest.json", manifest)
    return source


def test_projection_preserves_named_active_labels_native_values_splits_and_policy(source_dataset, tmp_path):
    original = {str(path.relative_to(source_dataset)): file_sha256(path) for path in source_dataset.rglob("*") if path.is_file()}
    output = tmp_path / "v4"
    manifest = dataset.prepare(source_dataset, output)
    assert manifest == dataset.verify(output) == dataset.prepare(source_dataset, output)
    assert manifest["model_variant"] == "v4.0" and manifest["total_rows"] == 3
    for split, meta in manifest["split_files"].items():
        row = read_json(output / meta["path"])[0]
        original_row = read_json(source_dataset / meta["path"])[0]
        assert row["target_names"] == list(TARGET_NAMES)
        assert row["targets"] == {name: original_row["targets"][name] for name in TARGET_NAMES}
        assert row["targets"]["train_step_wall_ms"] != row["native_targets"]["train_step_wall_ms"]
        for key in ("sample_id", "group_id", "split", "native_targets", "provenance", "target_policy"):
            assert row[key] == original_row[key]
        design = read_json(output / row["input_path"])
        assert set(design) == {"version", "training"} and design["version"] == INPUT_SCHEMA_VERSION
        assert design["training"] == read_json(source_dataset / original_row["input_path"])["training"]
    assert file_sha256(output / "source/policy/changes.json") == original["policy/changes.json"]
    assert original == {str(path.relative_to(source_dataset)): file_sha256(path) for path in source_dataset.rglob("*") if path.is_file()}


def test_transfer_manifest_and_target_arrays_project_by_name(source_dataset, tmp_path):
    manifest = read_json(source_dataset / "dataset_manifest.json")
    manifest["version"] = dataset.SOURCE_VERSIONS[1]
    manifest["prediction_hardware"] = manifest.pop("hardware_id")
    for split, meta in manifest["split_files"].items():
        rows = read_json(source_dataset / meta["path"])
        row = rows[0]
        row.pop("split")
        row.pop("version")
        row["targets"] = [row["targets"][name] for name in row["target_names"]]
        row["native_targets"] = [row["native_targets"][name] for name in row["target_names"]]
        atomic_write(source_dataset / meta["path"], rows, compress=True)
        meta["sha256"] = file_sha256(source_dataset / meta["path"])
    policy = read_json(source_dataset / manifest["label_policy"]["path"])
    policy["historical_validation"]["sha256"] = manifest["split_files"]["validation"]["sha256"]
    atomic_write(source_dataset / manifest["label_policy"]["path"], policy)
    manifest["label_policy"]["sha256"] = file_sha256(source_dataset / manifest["label_policy"]["path"])
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    atomic_write(source_dataset / "dataset_manifest.json", manifest)
    projected = dataset.prepare(source_dataset, tmp_path / "v4")
    row = read_json(tmp_path / "v4" / projected["split_files"]["train"]["path"])[0]
    assert row["targets"]["train_step_wall_ms"] == 1.2 and row["split"] == "train"
    original = read_json(source_dataset / manifest["split_files"]["train"]["path"])[0]
    assert row["native_targets"] == dict(zip(original["target_names"], original["native_targets"], strict=True))
    assert row["native_targets"]["train_step_wall_ms"] == 1. and row["native_targets"]["infer_step_wall_ms"] == .5
    assert read_json(tmp_path / "v4/source" / manifest["split_files"]["train"]["path"])[0] == original


@pytest.mark.parametrize("kind", ["labels", "native", "group", "hash", "input", "policy"])
def test_verifier_detects_corruption_even_with_rehashed_split(source_dataset, tmp_path, kind):
    output = tmp_path / "v4"
    manifest = dataset.prepare(source_dataset, output)
    meta = manifest["split_files"]["train"]
    rows = read_json(output / meta["path"])
    if kind == "input":
        (output / rows[0]["input_path"]).write_bytes(b"corrupt")
    elif kind == "policy":
        atomic_write(output / "source/policy/changes.json", {"changed": True})
    else:
        if kind == "labels":
            rows[0]["targets"][TARGET_NAMES[0]] += 1
        elif kind == "native":
            rows[0]["native_targets"][TARGET_NAMES[0]] += 1
        elif kind == "group":
            rows[0]["group_id"] = "group-2"
        else:
            rows[0]["input_sha256"] = "0" * 64
        atomic_write(output / meta["path"], rows, compress=True)
        meta["sha256"] = file_sha256(output / meta["path"])
        manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        atomic_write(output / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError):
        dataset.verify(output)


@pytest.mark.parametrize("kind", ["group", "duplicate", "range", "path"])
def test_source_identity_range_and_path_violations_rejected(source_dataset, tmp_path, kind):
    manifest = read_json(source_dataset / "dataset_manifest.json")
    meta = manifest["split_files"]["test"]
    rows = read_json(source_dataset / meta["path"])
    if kind == "group":
        rows[0]["group_id"] = "group-0"
    elif kind == "duplicate":
        rows[0]["sample_id"] = "sample-0"
    elif kind == "range":
        rows[0]["targets"]["train_avg_sm_util_percent"] = 101.
    else:
        rows[0]["input_path"] = "../outside.json"
    atomic_write(source_dataset / meta["path"], rows, compress=True)
    meta["sha256"] = file_sha256(source_dataset / meta["path"])
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    atomic_write(source_dataset / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError):
        dataset.prepare(source_dataset, tmp_path / "v4")


def test_historical_directory_cannot_be_overwritten(source_dataset):
    with pytest.raises(ValueError, match="separate"):
        dataset.prepare(source_dataset, source_dataset)
    with pytest.raises(ValueError, match="separate"):
        dataset.prepare(source_dataset, source_dataset / "v4")
