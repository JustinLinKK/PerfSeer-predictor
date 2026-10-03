from copy import deepcopy

import pytest

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32 import label_policy
from perfseer_v32.version import DATASET_VERSION, TARGET_NAMES


def row(name, epoch, *, split="train", dataset="same", input_path="graph.json"):
    targets = dict(zip(TARGET_NAMES, [epoch / 128, epoch / 128, epoch, 8., 800., 810., 26., 1., .9, 12., 790., 795.], strict=True))
    return {"version": DATASET_VERSION, "sample_id": f"{name}::{dataset}::tiny::bs8::adam::bf16_amp::a10",
            "split": split, "hardware_id": "a10", "group_id": "structure", "input_path": input_path,
            "input_sha256": "hash", "provenance": {"model_id": name, "modality": "time_series"},
            "target_names": list(TARGET_NAMES), "targets": targets, "native_targets": deepcopy(targets)}


IDENTITIES = {"graph.json": {"training": "train-input", "inference": "infer-input"}}


def test_shorter_epoch_keeps_observed_tuple_and_other_measurements():
    rows = [row("fast", 277.), row("slow", 630.)]
    original = deepcopy(rows)
    revised, changes, decisions = label_policy.revise_rows(rows, IDENTITIES, [])
    assert rows == original
    assert revised[1]["targets"]["train_epoch_ms"] == 277.
    for name in TARGET_NAMES[:3]:
        assert revised[1]["targets"][name] == original[0]["targets"][name]
    for name in TARGET_NAMES[3:]:
        assert revised[1]["targets"][name] == original[1]["targets"][name]
    assert revised[1]["native_targets"] == original[1]["native_targets"]
    assert len(decisions) == len(changes) == 1
    assert changes[0]["donor_sample_id"] == rows[0]["sample_id"]


@pytest.mark.parametrize("maximum,conflict", [(21., False), (21.00001, True)])
def test_common_five_percent_boundary(maximum, conflict):
    assert label_policy.timing_conflict([[19.], [maximum]]) is conflict
    revised, _, decisions = label_policy.revise_rows([row("a", 19.), row("b", maximum)], IDENTITIES, [])
    assert bool(decisions) is conflict
    assert revised[1]["targets"]["train_epoch_ms"] == (19. if conflict else maximum)


def test_inference_uses_wall_time_and_does_not_mix_column_minima():
    rows = [row("a", 277.), row("b", 277.)]
    rows[0]["targets"].update(infer_step_wall_ms=1., infer_step_gpu_ms=.9)
    rows[1]["targets"].update(infer_step_wall_ms=2., infer_step_gpu_ms=.7)
    for item in rows:
        item["native_targets"] = deepcopy(item["targets"])
    revised, changes, _ = label_policy.revise_rows(rows, IDENTITIES, [])
    assert revised[1]["targets"]["infer_step_wall_ms"] == 1.
    assert revised[1]["targets"]["infer_step_gpu_ms"] == .9
    assert changes[0]["head"] == "infer_timing"


@pytest.mark.parametrize("difference", ["split", "dataset", "mode_input"])
def test_no_donor_across_split_dataset_or_different_input(difference):
    kwargs = {"split": "validation"} if difference == "split" else {"dataset": "other"} if difference == "dataset" else {"input_path": "other.json"}
    rows = [row("a", 277.), row("b", 630., **kwargs)]
    identities = {**IDENTITIES, "other.json": {"training": "different-shape-or-settings", "inference": "infer-input"}}
    revised, changes, _ = label_policy.revise_rows(rows, identities, [])
    assert not changes
    assert [r["targets"] for r in revised] == [r["targets"] for r in rows]


def test_selection_is_deterministic_and_keeps_previous_consensus_donor():
    rows = [row("a", 100.), row("b", 200.), row("c", 630.)]
    change = {"sample_id": rows[0]["sample_id"], "donor_sample_id": rows[1]["sample_id"],
              "head": "train_timing", "target_names": list(TARGET_NAMES[:3]),
              "original_values": [rows[0]["targets"][n] for n in TARGET_NAMES[:3]],
              "estimated_values": [rows[1]["targets"][n] for n in TARGET_NAMES[:3]],
              "verified_measurement_correction": False}
    revised, changes, decisions = label_policy.revise_rows(rows, IDENTITIES, [change])
    assert all(r["targets"]["train_epoch_ms"] == 200. for r in revised)
    assert all(c["donor_sample_id"] == rows[1]["sample_id"] for c in changes)
    assert label_policy.revise_rows(list(reversed(rows)), IDENTITIES, [change])[2] == decisions
    invalid = {**change, "estimated_values": [1., 1., 150.]}
    with pytest.raises(ValueError, match="provenance"):
        label_policy.revise_rows(rows, IDENTITIES, [invalid])


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_invalid_timing_measurements_are_not_selected(value):
    with pytest.raises(ValueError, match="finite|physical"):
        label_policy.revise_rows([row("a", value), row("b", 630.)], IDENTITIES, [])


def test_activate_roundtrip_rejects_tampered_targets_and_preserves_originals(tmp_path, monkeypatch):
    from perfseer_v32 import dataset

    root, consensus = tmp_path / "dataset", tmp_path / "consensus"
    rows = [row("a", 277.), row("b", 630.), row("v", 400., split="validation"), row("t", 500., split="test")]
    base = {"version": DATASET_VERSION, "split_files": {}}
    for split in ("train", "validation", "test"):
        path = root / split / "samples.json.gz"
        values = [r for r in rows if r["split"] == split]
        atomic_write(path, values, compress=True)
        base["split_files"][split] = {"path": str(path.relative_to(root)), "rows": len(values), "sha256": file_sha256(path)}
    base["fingerprint"] = fingerprint(base)
    atomic_write(root / "dataset_manifest.json", base)
    original_hashes = {s: m["sha256"] for s, m in base["split_files"].items()}
    atomic_write(consensus / "changes.json.gz", [], compress=True)
    seed = {"source_dataset_fingerprint": base["fingerprint"], "source_manifest_sha256": file_sha256(root / "dataset_manifest.json"),
            "artifacts": {"changes.json.gz": file_sha256(consensus / "changes.json.gz")}}
    seed["fingerprint"] = fingerprint(seed)
    atomic_write(consensus / "manifest.json", seed)
    monkeypatch.setattr(label_policy, "input_identities", lambda *args: IDENTITIES)
    monkeypatch.setattr(dataset, "verify", lambda *args: {"status": "verified"})
    report = label_policy.activate(root, consensus, 1)
    active = read_json(root / "dataset_manifest.json")
    expected = label_policy.verify_policy(root, active, 1)
    assert report["changed_rows"] == 1 and report["changed_values"] == 3
    assert active["fingerprint"] != base["fingerprint"]
    assert expected[rows[1]["sample_id"]]["targets"]["train_epoch_ms"] == 277.
    for split, meta in base["split_files"].items():
        assert file_sha256(root / meta["path"]) == original_hashes[split]
    # Even a modified file with an updated hash cannot bypass donor reconstruction.
    path = root / active["split_files"]["train"]["path"]
    changed = read_json(path)
    changed[1]["targets"]["train_epoch_ms"] = 123.
    atomic_write(path, changed, compress=True)
    active["split_files"]["train"]["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="reconstructed policy"):
        label_policy.verify_policy(root, active, 1)
    active["label_policy"]["version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        label_policy.verify_policy(root, active, 1)
