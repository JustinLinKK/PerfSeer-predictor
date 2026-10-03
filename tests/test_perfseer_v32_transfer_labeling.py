import copy
import hashlib
import json
import sys

import pytest
import torch

from perfseer_v31.io import atomic_write, fingerprint, read_json
from perfseer_v32 import transfer_labeling as labeling
from perfseer_v32 import transfer_verification as verifier


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def anchor():
    node = dict(id=0, type="Gemm", preds=[], input_index=0,
                args={"linear_in_features": 4, "linear_out_features": 2, "linear_bias": 1},
                memory_info={"input_features": 4, "output_features": 2, "rank": 2})
    model = dict(NODE_SPECS=[node], INPUT_SPECS=[dict(name="features", shape=[1, 4], dtype="float32", kind="float")])
    identity = labeling.canonical_workload(model, dict(dataset_id="fixture", subset_id="tiny", num_samples=8))
    return dict(anchor_id=fingerprint(identity), identity=identity, model=model, split="train", group_id="group",
                modality="tabular", source_ids={p: ["source-" + p] for p in labeling.PRECISIONS},
                operation_types=["Gemm"], structure=dict(parameters=10, depth=1, width=2, input_numel=4))


def test_twelve_target_order_and_required_average_vram():
    expected = (
        "train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms", "train_avg_sm_util_percent",
        "train_avg_vram_mib", "train_peak_vram_mib", "train_peak_torch_allocated_mib",
        "infer_step_wall_ms", "infer_step_gpu_ms", "infer_avg_sm_util_percent",
        "infer_avg_vram_mib", "infer_peak_vram_mib",
    )
    assert labeling.TARGET_NAMES == expected
    values = dict.fromkeys(expected, 1.)
    values["train_epoch_ms_step_extrapolated"] = 1.
    verifier.checked_targets(values)
    for omitted in expected:
        with pytest.raises(ValueError, match="twelve-target"):
            verifier.checked_targets({k: v for k, v in values.items() if k != omitted})


@pytest.mark.parametrize("name,value", [("infer_avg_vram_mib", float("nan")),
                                       ("train_avg_sm_util_percent", 101), ("infer_step_gpu_ms", 0)])
def test_invalid_physical_targets(name, value):
    targets = dict.fromkeys((*labeling.TARGET_NAMES, "train_epoch_ms_step_extrapolated"), 1.)
    targets[name] = value
    with pytest.raises(ValueError):
        verifier.checked_targets(targets)


def test_campaign_cartesian_coverage_and_nested_budgets(anchor):
    anchors = []
    for split, count in labeling.ANCHOR_COUNTS.items():
        for index in range(count):
            item = copy.deepcopy(anchor)
            item.update(anchor_id=fingerprint([split, index]), split=split, group_id=f"{split}-{index}",
                        modality=labeling.MODALITIES[index % 6],
                        structure=dict(parameters=index + 10, depth=index % 5 + 1, width=index % 4 + 1, input_numel=index + 4))
            anchors.append(item)
    rows, experiments = labeling.build_campaign(anchors)
    assert labeling.build_campaign(anchors) == (rows, experiments)
    assert len(rows) == len({r["configuration_id"] for r in rows}) == 10240
    assert {s: sum(r["split"] == s for r in rows) for s in labeling.SPLIT_COUNTS} == labeling.SPLIT_COUNTS
    assert sum(r["a10_matched"] for r in rows) == 992
    assert len(experiments["pilot"]) == 48
    assert len(set(experiments["repeat_panel"])) == 512
    previous = []
    by_id = {r["configuration_id"]: r for r in rows}
    for budget in labeling.BUDGETS:
        selected = experiments["nested_training_budgets"][str(budget)]
        assert len(selected) == len(set(selected)) == budget
        assert selected[:len(previous)] == previous
        assert all(by_id[key]["split"] == "train" for key in selected)
        previous = selected
    assert all(by_id[key]["training"]["microbatch_size"] not in (16, 64)
               for key in experiments["batch_generalization_train"])
    for row in rows:
        if not row["a10_matched"]:
            assert row["paired_source_ids"] == []
            assert "targets" not in row


def test_24h_campaign_scope_preserves_required_coverage(anchor):
    anchors = []
    for split, count in labeling.ANCHOR_COUNTS.items():
        for index in range(count):
            item = copy.deepcopy(anchor)
            item.update(anchor_id=fingerprint([split, index]), split=split, group_id=f"{split}-{index}",
                        modality=labeling.MODALITIES[index % 6],
                        structure=dict(parameters=index + 10, depth=index % 5 + 1,
                                       width=index % 4 + 1, input_numel=index + 4))
            anchors.append(item)
    rows, experiments = labeling.build_campaign(anchors)
    scope = labeling.build_campaign_scope(rows, experiments, "campaign", "24h")
    assert scope == labeling.build_campaign_scope(rows, experiments, "campaign", "24h")
    assert scope["planned_records"] == 4128
    assert scope["planned_attempts"] == 4256
    assert scope["split_counts"] == {"train": 2988, "validation": 576, "test": 564}
    assert scope["cohort_counts"] == {"batch": 2976, "optimizer": 768, "accumulation": 384}
    assert scope["matched_records"] == 992
    assert len(scope["repeat_panel"]) == 64
    assert len(scope["pilot"]) == 48
    selected = {row["configuration_id"]: row for row in rows if row["configuration_id"] in scope["configuration_ids"]}
    assert len(selected) == 4128
    assert {row["anchor_id"] for row in selected.values()} == {item["anchor_id"] for item in anchors}
    assert {row["training"]["precision"] for row in selected.values()} == set(labeling.PRECISIONS)
    assert set(scope["repeat_panel"]) <= set(experiments["repeat_panel"])
    previous = []
    for budget in labeling.REDUCED_BUDGETS:
        current = scope["nested_training_budgets"][str(budget)]
        assert len(current) == len(set(current)) == budget
        assert current[:len(previous)] == previous
        previous = current


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd"])
def test_accumulation_matches_independent_logical_batch(optimizer_name):
    torch.manual_seed(13)
    accumulated, reference = torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)
    reference.load_state_dict(accumulated.state_dict())
    inputs = torch.randn(12, 4)
    settings = labeling.optimizer_settings(optimizer_name)
    first = labeling.make_optimizer(accumulated, settings)
    second = labeling.make_optimizer(reference, settings)
    calls = []
    original = first.step

    def step(*args, **kwargs):
        calls.append("step")
        return original(*args, **kwargs)

    first.step = step
    loss = labeling.training_step(accumulated, first, [(x,) for x in inputs.split(3)], "fp32_ieee")
    second.zero_grad(set_to_none=True)
    out = reference(inputs)
    expected_loss = out.square().mean()
    expected_loss.backward()
    second.step()
    torch.testing.assert_close(loss, expected_loss)
    for actual, expected in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert calls == ["step"]


def test_microbatches_are_materialized_lazily():
    calls = []
    batches = labeling.InputBatches(lambda: calls.append(1) or (torch.ones(2, 4),), 4)
    assert len(batches) == 4 and calls == []
    next(iter(batches))
    assert calls == [1]


def test_worker_command_uses_checkout_bootstrap_launcher(tmp_path):
    command = labeling.worker_command(tmp_path, "configuration", 2)
    assert command == [sys.executable, str(labeling.LAUNCHER_PATH), "_worker", "--output", str(tmp_path),
                       "--configuration-id", "configuration", "--repetition", "2"]


def test_source_material_replay_and_hash_changes(tmp_path, anchor):
    raw = tmp_path / "raw/fixture"
    raw.mkdir(parents=True)
    keys = [f"sample-{i}" for i in range(8)]
    for index, name in enumerate(keys):
        (raw / name).write_bytes(f"actual-input-{index}".encode())
    mask = tmp_path / "prepared/fixture/subset_masks/tiny.json"
    atomic_write(mask, {"sample_keys": keys})
    materials = labeling.source_materials(tmp_path, [anchor])
    material = materials["fixture::tiny"]
    probes = [{"key": key, "sha256": hashlib.sha256((raw / key).read_bytes()).hexdigest(),
               "bytes": len((raw / key).read_bytes())} for key in keys]
    payload = dict(dataset_id="fixture", subset_id="tiny", sample_keys=keys[:64], fingerprints=probes)
    expected_seed = int(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16], 16) % (2**63 - 1)
    assert material["seed"] == expected_seed
    first = labeling.make_inputs(anchor, 2, material, 0)
    again = labeling.make_inputs(anchor, 2, material, 0)
    changed = labeling.make_inputs(anchor, 2, material, 1)
    torch.testing.assert_close(first[0], again[0], rtol=0, atol=0)
    assert not torch.equal(first[0], changed[0])
    (raw / keys[0]).write_bytes(b"changed")
    assert labeling.source_materials(tmp_path, [anchor]) != materials


def test_missing_original_source_data_fails(tmp_path, anchor):
    with pytest.raises(FileNotFoundError, match="original source data"):
        labeling.source_materials(tmp_path, [anchor])
    with pytest.raises(SystemExit):
        labeling.main(["label", "--output", str(tmp_path), "--source-data-root", str(tmp_path / "missing")])
    assert not (tmp_path / "execution.json").exists()


def test_source_adapter_csv_and_nested_zip(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as bundle:
        bundle.writestr("rows.csv", "a,b\none,two\nthree,four\n")
    assert labeling.sample_bytes(tmp_path, "data.zip::rows.csv:row:1") == b"three\x1ffour"
    assert labeling.sample_bytes(tmp_path, "node:42") == b"node:42"
    with pytest.raises(ValueError, match="escapes"):
        labeling.sample_bytes(tmp_path, "../secret")


@pytest.mark.parametrize("batch,accumulation", [(1, 1), (8, 1), (128, 1), (2, 4)])
def test_capture_reflects_actual_batch_and_accumulation(anchor, batch, accumulation):
    material = {"seed": 42, "sample_keys": [str(i) for i in range(8)]}
    row = labeling.configuration(anchor, batch, "fp32_ieee", accumulation=accumulation)
    design, audit = labeling.capture_workload(anchor, row, material)
    verifier.verify_graph(design, row)
    assert audit["backward_capture"] == "strict"
    assert design["TRAINING_CONFIG"]["microbatch_size"] == batch
    assert design["TRAINING_CONFIG"]["gradient_accumulation_steps"] == accumulation
    from perfseer_v31.features import build_features
    features = build_features(design)
    features.validate()
    altered = copy.deepcopy(row)
    altered["training"]["microbatch_size"] *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        verifier.verify_graph(design, altered)


def test_inference_before_optimizer_and_training(anchor, monkeypatch):
    events = []

    class Model(torch.nn.Linear):
        def forward(self, x):
            events.append(("forward", self.training, torch.is_grad_enabled()))
            return super().forward(x)

    model = Model(4, 2)
    original = labeling.make_optimizer

    def optimizer(*args):
        events.append(("optimizer",))
        return original(*args)

    def measure(name, fn, *args, **kwargs):
        events.append((name,))
        fn()
        kwargs["evidence"]["status"] = "ok"

    monkeypatch.setattr(labeling, "make_optimizer", optimizer)
    material = {"seed": 42, "sample_keys": [str(i) for i in range(8)]}
    row = labeling.configuration(anchor, 2, "fp32_ieee", accumulation=4)
    phases = {}
    labeling.profile_phases(model, anchor, row, material, None, [], phases, device="cpu", measure=measure)
    assert events[0] == ("infer",)
    assert events[1] == ("forward", False, False)
    assert events[2] == ("optimizer",)
    assert events[3] == ("train",)
    assert events[4:] == [("forward", True, True)] * 4
    assert set(phases) == {"infer", "train"}


def phases_fixture():
    def phase(name, walls, gpus, steps, samples):
        blocks, offset = [], 0.
        for wall, gpu in zip(walls, gpus):
            blocks.append(dict(start_ms=offset, end_ms=offset + wall, wall_ms=wall,
                               gpu_ms=gpu, steps=steps, peak_torch_allocated_mib=100.))
            offset += wall
        return dict(phase=name, status="ok", warmup_steps=10 if name == "infer" else steps,
                    blocks=blocks, samples=samples, peak_torch_allocated_mib=100.)
    return {
        "train": phase("train", [2000., 4000.], [1000., 3000.], 4,
                       [{"t_ms": 10., "sm_percent": 20., "vram_mib": 200.},
                        {"t_ms": 3000., "sm_percent": 60., "vram_mib": 400.}]),
        "infer": phase("infer", [200.] * 30, [100.] * 30, 1,
                       [{"t_ms": 10., "sm_percent": 10., "vram_mib": 120.},
                        {"t_ms": 3000., "sm_percent": 30., "vram_mib": 160.}]),
    }


def test_all_twelve_targets_reconstruct_from_independent_expected_values():
    phases = phases_fixture()
    expected = dict(zip(labeling.TARGET_NAMES, [750., 500., 3000., 40., 300., 400., 100.,
                                               200., 100., 20., 140., 160.]))
    expected["train_epoch_ms_step_extrapolated"] = 3000.
    assert labeling.target_values(phases) == expected
    assert verifier.reconstruct_targets(phases) == expected


def test_precision_controls_do_not_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    objects = [torch.backends, torch.backends.cuda.matmul, torch.backends.cudnn,
               torch.backends.cudnn.conv, torch.backends.cudnn.rnn]
    old = [obj.fp32_precision for obj in objects]
    try:
        for precision in labeling.PRECISIONS:
            values = labeling.precision_controls(precision)
            assert set(values.values()) == {"tf32" if precision == "tf32" else "ieee"}
    finally:
        for obj, value in zip(objects, old):
            obj.fp32_precision = value


@pytest.mark.parametrize("phase_name", ["infer", "train"])
def test_phase_timer_measures_full_blocks_and_excludes_warmup(monkeypatch, phase_name):
    from types import SimpleNamespace
    clock = iter(index * .25 for index in range(10000))
    monkeypatch.setattr(labeling.time, "perf_counter", lambda: next(clock))
    observed = dict(name="NVIDIA GeForce RTX 5090", memory_total_bytes=32 * 1024**3,
                    compute_pids=[7], sm_percent=20., memory_used_mib=120.)
    nvml = SimpleNamespace(read=lambda: observed)
    events, calls = [], []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, end):
            return 100.

    def sampler(*args):
        return SimpleNamespace(samples=[{"t_ms": 1., "sm_percent": 20., "vram_mib": 120.}],
                               check=lambda: None, stop_event=SimpleNamespace(set=lambda: None),
                               thread=SimpleNamespace(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))

    monkeypatch.setattr(labeling, "Sampler", sampler)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append(len(calls)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100 * 1024**2)

    def step():
        calls.append(1)
        return torch.tensor(1.)

    phase = labeling.measure_phase(phase_name, step, nvml, [7], steps_per_epoch=4)
    warmup = 10 if phase_name == "infer" else 4
    assert events == [warmup]
    assert len(calls) == warmup + sum(b["steps"] for b in phase["blocks"])
    assert all(b["steps"] == (1 if phase_name == "infer" else 4) for b in phase["blocks"])
    assert sum(b["wall_ms"] for b in phase["blocks"]) >= 5000
    assert phase["status"] == "ok" and phase["peak_torch_allocated_mib"] == 100


def test_attempt_rejects_missing_phase_target_and_tampered_reduction(tmp_path, anchor):
    row = labeling.configuration(anchor, 2, "fp32_ieee")
    manifest, execution = {"fingerprint": "campaign"}, {"fingerprint": "execution"}
    material = {"seed": 42, "sample_keys": [str(i) for i in range(8)]}
    hardware = {"static": {"memory_bytes": 32 * 1024**3, "sm_count": 170, "compute_capability": 12.},
                "environment": {}}
    design, audit = labeling.capture_workload(anchor, row, material, hardware)
    path = tmp_path / "graph.json.gz"
    atomic_write(path, design, compress=True)
    from perfseer_v31.io import file_sha256
    result = dict(status="ok", version=labeling.VERSION, configuration_id=row["configuration_id"],
                  campaign_fingerprint="campaign", execution_fingerprint="execution",
                  target_names=list(labeling.TARGET_NAMES), phase_order=["infer", "train"], phases=phases_fixture(),
                  graph_path=path.name, graph_sha256=file_sha256(path), hardware=hardware)
    result["targets"] = labeling.target_values(result["phases"])
    verifier.verify_attempt(result, row, manifest, execution, tmp_path)
    changed = copy.deepcopy(result)
    changed["targets"]["infer_avg_vram_mib"] += 1
    with pytest.raises(ValueError, match="reconstruction"):
        verifier.verify_attempt(changed, row, manifest, execution, tmp_path)
    changed = copy.deepcopy(result)
    del changed["phases"]["infer"]
    with pytest.raises(ValueError, match="incomplete measurement phases"):
        verifier.verify_attempt(changed, row, manifest, execution, tmp_path)
    changed = copy.deepcopy(result)
    del changed["targets"]["train_avg_vram_mib"]
    with pytest.raises(ValueError, match="twelve-target"):
        verifier.verify_attempt(changed, row, manifest, execution, tmp_path)


def test_gpu_contention_and_wrong_device_rejected_without_cuda():
    observed = dict(name="NVIDIA GeForce RTX 5090", memory_total_bytes=32 * 1024**3,
                    compute_pids=[], sm_percent=0., memory_used_mib=100.)
    labeling.check_gpu(observed)
    with pytest.raises(labeling.ContentionError):
        labeling.check_gpu({**observed, "compute_pids": [42]})
    labeling.check_gpu({**observed, "compute_pids": [42]}, allowed_pids=[42])
    with pytest.raises(labeling.ContentionError):
        labeling.check_gpu({**observed, "compute_pids": [42, 43]}, allowed_pids=[42])
    with pytest.raises(ValueError, match="RTX 5090"):
        labeling.check_gpu({**observed, "name": "NVIDIA A10"})


def test_wsl_owner_registration_waits_and_rejects_foreign_owners(monkeypatch):
    base = dict(name="NVIDIA GeForce RTX 5090", memory_total_bytes=32 * 1024**3,
                sm_percent=0., memory_used_mib=100.)

    class Nvml:
        def __init__(self, pids):
            self.pids = iter(pids)

        def read(self):
            return {**base, "compute_pids": next(self.pids)}

    monkeypatch.setattr(labeling.time, "sleep", lambda _: None)
    assert labeling.wait_for_exclusive_owner(Nvml([[], [], [42]]), 42, timeout_seconds=1)["compute_pids"] == [42]
    assert labeling.wait_for_exclusive_owner(Nvml([[]]), 42, timeout_seconds=0)["compute_pids"] == []
    with pytest.raises(labeling.ContentionError, match="competing GPU"):
        labeling.wait_for_exclusive_owner(Nvml([[42, 43]]), 42, timeout_seconds=1)
    with pytest.raises(labeling.ContentionError, match="competing GPU"):
        labeling.wait_for_exclusive_owner(Nvml([[43]]), 42, timeout_seconds=1)


def test_persisted_graph_json_is_canonically_equal():
    # JSON persistence changes tuples to lists and TorchVersion to str.
    memory = {"nodes": ({"shape": (1, 2)},), "torch": torch.__version__}
    persisted = json.loads(json.dumps(memory))
    assert persisted != memory
    assert labeling.canonical_equal(persisted, memory)


def test_resume_requires_identical_execution(tmp_path):
    identity = dict(campaign_fingerprint="campaign", target_names=list(labeling.TARGET_NAMES), code="one")
    labeling.bind_execution(tmp_path, identity, resume=False)
    labeling.bind_execution(tmp_path, identity, resume=True)
    with pytest.raises(ValueError, match="use --resume"):
        labeling.bind_execution(tmp_path, identity, resume=False)
    with pytest.raises(ValueError, match="fingerprint differs"):
        labeling.bind_execution(tmp_path, {**identity, "code": "two"}, resume=True)


def test_exact_compatible_execution_migration_preserves_history(tmp_path, monkeypatch):
    old = dict(campaign_fingerprint="campaign", target_names=list(labeling.TARGET_NAMES),
               code_fingerprint="old-code", stable="same")
    previous = labeling.bind_execution(tmp_path, old, resume=False)
    monkeypatch.setattr(labeling, "COMPATIBLE_EXECUTION_CODE_MIGRATIONS", {"old-code": "tested-fix"})
    current = labeling.bind_execution(tmp_path, {**old, "code_fingerprint": "new-code"}, resume=True)
    assert current["execution_migration"] == "tested-fix"
    assert current["compatible_predecessor_execution_fingerprints"] == [previous["fingerprint"]]
    assert read_json(tmp_path / "execution_history" / f"{previous['fingerprint']}.json") == previous
    assert labeling.resolve_attempt_execution(tmp_path, current, previous["fingerprint"]) == previous
    assert labeling.bind_execution(tmp_path, {**old, "code_fingerprint": "new-code"}, resume=True) == current
    monkeypatch.setattr(labeling, "COMPATIBLE_EXECUTION_CODE_MIGRATIONS",
                        {"new-code": "second-tested-fix"})
    successor = labeling.bind_execution(tmp_path, {**old, "code_fingerprint": "newer-code"}, resume=True)
    assert successor["compatible_predecessor_execution_fingerprints"] == [
        current["fingerprint"], previous["fingerprint"]]
    assert read_json(tmp_path / "execution_history" / f"{current['fingerprint']}.json") == current
    with pytest.raises(ValueError, match="fingerprint differs"):
        labeling.bind_execution(tmp_path, {**old, "code_fingerprint": "newer-code", "stable": "changed"}, resume=True)


def test_fixed_repeat_failure_is_archived_for_retry(tmp_path):
    old_identity = dict(code_fingerprint="old")
    old = {**old_identity, "fingerprint": fingerprint(old_identity)}
    atomic_write(tmp_path / "execution_history" / f"{old['fingerprint']}.json", old)
    current = dict(fingerprint="new", compatible_predecessor_execution_fingerprints=[old["fingerprint"]])
    path = tmp_path / "attempt.json.gz"
    atomic_write(path, dict(status="failed", execution_fingerprint=old["fingerprint"],
                            error={"message": "recaptured graph changed"}), compress=True)
    assert labeling.resumable_attempt(path, {}, {}, current, tmp_path) == "pending"
    assert not path.exists() and len(list((tmp_path / "interrupted_attempts").glob("*.json.gz"))) == 1


def test_atomic_interruption_preserves_previous_artifact(tmp_path, monkeypatch):
    from perfseer_v31 import io as artifact_io
    path = tmp_path / "attempt.json.gz"
    atomic_write(path, {"status": "running"}, compress=True)

    def interrupted(*args):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(artifact_io.os, "replace", interrupted)
    with pytest.raises(OSError):
        atomic_write(path, {"status": "ok"}, compress=True)
    assert read_json(path) == {"status": "running"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["attempt.json.gz"]


def test_interrupted_attempt_is_preserved_and_can_resume(tmp_path):
    path = tmp_path / "attempt.json.gz"
    partial = dict(status="running", execution_fingerprint="execution", phases={"infer": {"status": "ok"}})
    atomic_write(path, partial, compress=True)
    assert labeling.resumable_attempt(path, {}, {}, {"fingerprint": "execution"}, tmp_path) == "pending"
    assert not path.exists()
    preserved = list((tmp_path / "interrupted_attempts").glob("*.json.gz"))
    assert len(preserved) == 1 and read_json(preserved[0]) == partial
    atomic_write(path, {**partial, "status": "oom"}, compress=True)
    assert labeling.resumable_attempt(path, {}, {}, {"fingerprint": "execution"}, tmp_path) == "failed"
    assert path.exists()


def test_incomplete_campaign_cannot_export_success(tmp_path, anchor):
    row = labeling.configuration(anchor, 2, "fp32_ieee")
    report = labeling.export_labels(tmp_path, {"planned_records": 10240}, [row], {"repeat_panel": []}, {"fingerprint": "execution"})
    assert report["status"] == "incomplete" and report["verified_records"] == 0
    assert read_json(tmp_path / "labels.json.gz") == []


def test_incomplete_24h_scope_uses_separate_artifacts(tmp_path, anchor):
    row = labeling.configuration(anchor, 8, "fp32_ieee")
    scope = dict(profile="24h", fingerprint="scope", configuration_ids=[row["configuration_id"]],
                 repeat_panel=[], planned_records=1, planned_attempts=1)
    report = labeling.export_labels(tmp_path, {"planned_records": 10240}, [row],
                                    {"repeat_panel": []}, {"fingerprint": "execution"}, scope=scope)
    assert report["status"] == "incomplete" and report["planned_records"] == 1
    assert report["campaign_scope_fingerprint"] == "scope"
    assert read_json(tmp_path / "labels-24h.json.gz") == []
    assert not (tmp_path / "labels.json.gz").exists()
