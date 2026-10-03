from copy import deepcopy

import numpy as np
import pytest
import torch

from perfseer_v31.io import fingerprint
from perfseer_v32.calibration import (FEATURE_NAMES, apply_adapter, fit, group_weights, load_adapter,
                                    ridge, save_adapter, validate_adapter)
from perfseer_v32.calibration_contracts import (CONTRACT_VERSION, ENVIRONMENT_VERSION, MIB, audit_records,
                                               environment_identity, measurement, schedule_duration, workload_identity)
from perfseer_v32.memory_baseline import TRACE_VERSION, process_peak, storage_peak
from perfseer_v32.version import TARGET_NAMES


def environment():
    return dict(version=ENVIRONMENT_VERSION, hardware_id="target", gpu_model="test", capacity_bytes=32 * 1024**3,
                partition="none", driver="test", cuda="test", framework="test", libraries={}, allocator="default",
                backend_policy="eager", host_context={"threads": 1})


def records(split, count=32, *, time_ratio=1.2, memory_delta=20 * MIB):
    result = []
    for index in range(count):
        key = f"{split}-{index}"
        base = dict(zip(TARGET_NAMES, [10. + index, 9., 120., 50., 200., 210. + index, 100., 5., 4., 50., 100., 110.]))
        protocol = dict(warmup="excluded", boundaries="window", step_unit="optimizer_update", data_loading="excluded", source="synthetic")
        observations = {target: measurement([value], [key], target=target, protocol=protocol) for target, value in
                        (("time_ms", base["train_step_wall_ms"] * time_ratio),
                         ("memory_bytes", base["train_peak_vram_mib"] * MIB + memory_delta))}
        features = {target: {name: (1. if name == "intercept" else np.log1p(index + 1)) for name in names}
                    for target, names in FEATURE_NAMES.items()}
        result.append(dict(version=CONTRACT_VERSION, sample_id=key, workload_id=key, group_id=key, anchor_id=key,
                           split=split, source_sha256="a" * 64, source_identity={"model": "test"}, environment=environment(),
                           domain_fingerprint=environment_identity(environment()), source_prediction=base,
                           measurements=observations, features=features, precision="fp32", batch=1, family="linear"))
    return result


def rehash(adapter):
    adapter["fingerprint"] = fingerprint({key: value for key, value in adapter.items() if key != "fingerprint"})
    return adapter


def apply(adapter, rows):
    return apply_adapter(adapter, rows, source={"model": "test"}, environment=environment())


def test_weighted_ridge_stationary_rank_deficient_and_input_validation():
    rng = np.random.default_rng(29)
    phi = np.c_[np.ones(12), np.ones(12), rng.normal(size=(12, 4))]
    residual, weights = rng.normal(size=12), rng.uniform(.1, 2., 12)
    weights /= weights.sum()
    lam = .01
    actual = ridge(phi, residual, weights, lam)
    a = phi.T @ (weights[:, None] * phi) + lam * np.eye(phi.shape[1])
    np.testing.assert_allclose(a @ actual, phi.T @ (weights * residual), atol=1e-12)
    assert np.linalg.eigvalsh(a).min() > 0
    np.testing.assert_array_equal(ridge(phi, np.zeros(12), weights, lam), np.zeros(6))
    for invalid in (0, -1, np.nan):
        with pytest.raises(ValueError, match="regularization"):
            ridge(phi, residual, weights, invalid)
    with pytest.raises(ValueError, match="finite"):
        ridge(phi, residual, -weights, lam)


def test_identity_independence_serialization_and_frozen_inputs(tmp_path):
    train, validation, query = records("train"), records("validation"), records("test")
    before = deepcopy((train, validation, query))
    adapter = fit(train, validation, regularizations=(.001,))
    zero = deepcopy(adapter)
    for task in zero["tasks"].values():
        task["coefficients"] = [0.] * len(task["coefficients"])
    rehash(zero)
    assert apply(zero, query)["predictions"] == [row["source_prediction"] for row in query]
    time = deepcopy(zero)
    time["tasks"]["time_ms"]["coefficients"][0] = np.log(1.2)
    rehash(time)
    assert apply(time, query)["predictions"][0]["train_peak_vram_mib"] == 210.
    assert apply(time, query)["predictions"][0]["train_step_wall_ms"] == pytest.approx(12.)
    memory = deepcopy(zero)
    memory["tasks"]["memory_bytes"]["coefficients"][0] = -2.
    rehash(memory)
    result = apply(memory, query)
    assert result["negative_raw_memory_count"] == len(query)
    assert result["predictions"][0]["train_peak_vram_mib"] == 0.
    assert result["predictions"][0]["train_step_wall_ms"] == 10.
    save_adapter(tmp_path / "adapter.json", adapter)
    assert apply(load_adapter(tmp_path / "adapter.json"), query) == apply(adapter, query)
    assert (train, validation, query) == before


def test_inference_matched_ratio_and_constant_baseline():
    train, validation = records("train"), records("validation")
    for row in train + validation:
        row["source_prediction"]["train_step_wall_ms"] = 110.
        item = row["measurements"]["time_ms"]
        row["measurements"]["time_ms"] = measurement([120.], item["run_ids"], target="time_ms", protocol=item["protocol"])
    adapter = fit(train, validation, method="constant", regularizations=(1e-10,))
    assert apply(adapter, validation)["predictions"][0]["train_step_wall_ms"] == pytest.approx(120., rel=1e-9)
    assert len(adapter["tasks"]["time_ms"]["coefficients"]) == 1


def test_actual_adapter_recovers_representable_heldout_residuals():
    rng = np.random.default_rng(29)
    train, validation, query = records("train", 64), records("validation", 16), records("test", 21)
    coefficients = {target: rng.normal(0, .02, len(names)) for target, names in FEATURE_NAMES.items()}
    for row in train + validation + query:
        for target, names in FEATURE_NAMES.items():
            values = np.r_[1., rng.uniform(0, 2, len(names) - 1)]
            row["features"][target] = dict(zip(names, values.tolist(), strict=True))
            base = row["source_prediction"]["train_step_wall_ms"] if target == "time_ms" else row["source_prediction"]["train_peak_vram_mib"] * MIB
            truth = base * np.exp(values @ coefficients[target]) if target == "time_ms" else base + max(base, 64 * MIB) * (values @ coefficients[target])
            item = row["measurements"][target]
            row["measurements"][target] = measurement([float(truth)], item["run_ids"], target=target, protocol=item["protocol"])
    adapter = fit(train, validation, regularizations=(1e-10,))
    actual = apply(adapter, query)["predictions"]
    for row, prediction in zip(query, actual, strict=True):
        assert prediction["train_step_wall_ms"] == pytest.approx(row["measurements"]["time_ms"]["value"], rel=1e-7)
        assert prediction["train_peak_vram_mib"] * MIB == pytest.approx(row["measurements"]["memory_bytes"]["value"], rel=1e-7)


def test_train_only_scaler_group_weights_and_task_validity():
    train, validation = records("train"), records("validation")
    for row in validation:
        row["features"]["time_ms"]["log_source_time"] += 1000
    train[1]["group_id"] = train[0]["group_id"]
    weights = group_weights(train)
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    target = train[0]["measurements"]["memory_bytes"]
    train[0]["measurements"]["memory_bytes"] = measurement([], [], target="memory_bytes", protocol=target["protocol"], reason="missing")
    adapter = fit(train, validation, regularizations=(.1,))
    assert train[0]["sample_id"] in adapter["tasks"]["time_ms"]["fit_ids"]
    assert train[0]["sample_id"] not in adapter["tasks"]["memory_bytes"]["fit_ids"]
    assert adapter["tasks"]["time_ms"]["mean"][1] < 10


def test_unknown_domain_and_wrong_source_fail_closed():
    adapter = fit(records("train"), records("validation"))
    changed = {**environment(), "driver": "new"}
    with pytest.raises(ValueError, match="environment"):
        apply_adapter(adapter, records("test"), source={"model": "test"}, environment=changed)
    result = apply_adapter(adapter, records("test"), source={"model": "test"}, environment=changed, unknown_domain="source_fallback")
    assert result["status"] == "source_fallback_unknown_domain" and not result["adapted_targets"]
    with pytest.raises(ValueError, match="checkpoint"):
        apply_adapter(adapter, records("test"), source={"model": "teacher"}, environment=environment())
    damaged = deepcopy(adapter)
    damaged["tasks"]["time_ms"]["coefficients"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_adapter(damaged)


@pytest.mark.parametrize("key", ["group_id", "anchor_id", "workload_id"])
def test_split_leakage_rejected(key):
    train, validation = records("train"), records("validation")
    validation[0][key] = train[0][key]
    with pytest.raises(ValueError, match="leakage"):
        fit(train, validation)
    with pytest.raises(ValueError, match="training rows"):
        fit(records("test"), records("validation"))


def test_native_measurement_aliases_and_configuration_identity():
    rows = records("train")
    alias = {**deepcopy(rows[0]), "sample_id": "alias", "alias_of": rows[0]["sample_id"]}
    report = audit_records([*rows, alias])
    assert report["aliases"] == 1 and report["real_gpu_runs"] == len(rows)
    alias["workload_id"] = "different"
    with pytest.raises(ValueError, match="copied"):
        audit_records([*rows, alias])
    training = dict(microbatch_size=2, gradient_accumulation_steps=1, precision="fp32", optimizer={"name": "adam"}, backend="eager")
    first = workload_identity("a" * 64, [{"shape": [2, 4], "dtype": "fp32"}], training, {"loss": "mse"})
    second = workload_identity("a" * 64, [{"shape": [4, 4], "dtype": "fp32"}], {**training, "microbatch_size": 4}, {"loss": "mse"})
    assert first != second


def test_repeat_mean_zero_memory_and_schedule():
    protocol = records("train", 1)[0]["measurements"]["time_ms"]["protocol"]
    result = measurement([1., 3.], ["a", "b"], target="time_ms", protocol=protocol)
    assert result["value"] == 2. and result["std"] == 1.
    assert measurement([0.], ["a"], target="memory_bytes", protocol=protocol)["valid"]
    assert not measurement([0.], ["a"], target="time_ms", protocol=protocol)["valid"]
    assert schedule_duration(2., dataset_size=128, microbatch=8, accumulation=4, epochs=3)["training_ms"] == 24.
    with pytest.raises(ValueError, match="variable final"):
        schedule_duration(2., dataset_size=129, microbatch=8)


def test_storage_alias_extents_lifetimes_temporaries_and_coincident_peak():
    backing = torch.empty(100, dtype=torch.float32)
    first, second = backing[:10], backing[20:30]
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    size = first.untyped_storage().nbytes()
    trace = dict(version=TRACE_VERSION, scope="allocated_storage", storages=[
        dict(storage_id="A", storage_bytes=size, lifetimes=[[-1, 2]]),
        dict(storage_id="A", storage_bytes=size, lifetimes=[[0, 2]]),
        dict(storage_id="gradient", storage_bytes=400, lifetimes=[[1, 2]]),
        dict(storage_id="adam", storage_bytes=800, lifetimes=[[2, 3]]),
        dict(storage_id="workspace", storage_bytes=80, lifetimes=[[2, 2]]),
    ])
    result = storage_peak(trace)
    assert result["peak_bytes"] == 1680 and result["unique_storages"] == 4
    assert process_peak([100, 200], [100, 0]) == 200
    trace["storages"][1]["storage_bytes"] = 40
    with pytest.raises(ValueError, match="extent"):
        storage_peak(trace)


def margins():
    return dict(time_mae_ms=.1, memory_mae_mib=8., relative_error=.01, near_zero_relative_floor=1e-6,
                memory_underprediction_mib=8., hit_rate=.02, warm_p95_fraction=.1, minimum_test_groups=5)


def test_metrics_mask_zero_targets_and_exact_relative_hits():
    from perfseer_v32.calibration_evaluation import metrics

    rows = records("test", 3, time_ratio=1., memory_delta=0.)
    predictions = [deepcopy(row["source_prediction"]) for row in rows]
    predictions[0]["train_step_wall_ms"] *= 1.05
    predictions[1]["train_step_wall_ms"] *= .9
    predictions[2]["train_step_wall_ms"] *= 1.11
    item = rows[0]["measurements"]["memory_bytes"]
    rows[0]["measurements"]["memory_bytes"] = measurement([0.], item["run_ids"], target="memory_bytes", protocol=item["protocol"])
    result = metrics(rows, predictions)
    assert result["time_ms"]["within_5pct"] == pytest.approx(1 / 3)
    assert result["time_ms"]["within_10pct"] == pytest.approx(2 / 3)
    assert result["memory_bytes"]["zero_rows"] == 1 and result["memory_bytes"]["relative_rows"] == 2
    assert result["memory_bytes"]["mae_mib"] == 70.


def test_fit_selection_does_not_use_target_values_or_duplicate_configurations():
    from perfseer_v32.calibration_evaluation import select_fit_rows

    rows = records("train", 40)
    chosen = [row["sample_id"] for row in select_fit_rows(rows, 16, 29)]
    for row in rows:
        row["measurements"]["time_ms"]["value"] = 99999.
    assert [row["sample_id"] for row in select_fit_rows(rows, 16, 29)] == chosen
    assert len(set(chosen)) == 16
    with pytest.raises(ValueError, match="budget"):
        select_fit_rows(rows, 41, 29)


def test_sealed_validation_selection_test_evaluation_and_inconclusive_promotion():
    from perfseer_v32.calibration_evaluation import evaluate_test, select_experiments

    train, validation, test = records("train", 40), records("validation", 8), records("test", 8)
    for row in train + validation + test:
        row["features"]["memory_bytes"]["log_analytic_bytes"] = 10.
        row["memory_baseline"] = {"available_before_execution": True}
    selection = select_experiments(train, validation, budgets=(32,), seeds=(29,), margins=margins())
    assert selection["test_labels_used"] is False
    trial = selection["trials"][0]
    assert all(model is None or model["fit_ids"] == trial["fit_ids"] for model in trial["models"].values())
    report = evaluate_test(selection, test, draws=100)
    assert report["trials"][0]["promotion"]["status"] == "inconclusive"
    assert "equal_budget_current_transfer_comparison_missing" in report["trials"][0]["promotion"]["reasons"]
    assert report["trials"][0]["candidate_intervals"]["independent_groups"] == 8
    changed = deepcopy(selection)
    changed["margins"]["relative_error"] = 100.
    with pytest.raises(ValueError, match="sealed"):
        evaluate_test(changed, test, draws=100)
    test[0]["group_id"] = trial["models"]["linear"]["fit_groups"][0]
    with pytest.raises(ValueError, match="leakage"):
        evaluate_test(selection, test, draws=100)


def test_promotion_requires_both_objectives_safety_and_evidence():
    from perfseer_v32.calibration_evaluation import promotion_gate

    intervals = {target: {"mae": [-.1, -.01], "median_relative_error": [-.02, -.01],
                          "p95_relative_error": [-.02, -.01], "within_5pct": [0., .02],
                          "within_10pct": [0., .02]} for target in ("time_ms", "memory_bytes")}
    intervals["memory_bytes"]["p95_underprediction_mib"] = [-1., 0.]
    evidence = dict(status="estimated", independent_groups=10, intervals=intervals)
    bench = dict(intended_host_verified=True, source_warm_p95_ms=1., candidate_warm_p95_ms=1., graph_extraction_ms=2.)
    assert promotion_gate(evidence, margins(), benchmark=bench, complete_comparisons=True)["status"] == "passed"
    intervals["memory_bytes"]["p95_underprediction_mib"] = [-1., 100.]
    assert promotion_gate(evidence, margins(), benchmark=bench, complete_comparisons=True)["status"] == "inconclusive"


def test_real_predictor_opt_in_preserves_api_and_checkpoint(tmp_path):
    from dataclasses import asdict
    from perfseer_v3.op_registry import OperationRegistry
    from perfseer_v32.calibration import feature_values, source_identity
    from perfseer_v32.capture import capture
    from perfseer_v32.features import build_features
    from perfseer_v32.inference import export_model, predict, predict_calibrated
    from perfseer_v32.model import SeerNetV32, SeerNetV32Config
    from perfseer_v32.runner import fit_training_normalization
    from perfseer_v32.training import checkpoint_payload

    torch.set_num_threads(1)
    torch.manual_seed(29)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="calibration")
    target = torch.tensor(list(records("train", 1)[0]["source_prediction"].values()))
    sample = {"features": build_features(design), "target": target, "sample_id": "one"}
    normalization = fit_training_normalization([sample], "test")
    config = SeerNetV32Config.from_registry(OperationRegistry.load(), sample["features"].layout,
                                          hidden=16, num_blocks=1, dropout=0., pooling_mode="phase_aware")
    model = SeerNetV32(config, target, target).eval()
    checkpoint = checkpoint_payload(model, role="student", epoch=1, normalization=asdict(normalization), dataset_fingerprint="test")
    export_model(checkpoint, tmp_path / "source.pt")
    artifact = torch.load(tmp_path / "source.pt", weights_only=False)
    original = predict(artifact, [design], amp=False, microbatch=1)
    identity = source_identity(artifact)
    train, validation = records("train"), records("validation")
    for row in train + validation:
        row["source_identity"] = identity
        row["source_prediction"] = original[0]
        row["features"] = feature_values(design, original[0])
    adapter = fit(train, validation, method="constant")
    for task in adapter["tasks"].values():
        task["coefficients"] = [0.] * len(task["coefficients"])
    rehash(adapter)
    assert predict(artifact, [design], calibration=adapter, environment=environment()) == original
    assert source_identity(artifact) == identity
    structured = predict_calibrated(artifact, [design], adapter, environment())
    assert len(structured["unadapted_targets"]) == 10 and tuple(structured["predictions"][0]) == TARGET_NAMES
    with pytest.raises(ValueError, match="configuration"):
        predict(artifact, [design], calibration=adapter, environment=environment(), amp=True)
    changed = deepcopy(artifact)
    next(iter(changed["model_state_dict"].values())).add_(1)
    with pytest.raises(ValueError, match="checkpoint"):
        predict(changed, [design], calibration=adapter, environment=environment())


def test_external_transfer_requires_equal_budget_and_frozen_provenance():
    from perfseer_v32.calibration_evaluation import _external_trial

    rows = records("validation", 3)
    trial = dict(budget=2, seed=29, fit_ids=["train-1", "train-2"])
    external = {**trial, "preprocessing_fit_ids": trial["fit_ids"], "distillation_fit_ids": [],
                "source_identity": rows[0]["source_identity"], "domain_fingerprint": rows[0]["domain_fingerprint"],
                "checkpoint_sha256": "a" * 64,
                "validation_predictions": {row["sample_id"]: row["source_prediction"] for row in rows}}
    predictions, binding = _external_trial([external], trial, rows, "validation")
    assert len(predictions) == 3 and len(binding) == 64
    changed = {**external, "fit_ids": ["train-1", "train-3"]}
    with pytest.raises(ValueError, match="label budget"):
        _external_trial([changed], trial, rows, "validation")
    changed = {**external, "preprocessing_fit_ids": ["validation-0"]}
    with pytest.raises(ValueError, match="non-fit"):
        _external_trial([changed], trial, rows, "validation")


def test_group_sampling_covers_groups_before_reusing_precision_strata():
    from perfseer_v32.calibration_evaluation import select_fit_rows

    rows = records("train", 32)
    for index, row in enumerate(rows):
        row["group_id"] = f"group-{index // 4}"
        row["precision"] = str(index % 4)
    selected = select_fit_rows(rows, 8, 29)
    assert len({row["group_id"] for row in selected}) == 8


def test_package_rejects_inconclusive_or_tampered_test_evidence(tmp_path):
    from perfseer_v31.io import atomic_write
    from perfseer_v32.calibration_evaluation import evaluate_test, select_experiments
    from perfseer_v32.calibration_runner import package

    train, validation, test = records("train", 8), records("validation", 3), records("test", 3)
    for row in train + validation + test:
        row["features"]["memory_bytes"]["log_analytic_bytes"] = 10.
        row["memory_baseline"] = {"available_before_execution": True}
    selection = select_experiments(train, validation, budgets=(8,), seeds=(29,), margins=margins())
    report = evaluate_test(selection, test, draws=100)
    atomic_write(tmp_path / "selection.json", selection)
    atomic_write(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="passed promotion gate"):
        package("unused.pt", tmp_path / "selection.json", tmp_path / "report.json", tmp_path / "deployment", budget=8, seed=29)
    report["trials"][0]["promotion"]["status"] = "passed"
    atomic_write(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="fingerprint"):
        package("unused.pt", tmp_path / "selection.json", tmp_path / "report.json", tmp_path / "deployment", budget=8, seed=29)
    assert not (tmp_path / "deployment").exists()
