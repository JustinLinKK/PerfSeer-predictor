from copy import deepcopy

import numpy as np
import pytest

from perfseer_v31.io import fingerprint
from perfseer_v32.calibration_contracts import ENVIRONMENT_FIELDS, ENVIRONMENT_VERSION, MIB, environment_identity
from perfseer_v4 import residuals
from perfseer_v4.resource_model import RESOURCE_NAMES
from perfseer_v4.version import TARGET_NAMES


def environment():
    return {"version": ENVIRONMENT_VERSION, **{name: "unknown" for name in ENVIRONMENT_FIELDS},
            "hardware_id": "target", "gpu_model": "target GPU", "capacity_bytes": 24 * 1024 ** 3}


def rows(split, count=8, variant="v4.0"):
    env = environment()
    result = []
    for index in range(count):
        base = dict(zip(TARGET_NAMES, [10. + index, 8. + index, 100. + index, 30. + index,
                                      80. + index, 100. + index, 70. + index]))
        truth = {name: value * (1.1 if i < 3 else 1.05) if i != 3 else value + 2.
                 for i, (name, value) in enumerate(base.items())}
        features = {name: (0. if name in {"epoch_length_unknown", "checkpointing"} else np.log1p(index + i + 1.))
                    for i, name in enumerate(RESOURCE_NAMES)}
        result.append({"sample_id": f"{split}-{index}", "group_id": f"{split}-group-{index}",
                       "workload_id": f"{split}-workload-{index}", "split": split,
                       "source_prediction": base, "targets": truth, "features": features,
                       "source_identity": {"model_variant": variant, "weights_sha256": "a" * 64,
                                           "normalization_sha256": "b" * 64},
                       "environment": env.copy(), "domain_fingerprint": environment_identity(env)})
    return result


@pytest.mark.parametrize("target", TARGET_NAMES)
def test_residual_transform_and_inverse_match_independent_physical_equations(target):
    base = np.asarray([.5, 80.]) if target != TARGET_NAMES[3] else np.asarray([0., 80.])
    truth = np.asarray([.75, 85.])
    kind = TARGET_NAMES.index(target)
    expected = (np.log(truth / base) if kind < 3 else (truth - base) / 100. if kind == 3 else
                (truth * MIB - base * MIB) / np.maximum(base * MIB, 64 * MIB))
    actual = residuals.residual_values(base, truth, target)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    prediction, raw = residuals.corrected_values(base, actual, target)
    np.testing.assert_allclose(prediction, truth, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(raw, truth, rtol=1e-13, atol=1e-15)


def test_physical_clamping_and_overflow_rejection():
    prediction, raw = residuals.corrected_values([1., 100.], [-2., 0.], TARGET_NAMES[5])
    np.testing.assert_array_equal(prediction, [0., 100.])
    np.testing.assert_array_equal(raw, [-127., 100.])
    prediction, raw = residuals.corrected_values([0., 90.], [-.1, .2], TARGET_NAMES[3])
    np.testing.assert_array_equal(prediction, [0., 100.])
    np.testing.assert_array_equal(raw, [-10., 110.])
    with pytest.raises(ValueError, match="overflow"):
        residuals.corrected_values([1.], [1000.], TARGET_NAMES[0])
    with pytest.raises(ValueError, match="physical"):
        residuals.residual_values([0.], [1.], TARGET_NAMES[0])


def test_ridge_stationarity_and_gp_posterior_independently():
    phi = np.asarray([[1., 2., 2.], [1., 3., 3.], [1., 3., 3.]])
    truth, weights, lam = np.asarray([.2, -.4, .8]), np.asarray([1., 2., 3.]), .03
    normalized = weights / weights.sum()
    coefficient = residuals.ridge(phi, truth, weights, lam)
    np.testing.assert_allclose((phi.T @ (normalized[:, None] * phi) + lam * np.eye(3)) @ coefficient,
                               phi.T @ (normalized * truth), rtol=1e-12, atol=1e-12)
    task = residuals.gp_fit(phi, truth, weights, lam, .5)
    query = np.asarray([[1., 2.5, 2.5], [1., 3., 3.]])
    kernel = np.exp(-np.square(phi[:, None] - phi[None, :]).sum(-1) / (2 * .5 ** 2))
    covariance = kernel + np.diag(lam / normalized + 1e-8)
    cross = np.exp(-np.square(query[:, None] - phi[None, :]).sum(-1) / (2 * .5 ** 2))
    expected_mean = cross @ np.linalg.solve(covariance, truth)
    expected_variance = 1. - np.einsum("ij,ji->i", cross, np.linalg.solve(covariance, cross.T))
    mean, variance = residuals.gp_predict(task, query)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-12, atol=1e-12)
    assert np.all(variance >= 0) and np.all(variance <= 1)
    with pytest.raises(ValueError, match="positive"):
        residuals.gp_fit(phi, truth, [1., 0., 2.], lam, 1.)


@pytest.mark.parametrize("variant,method", [("v4.1", "constant"), ("v4.1", "affine"), ("v4.1", "linear"), ("v4.3", "gp")])
def test_fit_predict_serialization_and_frozen_source_preservation(tmp_path, variant, method):
    source_variant = "v4.3" if variant == "v4.3" else "v4.0"
    fit_rows, validation = rows("train", variant=source_variant), rows("validation", 3, source_variant)
    before = deepcopy([fit_rows, validation])
    adapter = residuals.fit(fit_rows, validation, variant=variant, method=method,
                            regularizations=(1e-4, .01), length_scales=(.5, 1.))
    query = rows("test", 3, source_variant)
    source, env = fit_rows[0]["source_identity"], fit_rows[0]["environment"]
    prediction = residuals.apply_adapter(adapter, query, source, env)
    assert prediction["status"] == "adapted" and prediction["adapted_targets"] == list(TARGET_NAMES)
    assert len(prediction["predictions"]) == 3 and set(prediction["predictions"][0]) == set(TARGET_NAMES)
    assert before == [fit_rows, validation]
    assert adapter["status"] == "experimental_not_promoted"
    path = tmp_path / "adapter.json"
    residuals.save_adapter(path, adapter)
    restored = residuals.load_adapter(path)
    assert residuals.apply_adapter(restored, query, source, env) == prediction
    if method == "gp":
        assert prediction["uncertainty"]["kind"] == "latent_residual_variance"
        assert prediction["uncertainty"]["timing_point_estimate"] == "posterior_median"
        assert not prediction["uncertainty"]["calibrated_deployment_interval"]
        assert all(0 <= value <= 1 for row in prediction["uncertainty"]["values"] for value in row.values())


def test_fit_only_scaling_group_balance_epoch_features_and_per_target_base():
    fit_rows, validation = rows("train", 12), rows("validation", 3)
    fit_rows[-1]["group_id"] = fit_rows[-2]["group_id"]
    adapter = residuals.fit(fit_rows, validation, regularizations=(.01,))
    changed = deepcopy(validation)
    for row in changed:
        row["features"] = {name: value + 1000. for name, value in row["features"].items()}
    alternate = residuals.fit(fit_rows, changed, regularizations=(.01,))
    weights = residuals.group_weights(fit_rows)
    for target, task in adapter["tasks"].items():
        assert len(task["feature_names"]) <= 11 // 2
        assert task["mean"] == alternate["tasks"][target]["mean"]
        assert task["scale"] == alternate["tasks"][target]["scale"]
        unit = MIB if TARGET_NAMES.index(target) >= 4 else 1.
        expected = np.sum(weights * np.log1p([row["source_prediction"][target] * unit for row in fit_rows]))
        assert task["mean"][1] == pytest.approx(expected)
    assert adapter["tasks"]["train_epoch_ms"]["feature_names"][2:4] == ["log_steps_per_epoch", "epoch_length_unknown"]
    assert adapter["tasks"][TARGET_NAMES[3]]["feature_names"] == adapter["tasks"][TARGET_NAMES[0]]["feature_names"]


def test_null_labels_are_masked_per_target_and_require_fit_and_validation():
    fit_rows, validation = rows("train"), rows("validation", 3)
    fit_rows[0]["targets"][TARGET_NAMES[0]] = None
    adapter = residuals.fit(fit_rows, validation, regularizations=(.01,))
    assert fit_rows[0]["sample_id"] not in adapter["tasks"][TARGET_NAMES[0]]["fit_ids"]
    assert fit_rows[0]["sample_id"] in adapter["tasks"][TARGET_NAMES[1]]["fit_ids"]
    for row in validation:
        row["targets"][TARGET_NAMES[0]] = None
    with pytest.raises(ValueError, match="no valid"):
        residuals.fit(fit_rows, validation)


@pytest.mark.parametrize("corruption", ["group", "workload", "duplicate", "source", "environment", "domain", "variant"])
def test_fit_rejects_leakage_and_source_or_environment_mismatch(corruption):
    training, validation = rows("train"), rows("validation", 3)
    if corruption == "group":
        validation[0]["group_id"] = training[0]["group_id"]
    elif corruption == "workload":
        validation[0]["workload_id"] = training[0]["workload_id"]
    elif corruption == "duplicate":
        validation[0]["sample_id"] = training[0]["sample_id"]
    elif corruption == "source":
        validation[0]["source_identity"] = {**validation[0]["source_identity"], "weights_sha256": "c" * 64}
    elif corruption == "environment":
        validation[0]["environment"]["driver"] = "other"
    elif corruption == "domain":
        validation[0]["domain_fingerprint"] = "c" * 64
    else:
        training[0]["source_identity"]["model_variant"] = "v4.3"
    with pytest.raises(ValueError):
        residuals.fit(training, validation)


def test_fit_refuses_test_rows_before_reading_test_labels():
    class ForbiddenTargets(dict):
        def __iter__(self):
            raise AssertionError("test labels were accessed")

    training, validation = rows("train"), rows("validation", 3)
    forbidden = rows("test", 1)[0]
    forbidden["targets"] = ForbiddenTargets()
    with pytest.raises(ValueError, match="validation selection only"):
        residuals.fit([*training, forbidden], validation)


def test_source_and_environment_binding_unknown_fallback_and_tampering():
    training, validation = rows("train"), rows("validation", 3)
    adapter = residuals.fit(training, validation, regularizations=(.01,))
    source, env = training[0]["source_identity"], training[0]["environment"]
    query = [{"source_prediction": dict(training[0]["source_prediction"])}]
    other_environment = {**env, "driver": "new driver"}
    with pytest.raises(ValueError, match="environment"):
        residuals.apply_adapter(adapter, query, source, other_environment)
    result = residuals.apply_adapter(adapter, query, source, other_environment, unknown_domain="source_fallback")
    assert result["predictions"] == [query[0]["source_prediction"]] and result["status"] == "source_fallback_unknown_domain"
    with pytest.raises(ValueError, match="source checkpoint"):
        residuals.apply_adapter(adapter, query, {**source, "weights_sha256": "c" * 64}, other_environment,
                                unknown_domain="source_fallback")
    bad = deepcopy(adapter)
    bad["tasks"][TARGET_NAMES[0]]["coefficients"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        residuals.validate_adapter(bad)
    bad["tasks"][TARGET_NAMES[0]]["scale"][0] = 0.
    bad["fingerprint"] = fingerprint({k: v for k, v in bad.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="scaler"):
        residuals.validate_adapter(bad)


def test_gp_duplicate_constant_inputs_and_missing_targets_remain_finite():
    training, validation = rows("train", 4, "v4.3"), rows("validation", 2, "v4.3")
    for row in [*training, *validation]:
        row["features"] = {name: 0. for name in RESOURCE_NAMES}
    training[0]["targets"][TARGET_NAMES[4]] = None
    adapter = residuals.fit(training, validation, variant="v4.3", method="gp",
                            regularizations=(1e-4,), length_scales=(.5,))
    for task in adapter["tasks"].values():
        assert task["scale"] == [1.] * 17
        assert np.isfinite(task["alpha"]).all()
    result = residuals.apply_adapter(adapter, validation, training[0]["source_identity"], training[0]["environment"])
    assert all(np.isfinite(list(value.values())).all() for value in result["predictions"])


def test_memory_negative_raw_count_covers_all_memory_outputs():
    training, validation = rows("train", 4), rows("validation", 2)
    for row in [*training, *validation]:
        for target in TARGET_NAMES[4:]:
            row["source_prediction"][target], row["targets"][target] = 800., 1.
    adapter = residuals.fit(training, validation, method="constant", regularizations=(1e-4,))
    query = rows("test", 1)
    for target in TARGET_NAMES[4:]:
        query[0]["source_prediction"][target] = 1.
    result = residuals.apply_adapter(adapter, query, training[0]["source_identity"], training[0]["environment"])
    assert result["negative_raw_memory_count"] == 3
    assert all(result["predictions"][0][target] == 0 for target in TARGET_NAMES[4:])


def test_gp_kernel_excludes_source_predictions():
    training, validation = rows("train", 4, "v4.3"), rows("validation", 2, "v4.3")
    adapter = residuals.fit(training, validation, variant="v4.3", method="gp", regularizations=(.01,), length_scales=(1.,))
    query = rows("test", 1, "v4.3")
    source, env = training[0]["source_identity"], training[0]["environment"]
    before = residuals.apply_adapter(adapter, query, source, env)
    query[0]["source_prediction"][TARGET_NAMES[0]] *= 2
    after = residuals.apply_adapter(adapter, query, source, env)
    assert after["predictions"][0][TARGET_NAMES[0]] == pytest.approx(2 * before["predictions"][0][TARGET_NAMES[0]])
    assert after["uncertainty"] == before["uncertainty"]
