"""Independent seven-target ridge and Gaussian-process transfer corrections."""

import math

import numpy as np

from perfseer_v31.io import atomic_write, fingerprint, read_json
from perfseer_v32.calibration import FEATURE_NAMES, LAMBDA_GRID, group_weights, ridge
from perfseer_v32.calibration_contracts import MIB, environment_identity

from .version import OUTPUT_CONTRACT_VERSION, TARGET_NAMES

VERSION = "perfseer_v4_residual_adapter_v1"
LENGTH_SCALES = (.5, 1., 2., 4.)
JITTER = 1e-8


def _resource_names():
    from .resource_model import RESOURCE_NAMES
    return tuple(RESOURCE_NAMES)


def _kind(target):
    if target not in TARGET_NAMES:
        raise ValueError("unknown training target")
    return "time" if TARGET_NAMES.index(target) < 3 else "sm" if target == TARGET_NAMES[3] else "memory"


def _physical(values, target):
    result = np.asarray(values, dtype=np.float64)
    kind = _kind(target)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("expected finite physical target values")
    if ((result <= 0).any() if kind == "time" else (result < 0).any()) or (kind == "sm" and (result > 100).any()):
        raise ValueError("target outside physical range")
    return result * MIB if kind == "memory" else result


def residual_values(base, truth, target):
    base, truth = _physical(base, target), _physical(truth, target)
    if base.shape != truth.shape:
        raise ValueError("residual prediction and label shapes differ")
    kind = _kind(target)
    if kind == "time":
        return np.log(truth) - np.log(base)
    return (truth - base) / (np.maximum(base, 64 * MIB) if kind == "memory" else 100.)


def corrected_values(base, correction, target):
    base = _physical(base, target)
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != base.shape or not np.isfinite(correction).all():
        raise ValueError("invalid residual correction")
    kind = _kind(target)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            raw = base * np.exp(correction) if kind == "time" else base + correction * (
                np.maximum(base, 64 * MIB) if kind == "memory" else 100.)
        except FloatingPointError as error:
            raise ValueError("residual correction overflow") from error
    if not np.isfinite(raw).all() or (kind == "time" and (raw <= 0).any()):
        raise ValueError("nonfinite or nonpositive timing correction")
    if kind == "memory":
        return np.maximum(raw, 0.) / MIB, raw / MIB
    return (np.clip(raw, 0., 100.) if kind == "sm" else raw), raw


def _kernel(first, second, length_scale):
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("kernel length scale must be positive")
    squared = np.square((first[:, None, :] - second[None, :, :]) / length_scale).sum(-1)
    return np.exp(-.5 * squared)


def gp_fit(matrix, residual, weights, regularization, length_scale):
    matrix, residual, weights = (np.asarray(value, dtype=np.float64) for value in (matrix, residual, weights))
    if (matrix.ndim != 2 or not all(matrix.shape) or residual.shape != (len(matrix),) or weights.shape != residual.shape or
            not all(np.isfinite(value).all() for value in (matrix, residual, weights)) or (weights <= 0).any()):
        raise ValueError("GP requires finite matrices and strictly positive group weights")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("GP regularization must be positive")
    weights = weights / weights.sum()
    covariance = _kernel(matrix, matrix, length_scale) + np.diag(regularization / weights + JITTER)
    cholesky = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, residual))
    return {"fit_matrix": matrix.tolist(), "fit_weights": weights.tolist(), "alpha": alpha.tolist(),
            "cholesky": cholesky.tolist(), "regularization": float(regularization),
            "length_scale": float(length_scale), "jitter": JITTER}


def gp_predict(task, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    fit = np.asarray(task["fit_matrix"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != fit.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("invalid GP query features")
    cross = _kernel(matrix, fit, task["length_scale"])
    mean = cross @ np.asarray(task["alpha"], dtype=np.float64)
    solved = np.linalg.solve(np.asarray(task["cholesky"], dtype=np.float64), cross.T)
    variance = 1. - np.square(solved).sum(0)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < -1e-8).any():
        raise ValueError("invalid GP posterior")
    return mean, np.maximum(variance, 0.)


def _prediction(row):
    prediction = row["source_prediction"]
    if set(prediction) != set(TARGET_NAMES):
        raise ValueError("source prediction must explicitly name all seven training outputs")
    for target in TARGET_NAMES:
        _physical([prediction[target]], target)
    return prediction


def _features(row):
    names = _resource_names()
    if set(row["features"]) != set(names) or not np.isfinite(np.asarray([row["features"][name] for name in names], dtype=np.float64)).all():
        raise ValueError("expected the seventeen finite static resource descriptors")
    return row["features"]


def _audit_fit(rows, validation, variant):
    if not rows or not validation or any(row["split"] != "train" for row in rows) or any(row["split"] != "validation" for row in validation):
        raise ValueError("fit requires train rows and validation selection only")
    source, environment = rows[0]["source_identity"], rows[0]["environment"]
    if source.get("model_variant") != ("v4.3" if variant == "v4.3" else "v4.0"):
        raise ValueError("residual variant requires its matching frozen source model")
    domain = environment_identity(environment)
    ids, groups, workloads = set(), {}, {}
    for row in [*rows, *validation]:
        if (row["source_identity"] != source or row["environment"] != environment or
                row["domain_fingerprint"] != domain or environment_identity(row["environment"]) != domain):
            raise ValueError("mixed source predictor or target environment")
        if not row.get("sample_id") or not row.get("group_id") or row["sample_id"] in ids:
            raise ValueError("missing or duplicate sample identity")
        ids.add(row["sample_id"])
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("architecture group split leakage")
        workload = row.get("workload_id", row["sample_id"])
        if not workload or workloads.setdefault(workload, row["split"]) != row["split"]:
            raise ValueError("workload split leakage")
        _prediction(row)
        _features(row)
        if set(row["targets"]) != set(TARGET_NAMES):
            raise ValueError("labels require seven named values or explicit nulls")
        for target, value in row["targets"].items():
            if value is not None:
                _physical([value], target)
    return source, environment, domain


def _names(target, method, groups):
    if method == "gp":
        return list(_resource_names())
    if method == "constant":
        return ["intercept"]
    if method == "affine":
        return ["intercept", "source_prediction"]
    names = list(FEATURE_NAMES["memory_bytes" if _kind(target) == "memory" else "time_ms"])
    names[1] = "log_source_prediction"
    if target == "train_epoch_ms":
        names[2:2] = ["log_steps_per_epoch", "epoch_length_unknown"]
    return names[:min(len(names), max(1, groups // 2))]


def _matrix(rows, target, names):
    values = []
    for row in rows:
        base = float(_physical([row["source_prediction"][target]], target)[0])
        features = {**_features(row), "intercept": 1., "log_source_prediction": math.log1p(base),
                    "source_prediction": base / MIB if _kind(target) == "memory" else base}
        values.append([features[name] for name in names])
    return np.asarray(values, dtype=np.float64).reshape(len(rows), len(names))


def _task_prediction(task, rows, target):
    base = np.asarray([row["source_prediction"][target] for row in rows], dtype=np.float64)
    matrix = _matrix(rows, target, task["feature_names"])
    phi = (matrix - np.asarray(task["mean"])) / np.asarray(task["scale"])
    if task["method"] == "gp":
        correction, variance = gp_predict(task, phi)
    else:
        correction, variance = phi @ np.asarray(task["coefficients"], dtype=np.float64), None
    if task["method"] == "affine":
        kind = _kind(target)
        unit = MIB if kind == "memory" else 1.
        raw = (_physical(base, target) + unit * correction) / unit
        if not np.isfinite(raw).all() or (kind == "time" and (raw <= 0).any()):
            raise ValueError("nonfinite or nonpositive affine timing")
        prediction = np.maximum(raw, 0.) if kind == "memory" else np.clip(raw, 0., 100.) if kind == "sm" else raw
        return prediction, raw, variance
    prediction, raw = corrected_values(base, correction, target)
    return prediction, raw, variance


def _selection_key(prediction, truth):
    relative = np.abs(prediction - truth) / np.maximum(np.abs(truth), 1e-6)
    return (-int(np.sum(relative <= .05)), -int(np.sum(relative <= .1)),
            float(np.median(relative)), float(np.quantile(relative, .95)))


def fit(rows, validation, *, variant="v4.1", method="linear", regularizations=LAMBDA_GRID, length_scales=LENGTH_SCALES):
    if variant not in {"v4.1", "v4.3"} or method not in {"linear", "constant", "affine", "gp"} or (variant == "v4.3") != (method == "gp"):
        raise ValueError("incompatible residual variant and method")
    source, environment, domain = _audit_fit(rows, validation, variant)
    regularizations = sorted(set(regularizations))
    lengths = sorted(set(length_scales)) if method == "gp" else [1.]
    if not regularizations or not lengths or any(not math.isfinite(value) or value <= 0 for value in [*regularizations, *lengths]):
        raise ValueError("positive nonempty regularization and length-scale grids are required")
    tasks = {}
    for target in TARGET_NAMES:
        training_rows = [row for row in rows if row["targets"][target] is not None]
        validation_rows = [row for row in validation if row["targets"][target] is not None]
        if not training_rows or not validation_rows:
            raise ValueError(f"no valid fit or validation labels for {target}")
        groups = len({row["group_id"] for row in training_rows})
        names = _names(target, method, groups)
        matrix = _matrix(training_rows, target, names)
        weights = group_weights(training_rows)
        mean = np.sum(weights[:, None] * matrix, axis=0)
        scale = np.sqrt(np.sum(weights[:, None] * np.square(matrix - mean), axis=0))
        if method != "gp":
            mean[0], scale[0] = 0., 1.
        scale[scale < 1e-12] = 1.
        phi = (matrix - mean) / scale
        base = [row["source_prediction"][target] for row in training_rows]
        truth = [row["targets"][target] for row in training_rows]
        residual = ((_physical(truth, target) - _physical(base, target)) / (MIB if _kind(target) == "memory" else 1.)
                    if method == "affine" else residual_values(base, truth, target))
        validation_truth = np.asarray([row["targets"][target] for row in validation_rows], dtype=np.float64)
        candidates = []
        for length in lengths:
            for lam in regularizations:
                parameters = gp_fit(phi, residual, weights, lam, length) if method == "gp" else {
                    "regularization": lam, "coefficients": ridge(phi, residual, weights, lam).tolist()}
                task = {"method": method, "feature_names": names, "mean": mean.tolist(), "scale": scale.tolist(),
                        "fit_ids": [row["sample_id"] for row in training_rows], "independent_groups": groups, **parameters}
                try:
                    prediction, _, _ = _task_prediction(task, validation_rows, target)
                except ValueError:
                    continue
                candidates.append((_selection_key(prediction, validation_truth), lam, length, task))
        if not candidates:
            raise ValueError(f"all candidates produced invalid validation predictions for {target}")
        tasks[target] = min(candidates, key=lambda value: value[:3])[-1]
    result = {"version": VERSION, "model_variant": variant, "output_contract": OUTPUT_CONTRACT_VERSION,
              "target_names": list(TARGET_NAMES), "resource_names": list(_resource_names()), "method": method,
              "source_identity": source, "environment": environment, "domain_fingerprint": domain, "tasks": tasks,
              "fit_ids": [row["sample_id"] for row in rows], "validation_ids": [row["sample_id"] for row in validation],
              "fit_groups": sorted({row["group_id"] for row in rows}),
              "validation_groups": sorted({row["group_id"] for row in validation}),
              "fit_workloads": sorted({row.get("workload_id", row["sample_id"]) for row in rows}),
              "validation_workloads": sorted({row.get("workload_id", row["sample_id"]) for row in validation}),
              "regularization_grid": regularizations, "length_scale_grid": lengths if method == "gp" else [],
              "selection": "per_target_hit5_hit10_median_p95_validation_only", "status": "experimental_not_promoted",
              "memory_residual_scale": "max_actual_source_bytes_64MiB"}
    result["fingerprint"] = fingerprint(result)
    validate_adapter(result)
    return result


def validate_adapter(adapter):
    if adapter.get("version") != VERSION or adapter.get("fingerprint") != fingerprint({k: v for k, v in adapter.items() if k != "fingerprint"}):
        raise ValueError("residual artifact version or fingerprint differs")
    variant, method = adapter.get("model_variant"), adapter.get("method")
    if variant not in {"v4.1", "v4.3"} or method not in {"linear", "constant", "affine", "gp"} or (variant == "v4.3") != (method == "gp"):
        raise ValueError("incompatible residual artifact variant and method")
    if (adapter.get("output_contract") != OUTPUT_CONTRACT_VERSION or tuple(adapter.get("target_names", ())) != TARGET_NAMES or
            tuple(adapter.get("resource_names", ())) != _resource_names() or set(adapter.get("tasks", {})) != set(TARGET_NAMES)):
        raise ValueError("residual target or feature contract differs")
    if adapter["source_identity"].get("model_variant") != ("v4.3" if variant == "v4.3" else "v4.0"):
        raise ValueError("residual artifact source variant differs")
    if adapter["domain_fingerprint"] != environment_identity(adapter["environment"]):
        raise ValueError("residual environment fingerprint differs")
    for fit_key, validation_key in (("fit_ids", "validation_ids"), ("fit_groups", "validation_groups"), ("fit_workloads", "validation_workloads")):
        first, second = adapter[fit_key], adapter[validation_key]
        if not first or not second or len(set(first)) != len(first) or len(set(second)) != len(second) or set(first) & set(second):
            raise ValueError("residual artifact fit/validation identity leakage")
    for target, task in adapter["tasks"].items():
        groups = task["independent_groups"]
        if type(groups) is not int or not 1 <= groups <= len(task["fit_ids"]):
            raise ValueError("invalid residual independent-group count")
        names = _names(target, method, groups)
        if task["method"] != method or task["feature_names"] != names or not set(task["fit_ids"]) <= set(adapter["fit_ids"]):
            raise ValueError("residual task feature or fit identity differs")
        if not math.isfinite(task["regularization"]) or task["regularization"] <= 0 or task["regularization"] not in adapter["regularization_grid"]:
            raise ValueError("invalid residual regularization")
        for key in ("mean", "scale"):
            value = np.asarray(task[key], dtype=np.float64)
            if value.shape != (len(names),) or not np.isfinite(value).all():
                raise ValueError("invalid residual scaler")
        if min(task["scale"]) <= 0 or (method != "gp" and (task["mean"][0] != 0 or task["scale"][0] != 1)):
            raise ValueError("invalid residual scaler or intercept")
        if method == "gp":
            count = len(task["fit_ids"])
            for key, shape in (("fit_matrix", (count, len(names))), ("fit_weights", (count,)), ("alpha", (count,)), ("cholesky", (count, count))):
                value = np.asarray(task[key], dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("invalid GP posterior dimensions")
            weights, cholesky = np.asarray(task["fit_weights"]), np.asarray(task["cholesky"])
            if ((weights <= 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12) or
                    (np.diag(cholesky) <= 0).any() or np.any(np.triu(cholesky, 1)) or
                    task["jitter"] != JITTER or task["length_scale"] not in adapter["length_scale_grid"]):
                raise ValueError("invalid GP covariance contract")
        else:
            coefficients = np.asarray(task["coefficients"], dtype=np.float64)
            if coefficients.shape != (len(names),) or not np.isfinite(coefficients).all():
                raise ValueError("invalid residual coefficients")


def save_adapter(path, adapter):
    validate_adapter(adapter)
    atomic_write(path, adapter)
    if read_json(path) != adapter:
        raise ValueError("residual adapter serialization differs")


def load_adapter(path):
    adapter = read_json(path)
    validate_adapter(adapter)
    return adapter


def apply_adapter(adapter, rows, source, environment, unknown_domain="reject"):
    validate_adapter(adapter)
    if unknown_domain not in {"reject", "source_fallback"}:
        raise ValueError("unknown environment routing policy")
    if source != adapter["source_identity"]:
        raise ValueError("adapter source checkpoint, preprocessing or configuration differs")
    domain = environment_identity(environment)
    for row in rows:
        _prediction(row)
        if (row.get("source_identity", source) != source or row.get("domain_fingerprint", domain) != domain or
                ("environment" in row and environment_identity(row["environment"]) != domain)):
            raise ValueError("query source or domain differs")
    if domain != adapter["domain_fingerprint"]:
        if unknown_domain == "reject":
            raise ValueError("unsupported calibration environment")
        return {"predictions": [dict(row["source_prediction"]) for row in rows], "status": "source_fallback_unknown_domain",
                "adapted_targets": [], "unadapted_targets": list(TARGET_NAMES), "negative_raw_memory_count": 0}
    predictions = [dict(row["source_prediction"]) for row in rows]
    uncertainty = [dict() for _ in rows]
    negative = 0
    for target, task in adapter["tasks"].items():
        if not rows:
            continue
        values, raw, variance = _task_prediction(task, rows, target)
        if _kind(target) == "memory":
            negative += int(np.sum(raw < 0))
        for index, value in enumerate(values):
            predictions[index][target] = float(value)
            if variance is not None:
                uncertainty[index][target] = float(variance[index])
    result = {"predictions": predictions, "status": "adapted", "adapted_targets": list(TARGET_NAMES),
              "unadapted_targets": [], "negative_raw_memory_count": negative}
    if adapter["method"] == "gp":
        result["uncertainty"] = {"kind": "latent_residual_variance", "values": uncertainty,
                                 "timing_point_estimate": "posterior_median",
                                 "calibrated_deployment_interval": False}
    return result
