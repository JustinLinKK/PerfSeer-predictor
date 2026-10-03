"""Frozen-source, independent timing and memory residual calibration."""

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .calibration_contracts import MIB, TARGET_SCHEMA, audit_records, environment_identity, primary_prediction
from .capture import graphs_from_design
from .version import OUTPUT_CONTRACT_VERSION, TARGET_NAMES

VERSION = "perfseer_v32_convex_calibration_v1"
FEATURE_VERSION = "perfseer_v32_calibration_static_features_v1"
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10.)
FEATURE_NAMES = {
    "time_ms": ("intercept", "log_source_time", "log_flops", "log_operations", "log_tensor_bytes",
                "log_microbatch", "log_accumulation", "amp", "tf32", "compiled", "log_critical_path", "checkpointing"),
    "memory_bytes": ("intercept", "log_source_memory", "log_parameter_bytes", "log_optimizer_bytes",
                     "log_live_activation_bytes", "log_saved_bytes", "log_microbatch", "log_accumulation",
                     "amp", "checkpointing", "compiled", "alias_fraction"),
}


def source_identity(artifact):
    """Bind the actual weights and preprocessing, not a caller-supplied model name."""
    if tuple(artifact.get("target_names", ())) != TARGET_NAMES or artifact.get("output_contract") != OUTPUT_CONTRACT_VERSION:
        raise ValueError("calibration requires the native twelve-output source artifact")
    if artifact.get("normalization_sha256") != fingerprint(artifact["normalization"]):
        raise ValueError("source normalization hash differs")
    digest = hashlib.sha256()
    for name, tensor in sorted(artifact["model_state_dict"].items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(fingerprint([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    from perfseer_v31 import features_core
    from perfseer_v3 import features as graph_features
    files = [Path(__file__).with_name(name) for name in ("features.py", "model.py", "version.py")]
    files.extend([Path(features_core.__file__), Path(graph_features.__file__)])
    return {"weights_sha256": digest.hexdigest(), "normalization_sha256": artifact["normalization_sha256"],
            "model_config_sha256": fingerprint(artifact["model_config"]),
            "target_scales_sha256": fingerprint(torch.as_tensor(artifact["target_scales"]).tolist()),
            "preprocessing_sha256": fingerprint({str(index): file_sha256(path) for index, path in enumerate(files)}),
            "prediction_hardware": artifact["prediction_hardware"], "input_schema": artifact["input_schema"],
            "feature_version": artifact["feature_version"], "dataset_fingerprint": artifact["dataset_fingerprint"],
            "inference": {"device": "cpu", "amp": False, "microbatch": 1, "torch": torch.__version__}}


def feature_values(design, prediction):
    """Only static graph/configuration descriptors and frozen source predictions."""
    graph = graphs_from_design(design)["training"]
    global_features, training = graph.global_features, graph.training_config
    base = primary_prediction(prediction)
    precision = training["precision"]
    amp = precision in {"bf16_amp", "fp16_amp", "bf16", "fp16_grad_scaler", "mixed_structured"}
    compiled = training.get("backend", "unknown") in {"torch_compile", "torch_compile_inductor", "inductor", "inductor_cuda"}
    checkpointing = bool(training.get("checkpointing", training.get("activation_checkpointing", False)))
    def log(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid static calibration feature")
        return math.log1p(value)
    common = {"intercept": 1., "log_microbatch": log(training["microbatch_size"]),
              "log_accumulation": log(training["gradient_accumulation_steps"]),
              "amp": float(amp), "compiled": float(compiled), "checkpointing": float(checkpointing)}
    values = {
        "time_ms": {**common, "log_source_time": log(base["time_ms"]), "log_flops": log(global_features.total_flops),
                    "log_operations": log(global_features.operation_nodes),
                    "log_tensor_bytes": log(sum(edge.tensor_bytes or 0 for edge in graph.tensor_edges)),
                    "tf32": float(precision in {"tf32", "fp32_tf32"}),
                    "log_critical_path": log(global_features.critical_path_length)},
        "memory_bytes": {**common, "log_source_memory": log(base["memory_bytes"]),
                         "log_parameter_bytes": log(global_features.total_parameter_bytes),
                         "log_optimizer_bytes": log(global_features.total_optimizer_state_bytes),
                         "log_live_activation_bytes": log(global_features.peak_live_activation_bytes),
                         "log_saved_bytes": log(global_features.total_saved_for_backward_bytes),
                         "alias_fraction": sum(edge.is_view for edge in graph.tensor_edges) / max(1, len(graph.tensor_edges))},
    }
    return {target: {name: values[target][name] for name in names} for target, names in FEATURE_NAMES.items()}


def ridge(phi, residual, weights, regularization):
    phi, residual, weights = (np.asarray(value, dtype=np.float64) for value in (phi, residual, weights))
    if phi.ndim != 2 or not phi.shape[0] or not phi.shape[1] or residual.shape != (len(phi),) or weights.shape != residual.shape:
        raise ValueError("invalid ridge dimensions")
    if not all(np.isfinite(value).all() for value in (phi, residual, weights)) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("ridge requires finite data and nonnegative nonzero weights")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive, including for the intercept")
    weights = weights / weights.sum()
    # Augmented least squares avoids squaring the condition number; no matrix inverse.
    design = np.vstack((np.sqrt(weights[:, None]) * phi, math.sqrt(regularization) * np.eye(phi.shape[1])))
    labels = np.r_[np.sqrt(weights) * residual, np.zeros(phi.shape[1])]
    result = np.linalg.lstsq(design, labels, rcond=None)[0]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite ridge coefficients")
    return result


def group_weights(rows):
    counts = Counter(row["group_id"] for row in rows)
    return np.asarray([1. / (len(counts) * counts[row["group_id"]]) for row in rows], dtype=np.float64)


def _matrix(rows, target, names, method):
    if method == "affine":
        unit = MIB if target == "memory_bytes" else 1.
        return np.asarray([[1., primary_prediction(row["source_prediction"])[target] / unit] for row in rows], dtype=np.float64)
    result = np.asarray([[row["features"][target][name] for name in names] for row in rows], dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all() or not np.all(result[:, 0] == 1.):
        raise ValueError("invalid calibration features or intercept")
    return result


def _task_prediction(task, rows, target):
    base = np.asarray([primary_prediction(row["source_prediction"])[target] for row in rows])
    matrix = _matrix(rows, target, task["feature_names"], task["method"])
    phi = (matrix - np.asarray(task["mean"])) / np.asarray(task["scale"])
    correction = phi @ np.asarray(task["coefficients"])
    with np.errstate(over="raise", invalid="raise"):
        try:
            if task["method"] == "affine":
                raw = base + correction * (MIB if target == "memory_bytes" else 1.)
            elif target == "time_ms":
                raw = base * np.exp(correction)
            else:
                raw = base + np.maximum(base, 64 * MIB) * correction
        except FloatingPointError as error:
            raise ValueError("calibration prediction overflow; unsupported workload") from error
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite calibration prediction")
    if target == "time_ms" and (raw <= 0).any():
        raise ValueError("nonpositive calibrated timing; unsupported workload")
    return np.maximum(raw, 0.) if target == "memory_bytes" else raw, raw


def _selection_key(prediction, truth):
    positive = truth > 0
    if not positive.any():
        return (0., 0., float(np.mean(np.abs(prediction - truth))), 0.)
    relative = np.abs(prediction[positive] - truth[positive]) / truth[positive]
    return (-float(np.mean(relative <= .05)), -float(np.mean(relative <= .1)),
            float(np.median(relative)), float(np.quantile(relative, .95)))


def fit(rows, validation, *, method="linear", regularizations=LAMBDA_GRID, analytic=False):
    if method not in {"linear", "constant", "affine"} or (analytic and method != "linear"):
        raise ValueError("unknown calibration method")
    if not rows or not validation or any(row["split"] != "train" for row in rows) or any(row["split"] != "validation" for row in validation):
        raise ValueError("fit uses training rows and selection uses validation rows only")
    audit_records([*rows, *validation])
    rows, validation = ([row for row in items if not row.get("alias_of")] for items in (rows, validation))
    source, domain = rows[0]["source_identity"], rows[0]["domain_fingerprint"]
    if any(row["source_identity"] != source or row["domain_fingerprint"] != domain for row in [*rows, *validation]):
        raise ValueError("mixed source predictor or target environment")
    regularizations = tuple(regularizations)
    if not regularizations or any(not math.isfinite(value) or value <= 0 for value in regularizations):
        raise ValueError("positive regularization grid is required")
    tasks = {}
    for target in TARGET_SCHEMA:
        training_rows = [row for row in rows if row["measurements"][target]["valid"]]
        validation_rows = [row for row in validation if row["measurements"][target]["valid"]]
        if not training_rows or not validation_rows:
            raise ValueError(f"no valid fit or validation labels for {target}")
        groups = len({row["group_id"] for row in training_rows})
        names = list(FEATURE_NAMES[target][:min(len(FEATURE_NAMES[target]), max(1, groups // 2))])
        if method == "constant":
            names = ["intercept"]
        elif method == "affine":
            names = ["intercept", "source_prediction"]
        if analytic and target == "memory_bytes":
            if any(not row.get("memory_baseline", {}).get("available_before_execution") for row in [*training_rows, *validation_rows]):
                raise ValueError("analytic ablation requires a static baseline, not a query measurement")
            if len(names) > 1:
                names = names[:-1] + ["log_analytic_bytes"]
        matrix = _matrix(training_rows, target, names, method)
        weights = group_weights(training_rows)
        mean = np.sum(weights[:, None] * matrix, axis=0)
        scale = np.sqrt(np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0))
        mean[0], scale[0] = 0., 1.
        scale[scale < 1e-12] = 1.
        phi = (matrix - mean) / scale
        base = np.asarray([primary_prediction(row["source_prediction"])[target] for row in training_rows])
        truth = np.asarray([row["measurements"][target]["value"] for row in training_rows])
        if method == "affine":
            residual = (truth - base) / (MIB if target == "memory_bytes" else 1.)
        elif target == "time_ms":
            residual = np.log(truth) - np.log(base)
        else:
            residual = (truth - base) / np.maximum(base, 64 * MIB)
        validation_truth = np.asarray([row["measurements"][target]["value"] for row in validation_rows])
        candidates = []
        for lam in sorted(set(regularizations)):
            task = {"method": method, "feature_names": names, "mean": mean.tolist(), "scale": scale.tolist(),
                    "regularization": lam, "coefficients": ridge(phi, residual, weights, lam).tolist(),
                    "fit_ids": [row["sample_id"] for row in training_rows], "independent_groups": groups}
            try:
                prediction, _ = _task_prediction(task, validation_rows, target)
            except ValueError:
                continue
            candidates.append((_selection_key(prediction, validation_truth), lam, task))
        if not candidates:
            raise ValueError(f"all {target} candidates produced invalid validation predictions")
        tasks[target] = min(candidates, key=lambda item: (item[0], item[1]))[2]
    result = {"version": VERSION, "feature_version": FEATURE_VERSION, "target_schema": TARGET_SCHEMA,
              "source_identity": source, "domain_fingerprint": domain, "environment": rows[0]["environment"],
              "tasks": tasks, "analytic": analytic, "regularization_grid": list(regularizations),
              "fit_ids": [row["sample_id"] for row in rows], "validation_ids": [row["sample_id"] for row in validation],
              "fit_groups": sorted({row["group_id"] for row in rows}),
              "validation_groups": sorted({row["group_id"] for row in validation}),
              "fit_workloads": sorted({row["workload_id"] for row in rows}),
              "validation_workloads": sorted({row["workload_id"] for row in validation}),
              "selection": "per_task_hit5_hit10_median_p95_validation_only",
              "memory_scale": "max_frozen_source_memory_64MiB", "status": "experimental_not_promoted"}
    result["fingerprint"] = fingerprint(result)
    validate_adapter(result)
    return result


def validate_adapter(adapter):
    payload = {key: value for key, value in adapter.items() if key != "fingerprint"}
    if adapter.get("version") != VERSION or adapter.get("feature_version") != FEATURE_VERSION or adapter.get("fingerprint") != fingerprint(payload):
        raise ValueError("calibration artifact version or fingerprint differs")
    if adapter.get("target_schema") != TARGET_SCHEMA or set(adapter.get("tasks", {})) != set(TARGET_SCHEMA):
        raise ValueError("calibration target schema differs")
    if adapter["domain_fingerprint"] != environment_identity(adapter["environment"]):
        raise ValueError("calibration domain fingerprint differs")
    for target, task in adapter["tasks"].items():
        names = task["feature_names"]
        allowed = set(FEATURE_NAMES[target]) | ({"log_analytic_bytes"} if adapter["analytic"] and target == "memory_bytes" else set())
        if task["method"] == "affine":
            allowed = {"intercept", "source_prediction"}
        if task["method"] not in {"linear", "constant", "affine"} or not names or names[0] != "intercept" or len(set(names)) != len(names) or not set(names) <= allowed:
            raise ValueError("calibration feature order differs")
        if not math.isfinite(task["regularization"]) or task["regularization"] <= 0:
            raise ValueError("invalid artifact regularization")
        for key in ("mean", "scale", "coefficients"):
            values = np.asarray(task[key])
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("invalid calibration coefficient or scaler dimensions")
        if min(task["scale"]) <= 0 or task["mean"][0] != 0 or task["scale"][0] != 1:
            raise ValueError("invalid calibration scaler or intercept")


def save_adapter(path, adapter):
    validate_adapter(adapter)
    atomic_write(path, adapter)
    if read_json(path) != adapter:
        raise ValueError("adapter serialization verification failed")


def load_adapter(path):
    result = read_json(path)
    validate_adapter(result)
    return result


def apply_adapter(adapter, rows, *, source, environment, unknown_domain="reject"):
    validate_adapter(adapter)
    if unknown_domain not in {"reject", "source_fallback"}:
        raise ValueError("unknown domain routing policy")
    if source != adapter["source_identity"]:
        raise ValueError("adapter source checkpoint, preprocessing, or inference configuration differs")
    for row in rows:
        primary_prediction(row["source_prediction"])
    domain = environment_identity(environment)
    if domain != adapter["domain_fingerprint"]:
        if unknown_domain == "reject":
            raise ValueError("unsupported calibration environment")
        return {"predictions": [dict(row["source_prediction"]) for row in rows], "status": "source_fallback_unknown_domain",
                "adapted_targets": [], "unadapted_targets": list(TARGET_NAMES), "negative_raw_memory_count": 0}
    if any(row.get("source_identity", source) != source or row.get("domain_fingerprint", domain) != domain for row in rows):
        raise ValueError("query source or domain differs")
    predictions = [dict(row["source_prediction"]) for row in rows]
    negative = 0
    for target, task in adapter["tasks"].items():
        if not rows:
            continue
        values, raw = _task_prediction(task, rows, target)
        if target == "memory_bytes":
            negative = int(np.sum(raw < 0))
            values = values / MIB
        for prediction, value in zip(predictions, values, strict=True):
            prediction[TARGET_SCHEMA[target]["api_name"]] = float(value)
    adapted = [schema["api_name"] for schema in TARGET_SCHEMA.values()]
    return {"predictions": predictions, "status": "adapted", "adapted_targets": adapted,
            "unadapted_targets": [name for name in TARGET_NAMES if name not in adapted],
            "negative_raw_memory_count": negative}
