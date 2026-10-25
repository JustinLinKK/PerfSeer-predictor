"""Equal-budget selection and grouped, paired two-objective evaluation."""

from collections import defaultdict
import math

import numpy as np

from perfseer_v31.io import fingerprint

from .calibration import FEATURE_NAMES, apply_adapter, fit
from .calibration_contracts import MIB, TARGET_SCHEMA, audit_records, primary_prediction

REPORT_VERSION = "perfseer_v32_calibration_evaluation_v1"


def _summarize(truth, values, target):
    absolute = np.abs(values - truth)
    positive = truth > 0
    relative = absolute[positive] / truth[positive]
    result = {"valid_rows": len(truth), "relative_rows": int(positive.sum()), "zero_rows": int((~positive).sum()),
              "within_5pct": float(np.mean(relative <= .05)) if len(relative) else None,
              "within_10pct": float(np.mean(relative <= .1)) if len(relative) else None,
              "median_relative_error": float(np.median(relative)) if len(relative) else None,
              "p95_relative_error": float(np.quantile(relative, .95)) if len(relative) else None,
              "mae": float(np.mean(absolute)) if len(absolute) else None, "unit": TARGET_SCHEMA[target]["unit"]}
    if target == "memory_bytes":
        result.update(mae_mib=float(np.mean(absolute) / MIB) if len(absolute) else None,
                      p95_underprediction_mib=float(np.quantile(np.maximum(truth - values, 0), .95) / MIB) if len(truth) else None)
    return result


def metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("evaluation prediction count differs")
    result, primary = {}, [primary_prediction(prediction) for prediction in predictions]
    for target in TARGET_SCHEMA:
        valid = [index for index, row in enumerate(rows) if row["measurements"][target]["valid"] and not row.get("alias_of")]
        truth = np.asarray([rows[index]["measurements"][target]["value"] for index in valid])
        values = np.asarray([primary[index][target] for index in valid])
        result[target] = _summarize(truth, values, target)
    return result


def stratified_metrics(rows, predictions, *, draws=0, seed=29):
    result, intervals = {}, {}
    for field in ("family", "precision", "batch", "source_panel"):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row.get(field, "unknown"))].append(index)
        result[field] = {key: metrics([rows[i] for i in indices], [predictions[i] for i in indices])
                         for key, indices in sorted(groups.items())}
        if draws:
            for key, indices in groups.items():
                identity = tuple(indices)
                if identity not in intervals:
                    intervals[identity] = grouped_intervals([rows[i] for i in indices], [predictions[i] for i in indices],
                                                           draws=draws, seed=seed)
                result[field][key]["grouped_intervals"] = intervals[identity]
    return result


def select_fit_rows(rows, budget, seed):
    """Group/stratum coverage using only training inputs, never target magnitudes."""
    if type(budget) is not int or budget < 1 or any(row["split"] != "train" for row in rows):
        raise ValueError("positive fit budget and training rows required")
    rows = [row for row in rows if not row.get("alias_of") and any(value["valid"] for value in row["measurements"].values())]
    if budget > len(rows):
        raise ValueError("fit budget exceeds unique available configurations")
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group_id"], row.get("family", "unknown"), row.get("precision", "unknown"), row.get("batch", "unknown"))].append(row)
    # Within strata spread static feature vectors with deterministic farthest-first selection.
    names = FEATURE_NAMES["time_ms"]
    matrix = np.asarray([[row["features"]["time_ms"][name] for name in names] for row in rows])
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-12] = 1.
    vectors = {row["sample_id"]: matrix[index] / scale for index, row in enumerate(rows)}
    ordered = sorted(buckets, key=lambda key: fingerprint([seed, key]))
    selected, seen_groups = [], set()
    while len(selected) < budget:
        available = [key for key in ordered if buckets[key]]
        prioritize_groups = any(key[0] not in seen_groups for key in available)
        if prioritize_groups:
            available = [key for key in available if key[0] not in seen_groups]
        for key in available:
            if len(selected) == budget:
                break
            if prioritize_groups and key[0] in seen_groups:
                continue
            def order(row):
                distance = min((float(np.sum((vectors[row["sample_id"]] - vectors[item["sample_id"]]) ** 2))
                                for item in selected), default=0.)
                return (-distance, fingerprint([seed, row["sample_id"]]))
            chosen = min(buckets[key], key=order)
            selected.append(chosen)
            seen_groups.add(key[0])
            buckets[key].remove(chosen)
    return selected


def grouped_intervals(rows, predictions, *, baseline=None, seed=29, draws=1000):
    if draws < 100:
        raise ValueError("use at least 100 grouped bootstrap draws")
    if len(predictions) != len(rows) or (baseline is not None and len(baseline) != len(rows)):
        raise ValueError("paired baseline coverage differs")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row.get("alias_of"):
            groups[row["group_id"]].append(index)
    result = {"independent_groups": len(groups), "draws": draws, "seed": seed,
              "confidence": .95, "sign": "candidate_minus_baseline; negative error difference is better" if baseline is not None else "absolute metric"}
    if len(groups) < 2:
        return {**result, "status": "insufficient_independent_groups", "intervals": {}}
    grouped = [groups[key] for key in sorted(groups)]
    primary = [primary_prediction(prediction) for prediction in predictions]
    reference = [primary_prediction(prediction) for prediction in baseline] if baseline is not None else None
    data = {}
    for target in TARGET_SCHEMA:
        truth = np.asarray([row["measurements"][target]["value"] if row["measurements"][target]["valid"] else np.nan for row in rows])
        data[target] = (truth, np.asarray([row[target] for row in primary]),
                        np.asarray([row[target] for row in reference]) if reference is not None else None)
    rng, samples = np.random.default_rng(seed), defaultdict(list)
    keys = ("within_5pct", "within_10pct", "median_relative_error", "p95_relative_error", "mae", "p95_underprediction_mib")
    for _ in range(draws):
        indices = [i for selected in rng.integers(len(grouped), size=len(grouped)) for i in grouped[selected]]
        for target in TARGET_SCHEMA:
            truth, values, base = data[target]
            valid = np.asarray(indices)[np.isfinite(truth[indices])]
            candidate = _summarize(truth[valid], values[valid], target)
            reference_metrics = _summarize(truth[valid], base[valid], target) if base is not None else None
            for key in keys:
                value = candidate.get(key)
                other = reference_metrics.get(key) if reference_metrics is not None else 0.
                if value is not None and other is not None:
                    samples[(target, key)].append(value - other)
    intervals = {target: {} for target in TARGET_SCHEMA}
    for (target, key), values in samples.items():
        intervals[target][key] = [float(value) for value in np.quantile(values, [.025, .975])]
    return {**result, "status": "estimated", "intervals": intervals}


def validate_margins(margins):
    required = ("time_mae_ms", "memory_mae_mib", "relative_error", "near_zero_relative_floor",
                "memory_underprediction_mib", "hit_rate", "warm_p95_fraction", "minimum_test_groups")
    if any(key not in margins for key in required) or any(not math.isfinite(margins[key]) or margins[key] < 0 for key in required):
        raise ValueError("declare all nonnegative promotion margins before test evaluation")
    if margins["near_zero_relative_floor"] <= 0 or margins["hit_rate"] > 1 or type(margins["minimum_test_groups"]) is not int or margins["minimum_test_groups"] < 2:
        raise ValueError("invalid near-zero floor, hit-rate margin, or minimum groups")


def promotion_gate(intervals, margins, *, benchmark=None, complete_comparisons=False, memory_baseline_verified=True):
    validate_margins(margins)
    reasons, improvements = [], []
    if intervals["status"] != "estimated" or intervals["independent_groups"] < margins["minimum_test_groups"]:
        return {"status": "inconclusive", "reasons": ["insufficient_independent_test_groups"]}
    for target in TARGET_SCHEMA:
        values = intervals["intervals"][target]
        absolute_margin = margins["time_mae_ms"] if target == "time_ms" else margins["memory_mae_mib"] * MIB
        for key, margin in (("mae", absolute_margin), ("median_relative_error", margins["relative_error"]),
                            ("p95_relative_error", margins["relative_error"])):
            if key not in values or values[key][1] > margin:
                reasons.append(f"{target}:{key}_noninferiority_not_established")
        for key in ("within_5pct", "within_10pct"):
            if key not in values or values[key][0] < -margins["hit_rate"]:
                reasons.append(f"{target}:{key}_guard_not_established")
        improvements.append(values.get("mae", [0, 0])[1] < 0 and
                            values.get("median_relative_error", [0, 0])[1] < -margins["near_zero_relative_floor"])
    memory_interval = intervals["intervals"]["memory_bytes"].get("p95_underprediction_mib")
    if memory_interval is None or memory_interval[1] > margins["memory_underprediction_mib"]:
        reasons.append("memory_underprediction_guard_not_established")
    if not any(improvements):
        reasons.append("no_primary_objective_improvement_established")
    if not complete_comparisons:
        reasons.append("equal_budget_current_transfer_comparison_missing")
    if not memory_baseline_verified:
        reasons.append("analytic_memory_baseline_trace_gate_unverified")
    if benchmark is None or benchmark.get("intended_host_verified") is not True:
        reasons.append("intended_CPU_host_benchmark_missing")
    else:
        durations = [benchmark.get(key) for key in ("candidate_warm_p95_ms", "source_warm_p95_ms")]
        if any(value is None or not math.isfinite(value) or value <= 0 for value in durations):
            reasons.append("invalid_CPU_benchmark")
        elif durations[0] > durations[1] * (1 + margins["warm_p95_fraction"]):
            reasons.append("warm_latency_guard_failed")
        if benchmark.get("graph_extraction_ms") is None:
            reasons.append("graph_extraction_cost_unmeasured")
    return {"status": "inconclusive" if reasons else "passed", "reasons": reasons}


def _external_trial(trials, trial, rows, split):
    matches = [item for item in trials if (item["budget"], item["seed"]) == (trial["budget"], trial["seed"])]
    if len(matches) != 1:
        raise ValueError("current transfer must provide exactly one comparison per budget and seed")
    item = matches[0]
    if set(item["fit_ids"]) != set(trial["fit_ids"]) or len(item["fit_ids"]) != trial["budget"]:
        raise ValueError("current transfer comparison has a different label budget or fit IDs")
    permitted = set(item["fit_ids"])
    if any(not set(item[field]) <= permitted for field in ("preprocessing_fit_ids", "distillation_fit_ids")):
        raise ValueError("external preprocessing or distillation used non-fit rows")
    if any(row["source_identity"] != item["source_identity"] or row["domain_fingerprint"] != item["domain_fingerprint"] for row in rows):
        raise ValueError("external comparison source or environment differs")
    if not item.get("checkpoint_sha256") or len(item["checkpoint_sha256"]) != 64:
        raise ValueError("current transfer checkpoint hash is required")
    values = item[f"{split}_predictions"]
    if set(values) != {row["sample_id"] for row in rows}:
        raise ValueError("external comparison query coverage differs")
    predictions = [values[row["sample_id"]] for row in rows]
    for prediction in predictions:
        primary_prediction(prediction)
    binding = fingerprint({key: item[key] for key in ("budget", "seed", "fit_ids", "preprocessing_fit_ids",
                                                      "distillation_fit_ids", "source_identity", "domain_fingerprint", "checkpoint_sha256")})
    return predictions, binding


def select_experiments(training, validation, *, budgets=(32, 64, 128, 256), seeds=(11, 29, 47), margins,
                       current_transfer=None):
    """Return a sealed validation decision; this API never receives test rows."""
    validate_margins(margins)
    audit_records([*training, *validation])
    if any(row["split"] != "train" for row in training) or any(row["split"] != "validation" for row in validation):
        raise ValueError("experiment selection accepts only train and validation splits")
    validation = [row for row in validation if not row.get("alias_of")]
    trials = []
    for budget in budgets:
        for seed in seeds:
            selected = select_fit_rows(training, budget, seed)
            trial_identity = {"budget": budget, "seed": seed, "fit_ids": [row["sample_id"] for row in selected]}
            models, predictions = {"source": None}, {"source": [row["source_prediction"] for row in validation]}
            for method in ("constant", "affine", "linear", "linear_analytic"):
                adapter = fit(selected, validation, method="linear" if method == "linear_analytic" else method,
                              analytic=method == "linear_analytic")
                models[method] = adapter
                predictions[method] = apply_adapter(adapter, validation, source=adapter["source_identity"],
                                                    environment=adapter["environment"])["predictions"]
            binding = None
            if current_transfer is not None:
                predictions["current_transfer"], binding = _external_trial(current_transfer, trial_identity, validation, "validation")
            scores = {name: metrics(validation, values) for name, values in predictions.items()}
            # Choose a strongest control separately for each primary objective.
            # A joint average must not hide a failure in either target.
            def key(name, target):
                value = scores[name][target]
                if value["relative_rows"] == 0:
                    return (0., 0., value["mae"], value["mae"], name)
                return (-value["within_5pct"], -value["within_10pct"], value["median_relative_error"], value["p95_relative_error"], name)
            baseline_names = ["source", "constant", "affine"] + (["current_transfer"] if binding else [])
            controls = {target: min(baseline_names, key=lambda name: key(name, target)) for target in TARGET_SCHEMA}
            # Retain both candidates for validation review; freeze the selected one now.
            candidate = "linear_analytic" if all(key("linear_analytic", target)[:4] <= key("linear", target)[:4] for target in TARGET_SCHEMA) and any(
                key("linear_analytic", target)[:4] < key("linear", target)[:4] for target in TARGET_SCHEMA) else "linear"
            trials.append({"budget": budget, "seed": seed, "fit_ids": [row["sample_id"] for row in selected],
                           "independent_groups": len({row["group_id"] for row in selected}), "models": models,
                           "validation_metrics": scores, "comparison_baselines": controls, "candidate": candidate,
                           "current_transfer_binding": binding,
                           "fit_gpu_runs": len({run for row in selected for item in row["measurements"].values() for run in item["run_ids"]})})
    result = {"version": REPORT_VERSION, "stage": "validation_selection", "margins": margins, "trials": trials,
              "validation_ids": [row["sample_id"] for row in validation],
              "validation_gpu_runs": len({run for row in validation for item in row["measurements"].values() for run in item["run_ids"]}),
              "test_labels_used": False, "source_training_panel": "unknown_unless_provenance_declares_membership",
              "missing_comparisons": [] if current_transfer is not None else ["current_transfer_with_identical_fit_ids_and_budget"],
              "analytic_trace_gate": "unverified_static_graph_approximation", "source_retraining": "conditional_not_enabled"}
    result["fingerprint"] = fingerprint(result)
    return result


def evaluate_test(selection, rows, *, draws=1000, current_transfer=None, benchmarks=None):
    if selection.get("version") != REPORT_VERSION or selection.get("stage") != "validation_selection" or selection.get("fingerprint") != fingerprint({key: value for key, value in selection.items() if key != "fingerprint"}):
        raise ValueError("test requires the sealed validation selection")
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("final evaluation requires the untouched test split")
    audit_records(rows)
    rows = [row for row in rows if not row.get("alias_of")]
    reports = []
    for trial in selection["trials"]:
        predictions = {"source": [row["source_prediction"] for row in rows]}
        negative = {}
        for name, adapter in trial["models"].items():
            if adapter is None:
                continue
            if any(row["sample_id"] in adapter["fit_ids"] + adapter["validation_ids"] or
                   row["group_id"] in adapter["fit_groups"] + adapter["validation_groups"] or
                   row["workload_id"] in adapter["fit_workloads"] + adapter["validation_workloads"] for row in rows):
                raise ValueError("test leakage into adapter fitting or validation")
            result = apply_adapter(adapter, rows, source=adapter["source_identity"], environment=adapter["environment"])
            predictions[name], negative[name] = result["predictions"], result["negative_raw_memory_count"]
        if trial.get("current_transfer_binding"):
            predictions["current_transfer"], binding = _external_trial(current_transfer or [], trial, rows, "test")
            if binding != trial["current_transfer_binding"]:
                raise ValueError("current transfer checkpoint or fit provenance changed after validation")
        baseline = [dict(row["source_prediction"]) for row in rows]
        for target, name in trial["comparison_baselines"].items():
            api_name = TARGET_SCHEMA[target]["api_name"]
            for index in range(len(rows)):
                baseline[index][api_name] = predictions[name][index][api_name]
        candidate = predictions[trial["candidate"]]
        intervals = grouped_intervals(rows, candidate, baseline=baseline, seed=trial["seed"], draws=draws)
        benchmark = next((item for item in benchmarks or [] if item.get("adapter_fingerprint") ==
                          trial["models"][trial["candidate"]]["fingerprint"]), None)
        reports.append({"budget": trial["budget"], "seed": trial["seed"], "candidate": trial["candidate"],
                        "benchmark": benchmark, "current_transfer_binding": trial.get("current_transfer_binding"),
                        "comparison_baselines": trial["comparison_baselines"], "negative_raw_memory_counts": negative,
                        "metrics": {name: metrics(rows, values) for name, values in predictions.items()},
                        "strata": stratified_metrics(rows, candidate, draws=draws, seed=trial["seed"]), "paired_intervals": intervals,
                        "candidate_intervals": grouped_intervals(rows, candidate, seed=trial["seed"], draws=draws),
                        "promotion": promotion_gate(intervals, selection["margins"], benchmark=benchmark,
                                                     complete_comparisons=bool(trial.get("current_transfer_binding")),
                                                     memory_baseline_verified=trial["candidate"] != "linear_analytic")})
    result = {"version": REPORT_VERSION, "stage": "final_test", "selection_fingerprint": selection["fingerprint"],
            "test_ids": [row["sample_id"] for row in rows], "trials": reports,
            "profiling_cost": {"test_gpu_runs": len({run for row in rows for item in row["measurements"].values() for run in item["run_ids"]}),
                               "unique_test_configurations": len(rows)},
            "status": "experimental_not_promoted"}
    result["fingerprint"] = fingerprint(result)
    return result
