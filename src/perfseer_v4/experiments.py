"""Equal-budget transfer selection and separately sealed held-out evaluation."""

from collections import defaultdict
from pathlib import Path
import random
import time

import numpy as np
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32.calibration_contracts import environment_identity
from .dataset import checked_path, verify
from .runner import Samples, normalization_from_dict
from .training import metrics_from_predictions, restore_model, to_batch
from .transfer import fit_neural, source_identity
from .version import TARGET_NAMES

VARIANTS = ("source", "constant", "affine", "v4.1", "v4.0-finetune", "v4.2", "v4.3")
VERSION = "perfseer_v4_transfer_comparison_v1"


def choose_fit_ids(rows, budget, seed):
    if type(budget) is not int or budget < 1 or budget > len(rows):
        raise ValueError("fit budget must be a positive available sample count")
    if any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("fit selection requires training rows")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["sample_id"]):
        groups[row["group_id"]].append(row["sample_id"])
    if len({key for items in groups.values() for key in items}) != len(rows):
        raise ValueError("duplicate fit sample identity")
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        rng.shuffle(groups[name])
    order = []
    while len(order) < budget:
        for name in names:
            if groups[name]:
                order.append(groups[name].pop())
                if len(order) == budget:
                    break
    return order


def _samples(dataset, rows, source):
    return Samples(dataset, rows, normalization_from_dict(source["normalization"]), include_resources=source["model_variant"] == "v4.3")


@torch.no_grad()
def _predict(payload, samples, microbatch, device="cpu"):
    if microbatch < 1:
        raise ValueError("prediction microbatch must be positive")
    model = restore_model(payload, device=device).eval()
    predictions, timings = [], []
    for start in range(0, len(samples), microbatch):
        batch = to_batch(samples[start:start + microbatch], device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        values = model.predict_batch(batch).prediction.float().cpu().tolist()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - began) / len(values))
        predictions.extend(dict(zip(TARGET_NAMES, row, strict=True)) for row in values)
    return predictions, {"forward_seconds_per_row_p95": float(np.quantile(timings, .95)) if timings else None,
                         "scope": "model forward only; excludes restore and graph features", "microbatch": microbatch}


def _rows(root, rows, payload, environment, predictions):
    from .resource_model import resource_values
    identity = source_identity(payload)
    domain = environment_identity(environment)
    return [{**row, "workload_id": row["input_sha256"], "source_prediction": prediction,
             "features": resource_values(read_json(checked_path(root, row["input_path"]))),
             "source_identity": identity, "environment": environment, "domain_fingerprint": domain}
            for row, prediction in zip(rows, predictions, strict=True)]


def _metrics(predictions, rows, *, native=False):
    values = torch.tensor([[prediction[name] for name in TARGET_NAMES] for prediction in predictions], dtype=torch.float32)
    labels = "native_targets" if native else "targets"
    truth = torch.tensor([[row[labels][name] for name in TARGET_NAMES] for row in rows], dtype=torch.float32)
    metrics = metrics_from_predictions(values, truth)
    error = (values.double() - truth.double()).abs()
    relative = error / truth.double().abs().clamp_min(1e-6)
    metrics.update(relative_median=torch.quantile(relative, .5, dim=0).tolist(),
                   relative_p95=torch.quantile(relative, .95, dim=0).tolist(),
                   memory_underprediction_p95_mib={TARGET_NAMES[i]: float(torch.quantile((truth[:, i] - values[:, i]).clamp_min(0).double(), .95)) for i in (4, 5, 6)})
    return metrics


def _key(metrics, index):
    return (-metrics["within_5pct_count"][index], -metrics["within_10pct_count"][index],
            metrics["relative_median"][index], metrics["relative_p95"][index], metrics["mae"][index])


def _seal(value):
    return {**value, "fingerprint": fingerprint(value)}


def select(dataset, source, resource_source, environment, output, *, budgets=(32, 64, 128, 256),
           seeds=(11, 29, 47), variants=VARIANTS, device="cpu", epochs=100, microbatch=4):
    from .residuals import fit, apply_adapter, save_adapter
    dataset, source, output = Path(dataset).resolve(), Path(source).resolve(), Path(output).resolve()
    domain = environment_identity(environment)
    manifest = verify(dataset)
    if environment["hardware_id"] != manifest["prediction_hardware"]:
        raise ValueError("target environment differs from dataset hardware")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("unknown or repeated comparison variant")
    if not budgets or not seeds or len(set(budgets)) != len(budgets) or len(set(seeds)) != len(seeds):
        raise ValueError("unique nonempty budgets and seeds are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("selection requires a fresh output directory")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("model_variant") != "v4.0":
        raise ValueError("comparison source must be v4.0")
    sources = {"graph": payload}
    paths = {"graph": source}
    if "v4.3" in variants:
        if resource_source is None:
            raise ValueError("v4.3 requires a separately trained resource source checkpoint")
        paths["resource"] = Path(resource_source).resolve()
        sources["resource"] = torch.load(paths["resource"], map_location="cpu", weights_only=False)
        if sources["resource"].get("model_variant") != "v4.3":
            raise ValueError("resource checkpoint variant differs")
    # Test labels are not loaded here. Dataset verification only audits integrity.
    rows = {split: read_json(checked_path(dataset, manifest["split_files"][split]["path"])) for split in ("train", "validation")}
    for budget in budgets:
        choose_fit_ids(rows["train"], budget, seeds[0])
    prepared, samples, latency = {}, {}, {}
    for name, artifact in sources.items():
        prepared[name], samples[name], latency[name] = {}, {}, {}
        for split in ("train", "validation"):
            samples[name][split] = _samples(dataset, rows[split], artifact)
            predictions, latency[name][split] = _predict(artifact, samples[name][split], 1, "cpu")
            prepared[name][split] = _rows(dataset, rows[split], artifact, environment, predictions)
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in seeds:
        for budget in budgets:
            ids = choose_fit_ids(rows["train"], budget, seed)
            selected = set(ids)
            indices = [i for i, row in enumerate(rows["train"]) if row["sample_id"] in selected]
            candidates = {}
            for variant in variants:
                name = "resource" if variant == "v4.3" else "graph"
                artifact = sources[name]
                fit_rows = [prepared[name]["train"][i] for i in indices]
                validation = prepared[name]["validation"]
                started = time.perf_counter()
                entry = {"source": name, "fit_ids": sorted(ids), "validation_ids": [r["sample_id"] for r in rows["validation"]]}
                if variant == "source":
                    predictions = [row["source_prediction"] for row in validation]
                    entry["prediction_latency"] = latency[name]["validation"]
                elif variant in {"constant", "affine", "v4.1", "v4.3"}:
                    method = {"v4.1": "linear", "v4.3": "gp"}.get(variant, variant)
                    adapter = fit(fit_rows, validation, variant="v4.3" if variant == "v4.3" else "v4.1", method=method)
                    path = output / f"budget-{budget}-seed-{seed}-{variant}.json"
                    save_adapter(path, adapter)
                    begin = time.perf_counter()
                    predictions = apply_adapter(adapter, validation, source=source_identity(artifact), environment=environment)["predictions"]
                    entry.update(path=path.name, sha256=file_sha256(path), kind="residual",
                                 prediction_latency={"correction_seconds_per_row": (time.perf_counter() - begin) / len(validation),
                                                     "source_forward": latency[name]["validation"], "scope": "correction plus separately reported source forward"})
                else:
                    trained, fit_report = fit_neural(artifact, [samples[name]["train"][i] for i in indices],
                        samples[name]["validation"], variant=variant, dataset_fingerprint=manifest["fingerprint"],
                        hardware=manifest["prediction_hardware"], environment=environment, seed=seed,
                        device=device, epochs=epochs, microbatch=microbatch)
                    path = output / f"budget-{budget}-seed-{seed}-{variant}.pt"
                    atomic_write(path, trained, checkpoint=True)
                    predictions, entry["prediction_latency"] = _predict(trained, samples[name]["validation"], microbatch, device)
                    entry.update(path=path.name, sha256=file_sha256(path), kind="neural", fit_report=fit_report)
                entry.update(fit_and_validation_seconds=time.perf_counter() - started,
                             validation=_metrics(predictions, rows["validation"]),
                             validation_native=_metrics(predictions, rows["validation"], native=True))
                candidates[variant] = entry
            best = {target: min(candidates, key=lambda v: (*_key(candidates[v]["validation"], i), v)) for i, target in enumerate(TARGET_NAMES)}
            controls = [v for v in candidates if v in {"source", "constant", "affine", "v4.0-finetune"}]
            best_control = {target: min(controls, key=lambda v: (*_key(candidates[v]["validation"], i), v)) for i, target in enumerate(TARGET_NAMES)} if controls else {}
            fit_rows = [rows["train"][i] for i in indices]
            trials.append({"budget": budget, "seed": seed, "fit_ids": sorted(ids),
                           "fit_cost": {"recorded_samples": len(fit_rows),
                                        "distinct_training_inputs": len({row["input_sha256"] for row in fit_rows}),
                                        "independent_groups": len({row["group_id"] for row in fit_rows})}, "candidates": candidates,
                           "validation_best_by_target": best, "validation_control_by_target": best_control})
    selection = _seal({"version": VERSION, "stage": "validation_selection", "dataset": str(dataset),
                       "dataset_fingerprint": manifest["fingerprint"], "environment": environment,
                       "domain_fingerprint": domain, "label_policy": manifest.get("label_policy"),
                       "sources": {key: {"path": str(paths[key]), "sha256": file_sha256(paths[key]), "identity": source_identity(value)} for key, value in sources.items()},
                       "test_split": manifest["split_files"]["test"], "trials": trials,
                       "cost": {"fit_budget_unit": "distinct recorded sample IDs", "validation_samples": len(rows["validation"]),
                                "validation_distinct_training_inputs": len({row["input_sha256"] for row in rows["validation"]}),
                                "validation_independent_groups": len({row["group_id"] for row in rows["validation"]}),
                                "repeat_measurements": "not available in prepared dataset; not counted as new independent labels",
                                "new_gpu_measurements": 0}, "deployment_approved": False})
    atomic_write(output / "selection.json", selection)
    return selection


def _paired_interval(candidate, control, rows, index, seed):
    truth = np.asarray([row["targets"][TARGET_NAMES[index]] for row in rows], dtype=np.float64)
    difference = (np.abs(np.asarray([r[TARGET_NAMES[index]] for r in candidate]) - truth)
                  - np.abs(np.asarray([r[TARGET_NAMES[index]] for r in control]) - truth)) / np.maximum(np.abs(truth), 1e-6)
    groups = defaultdict(list)
    for value, row in zip(difference, rows, strict=True):
        groups[row["group_id"]].append(value)
    values = np.asarray([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(1000, len(values)))].mean(1)
    return {"candidate_minus_control_group_mean_relative_error": float(values.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "independent_groups": len(values)}


def evaluate_selection(selection_path, output, *, device="cpu", microbatch=4):
    from .residuals import load_adapter, apply_adapter
    selection_path, output = Path(selection_path).resolve(), Path(output).resolve()
    selection = read_json(selection_path)
    if (selection.get("version") != VERSION or selection.get("stage") != "validation_selection"
            or selection.get("fingerprint") != fingerprint({k: v for k, v in selection.items() if k != "fingerprint"})):
        raise ValueError("selection fingerprint or contract differs")
    if output.exists():
        raise ValueError("evaluation requires a new report path")
    if environment_identity(selection["environment"]) != selection["domain_fingerprint"]:
        raise ValueError("selection environment differs")
    dataset = Path(selection["dataset"])
    manifest = verify(dataset)
    if manifest["fingerprint"] != selection["dataset_fingerprint"] or manifest["split_files"]["test"] != selection["test_split"]:
        raise ValueError("evaluation dataset or held-out split differs")
    sources = {}
    for key, binding in selection["sources"].items():
        if file_sha256(binding["path"]) != binding["sha256"]:
            raise ValueError("source checkpoint changed after selection")
        sources[key] = torch.load(binding["path"], map_location="cpu", weights_only=False)
        if source_identity(sources[key]) != binding["identity"]:
            raise ValueError("source preprocessing changed after selection")
    for trial in selection["trials"]:
        for candidate in trial["candidates"].values():
            if "path" in candidate and file_sha256(checked_path(selection_path.parent, candidate["path"])) != candidate["sha256"]:
                raise ValueError("selected candidate artifact changed")
    # Only now open test labels, after every selected artifact has been checked.
    rows = read_json(checked_path(dataset, selection["test_split"]["path"]))
    prepared, samples = {}, {}
    for key, artifact in sources.items():
        samples[key] = _samples(dataset, rows, artifact)
        predictions, _ = _predict(artifact, samples[key], 1, "cpu")
        prepared[key] = _rows(dataset, rows, artifact, selection["environment"], predictions)
    trials = []
    for trial in selection["trials"]:
        predictions, candidates = {}, {}
        for variant, selected in trial["candidates"].items():
            key = selected["source"]
            if variant == "source":
                values = [r["source_prediction"] for r in prepared[key]]
            elif selected["kind"] == "residual":
                adapter = load_adapter(checked_path(selection_path.parent, selected["path"]))
                values = apply_adapter(adapter, prepared[key], source=source_identity(sources[key]), environment=selection["environment"])["predictions"]
            else:
                artifact = torch.load(checked_path(selection_path.parent, selected["path"]), map_location="cpu", weights_only=False)
                values, _ = _predict(artifact, samples[key], microbatch, device)
            predictions[variant] = values
            metrics = _metrics(values, rows)
            candidates[variant] = {"test": metrics, "test_native": _metrics(values, rows, native=True),
                                   "evidence_status": "accuracy_gate_passed" if selected["validation"]["gate_passed"] and metrics["gate_passed"] and len({row["group_id"] for row in rows}) >= 10 else "inconclusive"}
        for variant, result in candidates.items():
            result["primary_paired_intervals"] = {}
            for index in (0, 5):
                control = trial["validation_control_by_target"].get(TARGET_NAMES[index])
                if control:
                    result["primary_paired_intervals"][TARGET_NAMES[index]] = {"control": control,
                        **_paired_interval(predictions[variant], predictions[control], rows, index, trial["seed"])}
        trials.append({"budget": trial["budget"], "seed": trial["seed"], "fit_ids": trial["fit_ids"],
                       "fit_cost": trial["fit_cost"], "candidates": candidates})
    report = _seal({"version": VERSION, "stage": "held_out_evaluation", "selection_fingerprint": selection["fingerprint"],
                    "dataset_fingerprint": manifest["fingerprint"], "label_policy": manifest.get("label_policy"),
                    "test_ids": [row["sample_id"] for row in rows], "trials": trials, "cost": selection["cost"],
                    "status": "experimental_not_promoted", "deployment_approved": False,
                    "limitations": ["uncertainty is grouped sampling uncertainty, not a deployment guarantee",
                                    "source-seen versus source-unseen membership requires a verified source-training manifest",
                                    "repeat profiling cost is not recoverable from prepared rows alone"]})
    atomic_write(output, report)
    return report
