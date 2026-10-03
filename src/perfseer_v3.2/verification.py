"""Independently reconcile completed local epochs with dataset and checkpoints."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .runner import code_fingerprint, normalization_from_dict
from .training import T1, restore_model, selection_key
from .version import TARGET_NAMES, METRIC_VERSION, HEAD_GROUPS, SM_INDICES, validate_targets
from .features import build_features


def verify_run(dataset, output):
    dataset, output = Path(dataset), Path(output)
    manifest = read_json(dataset / "dataset_manifest.json")
    report = read_json(output / "local-validation.json")
    train = read_json(dataset / manifest["split_files"]["train"]["path"])
    validation = read_json(dataset / manifest["split_files"]["validation"]["path"])
    if report["status"] != "passed" or report["accuracy_gate_claimed"] or report["dataset_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("invalid execution validation report")
    if report["code_fingerprint"] != code_fingerprint() or tuple(report["target_names"]) != TARGET_NAMES:
        raise ValueError("validation code or output contract differs")
    sample_hash = fingerprint(sorted(row["sample_id"] for row in train))
    for epoch in range(1, report["epochs"] + 1):
        evidence = read_json(output / f"teacher-epoch-{epoch:04d}.json")
        if evidence["train_rows"] != len(train) or evidence["validation_rows"] != len(validation) or evidence["train_sample_ids_sha256"] != sample_hash:
            raise ValueError("epoch did not cover the full training/validation partitions")
        if evidence["optimizer_steps"] != math.ceil(len(train) / evidence["effective_batch"]):
            raise ValueError("incomplete optimizer steps")
        predictions = output / f"teacher-epoch-{epoch:04d}-predictions.json.gz"
        verify_predictions(predictions, validation)
        if file_sha256(predictions) != evidence["validation"]["predictions_sha256"]:
            raise ValueError("validation prediction export differs")
        if tuple(evidence["best_key"]) > selection_key(evidence["validation"], epoch):
            raise ValueError("best checkpoint selection disagrees with hit rates")
    path = output / "teacher-latest.pt"
    if file_sha256(path) != report["checkpoint_sha256"]:
        raise ValueError("latest checkpoint hash differs")
    latest = torch.load(path, map_location="cpu", weights_only=False)
    if latest["epoch"] != report["epochs"] or latest["next_epoch"] != report["epochs"] + 1 or latest["next_batch"] != 0:
        raise ValueError("checkpoint is not at the complete epoch boundary")
    if latest["train_rows"] != len(train) or latest["validation_rows"] != len(validation):
        raise ValueError("checkpoint row counts differ")
    if normalization_from_dict(latest["normalization"]).split_fingerprint != manifest["fingerprint"]:
        raise ValueError("normalization belongs to another dataset")
    if any(latest["model_config"][key] != value for key, value in T1.items()):
        raise ValueError("validation did not use the full T1 model")
    restored = restore_model(latest, dataset_fingerprint=manifest["fingerprint"])
    parameters = sum(p.numel() for p in restored.parameters())
    if parameters != report["parameters"] or not all(torch.isfinite(p).all() for p in restored.parameters()):
        raise ValueError("invalid restored model parameters")
    for prediction in report["checkpoint_reload_prediction"]:
        values = [prediction[name] for name in TARGET_NAMES]
        validate_targets(values)
    return {"status": "verified", "epochs": report["epochs"], "train_rows_per_epoch": len(train),
            "validation_rows_per_epoch": len(validation), "parameters": parameters,
            "dataset_fingerprint": manifest["fingerprint"], "checkpoint_sha256": report["checkpoint_sha256"],
            "full_partition_coverage": True, "full_T1_capacity": True, "checkpoint_reload": True}



def verify_predictions(path, expected_rows=None):
    """Reconstruct counts and errors independently with NumPy, without the evaluator."""
    exported = read_json(path)
    if exported["metric_version"] != METRIC_VERSION or tuple(exported["target_names"]) != TARGET_NAMES:
        raise ValueError("prediction metric contract differs")
    rows, metrics = exported["rows"], exported["metrics"]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate prediction identities")
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    if prediction.shape != (len(rows), 12) or target.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("invalid exported prediction tensors")
    if not np.array_equal(prediction.astype(np.float32).astype(np.float64), prediction) or not np.array_equal(target.astype(np.float32).astype(np.float64), target):
        raise ValueError("export is not the evaluated FP32 tensor values")
    for values in target:
        validate_targets(values)
    if expected_rows is not None:
        by_id = {row["sample_id"]: row for row in expected_rows}
        if set(ids) != set(by_id) or len(ids) != len(expected_rows):
            raise ValueError("exported split coverage differs")
        expected = np.asarray([[by_id[key]["targets"][name] for name in TARGET_NAMES] for key in ids], dtype=np.float32)
        if not np.array_equal(target, expected.astype(np.float64)):
            raise ValueError("exported truth differs from native targets")
    sizes = exported["batch_sizes"]
    if any(type(size) is not int or size < 1 for size in sizes) or sum(sizes) != len(rows):
        raise ValueError("exported batch coverage differs")
    if [row["batch_index"] for row in rows] != [index for index, size in enumerate(sizes) for _ in range(size)]:
        raise ValueError("exported batch identities differ")
    absolute = np.abs(prediction - target)
    relative = absolute / np.maximum(np.abs(target), 1e-6)
    five, ten = (relative <= .05).sum(0).tolist(), (relative <= .1).sum(0).tolist()
    if metrics["metric_version"] != METRIC_VERSION or tuple(metrics["target_names"]) != TARGET_NAMES or metrics["rows"] != len(rows) or metrics["within_5pct_count"] != five or metrics["within_10pct_count"] != ten:
        raise ValueError("reported hit counts disagree with predictions")
    for key, counts in (("within_5pct_accuracy", five), ("within_10pct_accuracy", ten)):
        if metrics[key] != dict(zip(TARGET_NAMES, [count / len(rows) for count in counts], strict=True)):
            raise ValueError("reported hit rates disagree with counts")
    np.testing.assert_allclose(metrics["mae"], absolute.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics["relative_mean_error"], relative.mean(0), rtol=1e-12, atol=1e-12)
    denominator = target.astype(np.float32).copy()
    denominator[:, SM_INDICES] = np.maximum(denominator[:, SM_INDICES], 1.)
    errors = np.abs(prediction.astype(np.float32) - target.astype(np.float32)) / denominator
    loss = float(np.mean([errors[:, indices].mean() for indices in HEAD_GROUPS]))
    np.testing.assert_allclose(metrics["group_balanced_loss"], loss, rtol=1e-6, atol=1e-7)
    for index in SM_INDICES:
        mask = target[:, index] == 0
        zero = metrics["zero_sm"][TARGET_NAMES[index]]
        if zero["rows"] != int(mask.sum()) or zero["within_5pct_count"] != int((relative[mask, index] <= .05).sum()):
            raise ValueError("zero-SM diagnostics disagree")
        if mask.any():
            np.testing.assert_allclose(zero["mae"], absolute[mask, index].mean(), rtol=1e-12, atol=1e-12)
        elif zero["mae"] is not None:
            raise ValueError("empty zero-SM population has a reported error")
    passed = all(20 * count >= 19 * len(rows) for count in five)
    if metrics["gate_passed"] is not passed or metrics["evaluation"] != exported["evaluation"]:
        raise ValueError("reported acceptance or execution configuration differs")
    return {"status": "verified", "rows": len(rows), "predictions_sha256": file_sha256(path),
            "metric_version": METRIC_VERSION, "gate_passed": passed}


def _input_identity(item):
    path, expected_hash = item
    torch.set_num_threads(1)
    if file_sha256(path) != expected_hash:
        raise ValueError("audit input hash differs")
    features = build_features(read_json(path))
    identities = {}
    for mode in ("training", "inference"):
        value = getattr(features, mode)
        digest = hashlib.sha256()
        for field in fields(value):
            tensor = getattr(value, field.name)
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().contiguous()
                digest.update(f"{field.name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
                digest.update(tensor.numpy().tobytes())
        identities[mode] = digest.hexdigest()
    return str(path), identities


def oracle_hits(truth, tolerance):
    radius = tolerance * np.maximum(np.abs(truth), 1e-6)
    events = sorted([(float(value), 0) for value in truth - radius] + [(float(value), 1) for value in truth + radius])
    active = best = 0
    for _, end in events:
        active += -1 if end else 1
        best = max(best, active)
    return best


def audit_dataset(dataset, output, *, baseline=None, workers=4):
    dataset, output = Path(dataset), Path(output)
    manifest = read_json(dataset / "dataset_manifest.json")
    splits = {split: read_json(dataset / manifest["split_files"][split]["path"]) for split in ("train", "validation")}
    paths = {str(dataset / row["input_path"]): row["input_sha256"] for rows in splits.values() for row in rows}
    identities = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (path, value) in enumerate(pool.map(_input_identity, paths.items()), 1):
            identities[path] = value
            if index % 100 == 0:
                print(f"input audit: {index}/{len(paths)}", flush=True)
    report = {"dataset_fingerprint": manifest["fingerprint"], "metric_version": METRIC_VERSION,
              "feature_code_fingerprint": code_fingerprint(), "test_labels_evaluated": False,
              "oracle_definition": "optimistic independent constant per mode input tensor identity and label; diagnostic only",
              "splits": {}}
    medians = None
    for split, rows in splits.items():
        truth = np.asarray([[row["targets"][name] for name in TARGET_NAMES] for row in rows], dtype=np.float32).astype(np.float64)
        if split == "train":
            medians = np.median(truth, axis=0)
        groups = {mode: defaultdict(list) for mode in ("training", "inference")}
        for index, row in enumerate(rows):
            for mode in groups:
                groups[mode][identities[str(dataset / row["input_path"])][mode]].append(index)
        metrics = {}
        for index, name in enumerate(TARGET_NAMES):
            mode = "training" if name.startswith("train_") else "inference"
            denominator = np.maximum(np.abs(truth[:, index]), 1e-6)
            baseline_error = np.abs(medians[index] - truth[:, index]) / denominator
            metrics[name] = {"training_median_within_5pct": float((baseline_error <= .05).mean()),
                             "training_median_within_10pct": float((baseline_error <= .1).mean()),
                             "oracle_within_5pct": sum(oracle_hits(truth[indices, index], .05) for indices in groups[mode].values()) / len(rows),
                             "oracle_within_10pct": sum(oracle_hits(truth[indices, index], .1) for indices in groups[mode].values()) / len(rows)}
        report["splits"][split] = {"rows": len(rows), "unique_mode_inputs": {mode: len(group) for mode, group in groups.items()}, "metrics": metrics}
    report["validation_gate_feasible_for_current_features"] = all(value["oracle_within_5pct"] >= .95 for value in report["splits"]["validation"]["metrics"].values())
    if baseline is not None:
        with gzip.open(baseline, "rt") as stream:
            predictions = [json.loads(line) for line in stream]
        rows = {row["sample_id"]: row for row in splits["validation"]}
        if len(predictions) != len(rows) or {row["sample_id"] for row in predictions} != set(rows):
            raise ValueError("delivered baseline validation identities differ")
        truth, predicted = [], []
        for row in predictions:
            if tuple(row["target_names"]) != TARGET_NAMES or not all(row["validity_mask"].values()):
                raise ValueError("delivered baseline output contract differs")
            expected = np.asarray([rows[row["sample_id"]]["targets"][name] for name in TARGET_NAMES], dtype=np.float32)
            if not np.array_equal(expected, [row["truth"][name] for name in TARGET_NAMES]):
                raise ValueError("baseline truth differs from preserved native labels")
            truth.append(expected)
            predicted.append([row["prediction"][name] for name in TARGET_NAMES])
        truth = np.asarray(truth, dtype=np.float64)
        prediction = np.asarray(predicted, dtype=np.float32).astype(np.float64)
        relative = np.abs(prediction - truth) / np.maximum(np.abs(truth), 1e-6)
        report["delivered_baseline"] = {"sha256": file_sha256(baseline), "rows": len(rows),
                                       "within_5pct_accuracy": dict(zip(TARGET_NAMES, (relative <= .05).mean(0).tolist(), strict=True)),
                                       "within_10pct_accuracy": dict(zip(TARGET_NAMES, (relative <= .1).mean(0).tolist(), strict=True))}
    atomic_write(output / "paired-input-audit.json", report)
    atomic_write(output / "mode-input-identities.json", {"dataset_fingerprint": manifest["fingerprint"], "identities": identities})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--audit-dataset", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.predictions:
        result = verify_predictions(args.predictions)
    elif args.dataset is None or args.output is None:
        parser.error("--dataset and --output are required for run or dataset verification")
    elif args.audit_dataset:
        result = audit_dataset(args.dataset, args.output, baseline=args.baseline, workers=args.workers)
    else:
        result = verify_run(args.dataset, args.output)
        atomic_write(args.output / "independent-verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
