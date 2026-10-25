#!/usr/bin/env python3
"""Pack freshly measured generated workloads into the pinned feature contract.

Static forward-only features are estimated from generated source. Physical
targets come only from verified successful local-labeler records. No hardware
name override, label imputation, split reassignment or normalizer fitting.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teachers import ROOT, bootstrap, sha
bootstrap()
import numpy as np
import torch
from perfseer_v3.features import apply_normalization
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v4.native_data import static_description
from perfseer_v4.native_graph import build_graph
from perfseer_v4.seerblock_data import graph_features
from v5_objectives import TARGET_NAMES, transform_targets


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("record target does not equal raw observations")


def verify_measurement(row, run, workload):
    if row["run_sha256"] != fingerprint(run) or row["hardware"] != run["hardware"]:
        raise ValueError("measurement run/hardware mismatch")
    if row["workload_sha256"] != workload["workload_sha256"] or row["source_sha256"] != workload["variant_source_sha256"]:
        raise ValueError("measured source identity mismatch")
    target, protocol = row["targets"], row["protocol"]
    if protocol != run["protocol"]:
        raise ValueError("measurement protocol differs from run")
    samples = row["resource_samples"]
    if len(samples) < protocol["resource_min_samples"]:
        raise ValueError("insufficient NVML samples")
    for domain in ("wall", "gpu"):
        epochs = [e[domain + "_ms"] for repeat in row["repeats"] for e in repeat["epochs"]]
        if len(epochs) != protocol["repeats"] * protocol["measured_epochs"]:
            raise ValueError("incomplete timing observations")
        close(target[f"train_epoch_{domain}_ms"], statistics.fmean(epochs))
        close(target[f"train_step_{domain}_ms"] * protocol["steps_per_epoch"], statistics.fmean(epochs))
    close(target["train_avg_nvml_busy_percent"], statistics.fmean(s["busy_percent"] for s in samples))
    for name, reduce in (("avg", statistics.fmean), ("peak", max)):
        measured = reduce(s["device_memory_bytes"] for s in samples) - row["baseline_before"]["memory_bytes"]
        close(target[f"train_{name}_vram_bytes"], measured)
        close(target[f"train_{name}_vram_mib"] * 1048576, measured)
    if abs(row["baseline_drift_bytes"]) > protocol["max_baseline_drift_mib"] * 1048576:
        raise ValueError("contaminated baseline accepted")
    raw = [target["train_step_wall_ms"], target["train_step_gpu_ms"], target["train_epoch_wall_ms"],
           target["train_avg_nvml_busy_percent"], target["train_avg_vram_mib"], target["train_peak_vram_mib"]]
    if any(not math.isfinite(x) for x in raw) or any(raw[i] <= 0 for i in (0, 1, 2, 4, 5)) or not 0 <= raw[3] <= 100:
        raise ValueError("invalid physical targets")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="One local GPU output directory containing run.json and records/")
    parser.add_argument("--architecture", type=Path, default=ROOT / "architecture.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing to overwrite an existing pack")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    metadata = torch.load(args.architecture, weights_only=False, map_location="cpu")
    normalizer = metadata["source_normalization"]
    run = json.loads((args.labels / "run.json").read_text())
    if run["generated_manifest_sha256"] != sha(args.generated / "workloads.jsonl"):
        raise ValueError("generated manifest differs from labeler run")
    hardware = run["hardware"]["hardware_id"]
    if not hardware.startswith("nvidia_"):
        raise ValueError("labeler hardware identity is not canonical NVIDIA")
    rows = [json.loads(x) for x in (args.generated / "workloads.jsonl").open()]
    if len({r["profile_point_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate generated identity")
    records = {}
    for path in (args.labels / "records").glob("*.json"):
        row = json.loads(path.read_text())
        if row["profile_point_id"] in records:
            raise ValueError("duplicate local measurement identity")
        records[row["profile_point_id"]] = row
    templates, registry = {}, OperationRegistry.load()
    features, ids, groups, families, splits, raw, missing = [], [], [], [], [], [], []
    record_hashes = {}
    for workload in rows:
        identifier = workload["profile_point_id"]
        record = records.get(identifier)
        if record is None or record["status"] != "ok":
            missing.append({"profile_point_id": identifier, "status": "not_measured" if record is None else record["status"]})
            continue
        split = workload["frozen_split"]
        if split not in (0, 1, 2):
            raise ValueError("no frozen split for generated configuration")
        values = verify_measurement(record, run, workload)
        source = args.generated / workload["variant_file"]
        if sha(source) != workload["variant_source_sha256"]:
            raise ValueError("generated source changed")
        text = source.read_text()
        description = static_description(text, workload, templates)
        if description["inputs"] != workload["input_specs"]:
            raise ValueError("source input specification differs")
        group = f"{workload['architecture']['family_index']:02d}:" + fingerprint(description["model"])
        if group != workload["configuration_group"]:
            raise ValueError("configuration-group identity differs")
        graph = build_graph(text, description["model"], description["inputs"])
        steps = int(run["protocol"]["steps_per_epoch"])
        training = {"batch_size": workload["batch_size"], "effective_batch_size": workload["batch_size"],
                    "grad_accumulation_steps": 1, "steps_per_epoch": steps,
                    "precision": workload["precision_config"], "optimizer": workload["optimizer"]}
        feature = apply_normalization(graph_features(graph, training, hardware.removeprefix("nvidia_"), registry), normalizer)
        feature.validate()
        features.append(feature); ids.append(identifier); groups.append(group); families.append(workload["model_id"].split("_")[1])
        splits.append(split); raw.append(values); record_hashes[identifier] = fingerprint(record)
    if not ids:
        raise ValueError("no successful measured workloads to pack")
    raw = np.asarray(raw, dtype=np.float64)
    targets, valid = transform_targets(raw)
    if not valid.all():
        raise ValueError("invalid v5 targets; imputation is disabled")
    seen = {}
    for group, split in zip(groups, splits):
        if seen.setdefault(group, split) != split:
            raise ValueError("configuration crosses splits")
    steps = int(run["protocol"]["steps_per_epoch"])
    protocol = "measured_single_step" if steps == 1 else "derived_step_times_count"
    manifest = {"schema": "perfseer-fresh-static-label-pack-v1", "target_names": list(TARGET_NAMES),
                "measured_hardware_id": hardware, "hardware": run["hardware"], "measurement_protocol": run["protocol"],
                "epoch_contract": "step mean is measured epoch mean divided by explicit step count; epoch output uses step-times-count contract",
                "static_capture": "estimated forward-only; no measured backward/lifetimes/runtime dtype capture",
                "normalization_sha256": normalizer.sha256, "architecture_sha256": sha(args.architecture),
                "generated_manifest_sha256": sha(args.generated / "workloads.jsonl"), "labeler_run_sha256": fingerprint(run),
                "measurement_record_sha256": record_hashes, "packer_sha256": sha(__file__),
                "split_counts": dict(Counter(map(str, splits))), "missing_rows": len(missing),
                "historical_protocol_equivalent": False}
    def input_bundle(indices):
        return {"features": [features[i] for i in indices], "sample_ids": [ids[i] for i in indices],
                "normalization": normalizer, "raw_steps_per_epoch": torch.full((len(indices),), steps, dtype=torch.long),
                "epoch_protocol": protocol}
    trainval = [i for i, split in enumerate(splits) if split != 2]
    bundle = {**input_bundle(trainval), "targets": torch.from_numpy(targets[trainval].copy()),
              "raw_targets_v4": torch.from_numpy(raw[trainval].copy()), "splits": torch.tensor([splits[i] for i in trainval]),
              "configuration_groups": [groups[i] for i in trainval], "families": [families[i] for i in trainval], "manifest": manifest}
    torch.save(bundle, args.output / "dataset.pt")
    torch.save(input_bundle(list(range(len(ids)))), args.output / "all-inputs.pt")
    for name, split in (("validation", 1), ("test", 2)):
        indices = [i for i, value in enumerate(splits) if value == split]
        torch.save(input_bundle(indices), args.output / (name + "-inputs.pt"))
        np.savez_compressed(args.output / (name + "-key.npz"), sample_ids=np.asarray([ids[i] for i in indices]),
                            targets=targets[indices], raw_targets_v4=raw[indices], families=np.asarray([families[i] for i in indices]))
    (args.output / "missing.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in missing))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Verify exact target and feature identities after serialization, not just in memory.
    replay = torch.load(args.output / "dataset.pt", weights_only=False, map_location="cpu", mmap=True)
    if replay["sample_ids"] != bundle["sample_ids"] or not torch.equal(replay["targets"], bundle["targets"]):
        raise ValueError("serialized target/identity replay differs")
    for feature in replay["features"]:
        feature.validate()
        if feature.metadata["normalization_sha256"] != normalizer.sha256:
            raise ValueError("serialized normalization changed")
    receipt = {"passed": True, "hardware": hardware, "accepted_rows": len(ids), "dataset_rows": len(trainval),
               "split_counts": manifest["split_counts"], "missing_rows": len(missing),
               "ready_for_training": bool(any(s == 0 for s in splits) and any(s == 1 for s in splits)),
               "all_targets_recomputed_from_observations": True, "all_generated_source_identities_verified": True,
               "normalizer_refitted": False, "cuda_used_for_feature_capture": False,
               "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    main()
