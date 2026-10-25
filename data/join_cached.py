#!/usr/bin/env python3
"""Attach historical measurements to independently generated workload identities.

This is a cache join, not a labeler or workload selector. Missing measurements
remain missing. No measured hardware ID, target, source identity, or split changes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import sys

from generate import canonical, digest, sha, HERE


def read_rows(path, key="profile_point_id", quarantine_duplicates=False):
    rows = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row[key] in rows:
            if not quarantine_duplicates:
                raise ValueError(f"duplicate {key}: {row[key]}")
            rows[row[key]] = {key: row[key], "status": "ambiguous_duplicate_quarantined"}
            continue
        rows[row[key]] = row
    return rows


def check_original(generated, original):
    for name in ("architecture", "batch_size", "input_specs", "precision_config", "optimizer", "model_id", "source_sha256"):
        if generated[name] != original[name]:
            raise ValueError(f"original identity mismatch: {generated['profile_point_id']} {name}")
    if generated["runtime_sha256"] != original["shared_runtime_sha256"]:
        raise ValueError("runtime hash mismatch")
    if original["shared_catalog_sha256"] != sha(HERE / "runtime/_model_catalog.py"):
        raise ValueError("catalog hash mismatch")


def check_variant(generated, label, config, sources, definitions, full_config):
    path = Path(sources) / config["model_file"]
    content = path.read_text()
    if sha(path) != config["source_sha256"] or label["variant_source_sha256"] != config["source_sha256"]:
        raise ValueError("variant source checksum mismatch: " + generated["profile_point_id"])
    if generated["variant_source_sha256"] != config["source_sha256"]:
        raise ValueError("directly regenerated variant differs from historical executable")
    tree = ast.parse(content)
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    expected = {"MODEL_CONFIG": full_config, "INPUT_SPECS": generated["input_specs"],
                "MODEL_ID": generated["model_id"], "PROFILE_POINT_ID": generated["profile_point_id"],
                "BATCH_SIZE": generated["batch_size"], "PRECISION_CONFIG": generated["precision_config"],
                "OPTIMIZER_NAME": generated["optimizer"]}
    if any(assignments.get(k) != v for k, v in expected.items()):
        raise ValueError("variant literal identity mismatch: " + generated["profile_point_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--hardware", choices=("A10", "A10G", "A100"), required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--original-workloads", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--runtime", type=Path, required=True, help="Portable predictor runtime containing src/ and aliases/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing to overwrite existing output")
    args.output.mkdir(parents=True, exist_ok=True)
    generated = read_rows(args.generated / "workloads.jsonl")
    configurations = read_rows(args.generated / "configurations.jsonl", "model_id")
    original = read_rows(args.original_workloads)
    labels = read_rows(args.labels, quarantine_duplicates=True)
    configs = read_rows(args.variant_root / "model_config.jsonl")
    tree = ast.parse((HERE / "runtime/_model_runtime.py").read_text())
    definitions = {x.name: ast.dump(x) for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    retained, missing, checks = [], [], 0
    expected_hardware = "nvidia_" + args.hardware.lower()
    for identifier, row in generated.items():
        check_original(row, original[identifier])
        if sha(args.generated / row["model_file"]) != row["source_sha256"]:
            raise ValueError("regenerated source checksum mismatch")
        if sha(args.generated / row["variant_file"]) != row["variant_source_sha256"]:
            raise ValueError("regenerated variant checksum mismatch")
        label = labels.get(identifier)
        if not label or label.get("status") != "ok":
            missing.append({"profile_point_id": identifier, "reason": "no_successful_historical_measurement", "original_status": None if not label else label.get("status")})
            continue
        if label["hardware_id"] != expected_hardware:
            raise ValueError("hardware identity mismatch: " + identifier)
        for name in ("model_id", "batch_size", "precision_config", "optimizer"):
            if label[name] != row[name]:
                raise ValueError("historical measurement identity mismatch: " + name)
        if args.hardware == "A10G":
            if label["variant_source_sha256"] != row["source_sha256"]:
                raise ValueError("A10G base-source checksum mismatch")
        else:
            check_variant(row, label, configs[identifier], args.variant_root, definitions, configurations[row["model_id"]]["config"])
        target = label["targets"]
        if any(not math.isfinite(float(target[k])) or float(target[k]) < 0 for k in (
                "train_step_wall_ms", "train_step_gpu_ms", "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib")):
            missing.append({"profile_point_id": identifier, "reason": "invalid_historical_target"})
            continue
        retained.append(label)
        checks += 1
    # Cache membership is checked only after workload construction and identity verification.
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime / "aliases")]
    import torch
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False, mmap=True)
    lookup = {identifier: i for i, identifier in enumerate(bundle["sample_ids"])}
    if len(lookup) != len(bundle["sample_ids"]):
        raise ValueError("duplicate feature IDs")
    kept_labels, train = [], []
    for label in retained:
        identifier = label["profile_point_id"]
        if identifier not in lookup:
            missing.append({"profile_point_id": identifier, "reason": "historical_label_exists_but_feature_bundle_excludes_row"})
            continue
        index = lookup[identifier]
        if int(bundle["splits"][index]) != 0 or generated[identifier]["frozen_split"] != 0:
            raise ValueError("non-TRAIN group in transfer training")
        if bundle["configuration_groups"][index] != generated[identifier]["configuration_group"]:
            raise ValueError("configuration group mismatch")
        t = label["targets"]
        raw = [t["train_step_wall_ms"], t["train_step_gpu_ms"], t["train_step_wall_ms"] * label["training"]["steps_per_epoch"],
               t["train_avg_sm_util_percent"], t["train_avg_vram_mib"], t["train_peak_vram_mib"]]
        if not torch.allclose(bundle["raw_targets_v4"][index].double(), torch.tensor(raw, dtype=torch.float64), rtol=1e-6, atol=1e-5):
            raise ValueError("cached feature target does not equal original measurement")
        train.append(index)
        kept_labels.append(label)
    if args.evaluation_manifest:
        old = json.loads(args.evaluation_manifest.read_text())
        validation = [lookup[x] for x in old["validation_ids"]]
        holdout = [lookup[x] for x in old["holdout_ids"]]
    else:
        validation = torch.nonzero(bundle["splits"] == 1).flatten().tolist()
        holdout = []
    for index in validation + holdout:
        if int(bundle["splits"][index]) != 1:
            raise ValueError("evaluation must retain historical validation groups")
    groups = [{bundle["configuration_groups"][i] for i in indices} for indices in (train, validation, holdout)]
    if groups[0] & (groups[1] | groups[2]) or groups[1] & groups[2]:
        raise ValueError("configuration leakage between transfer/evaluation sets")
    selection = {"schema": "perfseer-generated-transfer-cache-v1", "hardware": args.hardware,
                 "generated_manifest_sha256": sha(args.generated / "workloads.jsonl"),
                 "train_indices": train, "train_ids": [bundle["sample_ids"][i] for i in train],
                 "validation_indices": validation, "validation_ids": [bundle["sample_ids"][i] for i in validation],
                 "holdout_indices": holdout, "holdout_ids": [bundle["sample_ids"][i] for i in holdout],
                 "lineage_note": "Direct generated TRAIN workload design; identity-checked historical measurements and features. Evaluation IDs are preserved historical evaluator inputs, never generator inputs.",
                 "source_dataset_sha256": sha(args.bundle), "source_labels_sha256": sha(args.labels),
                 "original_workload_manifest_sha256": sha(args.original_workloads),
                 "evaluation_manifest_sha256": sha(args.evaluation_manifest) if args.evaluation_manifest else None}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(json.dumps({"event": "selection_ready", "path": str(args.output / "selection.json"), "train": len(train), "validation": len(validation), "holdout": len(holdout)}), flush=True)
    for name, records in (("labels_historical.jsonl", kept_labels), ("missing.jsonl", missing)):
        (args.output / name).write_text("".join(canonical(row) + "\n" for row in records))
    indices = train + validation + holdout
    result = {}
    for key, value in bundle.items():
        if isinstance(value, (list, tuple)) and len(value) == len(lookup):
            result[key] = [value[i] for i in indices]
        elif isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(lookup):
            result[key] = value[indices].clone()
        else:
            result[key] = value
    result["release_partition"] = torch.tensor([0] * len(train) + [1] * len(validation) + [2] * len(holdout))
    result["release_provenance"] = {"historical_cache": True, "generated_manifest_sha256": selection["generated_manifest_sha256"],
                                    "source_dataset_sha256": selection["source_dataset_sha256"], "hardware": args.hardware}
    torch.save(result, args.output / "dataset.pt")
    receipt = {"passed": True, "hardware": args.hardware, "requested_train_rows": len(generated), "available_train_rows": len(train),
               "missing_rows": len(missing), "identity_checked_labels": checks, "validation_rows": len(validation), "holdout_rows": len(holdout),
               "train_configuration_groups": len(groups[0]), "hardware_labels_changed": False,
               "new_measurements": False, "original_label_rows_retained_exactly": True,
               "missing_reasons": dict(Counter(x["reason"] for x in missing)),
               "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "join_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
