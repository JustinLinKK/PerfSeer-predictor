#!/usr/bin/env python3
"""Independent verification of generated sources, coverage and cached data."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import math
from generate import HERE, sha, digest, render_source, render_variant, group_key, configurations
from label import load_manifest


def verify_generated(root, original=None, variants=None):
    rows = load_manifest(root)
    receipt = json.loads((root / "generation.json").read_text())
    configs = {r["model_id"]: r["config"] for r in map(json.loads, (root / "configurations.jsonl").open())}
    splits = json.loads((HERE / "specs/frozen_group_splits.json").read_text())
    originals = {r["profile_point_id"]: r for r in map(json.loads, Path(original).open())} if original else None
    variant_rows = {r["profile_point_id"]: r for r in map(json.loads, Path(variants).open())} if variants else None
    for row in rows:
        config = configs[row["model_id"]]
        if render_source(row["model_id"], config) != (root / row["model_file"]).read_text():
            raise ValueError("base source replay mismatch")
        if render_variant(config, row["batch_size"], row["precision_config"], row["optimizer"]) != (root / row["variant_file"]).read_text():
            raise ValueError("variant source replay mismatch")
        if row["configuration_group"] != group_key(config) or row["frozen_split"] != splits.get(group_key(config)):
            raise ValueError("frozen group assignment mismatch")
        if originals:
            old = originals[row["profile_point_id"]]
            if any(row[k] != old[k] for k in ("source_sha256", "architecture", "input_specs", "optimizer", "precision_config", "batch_size")):
                raise ValueError("regenerated original identity mismatch")
        if variant_rows and row["variant_source_sha256"] != variant_rows[row["profile_point_id"]]["source_sha256"]:
            raise ValueError("regenerated historical variant SHA mismatch")
    if receipt["train_budget"]:
        expected_models = {f"model_{c['family_index']:02d}_{c['sample_rank']:04d}" for c in configurations("pretrain", receipt["seed"])
                           if splits[group_key(c)] == 0}
        if set(configs) != expected_models or any(r["frozen_split"] != 0 for r in rows):
            raise ValueError("budgeted design fails complete TRAIN configuration coverage")
    result = {"passed": True, "workload_rows": len(rows), "model_count": len(configs),
              "configuration_groups": len({r["configuration_group"] for r in rows}),
              "families": len({r["family_id"] for r in rows}), "modalities": dict(Counter(r["modality"] for r in rows)),
              "precisions": dict(Counter(r["precision_config"] for r in rows)), "optimizers": dict(Counter(r["optimizer"] for r in rows)),
              "all_sources_replay_exactly": True, "historical_base_hashes_checked": bool(original),
              "historical_variant_hashes_checked": bool(variants), "labels_read_for_generation": False,
              "manifest_sha256": sha(root / "workloads.jsonl")}
    return result


def verify_cached(root, generated, runtime):
    sys.path[:0] = [str(runtime / "src"), str(runtime / "aliases")]
    import torch
    selection = json.loads((root / "selection.json").read_text())
    receipt = json.loads((root / "join_receipt.json").read_text())
    for name, checksum in receipt["files"].items():
        if sha(root / name) != checksum:
            raise ValueError("cached file hash changed: " + name)
    data = torch.load(root / "dataset.pt", weights_only=False, map_location="cpu", mmap=True)
    generated_rows = {r["profile_point_id"]: r for r in map(json.loads, (generated / "workloads.jsonl").open())}
    labels = list(map(json.loads, (root / "labels_historical.jsonl").open()))
    missing = list(map(json.loads, (root / "missing.jsonl").open()))
    train_ids = selection["train_ids"]
    if set(train_ids) & {r["profile_point_id"] for r in missing}:
        raise ValueError("missing ID included in training")
    if set(train_ids) | {r["profile_point_id"] for r in missing} != set(generated_rows):
        raise ValueError("generated ID missing from availability accounting")
    if [r["profile_point_id"] for r in labels] != train_ids:
        raise ValueError("labels are not in TRAIN identity order")
    expected_ids = train_ids + selection["validation_ids"] + selection["holdout_ids"]
    if data["sample_ids"] != expected_ids:
        raise ValueError("feature identity order mismatch")
    groups = []
    for part in (0, 1, 2):
        indices = torch.nonzero(data["release_partition"] == part).flatten().tolist()
        groups.append({data["configuration_groups"][i] for i in indices})
    if groups[0] & (groups[1] | groups[2]) or groups[1] & groups[2]:
        raise ValueError("configuration leakage")
    for i, row in enumerate(labels):
        if row["hardware_id"] != "nvidia_" + selection["hardware"].lower():
            raise ValueError("wrong hardware")
        if data["configuration_groups"][i] != generated_rows[row["profile_point_id"]]["configuration_group"]:
            raise ValueError("generated/bundle group mismatch")
        targets, steps = row["targets"], row["training"]["steps_per_epoch"]
        raw = [targets["train_step_wall_ms"], targets["train_step_gpu_ms"], targets["train_step_wall_ms"] * steps,
               targets["train_avg_sm_util_percent"], targets["train_avg_vram_mib"], targets["train_peak_vram_mib"]]
        expected = raw[:4] + [max(1, math.ceil(x * 1048576 / 100000000)) * 100000000 / 1048576 for x in raw[4:]]
        if not torch.allclose(data["raw_targets_v4"][i].double(), torch.tensor(raw, dtype=torch.float64), rtol=1e-6, atol=1e-5):
            raise ValueError("raw historical labels mismatch")
        if not torch.allclose(data["targets"][i].double(), torch.tensor(expected, dtype=torch.float64), rtol=1e-6, atol=1e-5):
            raise ValueError("v5 target semantics mismatch")
    return {"passed": True, "hardware": selection["hardware"], "train": len(train_ids), "validation": len(selection["validation_ids"]),
            "holdout": len(selection["holdout_ids"]), "missing": len(missing), "train_configuration_groups": len(groups[0]),
            "all_cached_labels_rederived": True, "all_file_hashes_verified": True, "no_group_leakage": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--original-workloads", type=Path)
    parser.add_argument("--variant-config", type=Path)
    parser.add_argument("--cached", type=Path)
    parser.add_argument("--runtime", type=Path, default=HERE.parent / "runtime")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify_cached(args.cached, args.generated, args.runtime) if args.cached else verify_generated(args.generated, args.original_workloads, args.variant_config)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
