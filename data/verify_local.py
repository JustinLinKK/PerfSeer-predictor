#!/usr/bin/env python3
"""Recompute local metrics independently from raw events and NVML samples."""
import argparse
import json
import math
from pathlib import Path
import statistics
from generate import digest, sha


def same(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"metric differs from observations: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.output / "run.json").read_text())
    if sha(args.generated / "workloads.jsonl") != run["generated_manifest_sha256"]:
        raise ValueError("manifest changed")
    manifest = {r["profile_point_id"]: r for r in map(json.loads, (args.generated / "workloads.jsonl").open())}
    labels, outcomes = [], []
    for path in sorted((args.output / "records").glob("*.json")):
        row = json.loads(path.read_text())
        if row["run_sha256"] != digest(run) or row["hardware"] != run["hardware"]:
            raise ValueError("hardware or protocol binding mismatch")
        generated = manifest[row["profile_point_id"]]
        if row["workload_sha256"] != generated["workload_sha256"]:
            raise ValueError("wrong workload")
        outcomes.append(row["status"])
        if row["status"] != "ok":
            if "targets" in row:
                raise ValueError("failed measurement exposed targets")
            continue
        if row["source_sha256"] != generated["variant_source_sha256"]:
            raise ValueError("executed source differs from directly generated source")
        t, protocol, samples = row["targets"], row["protocol"], row["resource_samples"]
        if len(samples) < protocol["resource_min_samples"]:
            raise ValueError("insufficient NVML sample count")
        if row["runtime"]["cuda_uuid"].lower().removeprefix("gpu-") != run["hardware"]["gpu_uuid"].lower().removeprefix("gpu-"):
            raise ValueError("CUDA/NVML UUIDs differ")
        for domain in ("wall", "gpu"):
            epochs = [e[domain + "_ms"] for repeat in row["repeats"] for e in repeat["epochs"]]
            if len(epochs) != protocol["repeats"] * protocol["measured_epochs"]:
                raise ValueError("wrong measured epoch count")
            same(t[f"train_epoch_{domain}_ms"], statistics.fmean(epochs))
            same(t[f"train_step_{domain}_ms"], statistics.fmean(epochs) / protocol["steps_per_epoch"])
        same(t["train_avg_nvml_busy_percent"], statistics.fmean(x["busy_percent"] for x in samples))
        same(t["train_peak_nvml_busy_percent"], max(x["busy_percent"] for x in samples))
        for name, reduce in (("avg", statistics.fmean), ("peak", max)):
            raw = reduce(x["device_memory_bytes"] for x in samples)
            same(t[f"raw_device_{name}_vram_bytes"], raw)
            same(t[f"train_{name}_vram_bytes"], raw - row["baseline_before"]["memory_bytes"])
            same(t[f"train_{name}_vram_mib"], t[f"train_{name}_vram_bytes"] / 1048576)
        if abs(row["baseline_drift_bytes"]) > protocol["max_baseline_drift_mib"] * 1048576:
            raise ValueError("drifting baseline accepted")
        labels.append({"id": row["profile_point_id"], "family": generated["family_id"], "optimizer": generated["optimizer"],
                       "precision": generated["precision_config"], "targets": t, "nvml_samples": len(samples), "record_sha256": sha(path)})
    result = {"passed": True, "hardware": run["hardware"], "records": len(outcomes), "accepted_labels": len(labels),
              "all_metrics_recomputed_from_observations": True, "source_and_gpu_identity_verified": True,
              "not_A10_A100_A10G_measurements": True, "outcomes": outcomes, "labels": labels,
              "run_sha256": digest(run)}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "labels"}))


if __name__ == "__main__":
    main()
