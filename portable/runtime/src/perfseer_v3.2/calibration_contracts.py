"""Versioned identities and native observations for two-objective calibration."""

from collections import Counter, defaultdict
import math

import numpy as np

from perfseer_v31.io import fingerprint

from .version import TARGET_NAMES

CONTRACT_VERSION = "perfseer_v32_time_memory_observation_v1"
ENVIRONMENT_VERSION = "perfseer_v32_calibration_environment_v1"
TARGET_SCHEMA = {
    "time_ms": {"api_name": "train_step_wall_ms", "unit": "ms", "scope": "end_to_end_step",
                "statistic": "arithmetic_mean_of_run_means"},
    "memory_bytes": {"api_name": "train_peak_vram_mib", "unit": "bytes", "scope": "device_nvml",
                     "statistic": "arithmetic_mean_of_run_peaks"},
}
MIB = 1024 ** 2
ENVIRONMENT_FIELDS = ("hardware_id", "gpu_model", "capacity_bytes", "partition", "driver", "cuda",
                      "framework", "libraries", "allocator", "backend_policy", "host_context")


def environment_identity(environment):
    if environment.get("version") != ENVIRONMENT_VERSION or any(key not in environment for key in ENVIRONMENT_FIELDS):
        raise ValueError("incomplete calibration environment; declare unavailable fields as unknown")
    if not environment["hardware_id"] or not environment["gpu_model"] or environment["capacity_bytes"] <= 0:
        raise ValueError("invalid calibration hardware identity")
    return fingerprint(environment)


def workload_identity(source_sha256, executed_inputs, training, semantic_configuration):
    required = ("microbatch_size", "precision", "optimizer", "gradient_accumulation_steps", "backend")
    if len(source_sha256) != 64 or any(key not in training for key in required) or not executed_inputs:
        raise ValueError("incomplete executed workload identity")
    for field in ("microbatch_size", "gradient_accumulation_steps"):
        if type(training[field]) is not int or training[field] < 1:
            raise ValueError("batch and accumulation must be positive integers")
    if not semantic_configuration or any(not isinstance(item, dict) or not item.get("shape") or not item.get("dtype") for item in executed_inputs):
        raise ValueError("actual shapes, dtypes, and semantic configuration are required")
    return fingerprint({"source_sha256": source_sha256, "executed_inputs": executed_inputs,
                        "training": training, "semantics": semantic_configuration})


def primary_prediction(prediction):
    if set(prediction) != set(TARGET_NAMES):
        raise ValueError("source prediction must explicitly name all twelve outputs")
    if not all(math.isfinite(float(value)) for value in prediction.values()):
        raise ValueError("nonfinite source prediction")
    time, memory = float(prediction["train_step_wall_ms"]), float(prediction["train_peak_vram_mib"]) * MIB
    if time <= 0 or memory < 0:
        raise ValueError("source time must be positive and memory nonnegative")
    return {"time_ms": time, "memory_bytes": memory}


def measurement(values, run_ids, *, target, protocol, reason=None):
    """Aggregate independent native runs in physical units, before taking logs."""
    if target not in TARGET_SCHEMA or len(values) != len(run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("invalid measurement target or duplicate run identity")
    required = ("warmup", "boundaries", "step_unit", "data_loading", "source")
    if any(key not in protocol for key in required):
        raise ValueError("measurement protocol is incomplete")
    valid = bool(values) and all(value is not None and math.isfinite(float(value)) and
                                 (value > 0 if target == "time_ms" else value >= 0) for value in values)
    if not valid and not reason:
        reason = "missing_or_nonfinite_or_out_of_range_measurement"
    return {**TARGET_SCHEMA[target], "valid": valid, "reason": None if valid else reason,
            "value": float(np.mean(values)) if valid else None,
            "std": float(np.std(values)) if valid else None,
            "values": list(values) if valid else [], "run_ids": list(run_ids), "run_count": len(run_ids),
            "protocol": protocol}


def validate_record(row):
    if row.get("version") != CONTRACT_VERSION or row.get("split") not in {"train", "validation", "test", "uncertainty"}:
        raise ValueError("invalid observation version or split")
    for key in ("sample_id", "group_id", "anchor_id", "workload_id", "domain_fingerprint", "source_identity"):
        if not row.get(key):
            raise ValueError(f"missing observation {key}")
    primary_prediction(row["source_prediction"])
    if row["domain_fingerprint"] != environment_identity(row["environment"]):
        raise ValueError("observation environment fingerprint differs")
    for target, schema in TARGET_SCHEMA.items():
        item = row["measurements"][target]
        if any(item.get(key) != value for key, value in schema.items()):
            raise ValueError("measurement units, scope, or aggregation differs")
        if type(item.get("valid")) is not bool:
            raise ValueError("measurement requires an explicit validity mask")
        if item["valid"]:
            expected = measurement(item["values"], item["run_ids"], target=target, protocol=item["protocol"])
            if expected != item:
                raise ValueError("measurement aggregate or repeat provenance differs")
        elif item.get("value") is not None or not item.get("reason"):
            raise ValueError("invalid observations require a mask reason and null value")
    if row.get("status", "ok") != "ok" and any(item["valid"] for item in row["measurements"].values()):
        raise ValueError("OOM and failed runs are feasibility outcomes, not numerical labels")


def audit_records(rows):
    """Fail closed on aliases, workload/group leakage, and conflicting run reuse."""
    ids, identities, runs, source_variants = {}, {}, {}, defaultdict(set)
    invalid, alias_count = Counter(), 0
    for row in rows:
        validate_record(row)
        if row["sample_id"] in ids:
            raise ValueError("duplicate sample identity")
        ids[row["sample_id"]] = row
        for key in ("group_id", "anchor_id", "workload_id"):
            identity = (key, row[key])
            if identity in identities and identities[identity] != row["split"]:
                raise ValueError(f"split leakage through {key}")
            identities[identity] = row["split"]
        source_variants[row.get("source_sha256", "unknown")].add(row["workload_id"])
        for target, item in row["measurements"].items():
            if not item["valid"]:
                invalid[f"{target}:{item['reason']}"] += 1
            for run_id in item["run_ids"]:
                identity = (target, run_id)
                value = (row["workload_id"], row["split"], fingerprint(item), row["domain_fingerprint"])
                if identity in runs and runs[identity] != value:
                    raise ValueError("measurement copied across workloads, domains, or splits")
                runs[identity] = value
    canonical = []
    seen_workloads = set()
    for row in rows:
        if row.get("alias_of"):
            donor = ids.get(row["alias_of"])
            if donor is None or donor.get("alias_of") or any(row[key] != donor[key] for key in
                    ("workload_id", "group_id", "anchor_id", "split", "measurements", "source_identity",
                     "source_prediction", "features", "domain_fingerprint")):
                raise ValueError("alias must reference the same independent measurement and workload")
            alias_count += 1
            continue
        key = (row["domain_fingerprint"], row["workload_id"])
        if key in seen_workloads:
            raise ValueError("aggregate repeats or declare an alias; duplicate workload is not independent")
        seen_workloads.add(key)
        canonical.append(row)
    return {"records": len(rows), "unique_configurations": len(canonical), "aliases": alias_count,
            "independent_groups": len({row["group_id"] for row in canonical}),
            "real_gpu_runs": len({run_id for _, run_id in runs}), "invalid_targets": dict(invalid),
            "source_hashes_with_multiple_workloads": sum(len(value) > 1 for value in source_variants.values()),
            "splits": dict(Counter(row["split"] for row in canonical)),
            "split_fingerprint": fingerprint(sorted((row["sample_id"], row["workload_id"], row["group_id"], row["split"])
                                                     for row in canonical))}


def schedule_duration(step_ms, *, dataset_size, microbatch, accumulation=1, drop_last=False,
                      step_unit="optimizer_update", epochs=1, startup_ms=0.):
    """Only for a declared fixed-shape schedule with epoch = N * mean step."""
    if any(type(value) is not int or value < 1 for value in (dataset_size, microbatch, accumulation, epochs)):
        raise ValueError("schedule counts must be positive integers")
    if not math.isfinite(step_ms) or step_ms <= 0 or not math.isfinite(startup_ms) or startup_ms < 0:
        raise ValueError("invalid schedule duration")
    if step_unit not in {"microstep", "optimizer_update"}:
        raise ValueError("unknown step definition")
    if not drop_last and dataset_size % microbatch:
        raise ValueError("variable final batch requires step-shape classes")
    microsteps = dataset_size // microbatch
    if step_unit == "optimizer_update" and microsteps % accumulation:
        raise ValueError("partial accumulation requires step-shape classes")
    steps = microsteps if step_unit == "microstep" else microsteps // accumulation
    return {"steps_per_epoch": steps, "epoch_ms": steps * step_ms,
            "training_ms": epochs * steps * step_ms + startup_ms}
