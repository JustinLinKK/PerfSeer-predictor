"""Scheduler workload and label schemas for real-dataset profiling."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


WORKLOAD_SPEC_VERSION = 1
SCHEDULER_LABEL_VERSION = 3
SCHEDULER_RESOURCE_LABEL_VERSION = 2

SCHEDULER_TARGET_NAMES: tuple[str, ...] = (
    "train_step_wall_ms",
    "train_step_gpu_ms",
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_peak_vram_mib",
    "train_peak_torch_allocated_mib",
    "infer_step_wall_ms",
    "infer_step_gpu_ms",
    "infer_avg_sm_util_percent",
    "infer_peak_vram_mib",
)

SCHEDULER_RESOURCE_TARGET_NAMES: tuple[str, ...] = (
    "train_step_wall_ms",
    "train_step_gpu_ms",
    "train_phase_wall_ms",
    "train_phase_gpu_ms",
    "train_avg_sm_util_percent",
    "train_sm_util_std_percent",
    "train_p50_sm_util_percent",
    "train_p95_sm_util_percent",
    "train_peak_sm_util_percent",
    "train_avg_memory_controller_util_percent",
    "train_memory_controller_util_std_percent",
    "train_p50_memory_controller_util_percent",
    "train_p95_memory_controller_util_percent",
    "train_peak_memory_controller_util_percent",
    "train_avg_vram_used_mib",
    "train_vram_used_std_mib",
    "train_p50_vram_used_mib",
    "train_p95_vram_used_mib",
    "train_peak_vram_used_mib",
    "train_peak_torch_allocated_mib",
    "train_peak_torch_reserved_mib",
    "train_measurement_duration_ms",
    "train_sampler_samples",
    "infer_step_wall_ms",
    "infer_step_gpu_ms",
    "infer_phase_wall_ms",
    "infer_phase_gpu_ms",
    "infer_avg_sm_util_percent",
    "infer_sm_util_std_percent",
    "infer_p50_sm_util_percent",
    "infer_p95_sm_util_percent",
    "infer_peak_sm_util_percent",
    "infer_avg_memory_controller_util_percent",
    "infer_memory_controller_util_std_percent",
    "infer_p50_memory_controller_util_percent",
    "infer_p95_memory_controller_util_percent",
    "infer_peak_memory_controller_util_percent",
    "infer_avg_vram_used_mib",
    "infer_vram_used_std_mib",
    "infer_p50_vram_used_mib",
    "infer_p95_vram_used_mib",
    "infer_peak_vram_used_mib",
    "infer_peak_torch_allocated_mib",
    "infer_peak_torch_reserved_mib",
    "infer_measurement_duration_ms",
    "infer_sampler_samples",
)


def clean_id(value: Any, default: str = "unknown") -> str:
    raw = str(value or default).strip().lower()
    raw = re.sub(r"[^a-z0-9_.+-]+", "_", raw)
    raw = raw.strip("_")
    return raw or default


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _require_mapping(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"WorkloadSpec requires object field {key!r}")
    return value


def _positive_int(value: Any, field: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


def effective_batch_size(training: dict[str, Any]) -> int:
    batch = _positive_int(training.get("batch_size", 1), "training.batch_size")
    accum = _positive_int(training.get("grad_accumulation_steps", 1), "training.grad_accumulation_steps")
    return batch * accum


def workload_profile_point_id(spec: dict[str, Any]) -> str:
    model = _require_mapping(spec, "model")
    dataset = _require_mapping(spec, "dataset")
    training = _require_mapping(spec, "training")
    hardware = spec.get("hardware") if isinstance(spec.get("hardware"), dict) else {}
    return "::".join(
        [
            clean_id(model.get("model_id") or model.get("source_path")),
            clean_id(dataset.get("dataset_id")),
            clean_id(dataset.get("subset_id")),
            f"bs{effective_batch_size(training)}",
            clean_id(training.get("optimizer", "adam")),
            clean_id(training.get("precision", "fp32_ieee")),
            clean_id(hardware.get("hardware_id", "unknown")),
        ]
    )


def normalize_workload_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["workload_spec_version"] = int(out.get("workload_spec_version", WORKLOAD_SPEC_VERSION))
    if out["workload_spec_version"] != WORKLOAD_SPEC_VERSION:
        raise ValueError(f"unsupported workload_spec_version={out['workload_spec_version']}")
    model = dict(_require_mapping(out, "model"))
    dataset = dict(_require_mapping(out, "dataset"))
    training = dict(_require_mapping(out, "training"))
    hardware = dict(out.get("hardware") or {})
    if not model.get("model_id"):
        raise ValueError("WorkloadSpec model.model_id is required")
    if not model.get("source_path"):
        raise ValueError("WorkloadSpec model.source_path is required")
    if not dataset.get("dataset_id"):
        raise ValueError("WorkloadSpec dataset.dataset_id is required")
    if not dataset.get("subset_id"):
        raise ValueError("WorkloadSpec dataset.subset_id is required")
    dataset["num_samples"] = _positive_int(dataset.get("num_samples", dataset.get("sample_count", 1)), "dataset.num_samples")
    training["batch_size"] = _positive_int(training.get("batch_size", 1), "training.batch_size")
    training["grad_accumulation_steps"] = _positive_int(
        training.get("grad_accumulation_steps", 1),
        "training.grad_accumulation_steps",
    )
    training["optimizer"] = clean_id(training.get("optimizer", "adam"))
    training["precision"] = clean_id(training.get("precision", "fp32_ieee"))
    if hardware.get("hardware_id"):
        hardware["hardware_id"] = clean_id(hardware.get("hardware_id"))
    out["model"] = model
    out["dataset"] = dataset
    out["training"] = training
    out["hardware"] = hardware
    out["profile_point_id"] = str(out.get("profile_point_id") or workload_profile_point_id(out))
    out["workload_hash"] = stable_hash(
        {
            "model": model,
            "dataset": dataset,
            "training": training,
            "hardware": hardware,
        }
    )
    return out


def manifest_row_from_workload(spec: dict[str, Any]) -> dict[str, Any]:
    spec = normalize_workload_spec(spec)
    model = spec["model"]
    dataset = spec["dataset"]
    training = spec["training"]
    model_id = str(model["model_id"])
    precision = str(training.get("precision", "fp32_ieee"))
    return {
        "model_id": model_id,
        "graph_id": model.get("graph_id", model_id),
        "model_file": model["source_path"],
        "input_shape": model.get("input_shape", dataset.get("input_shape", [training["batch_size"], 1])),
        "input_specs": model.get("input_specs", dataset.get("input_specs", [])),
        "architecture_family": model.get("architecture_family", "unknown"),
        "precision_config": precision,
        "profile_point_id": spec["profile_point_id"],
        "label_file": f"label/label/{clean_id(spec['profile_point_id'])}.txt",
        "workload_spec": spec,
    }


def _phase_metric(detail: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(detail.get(key, default))
    except (TypeError, ValueError):
        return default


def _sampler_metric(detail: dict[str, Any], key: str, default: float = 0.0) -> float:
    sampler = detail.get("sampler") if isinstance(detail.get("sampler"), dict) else {}
    try:
        return float(sampler.get(key, default))
    except (TypeError, ValueError):
        return default


def label_v3_from_result(result: dict[str, Any]) -> dict[str, Any]:
    workload = result.get("workload_spec") if isinstance(result.get("workload_spec"), dict) else {}
    dataset = workload.get("dataset") if isinstance(workload.get("dataset"), dict) else {}
    training = workload.get("training") if isinstance(workload.get("training"), dict) else {}
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    train = details.get("train") if isinstance(details.get("train"), dict) else {}
    infer = details.get("infer") if isinstance(details.get("infer"), dict) else {}
    batch = int(result.get("batch_size") or training.get("batch_size") or 1)
    grad_accum = int(training.get("grad_accumulation_steps") or 1)
    effective_batch = max(batch * grad_accum, 1)
    num_samples = int(dataset.get("num_samples") or dataset.get("sample_count") or result.get("train_samples") or batch)
    steps_per_epoch = int(math.ceil(num_samples / effective_batch))
    train_wall_ms = _phase_metric(train, "mean_wall_iter_ms", _phase_metric(train, "mean_iter_ms"))
    infer_wall_ms = _phase_metric(infer, "mean_wall_iter_ms", _phase_metric(infer, "mean_iter_ms"))
    step_extrapolated_epoch_ms = train_wall_ms * steps_per_epoch
    measured_epoch_ms = _phase_metric(train, "measured_epoch_wall_mean_ms", 0.0)
    epoch_time_source = str(train.get("epoch_time_source") or "step_extrapolated")
    if epoch_time_source == "measured_epochs" and measured_epoch_ms > 0:
        train_epoch_ms = measured_epoch_ms
    else:
        epoch_time_source = "step_extrapolated"
        train_epoch_ms = step_extrapolated_epoch_ms
    targets = {
        "train_step_wall_ms": train_wall_ms,
        "train_step_gpu_ms": _phase_metric(train, "mean_iter_ms", train_wall_ms),
        "train_epoch_ms": train_epoch_ms,
        "train_epoch_ms_step_extrapolated": step_extrapolated_epoch_ms,
        "train_avg_sm_util_percent": _sampler_metric(train, "avg_sm_util"),
        "train_peak_vram_mib": _sampler_metric(train, "peak_mem_usage"),
        "train_peak_torch_allocated_mib": _phase_metric(train, "peak_torch_allocated_mib"),
        "infer_step_wall_ms": infer_wall_ms,
        "infer_step_gpu_ms": _phase_metric(infer, "mean_iter_ms", infer_wall_ms),
        "infer_avg_sm_util_percent": _sampler_metric(infer, "avg_sm_util"),
        "infer_peak_vram_mib": _sampler_metric(infer, "peak_mem_usage"),
    }
    return {
        "scheduler_label_version": SCHEDULER_LABEL_VERSION,
        "profile_point_id": result.get("profile_point_id"),
        "model_id": result.get("model_id"),
        "status": result.get("status"),
        "targets": targets,
        "target_names": list(SCHEDULER_TARGET_NAMES),
        "dataset": dataset,
        "training": {
            "batch_size": batch,
            "grad_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch,
            "optimizer": training.get("optimizer") or result.get("optimizer"),
            "precision": training.get("precision") or result.get("precision_config"),
            "steps_per_epoch": steps_per_epoch,
            "epoch_time_source": epoch_time_source,
            "warmup_epochs": int(train.get("warmup_epochs", 0) or 0),
            "measured_epochs": int(train.get("measured_epochs", 0) or 0),
        },
        "hardware_id": result.get("hardware_id"),
        "precision_config": result.get("precision_config"),
        "provenance": {
            "label_source": "profiler_result",
            "epoch_time_source": epoch_time_source,
            "real_dataloader_backed": bool(dataset.get("real_dataloader_backed", False)),
            "metadata_source": dataset.get("metadata_source"),
        },
    }


def scheduler_target_vector(label_v3: dict[str, Any]) -> list[float]:
    targets = label_v3.get("targets") if isinstance(label_v3.get("targets"), dict) else {}
    return [float(targets.get(name, 0.0) or 0.0) for name in SCHEDULER_TARGET_NAMES]


def _resource_phase_targets(prefix: str, detail: dict[str, Any]) -> dict[str, float]:
    sampler = detail.get("sampler") if isinstance(detail.get("sampler"), dict) else {}
    mean_wall = _phase_metric(detail, "mean_wall_iter_ms", _phase_metric(detail, "mean_iter_ms"))
    mean_gpu = _phase_metric(detail, "mean_iter_ms", mean_wall)
    steps = max(_phase_metric(detail, "measurement_steps", 1.0), 1.0)
    return {
        f"{prefix}_step_wall_ms": mean_wall,
        f"{prefix}_step_gpu_ms": mean_gpu,
        f"{prefix}_phase_wall_ms": _phase_metric(detail, "total_wall_ms", mean_wall * steps),
        f"{prefix}_phase_gpu_ms": _phase_metric(detail, "total_gpu_ms", mean_gpu * steps),
        f"{prefix}_avg_sm_util_percent": _sampler_metric(detail, "avg_sm_util"),
        f"{prefix}_sm_util_std_percent": _sampler_metric(detail, "sm_util_std"),
        f"{prefix}_p50_sm_util_percent": _sampler_metric(detail, "p50_sm_util"),
        f"{prefix}_p95_sm_util_percent": _sampler_metric(detail, "p95_sm_util"),
        f"{prefix}_peak_sm_util_percent": _sampler_metric(detail, "peak_sm_util"),
        f"{prefix}_avg_memory_controller_util_percent": _sampler_metric(detail, "avg_mem_util"),
        f"{prefix}_memory_controller_util_std_percent": _sampler_metric(detail, "mem_util_std"),
        f"{prefix}_p50_memory_controller_util_percent": _sampler_metric(detail, "p50_mem_util"),
        f"{prefix}_p95_memory_controller_util_percent": _sampler_metric(detail, "p95_mem_util"),
        f"{prefix}_peak_memory_controller_util_percent": _sampler_metric(detail, "peak_mem_util"),
        f"{prefix}_avg_vram_used_mib": _sampler_metric(detail, "avg_mem_usage"),
        f"{prefix}_vram_used_std_mib": _sampler_metric(detail, "mem_usage_std"),
        f"{prefix}_p50_vram_used_mib": _sampler_metric(detail, "p50_mem_usage"),
        f"{prefix}_p95_vram_used_mib": _sampler_metric(detail, "p95_mem_usage"),
        f"{prefix}_peak_vram_used_mib": _sampler_metric(detail, "peak_mem_usage"),
        f"{prefix}_peak_torch_allocated_mib": _phase_metric(detail, "peak_torch_allocated_mib"),
        f"{prefix}_peak_torch_reserved_mib": _phase_metric(detail, "peak_torch_reserved_mib"),
        f"{prefix}_measurement_duration_ms": _sampler_metric(detail, "measurement_duration_ms"),
        f"{prefix}_sampler_samples": _sampler_metric(detail, "sample_count"),
    }


def _resource_phase_quality(detail: dict[str, Any]) -> dict[str, Any]:
    sampler = detail.get("sampler") if isinstance(detail.get("sampler"), dict) else {}
    return {
        "measurement_steps": int(detail.get("measurement_steps", 0) or 0),
        "measurement_duration_ms": _sampler_metric(detail, "measurement_duration_ms"),
        "sampler_samples": int(sampler.get("sample_count", 0) or 0),
        "reliability_flags": list(sampler.get("reliability_flags", []) or []),
        "telemetry_source": sampler.get("source", "none"),
        "epoch_time_source": detail.get("epoch_time_source", "step_extrapolated"),
        "warmup_epochs": int(detail.get("warmup_epochs", 0) or 0),
        "measured_epochs": int(detail.get("measured_epochs", 0) or 0),
        "steps_per_epoch": int(detail.get("steps_per_epoch", 0) or 0),
        "epoch_wall_ms": list(detail.get("epoch_wall_ms", []) or []),
        "epoch_gpu_ms": list(detail.get("epoch_gpu_ms", []) or []),
        "measured_epoch_wall_mean_ms": _phase_metric(detail, "measured_epoch_wall_mean_ms"),
        "measured_epoch_wall_std_ms": _phase_metric(detail, "measured_epoch_wall_std_ms"),
    }


def scheduler_resource_label_from_result(result: dict[str, Any]) -> dict[str, Any]:
    workload = result.get("workload_spec") if isinstance(result.get("workload_spec"), dict) else {}
    dataset = workload.get("dataset") if isinstance(workload.get("dataset"), dict) else {}
    training = workload.get("training") if isinstance(workload.get("training"), dict) else {}
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    train = details.get("train") if isinstance(details.get("train"), dict) else {}
    infer = details.get("infer") if isinstance(details.get("infer"), dict) else {}
    targets: dict[str, float] = {}
    targets.update(_resource_phase_targets("train", train))
    targets.update(_resource_phase_targets("infer", infer))
    profile_dataset = result.get("profile_dataset") if isinstance(result.get("profile_dataset"), dict) else {}
    return {
        "scheduler_resource_label_version": SCHEDULER_RESOURCE_LABEL_VERSION,
        "profile_point_id": result.get("profile_point_id"),
        "model_id": result.get("model_id"),
        "status": result.get("status"),
        "targets": targets,
        "target_names": list(SCHEDULER_RESOURCE_TARGET_NAMES),
        "dataset": dataset,
        "training": {
            "batch_size": int(result.get("batch_size") or training.get("batch_size") or 1),
            "grad_accumulation_steps": int(training.get("grad_accumulation_steps") or 1),
            "optimizer": training.get("optimizer") or result.get("optimizer"),
            "precision": training.get("precision") or result.get("precision_config"),
            "num_workers": training.get("num_workers"),
            "prefetch_factor": training.get("prefetch_factor"),
        },
        "hardware_id": result.get("hardware_id"),
        "precision_config": result.get("precision_config"),
        "resource_profile_mode": result.get("resource_profile_mode"),
        "resource_quality": {
            "train": _resource_phase_quality(train),
            "infer": _resource_phase_quality(infer),
        },
        "resource_metric_sources": {
            "sm_utilization": "nvml_utilization_rates.gpu",
            "memory_controller_utilization": "nvml_utilization_rates.memory",
            "vram_used": "nvml_memory_info.used",
            "torch_allocated": "torch.cuda.max_memory_allocated",
            "torch_reserved": "torch.cuda.max_memory_reserved",
        },
        "batch_consumption": {
            "adapter": profile_dataset.get("adapter"),
            "real_dataloader_backed": profile_dataset.get("real_dataloader_backed"),
            "measurement_batches": profile_dataset.get("measurement_batches"),
            "generated_batches": profile_dataset.get("generated_batches"),
            "warmup_batches_per_phase": profile_dataset.get("warmup_batches_per_phase"),
            "sample_keys_consumed": profile_dataset.get("sample_keys_consumed"),
            "sample_fingerprints": profile_dataset.get("sample_fingerprints", []),
        },
    }
