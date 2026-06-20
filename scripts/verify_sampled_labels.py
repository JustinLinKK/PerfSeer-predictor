#!/usr/bin/env python3
"""Verify sampled label JSONL files contain required dataset fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PHASES = ("train", "infer")
NUMERIC_FIELDS = (
    "time_1_epoch_ms",
    "avg_device_memory_usage_mib",
    "peak_device_memory_usage_mib",
    "compile_time_ms",
    "warmup_time_ms",
    "compile_warmup_time_ms",
    "avg_sm_occupancy_percent",
    "peak_sm_occupancy_percent",
    "avg_sm_utilization_percent",
    "peak_sm_utilization_percent",
    "dram_activity_percent",
    "peak_dram_activity_percent",
)
POSITIVE_NUMERIC_FIELDS = (
    "avg_host_memory_usage_mib",
    "peak_host_memory_usage_mib",
)
OPTIONAL_NUMERIC_FIELDS = (
    "compile_warmup_time_ms",
)
SM_OCCUPANCY_PROXY_SOURCES = {"nvml_utilization_proxy"}


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def verify_phase(row: dict[str, Any], phase: str) -> list[str]:
    errors: list[str] = []
    labels = row.get("label_v2", {}).get(phase)
    if not isinstance(labels, dict):
        return [f"missing label_v2.{phase}"]
    for field in NUMERIC_FIELDS:
        if not finite_nonnegative(labels.get(field)):
            errors.append(f"invalid {phase}.{field}")
    for field in POSITIVE_NUMERIC_FIELDS:
        if not finite_positive(labels.get(field)):
            errors.append(f"invalid {phase}.{field}")
    for field in OPTIONAL_NUMERIC_FIELDS:
        value = labels.get(field)
        if value is not None and not finite_nonnegative(value):
            errors.append(f"invalid {phase}.{field}")
    dtype = labels.get("data_type")
    if (
        not isinstance(dtype, dict)
        or not dtype.get("input_dtype")
        or not isinstance(dtype.get("parameter_dtypes"), list)
        or not dtype.get("forward_input_dtype")
        or not isinstance(dtype.get("forward_parameter_dtypes"), list)
        or not isinstance(dtype.get("backward_gradient_dtypes"), list)
    ):
        errors.append(f"invalid {phase}.data_type")
    source = labels.get("sm_occupancy_source")
    if not source:
        errors.append(f"missing {phase}.sm_occupancy_source")
    elif not (str(source).startswith("ncu_") or source in SM_OCCUPANCY_PROXY_SOURCES):
        errors.append(f"invalid {phase}.sm_occupancy_source")
    kernel_count = labels.get("sm_occupancy_kernel_count")
    if not isinstance(kernel_count, int):
        errors.append(f"invalid {phase}.sm_occupancy_kernel_count")
    elif str(source).startswith("ncu_") and kernel_count <= 0:
        errors.append(f"invalid {phase}.sm_occupancy_kernel_count")
    elif source in SM_OCCUPANCY_PROXY_SOURCES and kernel_count != 0:
        errors.append(f"invalid {phase}.sm_occupancy_kernel_count")
    if phase == "train" and not labels.get("optimizer"):
        errors.append("missing train.optimizer")
    return errors


def verify_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("status") != "ok":
        return [f"row status is {row.get('status')}: {row.get('error', '<no error>')}"]
    for field in ("model_id", "input_shape", "batch_size", "model_file", "hardware"):
        if field not in row:
            errors.append(f"missing {field}")
    hardware = row.get("hardware", {})
    if not isinstance(hardware, dict) or "cuda_version" not in hardware or "device" not in hardware:
        errors.append("invalid hardware metadata")
    if hardware.get("cuda_available") is not True:
        errors.append("cuda was not available")
    if not hardware.get("gpu_name"):
        errors.append("missing gpu_name")
    for phase in PHASES:
        errors.extend(verify_phase(row, phase))
    return errors


def iter_rows(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify sampled label outputs.")
    parser.add_argument("paths", nargs="+", help="Result JSONL files or directories.")
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("results_shard*.jsonl")))
        else:
            files.append(path)
    checked = 0
    ok_rows = 0
    bad_rows = 0
    errors_out: list[str] = []
    for path, line_number, row in iter_rows(files):
        checked += 1
        errors = verify_row(row)
        if row.get("status") == "ok":
            ok_rows += 1
        if errors:
            bad_rows += 1
            if len(errors_out) < args.max_errors:
                errors_out.append(f"{path}:{line_number}: {row.get('model_id', '<unknown>')}: {', '.join(errors)}")
    summary = {"files": len(files), "rows": checked, "ok_rows": ok_rows, "bad_rows": bad_rows}
    print(json.dumps(summary, sort_keys=True))
    for error in errors_out:
        print(error)
    if not files or checked == 0:
        raise SystemExit("no result rows found")
    if bad_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
