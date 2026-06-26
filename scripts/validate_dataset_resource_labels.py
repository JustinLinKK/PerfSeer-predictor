#!/usr/bin/env python
"""Validate dataset-backed resource labels against 5-epoch golden profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nrp_calibration_pack.workload import clean_id, write_jsonl  # noqa: E402
from perfseer.architecture_schema import ARCHITECTURE_FAMILIES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 20 dataset-backed labels against 5-epoch golden profiles.")
    parser.add_argument("--workloads", default=str(ROOT / "nrp_calibration_pack" / "workload_specs_balanced_local" / "workloads.jsonl"))
    parser.add_argument("--models-dir", default=str(ROOT / "nrp_calibration_pack" / "models"))
    parser.add_argument("--output-dir", help="Defaults to record/resource_label_validation_<timestamp>.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hardware-id", default="rtx5090")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--infer-repeats", type=int, default=50)
    parser.add_argument("--train-repeats", type=int, default=50)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--min-phase-seconds", type=float, default=20.0)
    parser.add_argument("--min-sampler-samples", type=int, default=100)
    parser.add_argument("--sm-occupancy-source", default="nvml_proxy", choices=("ncu", "nvml_proxy"))
    parser.add_argument("--resource-profile-mode", default="sustained", choices=("compat", "sustained"))
    parser.add_argument("--selection-mode", default="deterministic", choices=("deterministic", "random"))
    parser.add_argument("--selection-seed", type=int, default=20260625, help="Seed used when --selection-mode=random.")
    parser.add_argument("--label-time-mode", default="measured_epochs", choices=("step_extrapolated", "measured_epochs"))
    parser.add_argument("--time-label-warmup-epochs", type=int, default=1)
    parser.add_argument("--time-label-measured-epochs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Only select and write the validation workload subset.")
    parser.add_argument("--report-only", action="store_true", help="Write comparison outputs without failing on thresholds.")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def architecture_family(row: dict[str, Any]) -> str:
    model = row.get("model") if isinstance(row.get("model"), dict) else {}
    return str(model.get("architecture_family") or row.get("architecture_family") or "unknown")


def _group_workloads(rows: Iterable[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict((family, []) for family in ARCHITECTURE_FAMILIES)
    for row in rows:
        family = architecture_family(row)
        grouped.setdefault(family, [])
        grouped[family].append(row)
    return grouped


def select_validation_workloads(
    rows: Iterable[dict[str, Any]],
    limit: int = 20,
    *,
    mode: str = "deterministic",
    seed: int = 20260625,
) -> list[dict[str, Any]]:
    grouped = _group_workloads(rows)
    if mode not in {"deterministic", "random"}:
        raise ValueError(f"unknown selection mode: {mode}")
    family_order = list(grouped)
    if mode == "random":
        rng = random.Random(seed)
        for items in grouped.values():
            rng.shuffle(items)
        known = [family for family in ARCHITECTURE_FAMILIES if family in grouped]
        unknown = [family for family in family_order if family not in ARCHITECTURE_FAMILIES]
        rng.shuffle(known)
        rng.shuffle(unknown)
        family_order = known + unknown
    extras: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    for family in family_order:
        items = grouped[family]
        if items and len(selected) < limit:
            selected.append(items[0])
            extras[family] = items[1:]
    while len(selected) < limit:
        added = False
        for family in family_order:
            bucket = extras.get(family, [])
            if bucket:
                selected.append(bucket.pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def profile_point_id(row: dict[str, Any]) -> str:
    return str(row.get("profile_point_id") or (row.get("workload_spec") or {}).get("profile_point_id") or row["model_id"])


def effective_batch_size(row: dict[str, Any]) -> int:
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    batch = int(training.get("batch_size") or 1)
    accum = int(training.get("grad_accumulation_steps") or 1)
    return max(batch * accum, 1)


def steps_per_epoch(row: dict[str, Any]) -> int:
    dataset = row.get("dataset") if isinstance(row.get("dataset"), dict) else {}
    samples = int(dataset.get("num_samples") or dataset.get("sample_count") or 1)
    return max(int(math.ceil(samples / effective_batch_size(row))), 1)


def run_profile_command(args: argparse.Namespace, workload_path: Path, output_dir: Path, *, golden_steps: int | None = None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
        "--workload-specs",
        str(workload_path),
        "--models-dir",
        str(args.models_dir),
        "--output-dir",
        str(output_dir),
        "--hardware-id",
        args.hardware_id,
        "--device",
        args.device,
        "--sm-occupancy-source",
        args.sm_occupancy_source,
        "--resource-profile-mode",
        args.resource_profile_mode,
        "--sample-interval",
        str(args.sample_interval),
        "--no-resume",
    ]
    if golden_steps is None:
        cmd.extend(
            [
                "--warmup",
                str(args.warmup),
                "--infer-repeats",
                str(args.infer_repeats),
                "--train-repeats",
                str(args.train_repeats),
                "--min-phase-seconds",
                str(args.min_phase_seconds),
                "--min-sampler-samples",
                str(args.min_sampler_samples),
                "--label-time-mode",
                args.label_time_mode,
                "--time-label-warmup-epochs",
                str(args.time_label_warmup_epochs),
                "--time-label-measured-epochs",
                str(args.time_label_measured_epochs),
            ]
        )
    else:
        cmd.extend(
            [
                "--warmup",
                "0",
                "--profile-epochs",
                str(args.epochs),
                "--batches-per-epoch",
                str(golden_steps),
                "--min-phase-seconds",
                "0",
                "--min-sampler-samples",
                "0",
                "--label-time-mode",
                "step_extrapolated",
            ]
        )
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": time.time() - start,
        "workload_file": str(workload_path),
        "output_dir": str(output_dir),
    }


def result_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("**/results_shard*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def first_ok_result(output_dir: Path, point_id: str) -> dict[str, Any] | None:
    for row in result_rows(output_dir):
        if str(row.get("profile_point_id")) == point_id and row.get("status") == "ok":
            return row
    return None


def rel_error(predicted: float, observed: float) -> float:
    return abs(predicted - observed) / max(abs(observed), 1e-9)


def compare_rows(selected: list[dict[str, Any]], label_dir: Path, golden_root: Path, epochs: int) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    label_by_id = {str(row.get("profile_point_id")): row for row in result_rows(label_dir)}
    for workload in selected:
        point_id = profile_point_id(workload)
        label_row = label_by_id.get(point_id)
        golden_row = first_ok_result(golden_root / clean_id(point_id), point_id)
        row: dict[str, Any] = {
            "profile_point_id": point_id,
            "architecture_family": architecture_family(workload),
            "label_status": label_row.get("status") if label_row else "missing",
            "golden_status": golden_row.get("status") if golden_row else "missing",
        }
        if not label_row or not golden_row or label_row.get("status") != "ok" or golden_row.get("status") != "ok":
            row.update({"time_pass": False, "vram_pass": False, "sm_pass": False, "overall_pass": False})
            comparisons.append(row)
            continue
        label_v3 = label_row.get("label_v3") if isinstance(label_row.get("label_v3"), dict) else {}
        label_targets = label_v3.get("targets") if isinstance(label_v3.get("targets"), dict) else {}
        label_resource = label_row.get("scheduler_resource_label") if isinstance(label_row.get("scheduler_resource_label"), dict) else {}
        label_resource_targets = label_resource.get("targets") if isinstance(label_resource.get("targets"), dict) else {}
        golden_resource = golden_row.get("scheduler_resource_label") if isinstance(golden_row.get("scheduler_resource_label"), dict) else {}
        golden_resource_targets = golden_resource.get("targets") if isinstance(golden_resource.get("targets"), dict) else {}
        golden_train = (golden_row.get("details") or {}).get("train", {}) if isinstance(golden_row.get("details"), dict) else {}
        predicted_epoch_ms = float(label_targets.get("train_epoch_ms", 0.0) or 0.0)
        golden_epoch_ms = float(golden_train.get("total_wall_ms", 0.0) or 0.0) / max(epochs, 1)
        predicted_peak_vram = float(label_resource_targets.get("train_peak_vram_used_mib", 0.0) or 0.0)
        golden_peak_vram = float(golden_resource_targets.get("train_peak_vram_used_mib", 0.0) or 0.0)
        predicted_reserved = float(label_resource_targets.get("train_peak_torch_reserved_mib", 0.0) or 0.0)
        golden_reserved = float(golden_resource_targets.get("train_peak_torch_reserved_mib", 0.0) or 0.0)
        predicted_avg_sm = float(label_resource_targets.get("train_avg_sm_util_percent", 0.0) or 0.0)
        golden_avg_sm = float(golden_resource_targets.get("train_avg_sm_util_percent", 0.0) or 0.0)
        predicted_p95_sm = float(label_resource_targets.get("train_p95_sm_util_percent", 0.0) or 0.0)
        golden_p95_sm = float(golden_resource_targets.get("train_p95_sm_util_percent", 0.0) or 0.0)
        time_err = rel_error(predicted_epoch_ms, golden_epoch_ms)
        vram_abs = abs(predicted_peak_vram - golden_peak_vram)
        reserved_abs = abs(predicted_reserved - golden_reserved)
        avg_sm_abs = abs(predicted_avg_sm - golden_avg_sm)
        p95_sm_abs = abs(predicted_p95_sm - golden_p95_sm)
        time_pass = time_err <= 0.25
        vram_pass = vram_abs <= max(0.10 * max(golden_peak_vram, 0.0), 512.0)
        reserved_pass = reserved_abs <= max(0.10 * max(golden_reserved, 0.0), 512.0)
        sm_pass = avg_sm_abs <= 15.0 or p95_sm_abs <= 20.0
        row.update(
            {
                "predicted_epoch_ms": predicted_epoch_ms,
                "golden_epoch_ms": golden_epoch_ms,
                "epoch_rel_error": time_err,
                "predicted_peak_vram_mib": predicted_peak_vram,
                "golden_peak_vram_mib": golden_peak_vram,
                "peak_vram_abs_error_mib": vram_abs,
                "predicted_peak_torch_reserved_mib": predicted_reserved,
                "golden_peak_torch_reserved_mib": golden_reserved,
                "torch_reserved_abs_error_mib": reserved_abs,
                "predicted_avg_sm_percent": predicted_avg_sm,
                "golden_avg_sm_percent": golden_avg_sm,
                "avg_sm_abs_error_percent": avg_sm_abs,
                "predicted_p95_sm_percent": predicted_p95_sm,
                "golden_p95_sm_percent": golden_p95_sm,
                "p95_sm_abs_error_percent": p95_sm_abs,
                "time_pass": time_pass,
                "vram_pass": vram_pass,
                "torch_reserved_pass": reserved_pass,
                "sm_pass": sm_pass,
                "overall_pass": time_pass and vram_pass and reserved_pass and sm_pass,
            }
        )
        comparisons.append(row)
    return comparisons


def write_comparisons(rows: list[dict[str, Any]], output_dir: Path, run_status: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    comparison_json = output_dir / "comparison.json"
    comparison_csv = output_dir / "comparison.csv"
    status_path = output_dir / "run_status.json"
    statuses = run_status or []
    comparison_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    status_path.write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with comparison_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    total = len(rows)
    time_pass = sum(1 for row in rows if row.get("time_pass"))
    vram_pass = sum(1 for row in rows if row.get("vram_pass"))
    sm_pass = sum(1 for row in rows if row.get("sm_pass"))
    overall_pass = sum(1 for row in rows if row.get("overall_pass"))
    failed_commands = sum(1 for status in statuses if int(status.get("returncode", 0) or 0) != 0)
    non_ok_rows = sum(1 for row in rows if row.get("label_status") != "ok" or row.get("golden_status") != "ok")
    failed_rows = sum(1 for row in rows if not row.get("overall_pass"))
    time_required = max(1, math.ceil(0.90 * total)) if total else 0
    vram_required = max(1, math.ceil(0.90 * total)) if total else 0
    sm_required = max(1, math.ceil(0.75 * total)) if total else 0
    overall_required = max(1, math.ceil(0.75 * total)) if total else 0
    summary = {
        "rows": total,
        "time_pass": time_pass,
        "vram_pass": vram_pass,
        "sm_pass": sm_pass,
        "overall_pass": overall_pass,
        "failed_rows": failed_rows,
        "non_ok_label_or_golden_rows": non_ok_rows,
        "failed_profile_commands": failed_commands,
        "required_time_pass": time_required,
        "required_vram_pass": vram_required,
        "required_sm_pass": sm_required,
        "required_overall_pass": overall_required,
        "passed_thresholds": (
            failed_commands == 0
            and non_ok_rows == 0
            and time_pass >= time_required
            and vram_pass >= vram_required
            and sm_pass >= sm_required
            and overall_pass >= overall_required
        ),
    }
    lines = [
        "# Resource Label Validation Summary",
        "",
        f"- Rows: {total}",
        f"- Epoch-time pass: {time_pass}/{total}",
        f"- Peak-VRAM pass: {vram_pass}/{total}",
        f"- SM pass: {sm_pass}/{total}",
        f"- Overall pass: {overall_pass}/{total}",
        f"- Failed rows: {failed_rows}/{total}",
        f"- Non-ok label/golden rows: {non_ok_rows}/{total}",
        f"- Failed profiler commands: {failed_commands}",
        f"- Required epoch-time pass: {time_required}/{total}",
        f"- Required peak-VRAM pass: {vram_required}/{total}",
        f"- Required SM pass: {sm_required}/{total}",
        f"- Required overall pass: {overall_required}/{total}",
        f"- Passed thresholds: {summary['passed_thresholds']}",
        "",
        f"- Comparison CSV: `{comparison_csv.name}`",
        f"- Comparison JSON: `{comparison_json.name}`",
        f"- Run status JSON: `{status_path.name}`",
    ]
    failed = [status for status in statuses if int(status.get("returncode", 0) or 0) != 0]
    if failed:
        lines.extend(["", "## Failed Commands", ""])
        for status in failed:
            lines.append(
                f"- {status.get('phase')} `{status.get('profile_point_id')}` exited {status.get('returncode')} "
                f"after {float(status.get('elapsed_sec', 0.0)):.1f}s"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.time_label_warmup_epochs < 0:
        raise SystemExit("--time-label-warmup-epochs must be >= 0")
    if args.time_label_measured_epochs <= 0:
        raise SystemExit("--time-label-measured-epochs must be > 0")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "record" / f"resource_label_validation_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = load_jsonl(Path(args.workloads))
    selected = select_validation_workloads(workloads, args.limit, mode=args.selection_mode, seed=args.selection_seed)
    selected_path = output_dir / "selected_workloads.jsonl"
    write_jsonl(selected_path, selected)
    if args.dry_run:
        families = [architecture_family(row) for row in selected]
        summary = {
            "dry_run": True,
            "selected_workloads": len(selected),
            "selection_mode": args.selection_mode,
            "selection_seed": args.selection_seed,
            "label_time_mode": args.label_time_mode,
            "time_label_warmup_epochs": args.time_label_warmup_epochs,
            "time_label_measured_epochs": args.time_label_measured_epochs,
            "families": families,
            "selected_workloads_file": str(selected_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Resource Label Validation Dry Run",
            "",
            f"- Selected workloads: {len(selected)}",
            f"- Selection mode: {args.selection_mode}",
            f"- Selection seed: {args.selection_seed}",
            f"- Label time mode: {args.label_time_mode}",
            f"- Time-label epochs: warmup {args.time_label_warmup_epochs}, measured {args.time_label_measured_epochs}",
            f"- Families: {', '.join(families)}",
            f"- Selected workloads file: `{selected_path.name}`",
            "",
            "Run without `--dry-run` to generate sustained profiler labels, 5-epoch golden profiles, comparison CSV/JSON, and the threshold summary.",
        ]
        (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(summary, sort_keys=True))
        return 0

    label_dir = output_dir / "profile_labels"
    golden_root = output_dir / "golden_5epoch"
    run_status: list[dict[str, Any]] = []
    for workload in selected:
        point_id = profile_point_id(workload)
        one_path = output_dir / "label_workloads" / f"{clean_id(point_id)}.jsonl"
        write_jsonl(one_path, [workload])
        status = run_profile_command(args, one_path, label_dir / clean_id(point_id))
        status.update({"phase": "label", "profile_point_id": point_id, "architecture_family": architecture_family(workload)})
        run_status.append(status)
        golden_path = output_dir / "golden_workloads" / f"{clean_id(point_id)}.jsonl"
        write_jsonl(golden_path, [workload])
        status = run_profile_command(args, golden_path, golden_root / clean_id(point_id), golden_steps=steps_per_epoch(workload))
        status.update({"phase": "golden", "profile_point_id": point_id, "architecture_family": architecture_family(workload)})
        run_status.append(status)
    comparisons = compare_rows(selected, label_dir, golden_root, args.epochs)
    summary = write_comparisons(comparisons, output_dir, run_status)
    print(json.dumps(summary, sort_keys=True))
    if not args.report_only and not summary["passed_thresholds"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
