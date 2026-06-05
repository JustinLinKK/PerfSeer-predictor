#!/usr/bin/env python
"""Check that a precision-transfer result ledger contains the required eval evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SLICE_KEYS = (
    "metrics_by_precision",
    "metrics_by_label_domain",
    "metrics_by_batch_size",
    "metrics_by_resource_regime",
    "metrics_by_graph_signature",
    "metrics_by_graph_family",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate precision-transfer eval rows before selecting a deployment student.")
    p.add_argument("--results", default="runs/results.jsonl")
    p.add_argument("--source-run-id", default="precision_large_teacher_source")
    p.add_argument("--baseline-run-id", help="Optional baseline eval run id used to verify source-teacher accuracy preservation.")
    p.add_argument("--transfer-run-id", default="precision_large_teacher_transfer")
    p.add_argument("--student-run-id", default="precision_distill_student_128")
    p.add_argument("--deploy-run-id", help="Run id expected in eval_deploy_complete rows. Defaults to --student-run-id.")
    p.add_argument("--baseline-data-root", help="Expected data_root for the baseline eval row. Defaults to --source-data-root.")
    p.add_argument("--source-data-root", help="Expected data_root for the source-teacher eval row.")
    p.add_argument("--precision-data-root", help="Expected data_root for transfer/student eval rows.")
    p.add_argument("--required-precision", default="", help="Comma-separated precision configs required in transfer and student eval slices.")
    p.add_argument("--min-eval-precision-count", action="append", default=[], metavar="PRECISION=N", help="Minimum held-out rows required for a precision_config in precision teacher/student/deployment eval rows. Can be repeated or comma-separated.")
    p.add_argument("--required-label-domain", default="", help="Comma-separated label domains required in transfer and student eval slices, such as precision_profile.")
    p.add_argument("--min-eval-source-labels", type=int, default=0, help="Minimum source-domain held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-precision-labels", type=int, default=0, help="Minimum precision_profile held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-pseudo-labels", type=int, default=0, help="Minimum pseudo held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-hardware-count", action="append", default=[], metavar="HARDWARE=N", help="Minimum held-out rows required for a hardware_id in precision teacher/student/deployment eval rows. Can be repeated or comma-separated.")
    p.add_argument("--max-source-mean-mape", type=float)
    p.add_argument("--max-source-baseline-mape-delta", type=float, help="Maximum allowed source_teacher mean_mape minus baseline mean_mape.")
    p.add_argument("--max-transfer-mean-mape", type=float)
    p.add_argument("--max-student-mean-mape", type=float)
    p.add_argument("--max-deploy-mean-mape", type=float)
    p.add_argument("--max-deploy-latency-p50", type=float)
    p.add_argument("--min-test-graphs", type=int, default=1)
    p.add_argument("--min-train-split-count", type=int, default=0, help="Minimum train split count required in train_complete metadata.")
    p.add_argument("--min-val-split-count", type=int, default=0, help="Minimum validation split count required in train_complete metadata.")
    p.add_argument("--min-train-test-count", type=int, default=0, help="Minimum test split count required in train_complete metadata.")
    p.add_argument("--min-precision-slices", type=int, default=1, help="Minimum precision slices required for transfer/student eval rows.")
    p.add_argument("--min-label-domain-slices", type=int, default=1, help="Minimum label-domain slices required for each eval row.")
    p.add_argument("--min-batch-size-slices", type=int, default=1, help="Minimum batch-size slices required for each eval row.")
    p.add_argument("--min-resource-regime-slices", type=int, default=1, help="Minimum resource-regime slices required for each eval row.")
    p.add_argument("--min-graph-signature-slices", type=int, default=1, help="Minimum graph-signature slices required for each eval row.")
    p.add_argument("--min-graph-family-slices", type=int, default=1, help="Minimum graph-family slices required for each eval row.")
    p.add_argument("--required-split-unit", choices=("pair", "graph", "graph_signature", "graph_family"), help="Require eval rows to report this train/eval split unit.")
    p.add_argument("--materialization-report", help="Optional precision_materialization_report.json path to validate.")
    p.add_argument("--require-source-precision-provenance", action="store_true", help="Require materialization report source precision provenance without requiring a confirmed precision recipe.")
    p.add_argument("--require-source-precision-confirmed", action="store_true", help="Require materialization report source precision to be confirmed.")
    p.add_argument("--expected-source-precision-config", help="Expected source_precision_config in the materialization report.")
    p.add_argument("--expected-source-precision-provenance", help="Expected source_precision_provenance in the materialization report.")
    p.add_argument("--min-materialized-precision-labels", type=int, default=0, help="Minimum accepted precision labels in the materialization report.")
    p.add_argument("--min-materialized-base-pairs", type=int, default=0, help="Minimum source dataset graph/label pairs included in the materialization report.")
    p.add_argument("--min-materialized-source-labels", type=int, default=0, help="Minimum source-domain labels included in the materialization report.")
    p.add_argument("--min-materialized-pseudo-labels", type=int, default=0, help="Minimum pseudo labels included in the materialization report.")
    p.add_argument("--require-deployment-eval", action="store_true", help="Require an eval_deploy_complete row for the deployment student.")
    p.add_argument("--require-deployment-metadata", action="store_true", help="Require deployment_metadata sidecar JSON in the deployment eval row.")
    p.add_argument("--require-deployment-student-checkpoint", action="store_true", help="Require deployment eval checkpoint paths to match the held-out precision-student eval checkpoint paths.")
    p.add_argument("--expected-deploy-runtime-backend", help="Expected requested runtime_backend in the deployment eval row.")
    p.add_argument("--expected-deploy-runtime-backend-actual", help="Expected actual runtime backend in the deployment eval row.")
    p.add_argument("--require-checkpoint-files", action="store_true", help="Require eval rows' ckpt_paths to exist on disk.")
    p.add_argument("--require-train-events", action="store_true", help="Require train_complete rows for source, transfer, and student runs.")
    p.add_argument("--require-eval-train-checkpoints", action="store_true", help="Require source/transfer/student eval checkpoint paths to match their train_complete checkpoint paths.")
    p.add_argument("--required-train-label-domain", default="", help="Comma-separated label domains required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-precision-count", action="append", default=[], metavar="PRECISION=N", help="Minimum train-split rows required for a precision_config in precision teacher/student train split metadata. Can be repeated or comma-separated.")
    p.add_argument("--min-train-source-labels", type=int, default=0, help="Minimum source-domain labels required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-precision-labels", type=int, default=0, help="Minimum measured precision-profile labels required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-pseudo-labels", type=int, default=0, help="Minimum pseudo-label rows required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-hardware-count", action="append", default=[], metavar="HARDWARE=N", help="Minimum train-split rows required for a hardware_id in precision teacher/student train split metadata. Can be repeated or comma-separated.")
    p.add_argument("--require-unlimited-train-data", action="store_true", help="Require checked train_complete rows to have no data.limit set.")
    p.add_argument("--require-source-train-provenance", action="store_true", help="Require source train_complete metadata to record source precision provenance.")
    p.add_argument("--require-source-train-precision-confirmed", action="store_true", help="Require source train_complete/checkpoint metadata to mark source precision as confirmed.")
    p.add_argument("--require-train-checkpoint-metadata", action="store_true", help="Require train_complete rows to include metadata summaries read from their saved checkpoints.")
    p.add_argument("--require-train-lineage", action="store_true", help="Require transfer/student train rows to link back to source/transfer train checkpoints.")
    p.add_argument("--skip-source", action="store_true")
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--skip-student", action="store_true")
    p.add_argument("--report-out", help="Optional JSON report path.")
    return p.parse_args(argv)


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "eval_complete":
            rows.append(row)
    return rows


def load_deploy_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "eval_deploy_complete":
            rows.append(row)
    return rows


def load_train_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "train_complete":
            rows.append(row)
    return rows


def latest_row(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("run_id")) == run_id]
    return matches[-1] if matches else None


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def paths_match(actual: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    actual_s = str(actual)
    expected_s = str(expected)
    return actual_s == expected_s or Path(actual_s) == Path(expected_s)


def parse_required_precision(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_min_count_specs(raw_items: list[str] | None, *, label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in raw_items or []:
        for part in str(raw).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"{label} entry {item!r} must use NAME=COUNT")
            name, value = item.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"{label} entry {item!r} has an empty name")
            try:
                count = int(value.strip())
            except ValueError as exc:
                raise ValueError(f"{label} entry {item!r} has a non-integer count") from exc
            counts[name] = max(0, count)
    return dict(sorted(counts.items()))


def min_train_label_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "precision_profile": max(0, int(args.min_train_precision_labels or 0)),
        "pseudo": max(0, int(args.min_train_pseudo_labels or 0)),
        "source": max(0, int(args.min_train_source_labels or 0)),
    }


def min_eval_label_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "precision_profile": max(0, int(args.min_eval_precision_labels or 0)),
        "pseudo": max(0, int(args.min_eval_pseudo_labels or 0)),
        "source": max(0, int(args.min_eval_source_labels or 0)),
    }


def row_label_domain_counts(row: dict[str, Any] | None) -> dict[str, int]:
    if row is None:
        return {}
    raw = row.get("label_domain_counts")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value or 0) for key, value in raw.items()}


def row_precision_config_counts(row: dict[str, Any] | None) -> dict[str, int]:
    if row is None:
        return {}
    raw = row.get("precision_config_counts")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value or 0) for key, value in raw.items()}


def row_hardware_id_counts(row: dict[str, Any] | None) -> dict[str, int]:
    if row is None:
        return {}
    raw = row.get("hardware_id_counts")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value or 0) for key, value in raw.items()}


def row_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    precision = row.get("metrics_by_precision") if isinstance(row.get("metrics_by_precision"), dict) else {}
    label_domains = row.get("metrics_by_label_domain") if isinstance(row.get("metrics_by_label_domain"), dict) else {}
    signatures = row.get("metrics_by_graph_signature") if isinstance(row.get("metrics_by_graph_signature"), dict) else {}
    families = row.get("metrics_by_graph_family") if isinstance(row.get("metrics_by_graph_family"), dict) else {}
    evaluation_split = row.get("evaluation_split") if isinstance(row.get("evaluation_split"), dict) else {}
    return {
        "run_id": row.get("run_id"),
        "data_root": row.get("data_root"),
        "mean_mape": row.get("mean_mape"),
        "num_test_graphs": row.get("num_test_graphs"),
        "split_unit": row.get("split_unit") or evaluation_split.get("split_unit"),
        "test_hash": row.get("test_hash") or evaluation_split.get("test_hash"),
        "checkpoint_test_hash": evaluation_split.get("checkpoint_test_hash"),
        "test_count": evaluation_split.get("test_count"),
        "evaluation_split_source": evaluation_split.get("source"),
        "runtime_backend": row.get("runtime_backend"),
        "runtime_backend_actual": row.get("runtime_backend_actual"),
        "deployment_metadata": row.get("deployment_metadata"),
        "precision_configs": sorted(str(key) for key in precision),
        "precision_config_counts": row_precision_config_counts(row),
        "hardware_id_counts": row_hardware_id_counts(row),
        "label_domains": sorted(str(key) for key in label_domains),
        "label_domain_counts": row_label_domain_counts(row),
        "graph_signatures": sorted(str(key) for key in signatures),
        "graph_families": sorted(str(key) for key in families),
    }


def train_checkpoint_metadata_rows(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    raw = row.get("checkpoint_metadata")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def train_config_section(row: dict[str, Any] | None, section: str) -> dict[str, Any]:
    if row is None:
        return {}
    cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
    value = cfg.get(section)
    return value if isinstance(value, dict) else {}


def train_checkpoint_paths(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return []
    paths = row.get("checkpoints")
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if str(path).strip()]


def path_matches_any(actual: Any, expected_paths: list[str]) -> bool:
    return any(paths_match(actual, expected) for expected in expected_paths)


def checkpoint_initialization_paths(row: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    for item in train_checkpoint_metadata_rows(row):
        initialization = item.get("initialization")
        if isinstance(initialization, dict) and initialization.get("path"):
            paths.append(str(initialization["path"]))
    return paths


def checkpoint_distillation_teacher_paths(row: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    for item in train_checkpoint_metadata_rows(row):
        teacher = item.get("distillation_teacher")
        if not isinstance(teacher, dict):
            continue
        raw_paths = teacher.get("paths")
        if isinstance(raw_paths, list):
            paths.extend(str(path) for path in raw_paths if str(path).strip())
    return paths


def train_split_label_domain_counts(row: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    if row is None:
        return {}
    split = row.get("split") if isinstance(row.get("split"), dict) else {}
    counts = split.get("label_domain_counts") if isinstance(split.get("label_domain_counts"), dict) else {}
    if not counts:
        for item in train_checkpoint_metadata_rows(row):
            item_split = item.get("split")
            if not isinstance(item_split, dict):
                continue
            raw = item_split.get("label_domain_counts")
            if isinstance(raw, dict) and raw:
                counts = raw
                break
    out: dict[str, dict[str, int]] = {}
    for split_name, values in counts.items():
        if not isinstance(values, dict):
            continue
        out[str(split_name)] = {str(key): int(value or 0) for key, value in values.items()}
    return out


def train_split_precision_config_counts(row: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    if row is None:
        return {}
    split = row.get("split") if isinstance(row.get("split"), dict) else {}
    counts = split.get("precision_config_counts") if isinstance(split.get("precision_config_counts"), dict) else {}
    if not counts:
        for item in train_checkpoint_metadata_rows(row):
            item_split = item.get("split")
            if not isinstance(item_split, dict):
                continue
            raw = item_split.get("precision_config_counts")
            if isinstance(raw, dict) and raw:
                counts = raw
                break
    out: dict[str, dict[str, int]] = {}
    for split_name, values in counts.items():
        if not isinstance(values, dict):
            continue
        out[str(split_name)] = {str(key): int(value or 0) for key, value in values.items()}
    return out


def train_split_hardware_id_counts(row: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    if row is None:
        return {}
    split = row.get("split") if isinstance(row.get("split"), dict) else {}
    counts = split.get("hardware_id_counts") if isinstance(split.get("hardware_id_counts"), dict) else {}
    if not counts:
        for item in train_checkpoint_metadata_rows(row):
            item_split = item.get("split")
            if not isinstance(item_split, dict):
                continue
            raw = item_split.get("hardware_id_counts")
            if isinstance(raw, dict) and raw:
                counts = raw
                break
    out: dict[str, dict[str, int]] = {}
    for split_name, values in counts.items():
        if not isinstance(values, dict):
            continue
        out[str(split_name)] = {str(key): int(value or 0) for key, value in values.items()}
    return out


def train_split_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    merged: dict[str, Any] = {}
    for item in train_checkpoint_metadata_rows(row):
        item_split = item.get("split")
        if isinstance(item_split, dict) and item_split:
            merged.update(item_split)
            break
    split = row.get("split") if isinstance(row.get("split"), dict) else {}
    merged.update(split)
    return merged


def train_row_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    distill_cfg = cfg.get("distillation") if isinstance(cfg.get("distillation"), dict) else {}
    features_cfg = cfg.get("features") if isinstance(cfg.get("features"), dict) else {}
    checkpoint_metadata = train_checkpoint_metadata_rows(row)
    source_precisions = [
        item.get("source_precision")
        for item in checkpoint_metadata
        if isinstance(item.get("source_precision"), dict)
    ]
    initialization_paths = [
        item["initialization"].get("path")
        for item in checkpoint_metadata
        if isinstance(item.get("initialization"), dict) and item["initialization"].get("path")
    ]
    distillation_kinds = sorted(
        {
            str(item["distillation_teacher"].get("kind"))
            for item in checkpoint_metadata
            if isinstance(item.get("distillation_teacher"), dict) and item["distillation_teacher"].get("kind")
        }
    )
    split_meta = train_split_metadata(row)
    return {
        "run_id": row.get("run_id"),
        "data_root": data_cfg.get("root"),
        "data_limit": data_cfg.get("limit"),
        "precision_config": features_cfg.get("precision_config"),
        "source_precision_confirmed": bool(data_cfg.get("source_precision_confirmed")),
        "source_precision_provenance": data_cfg.get("source_precision_provenance"),
        "init_checkpoint": train_cfg.get("init_checkpoint"),
        "teacher_ckpt_dir": distill_cfg.get("teacher_ckpt_dir"),
        "checkpoints": list(row.get("checkpoints") or []) if isinstance(row.get("checkpoints"), list) else [],
        "checkpoint_metadata_count": len(checkpoint_metadata),
        "split_unit": split_meta.get("split_unit"),
        "train_count": split_meta.get("train_count"),
        "val_count": split_meta.get("val_count"),
        "test_count": split_meta.get("test_count"),
        "test_hash": split_meta.get("test_hash"),
        "split_label_domain_counts": train_split_label_domain_counts(row),
        "split_precision_config_counts": train_split_precision_config_counts(row),
        "split_hardware_id_counts": train_split_hardware_id_counts(row),
        "checkpoint_source_precision": source_precisions[0] if source_precisions else None,
        "checkpoint_initialization_paths": initialization_paths,
        "checkpoint_distillation_teacher_kinds": distillation_kinds,
        "elapsed_sec": row.get("elapsed_sec"),
    }


def row_split_unit(row: dict[str, Any]) -> str | None:
    if row.get("split_unit"):
        return str(row.get("split_unit"))
    evaluation_split = row.get("evaluation_split") if isinstance(row.get("evaluation_split"), dict) else {}
    value = evaluation_split.get("split_unit")
    return str(value) if value else None


def row_test_hash(row: dict[str, Any]) -> str:
    evaluation_split = row.get("evaluation_split") if isinstance(row.get("evaluation_split"), dict) else {}
    return str(row.get("test_hash") or evaluation_split.get("test_hash") or "").strip()


def check_split_evidence(label: str, row: dict[str, Any], required_split_unit: str, min_test_graphs: int) -> list[str]:
    errors: list[str] = []
    evaluation_split = row.get("evaluation_split") if isinstance(row.get("evaluation_split"), dict) else {}
    if not evaluation_split:
        errors.append(f"{label} evaluation_split metadata is missing")
    test_hash = row_test_hash(row)
    if not test_hash:
        errors.append(f"{label} test_hash is missing")
    test_count = evaluation_split.get("test_count")
    if test_count is not None:
        try:
            if int(test_count) < min_test_graphs:
                errors.append(f"{label} evaluation_split test_count is below {min_test_graphs}")
        except (TypeError, ValueError):
            errors.append(f"{label} evaluation_split test_count is not an integer")
    checkpoint_hash = str(evaluation_split.get("checkpoint_test_hash") or "").strip()
    limit_applied = bool(evaluation_split.get("limit_applied"))
    if checkpoint_hash and test_hash and checkpoint_hash != test_hash and not limit_applied:
        errors.append(
            f"{label} test_hash {test_hash!r} does not match checkpoint_test_hash {checkpoint_hash!r}"
        )
    checkpoint_split_unit = str(evaluation_split.get("checkpoint_split_unit") or "").strip()
    if checkpoint_split_unit and checkpoint_split_unit != required_split_unit:
        errors.append(
            f"{label} checkpoint_split_unit {checkpoint_split_unit!r} does not match required {required_split_unit!r}"
        )
    return errors


def load_materialization_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing materialization report: {path}")
    return json.loads(path.read_text())


def materialization_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    source_metadata_labels = int(report.get("source_metadata_labels") or 0)
    calibration_source_labels = int(report.get("calibration_source_labels") or 0)
    return {
        "source_precision_config": report.get("source_precision_config"),
        "source_precision_confirmed": bool(report.get("source_precision_confirmed")),
        "source_precision_provenance": report.get("source_precision_provenance"),
        "base_pairs": int(report.get("base_pairs") or 0),
        "source_metadata_labels": source_metadata_labels,
        "calibration_source_labels": calibration_source_labels,
        "source_labels_total": source_metadata_labels + calibration_source_labels,
        "precision_labels": int(report.get("precision_labels") or 0),
        "pseudo_labels": int(report.get("pseudo_labels") or 0),
        "unsupported_fp8_rows": int(report.get("unsupported_fp8_rows") or 0),
        "report": report.get("report_path"),
    }


def check_materialization_report(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    needs_report = bool(
        args.materialization_report
        or args.require_source_precision_provenance
        or args.require_source_precision_confirmed
        or args.expected_source_precision_config
        or args.expected_source_precision_provenance
        or args.min_materialized_precision_labels
        or args.min_materialized_base_pairs
        or args.min_materialized_source_labels
        or args.min_materialized_pseudo_labels
    )
    if not needs_report:
        return None, []
    if not args.materialization_report:
        return None, ["materialization report is required for materialization checks"]
    report_path = Path(args.materialization_report)
    try:
        report = load_materialization_report(report_path)
    except Exception as exc:
        return None, [str(exc)]
    report["report_path"] = str(report_path)
    errors: list[str] = []
    require_report_source_provenance = bool(args.require_source_precision_provenance or args.require_source_precision_confirmed)
    if require_report_source_provenance:
        if not str(report.get("source_precision_provenance") or "").strip():
            errors.append("materialization report source precision provenance is empty")
    if args.require_source_precision_confirmed:
        if not report.get("source_precision_confirmed"):
            errors.append("materialization report source precision is not confirmed")
    if args.expected_source_precision_config and str(report.get("source_precision_config")) != args.expected_source_precision_config:
        errors.append(
            "materialization report source_precision_config "
            f"{report.get('source_precision_config')!r} does not match expected {args.expected_source_precision_config!r}"
        )
    if args.expected_source_precision_provenance and str(report.get("source_precision_provenance")) != args.expected_source_precision_provenance:
        errors.append("materialization report source_precision_provenance does not match expected value")
    min_precision_labels = max(0, int(args.min_materialized_precision_labels or 0))
    if int(report.get("precision_labels") or 0) < min_precision_labels:
        errors.append(
            "materialization report precision_labels "
            f"{int(report.get('precision_labels') or 0)} is below required {min_precision_labels}"
        )
    min_base_pairs = max(0, int(args.min_materialized_base_pairs or 0))
    if int(report.get("base_pairs") or 0) < min_base_pairs:
        errors.append(
            "materialization report base_pairs "
            f"{int(report.get('base_pairs') or 0)} is below required {min_base_pairs}"
        )
    source_labels_total = int(report.get("source_metadata_labels") or 0) + int(report.get("calibration_source_labels") or 0)
    min_source_labels = max(0, int(args.min_materialized_source_labels or 0))
    if source_labels_total < min_source_labels:
        errors.append(
            "materialization report source labels "
            f"{source_labels_total} is below required {min_source_labels}"
        )
    min_pseudo_labels = max(0, int(args.min_materialized_pseudo_labels or 0))
    if int(report.get("pseudo_labels") or 0) < min_pseudo_labels:
        errors.append(
            "materialization report pseudo_labels "
            f"{int(report.get('pseudo_labels') or 0)} is below required {min_pseudo_labels}"
        )
    return materialization_summary(report), errors


def check_eval_row(
    *,
    label: str,
    row: dict[str, Any] | None,
    expected_data_root: str | None,
    required_precision: list[str],
    required_label_domains: list[str],
    max_mean_mape: float | None,
    min_test_graphs: int,
    require_precision: bool,
    require_label_domain: bool,
    required_split_unit: str | None,
    min_precision_slices: int,
    min_label_domain_slices: int,
    min_batch_size_slices: int,
    min_resource_regime_slices: int,
    min_graph_signature_slices: int,
    min_graph_family_slices: int,
    min_precision_config_counts: dict[str, int],
    min_label_domain_counts: dict[str, int],
    min_hardware_id_counts: dict[str, int],
    require_checkpoint_files: bool,
) -> list[str]:
    errors: list[str] = []
    if row is None:
        return [f"missing {label} eval row"]
    if not is_finite_number(row.get("mean_mape")):
        errors.append(f"{label} mean_mape is missing or non-finite")
    elif max_mean_mape is not None and float(row["mean_mape"]) > max_mean_mape:
        errors.append(f"{label} mean_mape {float(row['mean_mape']):.6g} exceeds threshold {max_mean_mape:.6g}")
    if int(row.get("num_test_graphs") or 0) < min_test_graphs:
        errors.append(f"{label} num_test_graphs is below {min_test_graphs}")
    if not paths_match(row.get("data_root"), expected_data_root):
        errors.append(f"{label} data_root {row.get('data_root')!r} does not match expected {expected_data_root!r}")
    actual_split_unit = row_split_unit(row)
    if required_split_unit is not None and actual_split_unit != required_split_unit:
        errors.append(f"{label} split_unit {actual_split_unit!r} does not match required {required_split_unit!r}")
    if required_split_unit is not None:
        errors.extend(check_split_evidence(label, row, required_split_unit, min_test_graphs))
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append(f"{label} metrics are missing")
    for key in SLICE_KEYS:
        value = row.get(key)
        if not isinstance(value, dict) or not value:
            errors.append(f"{label} {key} is missing or empty")
    min_slice_counts = {
        "metrics_by_precision": min_precision_slices if require_precision else 1,
        "metrics_by_label_domain": min_label_domain_slices,
        "metrics_by_batch_size": min_batch_size_slices,
        "metrics_by_resource_regime": min_resource_regime_slices,
        "metrics_by_graph_signature": min_graph_signature_slices,
        "metrics_by_graph_family": min_graph_family_slices,
    }
    for key, required_count in min_slice_counts.items():
        value = row.get(key)
        if isinstance(value, dict) and len(value) < max(1, required_count):
            errors.append(f"{label} {key} has {len(value)} slice(s), below required {max(1, required_count)}")
    precision_rows = row.get("metrics_by_precision") if isinstance(row.get("metrics_by_precision"), dict) else {}
    if require_precision:
        missing = [precision for precision in required_precision if precision not in precision_rows]
        if missing:
            errors.append(f"{label} missing required precision slice(s): {', '.join(missing)}")
    min_precision_config_counts = {
        str(precision): max(0, int(count or 0))
        for precision, count in min_precision_config_counts.items()
    }
    if any(count > 0 for count in min_precision_config_counts.values()):
        counts = row_precision_config_counts(row)
        if not counts:
            errors.append(f"{label} precision_config_counts are missing or empty")
        for precision, required_count in min_precision_config_counts.items():
            if required_count <= 0:
                continue
            actual_count = int(counts.get(precision) or 0)
            if actual_count < required_count:
                errors.append(
                    f"{label} eval precision_config {precision!r} count {actual_count} "
                    f"is below required {required_count}"
                )
    label_domain_rows = row.get("metrics_by_label_domain") if isinstance(row.get("metrics_by_label_domain"), dict) else {}
    if require_label_domain:
        missing = [domain for domain in required_label_domains if domain not in label_domain_rows]
        if missing:
            errors.append(f"{label} missing required label-domain slice(s): {', '.join(missing)}")
    min_label_domain_counts = {
        str(domain): max(0, int(count or 0))
        for domain, count in min_label_domain_counts.items()
    }
    if any(count > 0 for count in min_label_domain_counts.values()):
        counts = row_label_domain_counts(row)
        if not counts:
            errors.append(f"{label} label_domain_counts are missing or empty")
        for domain, required_count in min_label_domain_counts.items():
            if required_count <= 0:
                continue
            actual_count = int(counts.get(domain) or 0)
            if actual_count < required_count:
                errors.append(
                    f"{label} eval label-domain {domain!r} count {actual_count} "
                    f"is below required {required_count}"
                )
    min_hardware_id_counts = {
        str(hardware_id): max(0, int(count or 0))
        for hardware_id, count in min_hardware_id_counts.items()
    }
    if any(count > 0 for count in min_hardware_id_counts.values()):
        counts = row_hardware_id_counts(row)
        if not counts:
            errors.append(f"{label} hardware_id_counts are missing or empty")
        for hardware_id, required_count in min_hardware_id_counts.items():
            if required_count <= 0:
                continue
            actual_count = int(counts.get(hardware_id) or 0)
            if actual_count < required_count:
                errors.append(
                    f"{label} eval hardware_id {hardware_id!r} count {actual_count} "
                    f"is below required {required_count}"
                )
    if require_checkpoint_files:
        errors.extend(check_checkpoint_files(label, row))
    return errors


def check_checkpoint_files(label: str, row: dict[str, Any]) -> list[str]:
    paths = row.get("ckpt_paths")
    if not isinstance(paths, list) or not paths:
        return [f"{label} ckpt_paths are missing or empty"]
    errors: list[str] = []
    for path in paths:
        if not Path(str(path)).is_file():
            errors.append(f"{label} checkpoint file is missing: {path}")
    return errors


def eval_checkpoint_paths(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return []
    paths = row.get("ckpt_paths")
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if str(path).strip()]


def check_deployment_student_checkpoint(
    student_row: dict[str, Any] | None,
    deployment_row: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    student_paths = eval_checkpoint_paths(student_row)
    deployment_paths = eval_checkpoint_paths(deployment_row)
    summary = {
        "student_ckpt_paths": student_paths,
        "deployment_ckpt_paths": deployment_paths,
    }
    errors: list[str] = []
    if student_row is None or deployment_row is None:
        return summary, errors
    if not student_paths:
        errors.append("deployment checkpoint linkage precision_student ckpt_paths are missing or empty")
    if not deployment_paths:
        errors.append("deployment checkpoint linkage deployment_student ckpt_paths are missing or empty")
    for path in deployment_paths:
        if student_paths and not path_matches_any(path, student_paths):
            errors.append(
                f"deployment checkpoint linkage deployment_student checkpoint {path!r} "
                "is not listed in precision_student eval checkpoints"
            )
    for path in student_paths:
        if deployment_paths and not path_matches_any(path, deployment_paths):
            errors.append(
                f"deployment checkpoint linkage precision_student checkpoint {path!r} "
                "is not listed in deployment_student eval checkpoints"
            )
    return summary, errors


def check_eval_train_checkpoint_linkage(
    specs: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]],
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {}
    errors: list[str] = []
    for label, eval_row, train_row in specs:
        eval_paths = eval_checkpoint_paths(eval_row)
        train_paths = train_checkpoint_paths(train_row)
        summary[label] = {
            "eval_ckpt_paths": eval_paths,
            "train_checkpoints": train_paths,
        }
        if eval_row is None:
            errors.append(f"missing {label} eval row for eval/train checkpoint linkage")
            continue
        if train_row is None:
            errors.append(f"missing {label} train row for eval/train checkpoint linkage")
            continue
        if not eval_paths:
            errors.append(f"eval/train checkpoint linkage {label} eval ckpt_paths are missing or empty")
        if not train_paths:
            errors.append(f"eval/train checkpoint linkage {label} train checkpoints are missing or empty")
        for path in eval_paths:
            if train_paths and not path_matches_any(path, train_paths):
                errors.append(
                    f"eval/train checkpoint linkage {label} eval checkpoint {path!r} "
                    f"is not listed in {label} train checkpoints"
                )
        for path in train_paths:
            if eval_paths and not path_matches_any(path, eval_paths):
                errors.append(
                    f"eval/train checkpoint linkage {label} train checkpoint {path!r} "
                    f"is not listed in {label} eval checkpoints"
                )
    return summary, errors


def is_unlimited_data_limit(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "0", "none", "None", "null"}
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def int_field(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_train_split_metadata(
    *,
    label: str,
    row: dict[str, Any] | None,
    required_split_unit: str | None,
    min_train_split_count: int,
    min_val_split_count: int,
    min_train_test_count: int,
) -> list[str]:
    min_counts = {
        "train_count": max(0, int(min_train_split_count or 0)),
        "val_count": max(0, int(min_val_split_count or 0)),
        "test_count": max(0, int(min_train_test_count or 0)),
    }
    if required_split_unit is None and not any(count > 0 for count in min_counts.values()):
        return []
    if row is None:
        return [f"missing {label} train row for train split validation"]
    split = train_split_metadata(row)
    if not split:
        return [f"{label} train split metadata is missing"]
    errors: list[str] = []
    actual_split_unit = str(split.get("split_unit") or "").strip() or None
    if required_split_unit is not None:
        if actual_split_unit != required_split_unit:
            errors.append(
                f"{label} train split_unit {actual_split_unit!r} "
                f"does not match required {required_split_unit!r}"
            )
        if not str(split.get("test_hash") or "").strip():
            errors.append(f"{label} train test_hash is missing")
    for field, required_count in min_counts.items():
        if required_count <= 0:
            continue
        actual_count = int_field(split.get(field))
        if actual_count is None:
            errors.append(f"{label} train {field} is missing or not an integer")
            continue
        if actual_count < required_count:
            errors.append(
                f"{label} train {field} {actual_count} "
                f"is below required {required_count}"
            )
    return errors


def check_checkpoint_path_list(label: str, field: str, paths: Any) -> list[str]:
    if not isinstance(paths, list) or not paths:
        return [f"{label} {field} are missing or empty"]
    errors: list[str] = []
    for path in paths:
        if not Path(str(path)).is_file():
            errors.append(f"{label} checkpoint file is missing: {path}")
    return errors


def check_checkpoint_metadata_list(
    *,
    label: str,
    row: dict[str, Any],
    checkpoints: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw = row.get("checkpoint_metadata")
    if not isinstance(raw, list) or not raw:
        return [], [f"{label} train checkpoint_metadata is missing or empty"]
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    checkpoint_paths = list(checkpoints) if isinstance(checkpoints, list) else []
    if checkpoint_paths and len(raw) != len(checkpoint_paths):
        errors.append(
            f"{label} train checkpoint_metadata has {len(raw)} item(s), expected {len(checkpoint_paths)}"
        )
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"{label} train checkpoint_metadata item {idx} is not an object")
            continue
        summaries.append(item)
        path = str(item.get("path") or "").strip()
        if not path:
            errors.append(f"{label} train checkpoint_metadata item {idx} path is missing")
        elif checkpoint_paths and not any(paths_match(path, expected) for expected in checkpoint_paths):
            errors.append(f"{label} train checkpoint_metadata path {path!r} is not listed in train checkpoints")
        load_error = str(item.get("load_error") or "").strip()
        if load_error:
            errors.append(f"{label} train checkpoint_metadata load error for {path or idx}: {load_error}")
    return summaries, errors


def check_checkpoint_source_precision(
    *,
    label: str,
    summaries: list[dict[str, Any]],
    expected_source_precision_config: str | None,
    expected_source_precision_provenance: str | None,
    require_confirmed: bool,
) -> list[str]:
    errors: list[str] = []
    for idx, summary in enumerate(summaries):
        source_precision = summary.get("source_precision")
        if not isinstance(source_precision, dict) or not source_precision:
            errors.append(f"{label} checkpoint source_precision metadata is missing for checkpoint {idx}")
            continue
        provenance = str(source_precision.get("provenance") or "").strip()
        if require_confirmed and not source_precision.get("confirmed"):
            errors.append(f"{label} checkpoint source precision is not confirmed")
        if not provenance:
            errors.append(f"{label} checkpoint source precision provenance is empty")
        if expected_source_precision_config and str(source_precision.get("precision_config")) != expected_source_precision_config:
            errors.append(
                f"{label} checkpoint precision_config {source_precision.get('precision_config')!r} "
                f"does not match expected {expected_source_precision_config!r}"
            )
        if expected_source_precision_provenance and provenance != expected_source_precision_provenance:
            errors.append(f"{label} checkpoint source precision provenance does not match expected value")
    return errors


def check_checkpoint_initialization(
    *,
    label: str,
    summaries: list[dict[str, Any]],
    init_checkpoint: str,
) -> list[str]:
    errors: list[str] = []
    for idx, summary in enumerate(summaries):
        initialization = summary.get("initialization")
        if not isinstance(initialization, dict) or not initialization:
            errors.append(f"{label} checkpoint initialization metadata is missing for checkpoint {idx}")
            continue
        if not paths_match(initialization.get("path"), init_checkpoint):
            errors.append(
                f"{label} checkpoint initialization path {initialization.get('path')!r} "
                f"does not match train init_checkpoint {init_checkpoint!r}"
            )
    return errors


def check_checkpoint_distillation_teacher(
    *,
    label: str,
    summaries: list[dict[str, Any]],
    require_checkpoint_files: bool,
) -> list[str]:
    errors: list[str] = []
    for idx, summary in enumerate(summaries):
        teacher = summary.get("distillation_teacher")
        if not isinstance(teacher, dict) or not teacher or str(teacher.get("kind") or "none") == "none":
            errors.append(f"{label} checkpoint distillation_teacher metadata is missing for checkpoint {idx}")
            continue
        paths = teacher.get("paths")
        if not isinstance(paths, list) or not paths:
            errors.append(f"{label} checkpoint distillation_teacher paths are missing for checkpoint {idx}")
            continue
        if require_checkpoint_files:
            for path in paths:
                if not Path(str(path)).is_file():
                    errors.append(f"{label} checkpoint distillation teacher file is missing: {path}")
    return errors


def check_train_row(
    *,
    label: str,
    row: dict[str, Any] | None,
    expected_data_root: str | None,
    require_checkpoint_files: bool,
    require_checkpoint_metadata: bool,
    require_unlimited_train_data: bool,
    require_init_checkpoint: bool = False,
    require_teacher_ckpt_dir: bool = False,
    require_source_provenance: bool = False,
    require_source_precision_confirmed: bool = False,
    expected_source_precision_config: str | None = None,
    expected_source_precision_provenance: str | None = None,
    required_split_unit: str | None = None,
    min_train_split_count: int = 0,
    min_val_split_count: int = 0,
    min_train_test_count: int = 0,
) -> list[str]:
    if row is None:
        return [f"missing {label} train row"]
    errors: list[str] = []
    cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
    if not cfg:
        errors.append(f"{label} train config is missing")
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    features_cfg = cfg.get("features") if isinstance(cfg.get("features"), dict) else {}
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    distill_cfg = cfg.get("distillation") if isinstance(cfg.get("distillation"), dict) else {}
    if not paths_match(data_cfg.get("root"), expected_data_root):
        errors.append(f"{label} train data root {data_cfg.get('root')!r} does not match expected {expected_data_root!r}")
    if require_unlimited_train_data and not is_unlimited_data_limit(data_cfg.get("limit")):
        errors.append(f"{label} train data.limit {data_cfg.get('limit')!r} is not unlimited")
    if require_source_provenance or require_source_precision_confirmed:
        provenance = str(data_cfg.get("source_precision_provenance") or "").strip()
        if require_source_precision_confirmed and not data_cfg.get("source_precision_confirmed"):
            errors.append(f"{label} train source precision is not confirmed")
        if not provenance:
            errors.append(f"{label} train source precision provenance is empty")
        if expected_source_precision_config and str(features_cfg.get("precision_config")) != expected_source_precision_config:
            errors.append(
                f"{label} train precision_config {features_cfg.get('precision_config')!r} "
                f"does not match expected {expected_source_precision_config!r}"
            )
        if expected_source_precision_provenance and provenance != expected_source_precision_provenance:
            errors.append(f"{label} train source precision provenance does not match expected value")
    checkpoints = row.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        errors.append(f"{label} train checkpoints are missing or empty")
    elif require_checkpoint_files:
        errors.extend(check_checkpoint_path_list(label, "train checkpoints", checkpoints))
    checkpoint_summaries: list[dict[str, Any]] = []
    if require_checkpoint_metadata:
        checkpoint_summaries, metadata_errors = check_checkpoint_metadata_list(
            label=label,
            row=row,
            checkpoints=checkpoints,
        )
        errors.extend(metadata_errors)
        if require_source_provenance or require_source_precision_confirmed:
            errors.extend(
                check_checkpoint_source_precision(
                    label=label,
                    summaries=checkpoint_summaries,
                    expected_source_precision_config=expected_source_precision_config,
                    expected_source_precision_provenance=expected_source_precision_provenance,
                    require_confirmed=require_source_precision_confirmed,
                )
            )
    init_checkpoint = str(train_cfg.get("init_checkpoint") or "").strip()
    if require_init_checkpoint:
        if not init_checkpoint:
            errors.append(f"{label} train init_checkpoint is missing")
        elif require_checkpoint_files and not Path(init_checkpoint).is_file():
            errors.append(f"{label} train init_checkpoint file is missing: {init_checkpoint}")
        if init_checkpoint and require_checkpoint_metadata:
            errors.extend(
                check_checkpoint_initialization(
                    label=label,
                    summaries=checkpoint_summaries,
                    init_checkpoint=init_checkpoint,
                )
            )
    teacher_ckpt_dir = str(distill_cfg.get("teacher_ckpt_dir") or "").strip()
    if require_teacher_ckpt_dir:
        if not teacher_ckpt_dir:
            errors.append(f"{label} train teacher_ckpt_dir is missing")
        elif require_checkpoint_files and not Path(teacher_ckpt_dir).is_dir():
            errors.append(f"{label} train teacher_ckpt_dir is missing: {teacher_ckpt_dir}")
        if teacher_ckpt_dir and require_checkpoint_metadata:
            errors.extend(
                check_checkpoint_distillation_teacher(
                    label=label,
                    summaries=checkpoint_summaries,
                    require_checkpoint_files=require_checkpoint_files,
                )
            )
    errors.extend(
        check_train_split_metadata(
            label=label,
            row=row,
            required_split_unit=required_split_unit,
            min_train_split_count=min_train_split_count,
            min_val_split_count=min_val_split_count,
            min_train_test_count=min_train_test_count,
        )
    )
    return errors


def check_train_label_domains(
    *,
    label: str,
    row: dict[str, Any] | None,
    required_domains: list[str],
    min_counts: dict[str, int],
) -> list[str]:
    min_counts = {str(domain): max(0, int(count or 0)) for domain, count in min_counts.items()}
    if not required_domains and not any(count > 0 for count in min_counts.values()):
        return []
    if row is None:
        return [f"missing {label} train row for train label-domain validation"]
    counts = train_split_label_domain_counts(row)
    train_counts = counts.get("train") if isinstance(counts.get("train"), dict) else {}
    if not train_counts:
        return [f"{label} train label-domain counts are missing or empty"]
    errors: list[str] = []
    for domain in required_domains:
        if int(train_counts.get(domain) or 0) <= 0:
            errors.append(f"{label} train split missing required label-domain {domain!r}")
    for domain, required_count in min_counts.items():
        if required_count <= 0:
            continue
        actual_count = int(train_counts.get(domain) or 0)
        if actual_count < required_count:
            errors.append(
                f"{label} train split label-domain {domain!r} count {actual_count} "
                f"is below required {required_count}"
            )
    return errors


def check_train_precision_configs(
    *,
    label: str,
    row: dict[str, Any] | None,
    min_counts: dict[str, int],
) -> list[str]:
    min_counts = {str(precision): max(0, int(count or 0)) for precision, count in min_counts.items()}
    if not any(count > 0 for count in min_counts.values()):
        return []
    if row is None:
        return [f"missing {label} train row for train precision-config validation"]
    counts = train_split_precision_config_counts(row)
    train_counts = counts.get("train") if isinstance(counts.get("train"), dict) else {}
    if not train_counts:
        return [f"{label} train precision-config counts are missing or empty"]
    errors: list[str] = []
    for precision, required_count in min_counts.items():
        if required_count <= 0:
            continue
        actual_count = int(train_counts.get(precision) or 0)
        if actual_count < required_count:
            errors.append(
                f"{label} train split precision_config {precision!r} count {actual_count} "
                f"is below required {required_count}"
            )
    return errors


def check_train_hardware_ids(
    *,
    label: str,
    row: dict[str, Any] | None,
    min_counts: dict[str, int],
) -> list[str]:
    min_counts = {str(hardware_id): max(0, int(count or 0)) for hardware_id, count in min_counts.items()}
    if not any(count > 0 for count in min_counts.values()):
        return []
    if row is None:
        return [f"missing {label} train row for train hardware-id validation"]
    counts = train_split_hardware_id_counts(row)
    train_counts = counts.get("train") if isinstance(counts.get("train"), dict) else {}
    if not train_counts:
        return [f"{label} train hardware-id counts are missing or empty"]
    errors: list[str] = []
    for hardware_id, required_count in min_counts.items():
        if required_count <= 0:
            continue
        actual_count = int(train_counts.get(hardware_id) or 0)
        if actual_count < required_count:
            errors.append(
                f"{label} train split hardware_id {hardware_id!r} count {actual_count} "
                f"is below required {required_count}"
            )
    return errors


def check_train_lineage(
    *,
    source_row: dict[str, Any] | None,
    transfer_row: dict[str, Any] | None,
    student_row: dict[str, Any] | None,
    skip_source: bool,
    skip_transfer: bool,
    skip_student: bool,
    require_checkpoint_metadata: bool,
) -> tuple[dict[str, Any], list[str]]:
    source_paths = train_checkpoint_paths(source_row)
    transfer_paths = train_checkpoint_paths(transfer_row)
    transfer_train_cfg = train_config_section(transfer_row, "train")
    student_distill_cfg = train_config_section(student_row, "distillation")
    transfer_init = str(transfer_train_cfg.get("init_checkpoint") or "").strip()
    student_teacher_dir = str(student_distill_cfg.get("teacher_ckpt_dir") or "").strip()
    transfer_init_metadata_paths = checkpoint_initialization_paths(transfer_row)
    student_teacher_metadata_paths = checkpoint_distillation_teacher_paths(student_row)
    lineage = {
        "source_checkpoints": source_paths,
        "transfer_checkpoints": transfer_paths,
        "transfer_init_checkpoint": transfer_init or None,
        "transfer_checkpoint_initialization_paths": transfer_init_metadata_paths,
        "student_teacher_ckpt_dir": student_teacher_dir or None,
        "student_checkpoint_teacher_paths": student_teacher_metadata_paths,
    }
    errors: list[str] = []
    if not skip_source and not skip_transfer:
        if not source_paths:
            errors.append("train lineage source_teacher checkpoints are missing or empty")
        if not transfer_init:
            errors.append("train lineage precision_teacher init_checkpoint is missing")
        elif source_paths and not path_matches_any(transfer_init, source_paths):
            errors.append(
                f"train lineage precision_teacher init_checkpoint {transfer_init!r} "
                "is not listed in source_teacher train checkpoints"
            )
        if require_checkpoint_metadata:
            if not transfer_init_metadata_paths:
                errors.append("train lineage precision_teacher checkpoint initialization paths are missing")
            for path in transfer_init_metadata_paths:
                if source_paths and not path_matches_any(path, source_paths):
                    errors.append(
                        f"train lineage precision_teacher checkpoint initialization path {path!r} "
                        "is not listed in source_teacher train checkpoints"
                    )
    if not skip_transfer and not skip_student:
        if not transfer_paths:
            errors.append("train lineage precision_teacher checkpoints are missing or empty")
        if not student_teacher_dir:
            errors.append("train lineage precision_student teacher_ckpt_dir is missing")
        else:
            transfer_dirs = [str(Path(path).parent) for path in transfer_paths]
            transfer_out_dir = str(transfer_row.get("out_dir") or "").strip() if transfer_row else ""
            if transfer_out_dir:
                transfer_dirs.append(transfer_out_dir)
            if transfer_dirs and not any(paths_match(student_teacher_dir, directory) for directory in transfer_dirs):
                errors.append(
                    f"train lineage precision_student teacher_ckpt_dir {student_teacher_dir!r} "
                    "does not match precision_teacher output/checkpoint directory"
                )
        if require_checkpoint_metadata:
            if not student_teacher_metadata_paths:
                errors.append("train lineage precision_student checkpoint teacher paths are missing")
            for path in student_teacher_metadata_paths:
                if transfer_paths and not path_matches_any(path, transfer_paths):
                    errors.append(
                        f"train lineage precision_student checkpoint teacher path {path!r} "
                        "is not listed in precision_teacher train checkpoints"
                    )
    return lineage, errors


def check_deployment_metadata(path: str) -> list[str]:
    errors: list[str] = []
    if not path:
        return ["deployment eval row is missing deployment_metadata"]
    meta_path = Path(path)
    if not meta_path.exists():
        return [f"deployment metadata file is missing: {path}"]
    try:
        metadata = json.loads(meta_path.read_text())
    except Exception as exc:
        return [f"deployment metadata file could not be read: {exc}"]
    for key in ("feature_config", "precision_hardware_config", "feature_layout", "supported_precision_hardware", "required_inputs"):
        if not isinstance(metadata.get(key), dict) or not metadata.get(key):
            errors.append(f"deployment metadata {key} is missing or empty")
    return errors


def check_deployment_row(
    *,
    row: dict[str, Any] | None,
    expected_data_root: str | None,
    required_precision: list[str],
    required_label_domains: list[str],
    max_mean_mape: float | None,
    max_latency_p50: float | None,
    min_test_graphs: int,
    required_split_unit: str | None,
    min_precision_slices: int,
    min_label_domain_slices: int,
    min_batch_size_slices: int,
    min_resource_regime_slices: int,
    min_graph_signature_slices: int,
    min_graph_family_slices: int,
    min_precision_config_counts: dict[str, int],
    min_label_domain_counts: dict[str, int],
    min_hardware_id_counts: dict[str, int],
    require_checkpoint_files: bool,
    expected_runtime_backend: str | None,
    expected_runtime_backend_actual: str | None,
    require_deployment_metadata: bool,
) -> list[str]:
    errors = check_eval_row(
        label="deployment_student",
        row=row,
        expected_data_root=expected_data_root,
        required_precision=required_precision,
        required_label_domains=required_label_domains,
        max_mean_mape=max_mean_mape,
        min_test_graphs=min_test_graphs,
        require_precision=bool(required_precision),
        require_label_domain=bool(required_label_domains),
        required_split_unit=required_split_unit,
        min_precision_slices=min_precision_slices,
        min_label_domain_slices=min_label_domain_slices,
        min_batch_size_slices=min_batch_size_slices,
        min_resource_regime_slices=min_resource_regime_slices,
        min_graph_signature_slices=min_graph_signature_slices,
        min_graph_family_slices=min_graph_family_slices,
        min_precision_config_counts=min_precision_config_counts,
        min_label_domain_counts=min_label_domain_counts,
        min_hardware_id_counts=min_hardware_id_counts,
        require_checkpoint_files=require_checkpoint_files,
    )
    if row is None:
        return errors
    if expected_runtime_backend and str(row.get("runtime_backend")) != expected_runtime_backend:
        errors.append(
            "deployment_student runtime_backend "
            f"{row.get('runtime_backend')!r} does not match expected {expected_runtime_backend!r}"
        )
    if expected_runtime_backend_actual and str(row.get("runtime_backend_actual")) != expected_runtime_backend_actual:
        errors.append(
            "deployment_student runtime_backend_actual "
            f"{row.get('runtime_backend_actual')!r} does not match expected {expected_runtime_backend_actual!r}"
        )
    if max_latency_p50 is not None:
        if not is_finite_number(row.get("latency_forward_ms_p50")):
            errors.append("deployment_student latency_forward_ms_p50 is missing or non-finite")
        elif float(row["latency_forward_ms_p50"]) > max_latency_p50:
            errors.append(
                f"deployment_student latency_forward_ms_p50 {float(row['latency_forward_ms_p50']):.6g} "
                f"exceeds threshold {max_latency_p50:.6g}"
            )
    if require_deployment_metadata:
        errors.extend(check_deployment_metadata(str(row.get("deployment_metadata") or "")))
    return errors


def check_source_baseline_comparison(
    *,
    source_row: dict[str, Any] | None,
    baseline_row: dict[str, Any] | None,
    max_delta: float,
) -> tuple[dict[str, Any], list[str]]:
    summary = {
        "baseline_mean_mape": baseline_row.get("mean_mape") if baseline_row else None,
        "source_mean_mape": source_row.get("mean_mape") if source_row else None,
        "max_delta": max_delta,
        "delta": None,
    }
    errors: list[str] = []
    if baseline_row is None:
        errors.append("missing baseline eval row for source baseline comparison")
        return summary, errors
    if source_row is None:
        errors.append("missing source_teacher eval row for source baseline comparison")
        return summary, errors
    if not is_finite_number(baseline_row.get("mean_mape")):
        errors.append("baseline mean_mape is missing or non-finite")
        return summary, errors
    if not is_finite_number(source_row.get("mean_mape")):
        errors.append("source_teacher mean_mape is missing or non-finite for baseline comparison")
        return summary, errors
    delta = float(source_row["mean_mape"]) - float(baseline_row["mean_mape"])
    summary["delta"] = delta
    if delta > max_delta:
        errors.append(
            f"source_teacher mean_mape delta vs baseline {delta:.6g} exceeds threshold {max_delta:.6g}"
        )
    return summary, errors


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_eval_rows(Path(args.results))
    deploy_rows = load_deploy_rows(Path(args.results))
    train_rows = load_train_rows(Path(args.results))
    required_precision = parse_required_precision(args.required_precision)
    required_eval_precision_counts = parse_min_count_specs(args.min_eval_precision_count, label="--min-eval-precision-count")
    required_eval_hardware_counts = parse_min_count_specs(args.min_eval_hardware_count, label="--min-eval-hardware-count")
    required_label_domains = parse_required_precision(args.required_label_domain)
    required_eval_label_counts = min_eval_label_counts(args)
    required_train_label_domains = parse_required_precision(args.required_train_label_domain)
    required_train_precision_counts = parse_min_count_specs(args.min_train_precision_count, label="--min-train-precision-count")
    required_train_hardware_counts = parse_min_count_specs(args.min_train_hardware_count, label="--min-train-hardware-count")
    required_train_label_counts = min_train_label_counts(args)
    require_train_label_counts = any(count > 0 for count in required_train_label_counts.values())
    require_train_precision_counts = any(count > 0 for count in required_train_precision_counts.values())
    require_train_hardware_counts = any(count > 0 for count in required_train_hardware_counts.values())
    required_train_split_counts = {
        "train": max(0, int(args.min_train_split_count or 0)),
        "val": max(0, int(args.min_val_split_count or 0)),
        "test": max(0, int(args.min_train_test_count or 0)),
    }
    require_train_split_counts = any(count > 0 for count in required_train_split_counts.values())
    train_required_split_unit = (
        args.required_split_unit
        if args.required_split_unit is not None and (args.require_train_events or require_train_split_counts)
        else None
    )
    materialization, materialization_errors = check_materialization_report(args)
    run_specs = []
    baseline_row = latest_row(rows, args.baseline_run_id) if args.baseline_run_id else None
    if not args.skip_source:
        run_specs.append(
            (
                "source_teacher",
                latest_row(rows, args.source_run_id),
                args.source_data_root,
                [],
                [],
                args.max_source_mean_mape,
            )
        )
    if not args.skip_transfer:
        run_specs.append(
            (
                "precision_teacher",
                latest_row(rows, args.transfer_run_id),
                args.precision_data_root,
                required_precision,
                required_label_domains,
                args.max_transfer_mean_mape,
            )
        )
    if not args.skip_student:
        run_specs.append(
            (
                "precision_student",
                latest_row(rows, args.student_run_id),
                args.precision_data_root,
                required_precision,
                required_label_domains,
                args.max_student_mean_mape,
            )
        )

    errors: list[str] = list(materialization_errors)
    summaries: dict[str, Any] = {}
    if args.baseline_run_id:
        summaries["baseline"] = row_summary(baseline_row)
        if baseline_row is not None and not paths_match(baseline_row.get("data_root"), args.baseline_data_root or args.source_data_root):
            errors.append(
                f"baseline data_root {baseline_row.get('data_root')!r} "
                f"does not match expected {(args.baseline_data_root or args.source_data_root)!r}"
            )
    for label, row, data_root, precision, label_domains, threshold in run_specs:
        summaries[label] = row_summary(row)
        eval_label_counts = required_eval_label_counts if label in {"precision_teacher", "precision_student"} else {}
        eval_precision_counts = required_eval_precision_counts if label in {"precision_teacher", "precision_student"} else {}
        eval_hardware_counts = required_eval_hardware_counts if label in {"precision_teacher", "precision_student"} else {}
        errors.extend(
            check_eval_row(
                label=label,
                row=row,
                expected_data_root=data_root,
                required_precision=precision,
                required_label_domains=label_domains,
                max_mean_mape=threshold,
                min_test_graphs=max(1, int(args.min_test_graphs)),
                require_precision=bool(precision),
                require_label_domain=bool(label_domains),
                required_split_unit=args.required_split_unit,
                min_precision_slices=max(1, int(args.min_precision_slices)),
                min_label_domain_slices=max(1, int(args.min_label_domain_slices)),
                min_batch_size_slices=max(1, int(args.min_batch_size_slices)),
                min_resource_regime_slices=max(1, int(args.min_resource_regime_slices)),
                min_graph_signature_slices=max(1, int(args.min_graph_signature_slices)),
                min_graph_family_slices=max(1, int(args.min_graph_family_slices)),
                min_precision_config_counts=eval_precision_counts,
                min_label_domain_counts=eval_label_counts,
                min_hardware_id_counts=eval_hardware_counts,
                require_checkpoint_files=bool(args.require_checkpoint_files),
            )
        )
    baseline_comparison = None
    if args.max_source_baseline_mape_delta is not None or args.baseline_run_id:
        if not args.baseline_run_id:
            errors.append("--baseline-run-id is required for source baseline comparison")
        else:
            baseline_comparison, baseline_errors = check_source_baseline_comparison(
                source_row=latest_row(rows, args.source_run_id),
                baseline_row=baseline_row,
                max_delta=float(args.max_source_baseline_mape_delta or 0.0),
            )
            errors.extend(baseline_errors)
    training_summaries: dict[str, Any] = {}
    if (
        args.require_train_events
        or args.require_unlimited_train_data
        or args.require_source_train_provenance
        or args.require_source_train_precision_confirmed
        or args.require_train_checkpoint_metadata
        or args.require_train_lineage
        or args.require_eval_train_checkpoints
        or required_train_label_domains
        or require_train_label_counts
        or require_train_precision_counts
        or require_train_hardware_counts
        or require_train_split_counts
    ):
        train_specs: list[tuple[str, dict[str, Any] | None, str | None, bool, bool]] = []
        if not args.skip_source:
            train_specs.append(("source_teacher", latest_row(train_rows, args.source_run_id), args.source_data_root, False, False))
        if not args.skip_transfer:
            train_specs.append(("precision_teacher", latest_row(train_rows, args.transfer_run_id), args.precision_data_root, True, False))
        if not args.skip_student:
            train_specs.append(("precision_student", latest_row(train_rows, args.student_run_id), args.precision_data_root, False, True))
        for label, row, data_root, require_init, require_teacher in train_specs:
            training_summaries[label] = train_row_summary(row)
            errors.extend(
                check_train_row(
                    label=label,
                    row=row,
                    expected_data_root=data_root,
                    require_checkpoint_files=bool(args.require_checkpoint_files),
                    require_checkpoint_metadata=bool(args.require_train_checkpoint_metadata),
                    require_unlimited_train_data=bool(args.require_unlimited_train_data),
                    require_init_checkpoint=require_init,
                    require_teacher_ckpt_dir=require_teacher,
                    require_source_provenance=bool(args.require_source_train_provenance and label == "source_teacher"),
                    require_source_precision_confirmed=bool(args.require_source_train_precision_confirmed and label == "source_teacher"),
                    expected_source_precision_config=args.expected_source_precision_config,
                    expected_source_precision_provenance=args.expected_source_precision_provenance,
                    required_split_unit=train_required_split_unit,
                    min_train_split_count=required_train_split_counts["train"],
                    min_val_split_count=required_train_split_counts["val"],
                    min_train_test_count=required_train_split_counts["test"],
                )
            )
            if label in {"precision_teacher", "precision_student"}:
                errors.extend(
                    check_train_precision_configs(
                        label=label,
                        row=row,
                        min_counts=required_train_precision_counts,
                    )
                )
                errors.extend(
                    check_train_hardware_ids(
                        label=label,
                        row=row,
                        min_counts=required_train_hardware_counts,
                    )
                )
                errors.extend(
                    check_train_label_domains(
                        label=label,
                        row=row,
                        required_domains=required_train_label_domains,
                        min_counts=required_train_label_counts,
                    )
                )
    train_lineage = None
    if args.require_train_lineage:
        train_lineage, lineage_errors = check_train_lineage(
            source_row=latest_row(train_rows, args.source_run_id),
            transfer_row=latest_row(train_rows, args.transfer_run_id),
            student_row=latest_row(train_rows, args.student_run_id),
            skip_source=bool(args.skip_source),
            skip_transfer=bool(args.skip_transfer),
            skip_student=bool(args.skip_student),
            require_checkpoint_metadata=bool(args.require_train_checkpoint_metadata),
        )
        errors.extend(lineage_errors)
    eval_train_checkpoint_linkage = None
    if args.require_eval_train_checkpoints:
        linkage_specs: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
        if not args.skip_source:
            linkage_specs.append(("source_teacher", latest_row(rows, args.source_run_id), latest_row(train_rows, args.source_run_id)))
        if not args.skip_transfer:
            linkage_specs.append(("precision_teacher", latest_row(rows, args.transfer_run_id), latest_row(train_rows, args.transfer_run_id)))
        if not args.skip_student:
            linkage_specs.append(("precision_student", latest_row(rows, args.student_run_id), latest_row(train_rows, args.student_run_id)))
        eval_train_checkpoint_linkage, linkage_errors = check_eval_train_checkpoint_linkage(linkage_specs)
        errors.extend(linkage_errors)
    deployment_summary = None
    deployment_checkpoint_linkage = None
    if args.require_deployment_eval:
        deploy_run_id = args.deploy_run_id or args.student_run_id
        deployment_row = latest_row(deploy_rows, deploy_run_id)
        deployment_summary = row_summary(deployment_row)
        errors.extend(
            check_deployment_row(
                row=deployment_row,
                expected_data_root=args.precision_data_root,
                required_precision=required_precision,
                required_label_domains=required_label_domains,
                max_mean_mape=args.max_deploy_mean_mape,
                max_latency_p50=args.max_deploy_latency_p50,
                min_test_graphs=max(1, int(args.min_test_graphs)),
                required_split_unit=args.required_split_unit,
                min_precision_slices=max(1, int(args.min_precision_slices)),
                min_label_domain_slices=max(1, int(args.min_label_domain_slices)),
                min_batch_size_slices=max(1, int(args.min_batch_size_slices)),
                min_resource_regime_slices=max(1, int(args.min_resource_regime_slices)),
                min_graph_signature_slices=max(1, int(args.min_graph_signature_slices)),
                min_graph_family_slices=max(1, int(args.min_graph_family_slices)),
                min_precision_config_counts=required_eval_precision_counts,
                min_label_domain_counts=required_eval_label_counts,
                min_hardware_id_counts=required_eval_hardware_counts,
                require_checkpoint_files=bool(args.require_checkpoint_files),
                expected_runtime_backend=args.expected_deploy_runtime_backend,
                expected_runtime_backend_actual=args.expected_deploy_runtime_backend_actual,
                require_deployment_metadata=args.require_deployment_metadata,
            )
        )
        if args.require_deployment_student_checkpoint:
            deployment_checkpoint_linkage, checkpoint_linkage_errors = check_deployment_student_checkpoint(
                latest_row(rows, args.student_run_id),
                deployment_row,
            )
            errors.extend(checkpoint_linkage_errors)
    return {
        "ok": not errors,
        "errors": errors,
        "runs": summaries,
        "baseline_comparison": baseline_comparison,
        "training": training_summaries,
        "deployment": deployment_summary,
        "required_precision": required_precision,
        "deployment_required": bool(args.require_deployment_eval),
        "deployment_student_checkpoint_required": bool(args.require_deployment_student_checkpoint),
        "deployment_student_checkpoint": deployment_checkpoint_linkage,
        "checkpoint_files_required": bool(args.require_checkpoint_files),
        "required_label_domains": required_label_domains,
        "min_eval_precision_counts": required_eval_precision_counts,
        "min_eval_hardware_counts": required_eval_hardware_counts,
        "min_eval_label_counts": required_eval_label_counts,
        "source_precision_provenance_required": bool(args.require_source_precision_provenance),
        "source_precision_confirmed_required": bool(args.require_source_precision_confirmed),
        "required_train_label_domains": required_train_label_domains,
        "min_train_precision_counts": required_train_precision_counts,
        "min_train_hardware_counts": required_train_hardware_counts,
        "min_train_label_counts": required_train_label_counts,
        "min_train_split_counts": required_train_split_counts,
        "eval_train_checkpoints_required": bool(args.require_eval_train_checkpoints),
        "eval_train_checkpoints": eval_train_checkpoint_linkage,
        "training_required": bool(
            args.require_train_events
            or args.require_unlimited_train_data
            or args.require_source_train_provenance
            or args.require_source_train_precision_confirmed
            or args.require_train_checkpoint_metadata
            or args.require_train_lineage
            or args.require_eval_train_checkpoints
            or required_train_label_domains
            or require_train_label_counts
            or require_train_precision_counts
            or require_train_hardware_counts
            or require_train_split_counts
        ),
        "unlimited_train_data_required": bool(args.require_unlimited_train_data),
        "source_train_provenance_required": bool(args.require_source_train_provenance),
        "source_train_precision_confirmed_required": bool(args.require_source_train_precision_confirmed),
        "train_checkpoint_metadata_required": bool(args.require_train_checkpoint_metadata),
        "train_lineage_required": bool(args.require_train_lineage),
        "train_lineage": train_lineage,
        "required_split_unit": args.required_split_unit,
        "min_slice_counts": {
            "precision": max(1, int(args.min_precision_slices)),
            "label_domain": max(1, int(args.min_label_domain_slices)),
            "batch_size": max(1, int(args.min_batch_size_slices)),
            "resource_regime": max(1, int(args.min_resource_regime_slices)),
            "graph_signature": max(1, int(args.min_graph_signature_slices)),
            "graph_family": max(1, int(args.min_graph_family_slices)),
        },
        "materialization": materialization,
        "results": str(args.results),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["ok"]:
        print("precision transfer result check passed")
        print(json.dumps(report["runs"], sort_keys=True))
        return
    for error in report["errors"]:
        print(f"ERROR: {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
