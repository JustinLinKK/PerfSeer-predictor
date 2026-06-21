#!/usr/bin/env python
"""Materialize precision calibration profiler results as a PerfSeer dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer.data import list_pairs  # noqa: E402

SOURCE_UNKNOWN_PRECISION_CONFIG = "source_domain_unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize precision calibration results into dataset/cg + dataset/label layout.")
    p.add_argument("--pack-dir", required=True, help="Generated calibration pack directory containing subset/ and manifest/.")
    p.add_argument("--results-dir", required=True, help="Profiler output directory containing results_shard*.jsonl.")
    p.add_argument("--out-root", required=True, help="Output dataset root.")
    p.add_argument("--base-data-root", help="Optional original dataset root to include alongside precision labels.")
    p.add_argument("--base-mode", choices=("skip", "copy", "symlink"), default="skip")
    p.add_argument("--source-precision-config", default="fp32_ieee", help="Precision config to assign to labels copied from --base-data-root.")
    p.add_argument("--source-hardware-id", default="source_domain_unknown", help="Hardware id to assign to labels copied from --base-data-root.")
    p.add_argument("--source-hardware-features-json", default="{}", help="JSON object of numeric hardware features for labels copied from --base-data-root.")
    p.add_argument("--source-precision-provenance", default="", help="Short note/path/URI proving the original source labels' precision setup.")
    p.add_argument("--require-source-precision-provenance", action="store_true", help="Fail when source-domain labels are included without provenance.")
    p.add_argument("--hardware-id", help="Override hardware id used in materialized label filenames.")
    p.add_argument("--pseudo-precision-sweep", default="", help="Comma-separated precision configs to add as pseudo rows backed by source labels for teacher distillation.")
    p.add_argument("--pseudo-hardware-id", help="Hardware id to assign to pseudo rows. Defaults to --hardware-id or --source-hardware-id.")
    p.add_argument("--pseudo-hardware-features-json", default="{}", help="JSON object of numeric hardware features for pseudo rows.")
    p.add_argument("--force", action="store_true", help="Remove existing output root before writing.")
    return p.parse_args(argv)


def clean_id(value: str | None, default: str = "unknown") -> str:
    raw = value or default
    raw = raw.strip().lower()
    raw = re.sub(r"[^a-z0-9_.+-]+", "_", raw)
    raw = raw.strip("_")
    return raw or default


def normalize_precision_config(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    aliases = {
        "fp32": "fp32_ieee",
        "float32": "fp32_ieee",
        "fp32_ieee": "fp32_ieee",
        "tf32": "tf32",
        "bf16": "bf16_amp",
        "bf16_amp": "bf16_amp",
        "fp16": "fp16_amp",
        "float16": "fp16_amp",
        "fp16_amp": "fp16_amp",
        "fp8": "fp8_te_hybrid",
        "fp8_te": "fp8_te_hybrid",
        "fp8_te_hybrid": "fp8_te_hybrid",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "source_unknown": SOURCE_UNKNOWN_PRECISION_CONFIG,
        "source_domain_unknown": SOURCE_UNKNOWN_PRECISION_CONFIG,
    }
    if key == "bf32":
        raise ValueError("bf32 is ambiguous; use tf32 or bf16_amp")
    if key not in aliases:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unknown precision config {value!r}; expected one of: {allowed}")
    return aliases[key]


def parse_hardware_features(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--source-hardware-features-json must be a JSON object")
    return data


def parse_precision_sweep(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        precision = normalize_precision_config(item)
        if precision not in seen:
            out.append(precision)
            seen.add(precision)
    return out


def is_source_precision_confirmed(precision_config: str, provenance: str) -> bool:
    return bool(str(provenance or "").strip()) and normalize_precision_config(precision_config) != SOURCE_UNKNOWN_PRECISION_CONFIG


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_manifest(pack_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = pack_dir / "manifest" / "subset_manifest.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl([manifest]):
        key = f"{row.get('model_id')}::{row.get('precision_config')}"
        rows[key] = row
        rows.setdefault(str(row.get("model_id")), row)
    return rows


def hardware_id_from_result(row: dict[str, Any], override: str | None = None) -> str:
    if override:
        return clean_id(override)
    if row.get("hardware_id"):
        return clean_id(str(row["hardware_id"]))
    hardware = row.get("hardware") if isinstance(row.get("hardware"), dict) else {}
    gpu_name = hardware.get("gpu_name") or hardware.get("device") or "unknown"
    cc = hardware.get("compute_capability")
    return clean_id(f"{gpu_name}_cc{cc}" if cc else str(gpu_name))


def hardware_features(row: dict[str, Any]) -> dict[str, Any]:
    hardware = row.get("hardware") if isinstance(row.get("hardware"), dict) else {}
    out: dict[str, Any] = {}
    if "compute_capability" in hardware:
        out["compute_capability"] = hardware["compute_capability"]
    if "multi_processor_count" in hardware:
        out["sm_count"] = hardware["multi_processor_count"]
    if "total_memory_mib" in hardware:
        out["vram_gib"] = float(hardware["total_memory_mib"]) / 1024.0
    for key in (
        "architecture_id",
        "memory_bandwidth_gbps",
        "l2_cache_mib",
        "peak_fp32_tflops",
        "peak_tf32_tflops",
        "peak_fp16_bf16_tflops",
        "peak_fp8_tflops",
    ):
        if key in hardware:
            out[key] = hardware[key]
    return out


def precision_config_from_result(row: dict[str, Any]) -> str:
    precision = row.get("precision") if isinstance(row.get("precision"), dict) else {}
    raw = str(row.get("precision_config") or precision.get("precision_config") or "fp32_ieee")
    try:
        return normalize_precision_config(raw)
    except ValueError:
        return raw.strip().lower().replace("-", "_") or "unknown"


def fallback_policy_from_result(row: dict[str, Any]) -> str:
    precision = row.get("precision") if isinstance(row.get("precision"), dict) else {}
    return str(precision.get("fallback_policy") or row.get("fallback_policy") or "none")


def bump(mapping: dict[str, Any], key: str, amount: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + amount


def bump_nested(mapping: dict[str, Any], outer: str, inner: str, amount: int = 1) -> None:
    bucket = mapping.setdefault(outer, {})
    bump(bucket, inner, amount)


def rejected_row_summary(row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    precision_config = precision_config_from_result(row)
    return {
        "status": status,
        "reason": reason,
        "model_id": row.get("model_id"),
        "graph_id": row.get("graph_id"),
        "profile_point_id": row.get("profile_point_id"),
        "precision_config": precision_config,
        "hardware_id": row.get("hardware_id"),
        "fallback_policy": fallback_policy_from_result(row),
        "error": row.get("error"),
        "precision": row.get("precision", {}),
        "hardware": row.get("hardware", {}),
    }


def record_skip(report: dict[str, Any], row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    precision_config = precision_config_from_result(row)
    fallback_policy = fallback_policy_from_result(row)
    bump(report["skipped"], status)
    bump_nested(report["skipped_by_precision"], precision_config, status)
    bump_nested(report["skipped_by_status"], status, precision_config)
    if fallback_policy and fallback_policy != "none":
        bump_nested(report["fallback_policy_counts"], fallback_policy, precision_config)
    if precision_config.startswith("fp8_") and status in {"unsupported_precision", "unsupported", "error", "oom"}:
        bump(report, "unsupported_fp8_rows")
    return rejected_row_summary(row, status, reason)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def include_base_dataset(
    base_root: Path,
    out_root: Path,
    mode: str,
    source_precision_config: str,
    source_hardware_id: str,
    source_hardware_features: dict[str, Any],
    source_precision_provenance: str,
) -> tuple[int, list[dict[str, Any]]]:
    if mode == "skip":
        return 0, []
    count = 0
    metadata_rows: list[dict[str, Any]] = []
    for graph_path, label_path in list_pairs(str(base_root)):
        graph_file = f"cg/cg/{Path(graph_path).name}"
        label_file = f"label/label/{Path(label_path).name}"
        graph_dst = out_root / graph_file
        label_dst = out_root / label_file
        link_or_copy(Path(graph_path), graph_dst, mode)
        link_or_copy(Path(label_path), label_dst, mode)
        label_stem = Path(label_path).stem
        graph_id = Path(graph_path).stem
        metadata_rows.append(
            {
                "graph_id": graph_id,
                "graph_file": graph_file,
                "label_file": label_file,
                "label_stem": label_stem,
                "hardware_id": source_hardware_id,
                "precision_config": source_precision_config,
                "profile_point_id": f"{graph_id}::{source_precision_config}",
                "source_result_status": "source_domain",
                "label_domain": "source",
                "is_base_label": True,
                "source_precision_provenance": source_precision_provenance,
                "source_precision_confirmed": is_source_precision_confirmed(source_precision_config, source_precision_provenance),
                "hardware_features": source_hardware_features,
                "hardware": {"hardware_id": source_hardware_id, **source_hardware_features},
                "precision": {"precision_config": source_precision_config},
            }
        )
        count += 1
    return count, metadata_rows


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    pack_dir = Path(args.pack_dir)
    results_dir = Path(args.results_dir)
    out_root = Path(args.out_root)
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    (out_root / "cg" / "cg").mkdir(parents=True, exist_ok=True)
    (out_root / "label" / "label").mkdir(parents=True, exist_ok=True)

    source_precision = normalize_precision_config(args.source_precision_config)
    source_hardware_id = clean_id(args.source_hardware_id, "source_domain_unknown")
    source_hardware_features = parse_hardware_features(args.source_hardware_features_json)
    source_precision_provenance = str(args.source_precision_provenance or "").strip()
    source_precision_confirmed = is_source_precision_confirmed(source_precision, source_precision_provenance)
    if args.require_source_precision_provenance and args.base_data_root and args.base_mode != "skip" and not source_precision_provenance:
        raise ValueError("--require-source-precision-provenance was set but --source-precision-provenance is empty")
    pseudo_precision_sweep = parse_precision_sweep(args.pseudo_precision_sweep)
    pseudo_hardware_id = clean_id(args.pseudo_hardware_id or args.hardware_id or source_hardware_id, source_hardware_id)
    pseudo_hardware_features = parse_hardware_features(args.pseudo_hardware_features_json)
    base_count, source_metadata_rows = (
        include_base_dataset(
            Path(args.base_data_root),
            out_root,
            args.base_mode,
            source_precision,
            source_hardware_id,
            source_hardware_features,
            source_precision_provenance,
        )
        if args.base_data_root
        else (0, [])
    )
    manifest = load_manifest(pack_dir)
    result_paths = sorted(results_dir.glob("results_shard*.jsonl"))
    metadata_path = out_root / "label" / "precision_metadata.jsonl"
    rejected_path = out_root / "precision_rejected_rows.jsonl"
    report = {
        "base_pairs": base_count,
        "source_metadata_labels": len(source_metadata_rows),
        "calibration_source_labels": 0,
        "source_precision_config": source_precision,
        "source_hardware_id": source_hardware_id,
        "source_precision_provenance": source_precision_provenance,
        "source_precision_confirmed": source_precision_confirmed,
        "precision_labels": 0,
        "pseudo_labels": 0,
        "pseudo_precision_sweep": pseudo_precision_sweep,
        "pseudo_hardware_id": pseudo_hardware_id if pseudo_precision_sweep else "",
        "pseudo_hardware_features": pseudo_hardware_features if pseudo_precision_sweep else {},
        "skipped": {},
        "skipped_by_precision": {},
        "skipped_by_status": {},
        "fallback_policy_counts": {},
        "unsupported_fp8_rows": 0,
        "rejected_rows_file": str(rejected_path.name),
        "result_files": [str(path) for path in result_paths],
    }
    seen_labels: set[str] = {str(row.get("label_file", "")) for row in source_metadata_rows}
    seen_source_labels: set[str] = set(seen_labels)
    source_candidates: list[dict[str, Any]] = list(source_metadata_rows)
    accepted_precision_keys: set[tuple[str, str, str]] = set()

    with metadata_path.open("w") as meta_fh, rejected_path.open("w") as rejected_fh:
        for meta in source_metadata_rows:
            meta_fh.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in iter_jsonl(result_paths):
            status = str(row.get("status", ""))
            if status != "ok":
                summary = record_skip(report, row, status or "unknown", "profiler_status")
                rejected_fh.write(json.dumps(summary, sort_keys=True) + "\n")
                continue
            model_id = str(row.get("model_id"))
            precision_config = precision_config_from_result(row)
            manifest_row = manifest.get(f"{model_id}::{precision_config}") or manifest.get(model_id) or {}
            graph_rel = manifest_row.get("subset_graph_file") or f"subset/cg/cg/{model_id}.pkl"
            graph_src = pack_dir / str(graph_rel)
            if not graph_src.exists():
                summary = record_skip(report, row, "missing_graph", str(graph_rel))
                rejected_fh.write(json.dumps(summary, sort_keys=True) + "\n")
                continue
            graph_file = f"cg/cg/{model_id}.pkl"
            shutil.copy2(graph_src, out_root / graph_file)

            base_label_file = str(manifest_row.get("base_label_file") or f"label/label/{model_id}.txt")
            original_label_raw = str(manifest_row.get("original_label_path") or "")
            original_label_path = Path(original_label_raw)
            if base_label_file not in seen_source_labels and original_label_raw and original_label_path.is_file():
                shutil.copy2(original_label_path, out_root / base_label_file)
                base_label_name = Path(base_label_file).name
                source_meta = {
                    "graph_id": model_id,
                    "graph_file": graph_file,
                    "label_file": base_label_file,
                    "label_stem": Path(base_label_name).stem,
                    "hardware_id": source_hardware_id,
                    "precision_config": source_precision,
                    "profile_point_id": f"{model_id}::{source_precision}",
                    "source_result_status": "source_domain",
                    "label_domain": "source",
                    "is_base_label": True,
                    "source_precision_provenance": source_precision_provenance,
                    "source_precision_confirmed": source_precision_confirmed,
                    "hardware_features": source_hardware_features,
                    "hardware": {"hardware_id": source_hardware_id, **source_hardware_features},
                    "precision": {"precision_config": source_precision},
                }
                meta_fh.write(json.dumps(source_meta, sort_keys=True) + "\n")
                source_candidates.append(source_meta)
                seen_source_labels.add(base_label_file)
                seen_labels.add(base_label_file)
            report["calibration_source_labels"] += 1

            hw_id = hardware_id_from_result(row, args.hardware_id)
            label_name = f"{model_id}_{hw_id}_{precision_config}.txt"
            label_file = f"label/label/{label_name}"
            if label_file in seen_labels:
                summary = record_skip(report, row, "duplicate_label", label_file)
                rejected_fh.write(json.dumps(summary, sort_keys=True) + "\n")
                continue
            seen_labels.add(label_file)
            label = row.get("label")
            if not isinstance(label, dict):
                summary = record_skip(report, row, "missing_label", "result row has no dataset label dict")
                rejected_fh.write(json.dumps(summary, sort_keys=True) + "\n")
                continue
            (out_root / label_file).write_text(repr(label) + "\n")
            meta = {
                "graph_id": model_id,
                "graph_file": graph_file,
                "label_file": label_file,
                "label_stem": Path(label_name).stem,
                "hardware_id": hw_id,
                "precision_config": precision_config,
                "base_label_file": base_label_file,
                "profile_point_id": row.get("profile_point_id"),
                "source_result_status": status,
                "hardware_features": hardware_features(row),
                "hardware": row.get("hardware", {}),
                "precision": row.get("precision", {}),
            }
            meta_fh.write(json.dumps(meta, sort_keys=True) + "\n")
            accepted_precision_keys.add((model_id, hw_id, precision_config))
            report["precision_labels"] += 1

        for source_meta in source_candidates:
            graph_id = str(source_meta.get("graph_id") or "")
            graph_file = str(source_meta.get("graph_file") or "")
            base_label_file = str(source_meta.get("label_file") or "")
            if not graph_id or not graph_file or not base_label_file:
                continue
            base_label_path = out_root / base_label_file
            if not base_label_path.is_file():
                continue
            for precision_config in pseudo_precision_sweep:
                if (graph_id, pseudo_hardware_id, precision_config) in accepted_precision_keys:
                    continue
                label_name = f"{graph_id}_{pseudo_hardware_id}_{precision_config}_pseudo.txt"
                label_file = f"label/label/{label_name}"
                if label_file in seen_labels:
                    continue
                seen_labels.add(label_file)
                (out_root / label_file).write_text(base_label_path.read_text())
                meta = {
                    "graph_id": graph_id,
                    "graph_file": graph_file,
                    "label_file": label_file,
                    "label_stem": Path(label_name).stem,
                    "hardware_id": pseudo_hardware_id,
                    "precision_config": precision_config,
                    "base_label_file": base_label_file,
                    "profile_point_id": f"{graph_id}::{pseudo_hardware_id}::{precision_config}::pseudo",
                    "source_result_status": "pseudo",
                    "label_domain": "pseudo",
                    "is_pseudo_label": True,
                    "hardware_features": pseudo_hardware_features,
                    "hardware": {"hardware_id": pseudo_hardware_id, **pseudo_hardware_features},
                    "precision": {"precision_config": precision_config},
                }
                meta_fh.write(json.dumps(meta, sort_keys=True) + "\n")
                report["pseudo_labels"] += 1

    (out_root / "precision_materialization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    report = materialize(parse_args(argv))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
