#!/usr/bin/env python
"""Rebuild a PerfSeer dataset from a source-only NRP labels tarball."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_source_converter import convert_generated_source_to_networkx  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild dataset/cg and labels from a source-only NRP tarball.")
    parser.add_argument("--source-tar", required=True, help="Source-only tarball from nrp_calibration_pack/package_source_tar.py.")
    parser.add_argument("--out-root", required=True, help="Dataset root to write.")
    parser.add_argument("--work-dir", help="Optional extraction directory to keep for audit/debugging.")
    parser.add_argument(
        "--precision-config",
        action="append",
        default=[],
        help="Only materialize matching precision config(s). May be repeated or comma-separated.",
    )
    parser.add_argument("--force", action="store_true", help="Remove existing --out-root before writing.")
    return parser.parse_args(argv)


def clean_id(value: str | None, default: str = "unknown") -> str:
    raw = (value or default).strip().lower()
    raw = re.sub(r"[^a-z0-9_.+-]+", "_", raw).strip("_")
    return raw or default


def safe_extract(tar_path: Path, dest: Path) -> None:
    root = dest.resolve()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"tar member escapes extraction root: {member.name}")
        tar.extractall(dest)


def iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r") as fh:
            for line in fh:
                if line.strip():
                    yield path, json.loads(line)


def load_manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "pack" / "manifest" / "subset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"source package does not contain {manifest_path.relative_to(root)}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _path, row in iter_jsonl([manifest_path]):
        model_id = str(row.get("model_id") or row.get("graph_id") or "")
        if not model_id or model_id in seen:
            continue
        rows.append(row)
        seen.add(model_id)
    return rows


def graph_metadata(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "architecture_family",
        "feature_schema_version",
        "input_specs",
        "model_id",
        "original_stem",
        "variant_kind",
        "variant_signature",
    )
    return {key: row[key] for key in keys if key in row}


def write_graphs(extracted: Path, out_root: Path, rows: list[dict[str, Any]]) -> int:
    graph_dir = out_root / "cg" / "cg"
    graph_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        model_id = str(row["model_id"])
        model_file = extracted / "pack" / str(row["model_file"])
        graph = convert_generated_source_to_networkx(model_file, metadata=graph_metadata(row))
        graph.graph["source_model_file"] = str(Path("models") / f"{model_id}.py")
        with (graph_dir / f"{model_id}.pkl").open("wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        count += 1
    return count


def hardware_id_for_row(row: dict[str, Any]) -> str:
    raw = row.get("hardware_id")
    hardware = row.get("hardware") if isinstance(row.get("hardware"), dict) else {}
    if not raw:
        raw = hardware.get("hardware_id") or hardware.get("gpu_name") or hardware.get("device")
    return clean_id(str(raw or "unknown"))


def label_text_for_row(row: dict[str, Any], result_root: Path) -> str | None:
    label = row.get("label")
    if isinstance(label, dict):
        return repr(label) + "\n"
    label_file = str(row.get("label_file") or "")
    if label_file:
        label_path = result_root / label_file
        if label_path.exists():
            return label_path.read_text()
    return None


def parse_precision_filter(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def write_labels_and_metadata(
    extracted: Path,
    out_root: Path,
    model_ids: set[str],
    precision_filter: set[str],
) -> tuple[int, int]:
    label_dir = out_root / "label" / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_root / "label" / "precision_metadata.jsonl"
    rejected_path = out_root / "precision_rejected_rows.jsonl"
    result_roots = sorted((extracted / "results").glob("*")) if (extracted / "results").exists() else []
    result_paths = [path for root in result_roots for path in sorted(root.glob("results_shard*.jsonl"))]
    accepted = 0
    rejected = 0
    with metadata_path.open("w") as metadata_fh, rejected_path.open("w") as rejected_fh:
        for result_path, row in iter_jsonl(result_paths):
            model_id = str(row.get("model_id") or row.get("graph_id") or "")
            precision_config = str(row.get("precision_config") or row.get("precision", {}).get("precision_config") or "unknown")
            status = str(row.get("status") or "unknown")
            hw_id = hardware_id_for_row(row)
            if precision_filter and precision_config not in precision_filter:
                continue
            if status != "ok":
                rejected_fh.write(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "hardware_id": hw_id,
                            "precision_config": precision_config,
                            "status": status,
                            "error": row.get("error"),
                            "profile_point_id": row.get("profile_point_id"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                rejected += 1
                continue
            if model_id not in model_ids:
                continue
            label_text = label_text_for_row(row, result_path.parent)
            if label_text is None:
                rejected_fh.write(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "hardware_id": hw_id,
                            "precision_config": precision_config,
                            "status": "missing_label",
                            "profile_point_id": row.get("profile_point_id"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                rejected += 1
                continue
            label_name = f"{model_id}_{hw_id}_{precision_config}.txt"
            label_file = f"label/label/{label_name}"
            (out_root / label_file).write_text(label_text)
            metadata = {
                "graph_id": model_id,
                "graph_file": f"cg/cg/{model_id}.pkl",
                "label_file": label_file,
                "label_stem": Path(label_name).stem,
                "hardware_id": hw_id,
                "precision_config": precision_config,
                "profile_point_id": row.get("profile_point_id") or f"{model_id}::{precision_config}",
                "source_result_status": status,
                "label_domain": "precision_profile",
                "precision": row.get("precision", {}),
                "hardware": row.get("hardware", {}),
            }
            metadata_fh.write(json.dumps(metadata, sort_keys=True) + "\n")
            accepted += 1
    return accepted, rejected


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_tar = Path(args.source_tar).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists():
        if not args.force:
            raise SystemExit(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    if args.work_dir:
        extracted = Path(args.work_dir).resolve()
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True)
        cleanup = False
    else:
        temp = tempfile.TemporaryDirectory()
        extracted = Path(temp.name)
        cleanup = True

    try:
        safe_extract(source_tar, extracted)
        rows = load_manifest_rows(extracted)
        graph_count = write_graphs(extracted, out_root, rows)
        precision_filter = parse_precision_filter(args.precision_config)
        label_count, rejected_count = write_labels_and_metadata(
            extracted,
            out_root,
            {str(row["model_id"]) for row in rows},
            precision_filter,
        )
        report = {
            "source_tar": str(source_tar),
            "graphs": graph_count,
            "precision_labels": label_count,
            "precision_filter": sorted(precision_filter),
            "rejected_rows": rejected_count,
            "metadata_file": "label/precision_metadata.jsonl",
            "rejected_rows_file": "precision_rejected_rows.jsonl",
        }
        (out_root / "source_tar_rebuild_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        if cleanup:
            temp.cleanup()  # type: ignore[has-type]


if __name__ == "__main__":
    main()
