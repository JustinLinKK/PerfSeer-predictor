#!/usr/bin/env python
"""Create scheduler WorkloadSpec rows from model manifests and dataset profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nrp_calibration_pack.workload import clean_id, normalize_workload_spec, write_jsonl  # noqa: E402


DEFAULT_REGISTRY = ROOT / "dataset_sources" / "registry.json"
DEFAULT_DATASET_PROFILE_ROOT = ROOT / "datasets" / "prepared"
DEFAULT_RAW_ROOT = ROOT / "datasets" / "raw"
DTYPE_BYTES = {
    "float32": 4.0,
    "fp32": 4.0,
    "int64": 8.0,
    "long": 8.0,
    "float16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int32": 4.0,
    "int16": 2.0,
    "int8": 1.0,
    "uint8": 1.0,
    "bool": 1.0,
}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create real-dataset scheduler workload specs.")
    parser.add_argument("--manifest", required=True, help="Path to nrp_calibration_pack/manifest/subset_manifest.jsonl.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to dataset_sources/registry.json.")
    parser.add_argument("--dataset-profile-root", default=str(DEFAULT_DATASET_PROFILE_ROOT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Path to local raw datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-id", action="append", help="Subset id to include. May be repeated. Defaults to tiny.")
    parser.add_argument("--batch-size", action="append", type=positive_int, help="Batch size to include. May be repeated.")
    parser.add_argument("--precision-sweep", default="fp32_ieee", help="Comma-separated precisions to store in WorkloadSpec.")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--hardware-id", default="unknown")
    parser.add_argument("--limit", type=positive_int, help="Optional number of unique models.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def unique_model_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        model_id = str(row.get("model_id") or "")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def registry_rows(path: Path) -> list[dict[str, Any]]:
    registry = json.loads(path.read_text())
    return list(registry.get("datasets", []))


def resolve_path(value: str | None, *, fallback: Path | None = None) -> Path | None:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    return fallback


def load_dataset_profiles(registry_path: Path, profile_root: Path, raw_root: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in registry_rows(registry_path):
        if str(row.get("status")) != "prepared":
            continue
        profile_value = str(row.get("prepared_profile") or "")
        profile_path = Path(profile_value) if profile_value else profile_root / str(row["id"]) / "dataset_profile.json"
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        if not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text())
        raw_summary = profile.get("raw_summary") if isinstance(profile.get("raw_summary"), dict) else {}
        raw_dir = resolve_path(str(raw_summary.get("raw_dir") or ""), fallback=raw_root / str(row["id"]))
        profile["_prepared_profile_path"] = str(profile_path)
        profile["_prepared_dir"] = str(profile_path.parent)
        profile["_raw_dir"] = str(raw_dir) if raw_dir is not None else ""
        profiles[str(row["id"])] = profile
    return profiles


def family_dataset_map(profiles: dict[str, dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for dataset_id, profile in profiles.items():
        for family in profile.get("model_families", []) or []:
            mapping.setdefault(str(family), dataset_id)
    return mapping


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def subset_info(profile: dict[str, Any], subset_id: str) -> dict[str, Any]:
    subsets = profile.get("subsets") if isinstance(profile.get("subsets"), dict) else {}
    info = subsets.get(subset_id)
    if not isinstance(info, dict):
        raise KeyError(subset_id)
    return info


def real_dataloader_available(profile: dict[str, Any], subset: dict[str, Any]) -> bool:
    subset_path = resolve_path(str(subset.get("path") or ""))
    raw_dir = Path(str(profile.get("_raw_dir") or ""))
    profile_path = Path(str(profile.get("_prepared_profile_path") or ""))
    return bool(subset_path and subset_path.is_file() and raw_dir.is_dir() and profile_path.is_file())


def input_specs_for_profile(profile: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    modality = str(profile.get("modality", "tensor"))
    task = str(profile.get("task_family", "unknown"))
    if modality == "text":
        shape = [batch_size, int(profile.get("sequence_length_p95", 512) or 512)]
        return [{"name": "input_ids", "shape": shape, "dtype": "int64", "kind": "token_ids"}]
    if modality == "audio":
        shape = [batch_size, int(profile.get("audio_frames_p95", 1024) or 1024), int(profile.get("audio_feature_dim", 80) or 80)]
        return [{"name": "audio_features", "shape": shape, "dtype": "float32", "kind": "float"}]
    if modality == "time_series":
        shape = [batch_size, int(profile.get("window_length", 128) or 128), int(profile.get("feature_count", 16) or 16)]
        return [{"name": "series", "shape": shape, "dtype": "float32", "kind": "float"}]
    if modality == "tabular":
        shape = [batch_size, int(profile.get("feature_count", 128) or 128)]
        return [{"name": "features", "shape": shape, "dtype": "float32", "kind": "float"}]
    if modality == "graph":
        node_count = int(profile.get("graph_nodes_p95", 1024) or 1024)
        feature_dim = int(profile.get("feature_count", 64) or 64)
        return [{"name": "node_features", "shape": [batch_size, node_count, feature_dim], "dtype": "float32", "kind": "float"}]
    if task == "image_segmentation" or task == "object_detection":
        image_size = int(profile.get("image_height_p95", 512) or 512)
    else:
        image_size = int(profile.get("image_height_p95", 224) or 224)
    return [{"name": "image", "shape": [batch_size, 3, image_size, image_size], "dtype": "float32", "kind": "float"}]


def specs_with_batch_size(input_specs: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, spec in enumerate(input_specs):
        item = dict(spec)
        shape = [int(dim) for dim in item.get("shape", [])]
        if not shape:
            shape = [batch_size, 1]
        shape[0] = int(batch_size)
        item["shape"] = shape
        item.setdefault("name", f"input{idx}")
        item.setdefault("dtype", "float32")
        item.setdefault("kind", "float")
        out.append(item)
    return out


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= max(int(value), 1)
    return out


def input_feature_metadata(input_specs: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    first_shape = [int(dim) for dim in (input_specs[0].get("shape") if input_specs else [batch_size, 1])]
    rank = len(first_shape)
    dims = first_shape[:6] + [0] * max(0, 6 - len(first_shape))
    total_numel = 0
    total_bytes = 0.0
    for spec in input_specs:
        shape = [int(dim) for dim in spec.get("shape", [])]
        dtype = str(spec.get("dtype", "float32")).lower()
        numel = product(shape)
        total_numel += numel
        total_bytes += float(numel) * DTYPE_BYTES.get(dtype, 4.0)
    per_sample_numel = total_numel / max(int(batch_size), 1)
    per_sample_bytes = total_bytes / max(int(batch_size), 1)
    return {
        "input_rank": rank,
        "input_dim0": dims[0],
        "input_dim1": dims[1],
        "input_dim2": dims[2],
        "input_dim3": dims[3],
        "input_dim4": dims[4],
        "input_dim5": dims[5],
        "input_numel_per_sample": per_sample_numel,
        "input_bytes_per_sample": per_sample_bytes,
        "input_bytes_per_batch": total_bytes,
    }


def make_workload(
    row: dict[str, Any],
    profile: dict[str, Any],
    subset_id: str,
    batch_size: int,
    precision: str,
    optimizer: str,
    hardware_id: str,
) -> dict[str, Any]:
    subset = subset_info(profile, subset_id)
    subset_path = resolve_path(str(subset.get("path") or ""))
    dataset_input_specs = input_specs_for_profile(profile, batch_size)
    model_input_specs = specs_with_batch_size(list(row.get("input_specs") or dataset_input_specs), batch_size)
    model_id = str(row["model_id"])
    dataset_id = str(profile["dataset_id"])
    raw_dir = str(profile.get("_raw_dir") or "")
    prepared_dir = str(profile.get("_prepared_dir") or "")
    prepared_profile_path = str(profile.get("_prepared_profile_path") or "")
    real_backed = real_dataloader_available(profile, subset)
    input_meta = input_feature_metadata(dataset_input_specs, batch_size)
    spec = {
        "workload_spec_version": 1,
        "model": {
            "model_id": model_id,
            "graph_id": row.get("graph_id", model_id),
            "source_path": row.get("model_file"),
            "entrypoint": "make_model",
            "architecture_family": row.get("architecture_family", "unknown"),
            "task_adapter": profile.get("task_family", "unknown"),
            "source_hash": row.get("source_sha256", ""),
            "input_specs": model_input_specs,
            "input_shape": model_input_specs[0]["shape"],
        },
        "dataset": {
            "dataset_id": dataset_id,
            "task_type": profile.get("task_family"),
            "modality": profile.get("modality"),
            "subset_id": subset_id,
            "subset_index_path": str(subset_path) if subset_path is not None else subset.get("path"),
            "subset_mask_path": str(subset_path) if subset_path is not None else subset.get("path"),
            "prepared_profile_path": prepared_profile_path,
            "prepared_dir": prepared_dir,
            "raw_dir": raw_dir,
            "num_samples": int(subset.get("num_samples") or profile.get("sample_count") or 1),
            "sample_count": int(profile.get("sample_count") or 1),
            "sample_bytes_mean": float(profile.get("sample_bytes_mean") or 0.0),
            "dataset_input_specs": dataset_input_specs,
            "dataset_input_shape": dataset_input_specs[0]["shape"],
            "input_rank": input_meta["input_rank"],
            "input_dim0": input_meta["input_dim0"],
            "input_dim1": input_meta["input_dim1"],
            "input_dim2": input_meta["input_dim2"],
            "input_dim3": input_meta["input_dim3"],
            "input_dim4": input_meta["input_dim4"],
            "input_dim5": input_meta["input_dim5"],
            "input_numel_per_sample": input_meta["input_numel_per_sample"],
            "input_bytes_per_sample": input_meta["input_bytes_per_sample"],
            "input_bytes_per_batch": input_meta["input_bytes_per_batch"],
            "input_specs": model_input_specs,
            "input_shape": model_input_specs[0]["shape"],
            "metadata_source": profile.get("metadata_source", "real_downloaded_files"),
            "real_dataloader_backed": real_backed,
            "preprocessing": profile.get("preprocessing", {}),
        },
        "training": {
            "batch_size": int(batch_size),
            "grad_accumulation_steps": 1,
            "optimizer": optimizer,
            "loss_type": profile.get("loss_type", "task_default"),
            "precision": precision,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
        },
        "hardware": {
            "hardware_id": hardware_id,
        },
    }
    return normalize_workload_spec(spec)


def write_outputs(workloads: list[dict[str, Any]], output_dir: Path, force: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "workloads": output_dir / "workloads.jsonl",
        "index": output_dir / "index.jsonl",
        "summary": output_dir / "summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{existing[0]} exists; pass --force to overwrite")
    write_jsonl(paths["workloads"], workloads)
    write_jsonl(paths["index"], workloads)
    by_point = output_dir / "by_profile_point"
    by_point.mkdir(exist_ok=True)
    for workload in workloads:
        (by_point / f"{clean_id(workload['profile_point_id'])}.json").write_text(
            json.dumps(workload, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "workload_spec_version": 1,
        "workloads": len(workloads),
        "dataset_ids": sorted({str(w["dataset"]["dataset_id"]) for w in workloads}),
        "subset_ids": sorted({str(w["dataset"]["subset_id"]) for w in workloads}),
        "precisions": sorted({str(w["training"]["precision"]) for w in workloads}),
        "optimizers": sorted({str(w["training"]["optimizer"]) for w in workloads}),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profiles = load_dataset_profiles(Path(args.registry), Path(args.dataset_profile_root), Path(args.raw_root))
    family_to_dataset = family_dataset_map(profiles)
    rows = unique_model_rows(Path(args.manifest), args.limit)
    subsets = args.subset_id or ["tiny"]
    batch_sizes = args.batch_size or [1]
    precisions = parse_csv_list(args.precision_sweep)
    workloads: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row in rows:
        family = str(row.get("architecture_family", "unknown"))
        dataset_id = family_to_dataset.get(family)
        if not dataset_id:
            skipped[family] = skipped.get(family, 0) + 1
            continue
        profile = profiles[dataset_id]
        for subset_id in subsets:
            for batch_size in batch_sizes:
                for precision in precisions:
                    try:
                        workloads.append(
                            make_workload(row, profile, subset_id, batch_size, precision, args.optimizer, args.hardware_id)
                        )
                    except KeyError:
                        skipped[f"{dataset_id}:{subset_id}"] = skipped.get(f"{dataset_id}:{subset_id}", 0) + 1
    summary = write_outputs(workloads, Path(args.output_dir), args.force)
    summary["skipped"] = skipped
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
