#!/usr/bin/env python
"""Approval-gated dataset source download and preparation helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "dataset_sources" / "registry.json"
DEFAULT_RAW_ROOT = ROOT / "datasets" / "raw"
DEFAULT_PREPARED_ROOT = ROOT / "datasets" / "prepared"
APPROVED_STATES = {"approved", "downloaded", "prepared"}
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".dcm", ".ogg", ".wav", ".flac"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage approved real dataset sources.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to dataset_sources/registry.json.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List dataset ids, tier, and approval status.")
    list_p.add_argument(
        "--tier",
        choices=("all", "local", "nautilus"),
        default="all",
        help="Filter candidates by execution tier.",
    )
    list_p.add_argument("--ids-only", action="store_true", help="Print only dataset ids.")

    show_p = sub.add_parser("show", help="Print one dataset registry entry.")
    show_p.add_argument("dataset_id")

    approve_p = sub.add_parser("approve", help="Mark one dataset source approved after manual review.")
    approve_p.add_argument("dataset_id")

    download_p = sub.add_parser("download", help="Download an approved Kaggle dataset or competition.")
    download_p.add_argument("dataset_id")
    download_p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    download_p.add_argument("--dry-run", action="store_true")
    download_p.add_argument(
        "--allow-nautilus-only",
        action="store_true",
        help="Permit downloading a Nautilus-only dataset. Use only on a Nautilus PVC-backed worker.",
    )

    ogb_p = sub.add_parser("download-ogb", help="Download an approved OGB/PyG dataset through OGB loaders.")
    ogb_p.add_argument("dataset_id")
    ogb_p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    ogb_p.add_argument("--dry-run", action="store_true")
    ogb_p.add_argument(
        "--allow-nautilus-only",
        action="store_true",
        help="Permit downloading a Nautilus-only dataset. Use only on a Nautilus PVC-backed worker.",
    )

    prep_p = sub.add_parser("prepare", help="Create deterministic subset masks and metadata for an approved dataset.")
    prep_p.add_argument("dataset_id")
    prep_p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    prep_p.add_argument("--prepared-root", default=str(DEFAULT_PREPARED_ROOT))
    prep_p.add_argument("--dry-run", action="store_true")
    prep_p.add_argument("--force", action="store_true")

    return parser.parse_args(argv)


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n")


def datasets_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in registry.get("datasets", [])}


def get_dataset(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    rows = datasets_by_id(registry)
    if dataset_id not in rows:
        allowed = ", ".join(sorted(rows))
        raise SystemExit(f"unknown dataset_id {dataset_id!r}; expected one of: {allowed}")
    return rows[dataset_id]


def require_approved(row: dict[str, Any]) -> None:
    status = str(row.get("status", "pending"))
    if status not in APPROVED_STATES:
        raise SystemExit(
            f"{row['id']} is {status!r}, not approved. Review {row.get('approval_doc')} and run "
            f"`python scripts/manage_dataset_sources.py approve {row['id']}` first."
        )


def require_download_tier(row: dict[str, Any], allow_nautilus_only: bool) -> None:
    tier = str(row.get("target_tier", "local"))
    if tier == "nautilus" and not allow_nautilus_only:
        replacement = row.get("local_replacement_id")
        replacement_msg = f" Use local replacement `{replacement}` instead." if replacement else ""
        raise SystemExit(
            f"{row['id']} is marked Nautilus-only; refusing local download.{replacement_msg} "
            "Pass --allow-nautilus-only only on a Nautilus PVC-backed job."
        )


def dataset_raw_dir(raw_root: Path, row: dict[str, Any]) -> Path:
    return raw_root / str(row["id"])


def dataset_prepared_dir(prepared_root: Path, row: dict[str, Any]) -> Path:
    return prepared_root / str(row["id"])


def command_with_output_dir(row: dict[str, Any], output_dir: Path) -> list[str]:
    cmd = [str(part) for part in row.get("download_command", [])]
    if not cmd:
        raise SystemExit(f"{row['id']} has no download_command")
    if row.get("source_type") in {"kaggle_competition", "kaggle_dataset"} and "-p" not in cmd and "--path" not in cmd:
        cmd.extend(["-p", str(output_dir)])
    return cmd


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))


def file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path, limit: int | None = None) -> tuple[int, list[str]]:
    count = 0
    header: list[str] = []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            for count, _row in enumerate(reader, start=1):
                if limit is not None and count >= limit:
                    break
    except Exception:
        return 0, []
    return count, header


def csv_row_count_from_text_io(fh: Any, limit: int | None = None) -> tuple[int, list[str]]:
    count = 0
    header: list[str] = []
    try:
        reader = csv.reader(fh)
        header = next(reader, [])
        for count, _row in enumerate(reader, start=1):
            if limit is not None and count >= limit:
                break
    except Exception:
        return 0, []
    return count, header


def zip_archives(raw_dir: Path) -> list[Path]:
    return [path for path in iter_files(raw_dir) if path.suffix.lower() == ".zip"]


def zip_member_key(raw_dir: Path, archive: Path, member: str) -> str:
    return f"{archive.relative_to(raw_dir)}::{member}"


def zip_csv_row_count(archive: Path, member: str) -> tuple[int, list[str]]:
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as raw_fh:
            text_fh = io.TextIOWrapper(raw_fh, encoding="utf-8", errors="ignore", newline="")
            return csv_row_count_from_text_io(text_fh)


def nested_zip_csv_row_count(archive: Path, nested_member: str, inner_member: str | None = None) -> tuple[int, list[str]]:
    with zipfile.ZipFile(archive) as zf:
        with zf.open(nested_member) as raw_nested:
            nested_bytes = io.BytesIO(raw_nested.read())
    with zipfile.ZipFile(nested_bytes) as nested_zf:
        member = inner_member or next(
            name for name in nested_zf.namelist() if name.lower().endswith(".csv")
        )
        with nested_zf.open(member) as raw_fh:
            text_fh = io.TextIOWrapper(raw_fh, encoding="utf-8", errors="ignore", newline="")
            return csv_row_count_from_text_io(text_fh)


def archive_summaries(raw_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for archive in zip_archives(raw_dir):
        item: dict[str, Any] = {"path": str(archive.relative_to(raw_dir))}
        if not zipfile.is_zipfile(archive):
            item["valid_zip"] = False
            summaries.append(item)
            continue
        with zipfile.ZipFile(archive) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            suffix_counts: dict[str, int] = {}
            for info in infos:
                suffix = Path(info.filename).suffix.lower() or "<none>"
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            item.update(
                {
                    "valid_zip": True,
                    "member_count": len(infos),
                    "uncompressed_bytes": int(sum(info.file_size for info in infos)),
                    "suffix_counts": dict(sorted(suffix_counts.items())),
                    "first_members": [info.filename for info in infos[:32]],
                }
            )
        summaries.append(item)
    return summaries


def summarize_raw_files(raw_dir: Path) -> dict[str, Any]:
    files = iter_files(raw_dir)
    sizes = [path.stat().st_size for path in files]
    csv_summaries: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".csv":
            rows, header = csv_row_count(path)
            csv_summaries.append(
                {
                    "path": str(path.relative_to(raw_dir)),
                    "rows": rows,
                    "columns": len(header),
                    "header": header[:32],
                }
            )
    return {
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "total_bytes": int(sum(sizes)),
        "min_file_bytes": int(min(sizes)) if sizes else 0,
        "max_file_bytes": int(max(sizes)) if sizes else 0,
        "csv_files": csv_summaries,
        "csv_total_rows": int(sum(item["rows"] for item in csv_summaries)),
        "archives": archive_summaries(raw_dir),
        "files_sha256": [
            {"path": str(path.relative_to(raw_dir)), "sha256": file_digest(path)}
            for path in files[:256]
        ],
    }


def media_keys_from_zip(raw_dir: Path, archive: Path, prefixes: tuple[str, ...], suffixes: set[str]) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return [
            zip_member_key(raw_dir, archive, info.filename)
            for info in zf.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() in suffixes
            and (not prefixes or info.filename.startswith(prefixes))
        ]


def row_keys(prefix: str, count: int) -> list[str]:
    return [f"{prefix}:row:{idx:012d}" for idx in range(count)]


def archive_for_dataset(raw_dir: Path) -> Path | None:
    archives = [path for path in zip_archives(raw_dir) if zipfile.is_zipfile(path)]
    if len(archives) == 1:
        return archives[0]
    return None


def ogb_node_keys(raw_dir: Path, slug: str) -> list[str]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=slug, root=str(raw_dir))
    data = dataset[0]
    num_nodes = int(getattr(data, "num_nodes"))
    return [f"node:{idx:012d}" for idx in range(num_nodes)]


def sample_keys_for_dataset(row: dict[str, Any], raw_dir: Path, summary: dict[str, Any]) -> list[str]:
    dataset_id = str(row["id"])
    if row.get("source_type") == "pyg_ogb":
        return ogb_node_keys(raw_dir, str(row["slug"]))

    archive = archive_for_dataset(raw_dir)
    if archive is not None:
        if dataset_id == "cassava_leaf_disease":
            return media_keys_from_zip(raw_dir, archive, ("train_images/",), {".jpg", ".jpeg", ".png"})
        if dataset_id == "siim_acr_pneumothorax":
            return media_keys_from_zip(raw_dir, archive, ("stage_2_images/",), {".dcm"})
        if dataset_id == "great_barrier_reef":
            return media_keys_from_zip(raw_dir, archive, ("train_images/",), {".jpg", ".jpeg", ".png"})
        if dataset_id == "birdclef_2023":
            return media_keys_from_zip(raw_dir, archive, ("train_audio/",), {".ogg", ".wav", ".flac"})
        if dataset_id == "jigsaw_toxic_comment":
            rows, _header = nested_zip_csv_row_count(archive, "train.csv.zip", "train.csv")
            return row_keys(zip_member_key(raw_dir, archive, "train.csv.zip::train.csv"), rows)
        if dataset_id == "cnn_dailymail_summarization":
            rows, _header = zip_csv_row_count(archive, "cnn_dailymail/train.csv")
            return row_keys(zip_member_key(raw_dir, archive, "cnn_dailymail/train.csv"), rows)
        if dataset_id == "store_sales_time_series":
            rows, _header = zip_csv_row_count(archive, "train.csv")
            return row_keys(zip_member_key(raw_dir, archive, "train.csv"), rows)
        if dataset_id == "home_credit_default_risk":
            rows, _header = zip_csv_row_count(archive, "application_train.csv")
            return row_keys(zip_member_key(raw_dir, archive, "application_train.csv"), rows)
        if dataset_id == "credit_card_default":
            rows, _header = zip_csv_row_count(archive, "UCI_Credit_Card.csv")
            return row_keys(zip_member_key(raw_dir, archive, "UCI_Credit_Card.csv"), rows)
        with zipfile.ZipFile(archive) as zf:
            media = [
                zip_member_key(raw_dir, archive, info.filename)
                for info in zf.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in MEDIA_SUFFIXES
            ]
            if media:
                return media
            return [
                zip_member_key(raw_dir, archive, info.filename)
                for info in zf.infolist()
                if not info.is_dir()
            ]

    files = iter_files(raw_dir)
    non_archives = [path for path in files if path.suffix.lower() != ".zip"]
    if non_archives:
        return [str(path.relative_to(raw_dir)) for path in non_archives]
    csv_rows = int(summary.get("csv_total_rows") or 0)
    if csv_rows > 0:
        return [f"csv_row_{idx:012d}" for idx in range(csv_rows)]
    return []


def subset_masks(keys: list[str], subset_sizes: dict[str, int | None], seed: str) -> dict[str, list[str]]:
    ordered = sorted(keys, key=lambda key: stable_hash(f"{seed}:{key}"))
    masks: dict[str, list[str]] = {}
    for name, size in subset_sizes.items():
        if size is None:
            masks[name] = list(ordered)
        else:
            masks[name] = ordered[: min(int(size), len(ordered))]
    return masks


def write_subset_masks(prepared_dir: Path, masks: dict[str, list[str]]) -> dict[str, Any]:
    subset_dir = prepared_dir / "subset_masks"
    subset_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for name, keys in masks.items():
        path = subset_dir / f"{name}.json"
        payload = {"subset_id": name, "num_samples": len(keys), "sample_keys": keys}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        out[name] = {"path": str(path), "num_samples": len(keys), "sha256": file_digest(path)}
    return out


def dataset_profile(row: dict[str, Any], summary: dict[str, Any], subsets: dict[str, Any], sample_count: int) -> dict[str, Any]:
    avg_file_bytes = float(summary["total_bytes"]) / max(int(summary.get("file_count") or 0), 1)
    return {
        "dataset_profile_version": 1,
        "dataset_id": row["id"],
        "status": "prepared",
        "source_type": row.get("source_type"),
        "slug": row.get("slug"),
        "task_family": row.get("task_family"),
        "modality": row.get("modality"),
        "target_tier": row.get("target_tier", "local"),
        "estimated_storage_gb": row.get("estimated_storage_gb"),
        "model_families": row.get("model_families", []),
        "sample_count": sample_count,
        "sample_bytes_mean": avg_file_bytes,
        "raw_summary": summary,
        "subsets": subsets,
        "subset_policy": "deterministic_sha256_order; adapter-specific stratification may refine this file later",
        "metadata_source": "real_downloaded_files",
    }


def command_list(registry: dict[str, Any], tier: str, ids_only: bool) -> None:
    for row in registry.get("datasets", []):
        row_tier = str(row.get("target_tier", "local"))
        if tier != "all" and row_tier != tier:
            continue
        if ids_only:
            print(row["id"])
        else:
            size = row.get("estimated_storage_gb", "unknown")
            print(
                f"{row['id']}\t{row_tier}\t{row.get('status', 'pending')}\t"
                f"{row.get('task_family')}\t{size}\t{row.get('slug')}"
            )


def command_show(registry: dict[str, Any], dataset_id: str) -> None:
    print(json.dumps(get_dataset(registry, dataset_id), indent=2, sort_keys=True))


def command_approve(registry_path: Path, registry: dict[str, Any], dataset_id: str) -> None:
    row = get_dataset(registry, dataset_id)
    if row.get("status") == "pending":
        row["status"] = "approved"
    save_registry(registry_path, registry)
    print(f"{dataset_id}: {row['status']}")


def command_download(
    registry: dict[str, Any],
    dataset_id: str,
    raw_root: Path,
    dry_run: bool,
    allow_nautilus_only: bool,
) -> None:
    row = get_dataset(registry, dataset_id)
    require_approved(row)
    require_download_tier(row, allow_nautilus_only)
    out_dir = dataset_raw_dir(raw_root, row)
    if row.get("source_type") == "pyg_ogb":
        raise SystemExit("use download-ogb for PyG/OGB datasets")
    cmd = command_with_output_dir(row, out_dir)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def command_download_ogb(
    registry: dict[str, Any],
    dataset_id: str,
    raw_root: Path,
    dry_run: bool,
    allow_nautilus_only: bool,
) -> None:
    row = get_dataset(registry, dataset_id)
    require_approved(row)
    require_download_tier(row, allow_nautilus_only)
    if row.get("source_type") != "pyg_ogb":
        raise SystemExit(f"{dataset_id} is not a PyG/OGB dataset")
    out_dir = dataset_raw_dir(raw_root, row)
    loader_code = (
        "import builtins, os; "
        "os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1'); "
        "builtins.input = lambda prompt='': (print(prompt, end=''), 'y')[1]; "
        "from ogb.nodeproppred import PygNodePropPredDataset; "
        f"PygNodePropPredDataset(name={row['slug']!r}, root={str(out_dir)!r})"
    )
    cmd = [
        sys.executable,
        "-c",
        loader_code,
    ]
    print(f"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    missing = [name for name in ("ogb", "torch_geometric") if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"{dataset_id} requires missing Python package(s): {joined}. "
            "Install graph dataset dependencies with `python -m pip install ogb torch_geometric`."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    subprocess.run(cmd, check=True, env=env)


def command_prepare(
    registry_path: Path,
    registry: dict[str, Any],
    dataset_id: str,
    raw_root: Path,
    prepared_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    row = get_dataset(registry, dataset_id)
    require_approved(row)
    raw_dir = dataset_raw_dir(raw_root, row)
    if not raw_dir.exists():
        raise SystemExit(f"raw dataset directory does not exist: {raw_dir}")
    prepared_dir = dataset_prepared_dir(prepared_root, row)
    if prepared_dir.exists() and not force and not dry_run:
        raise SystemExit(f"{prepared_dir} exists; pass --force to overwrite")
    summary = summarize_raw_files(raw_dir)
    keys = sample_keys_for_dataset(row, raw_dir, summary)
    masks = subset_masks(keys, registry.get("subset_sizes", {}), str(row["id"]))
    print(json.dumps({"dataset_id": dataset_id, "samples": len(keys), "subsets": {k: len(v) for k, v in masks.items()}}, sort_keys=True))
    if dry_run:
        return
    prepared_dir.mkdir(parents=True, exist_ok=True)
    subset_info = write_subset_masks(prepared_dir, masks)
    profile = dataset_profile(row, summary, subset_info, len(keys))
    (prepared_dir / "metadata_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (prepared_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    row["status"] = "prepared"
    profile_path = prepared_dir / "dataset_profile.json"
    try:
        row["prepared_profile"] = str(profile_path.relative_to(ROOT))
    except ValueError:
        row["prepared_profile"] = str(profile_path)
    save_registry(registry_path, registry)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    if args.command == "list":
        command_list(registry, args.tier, args.ids_only)
    elif args.command == "show":
        command_show(registry, args.dataset_id)
    elif args.command == "approve":
        command_approve(registry_path, registry, args.dataset_id)
    elif args.command == "download":
        command_download(registry, args.dataset_id, Path(args.raw_root), args.dry_run, args.allow_nautilus_only)
    elif args.command == "download-ogb":
        command_download_ogb(registry, args.dataset_id, Path(args.raw_root), args.dry_run, args.allow_nautilus_only)
    elif args.command == "prepare":
        command_prepare(
            registry_path,
            registry,
            args.dataset_id,
            Path(args.raw_root),
            Path(args.prepared_root),
            args.dry_run,
            args.force,
        )


if __name__ == "__main__":
    main()
