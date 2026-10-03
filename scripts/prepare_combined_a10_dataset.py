#!/usr/bin/env python3
"""Extract, combine, split, and verify the two checked-in A10 label archives."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


SCHEMA_VERSION = "perfseer_v3_combined_a10_labeled_sample_v1"
MANIFEST_VERSION = "perfseer_v3_combined_a10_dataset_manifest_v1"
SPLIT_SEED = "perfseer-v3-a10-combined-split-20260903-v1"
CALIBRATION_BUNDLE = "a10_cv_other_labels_models"
ACCEPTED_BUNDLE = "accepted_nodes_20260831T2240Z"
CALIBRATION_ARCHIVE = "a10_cv_other_labels_models (1).zip"
ACCEPTED_ARCHIVE = "accepted_nodes_20260831T2240Z.zip"
COMMON_TARGET_NAMES = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_peak_vram_mib",
)
ACCEPTED_TARGET_NAMES = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_p95_sm_util_percent",
    "train_peak_vram_used_mib",
    "train_peak_torch_reserved_mib",
    "train_peak_memory_controller_util_percent",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value) + b"\n")


def _atomic_write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                for row in rows:
                    compressed.write(_json_bytes(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _archive_members(
    archive: zipfile.ZipFile,
    *,
    strip_root: str | None,
) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member {info.filename!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"archive member is a symbolic link: {info.filename!r}")
        parts = path.parts
        if strip_root is not None:
            if not parts or parts[0] != strip_root:
                raise ValueError(f"archive member is outside {strip_root!r}: {info.filename!r}")
            parts = parts[1:]
        if not parts or normalized.endswith("/") or stat.S_ISDIR(mode):
            continue
        relative = PurePosixPath(*parts).as_posix()
        if relative in members:
            raise ValueError(f"duplicate normalized archive member {relative!r}")
        members[relative] = info
    return members


def _ensure_extracted(
    archive_path: Path,
    destination: Path,
    *,
    strip_root: str | None = None,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive, strip_root=strip_root)
        existing = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if existing and existing != set(members):
            missing = sorted(set(members) - existing)[:5]
            extra = sorted(existing - set(members))[:5]
            raise ValueError(
                f"existing extraction at {destination} differs; missing={missing}, extra={extra}"
            )
        for relative, info in sorted(members.items()):
            target = destination / relative
            if target.exists():
                with archive.open(info) as archived:
                    archived_sha256 = _sha256_stream(archived)
                if _sha256_file(target) != archived_sha256:
                    raise ValueError(f"extracted file differs from archive: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=target.parent
            )
            temporary = Path(temporary_name)
            try:
                with archive.open(info) as source, os.fdopen(descriptor, "wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return len(members)


def _finite_targets(targets: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    result = {name: float(targets[name]) for name in names}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("label targets must be finite")
    return result


def _calibration_rows(source_pool: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    group_attributes: dict[str, tuple[str, str]] = {}
    labels_root = source_pool / "calibration" / "labels"
    for label_path in sorted(labels_root.glob("*.jsonl.gz")):
        with gzip.open(label_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                native = json.loads(line)
                if native.get("status") != "ok":
                    raise ValueError(f"calibration label is not usable: {label_path}:{line_number}")
                profile_point_id = str(native["profile_point_id"])
                sample_id = f"calibration:{profile_point_id}"
                if sample_id in seen_samples:
                    raise ValueError(f"duplicate sample ID {sample_id}")
                seen_samples.add(sample_id)
                model_id = str(native["model_id"])
                design_path = source_pool / "calibration" / "models" / f"{model_id}.py"
                if not design_path.is_file():
                    raise ValueError(f"missing calibration model design {design_path}")
                modality = str(native["dataset"]["modality"])
                hardware_id = str(native["hardware_id"])
                prior = group_attributes.setdefault(model_id, (modality, hardware_id))
                if prior != (modality, hardware_id):
                    raise ValueError(f"calibration model {model_id} crosses strata")
                target_names = tuple(str(value) for value in native["target_names"])
                targets = _finite_targets(native["targets"], target_names)
                common_targets = _finite_targets(targets, COMMON_TARGET_NAMES)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "source_bundle": CALIBRATION_BUNDLE,
                        "split": None,
                        "split_group_id": f"calibration-model:{model_id}",
                        "hardware_id": hardware_id,
                        "modality": modality,
                        "model_id": model_id,
                        "model_design_path": design_path.relative_to(source_pool.parent).as_posix(),
                        "model_design_sha256": _sha256_file(design_path),
                        "label_artifact_path": label_path.relative_to(source_pool.parent).as_posix(),
                        "label_artifact_line": line_number,
                        "native_target_schema": "calibration_12_target_v3",
                        "target_names": list(target_names),
                        "targets": targets,
                        "common_targets": common_targets,
                        "native_label_sha256": _canonical_sha256(native),
                        "native_label": native,
                    }
                )
    group_sizes = Counter(row["split_group_id"] for row in rows)
    if set(group_sizes.values()) != {4}:
        raise ValueError("each calibration model must have exactly four measurements")
    return rows


def _accepted_rows(source_pool: Path) -> list[dict[str, Any]]:
    root = source_pool / "accepted_nodes"
    csv_path = root / "required_labels" / "required_labels.csv"
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for csv_row_number, required in enumerate(csv.DictReader(handle), start=2):
            candidate_id = str(required["candidate_id"])
            sample_id = f"accepted-node:{candidate_id}"
            if sample_id in seen_samples:
                raise ValueError(f"duplicate sample ID {sample_id}")
            seen_samples.add(sample_id)
            if required["label_status"] != "accepted":
                raise ValueError(f"accepted-node label is not accepted: {candidate_id}")
            record_path = root / "accepted" / f"{candidate_id}.json"
            design_path = root / "model_design_sources" / f"{candidate_id}.py"
            if not record_path.is_file() or not design_path.is_file():
                raise ValueError(f"missing accepted-node artifact for {candidate_id}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "accepted":
                raise ValueError(f"accepted-node record is not accepted: {candidate_id}")
            if record.get("configuration_id") != candidate_id:
                raise ValueError(f"accepted-node configuration differs: {candidate_id}")
            hardware_id = str(required["target_hardware_id"])
            if record.get("target_hardware_id") != hardware_id:
                raise ValueError(f"accepted-node hardware differs: {candidate_id}")
            if record.get("fingerprints", {}).get("source_sha256") != required["model_source_sha256"]:
                raise ValueError(f"accepted-node source fingerprint differs: {candidate_id}")
            validity = record.get("targets", {}).get("validity")
            values = tuple(float(value) for value in record.get("targets", {}).get("values", ()))
            if validity != [True] * len(ACCEPTED_TARGET_NAMES) or len(values) != len(
                ACCEPTED_TARGET_NAMES
            ):
                raise ValueError(f"accepted-node target vector is invalid: {candidate_id}")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"accepted-node target is not finite: {candidate_id}")
            if json.loads(required["target_values_json"]) != list(values):
                raise ValueError(f"accepted-node CSV and record targets differ: {candidate_id}")
            targets = dict(zip(ACCEPTED_TARGET_NAMES, values, strict=True))
            common_targets = {
                "train_epoch_ms": targets["train_epoch_ms"],
                "train_avg_sm_util_percent": targets["train_avg_sm_util_percent"],
                "train_peak_vram_mib": targets["train_peak_vram_used_mib"],
            }
            family_group = _canonical_sha256(
                {
                    "source_lineage": required["model_source_lineage"],
                    "source_sha256": required["model_source_sha256"],
                }
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_bundle": ACCEPTED_BUNDLE,
                    "split": None,
                    "split_group_id": f"accepted-design:{candidate_id}",
                    "source_family_group_id": family_group,
                    "hardware_id": hardware_id,
                    "modality": str(required["model_source_modality"]),
                    "model_id": candidate_id,
                    "model_family_id": str(required["model_family_id"]),
                    "model_design_path": design_path.relative_to(source_pool.parent).as_posix(),
                    "model_design_sha256": _sha256_file(design_path),
                    "label_artifact_path": record_path.relative_to(source_pool.parent).as_posix(),
                    "label_artifact_sha256": _sha256_file(record_path),
                    "required_label_artifact_path": csv_path.relative_to(source_pool.parent).as_posix(),
                    "required_label_csv_row": csv_row_number,
                    "runtime_source_root": (
                        root / "model_sources"
                    ).relative_to(source_pool.parent).as_posix(),
                    "native_target_schema": "dataset_pack_6_target_v1",
                    "target_names": list(ACCEPTED_TARGET_NAMES),
                    "targets": targets,
                    "common_targets": common_targets,
                    "required_label_sha256": _canonical_sha256(required),
                    "required_label": required,
                    "native_label_sha256": _canonical_sha256(record),
                    "native_label": record,
                }
            )
    return rows


def _largest_remainder(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    if total < 0 or total > denominator:
        raise ValueError("invalid apportionment total")
    exact = {key: Fraction(total * value, denominator) for key, value in weights.items()}
    result = {key: value.numerator // value.denominator for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def _assign_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bundle_rows = Counter(str(row["source_bundle"]) for row in rows)
    validation_total = (len(rows) + 5) // 10
    bundle_validation = _largest_remainder(validation_total, bundle_rows)
    selected_validation_groups: set[str] = set()
    stratum_validation: dict[str, int] = {}
    for bundle in sorted(bundle_rows):
        bundle_subset = [row for row in rows if row["source_bundle"] == bundle]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bundle_subset:
            groups[str(row["split_group_id"])].append(row)
        group_sizes = {len(group) for group in groups.values()}
        if len(group_sizes) != 1:
            raise ValueError(f"split groups in {bundle} do not have a uniform size")
        group_size = next(iter(group_sizes))
        if bundle_validation[bundle] % group_size:
            raise ValueError(f"validation allocation for {bundle} breaks a model group")
        modality_groups: dict[str, list[str]] = defaultdict(list)
        for group_id, group in groups.items():
            modalities = {str(row["modality"]) for row in group}
            if len(modalities) != 1:
                raise ValueError(f"split group {group_id} crosses modalities")
            modality_groups[next(iter(modalities))].append(group_id)
        validation_group_count = bundle_validation[bundle] // group_size
        modality_validation = _largest_remainder(
            validation_group_count,
            {modality: len(values) for modality, values in modality_groups.items()},
        )
        for modality, group_ids in sorted(modality_groups.items()):
            ranked = sorted(
                group_ids,
                key=lambda group_id: hashlib.sha256(
                    f"{SPLIT_SEED}\0{bundle}\0{modality}\0{group_id}".encode("utf-8")
                ).hexdigest(),
            )
            chosen = ranked[: modality_validation[modality]]
            selected_validation_groups.update(chosen)
            stratum_validation[f"{bundle}:{modality}"] = len(chosen) * group_size
    for row in rows:
        row["split"] = (
            "validation"
            if row["split_group_id"] in selected_validation_groups
            else "train"
        )
    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != Counter({"train": len(rows) - validation_total, "validation": validation_total}):
        raise ValueError(f"split counts differ: {dict(split_counts)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["split_group_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("one model-design split group crosses train and validation")
    return {
        "requested_fractions": {"train": 0.9, "validation": 0.1},
        "actual_counts": dict(sorted(split_counts.items())),
        "actual_fractions": {
            split: split_counts[split] / len(rows) for split in ("train", "validation")
        },
        "seed": SPLIT_SEED,
        "split_unit": "model_design",
        "stratified_by": ["source_bundle", "modality"],
        "allocation": "largest_remainder_then_sha256_rank",
        "split_group_count": len(group_splits),
        "split_group_overlap_count": 0,
        "validation_counts_by_stratum": dict(sorted(stratum_validation.items())),
    }


def _source_tree_fingerprint(source_pool: Path) -> tuple[int, str]:
    files = sorted(path for path in source_pool.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_pool).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _prepare(data_root: Path) -> Path:
    raw_root = data_root / "raw_source"
    output = data_root / "ready_for_train"
    source_pool = output / "source_pool"
    archives = (
        (raw_root / CALIBRATION_ARCHIVE, source_pool / "calibration", None),
        (raw_root / ACCEPTED_ARCHIVE, source_pool / "accepted_nodes", ACCEPTED_BUNDLE),
    )
    archive_metadata = []
    for archive_path, destination, strip_root in archives:
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"archive CRC check failed: {archive_path}")
            file_count = len(_archive_members(archive, strip_root=strip_root))
        extracted_count = _ensure_extracted(
            archive_path, destination, strip_root=strip_root
        )
        if extracted_count != file_count:
            raise ValueError(f"archive extraction count differs: {archive_path}")
        archive_metadata.append(
            {
                "path": Path(os.path.relpath(archive_path, output)).as_posix(),
                "sha256": _sha256_file(archive_path),
                "extracted_file_count": extracted_count,
            }
        )
    rows = _calibration_rows(source_pool) + _accepted_rows(source_pool)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("combined sample IDs are not unique")
    split_policy = _assign_splits(rows)
    split_rows = {
        split: sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: str(row["sample_id"]),
        )
        for split in ("train", "validation")
    }
    split_files: dict[str, Any] = {}
    for split, values in split_rows.items():
        path = output / split / "samples.jsonl.gz"
        _atomic_write_jsonl_gz(path, values)
        split_files[split] = {
            "path": path.relative_to(output).as_posix(),
            "rows": len(values),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "source_bundles": _counts(values, "source_bundle"),
            "hardware_ids": _counts(values, "hardware_id"),
            "modalities": _counts(values, "modality"),
            "native_target_schemas": _counts(values, "native_target_schema"),
        }
    source_file_count, source_tree_sha256 = _source_tree_fingerprint(source_pool)
    split_fingerprint = _canonical_sha256(
        sorted(
            (
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "split_group_id": row["split_group_id"],
                }
                for row in rows
            ),
            key=lambda value: value["sample_id"],
        )
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "sample_schema_version": SCHEMA_VERSION,
        "total_rows": len(rows),
        "source_archives": archive_metadata,
        "source_pool": {
            "path": "source_pool",
            "file_count": source_file_count,
            "tree_sha256": source_tree_sha256,
        },
        "split_policy": split_policy,
        "split_fingerprint": split_fingerprint,
        "split_files": split_files,
        "common_target_names": list(COMMON_TARGET_NAMES),
        "native_target_contracts": {
            "calibration_12_target_v3": list(
                next(
                    row["target_names"]
                    for row in rows
                    if row["source_bundle"] == CALIBRATION_BUNDLE
                )
            ),
            "dataset_pack_6_target_v1": list(ACCEPTED_TARGET_NAMES),
        },
        "hardware_ids_are_preserved_not_merged": True,
        "intended_validation_interval_epochs": 1,
    }
    _atomic_write_json(output / "dataset_manifest.json", manifest)
    return output


def _read_split(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _verify(data_root: Path) -> dict[str, Any]:
    output = data_root / "ready_for_train"
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("combined dataset manifest version differs")
    for archive in manifest["source_archives"]:
        path = (output / archive["path"]).resolve()
        if _sha256_file(path) != archive["sha256"]:
            raise ValueError(f"source archive hash differs: {path}")
        with zipfile.ZipFile(path) as zipped:
            if zipped.testzip() is not None:
                raise ValueError(f"source archive CRC check failed: {path}")
    source_count, source_sha256 = _source_tree_fingerprint(output / "source_pool")
    if source_count != manifest["source_pool"]["file_count"]:
        raise ValueError("source-pool file count differs")
    if source_sha256 != manifest["source_pool"]["tree_sha256"]:
        raise ValueError("source-pool fingerprint differs")
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        metadata = manifest["split_files"][split]
        path = output / metadata["path"]
        if _sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"{split} file hash differs")
        values = _read_split(path)
        if len(values) != metadata["rows"]:
            raise ValueError(f"{split} row count differs")
        if any(row.get("split") != split for row in values):
            raise ValueError(f"{split} file contains another split")
        if _counts(values, "source_bundle") != metadata["source_bundles"]:
            raise ValueError(f"{split} source distribution differs")
        if _counts(values, "modality") != metadata["modalities"]:
            raise ValueError(f"{split} modality distribution differs")
        rows.extend(values)
    if len(rows) != manifest["total_rows"]:
        raise ValueError("combined row count differs")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs overlap across splits")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("combined sample schema version differs")
        if set(row.get("common_targets", ())) != set(COMMON_TARGET_NAMES):
            raise ValueError(f"common targets differ: {row['sample_id']}")
        if any(not math.isfinite(float(value)) for value in row["common_targets"].values()):
            raise ValueError(f"common target is not finite: {row['sample_id']}")
        design_path = (output / row["model_design_path"]).resolve()
        try:
            design_path.relative_to(output.resolve())
        except ValueError as error:
            raise ValueError(f"model path escapes output: {row['sample_id']}") from error
        if not design_path.is_file() or _sha256_file(design_path) != row["model_design_sha256"]:
            raise ValueError(f"model design differs: {row['sample_id']}")
        if _canonical_sha256(row["native_label"]) != row["native_label_sha256"]:
            raise ValueError(f"native label hash differs: {row['sample_id']}")
        group_splits[str(row["split_group_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("a model-design group leaks across splits")
    if len(group_splits) != manifest["split_policy"]["split_group_count"]:
        raise ValueError("split-group count differs")
    split_fingerprint = _canonical_sha256(
        sorted(
            (
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "split_group_id": row["split_group_id"],
                }
                for row in rows
            ),
            key=lambda value: value["sample_id"],
        )
    )
    if split_fingerprint != manifest["split_fingerprint"]:
        raise ValueError("split fingerprint differs")
    return {
        "status": "ok",
        "total_rows": len(rows),
        "train_rows": manifest["split_files"]["train"]["rows"],
        "validation_rows": manifest["split_files"]["validation"]["rows"],
        "split_group_overlap_count": 0,
        "source_file_count": source_count,
        "source_tree_sha256": source_sha256,
        "split_fingerprint": split_fingerprint,
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    default_data_root = repository_root / "src" / "perfseer_v3" / "dataset_with_label"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        output = _prepare(arguments.data_root.resolve())
        report = _verify(arguments.data_root.resolve())
        report["output"] = str(output)
    else:
        report = _verify(arguments.data_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
