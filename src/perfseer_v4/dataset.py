"""Project prepared v3.2 datasets without changing measurements or split identity."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

from perfseer_v31.dataset import _conservative_architecture_group
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32.version import DATASET_VERSION as SOURCE_VERSION, INPUT_SCHEMA_VERSION as SOURCE_INPUT

from .capture import graph_from_design
from .version import DATASET_VERSION, INPUT_SCHEMA_VERSION, TARGET_NAMES, validate_targets

DATA = Path(__file__).resolve().parent / "dataset_with_label"
SOURCE_VERSIONS = (SOURCE_VERSION, "perfseer_v32_rtx5090_transfer_training_v1")
VARIANTS = ("v4.0", "v4.1", "v4.2", "v4.3")


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("dataset path escapes its root")
    return path


def _manifest(root):
    manifest = read_json(Path(root) / "dataset_manifest.json")
    if manifest.get("fingerprint") != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("dataset manifest fingerprint differs")
    if set(manifest["split_files"]) != {"train", "validation", "test"}:
        raise ValueError("dataset requires train, validation and test splits")
    return manifest


def _checked_rows(root, meta):
    path = checked_path(root, meta["path"])
    if file_sha256(path) != meta["sha256"]:
        raise ValueError("split hash differs")
    rows = read_json(path)
    if not rows or len(rows) != meta["rows"]:
        raise ValueError("empty or changed split")
    return rows


def _snapshot(source, output, manifest):
    """Retain exact source rows and referenced policy evidence for portable audits."""
    files = {}

    def copy(relative, digest):
        if Path(relative).is_absolute():
            if not Path(relative).resolve().is_relative_to(source.resolve()):
                raise ValueError("source provenance path escapes its root")
            relative = str(Path(relative).resolve().relative_to(source.resolve()))
        if relative in files:
            if files[relative] != digest:
                raise ValueError("conflicting source provenance hashes")
            return
        path = checked_path(source, relative)
        if file_sha256(path) != digest:
            raise ValueError("source provenance hash differs")
        target = checked_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != digest:
            raise ValueError("source provenance copy differs")
        files[relative] = digest
        if relative.endswith((".json", ".json.gz")):
            references(read_json(path))

    def references(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                copy(value["path"], value["sha256"])
            else:
                for nested in value.values():
                    references(nested)
        elif isinstance(value, list):
            for nested in value:
                references(nested)

    copy("dataset_manifest.json", file_sha256(source / "dataset_manifest.json"))
    return files


def _project_row(row, split, input_path, input_sha256):
    names = row["target_names"]
    if len(names) != len(set(names)) or not set(TARGET_NAMES).issubset(names):
        raise ValueError("source training target names are missing or duplicated")
    values = row["targets"]
    values = values if isinstance(values, dict) else dict(zip(names, values, strict=True))
    targets = {name: values[name] for name in TARGET_NAMES}
    validate_targets(list(targets.values()))
    native = row["native_targets"]
    if not isinstance(native, dict):
        native = dict(zip(names, native, strict=True))
    validate_targets([native[name] for name in TARGET_NAMES])
    return {**row, "version": DATASET_VERSION, "split": split,
            "input_path": input_path, "input_sha256": input_sha256,
            "target_names": list(TARGET_NAMES), "targets": targets}


def prepare(source, output, workers=1, variant="v4.0"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("use a separate v4 dataset directory outside the historical source")
    if variant not in VARIANTS:
        raise ValueError("unsupported v4 variant")
    manifest = _manifest(source)
    if manifest.get("version") not in SOURCE_VERSIONS:
        raise ValueError("expected a prepared v3.2 dataset")
    if (output / "dataset_manifest.json").exists():
        existing = verify(output, workers)
        if existing["projection"]["source_dataset_fingerprint"] != manifest["fingerprint"] or existing["model_variant"] != variant:
            raise ValueError("existing projection belongs to a different source or variant")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("use an empty output directory for dataset projection")
    original_rows = {split: _checked_rows(source, meta) for split, meta in manifest["split_files"].items()}
    snapshots = _snapshot(source, output / "source", manifest)
    inputs, splits = {}, {}
    for split, rows in original_rows.items():
        projected = []
        for row in rows:
            key = row["input_path"]
            if key not in inputs:
                path = checked_path(source, key)
                if file_sha256(path) != row["input_sha256"]:
                    raise ValueError("source input hash differs")
                original = read_json(path)
                if original.get("version") != SOURCE_INPUT:
                    raise ValueError("expected a v3.2 paired input for explicit projection")
                design = {"version": INPUT_SCHEMA_VERSION, "training": original["training"]}
                graph_from_design(design)
                relative = f"models/{fingerprint(design)}.json.gz"
                atomic_write(output / relative, design, compress=True)
                inputs[key] = {"source_sha256": row["input_sha256"], "path": relative,
                               "sha256": file_sha256(output / relative),
                               "training_sha256": fingerprint(original["training"])}
            if inputs[key]["source_sha256"] != row["input_sha256"]:
                raise ValueError("conflicting source input hashes")
            projected.append(_project_row(row, split, inputs[key]["path"], inputs[key]["sha256"]))
        relative = f"{split}/samples.json.gz"
        atomic_write(output / relative, projected, compress=True)
        splits[split] = {"path": relative, "sha256": file_sha256(output / relative), "rows": len(projected)}
    projection = {"source_dataset_fingerprint": manifest["fingerprint"], "files": snapshots, "inputs": inputs}
    atomic_write(output / "projection.json", projection)
    result = {"version": DATASET_VERSION, "model_variant": variant, "input_schema": INPUT_SCHEMA_VERSION,
              "target_names": list(TARGET_NAMES), "split_files": splits,
              "total_rows": sum(meta["rows"] for meta in splits.values()),
              "prediction_hardware": manifest.get("prediction_hardware", manifest.get("hardware_id")),
              "label_policy": manifest.get("label_policy"),
              "projection": {"path": "projection.json", "sha256": file_sha256(output / "projection.json"),
                             "source_dataset_fingerprint": manifest["fingerprint"]}}
    result["fingerprint"] = fingerprint(result)
    atomic_write(output / "dataset_manifest.json", result)
    return verify(output, workers)


def _verify_input(item):
    path, evidence, hardware = item
    if file_sha256(path) != evidence["sha256"]:
        raise ValueError("training input hash differs")
    design = read_json(path)
    graph = graph_from_design(design)
    if fingerprint(design["training"]) != evidence["training_sha256"]:
        raise ValueError("training graph differs from source projection")
    if graph.metadata.get("target_hardware_id") != hardware:
        raise ValueError("training graph hardware differs")
    return str(path), _conservative_architecture_group(design["training"])


def verify(output, workers=1):
    output = Path(output).resolve()
    manifest = _manifest(output)
    if (manifest.get("version") != DATASET_VERSION or manifest.get("input_schema") != INPUT_SCHEMA_VERSION or
            tuple(manifest["target_names"]) != TARGET_NAMES or manifest.get("model_variant") not in VARIANTS):
        raise ValueError("v4 dataset contract differs")
    ref = manifest["projection"]
    path = checked_path(output, ref["path"])
    if file_sha256(path) != ref["sha256"]:
        raise ValueError("projection provenance hash differs")
    projection = read_json(path)
    for relative, digest in projection["files"].items():
        if file_sha256(checked_path(output / "source", relative)) != digest:
            raise ValueError("source provenance snapshot hash differs")
    source = _manifest(output / "source")
    if (source["version"] not in SOURCE_VERSIONS or source["fingerprint"] != ref["source_dataset_fingerprint"] or
            source["fingerprint"] != projection["source_dataset_fingerprint"] or
            source.get("label_policy") != manifest.get("label_policy") or
            source.get("prediction_hardware", source.get("hardware_id")) != manifest["prediction_hardware"]):
        raise ValueError("source projection contract differs")
    samples, groups, models, input_splits, hashes, pending, used = set(), {}, {}, {}, {}, {}, set()
    for split, meta in manifest["split_files"].items():
        rows = _checked_rows(output, meta)
        originals = _checked_rows(output / "source", source["split_files"][split])
        if len(rows) != len(originals):
            raise ValueError("source projection row count differs")
        for row, original in zip(rows, originals, strict=True):
            evidence = projection["inputs"][original["input_path"]]
            used.add(original["input_path"])
            if evidence["source_sha256"] != original["input_sha256"]:
                raise ValueError("source input provenance differs")
            if original.get("split", split) != split or row != _project_row(original, split, evidence["path"], evidence["sha256"]):
                raise ValueError("row labels, metadata or identity differ from source projection")
            if row["sample_id"] in samples or row["hardware_id"] != manifest["prediction_hardware"]:
                raise ValueError("duplicate sample or invalid hardware")
            samples.add(row["sample_id"])
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError("architecture group split leakage")
            model_id = row.get("provenance", {}).get("model_id")
            if model_id is not None and models.setdefault(model_id, split) != split:
                raise ValueError("source model split leakage")
            path = str(checked_path(output, row["input_path"]))
            if input_splits.setdefault(path, split) != split or hashes.setdefault(row["input_sha256"], split) != split:
                raise ValueError("training input split leakage")
            item = (path, evidence, row["hardware_id"])
            if path in pending and pending[path][1]["training_sha256"] != evidence["training_sha256"]:
                raise ValueError("conflicting training projection evidence")
            pending[path] = item
    if len(samples) != manifest["total_rows"] or used != set(projection["inputs"]):
        raise ValueError("incomplete source projection coverage")
    architectures = {}

    def check(results):
        for path, architecture in results:
            if architectures.setdefault(architecture, input_splits[path]) != input_splits[path]:
                raise ValueError("normalized architecture split leakage")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            check(pool.map(_verify_input, pending.values()))
    else:
        check(map(_verify_input, pending.values()))
    return manifest
