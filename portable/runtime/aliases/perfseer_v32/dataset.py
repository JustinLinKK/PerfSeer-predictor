"""Extract both archives and compile every unique measurement into strict graphs."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

import torch

from perfseer_v31 import dataset as compiler
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .capture import capture, capture_inference, graphs_from_design, inference_settings
from .version import DATASET_VERSION, HARDWARE_ID, TARGET_NAMES, INPUT_SCHEMA_VERSION, validate_targets

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "dataset_with_label"
ARCHIVES = ("a10_cv_other_labels_models (1).zip", "a10_cv_other_nlp_labels_models.zip")
SOURCE_ROWS = 74704
UNIQUE_ROWS = 40020


def extract_archive(archive, output):
    output = Path(output)
    files = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = PurePosixPath(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"unsafe archive member: {info.filename}")
            if info.is_dir():
                continue
            name = relative.as_posix()
            if name in files:
                raise ValueError(f"duplicate archive member: {name}")
            path = output / name
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError(f"unsafe extraction destination: {path}")
            payload = bundle.read(info)
            digest = hashlib.sha256(payload).hexdigest()
            if path.exists():
                if file_sha256(path) != digest:
                    raise ValueError(f"extracted source differs: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            if file_sha256(path) != digest:
                raise ValueError(f"extraction verification failed: {path}")
            files[name] = digest
    return files


def merge_sources(data=DATA):
    data = Path(data)
    rows, models, archives = {}, {}, {}
    total = 0
    for index, name in enumerate(ARCHIVES):
        archive = data / "raw_source" / name
        destination = data / "extracted" / f"archive-{index + 1}"
        files = extract_archive(archive, destination)
        archives[name] = {"sha256": file_sha256(archive), "files": files, "rows": 0}
        for relative, digest in sorted(files.items()):
            if relative.startswith("models/") and relative.endswith(".py"):
                if relative in models and models[relative] != digest:
                    raise ValueError(f"conflicting model sources: {relative}")
                models[relative] = digest
        for relative in sorted(files):
            if not relative.endswith((".jsonl", ".jsonl.gz")):
                continue
            path = destination / relative
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    native = json.loads(line)
                    key = native["profile_point_id"]
                    if native["status"] != "ok" or native["hardware_id"] != "a10":
                        raise ValueError(f"invalid source measurement: {key}")
                    values = [float(native["targets"][target]) for target in TARGET_NAMES]
                    validate_targets(values)
                    if tuple(native.get("target_names", ())) != TARGET_NAMES:
                        raise ValueError(f"native target order differs: {key}")
                    model_path = f"models/{native['model_id']}.py"
                    if model_path not in files:
                        raise ValueError(f"missing model source: {key}")
                    source = {"archive": name, "label_path": relative, "line": line_number}
                    if key in rows:
                        if rows[key]["native_label"] != native or rows[key]["model_source_sha256"] != files[model_path]:
                            raise ValueError(f"conflicting duplicate measurement: {key}")
                        rows[key]["sources"].append(source)
                    else:
                        rows[key] = {"sample_id": key, "native_label": native, "sources": [source],
                                     "model_source_path": str((destination / model_path).relative_to(data)),
                                     "model_source_sha256": files[model_path]}
                    total += 1
                    archives[name]["rows"] += 1
    if total != SOURCE_ROWS or len(rows) != UNIQUE_ROWS:
        raise ValueError(f"unexpected source coverage: {total} rows, {len(rows)} unique")
    merged = [rows[key] for key in sorted(rows)]
    path = data / "merged_samples.json.gz"
    if path.exists():
        if read_json(path) != merged:
            raise ValueError("existing merged source differs")
    else:
        atomic_write(path, merged, compress=True)
    manifest = {"archives": archives, "source_rows": total, "unique_rows": len(rows),
                "exact_duplicates": total - len(rows), "merged_sha256": file_sha256(path),
                "modalities": dict(Counter(r["native_label"]["dataset"]["modality"] for r in merged))}
    manifest["fingerprint"] = fingerprint(manifest)
    source_path = data / "source_manifest.json"
    if source_path.exists():
        if read_json(source_path) != manifest:
            raise ValueError("existing source manifest differs")
    else:
        atomic_write(source_path, manifest)
    return manifest


def source_rows(data=DATA):
    data = Path(data)
    manifest = read_json(data / "source_manifest.json")
    if file_sha256(data / "merged_samples.json.gz") != manifest["merged_sha256"]:
        raise ValueError("merged source hash differs")
    compiler_hash = fingerprint({name: file_sha256(compiler.PACKAGE / name) for name in
                                ("capture_training.py", "generated_model_runtime.py", "recovered_contracts.py", "recovered_generated_lineages.py")})
    defaults = read_json(compiler.PACKAGE / "optimizer_defaults.json")["defaults"]
    constants = {}
    for raw in read_json(data / "merged_samples.json.gz"):
        native = raw["native_label"]
        path = data / raw["model_source_path"]
        if path not in constants:
            if file_sha256(path) != raw["model_source_sha256"]:
                raise ValueError(f"model source hash differs: {path}")
            constants[path] = compiler._constants(path)
        model = constants[path]
        measured = native["training"]
        training = {
            "version": "perfseer_v31_training_config_v1",
            "microbatch_size": int(measured["batch_size"]),
            "gradient_accumulation_steps": int(measured["grad_accumulation_steps"]),
            "precision": measured["precision"],
            "optimizer": {"name": measured["optimizer"], "learning_rate": 1e-3,
                          "weight_decay": .01 if measured["optimizer"] == "adamw" else 0.,
                          **({"momentum": 0.} if measured["optimizer"] == "sgd" else {})},
            "scheduler": {"name": "none"}, "backend": "cuda_eager", "loss": "mse_to_zero",
            "steps_per_epoch": int(measured["steps_per_epoch"]),
            "total_epochs": int(measured["warmup_epochs"] + measured["measured_epochs"]),
        }
        if int(measured["effective_batch_size"]) != training["microbatch_size"] * training["gradient_accumulation_steps"]:
            raise ValueError(f"inconsistent measured batch: {raw['sample_id']}")
        training = compiler.training_settings(training, defaults)
        execution = {"nodes": compiler._native_architecture(model["NODE_SPECS"]), "inputs": model["INPUT_SPECS"]}
        yield {"sample_id": raw["sample_id"], "kind": "calibration", "model": model, "training": training,
               "targets": {name: float(native["targets"][name]) for name in TARGET_NAMES},
               "native_targets": native["targets"],
               "source_group": fingerprint({**execution, "inputs": [{**s, "shape": [1, *s["shape"][1:]]} for s in model["INPUT_SPECS"]]}),
               "capture_key": fingerprint({"model": execution, "training": training, "compiler": compiler_hash, "torch": torch.__version__}),
               "provenance": {"source_bundle": "a10_calibration", "model_id": native["model_id"],
                              "modality": native["dataset"]["modality"], "sources": raw["sources"],
                              "model_source_sha256": raw["model_source_sha256"],
                              "operation_types": sorted({n["type"] for n in model["NODE_SPECS"]}),
                              "recovered_training_defaults": "run_profile.py:Adam/SGD/AdamW(lr=1e-3), mse_to_zero, no scheduler"}}


def capture_fingerprint():
    shared = compiler.PACKAGE.parent / "perfseer_v3"
    files = [PACKAGE / "capture.py", PACKAGE / "version.py"]
    files += [compiler.PACKAGE / name for name in ("capture_training.py", "capture_export.py", "liveness.py", "generated_model_runtime.py")]
    files += [path for path in shared.iterdir() if path.suffix in {".py", ".json"}]
    return fingerprint({str(path.relative_to(PACKAGE.parent)): file_sha256(path) for path in files})


def _convert_pair(row, output):
    torch.set_num_threads(1)
    sys.dont_write_bytecode = True
    path = Path(output) / "models" / f"{row['capture_key']}.json.gz"
    audit_path = path.with_suffix(".audit.json")
    try:
        if path.exists() and audit_path.exists():
            design, audit = read_json(path), read_json(audit_path)
            if audit["design_sha256"] != file_sha256(path) or audit["capture_fingerprint"] != row["capture_fingerprint"]:
                raise ValueError("cached paired capture differs")
            graphs_from_design(design)
            return row["capture_key"], audit
        model, inputs, batch, adapter = compiler.build_workload(row)
        reuse = row.get("reuse")
        if reuse:
            if file_sha256(reuse["path"]) != reuse["audit"]["design_sha256"]:
                raise ValueError("reused training capture hash differs")
            training = read_json(reuse["path"])
            inference, evidence = capture_inference(model, inputs, row["training"], architecture_key=row["source_group"])
            design = {"version": INPUT_SCHEMA_VERSION, "training": training, "inference": inference}
            audit = {"training": reuse["audit"], "inference": evidence}
        else:
            design, audit = capture(model, inputs, batch, adapter, row["training"], architecture_key=row["source_group"])
        graphs_from_design(design)
        atomic_write(path, design, compress=True)
        audit.update(status="ok", design_sha256=file_sha256(path), capture_fingerprint=row["capture_fingerprint"],
                     training_capture_key=row["training_capture_key"],
                     mode_sha256={mode: fingerprint(design[mode]) for mode in ("training", "inference")},
                     group_architecture=compiler._conservative_architecture_group(design["training"]))
        atomic_write(audit_path, audit)
        return row["capture_key"], audit
    except Exception as error:
        return row["capture_key"], {"status": "failed", "sample_id": row["sample_id"], "error": f"{type(error).__name__}: {str(error)[:4000]}"}


def prepare(output, workers=6, data=DATA, reuse=None, limit=0):
    output, data = Path(output), Path(data)
    if output.resolve() == (data / "ready_for_train").resolve():
        raise ValueError("preserve the legacy dataset; use ready_for_train_12")
    if (output / "dataset_manifest.json").exists():
        return verify(output, workers)
    if file_sha256(compiler.PACKAGE / "generated_model_runtime.py") != compiler.CALIBRATION_RUNTIME_SHA256:
        raise ValueError("calibration runtime differs from pinned source")
    source = merge_sources(data)
    rows = list(source_rows(data))
    legacy_root = data / "ready_for_train"
    legacy = read_json(legacy_root / "dataset_manifest.json")
    if legacy["fingerprint"] != fingerprint({key: value for key, value in legacy.items() if key != "fingerprint"}):
        raise ValueError("legacy manifest fingerprint differs")
    reference = {}
    for split, meta in legacy["split_files"].items():
        path = legacy_root / meta["path"]
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("legacy split hash differs")
        for row in read_json(path):
            if row["sample_id"] in reference or row["split"] != split:
                raise ValueError("invalid legacy split membership")
            reference[row["sample_id"]] = row
    if set(reference) != {row["sample_id"] for row in rows}:
        raise ValueError("legacy/source measurement identities differ")
    reuse_root = Path(reuse) if reuse else legacy_root
    previous = read_json(reuse_root / "conversion-audit.json")["captures"]
    compiler_hash = capture_fingerprint()
    for row in rows:
        old = reference[row["sample_id"]]
        if old["native_targets"] != row["native_targets"] or old["provenance"] != row["provenance"]:
            raise ValueError("legacy native labels or provenance differ from source")
        row.update(group_id=old["group_id"], split=old["split"], training_capture_key=row["capture_key"], capture_fingerprint=compiler_hash)
        audit = previous.get(row["capture_key"], {})
        if audit.get("status") == "ok" and audit.get("backward_capture") == "strict" and audit.get("gradient_parameters_checked", 0) > 0:
            row["reuse"] = {"path": str(reuse_root / "models" / f"{row['capture_key']}.json.gz"), "audit": audit}
        row["capture_key"] = fingerprint({"training": row["training_capture_key"], "paired_compiler": compiler_hash})
    unique = {row["capture_key"]: row for row in rows}
    # A partial preflight covers distinct modality/precision settings first.
    representatives = {}
    for row in unique.values():
        representatives.setdefault((row["provenance"]["modality"], row["training"]["precision"]), row)
    selected = list(representatives.values())
    seen = {row["capture_key"] for row in selected}
    selected.extend(row for row in unique.values() if row["capture_key"] not in seen)
    selected = selected[:limit or None]
    audits = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_convert_pair, row, str(output)): row["capture_key"] for row in selected}
        for future in as_completed(pending):
            key, audit = future.result()
            audits[key] = audit
            print(json.dumps({"event": "paired_conversion", "completed": len(audits), "total": len(selected), "capture_key": key, **audit}), flush=True)
    atomic_write(output / "conversion-audit.json", {"source_rows": len(rows), "unique_captures": len(unique), "partial": bool(limit),
                 "capture_fingerprint": compiler_hash, "captures": audits})
    if any(audit["status"] != "ok" for audit in audits.values()):
        raise ValueError("paired graph conversion failed; see conversion-audit.json")
    if limit:
        return {"status": "partial_preflight", "captures": len(audits)}
    splits = {}
    for split in ("train", "validation", "test"):
        values = [{"version": DATASET_VERSION, "sample_id": row["sample_id"], "group_id": row["group_id"], "split": split,
                   "hardware_id": HARDWARE_ID, "input_path": f"models/{row['capture_key']}.json.gz",
                   "input_sha256": audits[row["capture_key"]]["design_sha256"], "target_names": list(TARGET_NAMES),
                   "targets": row["targets"], "native_targets": row["native_targets"], "provenance": row["provenance"]}
                  for row in rows if row["split"] == split]
        path = output / split / "samples.json.gz"
        atomic_write(path, values, compress=True)
        splits[split] = {"path": str(path.relative_to(output)), "rows": len(values), "sha256": file_sha256(path)}
    atomic_write(output / "source_manifest.json", source)
    atomic_write(output / "split-reference.json", {"dataset_fingerprint": legacy["fingerprint"], "split_files": legacy["split_files"],
                 "samples": {key: {"split": row["split"], "group_id": row["group_id"], "native_targets_sha256": fingerprint(row["native_targets"]),
                                    "provenance_sha256": fingerprint(row["provenance"])} for key, row in reference.items()}})
    manifest = {"version": DATASET_VERSION, "input_schema": INPUT_SCHEMA_VERSION, "hardware_id": HARDWARE_ID, "target_names": list(TARGET_NAMES),
                "total_rows": len(rows), "seed": 42, "split_files": splits, "capture_fingerprint": compiler_hash,
                "source_manifest_sha256": file_sha256(output / "source_manifest.json"),
                "split_reference_sha256": file_sha256(output / "split-reference.json"),
                "runtime_sha256": compiler.CALIBRATION_RUNTIME_SHA256, "capture_torch_version": str(torch.__version__),
                "all_captures_strict": True, "conversion_audit_sha256": file_sha256(output / "conversion-audit.json")}
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "dataset_manifest.json", manifest)
    return verify(output, workers)


def _verify_pair_input(item):
    path, expected_hash, expected_modes = item
    if file_sha256(path) != expected_hash:
        raise ValueError(f"paired input hash differs: {path}")
    design = read_json(path)
    graphs_from_design(design)
    if {mode: fingerprint(design[mode]) for mode in ("training", "inference")} != expected_modes:
        raise ValueError("mode graph hashes differ")
    return path, compiler._conservative_architecture_group(design["training"])


def verify(output, workers=6):
    output = Path(output)
    manifest = read_json(output / "dataset_manifest.json")
    if manifest["version"] != DATASET_VERSION or manifest.get("input_schema") != INPUT_SCHEMA_VERSION or tuple(manifest["target_names"]) != TARGET_NAMES or manifest["hardware_id"] != HARDWARE_ID:
        raise ValueError("dataset contract differs")
    if manifest["fingerprint"] != fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}):
        raise ValueError("manifest fingerprint differs")
    policy_rows = None
    if "label_policy" in manifest:
        from .label_policy import verify_policy
        policy_rows = verify_policy(output, manifest, workers)
    if manifest["runtime_sha256"] != compiler.CALIBRATION_RUNTIME_SHA256 or file_sha256(compiler.PACKAGE / "generated_model_runtime.py") != compiler.CALIBRATION_RUNTIME_SHA256:
        raise ValueError("calibration runtime identity differs")
    for name, filename in (("source_manifest", "source_manifest.json"), ("conversion_audit", "conversion-audit.json"), ("split_reference", "split-reference.json")):
        if file_sha256(output / filename) != manifest[name + "_sha256"]:
            raise ValueError(f"{name} hash differs")
    source = read_json(output / "source_manifest.json")
    audit = read_json(output / "conversion-audit.json")
    reference = read_json(output / "split-reference.json")
    if source["unique_rows"] != UNIQUE_ROWS or source["source_rows"] != SOURCE_ROWS or source["exact_duplicates"] != SOURCE_ROWS - UNIQUE_ROWS:
        raise ValueError("source accounting differs")
    if source["fingerprint"] != fingerprint({key: value for key, value in source.items() if key != "fingerprint"}):
        raise ValueError("source fingerprint differs")
    if audit["partial"] or audit["source_rows"] != UNIQUE_ROWS or len(audit["captures"]) != audit["unique_captures"] or not manifest["all_captures_strict"] or audit["capture_fingerprint"] != manifest["capture_fingerprint"]:
        raise ValueError("incomplete conversion")
    for item in audit["captures"].values():
        if item["status"] != "ok" or item["training"].get("backward_capture") != "strict" or item["training"].get("gradient_parameters_checked", 0) < 1 or item["inference"].get("inference_capture") != "strict" or item["inference"].get("forward_outputs_checked", 0) < 1 or item["capture_fingerprint"] != manifest["capture_fingerprint"]:
            raise ValueError("failed or unverified mode capture")
    if set(manifest["split_files"]) != {"train", "validation", "test"}:
        raise ValueError("incomplete splits")
    samples, groups, models, inputs, input_splits, modalities, hashes = set(), {}, {}, {}, {}, Counter(), {}
    for split, meta in manifest["split_files"].items():
        path = output / meta["path"]
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("split hash differs")
        rows = read_json(path)
        if len(rows) != meta["rows"] or len(rows) != reference["split_files"][split]["rows"] or not rows:
            raise ValueError("empty or changed split")
        for row in rows:
            if row["sample_id"] in samples or row["split"] != split or row["hardware_id"] != HARDWARE_ID or row["version"] != DATASET_VERSION:
                raise ValueError("duplicate or invalid row identity")
            samples.add(row["sample_id"])
            modalities[row["provenance"]["modality"]] += 1
            original = reference["samples"][row["sample_id"]]
            if original != {"split": split, "group_id": row["group_id"], "native_targets_sha256": fingerprint(row["native_targets"]), "provenance_sha256": fingerprint(row["provenance"])}:
                raise ValueError("native label, provenance, or split assignment changed")
            if groups.setdefault(row["group_id"], split) != split or models.setdefault(row["provenance"]["model_id"], split) != split:
                raise ValueError("architecture/model split leakage")
            if tuple(row["target_names"]) != TARGET_NAMES or set(row["targets"]) != set(TARGET_NAMES):
                raise ValueError("target contract differs")
            validate_targets([row["targets"][name] for name in TARGET_NAMES])
            if policy_rows is not None:
                if row != policy_rows[row["sample_id"]]:
                    raise ValueError("row differs from verified label policy")
            elif row["targets"] != {name: row["native_targets"][name] for name in TARGET_NAMES}:
                raise ValueError("native target mapping differs")
            path = output / row["input_path"]
            key = path.name.removesuffix(".json.gz")
            evidence = audit["captures"].get(key, {})
            if evidence.get("design_sha256") != row["input_sha256"]:
                raise ValueError("missing matching capture audit")
            if inputs.setdefault(path, row["input_sha256"]) != row["input_sha256"] or input_splits.setdefault(path, split) != split or hashes.setdefault(row["input_sha256"], split) != split:
                raise ValueError("input hash conflict or split leakage")
    architectures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        items = [(path, digest, audit["captures"][path.name.removesuffix(".json.gz")]["mode_sha256"]) for path, digest in inputs.items()]
        for path, architecture in pool.map(_verify_pair_input, items):
            if architectures.setdefault(architecture, input_splits[path]) != input_splits[path]:
                raise ValueError("normalized architecture leakage")
    if samples != set(reference["samples"]) or len(samples) != UNIQUE_ROWS or len(samples) != manifest["total_rows"] or dict(modalities) != source["modalities"]:
        raise ValueError("incomplete source coverage")
    return {"status": "verified", "total_rows": len(samples), "unique_inputs": len(inputs),
            "groups": len(groups), "modalities": dict(modalities), "split_files": manifest["split_files"],
            "dataset_fingerprint": manifest["fingerprint"], "native_labels_unchanged": True, "split_assignments_unchanged": True,
            "target_labels_unchanged": policy_rows is None, "label_policy": manifest.get("label_policy")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("merge", "prepare", "verify"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DATA / "ready_for_train_12")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = merge_sources(args.data) if args.command == "merge" else verify(args.output, args.workers) if args.command == "verify" else prepare(args.output, args.workers, args.data, args.reuse, args.limit)
    print(json.dumps({k: v for k, v in result.items() if k != "archives"}, indent=2))


if __name__ == "__main__":
    main()
