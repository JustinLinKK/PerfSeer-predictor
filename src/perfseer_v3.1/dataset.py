"""Verified migration and complete conversion of the two A10 source collections."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import importlib
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import torch

from .capture_training import capture, graph_from_design
from .generated_model_runtime import GraphModel
from .io import atomic_write, file_sha256, fingerprint, read_json
from .version import DATASET_VERSION, HARDWARE_ID, TARGET_NAMES

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
DATA = PACKAGE / "dataset_with_label"
LEGACY = DATA / "legacy_v3"
CALIBRATION_RUNTIME_SHA256 = "c05fb507cabb8efd0c050a37f01bb58460d19e785872d5b3bf604187117a27fb"


def training_settings(training, defaults):
    recorded = training["optimizer"]
    components = defaults[recorded["name"]]
    primary = dict(components[0])
    primary.pop("lr", None)
    if isinstance(primary.get("eps"), list):
        primary["epsilon_components"] = primary.pop("eps")
    training["optimizer"] = {**primary, **recorded}
    if len(components) > 1:
        training["optimizer"]["components"] = [recorded["name"], "adamw"]
    scheduler = dict(training["scheduler"])
    known = {"exponential": {"gamma": .95}, "step": {"gamma": .9, "step_size": 1},
             "multi_step": {"gamma": .9, "milestones": [1, 3]}, "polynomial": {"power": 2.},
             "reduce_on_plateau": {"factor": .5, "patience": 0}, "cosine_warm_restarts": {"T_0": 2}}
    training["scheduler"] = {**known.get(scheduler["name"], {}), **scheduler}
    training["unavailable_settings"] = ["per_epoch_learning_rate_trace"] if scheduler["name"] not in {"none", "constant"} else []
    return training


def migrate():
    old = ROOT / "src/perfseer_v3/dataset_with_label"
    verify = runpy.run_path(str(ROOT / "scripts/prepare_combined_a10_dataset.py"))["_verify"]
    if old.is_symlink():
        if old.resolve() != LEGACY.resolve():
            raise ValueError("legacy dataset symlink points to an unexpected destination")
        return verify(LEGACY)
    if LEGACY.exists():
        raise ValueError("migration destination exists without the expected legacy link")
    before = verify(old)
    DATA.mkdir(parents=True, exist_ok=True)
    old.rename(LEGACY)
    try:
        after = verify(LEGACY)
        if after != before:
            raise ValueError("dataset changed during migration")
        old.symlink_to(Path("../perfseer_v3.1/dataset_with_label/legacy_v3"), target_is_directory=True)
    except Exception:
        LEGACY.rename(old)
        raise
    atomic_write(DATA / "migration-verification.json", after)
    return after


def _constants(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"INPUT_SPECS", "NODE_SPECS"}:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"INPUT_SPECS", "NODE_SPECS"}:
        raise ValueError(f"missing generated model specifications: {path}")
    return values


def _native_architecture(specs):
    # Only runtime-consumed memory fields describe layer construction. Accounting
    # totals in the generated source refer to its original (unprofiled) batch size.
    fields = ("input_channels", "output_channels", "input_features", "output_features", "input_h", "input_w", "output_h", "output_w", "rank", "sequence_length")
    return [
        {"id": node["id"], "type": node["type"], "args": node["args"], "preds": node["preds"],
         "input_index": node.get("input_index", 0),
         "memory_info": {key: node.get("memory_info", {}).get(key, 0) for key in fields}}
        for node in specs
    ]


def source_rows():
    source_root = LEGACY / "ready_for_train"
    compiler = fingerprint({name: file_sha256(PACKAGE / name) for name in ("capture_training.py", "generated_model_runtime.py", "recovered_contracts.py", "recovered_generated_lineages.py")})
    constants = {}
    defaults = read_json(PACKAGE / "optimizer_defaults.json")["defaults"]
    for split in ("train", "validation"):
        with gzip.open(source_root / split / "samples.jsonl.gz", "rt") as stream:
            for line in stream:
                raw = json.loads(line)
                targets = {name: float(raw["common_targets"][name]) for name in TARGET_NAMES}
                if not all(math.isfinite(value) for value in targets.values()):
                    raise ValueError(f"nonfinite target: {raw['sample_id']}")
                if targets[TARGET_NAMES[0]] <= 0 or targets[TARGET_NAMES[2]] <= 0 or not 0 <= targets[TARGET_NAMES[1]] <= 100:
                    raise ValueError(f"target outside physical range: {raw['sample_id']}")
                row = {"sample_id": raw["sample_id"], "targets": targets, "hardware_id": HARDWARE_ID,
                       "provenance": {"source_bundle": raw["source_bundle"], "model_id": raw["model_id"],
                                      "modality": raw["modality"], "legacy_hardware_id": raw["hardware_id"],
                                      "model_source_sha256": raw["model_design_sha256"]}}
                if raw["native_target_schema"] == "calibration_12_target_v3":
                    if targets != {name: float(raw["native_label"]["targets"][name]) for name in TARGET_NAMES}:
                        raise ValueError(f"calibration target mapping differs: {raw['sample_id']}")
                    path = source_root / raw["model_design_path"]
                    if path not in constants:
                        constants[path] = _constants(path)
                    model = constants[path]
                    native = raw["native_label"]
                    measured = native["training"]
                    row["kind"] = "calibration"
                    row["model"] = model
                    row["training"] = {
                        "microbatch_size": int(measured["batch_size"]),
                        "gradient_accumulation_steps": int(measured["grad_accumulation_steps"]),
                        "precision": measured["precision"],
                        "optimizer": {"name": measured["optimizer"], "learning_rate": 1e-3,
                                      "weight_decay": .01 if measured["optimizer"] == "adamw" else 0.,
                                      **({"momentum": 0.} if measured["optimizer"] == "sgd" else {})},
                        "scheduler": {"name": "none"}, "backend": "cuda_eager",
                        "loss": "mse_to_zero", "steps_per_epoch": int(measured["steps_per_epoch"]),
                        "total_epochs": int(measured["warmup_epochs"] + measured["measured_epochs"]),
                    }
                    row["provenance"]["recovered_training_defaults"] = "run_profile.py:Adam/SGD/AdamW(lr=1e-3), mse_to_zero, no scheduler"
                    row["source_group"] = fingerprint({"nodes": _native_architecture(model["NODE_SPECS"]), "inputs": [{**s, "shape": [1, *s["shape"][1:]]} for s in model["INPUT_SPECS"]]})
                    row["provenance"]["operation_types"] = sorted({node["type"] for node in model["NODE_SPECS"]})
                    execution = {"nodes": _native_architecture(model["NODE_SPECS"]), "inputs": model["INPUT_SPECS"]}
                elif raw["native_target_schema"] == "dataset_pack_6_target_v1":
                    native_targets = raw["native_label"]["targets"]
                    if not all(native_targets["validity"][i] for i in (0, 1, 3)) or list(targets.values()) != [float(native_targets["values"][i]) for i in (0, 1, 3)]:
                        raise ValueError(f"small target mapping differs: {raw['sample_id']}")
                    model = json.loads(raw["required_label"]["model_design_json"])
                    row["kind"] = "accepted"
                    row["model"] = model
                    row["provenance"]["family"] = model["family_id"]
                    native = raw["native_label"]
                    microbatch = int(model["microbatch_size"])
                    row["training"] = {
                        "microbatch_size": microbatch, "gradient_accumulation_steps": int(model["gradient_accumulation_steps"]),
                        "precision": model["precision_policy"]["policy_id"],
                        "optimizer": model["optimizer"], "scheduler": model["scheduler"],
                        "backend": model["execution"]["backend_id"], "loss": model["training_step_id"],
                        "steps_per_epoch": math.ceil(int(native["expected_train_examples"]) / microbatch),
                        "total_epochs": int(native["total_epochs"]),
                    }
                    execution = {key: model[key] for key in ("factory_id", "architecture_parameters", "input_signature", "target_width", "task_id")}
                    row["source_group"] = fingerprint({key: value for key, value in execution.items() if key != "task_id"})
                else:
                    raise ValueError(f"unsupported source schema: {raw['sample_id']}")
                row["training"]["version"] = "perfseer_v31_training_config_v1"
                row["training"] = training_settings(row["training"], defaults)
                row["capture_key"] = fingerprint({"model": execution, "training": row["training"], "compiler": compiler, "torch": torch.__version__})
                yield row


def _archived_modules():
    name = "perfseer_v31.archived"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(LEGACY / "ready_for_train/source_pool/accepted_nodes/model_sources/perfseer_v3")]
        sys.modules[name] = package
        importlib.import_module(name + ".dataset_pack")
        for dependency in ("contracts", "generated_lineages"):
            module_name = name + ".dataset_pack." + dependency
            spec = importlib.util.spec_from_file_location(module_name, PACKAGE / f"recovered_{dependency}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    return name


def build_workload(row):
    torch.manual_seed(42)
    if row["kind"] == "calibration":
        model = GraphModel(row["model"]["NODE_SPECS"])
        inputs = []
        for spec in row["model"]["INPUT_SPECS"]:
            shape = (row["training"]["microbatch_size"], *spec["shape"][1:])
            dtype = getattr(torch, spec["dtype"])
            if dtype.is_floating_point:
                tensor = torch.randn(shape, dtype=dtype)
            else:
                tensor = torch.ones(shape, dtype=dtype)
            inputs.append(tensor)
        return model, tuple(inputs), None, None
    prefix = _archived_modules()
    design = row["model"]
    adapters = importlib.import_module(prefix + ".dataset_pack.adapters.base")
    from perfseer_v3.dataset_pack.local_runtime import bind_candidate_batch_shape
    adapter = adapters.adapter_for_task(design["task_id"])
    candidate = SimpleNamespace(microbatch_size=int(design["microbatch_size"]), input_signature=design["input_signature"], family_id=design["family_id"])
    batch = bind_candidate_batch_shape(candidate, adapter, adapter.build_train_dataset()[0])
    module = importlib.import_module(design["factory_id"].replace("perfseer_v3", prefix, 1))
    model = module.build_model(output_width=int(design["target_width"]), task_kind=adapter.task_kind,
                               seed=int(design["seed_policy"]["seed"]), architecture_parameters=design["architecture_parameters"])
    return model, adapter.build_model_inputs(batch), batch, adapter


def _architecture_from_graph(design):
    # Ignore source names, initial weights, and training phases for architecture
    # grouping. Shared tensor topology is represented by numbered graph edges.
    nodes = [node for node in design["NODE_SPECS"] if node["phase"] == "forward"]
    ids = {node["node_id"] for node in nodes}
    return fingerprint({
        "nodes": [{key: node[key] for key in ("node_id", "raw_target", "normalized_args")} for node in nodes],
        "edges": [{key: edge[key] for key in ("producer_node_id", "consumer_node_id", "producer_output_index", "consumer_input_index", "tensor_role", "shape", "dtype")} for edge in design["EDGE_SPECS"] if edge["consumer_node_id"] in ids],
    })


def _conservative_architecture_group(design):
    # Group more broadly than exact equality: batch shapes, autocast conversions,
    # and source naming cannot separate otherwise identical architectures. This
    # may co-locate related designs, but never creates cross-split duplicates.
    nodes = [node for node in design["NODE_SPECS"] if node["phase"] == "forward"]
    identities = [node["raw_target"] for node in nodes if "_to_copy" not in node["raw_target"] and "detach" not in node["raw_target"]]
    parameters = {}
    for edge in design["EDGE_SPECS"]:
        if edge["tensor_role"] == "parameter":
            parameters.setdefault(edge["source_name"], edge["shape"])
    return fingerprint({"operations": identities, "parameter_shapes": sorted(parameters.values())})


def _convert_one(row, output):
    torch.set_num_threads(1)
    sys.dont_write_bytecode = True
    path = Path(output) / "models" / f"{row['capture_key']}.json.gz"
    audit_path = path.with_suffix(".audit.json")
    try:
        if path.exists() and audit_path.exists():
            design, audit = read_json(path), read_json(audit_path)
            if audit.get("design_sha256") != file_sha256(path):
                raise ValueError("cached design hash mismatch")
            graph_from_design(design).validate()
        else:
            model, inputs, batch, adapter = build_workload(row)
            design, audit = capture(model, inputs, batch, adapter, row["training"], architecture_key=row["source_group"])
            atomic_write(path, design, compress=True)
            audit.update({"design_sha256": file_sha256(path), "architecture": _architecture_from_graph(design)})
            atomic_write(audit_path, audit)
        if "group_architecture" not in audit:
            audit["group_architecture"] = _conservative_architecture_group(design)
            atomic_write(audit_path, audit)
        return row["capture_key"], {"status": "ok", **audit}
    except Exception as error:
        return row["capture_key"], {"status": "failed", "sample_id": row["sample_id"], "error": f"{type(error).__name__}: {str(error)[:4000]}"}


def grouped_split(rows, audits, seed=42):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in rows:
        left, right = find(row["source_group"]), find(audits[row["capture_key"]]["group_architecture"])
        parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for row in rows:
        row["group_id"] = find(row["source_group"])
        groups[row["group_id"]].append(row)
    # Whole groups are assigned greedily against source/modality-stratified quotas.
    strata = Counter((r["provenance"]["source_bundle"], r["provenance"]["modality"]) for r in rows)
    counts = {split: Counter() for split in ("train", "validation", "test")}
    fractions = {"train": .8, "validation": .1, "test": .1}
    for key in sorted(groups, key=lambda key: (-len(groups[key]), fingerprint([seed, key]))):
        group = groups[key]
        changes = Counter((r["provenance"]["source_bundle"], r["provenance"]["modality"]) for r in group)
        def cost(split):
            return sum(((counts[split][s] + n - fractions[split] * strata[s]) ** 2 - (counts[split][s] - fractions[split] * strata[s]) ** 2) / max(fractions[split] * strata[s], 1) for s, n in changes.items())
        split = min(fractions, key=cost)
        counts[split].update(changes)
        for row in group:
            row["split"] = split
    return rows


def prepare(output, workers=4, limit=0):
    output = Path(output)
    manifest_path = output / "dataset_manifest.json"
    if not limit and manifest_path.exists():
        previous = read_json(manifest_path)
        manifest_path.replace(output / f"dataset_manifest.{previous['fingerprint']}.previous.json")
    if file_sha256(PACKAGE / "generated_model_runtime.py") != CALIBRATION_RUNTIME_SHA256:
        raise ValueError("recovered runtime differs from its pinned source")
    rows = list(source_rows())
    if len(rows) != 36409 or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("source row count or uniqueness differs")
    unique = {r["capture_key"]: r for r in rows}
    representatives = {}
    for row in unique.values():
        family = row["provenance"].get("family", tuple(row["provenance"].get("operation_types", ())))
        representatives.setdefault((row["kind"], family), row)
    ordered = list(representatives.values())
    seen = {row["capture_key"] for row in ordered}
    ordered.extend(row for row in unique.values() if row["capture_key"] not in seen)
    selected = ordered[:limit or None]
    audits = {}
    output = Path(output)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_convert_one, row, str(output)): row["capture_key"] for row in selected}
        for future in as_completed(pending):
            key, audit = future.result()
            audits[key] = audit
            print(json.dumps({"event": "conversion", "completed": len(audits), "total": len(selected), "capture_key": key, **audit}), flush=True)
    atomic_write(output / "conversion-audit.json", {"source_rows": len(rows), "unique_captures": len(unique), "partial": bool(limit), "captures": audits})
    failures = [a for a in audits.values() if a["status"] != "ok"]
    if failures:
        raise ValueError(f"{len(failures)} captures failed; see {output / 'conversion-audit.json'}")
    if limit:
        return {"status": "partial_preflight", "captures": len(audits)}
    rows = grouped_split(rows, audits)
    split_files = {}
    for split in ("train", "validation", "test"):
        values = []
        for row in sorted(rows, key=lambda r: r["sample_id"]):
            if row["split"] != split:
                continue
            values.append({"version": DATASET_VERSION, "sample_id": row["sample_id"], "group_id": row["group_id"], "split": split,
                           "hardware_id": HARDWARE_ID, "input_path": f"models/{row['capture_key']}.json.gz",
                           "input_sha256": audits[row["capture_key"]]["design_sha256"],
                           "target_names": list(TARGET_NAMES), "targets": row["targets"], "provenance": row["provenance"]})
        path = output / split / "samples.json.gz"
        atomic_write(path, values, compress=True)
        split_files[split] = {"path": str(path.relative_to(output)), "rows": len(values), "sha256": file_sha256(path)}
    manifest = {"version": DATASET_VERSION, "hardware_id": HARDWARE_ID, "target_names": list(TARGET_NAMES), "total_rows": len(rows), "seed": 42,
                "split_files": split_files, "runtime_sha256": CALIBRATION_RUNTIME_SHA256, "capture_torch_version": torch.__version__, "all_captures_strict": True,
                "conversion_audit_sha256": file_sha256(output / "conversion-audit.json")}
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "dataset_manifest.json", manifest)
    return verify(output)


def _verify_input(item):
    path, expected_hash = item
    if file_sha256(path) != expected_hash:
        raise ValueError(f"input artifact hash differs: {path}")
    design = read_json(path)
    graph_from_design(design).validate()
    return path, _conservative_architecture_group(design)


def verify(output, workers=8):
    output = Path(output)
    manifest = read_json(output / "dataset_manifest.json")
    if manifest["version"] != DATASET_VERSION or tuple(manifest["target_names"]) != TARGET_NAMES:
        raise ValueError("dataset contract differs")
    if set(manifest["split_files"]) != {"train", "validation", "test"} or manifest["hardware_id"] != HARDWARE_ID or not manifest["all_captures_strict"]:
        raise ValueError("incomplete dataset contract")
    if manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("manifest fingerprint differs")
    if manifest.get("conversion_audit_sha256") != file_sha256(output / "conversion-audit.json"):
        raise ValueError("conversion audit fingerprint differs")
    audit = read_json(output / "conversion-audit.json")
    if audit["partial"] or audit["source_rows"] != 36409 or len(audit["captures"]) != audit["unique_captures"] or any(
        item["status"] != "ok" or item.get("backward_capture") != "strict" for item in audit["captures"].values()
    ):
        raise ValueError("incomplete or failed conversion audit")
    groups, samples, inputs = {}, set(), {}
    architectures, input_splits, model_splits = {}, {}, {}
    for split, meta in manifest["split_files"].items():
        path = output / meta["path"]
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("split hash differs")
        rows = read_json(path)
        if len(rows) != meta["rows"] or not rows:
            raise ValueError("empty or incomplete split")
        for row in rows:
            if row["sample_id"] in samples or row["split"] != split or row["hardware_id"] != HARDWARE_ID:
                raise ValueError("duplicate sample or split/hardware mismatch")
            samples.add(row["sample_id"])
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError("architecture group leakage")
            model_key = (row["provenance"]["source_bundle"], row["provenance"]["model_id"])
            if model_splits.setdefault(model_key, split) != split:
                raise ValueError("model configuration leakage")
            if tuple(row["target_names"]) != TARGET_NAMES or set(row["targets"]) != set(TARGET_NAMES):
                raise ValueError("unexpected training target")
            values = [row["targets"][name] for name in TARGET_NAMES]
            if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[2] <= 0 or not 0 <= values[1] <= 100:
                raise ValueError("invalid target values")
            input_path = output / row["input_path"]
            capture_key = input_path.name.removesuffix(".json.gz")
            if audit["captures"].get(capture_key, {}).get("design_sha256") != row["input_sha256"]:
                raise ValueError("input lacks matching conversion parity evidence")
            if inputs.setdefault(input_path, row["input_sha256"]) != row["input_sha256"]:
                raise ValueError("inconsistent input hash")
            if input_splits.setdefault(input_path, split) != split:
                raise ValueError("identical input leakage")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, architecture in pool.map(_verify_input, inputs.items()):
            split = input_splits[path]
            if architectures.setdefault(architecture, split) != split:
                raise ValueError("normalized architecture leakage")
    if len(samples) != manifest["total_rows"] or len(samples) != 36409:
        raise ValueError("not all source rows were converted")
    return {"status": "verified", "total_rows": len(samples), "unique_inputs": len(inputs), "groups": len(groups), "split_files": manifest["split_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "prepare", "verify"))
    parser.add_argument("--output", type=Path, default=DATA / "ready_for_train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Partial capture preflight; never produces a training manifest")
    args = parser.parse_args()
    result = migrate() if args.command == "migrate" else verify(args.output) if args.command == "verify" else prepare(args.output, args.workers, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
