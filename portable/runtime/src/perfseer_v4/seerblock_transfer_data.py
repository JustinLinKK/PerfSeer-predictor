"""CPU conversion and budget-restricted loading of existing RTX5090 graphs."""

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from perfseer_v3.features import apply_normalization
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .features import build_features
from .native_data import TARGET_NAMES
from .seerblock_data import seerblock_feature_layout

VERSION = "perfseer_seerblock_transfer_tensors_v1"
SPLITS = ("train", "validation", "test")
ROW_FIELDS = ("sample_id", "group_id", "hardware_id", "input_path", "input_sha256",
              "source_training_sha256", "split")


def verify_feature(feature):
    feature.validate()
    if feature.layout != seerblock_feature_layout() or not len(feature.x_cont):
        raise ValueError("transfer requires nonempty original 110-field SeerBlock features")
    if "normalization_sha256" in feature.metadata:
        raise ValueError("transfer cache must contain raw features; use source normalization at loading")
    for field in fields(feature):
        value = getattr(feature, field.name)
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite transfer tensor: {field.name}")


def write_index(output, rows, parent_fingerprint, *, access):
    atomic_write(output / "index.json", rows)
    manifest = {"version": VERSION, "source": "RTX5090", "target_names": list(TARGET_NAMES),
                "layout_sha256": seerblock_feature_layout().layout_sha256,
                "parent_fingerprint": parent_fingerprint, "access": access,
                "normalization": "raw tensors; apply selected source teacher normalization exactly once",
                "splits": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
                "rows": len(rows), "graphs": len({row["feature_path"] for row in rows}),
                "index_sha256": file_sha256(output / "index.json"),
                "converter_sha256": file_sha256(__file__), "workloads_executed": False}
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "dataset_manifest.json", manifest)


def read_index(root):
    root = Path(root)
    manifest = read_json(root / "dataset_manifest.json")
    if (manifest["version"] != VERSION or manifest["target_names"] != list(TARGET_NAMES)
            or manifest["layout_sha256"] != seerblock_feature_layout().layout_sha256
            or fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]
            or file_sha256(root / "index.json") != manifest["index_sha256"]):
        raise ValueError("transfer index identity or feature contract differs")
    return manifest, read_json(root / "index.json")


def read_feature(root, row):
    root = Path(root).resolve()
    path = (root / row["feature_path"]).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != row["feature_sha256"]:
        raise ValueError("transfer feature path or hash differs")
    feature = torch.load(path, map_location="cpu", weights_only=False)
    verify_feature(feature)
    return feature


def verify(root, expected_rows=None):
    manifest, rows = read_index(root)
    ids, groups, checked = set(), {}, {}
    expected = None if expected_rows is None else {row["sample_id"]: row for row in expected_rows}
    for row in rows:
        allowed = set(ROW_FIELDS) | {"feature_path", "feature_sha256", "graph_sha256", "nodes", "steps_per_epoch", "targets"}
        if set(row) - allowed:
            raise ValueError("unexpected transfer row fields may expose unauthorized labels")
        identifier, split = row["sample_id"], row["split"]
        if identifier in ids or split not in SPLITS or groups.setdefault(row["group_id"], split) != split:
            raise ValueError("transfer identity or group crosses frozen partitions")
        ids.add(identifier)
        if "targets" in row:
            values = torch.tensor([row["targets"][name] for name in TARGET_NAMES], dtype=torch.float64)
            if (set(row["targets"]) != set(TARGET_NAMES) or not torch.isfinite(values).all()
                    or (values[[0, 1, 2, 4, 5]] <= 0).any() or not 0 <= values[3] <= 100):
                raise ValueError("invalid six-target transfer labels")
        if expected is not None:
            reference = expected.get(identifier)
            if (reference is None or any(row[key] != reference[key] for key in ROW_FIELDS)
                    or row.get("targets") != reference.get("targets")):
                raise ValueError("transfer tensors changed identities, labels, or authorized label access")
        key = row["feature_path"]
        if key in checked and checked[key] != row["feature_sha256"]:
            raise ValueError("conflicting transfer feature identities")
        if key not in checked:
            feature = read_feature(root, row)
            if feature.graph_sha256 != row["graph_sha256"]:
                raise ValueError("encoded training graph identity differs")
            checked[key] = row["feature_sha256"]
    if (len(ids) != manifest["rows"] or len(checked) != manifest["graphs"]
            or {split: sum(row["split"] == split for row in rows) for split in SPLITS} != manifest["splits"]
            or expected is not None and ids != set(expected)):
        raise ValueError("transfer inventory differs")
    return {"status": "verified", "rows": len(ids), "graphs": len(checked),
            "splits": manifest["splits"], "layout_sha256": manifest["layout_sha256"],
            "global_fields": 110, "targets": 6, "groups": len(groups),
            "labeled_rows": sum("targets" in row for row in rows), "workloads_executed": False}


def prepare(input_root, output):
    input_root, output = Path(input_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("use a fresh conversion directory")
    source = read_json(input_root / "dataset_manifest.json")
    rows, cached = [], {}
    output.mkdir(parents=True)
    for split in SPLITS:
        split_file = source["split_files"][split]
        path = (input_root / split_file["path"]).resolve()
        if not path.is_relative_to(input_root) or file_sha256(path) != split_file["sha256"]:
            raise ValueError("original RTX5090 split hash differs")
        original = read_json(path)
        if len(original) != split_file["rows"]:
            raise ValueError("original RTX5090 row count differs")
        for row in original:
            graph = (input_root / row["input_path"]).resolve()
            if row["split"] != split or not graph.is_relative_to(input_root):
                raise ValueError("original RTX5090 split or graph path differs")
            digest = row["input_sha256"]
            if digest not in cached:
                if file_sha256(graph) != digest:
                    raise ValueError("original RTX5090 graph hash differs")
                design = read_json(graph)
                feature = build_features(design).training
                verify_feature(feature)
                relative = f"features/{digest}.pt"
                atomic_write(output / relative, feature, checkpoint=True)
                cached[digest] = {"feature_path": relative, "feature_sha256": file_sha256(output / relative),
                                  "graph_sha256": feature.graph_sha256, "nodes": len(feature.x_cont),
                                  "steps_per_epoch": design["training"]["TRAINING_CONFIG"]["steps_per_epoch"]}
            values = {name: row["targets"][name] for name in TARGET_NAMES}
            epoch = values["train_step_wall_ms"] * cached[digest]["steps_per_epoch"]
            if abs(values["train_epoch_ms"] - epoch) > 1e-6 * max(abs(epoch), 1):
                raise ValueError("RTX5090 derived epoch provenance differs")
            rows.append({**{key: row[key] for key in ROW_FIELDS}, **cached[digest], "targets": values})
            if len(rows) % 100 == 0:
                print(f"Converted {len(rows)}/{source['total_rows']} RTX5090 rows", flush=True)
    write_index(output, rows, source["fingerprint"], access="local_six_target_corpus")
    result = verify(output)
    atomic_write(output / "verification.json", result)
    return result


def publish(converted, output, public_rows):
    """Expose only labels already authorized in the ABA target JSONL files."""
    converted, output = Path(converted), Path(output)
    if output.exists():
        raise ValueError("use a fresh public transfer directory")
    manifest, rows = read_index(converted)
    by_id = {row["sample_id"]: row for row in rows}
    selected = []
    output.mkdir(parents=True)
    for public in public_rows:
        row = by_id[public["sample_id"]]
        if (any(row[key] != public[key] for key in ROW_FIELDS)
                or "targets" in public and row["targets"] != public["targets"]):
            raise ValueError("public transfer metadata or labels differ from converted corpus")
        projected = {key: value for key, value in row.items() if key != "targets"}
        if "targets" in public:
            projected["targets"] = public["targets"]
        destination = output / row["feature_path"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.hardlink_to(converted / row["feature_path"])
        selected.append(projected)
    write_index(output, selected, manifest["fingerprint"], access="authorized_budget_union_only")
    return verify(output, public_rows)


def load_inputs(root, normalization, sample_ids):
    """Load a bounded batch with source normalization and no target labels."""
    manifest, rows = read_index(root)
    by_id = {row["sample_id"]: row for row in rows}
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("require distinct transfer sample IDs")
    if normalization.layout_sha256 != manifest["layout_sha256"]:
        raise ValueError("source normalization must use the matching 110-field layout")
    selected = [by_id[identifier] for identifier in sample_ids]
    features = [apply_normalization(read_feature(root, row), normalization) for row in selected]
    selected_manifest = {"version": VERSION, "source": "RTX5090", "target_names": list(TARGET_NAMES),
                         "parent_fingerprint": manifest["fingerprint"], "layout_sha256": manifest["layout_sha256"],
                         "normalization_sha256": normalization.sha256, "sample_ids": list(sample_ids),
                         "rows": len(sample_ids)}
    selected_manifest["fingerprint"] = fingerprint(selected_manifest)
    return {"features": features, "sample_ids": list(sample_ids),
            "configuration_groups": [row["group_id"] for row in selected],
            "splits": torch.tensor([SPLITS.index(row["split"]) for row in selected]),
            "normalization": normalization, "manifest": selected_manifest}


def load_trial(root, subsets, normalization, budget, seed):
    """Return the standard features/targets/splits dictionary for one fixed trial."""
    trial = subsets[str(seed)][str(budget)]
    fit, selection = trial["fit_ids"], trial["selection_ids"]
    if len(fit) != 3 * budget // 4 or len(selection) != budget // 4 or len(set(fit + selection)) != budget:
        raise ValueError("transfer trial differs from its fixed label budget")
    _, rows = read_index(root)
    by_id = {row["sample_id"]: row for row in rows}
    for split, identifiers in (("train", fit), ("validation", selection)):
        if any(by_id[identifier]["split"] != split or "targets" not in by_id[identifier] for identifier in identifiers):
            raise ValueError("trial label is unavailable or outside the required partition")
    data = load_inputs(root, normalization, fit + selection)
    data["targets"] = torch.tensor([[by_id[identifier]["targets"][name] for name in TARGET_NAMES]
                                   for identifier in fit + selection], dtype=torch.float32)
    data["manifest"] = {**data["manifest"], "label_budget": budget, "seed": seed,
                        "selected_ids_sha256": fingerprint(fit + selection)}
    data["manifest"]["fingerprint"] = fingerprint({key: value for key, value in data["manifest"].items() if key != "fingerprint"})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("labels/RTX5090/ready_for_train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args.output) if args.verify else prepare(args.input, args.output)
    print(result, flush=True)


if __name__ == "__main__":
    main()
