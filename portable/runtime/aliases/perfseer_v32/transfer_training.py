"""Fine-tune the v3.2 teacher on native RTX 5090 labels, then distill S1."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import fcntl
import inspect
from pathlib import Path
import random

import numpy as np
import torch

from perfseer_v31.generated_model_runtime import GraphModel
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .capture import capture_inference, graphs_from_design
from .features import build_features, normalize_features
from .transfer_labeling import HARDWARE_ID, SEED, make_inputs
from .version import INPUT_SCHEMA_VERSION, TARGET_NAMES, validate_targets


VERSION = "perfseer_v32_rtx5090_transfer_training_v1"
SPLITS = {"train": 2988, "validation": 576, "test": 564}


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("dataset path escapes its root")
    return path


def load_labels(source):
    source = Path(source)
    labels = read_json(source / "labels-24h.json.gz")
    scope = read_json(source / "campaign-scope-24h.json.gz")
    if Counter(row["split"] for row in labels) != SPLITS:
        raise ValueError("incomplete 24h label splits")
    ids = [row["profile_point_id"] for row in labels]
    if len(set(ids)) != len(ids) or ids != scope["configuration_ids"]:
        raise ValueError("label configuration identities differ")
    groups = defaultdict(set)
    for row in labels:
        if row["status"] != "ok" or row["hardware_id"] != HARDWARE_ID or tuple(row["target_names"]) != TARGET_NAMES:
            raise ValueError("label hardware, status, or target contract differs")
        validate_targets([row["targets"][name] for name in TARGET_NAMES])
        if file_sha256(checked_path(source, row["graph_path"])) != row["graph_sha256"]:
            raise ValueError("original training graph hash differs")
        groups[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("architecture group leaks across splits")
    return labels


def _prepare_group(source, output, rows, anchor, material):
    torch.set_num_threads(1)
    source, output = Path(source), Path(output)
    row = rows[0]
    identity = fingerprint({"anchor": anchor, "material": material, "training": row["training"],
                            "code": fingerprint([inspect.getsource(_prepare_group), inspect.getsource(make_inputs),
                                                 file_sha256(Path(__file__).with_name("capture.py"))]),
                            "torch": str(torch.__version__)})
    paths = [output / "pairs" / (item["profile_point_id"] + ".json.gz") for item in rows]
    audits = [path.with_suffix(".audit.json") for path in paths]
    if all(path.exists() and audit.exists() for path, audit in zip(paths, audits)):
        if all(read_json(audit).get("capture_identity") == identity and
               read_json(audit)["sha256"] == file_sha256(path) for path, audit in zip(paths, audits)):
            return len(rows)
        raise ValueError("cached transfer capture differs; use a fresh output directory")
    torch.manual_seed(SEED)
    model = GraphModel(anchor["model"]["NODE_SPECS"])
    inputs = make_inputs(anchor, row["training"]["microbatch_size"], material, 0)
    inference, audit = capture_inference(model, inputs, row["training"], architecture_key=anchor["anchor_id"])
    for item, path, audit_path in zip(rows, paths, audits):
        training = read_json(checked_path(source, item["graph_path"]))
        metadata = training["graph"]["metadata"]
        inferred = {**inference, "graph": {**inference["graph"], "metadata": {
            **inference["graph"]["metadata"], **{key: metadata[key] for key in
                ("target_hardware_id", "hardware_profile", "hardware_profile_sha256")}}}}
        design = {"version": INPUT_SCHEMA_VERSION, "training": training, "inference": inferred}
        graphs = graphs_from_design(design)
        if graphs["training"].training_config != item["training"]:
            raise ValueError("measured and captured training settings differ")
        build_features(design).validate()
        atomic_write(path, design, compress=True)
        atomic_write(audit_path, {**audit, "capture_identity": identity, "sha256": file_sha256(path),
                                 "source_training_sha256": item["graph_sha256"],
                                 "torch": str(torch.__version__), "device": "cpu"})
    return len(rows)


def prepare(source, output, workers=4):
    source, output = Path(source), Path(output)
    labels = load_labels(source)
    anchors = {row["anchor_id"]: row for row in read_json(source / "anchors.json.gz")}
    materials = read_json(source / "source_materials.json.gz")
    groups = defaultdict(list)
    for row in labels:
        groups[(row["anchor_id"], row["training"]["microbatch_size"], row["training"]["precision"])].append(row)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for (anchor_id, _, _), rows in groups.items():
            anchor = anchors[anchor_id]
            material = materials[anchor["identity"]["dataset_id"] + "::" + anchor["identity"]["subset_id"]]
            futures.append(pool.submit(_prepare_group, source, output, rows, anchor, material))
        for future in as_completed(futures):
            completed += future.result()
            if completed % 50 < 5 or completed == len(labels):
                print(f"verified paired graphs: {completed}/{len(labels)}", flush=True)
    splits = {}
    for split in SPLITS:
        rows = []
        for row in labels:
            if row["split"] != split:
                continue
            path = "pairs/" + row["profile_point_id"] + ".json.gz"
            rows.append({"sample_id": row["profile_point_id"], "group_id": row["group_id"],
                         "target_names": list(TARGET_NAMES), "targets": {name: row["targets"][name] for name in TARGET_NAMES},
                         "native_targets": row["targets"], "hardware_id": HARDWARE_ID,
                         "input_path": path, "input_sha256": file_sha256(output / path),
                         "source_training_sha256": row["graph_sha256"]})
        path = split + "/samples.json.gz"
        atomic_write(output / path, rows, compress=True)
        splits[split] = {"path": path, "sha256": file_sha256(output / path), "rows": len(rows)}
    manifest = {"version": VERSION, "prediction_hardware": HARDWARE_ID, "target_names": list(TARGET_NAMES),
                "split_files": splits, "source_labels_sha256": file_sha256(source / "labels-24h.json.gz"),
                "label_policy": {"version": "native_rtx5090_measurements", "targets_modified": False}}
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "dataset_manifest.json", manifest)
    return verify(output, source, workers)


def _verify_prepared_row(item):
    dataset, source, split, row, original = item
    pair = checked_path(dataset, row["input_path"])
    if file_sha256(pair) != row["input_sha256"]:
        raise ValueError("paired graph hash differs")
    design = read_json(pair)
    graphs = graphs_from_design(design)
    if any(graph.metadata.get("target_hardware_id") != HARDWARE_ID for graph in graphs.values()):
        raise ValueError("paired graph prediction hardware differs")
    if original and (row["native_targets"] != original["targets"] or split != original["split"] or
                     row["group_id"] != original["group_id"] or
                     design["training"] != read_json(checked_path(source, original["graph_path"]))):
        raise ValueError("transfer data differs from native measurement or graph")


def verify(dataset, source=None, workers=4):
    dataset = Path(dataset)
    manifest = read_json(dataset / "dataset_manifest.json")
    if manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("transfer dataset fingerprint differs")
    if manifest["version"] != VERSION or manifest["prediction_hardware"] != HARDWARE_ID or tuple(manifest["target_names"]) != TARGET_NAMES:
        raise ValueError("transfer dataset contract differs")
    native = {row["profile_point_id"]: row for row in load_labels(source)} if source else None
    if source and manifest["source_labels_sha256"] != file_sha256(Path(source) / "labels-24h.json.gz"):
        raise ValueError("native label file differs")
    ids, groups, counts, pairs = set(), defaultdict(set), {}, []
    for split, meta in manifest["split_files"].items():
        path = checked_path(dataset, meta["path"])
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("transfer split hash differs")
        rows = read_json(path)
        counts[split] = len(rows)
        if len(rows) != meta["rows"]:
            raise ValueError("transfer split row count differs")
        for row in rows:
            if row["sample_id"] in ids or row["hardware_id"] != HARDWARE_ID or tuple(row["target_names"]) != TARGET_NAMES:
                raise ValueError("duplicate sample or invalid hardware/target contract")
            ids.add(row["sample_id"])
            groups[row["group_id"]].add(split)
            validate_targets([row["targets"][name] for name in TARGET_NAMES])
            if row["targets"] != {name: row["native_targets"][name] for name in TARGET_NAMES}:
                raise ValueError("native transfer targets changed")
            pairs.append((dataset, source, split, row, native[row["sample_id"]] if native else None))
    if counts != SPLITS or any(len(value) != 1 for value in groups.values()):
        raise ValueError("incomplete splits or architecture leakage")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(_verify_prepared_row, pairs, chunksize=4), 1):
            if index % 500 == 0 or index == len(pairs):
                print(f"reconciled graph pairs and native labels: {index}/{len(pairs)}", flush=True)
    return {"status": "ok", "rows": len(ids), "splits": counts, "architecture_groups": len(groups),
            "prediction_hardware": HARDWARE_ID, "dataset_fingerprint": manifest["fingerprint"]}


def transfer_normalization(base, dataset_fingerprint):
    from .runner import normalization_from_dict
    normalization = normalization_from_dict(base["normalization"])
    if normalization.split_fingerprint != base["dataset_fingerprint"]:
        raise ValueError("base normalization dataset fingerprint differs")
    # Preserve the pretrained numeric transform; bind its reuse to the new dataset.
    return replace(normalization, **{mode: replace(getattr(normalization, mode), split_fingerprint=dataset_fingerprint)
                                     for mode in ("training", "inference")})


def train(args):
    from .inference import export_model
    from .runner import Samples, code_fingerprint, run_gate, run_stage
    from .training import evaluate, restore_model
    from .version import HARDWARE_ID as BASE_HARDWARE_ID

    verify(args.dataset)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this training entry point requires the requested A100 CUDA device")
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if base.get("role") != "teacher" or base.get("prediction_hardware", BASE_HARDWARE_ID) != BASE_HARDWARE_ID:
        raise ValueError("expected a pretrained A10 v3.2 teacher checkpoint")
    baseline = restore_model(base, device="cuda")
    manifest = read_json(args.dataset / "dataset_manifest.json")
    normalization = transfer_normalization(base, manifest["fingerprint"])
    manifest["transfer"] = {"version": VERSION, "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
                            "base_dataset_fingerprint": base["dataset_fingerprint"],
                            "base_normalization_sha256": base["normalization_sha256"],
                            "base_label_policy": base.get("label_policy"),
                            "target_hardware_id": HARDWARE_ID, "base_hardware_id": BASE_HARDWARE_ID,
                            "normalization_policy": "frozen_base_training_transform", "trainable_parameters": "all",
                            "transfer_code_sha256": file_sha256(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"dataset_fingerprint": manifest["fingerprint"], "transfer": manifest["transfer"],
                "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs,
                "learning_rate": args.learning_rate, "effective_batch": args.effective_batch,
                "microbatch": args.microbatch, "code_fingerprint": code_fingerprint()}
    identity_path = args.output / "transfer-run.json"
    if identity_path.exists():
        if not args.resume or read_json(identity_path) != identity:
            raise ValueError("existing transfer run differs; use --resume with the same inputs or a fresh output")
    else:
        if any(args.output.glob("*.pt")):
            raise ValueError("output already contains checkpoints without transfer identity")
        atomic_write(identity_path, identity)
    if (args.output / "transfer-report.json").exists():
        print("Transfer training already completed; see transfer-report.json", flush=True)
        return
    atomic_write(args.output / "normalization.json", asdict(normalization))
    cache = args.output / "feature-cache" / manifest["fingerprint"]
    rows = {split: read_json(args.dataset / meta["path"]) for split, meta in manifest["split_files"].items() if split != "test"}
    samples = {split: Samples(args.dataset, items, normalization, cache) for split, items in rows.items()}
    # Checkpoint restoration and actual normalized features must agree before updates.
    normalize_features(build_features(read_json(args.dataset / rows["train"][0]["input_path"])), normalization).validate()
    baseline_path = args.output / "base-validation.json"
    if not baseline_path.exists():
        metrics = evaluate(baseline, samples["validation"], args.microbatch,
                           prediction_path=args.output / "base-validation-predictions.json.gz", identity=identity)
        atomic_write(baseline_path, metrics)
    del baseline, base
    torch.cuda.empty_cache()
    if args.resume and (args.output / "teacher-gate.json").exists():
        selected = torch.load(args.output / "teacher-best.pt", map_location="cpu", weights_only=False)
        if selected.get("transfer") != manifest["transfer"]:
            raise ValueError("completed teacher transfer lineage differs")
        teacher = restore_model(selected, dataset_fingerprint=manifest["fingerprint"], device="cuda")
    else:
        teacher, selected = run_stage("teacher", samples["train"], samples["validation"], args.output, manifest,
                                      normalization, "cuda", resume=args.resume, epochs=args.teacher_epochs,
                                      effective_batch=args.effective_batch, maximum_microbatch=args.microbatch,
                                      initial_checkpoint=args.base_checkpoint, learning_rate=args.learning_rate,
                                      early_stopping_patience=10, early_stopping_min_epochs=20)
    export_model(selected, args.output / "teacher-rtx5090-candidate.pt")
    teacher_gate = run_gate("teacher", teacher, selected, args.dataset, args.output, manifest, normalization)
    report = {**identity, "teacher_gate": teacher_gate, "base_validation": read_json(baseline_path),
              "prediction_hardware": HARDWARE_ID, "training_device": torch.cuda.get_device_name(0)}
    if not teacher_gate["passed"]:
        atomic_write(args.output / "transfer-report.json", {**report, "status": "teacher_gate_failed", "student_started": False})
        print("Teacher quality gate failed; distillation will not run.", flush=True)
        return
    teacher.eval().requires_grad_(False)
    student, selected_student = run_stage("student", samples["train"], samples["validation"], args.output, manifest,
                                          normalization, "cuda", teacher=teacher, resume=args.resume,
                                          epochs=args.student_epochs, effective_batch=args.effective_batch,
                                          maximum_microbatch=args.microbatch)
    export_model(selected_student, args.output / "student-rtx5090-candidate.pt")
    student_gate = run_gate("student", student, selected_student, args.dataset, args.output, manifest, normalization)
    if student_gate["passed"]:
        export_model(selected_student, args.output / "student-rtx5090-accepted.pt")
    report.update(student_gate=student_gate, student_started=True,
                  status="accepted" if student_gate["passed"] else "student_gate_failed")
    atomic_write(args.output / "transfer-report.json", report)
    print(f"Transfer training complete: {report['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "train"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.teacher_epochs, args.student_epochs, args.effective_batch, args.microbatch) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("workers, epochs, batch sizes, and learning rate must be positive")
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        print(prepare(args.source, args.dataset, args.workers))
    elif args.command == "verify":
        print(verify(args.dataset, args.source, args.workers))
    else:
        if args.base_checkpoint is None or args.output is None:
            parser.error("train requires --base-checkpoint and --output")
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".transfer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                train(args)
            except InterruptedError as error:
                print(error, flush=True)
                raise SystemExit(75)


if __name__ == "__main__":
    main()
