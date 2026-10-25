"""Independent source, campaign, and twelve-target measurement reconciliation."""

from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path

from perfseer_v31 import dataset as compiler
from perfseer_v31.capture_training import graph_from_design
from perfseer_v31.io import file_sha256, fingerprint, read_json
from perfseer_v3.hardware import HardwareProfileV3

from .dataset import DATA
from .transfer_labeling import (
    ANCHOR_COUNTS, BATCHES, BUDGETS, HARDWARE_ID, MODALITIES, PRECISIONS, PROTOCOL,
    REDUCED_ACCUMULATION_CELLS, REDUCED_ATTEMPTS_PER_HOUR, REDUCED_BATCHES,
    REDUCED_BUDGETS, REDUCED_OPTIMIZER_BATCHES, REDUCED_PLANNED_ATTEMPTS,
    REDUCED_PLANNED_RECORDS, REDUCED_REPEAT_PANEL_SIZE, SECONDARY_COUNTS, SEED,
    SPLIT_COUNTS, TARGET_NAMES, VERSION, campaign_scope_path,
    resolve_attempt_execution,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_targets(values):
    require(set(values) == {*TARGET_NAMES, "train_epoch_ms_step_extrapolated"}, "incomplete twelve-target output")
    for name, value in values.items():
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, f"invalid target: {name}")
        if name.endswith("_ms") or "vram" in name or name == "train_peak_torch_allocated_mib":
            require(value > 0, f"nonpositive target: {name}")
        if "sm_util" in name:
            require(value <= 100, f"utilization out of range: {name}")


def reconstruct_targets(phases, *, first_two=False):
    """Compute from primitive samples/counters, without the labeler's reducer."""
    result = {}
    for name in ("train", "infer"):
        phase = phases[name]
        blocks = phase["blocks"][:2] if name == "train" and first_two else phase["blocks"]
        samples = [s for s in phase["samples"] if not (name == "train" and first_two) or s["t_ms"] <= blocks[-1]["end_ms"]]
        require(bool(samples), "phase has no telemetry")
        steps = sum(block["steps"] for block in blocks)
        result[name + "_step_wall_ms"] = math.fsum(b["wall_ms"] for b in blocks) / steps
        result[name + "_step_gpu_ms"] = math.fsum(b["gpu_ms"] for b in blocks) / steps
        result[name + "_avg_sm_util_percent"] = math.fsum(s["sm_percent"] for s in samples) / len(samples)
        result[name + "_avg_vram_mib"] = math.fsum(s["vram_mib"] for s in samples) / len(samples)
        result[name + "_peak_vram_mib"] = max(s["vram_mib"] for s in samples)
        if name == "train":
            result["train_epoch_ms"] = math.fsum(b["wall_ms"] for b in blocks) / len(blocks)
            result["train_peak_torch_allocated_mib"] = (blocks[-1]["peak_torch_allocated_mib"] if first_two
                                                       else phase["peak_torch_allocated_mib"])
            result["train_epoch_ms_step_extrapolated"] = result["train_step_wall_ms"] * blocks[0]["steps"]
    return result


def verify_graph(design, row, *, hardware=None):
    graph = graph_from_design(design)
    graph.validate()
    require(design["TRAINING_CONFIG"] == row["training"], "graph training configuration differs")
    require(graph.metadata["target_hardware_id"] == HARDWARE_ID, "graph hardware identity differs")
    require(graph.metadata["configuration_id"] == row["configuration_id"], "graph configuration identity differs")
    require(graph.metadata["batch_size"] == row["training"]["microbatch_size"], "graph batch metadata differs")
    inputs = [edge for edge in graph.tensor_edges if edge.tensor_role == "model_input"]
    require(bool(inputs), "graph has no captured model inputs")
    require(all(edge.shape[0] == row["training"]["microbatch_size"] for edge in inputs), "captured input batch differs")
    if hardware is not None:
        expected = HardwareProfileV3(HARDWARE_ID, hardware["static"], {}, hardware["environment"])
        require(graph.metadata["hardware_profile"] == expected.canonical_payload and
                graph.metadata["hardware_profile_sha256"] == expected.sha256, "graph physical hardware differs")


def verify_attempt(result, row, manifest, execution, output):
    attempt_execution = resolve_attempt_execution(output, execution, result.get("execution_fingerprint"))
    require(result["status"] == "ok", "attempt did not succeed")
    require(result["version"] == VERSION and result["configuration_id"] == row["configuration_id"], "attempt identity differs")
    require(result["campaign_fingerprint"] == manifest["fingerprint"] and
            result["execution_fingerprint"] == attempt_execution["fingerprint"], "attempt execution lineage differs")
    require(tuple(result["target_names"]) == TARGET_NAMES, "attempt target order differs")
    require(result.get("phase_order") == ["infer", "train"], "inference/training phase order differs")
    require(set(result["phases"]) == {"train", "infer"}, "incomplete measurement phases")
    for name, phase in result["phases"].items():
        require(phase["status"] == "ok" and phase["phase"] == name, "phase did not succeed")
        blocks, samples = phase["blocks"], phase["samples"]
        minimum = PROTOCOL["infer_min_steps"] if name == "infer" else PROTOCOL["train_min_epochs"]
        steps = 1 if name == "infer" else row["training"]["steps_per_epoch"]
        warmup = PROTOCOL["infer_warmup_steps"] if name == "infer" else steps
        require(len(blocks) >= minimum and phase["warmup_steps"] == warmup, "incomplete measurement/warmup")
        require(sum(b["wall_ms"] for b in blocks) >= PROTOCOL["min_phase_seconds"] * 1000, "phase is too short")
        previous = 0.
        for block in blocks:
            require(block["steps"] == steps, "logical measurement steps differ")
            for key in ("wall_ms", "gpu_ms"):
                require(math.isfinite(block[key]) and block[key] > 0, "invalid raw duration")
            require(previous <= block["start_ms"] < block["end_ms"], "invalid measurement time boundaries")
            require(math.isclose(block["end_ms"] - block["start_ms"], block["wall_ms"], rel_tol=1e-7, abs_tol=1e-4),
                    "wall duration differs from timestamps")
            previous = block["end_ms"]
        require(bool(samples), "phase has no telemetry")
        previous = -1.
        for sample in samples:
            require(math.isfinite(sample["t_ms"]) and sample["t_ms"] >= previous, "invalid telemetry time")
            require(math.isfinite(sample["sm_percent"]) and 0 <= sample["sm_percent"] <= 100, "invalid raw utilization")
            require(math.isfinite(sample["vram_mib"]) and sample["vram_mib"] > 0, "invalid raw VRAM")
            previous = sample["t_ms"]
        require(math.isfinite(phase["peak_torch_allocated_mib"]) and phase["peak_torch_allocated_mib"] > 0,
                "invalid allocated-memory peak")
    checked_targets(result["targets"])
    expected = reconstruct_targets(result["phases"])
    for name, value in expected.items():
        require(math.isclose(result["targets"][name], value, rel_tol=1e-10, abs_tol=1e-8), f"raw target reconstruction differs: {name}")
    comparison = result.get("first_two_epoch_comparison")
    if comparison is not None:
        expected_comparison = reconstruct_targets(result["phases"], first_two=True)
        for name, value in expected_comparison.items():
            require(math.isclose(comparison[name], value, rel_tol=1e-10, abs_tol=1e-8), f"source-protocol diagnostic differs: {name}")
    path = Path(output) / result["graph_path"]
    require(path.resolve().is_relative_to(Path(output).resolve()), "graph path escapes output")
    require(file_sha256(path) == result["graph_sha256"], "captured graph hash differs")
    verify_graph(read_json(path), row, hardware=result["hardware"])


def _balanced_order(rows, *, salt):
    buckets = defaultdict(list)
    for row in rows:
        training = row["training"]
        buckets[(row["modality"], training["microbatch_size"], training["precision"], row["cohort"])].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda row: fingerprint([SEED, salt, row["group_id"], row["configuration_id"]]))
    keys = sorted(buckets, key=lambda key: fingerprint([SEED, salt, key]))
    result = []
    for index in range(max(map(len, buckets.values()))):
        result.extend(buckets[key][index]["configuration_id"] for key in keys if index < len(buckets[key]))
    return result


def verify_campaign_scope(output, rows, experiments, manifest, profile):
    require(profile == "24h", f"unsupported campaign profile: {profile}")
    selected = []
    for row in rows:
        training = row["training"]
        batch = training["microbatch_size"]
        accumulation = training["gradient_accumulation_steps"]
        if row["cohort"] == "batch" and batch in REDUCED_BATCHES:
            selected.append(row)
        elif row["cohort"] == "optimizer" and batch in REDUCED_OPTIMIZER_BATCHES:
            selected.append(row)
        elif row["cohort"] == "accumulation" and (batch, accumulation) in REDUCED_ACCUMULATION_CELLS:
            selected.append(row)
    selected_ids = [row["configuration_id"] for row in selected]
    selected_set = set(selected_ids)
    full_panel = set(experiments["repeat_panel"])
    candidates = [row for row in selected if row["configuration_id"] in full_panel]
    panel = _balanced_order(candidates, salt="24h-repeat")[:REDUCED_REPEAT_PANEL_SIZE]
    train = [row for row in selected if row["split"] == "train"]
    order = _balanced_order(train, salt="24h-budgets")
    expected = {
        "version": "perfseer_v32_transfer_campaign_scope_v1",
        "profile": profile,
        "campaign_fingerprint": manifest["fingerprint"],
        "selection": {
            "batch_sweep": list(REDUCED_BATCHES),
            "optimizer_batches": list(REDUCED_OPTIMIZER_BATCHES),
            "accumulation_cells": [list(cell) for cell in REDUCED_ACCUMULATION_CELLS],
        },
        "configuration_ids": selected_ids,
        "planned_records": len(selected),
        "planned_attempts": len(selected) + 2 * len(panel),
        "repeat_panel": panel,
        "pilot": [key for key in experiments["pilot"] if key in selected_set],
        "nested_training_budgets": {str(budget): order[:budget] for budget in REDUCED_BUDGETS},
        "validation": [row["configuration_id"] for row in selected if row["split"] == "validation"],
        "test": [row["configuration_id"] for row in selected if row["split"] == "test"],
        "batch8_only": {split: [row["configuration_id"] for row in selected
                                if row["split"] == split and row["a10_matched"]]
                        for split in SPLIT_COUNTS},
        "batch_generalization_train": [row["configuration_id"] for row in train
                                        if row["training"]["microbatch_size"] not in (16, 64)],
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "cohort_counts": dict(Counter(row["cohort"] for row in selected)),
        "matched_records": sum(row["a10_matched"] for row in selected),
        "estimated_hours": REDUCED_PLANNED_ATTEMPTS / REDUCED_ATTEMPTS_PER_HOUR,
        "planning_attempts_per_hour": REDUCED_ATTEMPTS_PER_HOUR,
    }
    expected["fingerprint"] = fingerprint(expected)
    scope = read_json(campaign_scope_path(output, profile))
    require(scope == expected, "24h campaign scope differs")
    require(scope["planned_records"] == REDUCED_PLANNED_RECORDS and
            scope["planned_attempts"] == REDUCED_PLANNED_ATTEMPTS, "24h campaign size differs")
    require(scope["split_counts"] == {"train": 2988, "validation": 576, "test": 564}, "24h split counts differ")
    require(scope["cohort_counts"] == {"batch": 2976, "optimizer": 768, "accumulation": 384},
            "24h cohort counts differ")
    require(scope["matched_records"] == 992 and len(scope["pilot"]) == 48, "24h matched/pilot coverage differs")
    require(len(scope["repeat_panel"]) == len(set(scope["repeat_panel"])) == REDUCED_REPEAT_PANEL_SIZE,
            "24h repeat panel differs")
    return scope, selected


def verify_exported_labels(output, labels, by_id, panel, execution, *, scope=None):
    require(len(labels) == len({row["profile_point_id"] for row in labels}), "duplicate exported labels")
    allowed = set(by_id)
    for label in labels:
        key = label["profile_point_id"]
        require(key in allowed and tuple(label["target_names"]) == TARGET_NAMES and label["status"] == "ok",
                "invalid exported label")
        require(label["version"] == VERSION and label["scheduler_label_version"] == 4 and
                label["hardware_id"] == HARDWARE_ID, "exported label contract differs")
        resolve_attempt_execution(output, execution,
                                  label.get("campaign_execution_fingerprint", label["execution_fingerprint"]))
        if scope is not None:
            require(label.get("campaign_scope_fingerprint") == scope["fingerprint"],
                    "exported campaign scope differs")
        require(all(label[name] == by_id[key][name]
                    for name in ("split", "group_id", "anchor_id", "training", "paired_source_ids")),
                "exported configuration lineage differs")
        checked_targets(label["targets"])
        primary = read_json(Path(output) / "attempts" / f"{key}.0.json.gz")
        require(primary["status"] == "ok" and label["targets"] == primary["targets"],
                "export differs from primary measurement")
        require(label["execution_fingerprint"] == primary["execution_fingerprint"],
                "exported attempt execution differs")
        require(label["graph_path"] == primary["graph_path"] and label["graph_sha256"] == primary["graph_sha256"],
                "exported graph lineage differs")
        repeated = []
        for repetition in range(3 if key in panel else 1):
            result = read_json(Path(output) / "attempts" / f"{key}.{repetition}.json.gz")
            require(result["status"] == "ok", "exported repeat panel is incomplete")
            repeated.append(result)
        for name in TARGET_NAMES:
            values = [result["targets"][name] for result in repeated]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            require(label["repeat_quality"][name]["values"] == values and
                    math.isclose(label["repeat_quality"][name]["std"], std, rel_tol=1e-9, abs_tol=1e-8),
                    "repeat quality evidence differs")


def verify(output, data=DATA, *, require_complete=False, campaign_profile="full"):
    output, data = Path(output), Path(data)
    manifest = read_json(output / "manifest.json")
    require(manifest["fingerprint"] == fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}),
            "campaign fingerprint differs")
    require(manifest["version"] == VERSION and manifest["hardware_id"] == HARDWARE_ID and
            tuple(manifest["target_names"]) == TARGET_NAMES, "campaign output contract differs")
    require(manifest["protocol"] == PROTOCOL and manifest["seed"] == SEED, "campaign protocol/seed differs")
    require(manifest["scheduler_label_version"] == 4 and manifest["auxiliary_targets"] == ["train_epoch_ms_step_extrapolated"],
            "native output schema/auxiliary contract differs")
    required_files = {"anchors.json.gz", "source_aliases.json.gz", "configurations.json.gz", "experiments.json.gz", "coverage.json"}
    require(set(manifest["files"]) == required_files, "campaign file set differs")
    for name, digest in manifest["files"].items():
        require(file_sha256(output / name) == digest, f"campaign file hash differs: {name}")
    anchors = read_json(output / "anchors.json.gz")
    aliases = read_json(output / "source_aliases.json.gz")
    rows = read_json(output / "configurations.json.gz")
    experiments = read_json(output / "experiments.json.gz")
    by_anchor = {a["anchor_id"]: a for a in anchors}
    by_id = {r["configuration_id"]: r for r in rows}
    require(len(anchors) == len(by_anchor) == 248, "anchor count/uniqueness differs")
    require(len(rows) == len(by_id) == manifest["planned_records"] == 10240, "configuration count/uniqueness differs")
    require(dict(Counter(a["split"] for a in anchors)) == ANCHOR_COUNTS, "anchor splits differ")
    require(dict(Counter(r["split"] for r in rows)) == manifest["split_counts"] == SPLIT_COUNTS, "configuration splits differ")
    require(dict(Counter(r["cohort"] for r in rows)) == {"batch": 7936, "optimizer": 1536, "accumulation": 768},
            "component counts differ")
    base = read_json(data / "ready_for_train/dataset_manifest.json")
    require(base["fingerprint"] == manifest["source_dataset_fingerprint"], "source corpus identity differs")
    require(base["fingerprint"] == fingerprint({k: v for k, v in base.items() if k != "fingerprint"}), "source manifest hash differs")
    source = read_json(data / "source_manifest.json")
    require(file_sha256(data / "source_manifest.json") == base["source_manifest_sha256"], "source manifest lineage differs")
    require(file_sha256(data / "merged_samples.json.gz") == source["merged_sha256"], "merged source hash differs")
    require(file_sha256(compiler.PACKAGE / "generated_model_runtime.py") == manifest["runtime_sha256"] ==
            compiler.CALIBRATION_RUNTIME_SHA256, "runtime pin differs")
    raw = {r["sample_id"]: r for r in read_json(data / "merged_samples.json.gz")}
    prepared = {}
    for split, meta in base["split_files"].items():
        path = data / "ready_for_train" / meta["path"]
        require(file_sha256(path) == meta["sha256"], "source split hash differs")
        values = read_json(path)
        require(len(values) == meta["rows"], "source split length differs")
        for row in values:
            require(row["split"] == split and row["sample_id"] not in prepared, "source split membership differs")
            prepared[row["sample_id"]] = row
    require(len(aliases) == len(raw) == len(prepared) == 40020, "source aliases are incomplete")
    seen, models, groups, source_ids = set(), {}, {}, {}
    for alias in aliases:
        sample_id = alias["sample_id"]
        require(sample_id not in seen, "duplicate source alias")
        seen.add(sample_id)
        native = raw[sample_id]
        require(all(alias[k] == v for k, v in native.items()), "native source alias changed")
        require(tuple(native["native_label"]["target_names"]) == TARGET_NAMES, "native source target order differs")
        checked_targets(native["native_label"]["targets"])
        require(alias["split"] == prepared[sample_id]["split"] and alias["group_id"] == prepared[sample_id]["group_id"],
                "source split/group lineage differs")
        anchor = by_anchor[alias["anchor_id"]]
        require(anchor["split"] == alias["split"] and anchor["group_id"] == alias["group_id"], "anchor source lineage differs")
        path = data / native["model_source_path"]
        if path not in models:
            require(file_sha256(path) == native["model_source_sha256"], "source model hash differs")
            model = compiler._constants(path)
            models[path] = {"nodes": compiler._native_architecture(model["NODE_SPECS"]),
                            "inputs": [{**s, "shape": [1, *s["shape"][1:]]} for s in model["INPUT_SPECS"]]}
        dataset = native["native_label"]["dataset"]
        identity = {**models[path], "dataset_id": dataset["dataset_id"], "subset_id": dataset["subset_id"],
                    "num_samples": int(dataset["num_samples"])}
        require(identity == anchor["identity"] and fingerprint(identity) == anchor["anchor_id"], "source canonical workload differs")
        precision = native["native_label"]["training"]["precision"]
        source_ids.setdefault((anchor["anchor_id"], precision), set()).add(sample_id)
    for anchor in anchors:
        require(groups.setdefault(anchor["group_id"], anchor["split"]) == anchor["split"], "architecture group leakage")
        require(anchor["identity"]["nodes"] == compiler._native_architecture(anchor["model"]["NODE_SPECS"]),
                "stored executable model differs")
        require(anchor["identity"]["inputs"] == [{**s, "shape": [1, *s["shape"][1:]]} for s in anchor["model"]["INPUT_SPECS"]],
                "stored executable input differs")
        for precision in PRECISIONS:
            require(set(anchor["source_ids"][precision]) == source_ids[(anchor["anchor_id"], precision)],
                    "precision source pairing differs")
    require(len(groups) == 119, "source architecture coverage differs")
    secondary = experiments["secondary_anchors"]
    require(len(secondary) == len(set(secondary)) == 48, "secondary anchor count differs")
    require(dict(Counter(by_anchor[k]["split"] for k in secondary)) == SECONDARY_COUNTS, "secondary splits differ")
    for split in SPLIT_COUNTS:
        require({by_anchor[k]["modality"] for k in secondary if by_anchor[k]["split"] == split} == set(MODALITIES),
                "secondary modality coverage differs")
    actual_cells = {}
    for row in rows:
        anchor, training = by_anchor[row["anchor_id"]], row["training"]
        require(all(row[k] == anchor[k] for k in ("split", "group_id", "modality")), "configuration split/structure differs")
        batch, accumulation, optimizer = training["microbatch_size"], training["gradient_accumulation_steps"], training["optimizer"]["name"]
        effective = batch * accumulation
        require(row["effective_batch_size"] == effective, "effective batch differs")
        require(training["steps_per_epoch"] == math.ceil(anchor["identity"]["num_samples"] / effective), "epoch length differs")
        require(training["optimizer"]["learning_rate"] == .001 and training["scheduler"] == {"name": "none"},
                "optimizer/scheduler contract differs")
        require(row["configuration_id"] == fingerprint({"anchor_id": row["anchor_id"], "training": training, "hardware_id": HARDWARE_ID}),
                "configuration fingerprint differs")
        matched = batch == 8 and accumulation == 1 and optimizer == "adam"
        require(row["a10_matched"] == matched and row["paired_source_ids"] == (anchor["source_ids"][training["precision"]] if matched else []),
                "A10 pairing/new-configuration isolation differs")
        cell = (batch, training["precision"], optimizer, accumulation)
        actual_cells.setdefault(row["anchor_id"], []).append(cell)
    for anchor in anchors:
        expected = {(b, p, "adam", 1) for b, p in itertools.product(BATCHES, PRECISIONS)}
        if anchor["anchor_id"] in secondary:
            expected.update((b, p, opt, 1) for b, p, opt in itertools.product((4, 8, 32, 128), PRECISIONS, ("adamw", "sgd")))
            expected.update((effective // g, p, "adam", g) for effective, g, p in itertools.product((64, 128), (4, 8), PRECISIONS))
        cells = actual_cells[anchor["anchor_id"]]
        require(len(cells) == len(set(cells)) and set(cells) == expected, "batch/optimizer/accumulation coverage differs")
    require(sum(r["a10_matched"] for r in rows) == 992, "matched configuration count differs")
    train = {r["configuration_id"] for r in rows if r["split"] == "train"}
    previous = []
    require(set(experiments["nested_training_budgets"]) == {str(b) for b in BUDGETS}, "budget list differs")
    for budget in BUDGETS:
        ids = experiments["nested_training_budgets"][str(budget)]
        require(len(ids) == len(set(ids)) == budget and set(ids) <= train and ids[:len(previous)] == previous, "nested budget differs")
        previous = ids
    for split in ("validation", "test"):
        require(set(experiments[split]) == {r["configuration_id"] for r in rows if r["split"] == split}, "evaluation set differs")
    for split in SPLIT_COUNTS:
        require(set(experiments["batch8_only"][split]) == {r["configuration_id"] for r in rows if r["split"] == split and r["a10_matched"]},
                "batch-8 comparison differs")
    require(set(experiments["batch_generalization_train"]) == {r["configuration_id"] for r in rows
            if r["split"] == "train" and r["training"]["microbatch_size"] not in (16, 64)}, "batch holdout leaks")
    panel, pilot = experiments["repeat_panel"], experiments["pilot"]
    require(len(panel) == len(set(panel)) == 512 and set(panel) <= set(by_id), "repeat panel differs")
    require(len(pilot) == len(set(pilot)) == 48, "pilot size differs")
    require({(by_id[k]["modality"], by_id[k]["training"]["precision"], by_id[k]["training"]["microbatch_size"]) for k in pilot} ==
            set(itertools.product(MODALITIES, PRECISIONS, (1, 128))), "pilot coverage differs")
    require(all(by_id[k]["split"] == "train" and by_id[k]["cohort"] == "batch" for k in pilot), "pilot uses nontraining configurations")
    coverage = read_json(output / "coverage.json")
    require(coverage["split_counts"] == SPLIT_COUNTS and coverage["matched_records"] == 992 and
            coverage["planned_records"] == 10240 and coverage["declared_targets_per_record"] == 12, "coverage report differs")
    for name, value in {
        "architecture_group": lambda r: r["group_id"], "modality": lambda r: r["modality"],
        "batch": lambda r: r["training"]["microbatch_size"], "precision": lambda r: r["training"]["precision"],
        "optimizer": lambda r: r["training"]["optimizer"]["name"],
        "accumulation": lambda r: r["training"]["gradient_accumulation_steps"],
    }.items():
        require(coverage["coverage"][name] == dict(Counter(str(value(r)) for r in rows)), f"coverage accounting differs: {name}")
    scope, active_rows = (None, rows) if campaign_profile == "full" else verify_campaign_scope(
        output, rows, experiments, manifest, campaign_profile)
    active_by_id = {row["configuration_id"]: row for row in active_rows}
    active_panel = set(panel if scope is None else scope["repeat_panel"])
    for path in sorted((output / "graphs").glob("*.json.gz")):
        key = path.name.removesuffix(".json.gz")
        require(key in by_id, "unknown captured graph")
        verify_graph(read_json(path), by_id[key])
    attempts, active_attempts, labels = 0, 0, []
    if (output / "execution.json").exists():
        execution = read_json(output / "execution.json")
        require(execution["fingerprint"] == fingerprint({k: v for k, v in execution.items() if k != "fingerprint"}), "execution hash differs")
        require(execution["campaign_fingerprint"] == manifest["fingerprint"] and execution["protocol"] == PROTOCOL and
                tuple(execution["target_names"]) == TARGET_NAMES, "execution contract differs")
        if (output / "source_materials.json.gz").exists():
            require(fingerprint(read_json(output / "source_materials.json.gz")) == execution["material_fingerprint"],
                    "source input material fingerprint differs")
        for path in sorted((output / "attempts").glob("*.json.gz")):
            result = read_json(path)
            require(result["configuration_id"] in by_id and result["repetition"] in (0, 1, 2), "unknown attempt identity")
            require(path.name == f"{result['configuration_id']}.{result['repetition']}.json.gz", "attempt filename identity differs")
            resolve_attempt_execution(output, execution, result.get("execution_fingerprint"))
            require(result["repetition"] == 0 or result["configuration_id"] in panel, "repeat outside frozen panel")
            if result["status"] == "ok":
                verify_attempt(result, by_id[result["configuration_id"]], manifest, execution, output)
                attempts += 1
                if (result["configuration_id"] in active_by_id and
                        (result["repetition"] == 0 or result["configuration_id"] in active_panel)):
                    active_attempts += 1
        if (output / "labels.json.gz").exists():
            labels = read_json(output / "labels.json.gz")
            verify_exported_labels(output, labels, by_id, set(panel), execution)
    report_path = output / "labeling-report.json"
    if report_path.exists():
        report = read_json(report_path)
        require(report["verified_records"] == len(labels) and report["labels_sha256"] == file_sha256(output / "labels.json.gz"),
                "labeling report differs")
        resolve_attempt_execution(output, execution, report["execution_fingerprint"])
        require((report["status"] == "complete") == (len(labels) == 10240), "false campaign completion")
    if scope is not None:
        labels_path = output / f"labels-{campaign_profile}.json.gz"
        scoped_labels = read_json(labels_path) if labels_path.exists() else []
        if labels_path.exists():
            verify_exported_labels(output, scoped_labels, active_by_id, active_panel, execution, scope=scope)
        scoped_report_path = output / f"labeling-report-{campaign_profile}.json"
        if scoped_report_path.exists():
            scoped_report = read_json(scoped_report_path)
            require(scoped_report["verified_records"] == len(scoped_labels) and
                    scoped_report["labels_sha256"] == file_sha256(labels_path), "scoped labeling report differs")
            require(scoped_report["campaign_profile"] == campaign_profile and
                    scoped_report["campaign_scope_fingerprint"] == scope["fingerprint"] and
                    scoped_report["planned_attempts"] == scope["planned_attempts"], "scoped report identity differs")
            resolve_attempt_execution(output, execution, scoped_report["execution_fingerprint"])
            require((scoped_report["status"] == "complete") ==
                    (len(scoped_labels) == scope["planned_records"]), "false scoped campaign completion")
        if require_complete:
            require(len(scoped_labels) == scope["planned_records"] and scoped_report_path.exists(),
                    f"{campaign_profile} campaign is incomplete")
        return dict(status="verified", campaign_profile=campaign_profile,
                    planned_records=scope["planned_records"], planned_attempts=scope["planned_attempts"],
                    source_rows=40020, anchors=248, architecture_groups=119,
                    target_names=list(TARGET_NAMES), split_counts=scope["split_counts"],
                    matched_records=scope["matched_records"], verified_attempts=active_attempts,
                    verified_labels=len(scoped_labels), campaign_fingerprint=manifest["fingerprint"],
                    campaign_scope_fingerprint=scope["fingerprint"], gpu_execution_performed=False)
    if require_complete:
        require(len(labels) == 10240 and report_path.exists(), "campaign is incomplete")
    return dict(status="verified", campaign_profile="full", planned_records=10240, source_rows=40020,
                anchors=248, architecture_groups=119, target_names=list(TARGET_NAMES), split_counts=SPLIT_COUNTS,
                matched_records=992, verified_attempts=attempts, verified_labels=len(labels),
                campaign_fingerprint=manifest["fingerprint"], gpu_execution_performed=False)
