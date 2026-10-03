"""Prepare and explicitly launch the twelve-target local RTX 5090 campaign.

prepare, prepare-data and verify never initialize CUDA. prepare-data downloads
the original A10 sources to record/perfseer-v32/transfer-source-data by default.
The recovered input adapter generates tensors from real subset keys and file
fingerprints; it is not a conventional decoded-example dataloader.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import zipfile

import torch
from torch import nn

from perfseer_v31 import dataset as compiler
from perfseer_v31.capture_training import capture, graph_from_design
from perfseer_v31.generated_model_runtime import GraphModel
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v3.hardware import HardwareProfileV3
from .dataset import DATA
from .transfer_sources import DEFAULT_SOURCE_ROOT


TARGET_NAMES = (
    "train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
    "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib",
    "train_peak_torch_allocated_mib", "infer_step_wall_ms", "infer_step_gpu_ms",
    "infer_avg_sm_util_percent", "infer_avg_vram_mib", "infer_peak_vram_mib",
)
VERSION = "perfseer_v32_transfer_12_targets_v1"
HARDWARE_ID = "nvidia_geforce_rtx_5090_32gb_local"
SEED = 42
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
PRECISIONS = ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp")
MODALITIES = ("audio", "graph", "image", "tabular", "text", "time_series")
SPLIT_COUNTS = {"train": 7456, "validation": 1408, "test": 1376}
ANCHOR_COUNTS = {"train": 185, "validation": 32, "test": 31}
SECONDARY_COUNTS = {"train": 32, "validation": 8, "test": 8}
BUDGETS = (512, 1024, 2048, 4096, 7456)
CAMPAIGN_PROFILES = ("full", "24h")
REDUCED_BATCHES = (1, 8, 128)
REDUCED_OPTIMIZER_BATCHES = (8, 128)
REDUCED_ACCUMULATION_CELLS = ((16, 4), (16, 8))
REDUCED_BUDGETS = (512, 1024, 2048, 2988)
REDUCED_REPEAT_PANEL_SIZE = 64
REDUCED_PLANNED_RECORDS = 4128
REDUCED_PLANNED_ATTEMPTS = 4256
REDUCED_ATTEMPTS_PER_HOUR = 185
PROTOCOL = {
    "version": "perfseer_v32_transfer_train_infer_protocol_v1",
    "phase_order": ["infer", "train"], "infer_warmup_steps": 10,
    "infer_min_steps": 30, "train_warmup_epochs": 1, "train_min_epochs": 2,
    "min_phase_seconds": 5., "sample_interval_seconds": .01,
    "max_phase_seconds": 1800., "worker_timeout_seconds": 7200.,
    "repeat_panel_size": 512, "repeat_count": 3,
    "loss": "mse_to_zero", "scheduler": "none",
    "training_step_unit": "logical_optimizer_step",
    "inference_step_unit": "forward_microbatch",
    "input_adapter": "local_subset_key_tensor",
    "source_weights_seed_available": False,
    "warmup_excluded": True, "sm_metric": "nvml_gpu_utilization_percent",
}
COMPATIBLE_EXECUTION_CODE_MIGRATIONS = {
    "fe14fced2ee06b4b20a59b957537b177748991bc79387214ca771eb62c5a6fb5":
        "canonical_graph_comparison_and_wsl_owner_registration_v1",
    "e502af346ee6b1c341b03a8058e079d1bf1049734a1e3b0f039c2637f6d108ef":
        "wsl_missing_owner_fallback_v1",
    "5fd9f9294e7a7a219206baeac7a41905f1902ed75b7d5f7c999a97c415556269":
        "bf16_capture_gradient_tolerance_v1",
    "f0eca5244d8089a260d9f19fa20f2e29619f6d1bdc23c2e13f4872822ce4a020":
        "24_hour_scope_pause_and_launcher_bootstrap_v1",
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "record/perfseer-v32/transfer-rtx5090"
LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "scripts/run_perfseer_v32_transfer_labeling.py"


def optimizer_settings(name):
    if name not in ("adam", "adamw", "sgd"):
        raise ValueError(f"unsupported optimizer: {name}")
    settings = dict(name=name, learning_rate=.001, weight_decay=.01 if name == "adamw" else 0.,
                    foreach=None, fused=None, maximize=False, differentiable=False)
    if name == "sgd":
        settings.update(momentum=0., dampening=0., nesterov=False)
    else:
        settings.update(betas=[.9, .999], eps=1e-8, amsgrad=False, capturable=False)
    return settings


def canonical_workload(model, dataset):
    return {
        "nodes": compiler._native_architecture(model["NODE_SPECS"]),
        "inputs": [{**s, "shape": [1, *s["shape"][1:]]} for s in model["INPUT_SPECS"]],
        "dataset_id": dataset["dataset_id"], "subset_id": dataset["subset_id"],
        "num_samples": int(dataset["num_samples"]),
    }


def source_anchors(data):
    """Reconcile every raw row with its prepared split and executable source."""
    data = Path(data)
    base = read_json(data / "ready_for_train/dataset_manifest.json")
    if base["fingerprint"] != fingerprint({k: v for k, v in base.items() if k != "fingerprint"}):
        raise ValueError("A10 manifest fingerprint differs")
    source = read_json(data / "source_manifest.json")
    if file_sha256(data / "source_manifest.json") != base["source_manifest_sha256"]:
        raise ValueError("A10 source manifest differs")
    if file_sha256(data / "merged_samples.json.gz") != source["merged_sha256"]:
        raise ValueError("merged source hash differs")
    if file_sha256(compiler.PACKAGE / "generated_model_runtime.py") != compiler.CALIBRATION_RUNTIME_SHA256:
        raise ValueError("pinned generated runtime differs")
    prepared = {}
    for split, meta in base["split_files"].items():
        path = data / "ready_for_train" / meta["path"]
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("A10 split hash differs")
        values = read_json(path)
        if len(values) != meta["rows"]:
            raise ValueError("A10 split count differs")
        for row in values:
            if row["sample_id"] in prepared or row["split"] != split:
                raise ValueError("A10 duplicate/split mismatch")
            prepared[row["sample_id"]] = row
    anchors, aliases, models = {}, [], {}
    for raw in read_json(data / "merged_samples.json.gz"):
        row = prepared.pop(raw["sample_id"])
        native = raw["native_label"]
        if tuple(native["target_names"]) != TARGET_NAMES or native["status"] != "ok" or native["hardware_id"] != "a10":
            raise ValueError("source is not a complete twelve-target A10 record")
        measured, dataset = native["training"], native["dataset"]
        if (measured["batch_size"], measured["grad_accumulation_steps"], measured["optimizer"]) != (8, 1, "adam"):
            raise ValueError("source training settings changed; redesign the campaign")
        path = data / raw["model_source_path"]
        if path not in models:
            if file_sha256(path) != raw["model_source_sha256"]:
                raise ValueError(f"source model hash differs: {path}")
            models[path] = compiler._constants(path)
        model = models[path]
        expected_inputs = [{**s, "shape": [8, *s["shape"][1:]]} for s in model["INPUT_SPECS"]]
        if expected_inputs != dataset["input_specs"]:
            raise ValueError(f"measured/source input mismatch: {raw['sample_id']}")
        identity = canonical_workload(model, dataset)
        key = fingerprint(identity)
        if key not in anchors:
            anchors[key] = dict(anchor_id=key, identity=identity, model=model, split=row["split"],
                                group_id=row["group_id"], modality=dataset["modality"],
                                operation_types=row["provenance"]["operation_types"],
                                source_ids={precision: [] for precision in PRECISIONS})
        anchor = anchors[key]
        if anchor["split"] != row["split"] or anchor["group_id"] != row["group_id"]:
            raise ValueError("canonical workload crosses architecture splits")
        anchor["source_ids"][measured["precision"]].append(raw["sample_id"])
        aliases.append({**raw, "anchor_id": key, "split": row["split"], "group_id": row["group_id"],
                        "input_path": row["input_path"], "input_sha256": row["input_sha256"]})
    if prepared or len(aliases) != 40020:
        raise ValueError("incomplete source alias coverage")
    values = sorted(anchors.values(), key=lambda a: a["anchor_id"])
    if dict(Counter(a["split"] for a in values)) != ANCHOR_COUNTS:
        raise ValueError("canonical anchor counts changed; redesign the campaign")
    for anchor in values:
        if any(not ids for ids in anchor["source_ids"].values()):
            raise ValueError("anchor lacks an A10 precision")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = GraphModel(anchor["model"]["NODE_SPECS"])
        anchor["structure"] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "depth": len(anchor["model"]["NODE_SPECS"]),
            "width": max(max(n.get("memory_info", {}).get("output_features", 0),
                             n.get("memory_info", {}).get("output_channels", 0)) for n in anchor["model"]["NODE_SPECS"]),
            "input_numel": sum(math.prod(s["shape"][1:]) for s in anchor["identity"]["inputs"]),
        }
    return base, values, aliases


def select_secondary(anchors):
    training = [a for a in anchors if a["split"] == "train"]
    cuts = {key: statistics.quantiles([a["structure"][key] for a in training], n=4, method="inclusive")
            for key in ("parameters", "depth", "width", "input_numel")}
    tokens = {}
    for a in anchors:
        tokens[a["anchor_id"]] = {f"modality:{a['modality']}", f"group:{a['group_id']}",
                                  *(f"op:{op}" for op in a["operation_types"]),
                                  *(f"{key}:{sum(value > c for c in cuts[key])}" for key, value in a["structure"].items())}
    selected = []
    for split, count in SECONDARY_COUNTS.items():
        pool = [a for a in anchors if a["split"] == split]
        frequencies = Counter(t for a in pool for t in tokens[a["anchor_id"]])
        covered, modalities, chosen = set(), set(), []
        while len(chosen) < count:
            missing = set(MODALITIES) - modalities
            eligible = [a for a in pool if a["anchor_id"] not in chosen and (not missing or a["modality"] in missing)]
            best = min(eligible, key=lambda a: (
                -sum(1 / frequencies[t] for t in tokens[a["anchor_id"]] - covered),
                fingerprint([SEED, a["anchor_id"]])))
            chosen.append(best["anchor_id"])
            covered.update(tokens[best["anchor_id"]])
            modalities.add(best["modality"])
        selected.extend(chosen)
    return selected


def configuration(anchor, batch, precision, optimizer="adam", accumulation=1, cohort="batch"):
    training = {
        "version": "perfseer_v31_training_config_v1", "microbatch_size": batch,
        "gradient_accumulation_steps": accumulation, "precision": precision,
        "optimizer": optimizer_settings(optimizer), "scheduler": {"name": "none"},
        "backend": "cuda_eager", "loss": "mse_to_zero",
        "steps_per_epoch": math.ceil(anchor["identity"]["num_samples"] / (batch * accumulation)),
        "total_epochs": 3, "unavailable_settings": [],
    }
    key = fingerprint({"anchor_id": anchor["anchor_id"], "training": training, "hardware_id": HARDWARE_ID})
    matched = batch == 8 and accumulation == 1 and optimizer == "adam"
    return dict(configuration_id=key, anchor_id=anchor["anchor_id"], split=anchor["split"],
                group_id=anchor["group_id"], modality=anchor["modality"], cohort=cohort,
                training=training, effective_batch_size=batch * accumulation, a10_matched=matched,
                paired_source_ids=anchor["source_ids"][precision] if matched else [])


def balanced_order(rows, *, salt):
    buckets = defaultdict(list)
    for row in rows:
        t = row["training"]
        buckets[(row["modality"], t["microbatch_size"], t["precision"], row["cohort"])].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: fingerprint([SEED, salt, r["group_id"], r["configuration_id"]]))
    keys = sorted(buckets, key=lambda k: fingerprint([SEED, salt, k]))
    result = []
    for index in range(max(map(len, buckets.values()))):
        result.extend(buckets[k][index]["configuration_id"] for k in keys if index < len(buckets[k]))
    return result


def build_campaign(anchors):
    secondary = select_secondary(anchors)
    rows = []
    for a in anchors:
        for precision in PRECISIONS:
            rows.extend(configuration(a, batch, precision) for batch in BATCHES)
            if a["anchor_id"] in secondary:
                rows.extend(configuration(a, batch, precision, optimizer=opt, cohort="optimizer")
                            for opt in ("adamw", "sgd") for batch in (4, 8, 32, 128))
                rows.extend(configuration(a, effective // accumulation, precision, accumulation=accumulation,
                                          cohort="accumulation")
                            for effective in (64, 128) for accumulation in (4, 8))
    rows.sort(key=lambda r: r["configuration_id"])
    train = [r for r in rows if r["split"] == "train"]
    order = balanced_order(train, salt="budgets")
    pilot_anchors = {m: min((a for a in anchors if a["split"] == "train" and a["modality"] == m),
                           key=lambda a: fingerprint([SEED, "pilot", a["anchor_id"]]))["anchor_id"]
                     for m in MODALITIES}
    experiments = {
        "secondary_anchors": secondary,
        "nested_training_budgets": {str(b): order[:b] for b in BUDGETS},
        "validation": [r["configuration_id"] for r in rows if r["split"] == "validation"],
        "test": [r["configuration_id"] for r in rows if r["split"] == "test"],
        "batch8_only": {s: [r["configuration_id"] for r in rows if r["split"] == s and r["a10_matched"]]
                        for s in SPLIT_COUNTS},
        "batch_generalization_train": [r["configuration_id"] for r in train
                                       if r["training"]["microbatch_size"] not in (16, 64)],
        "repeat_panel": balanced_order(rows, salt="repeat")[:512],
        "pilot": [r["configuration_id"] for r in rows if r["cohort"] == "batch"
                  and r["anchor_id"] == pilot_anchors[r["modality"]] and r["training"]["microbatch_size"] in (1, 128)],
    }
    return rows, experiments


def build_campaign_scope(rows, experiments, campaign_fingerprint, profile):
    if profile != "24h":
        raise ValueError(f"unsupported campaign profile: {profile}")
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
    repeat_candidates = [row for row in selected if row["configuration_id"] in set(experiments["repeat_panel"])]
    repeat_panel = balanced_order(repeat_candidates, salt="24h-repeat")[:REDUCED_REPEAT_PANEL_SIZE]
    train = [row for row in selected if row["split"] == "train"]
    order = balanced_order(train, salt="24h-budgets")
    scope = {
        "version": "perfseer_v32_transfer_campaign_scope_v1",
        "profile": profile,
        "campaign_fingerprint": campaign_fingerprint,
        "selection": {
            "batch_sweep": list(REDUCED_BATCHES),
            "optimizer_batches": list(REDUCED_OPTIMIZER_BATCHES),
            "accumulation_cells": [list(cell) for cell in REDUCED_ACCUMULATION_CELLS],
        },
        "configuration_ids": selected_ids,
        "planned_records": len(selected),
        "planned_attempts": len(selected) + 2 * len(repeat_panel),
        "repeat_panel": repeat_panel,
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
    if (scope["planned_records"], scope["planned_attempts"]) != (REDUCED_PLANNED_RECORDS, REDUCED_PLANNED_ATTEMPTS):
        raise ValueError("24h campaign size differs")
    scope["fingerprint"] = fingerprint(scope)
    return scope


def campaign_scope_path(output, profile):
    return Path(output) / f"campaign-scope-{profile}.json.gz"


def prepare_campaign_scope(output, rows, experiments, campaign_fingerprint, profile):
    scope = build_campaign_scope(rows, experiments, campaign_fingerprint, profile)
    path = campaign_scope_path(output, profile)
    if path.exists():
        if read_json(path) != scope:
            raise ValueError(f"{profile} campaign scope differs")
    else:
        atomic_write(path, scope, compress=True)
    return scope


def coverage_report(anchors, rows):
    return {
        "planned_records": len(rows), "declared_targets_per_record": len(TARGET_NAMES),
        "source_anchors": len(anchors), "architecture_groups": len({a["group_id"] for a in anchors}),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "anchor_split_counts": dict(Counter(a["split"] for a in anchors)),
        "cohort_counts": dict(Counter(r["cohort"] for r in rows)),
        "matched_records": sum(r["a10_matched"] for r in rows),
        "coverage": {key: dict(Counter(str(value(r)) for r in rows)) for key, value in {
            "architecture_group": lambda r: r["group_id"], "modality": lambda r: r["modality"],
            "batch": lambda r: r["training"]["microbatch_size"], "precision": lambda r: r["training"]["precision"],
            "optimizer": lambda r: r["training"]["optimizer"]["name"],
            "accumulation": lambda r: r["training"]["gradient_accumulation_steps"],
        }.items()},
        "operator_types": sorted({op for a in anchors for op in a["operation_types"]}),
        "structural_sampling": "all canonical source anchors; aliases are not independent architectures",
        "source_protocol_comparison": "first two measured training epochs retained separately",
        "prediction_error_report": "not available until a predictor is evaluated; no accuracy claim",
    }


def prepare(output, data=DATA, *, campaign_profile="full"):
    output = Path(output)
    if (output / "manifest.json").exists():
        from .transfer_verification import verify
        if campaign_profile != "full":
            manifest = read_json(output / "manifest.json")
            prepare_campaign_scope(output, read_json(output / "configurations.json.gz"),
                                   read_json(output / "experiments.json.gz"), manifest["fingerprint"], campaign_profile)
        result = verify(output, data=data, campaign_profile=campaign_profile)
        name = "preparation-verification.json" if campaign_profile == "full" else f"preparation-verification-{campaign_profile}.json"
        atomic_write(output / name, result)
        return result
    torch.set_num_threads(1)
    base, anchors, aliases = source_anchors(data)
    rows, experiments = build_campaign(anchors)
    coverage = coverage_report(anchors, rows)
    files = {}
    for name, value in (("anchors.json.gz", anchors), ("source_aliases.json.gz", aliases),
                        ("configurations.json.gz", rows), ("experiments.json.gz", experiments),
                        ("coverage.json", coverage)):
        atomic_write(output / name, value, compress=name.endswith(".gz"))
        files[name] = file_sha256(output / name)
    manifest = dict(version=VERSION, seed=SEED, hardware_id=HARDWARE_ID, target_names=list(TARGET_NAMES),
                    scheduler_label_version=4, auxiliary_targets=["train_epoch_ms_step_extrapolated"],
                    source_dataset_fingerprint=base["fingerprint"], runtime_sha256=compiler.CALIBRATION_RUNTIME_SHA256,
                    protocol=PROTOCOL, files=files, split_counts=SPLIT_COUNTS, planned_records=10240)
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "manifest.json", manifest)
    if campaign_profile != "full":
        prepare_campaign_scope(output, rows, experiments, manifest["fingerprint"], campaign_profile)
    from .transfer_verification import verify
    result = verify(output, data=data, campaign_profile=campaign_profile)
    name = "preparation-verification.json" if campaign_profile == "full" else f"preparation-verification-{campaign_profile}.json"
    atomic_write(output / name, result)
    return result


def _source_path(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("source sample path escapes its raw dataset")
    return path


def sample_bytes(raw_dir, key):
    """Byte-for-byte adapter convention recovered from run_profile.py."""
    if key.startswith(("node:", "csv_row_")):
        return key.encode("utf-8")
    if "::" not in key:
        with _source_path(raw_dir, key).open("rb") as stream:
            return stream.read(4096)
    archive, *parts = key.split("::")
    bundle = zipfile.ZipFile(_source_path(raw_dir, archive))
    try:
        for index, part in enumerate(parts):
            member, marker, row_index = part.rpartition(":row:")
            if not marker:
                member = part
            if marker:
                with bundle.open(member) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", errors="ignore", newline="") as stream:
                        reader = csv.reader(stream)
                        next(reader, None)
                        for n, row in enumerate(reader):
                            if n == int(row_index):
                                return "\x1f".join(row).encode("utf-8", errors="ignore")[:4096]
                raise ValueError(f"CSV sample row absent: {key}")
            if index == len(parts) - 1:
                with bundle.open(member) as stream:
                    return stream.read(4096)
            payload = bundle.read(member)
            bundle.close()
            bundle = zipfile.ZipFile(io.BytesIO(payload))
    finally:
        bundle.close()
    raise ValueError(f"invalid sample key: {key}")


def source_materials(root, anchors):
    root = Path(root)
    result = {}
    for anchor in anchors:
        identity = anchor["identity"]
        key = identity["dataset_id"] + "::" + identity["subset_id"]
        if key in result:
            continue
        raw = root / "raw" / identity["dataset_id"]
        mask = root / "prepared" / identity["dataset_id"] / "subset_masks" / f"{identity['subset_id']}.json"
        if not raw.is_dir() or not mask.is_file():
            raise FileNotFoundError(f"original source data required: {raw} and {mask}")
        keys = read_json(mask).get("sample_keys", [])
        if len(keys) != identity["num_samples"] or not all(isinstance(k, str) and k for k in keys):
            raise ValueError(f"subset sample keys/count differ: {mask}")
        probes = []
        for sample in keys[:8]:
            payload = sample_bytes(raw, sample)
            probes.append(dict(key=sample, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
        seed_payload = dict(dataset_id=identity["dataset_id"], subset_id=identity["subset_id"],
                            sample_keys=keys[:64], fingerprints=probes)
        # The original profiler uses json.dumps' default separators.
        digest = hashlib.sha256(json.dumps(seed_payload, sort_keys=True).encode("utf-8")).hexdigest()
        result[key] = dict(seed=int(digest[:16], 16) % (2**63 - 1), sample_keys=keys,
                           mask_sha256=file_sha256(mask), sample_fingerprints=probes)
    return result


def make_inputs(anchor, batch, material, offset, device="cpu"):
    keys = material["sample_keys"]
    batch_keys = [keys[(offset * batch + i) % len(keys)] for i in range(min(batch, len(keys)))]
    payload = dict(seed=material["seed"], batch_offset=offset, batch_keys=batch_keys)
    seed = int(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
    inputs = []
    for index, spec in enumerate(anchor["identity"]["inputs"]):
        shape = (batch, *spec["shape"][1:])
        generator = torch.Generator(device="cpu").manual_seed((seed + index * 9973) % (2**63 - 1))
        if spec["dtype"] in ("int64", "long") or spec.get("kind") in ("tokens", "token_ids"):
            tensor = torch.randint(0, 32000, shape, dtype=torch.long, generator=generator)
        elif spec.get("kind") == "adjacency":
            tensor = torch.eye(shape[-1], dtype=torch.float32).expand(shape).clone()
        else:
            tensor = torch.randn(shape, dtype=torch.float32, generator=generator)
        inputs.append(tensor.to(device))
    return tuple(inputs)


def make_optimizer(model, settings):
    values = {k: v for k, v in settings.items() if k not in ("name", "learning_rate")}
    values["lr"] = settings["learning_rate"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    return {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[settings["name"]](parameters, **values)


def autocast(precision, device):
    dtype = {"bf16_amp": torch.bfloat16, "fp16_amp": torch.float16}.get(precision)
    return torch.autocast(torch.device(device).type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def precision_controls(precision):
    if precision not in PRECISIONS:
        raise ValueError("unsupported precision")
    value = "tf32" if precision == "tf32" else "ieee"
    controls = {}
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        for name, obj in (("global", torch.backends), ("matmul", torch.backends.cuda.matmul),
                          ("cudnn", torch.backends.cudnn), ("conv", torch.backends.cudnn.conv),
                          ("rnn", torch.backends.cudnn.rnn)):
            obj.fp32_precision = value
            controls[name] = obj.fp32_precision
            if controls[name] != value:
                raise ValueError("precision backend did not apply requested policy")
    else:
        enabled = precision == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.set_float32_matmul_precision("high" if enabled else "highest")
        controls = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
    return controls


def training_step(model, optimizer, batches, precision, *, scaler=None):
    """One logical optimizer update; input iterator remains inside timing."""
    optimizer.zero_grad(set_to_none=True)
    count = len(batches)
    losses = []
    for inputs in batches:
        with autocast(precision, inputs[0].device):
            output = model(*inputs)
            loss = torch.nn.functional.mse_loss(output.float(), torch.zeros_like(output, dtype=torch.float32)) / count
        (scaler.scale(loss) if scaler is not None else loss).backward()
        losses.append(loss.detach())
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return torch.stack(losses).sum()


class InputBatches:
    def __init__(self, factory, count):
        self.factory, self.count = factory, count

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(self.count):
            yield self.factory()


class CaptureWorkload(nn.Module):
    """Expose the measured accumulation-scaled loss to the shared compiler."""
    def __init__(self, model, accumulation):
        super().__init__()
        self.model = model
        self.accumulation = accumulation

    def forward(self, inputs):
        return self.model(*inputs)

    def compute_loss(self, output, batch, adapter):
        return torch.nn.functional.mse_loss(output.float(), torch.zeros_like(output, dtype=torch.float32)) / self.accumulation


def capture_workload(anchor, row, material, hardware=None):
    torch.manual_seed(SEED)
    model = GraphModel(anchor["model"]["NODE_SPECS"])
    inputs = make_inputs(anchor, row["training"]["microbatch_size"], material, 0)
    wrapper = CaptureWorkload(model, row["training"]["gradient_accumulation_steps"])
    design, audit = capture(wrapper, inputs, None, True, row["training"], architecture_key=anchor["anchor_id"])
    graph = graph_from_design(design)
    metadata = {**graph.metadata, "target_hardware_id": HARDWARE_ID,
                "configuration_id": row["configuration_id"], "source_group_id": anchor["group_id"],
                "capture_scope": "strict_microstep_forward_loss_backward_plus_optimizer_summary",
                "profiling_repetitions_are_not_workload_epochs": True}
    metadata.pop("hardware_features", None)
    if hardware:
        profile = HardwareProfileV3(hardware_id=HARDWARE_ID, static=hardware["static"],
                                    microbenchmarks={}, environment=hardware["environment"])
        profile.validate()
        metadata["hardware_profile"] = profile.canonical_payload
        metadata["hardware_profile_sha256"] = profile.sha256
    graph = replace(graph, metadata=metadata)
    graph.validate()
    payload = graph.to_dict()
    design.update(graph={k: v for k, v in payload.items() if k not in ("nodes", "tensor_edges", "input_signature", "training_config")},
                  NODE_SPECS=payload["nodes"], EDGE_SPECS=payload["tensor_edges"],
                  INPUT_SPECS=payload["input_signature"], TRAINING_CONFIG=payload["training_config"])
    return design, audit


def canonical_equal(first, second):
    """Compare values by their persisted JSON representation."""
    return fingerprint(first) == fingerprint(second)


class ContentionError(RuntimeError):
    pass


class TelemetryError(RuntimeError):
    pass


def check_gpu(observed, allowed_pids=()):
    if observed["name"] != "NVIDIA GeForce RTX 5090" or not 30 * 1024**3 <= observed["memory_total_bytes"] <= 34 * 1024**3:
        raise ValueError("labeling requires the local 32 GiB RTX 5090")
    if set(observed["compute_pids"]) - set(allowed_pids):
        raise ContentionError(f"competing GPU compute processes: {observed['compute_pids']}")
    for field in ("sm_percent", "memory_used_mib"):
        value = observed[field]
        if not math.isfinite(value) or value < 0 or (field == "sm_percent" and value > 100):
            raise TelemetryError(f"invalid GPU telemetry: {field}")


def wait_for_exclusive_owner(nvml, expected_pid, timeout_seconds=10., poll_seconds=.05):
    """Wait for WSL NVML to publish this worker, rejecting any foreign owner."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        observed = nvml.read()
        check_gpu(observed, [expected_pid])
        if expected_pid in observed["compute_pids"]:
            return observed
        if time.monotonic() >= deadline:
            return observed
        time.sleep(poll_seconds)


class Nvml:
    def __init__(self, uuid=None):
        import pynvml
        pynvml.nvmlInit()
        self.api = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(0)

    def read(self):
        api, handle = self.api, self.handle
        memory = api.nvmlDeviceGetMemoryInfo(handle)
        name, uuid = api.nvmlDeviceGetName(handle), api.nvmlDeviceGetUUID(handle)
        return dict(name=name.decode() if isinstance(name, bytes) else name,
                    uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                    memory_total_bytes=int(memory.total), memory_used_mib=memory.used / 1024**2,
                    sm_percent=float(api.nvmlDeviceGetUtilizationRates(handle).gpu),
                    compute_pids=sorted(p.pid for p in api.nvmlDeviceGetComputeRunningProcesses(handle)))


class Sampler:
    def __init__(self, nvml, allowed_pids, origin):
        self.nvml, self.allowed_pids, self.origin = nvml, allowed_pids, origin
        self.samples, self.error = [], None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                observed = self.nvml.read()
                check_gpu(observed, self.allowed_pids)
                self.samples.append(dict(t_ms=(time.perf_counter() - self.origin) * 1000,
                                         sm_percent=observed["sm_percent"], vram_mib=observed["memory_used_mib"]))
            except Exception as error:
                self.error = error
                return
            self.stop_event.wait(PROTOCOL["sample_interval_seconds"])

    def check(self):
        if self.error:
            raise self.error


def measure_phase(name, fn, nvml, allowed_pids, *, steps_per_epoch=1, device="cuda", evidence=None):
    """Synchronized GPU blocks, separate CPU wall clocks, isolated telemetry."""
    warmup = PROTOCOL["infer_warmup_steps"] if name == "infer" else steps_per_epoch
    minimum = PROTOCOL["infer_min_steps"] if name == "infer" else PROTOCOL["train_min_epochs"]
    block_steps = 1 if name == "infer" else steps_per_epoch
    check_gpu(nvml.read(), allowed_pids)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    origin = time.perf_counter()
    sampler = Sampler(nvml, allowed_pids, origin)
    blocks = []
    result = evidence if evidence is not None else {}
    result.update(phase=name, warmup_steps=warmup, blocks=blocks, samples=sampler.samples, status="running")
    sampler.thread.start()
    try:
        while len(blocks) < minimum or sum(b["wall_ms"] for b in blocks) < 1000 * PROTOCOL["min_phase_seconds"]:
            sampler.check()
            if time.perf_counter() - origin > PROTOCOL["max_phase_seconds"]:
                raise TelemetryError("measurement phase exceeded its time limit")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(block_steps):
                value = fn()
            end.record()
            torch.cuda.synchronize()
            wall_end = time.perf_counter()
            blocks.append(dict(start_ms=(wall_start - origin) * 1000, end_ms=(wall_end - origin) * 1000,
                               wall_ms=(wall_end - wall_start) * 1000, gpu_ms=float(start.elapsed_time(end)),
                               steps=block_steps, peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2))
            sampler.check()
        result["status"] = "ok"
    finally:
        sampler.stop_event.set()
        sampler.thread.join(timeout=5)
        result["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        if sampler.thread.is_alive():
            raise TelemetryError("telemetry sampler did not stop")
    sampler.check()
    if not sampler.samples:
        raise TelemetryError("no NVML samples")
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("nonfinite profiling output/loss")
    return result


def target_values(phases, *, first_two=False):
    train, infer = phases["train"], phases["infer"]
    blocks = train["blocks"][:2] if first_two else train["blocks"]
    boundary = blocks[-1]["end_ms"]
    ts = [s for s in train["samples"] if not first_two or s["t_ms"] <= boundary]
    ins = infer["samples"]
    if not ts or not ins:
        raise TelemetryError("missing phase telemetry")
    values = {
        "train_step_wall_ms": sum(b["wall_ms"] for b in blocks) / sum(b["steps"] for b in blocks),
        "train_step_gpu_ms": sum(b["gpu_ms"] for b in blocks) / sum(b["steps"] for b in blocks),
        "train_epoch_ms": statistics.mean(b["wall_ms"] for b in blocks),
        "train_avg_sm_util_percent": statistics.mean(s["sm_percent"] for s in ts),
        "train_avg_vram_mib": statistics.mean(s["vram_mib"] for s in ts),
        "train_peak_vram_mib": max(s["vram_mib"] for s in ts),
        "train_peak_torch_allocated_mib": blocks[-1]["peak_torch_allocated_mib"] if first_two else train["peak_torch_allocated_mib"],
        "infer_step_wall_ms": sum(b["wall_ms"] for b in infer["blocks"]) / sum(b["steps"] for b in infer["blocks"]),
        "infer_step_gpu_ms": sum(b["gpu_ms"] for b in infer["blocks"]) / sum(b["steps"] for b in infer["blocks"]),
        "infer_avg_sm_util_percent": statistics.mean(s["sm_percent"] for s in ins),
        "infer_avg_vram_mib": statistics.mean(s["vram_mib"] for s in ins),
        "infer_peak_vram_mib": max(s["vram_mib"] for s in ins),
    }
    values["train_epoch_ms_step_extrapolated"] = values["train_step_wall_ms"] * blocks[0]["steps"]
    return values


def execution_code():
    paths = [Path(__file__), Path(__file__).with_name("transfer_verification.py"),
             Path(__file__).with_name("transfer_sources.py")]
    for package in (compiler.PACKAGE, Path(__file__).resolve().parents[1] / "perfseer_v3"):
        paths.extend(package.glob("*.py"))
    return fingerprint({str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p) for p in sorted(paths)})


def execution_identity(manifest, materials, observed, nvml):
    driver = nvml.api.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode()
    return dict(campaign_fingerprint=manifest["fingerprint"], version=VERSION, target_names=list(TARGET_NAMES),
                protocol=PROTOCOL, code_fingerprint=execution_code(), material_fingerprint=fingerprint(materials),
                gpu_uuid=observed["uuid"], gpu_name=observed["name"], memory_total_bytes=observed["memory_total_bytes"],
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), driver=driver, platform=platform.platform(),
                cpu_threads=1, omp_threads="1", mkl_threads="1")


def bind_execution(output, identity, *, resume):
    output, path = Path(output), Path(output) / "execution.json"
    payload = {**identity, "fingerprint": fingerprint(identity)}
    if path.exists():
        if not resume:
            raise ValueError("output already has an execution; use --resume")
        previous = read_json(path)
        if previous == payload:
            return previous
        prior_identity = {k: v for k, v in previous.items()
                          if k not in ("fingerprint", "compatible_predecessor_execution_fingerprints", "execution_migration")}
        if prior_identity == identity:
            return previous
        comparable = lambda value: {k: v for k, v in value.items()
                                    if k not in ("fingerprint", "code_fingerprint",
                                                 "compatible_predecessor_execution_fingerprints", "execution_migration")}
        migration = COMPATIBLE_EXECUTION_CODE_MIGRATIONS.get(previous.get("code_fingerprint"))
        if migration is None or comparable(previous) != comparable(identity):
            raise ValueError("resume execution fingerprint differs")
        history = output / "execution_history"
        history.mkdir(exist_ok=True)
        archived = history / f"{previous['fingerprint']}.json"
        if archived.exists() and read_json(archived) != previous:
            raise ValueError("execution history differs")
        atomic_write(archived, previous)
        predecessors = [previous["fingerprint"], *previous.get("compatible_predecessor_execution_fingerprints", [])]
        migrated = {**identity, "compatible_predecessor_execution_fingerprints": predecessors,
                    "execution_migration": migration}
        payload = {**migrated, "fingerprint": fingerprint(migrated)}
        atomic_write(output / "execution-migration.json",
                     dict(version=1, migration=migration, predecessor=previous["fingerprint"], successor=payload["fingerprint"]))
        atomic_write(path, payload)
    else:
        atomic_write(path, payload)
    return payload


def resolve_attempt_execution(output, execution, execution_fingerprint):
    """Resolve immutable per-attempt provenance across an approved code repair."""
    if execution_fingerprint == execution["fingerprint"]:
        return execution
    if execution_fingerprint not in execution.get("compatible_predecessor_execution_fingerprints", []):
        raise ValueError("attempt execution fingerprint differs")
    path = Path(output) / "execution_history" / f"{execution_fingerprint}.json"
    historical = read_json(path)
    if historical.get("fingerprint") != fingerprint({k: v for k, v in historical.items() if k != "fingerprint"}):
        raise ValueError("historical execution fingerprint differs")
    return historical


def profile_phases(model, anchor, row, material, nvml, allowed, phases, *, device="cuda", measure=None, checkpoint=None):
    measure = measure or measure_phase
    checkpoint = checkpoint or (lambda: None)
    batch, accumulation = row["training"]["microbatch_size"], row["training"]["gradient_accumulation_steps"]
    precision, offset = row["training"]["precision"], 0

    def next_inputs():
        nonlocal offset
        inputs = make_inputs(anchor, batch, material, offset, device)
        offset += 1
        return inputs

    model.eval()

    def infer():
        with torch.no_grad(), autocast(precision, device):
            return model(*next_inputs())

    phases["infer"] = {}
    measure("infer", infer, nvml, allowed, evidence=phases["infer"])
    checkpoint()
    offset = 0
    model.train()
    optimizer = make_optimizer(model, row["training"]["optimizer"])
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16_amp" and str(device).startswith("cuda") else None

    def train():
        return training_step(model, optimizer, InputBatches(next_inputs, accumulation), precision, scaler=scaler)

    phases["train"] = {}
    measure("train", train, nvml, allowed, steps_per_epoch=row["training"]["steps_per_epoch"], evidence=phases["train"])
    checkpoint()
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("nonfinite model parameters after training")
    return dict(resolved_optimizer_defaults=optimizer.defaults,
                grad_scaler_final_scale=scaler.get_scale() if scaler is not None else None)


def event(output, **payload):
    value = {"timestamp": time.time(), **payload}
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    with (Path(output) / "progress.jsonl").open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def worker_command(output, configuration_id, repetition):
    return [sys.executable, str(LAUNCHER_PATH), "_worker", "--output", str(output),
            "--configuration-id", configuration_id, "--repetition", str(repetition)]


def worker(output, configuration_id, repetition):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    execution = read_json(output / "execution.json")
    row = next(r for r in read_json(output / "configurations.json.gz") if r["configuration_id"] == configuration_id)
    anchor = next(a for a in read_json(output / "anchors.json.gz") if a["anchor_id"] == row["anchor_id"])
    materials = read_json(output / "source_materials.json.gz")
    material = materials[anchor["identity"]["dataset_id"] + "::" + anchor["identity"]["subset_id"]]
    name = f"{configuration_id}.{repetition}"
    result_path = output / "attempts" / f"{name}.json.gz"
    result = dict(configuration_id=configuration_id, repetition=repetition, version=VERSION,
                  execution_fingerprint=execution["fingerprint"], campaign_fingerprint=manifest["fingerprint"],
                  target_names=list(TARGET_NAMES), status="running", phase_order=["infer", "train"], phases={})
    torch.set_num_threads(1)
    try:
        if execution["code_fingerprint"] != execution_code() or execution["material_fingerprint"] != fingerprint(materials):
            raise ValueError("worker execution inputs changed")
        nvml = Nvml(execution["gpu_uuid"])
        check_gpu(nvml.read())
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("worker requires exactly one visible CUDA device")
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (12, 0):
            raise ValueError("worker GPU is not sm_120")
        torch.empty(1, device="cuda")
        # The pre-context check above requires an empty device. WSL may omit
        # this process from NVML temporarily, so retain the known worker PID as
        # the only allowed owner and reject every foreign PID if one appears.
        allowed = [os.getpid()]
        observed = wait_for_exclusive_owner(nvml, allowed[0])
        check_gpu(observed, allowed)
        controls = precision_controls(row["training"]["precision"])
        if row["training"]["precision"] == "bf16_amp" and not torch.cuda.is_bf16_supported():
            raise ValueError("unsupported precision: bf16_amp")
        hardware = dict(static={"memory_bytes": observed["memory_total_bytes"], "sm_count": props.multi_processor_count,
                                "compute_capability": 12.0},
                        environment={"cuda_version": str(execution["cuda"]), "cudnn_version": str(execution["cudnn"]),
                                     "pytorch_version": execution["torch"], "driver_version": execution["driver"]})
        design, audit = capture_workload(anchor, row, material, hardware)
        graph_path = output / "graphs" / f"{configuration_id}.json.gz"
        if graph_path.exists():
            if not canonical_equal(read_json(graph_path), design):
                raise ValueError("recaptured graph changed")
        else:
            atomic_write(graph_path, design, compress=True)
        check_gpu(nvml.read(), allowed)
        result.update(graph_path=str(graph_path.relative_to(output)), graph_sha256=file_sha256(graph_path),
                      capture_audit=audit, hardware=hardware, precision_controls=controls,
                      model_seed=SEED + repetition, input_seed=material["seed"],
                      gpu_ownership={"expected_pid": allowed[0], "observed_pids": observed["compute_pids"],
                                     "owner_observed": allowed[0] in observed["compute_pids"]})
        torch.manual_seed(SEED + repetition)
        model = GraphModel(anchor["model"]["NODE_SPECS"]).cuda()
        result.update(profile_phases(model, anchor, row, material, nvml, allowed, result["phases"],
                                     checkpoint=lambda: atomic_write(result_path, result, compress=True)))
        result["targets"] = target_values(result["phases"])
        try:
            result["first_two_epoch_comparison"] = target_values(result["phases"], first_two=True)
        except TelemetryError:
            result["first_two_epoch_comparison"] = None
        result["status"] = "ok"
        from .transfer_verification import verify_attempt
        verify_attempt(result, row, manifest, execution, output)
    except Exception as error:
        result["status"] = ("contention" if isinstance(error, ContentionError) else "oom" if isinstance(error, torch.OutOfMemoryError)
                            else "nonfinite" if isinstance(error, FloatingPointError) else
                            "unsupported_precision" if "unsupported precision" in str(error) else "failed")
        result["error"] = {"type": type(error).__name__, "message": str(error)[:4000]}
    atomic_write(result_path, result, compress=True)
    return result["status"] == "ok"


def export_labels(output, manifest, rows, experiments, execution, *, scope=None):
    from .transfer_verification import verify_attempt
    output = Path(output)
    labels, complete, missing = [], 0, []
    coverage = {name: defaultdict(Counter) for name in ("architecture_group", "modality", "batch", "precision")}
    if scope is None:
        active_rows = rows
        panel = set(experiments["repeat_panel"])
        labels_path = output / "labels.json.gz"
        report_path = output / "labeling-report.json"
        planned_records = manifest["planned_records"]
    else:
        by_id = {row["configuration_id"]: row for row in rows}
        active_rows = [by_id[key] for key in scope["configuration_ids"]]
        panel = set(scope["repeat_panel"])
        labels_path = output / f"labels-{scope['profile']}.json.gz"
        report_path = output / f"labeling-report-{scope['profile']}.json"
        planned_records = scope["planned_records"]
    for row in active_rows:
        attempts = []
        status = "pending"
        for repetition in range(3 if row["configuration_id"] in panel else 1):
            path = output / "attempts" / f"{row['configuration_id']}.{repetition}.json.gz"
            if not path.exists():
                break
            result = read_json(path)
            if result.get("status") != "ok":
                status = result.get("status", "failed")
                break
            verify_attempt(result, row, manifest, execution, output)
            attempts.append(result)
        accepted = len(attempts) == (3 if row["configuration_id"] in panel else 1)
        for name, value in (("architecture_group", row["group_id"]), ("modality", row["modality"]),
                            ("batch", str(row["training"]["microbatch_size"])), ("precision", row["training"]["precision"])):
            coverage[name][value]["accepted" if accepted else status] += 1
        if not accepted:
            missing.append(row["configuration_id"])
            continue
        first = attempts[0]
        label = dict(version=VERSION, scheduler_label_version=4, profile_point_id=row["configuration_id"],
                     hardware_id=HARDWARE_ID, split=row["split"], anchor_id=row["anchor_id"], group_id=row["group_id"],
                     target_names=list(TARGET_NAMES), targets=first["targets"], training=row["training"],
                     status="ok", graph_path=first["graph_path"], graph_sha256=first["graph_sha256"],
                     paired_source_ids=row["paired_source_ids"],
                     repeat_quality={name: {"values": [a["targets"][name] for a in attempts],
                                            "std": statistics.pstdev(a["targets"][name] for a in attempts)}
                                     for name in TARGET_NAMES},
                     execution_fingerprint=first["execution_fingerprint"],
                     campaign_execution_fingerprint=execution["fingerprint"])
        if scope is not None:
            label["campaign_scope_fingerprint"] = scope["fingerprint"]
        labels.append(label)
        complete += 1
    atomic_write(labels_path, labels, compress=True)
    report = dict(status="complete" if complete == planned_records else "incomplete",
                  verified_records=complete, planned_records=planned_records,
                  missing_or_failed=missing, labels_sha256=file_sha256(labels_path),
                  execution_fingerprint=execution["fingerprint"],
                  coverage={name: {value: dict(counts) for value, counts in groups.items()} for name, groups in coverage.items()})
    if scope is not None:
        report.update(campaign_profile=scope["profile"], campaign_scope_fingerprint=scope["fingerprint"],
                      planned_attempts=scope["planned_attempts"])
    atomic_write(report_path, report)
    return report


def resumable_attempt(path, row, manifest, execution, output):
    """Keep successful evidence; archive interrupted evidence before retrying."""
    from .transfer_verification import verify_attempt
    if not path.exists():
        return "pending"
    previous = read_json(path)
    resolve_attempt_execution(output, execution, previous.get("execution_fingerprint"))
    if previous.get("status") == "ok":
        verify_attempt(previous, row, manifest, execution, output)
        return "complete"
    fixed_graph_comparison = (previous.get("status") == "failed" and
                              previous.get("error", {}).get("message") == "recaptured graph changed" and
                              previous.get("execution_fingerprint") in
                              execution.get("compatible_predecessor_execution_fingerprints", []))
    if previous.get("status") in ("running", "contention", "worker_failed", "timeout") or fixed_graph_comparison:
        history = Path(output) / "interrupted_attempts"
        history.mkdir(exist_ok=True)
        preserved = history / f"{path.name.removesuffix('.json.gz')}.{file_sha256(path)}.json.gz"
        path.replace(preserved)
        return "pending"
    return "failed"


def label(output, source_data_root=DEFAULT_SOURCE_ROOT, data=DATA, *, resume=False, campaign_profile="full"):
    from .transfer_verification import verify, verify_attempt
    from .transfer_sources import verify_data
    output = Path(output)
    if not (output / "manifest.json").exists():
        prepare(output, data, campaign_profile=campaign_profile)
    verify(output, data=data)
    manifest = read_json(output / "manifest.json")
    anchors, rows = read_json(output / "anchors.json.gz"), read_json(output / "configurations.json.gz")
    experiments = read_json(output / "experiments.json.gz")
    scope = None if campaign_profile == "full" else prepare_campaign_scope(
        output, rows, experiments, manifest["fingerprint"], campaign_profile)
    source_report = verify_data(source_data_root, anchors)
    atomic_write(output / "source-data-verification.json", source_report)
    materials = source_materials(source_data_root, anchors)
    nvml = Nvml()
    observed = nvml.read()
    check_gpu(observed)
    if observed["sm_percent"] > 5:
        raise ContentionError("GPU is active; wait until the current experiment finishes")
    with (output / ".labeling.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = execution_identity(manifest, materials, observed, nvml)
        identity["source_data_fingerprint"] = source_report["source_data_fingerprint"]
        execution = bind_execution(output, identity, resume=resume)
        atomic_write(output / "source_materials.json.gz", materials, compress=True)
        (output / "attempts").mkdir(exist_ok=True)
        by_id = {r["configuration_id"]: r for r in rows}
        active_ids = [r["configuration_id"] for r in rows] if scope is None else scope["configuration_ids"]
        pilot = experiments["pilot"] if scope is None else scope["pilot"]
        panel = set(experiments["repeat_panel"] if scope is None else scope["repeat_panel"])
        order = [*pilot, *(key for key in active_ids if key not in pilot)]
        export = lambda: export_labels(output, manifest, rows, experiments, execution, scope=scope)
        failed_pilot = False
        for index, key in enumerate(order):
            if index == len(pilot):
                if failed_pilot:
                    break
                event(output, event="pilot_passed", configurations=len(pilot), targets_per_configuration=12)
            for repetition in range(3 if key in panel else 1):
                attempt = output / "attempts" / f"{key}.{repetition}.json.gz"
                state = resumable_attempt(attempt, by_id[key], manifest, execution, output)
                if state == "complete":
                    continue
                if state == "failed":
                    event(output, event="retained_failure", configuration_id=key, repetition=repetition, status=read_json(attempt)["status"])
                    failed_pilot |= key in pilot
                    break
                check_gpu(nvml.read())
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": observed["uuid"], "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
                command = worker_command(output, key, repetition)
                event(output, event="measurement_start", configuration_id=key, repetition=repetition,
                      index=index, planned=len(active_ids), campaign_profile=campaign_profile)
                with (output / "worker.log").open("a") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    started = time.monotonic()
                    try:
                        while True:
                            try:
                                process.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                event(output, event="measurement_running", configuration_id=key,
                                      repetition=repetition, elapsed_seconds=round(time.monotonic() - started, 1))
                                if time.monotonic() - started > PROTOCOL["worker_timeout_seconds"]:
                                    raise TimeoutError("configuration worker timed out")
                    except BaseException as error:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        partial = read_json(attempt) if attempt.exists() else dict(configuration_id=key, repetition=repetition)
                        partial.update(status="timeout" if isinstance(error, TimeoutError) else "running",
                                       execution_fingerprint=execution["fingerprint"],
                                       error={"type": type(error).__name__, "message": str(error)[:4000]})
                        atomic_write(attempt, partial, compress=True)
                        export()
                        if isinstance(error, KeyboardInterrupt):
                            event(output, event="campaign_paused", configuration_id=key, repetition=repetition,
                                  campaign_profile=campaign_profile,
                                  resume_from="last verified configuration; interrupted configuration will be retried")
                        raise
                if not attempt.exists():
                    atomic_write(attempt, dict(status="worker_failed", configuration_id=key, repetition=repetition,
                                                exit_code=process.returncode, execution_fingerprint=execution["fingerprint"]),
                                 compress=True)
                result = read_json(attempt)
                event(output, event="measurement_finished", configuration_id=key, repetition=repetition, status=result["status"])
                if result["status"] == "contention":
                    export()
                    raise ContentionError("GPU contention detected; campaign stopped")
                if result["status"] != "ok":
                    failed_pilot |= key in pilot
                    break
                verify_attempt(result, by_id[key], manifest, execution, output)
        return export()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-data", "verify", "label", "_worker"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--source-data-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        help="Original A10 source data (default: record/perfseer-v32/transfer-source-data)")
    parser.add_argument("--reuse-data-root", type=Path,
                        help="Optional existing raw-data cache to verify and copy during prepare-data")
    parser.add_argument("--campaign-profile", choices=CAMPAIGN_PROFILES, default="full",
                        help="Label the full campaign or the frozen approximately 24-hour subset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--configuration-id")
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return 0 if worker(args.output, args.configuration_id, args.repetition) else 1
    if args.command == "prepare":
        result = prepare(args.output, args.data, campaign_profile=args.campaign_profile)
    elif args.command == "prepare-data":
        from .transfer_sources import prepare_data
        prepare(args.output, args.data)
        result = prepare_data(args.source_data_root, read_json(args.output / "anchors.json.gz"),
                              read_json(args.output / "source_aliases.json.gz"), cache_root=args.reuse_data_root)
    elif args.command == "verify":
        from .transfer_verification import verify
        result = verify(args.output, data=args.data, require_complete=args.require_complete,
                        campaign_profile=args.campaign_profile)
        if (args.source_data_root / "source-data-manifest.json").exists():
            from .transfer_sources import verify_data
            result["source_data"] = verify_data(args.source_data_root, read_json(args.output / "anchors.json.gz"),
                                                rebuild_masks=True)
    else:
        if not (args.source_data_root / "source-data-manifest.json").is_file():
            parser.error(f"source data is not prepared at {args.source_data_root}; run prepare-data first")
        def pause(*_):
            raise KeyboardInterrupt("SIGTERM")

        previous = signal.signal(signal.SIGTERM, pause)
        try:
            result = label(args.output, args.source_data_root, args.data, resume=args.resume,
                           campaign_profile=args.campaign_profile)
        except KeyboardInterrupt:
            result = {
                "status": "paused",
                "campaign_profile": args.campaign_profile,
                "output": str(args.output),
                "resume_from": "last verified configuration; the interrupted configuration will be retried",
            }
        finally:
            signal.signal(signal.SIGTERM, previous)
    print(json.dumps(result, indent=2), flush=True)
    return 1 if result.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
