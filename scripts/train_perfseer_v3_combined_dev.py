#!/usr/bin/env python3
"""Development training smoke test for the combined A10/A10G staging corpus.

This runner intentionally does not bypass the production v3 manifest gates.  The
accepted-node archive has all six current targets, but neither supplied archive
contains the original materialized GraphIR.  For workflow verification, this
script reconstructs development-only GraphIR from the archived model designs and
the repository's deterministic task fixtures.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from perfseer_v3.capture_training import capture_training_graph
from perfseer_v3.coarsen_v3 import coarsen_graph
from perfseer_v3.dataset_pack.adapters import adapter_for_task
from perfseer_v3.dataset_pack.local_runtime import bind_candidate_batch_shape
from perfseer_v3.features import apply_normalization, build_graph_features, fit_normalization
from perfseer_v3.model import SeerNetV3, SeerNetV3Config
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.training import (
    TARGET_NAMES,
    TrainingSampleV3,
    student_distill_step,
    teacher_train_step,
)
from perfseer_v3.training_runner import (
    _amp_settings,
    _batches,
    _classify_workload_regime,
    _predict,
    _regression_validation_score,
)


DATASET_MANIFEST_VERSION = "perfseer_v3_combined_a10_dataset_manifest_v1"
DATASET_SAMPLE_SCHEMA = "perfseer_v3_combined_a10_labeled_sample_v1"
FULL_LABEL_SCHEMA = "dataset_pack_6_target_v1"
GRAPH_RECONSTRUCTION = (
    "development fixture reconstruction; not the original measured GraphIR and "
    "not eligible for a production PerfSeer v3 artifact"
)
T1_OVERRIDES: dict[str, Any] = {
    "hidden": 1280,
    "num_blocks": 10,
    "num_outputs": 6,
    "exact_embedding_dim": 96,
    "family_embedding_dim": 48,
    "hash_embedding_dim": 24,
    "overload_hash_embedding_dim": 24,
    "phase_embedding_dim": 24,
    "input_dtype_embedding_dim": 16,
    "dtype_embedding_dim": 16,
    "accumulation_dtype_embedding_dim": 16,
    "backend_embedding_dim": 16,
    "feature_quality_embedding_dim": 8,
    "layout_embedding_dim": 12,
    "rank_embedding_dim": 12,
    "hardware_profile_embedding_dim": 64,
    "adapter_rank": 32,
    "optimizer_embedding_dim": 8,
    "optimizer_family_embedding_dim": 8,
    "optimizer_hash_embedding_dim": 8,
    "scheduler_embedding_dim": 8,
    "scheduler_family_embedding_dim": 8,
    "scheduler_hash_embedding_dim": 8,
    "node_identity_fusion": "additive",
    "pooling_mode": "existing",
    "adapter_policy": "base",
}
S1_OVERRIDES: dict[str, Any] = {
    "hidden": 224,
    "num_blocks": 2,
    "num_outputs": 6,
    "exact_embedding_dim": 40,
    "family_embedding_dim": 20,
    "hash_embedding_dim": 10,
    "overload_hash_embedding_dim": 10,
    "phase_embedding_dim": 8,
    "input_dtype_embedding_dim": 8,
    "dtype_embedding_dim": 8,
    "accumulation_dtype_embedding_dim": 8,
    "backend_embedding_dim": 8,
    "feature_quality_embedding_dim": 4,
    "layout_embedding_dim": 6,
    "rank_embedding_dim": 6,
    "hardware_profile_embedding_dim": 64,
    "adapter_rank": 16,
    "optimizer_embedding_dim": 4,
    "optimizer_family_embedding_dim": 4,
    "optimizer_hash_embedding_dim": 4,
    "scheduler_embedding_dim": 4,
    "scheduler_family_embedding_dim": 4,
    "scheduler_hash_embedding_dim": 4,
    "node_identity_fusion": "additive",
    "pooling_mode": "existing",
    "adapter_policy": "base",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/perfseer_v3/dataset_with_label/ready_for_train"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3, help="teacher epochs")
    parser.add_argument("--distillation-epochs", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=12)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-capture-microbatch", type=int, default=8)
    parser.add_argument(
        "--families",
        default="",
        help="optional comma-separated model-family allowlist",
    )
    parser.add_argument("--capacity", choices=("smoke", "T1"), default="smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp", choices=("none", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--teacher-quality-gate", type=float, default=0.90)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument("--hard-label-weight", type=float, default=0.6)
    parser.add_argument(
        "--representation-distillation-weight", type=float, default=0.05
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-materialized", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_root / split / "samples.jsonl.gz"
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema_version") != DATASET_SAMPLE_SCHEMA:
                raise ValueError(f"unsupported combined dataset schema in {path}")
            if row.get("split") != split:
                raise ValueError(f"row split mismatch in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"combined dataset split is empty: {path}")
    return rows


def _eligible_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_microbatch: int,
    families: set[str],
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("native_target_schema") != FULL_LABEL_SCHEMA:
            continue
        if tuple(row.get("target_names", ())) != TARGET_NAMES:
            continue
        if families and row.get("model_family_id") not in families:
            continue
        targets = row.get("targets")
        required = row.get("required_label")
        if not isinstance(targets, Mapping) or not isinstance(required, Mapping):
            continue
        if any(name not in targets or not math.isfinite(float(targets[name])) for name in TARGET_NAMES):
            continue
        try:
            design = json.loads(str(required["model_design_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        microbatch = int(design.get("microbatch_size", 0))
        if not 1 <= microbatch <= maximum_microbatch:
            continue
        row["_model_design"] = design
        eligible.append(row)
    return eligible


def _candidate_order(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["model_family_id"])].append(row)
    for family_rows in by_family.values():
        family_rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['sample_id']}".encode("utf-8")
            ).hexdigest()
        )
    ordered: list[dict[str, Any]] = []
    while any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family]:
                ordered.append(by_family[family].pop(0))
    return ordered


def _capture_features(
    row: Mapping[str, Any], registry: OperationRegistry
) -> tuple[Any | None, dict[str, Any]]:
    design = dict(row["_model_design"])
    adapter = adapter_for_task(str(design["task_id"]))
    candidate = SimpleNamespace(
        microbatch_size=int(design["microbatch_size"]),
        input_signature=dict(design["input_signature"]),
        family_id=str(design["family_id"]),
    )
    fixture = bind_candidate_batch_shape(
        candidate,
        adapter,
        adapter.build_train_dataset()[0],
    )
    inputs = adapter.build_model_inputs(fixture)
    target = adapter.build_targets(fixture)
    module = importlib.import_module(str(design["factory_id"]))
    model = module.build_model(
        output_width=adapter.target_width,
        task_kind=adapter.task_kind,
        seed=int(design["seed_policy"]["seed"]),
        architecture_parameters=dict(design["architecture_parameters"]),
    )
    optimizer = dict(design["optimizer"])
    scheduler = dict(design["scheduler"])
    result = capture_training_graph(
        model,
        (inputs,),
        target=target,
        loss_fn=adapter.build_loss,
        optimizer_name=str(optimizer.pop("name")),
        optimizer_config=optimizer,
        scheduler_name=str(scheduler.pop("name")),
        scheduler_config=scheduler,
        training_config={
            "gradient_accumulation_steps": int(design["gradient_accumulation_steps"]),
            "precision_policy": dict(design["precision_policy"]),
        },
        target_hardware_id=str(row["hardware_id"]),
        registry=registry,
        allow_analytical_fallback=True,
    )
    audit = {
        "sample_id": row["sample_id"],
        "family": row["model_family_id"],
        "modality": row["modality"],
        "microbatch_size": candidate.microbatch_size,
        "capture_backend": result.backward_backend,
        "capture_success": result.success,
        "failures": [failure.to_dict() for failure in result.failures],
    }
    if not result.success or result.graph is None:
        return None, audit
    graph = coarsen_graph(result.graph, registry=registry)
    audit.update(
        {
            "graph_sha256": graph.graph_sha256,
            "nodes": len(graph.nodes),
            "edges": len(graph.tensor_edges),
        }
    )
    return build_graph_features(graph, registry=registry), audit


def _materialize_split(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    registry: OperationRegistry,
) -> tuple[list[tuple[dict[str, Any], Any]], list[dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], Any]] = []
    audits: list[dict[str, Any]] = []
    for row in _candidate_order(rows, seed):
        try:
            features, audit = _capture_features(row, registry)
        except Exception as error:
            features = None
            audit = {
                "sample_id": row["sample_id"],
                "family": row["model_family_id"],
                "modality": row["modality"],
                "capture_success": False,
                "exception": f"{type(error).__name__}: {error}",
            }
        audits.append(audit)
        print(json.dumps({"event": "capture", **audit}, sort_keys=True), flush=True)
        if features is not None:
            selected.append((row, features))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only reconstructed {len(selected)} of {count} requested samples"
        )
    return selected, audits


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != DATASET_MANIFEST_VERSION:
        raise ValueError("dataset manifest schema does not match the development trainer")
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    source_rows = {
        split: _load_rows(dataset_root, split) for split in ("train", "validation")
    }
    eligible = {
        split: _eligible_rows(
            rows,
            maximum_microbatch=args.max_capture_microbatch,
            families=families,
        )
        for split, rows in source_rows.items()
    }
    requested = {
        "train": args.train_samples,
        "validation": args.validation_samples,
    }
    if any(requested[split] < 1 for split in requested):
        raise ValueError("train and validation sample counts must be positive")
    if any(len(eligible[split]) < requested[split] for split in requested):
        raise ValueError(
            "the full-label/microbatch/family filters leave too few eligible rows: "
            + ", ".join(
                f"{split}={len(eligible[split])}/{requested[split]}"
                for split in requested
            )
        )
    registry = OperationRegistry.load()
    raw: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    capture_audits: dict[str, list[dict[str, Any]]] = {}
    for offset, split in enumerate(("train", "validation")):
        raw[split], capture_audits[split] = _materialize_split(
            eligible[split],
            count=requested[split],
            seed=args.seed + offset,
            registry=registry,
        )
    fingerprint = hashlib.sha256(
        json.dumps(
            sorted(row["sample_id"] for row, _ in raw["train"]),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    normalization = fit_normalization(
        [features for _, features in raw["train"]],
        split_name="train",
        split_fingerprint=fingerprint,
    )
    samples: dict[str, list[TrainingSampleV3]] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        samples[split] = []
        selected_rows[split] = []
        for row, features in raw[split]:
            target = torch.tensor(
                [float(row["targets"][name]) for name in TARGET_NAMES],
                dtype=torch.float32,
            )
            sample = TrainingSampleV3(
                features=apply_normalization(features, normalization),
                target=target,
                source_group=str(row["split_group_id"]),
                graph_signature=str(features.metadata.get("graph_sha256", row["model_id"])),
                workload_regime=_classify_workload_regime(features),
            )
            sample.validate()
            samples[split].append(sample)
            selected_rows[split].append(
                {
                    "sample_id": row["sample_id"],
                    "model_id": row["model_id"],
                    "family": row["model_family_id"],
                    "modality": row["modality"],
                    "hardware_id": row["hardware_id"],
                }
            )
    train_groups = {row["split_group_id"] for row in source_rows["train"]}
    validation_groups = {row["split_group_id"] for row in source_rows["validation"]}
    return {
        "samples": samples,
        "normalization": normalization,
        "audit": {
            "dataset_manifest_sha256": _sha256(manifest_path),
            "source_split_counts": {
                split: len(rows) for split, rows in source_rows.items()
            },
            "eligible_full_label_counts": {
                split: len(rows) for split, rows in eligible.items()
            },
            "selected_rows": selected_rows,
            "capture_audits": capture_audits,
            "source_group_overlap": sorted(train_groups & validation_groups),
            "target_names": list(TARGET_NAMES),
            "graph_reconstruction": GRAPH_RECONSTRUCTION,
        },
    }


def _distributed_context(device_name: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name != "cuda" and world_size > 1:
        raise ValueError("multi-process training requires --device cuda")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, device


def _rank_batches(
    samples: Sequence[TrainingSampleV3],
    *,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> list[list[TrainingSampleV3]]:
    order = list(samples)
    random.Random(seed).shuffle(order)
    per_rank = math.ceil(len(order) / world_size)
    padded = order + order[: per_rank * world_size - len(order)]
    local = padded[rank::world_size]
    return _batches(local, batch_size=batch_size, seed=seed)


def _target_mae(predictions: Mapping[str, torch.Tensor]) -> dict[str, float]:
    difference = (predictions["prediction"] - predictions["target"]).abs().mean(dim=0)
    return {name: float(difference[index]) for index, name in enumerate(TARGET_NAMES)}


def _regression_accuracy(predictions: Mapping[str, torch.Tensor]) -> float:
    return max(0.0, 1.0 - _regression_validation_score(predictions))


def _passes_teacher_quality_gate(accuracy: float, threshold: float) -> bool:
    return math.isfinite(accuracy) and accuracy > threshold


def _checkpoint_payload(
    model: SeerNetV3,
    model_config: SeerNetV3Config,
    normalization: Any,
    *,
    role: str,
    validation_accuracy: float,
) -> dict[str, Any]:
    return {
        "development_only": True,
        "graph_reconstruction": GRAPH_RECONSTRUCTION,
        "role": role,
        "model_config": model_config.to_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "normalization": normalization,
        "target_names": TARGET_NAMES,
        "validation_accuracy": validation_accuracy,
    }


def main() -> int:
    args = _arguments()
    if args.epochs < 1 or args.distillation_epochs < 0 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.student_learning_rate <= 0:
        raise ValueError("teacher and student learning rates must be positive")
    if not 0.0 <= args.teacher_quality_gate < 1.0:
        raise ValueError("teacher quality gate must be in [0, 1)")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("target accuracy must be in (0, 1]")
    if not 0.0 <= args.hard_label_weight <= 1.0:
        raise ValueError("hard label weight must be in [0, 1]")
    if args.representation_distillation_weight < 0:
        raise ValueError("representation distillation weight must be nonnegative")
    rank, _, world_size, device = _distributed_context(args.device)
    primary = rank == 0
    output_dir = args.output_dir.resolve()
    materialized_path = output_dir / "materialized-development-samples.pt"
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    status: list[dict[str, Any] | None] = [None]
    materialized: dict[str, Any] | None = None
    if primary:
        try:
            if args.reuse_materialized and materialized_path.exists():
                materialized = torch.load(
                    materialized_path, map_location="cpu", weights_only=False
                )
            else:
                materialized = _materialize(args)
                torch.save(materialized, materialized_path)
            status[0] = {"ok": True}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if not status[0] or not status[0]["ok"]:
        raise RuntimeError(str(status[0]))
    if materialized is None:
        materialized = torch.load(
            materialized_path, map_location="cpu", weights_only=False
        )
    samples = materialized["samples"]
    if materialized["audit"]["source_group_overlap"]:
        raise RuntimeError("the prepared train/validation source groups overlap")

    torch.manual_seed(args.seed)
    registry = OperationRegistry.load()
    teacher_overrides = T1_OVERRIDES if args.capacity == "T1" else {}
    teacher_config = SeerNetV3Config.from_registry(
        registry,
        samples["train"][0].features.layout,
        **teacher_overrides,
    )
    teacher_model = SeerNetV3(teacher_config).to(device)
    teacher_parameter_audit = teacher_model.configure_trainable_parameters("base")
    teacher_train_model: torch.nn.Module = teacher_model
    if world_size > 1:
        teacher_train_model = DistributedDataParallel(
            teacher_model, device_ids=[device.index]
        )
    teacher_optimizer = torch.optim.AdamW(
        teacher_train_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    teacher_autocast_dtype, teacher_scaler = _amp_settings(device, args.amp)
    teacher_epoch_reports: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        teacher_train_model.train()
        losses = [
            teacher_train_step(
                teacher_train_model,
                batch,
                teacher_optimizer,
                autocast_dtype=teacher_autocast_dtype,
                scaler=teacher_scaler,
            )
            for batch in _rank_batches(
                samples["train"],
                batch_size=args.batch_size,
                seed=args.seed + epoch - 1,
                rank=rank,
                world_size=world_size,
            )
        ]
        loss_total = torch.tensor(
            [sum(losses), float(len(losses))], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(loss_total)
        predictions = _predict(
            teacher_model,
            samples["validation"],
            device=device,
            batch_size=args.batch_size,
        )
        validation_score = _regression_validation_score(predictions)
        validation_accuracy = _regression_accuracy(predictions)
        epoch_report = {
            "epoch": epoch,
            "train_loss": float(loss_total[0] / loss_total[1].clamp_min(1)),
            "validation_score": validation_score,
            "validation_accuracy": validation_accuracy,
            "target_accuracy_met": validation_accuracy >= args.target_accuracy,
            "validation_target_mae": _target_mae(predictions),
        }
        if not all(
            math.isfinite(value)
            for value in (
                epoch_report["train_loss"],
                validation_score,
                validation_accuracy,
            )
        ):
            raise RuntimeError(f"epoch {epoch} produced a non-finite metric")
        teacher_epoch_reports.append(epoch_report)
        if primary:
            print(
                json.dumps({"event": "teacher_epoch", **epoch_report}, sort_keys=True),
                flush=True,
            )

    teacher_accuracy = float(teacher_epoch_reports[-1]["validation_accuracy"])
    teacher_gate_passed = _passes_teacher_quality_gate(
        teacher_accuracy, args.teacher_quality_gate
    )
    teacher_checkpoint_path = output_dir / "teacher-development-checkpoint.pt"
    student_checkpoint_path = output_dir / "student-development-checkpoint.pt"
    report_path = output_dir / "training-report.json"
    if primary:
        torch.save(
            _checkpoint_payload(
                teacher_model,
                teacher_config,
                materialized["normalization"],
                role="teacher",
                validation_accuracy=teacher_accuracy,
            ),
            teacher_checkpoint_path,
        )
        gpu = None
        peak_vram_mib = None
        if device.type == "cuda":
            gpu = torch.cuda.get_device_name(device)
            peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        report: dict[str, Any] = {
            "status": (
                "teacher_quality_gate_passed"
                if teacher_gate_passed
                else "teacher_quality_gate_failed"
            ),
            "development_only": True,
            "graph_reconstruction": GRAPH_RECONSTRUCTION,
            "device": str(device),
            "gpu": gpu,
            "world_size": world_size,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "capacity": args.capacity,
            "epochs_requested": args.epochs,
            "epochs_completed": len(teacher_epoch_reports),
            "distillation_epochs_requested": args.distillation_epochs,
            "distillation_epochs_completed": 0,
            "distillation_started": False,
            "validation_interval_epochs": 1,
            "train_samples": len(samples["train"]),
            "validation_samples": len(samples["validation"]),
            "batch_size_per_rank": args.batch_size,
            "target_names": list(TARGET_NAMES),
            "sm_targets": [
                "train_avg_sm_util_percent",
                "train_p95_sm_util_percent",
            ],
            "accuracy_definition": (
                "max(0, 1 - composite validation error), where composite error is "
                "0.5 * (positive-target MAPE + utilization MAE / 100)"
            ),
            "target_accuracy": args.target_accuracy,
            "teacher_target_accuracy_met": teacher_accuracy >= args.target_accuracy,
            "teacher_quality_gate": {
                "operator": ">",
                "threshold": args.teacher_quality_gate,
                "observed_accuracy": teacher_accuracy,
                "passed": teacher_gate_passed,
            },
            "optimizers": {
                "teacher": {
                    "name": "AdamW",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "scheduler": None,
                },
                "student": {
                    "name": "AdamW",
                    "learning_rate": args.student_learning_rate,
                    "weight_decay": args.weight_decay,
                    "scheduler": None,
                },
            },
            "teacher_total_parameter_count": teacher_parameter_audit[
                "total_parameter_count"
            ],
            "teacher_trainable_parameter_count": teacher_parameter_audit[
                "trainable_parameter_count"
            ],
            "peak_vram_allocated_mib": peak_vram_mib,
            "epochs": teacher_epoch_reports,
            "distillation_epochs": [],
            "dataset_audit": materialized["audit"],
            "checkpoint": str(teacher_checkpoint_path),
            "teacher_checkpoint": str(teacher_checkpoint_path),
            "student_checkpoint": None,
            "materialized_samples": str(materialized_path),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "teacher_quality_gate",
                    **report["teacher_quality_gate"],
                    "report": str(report_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
    if args.distillation_epochs > 0 and not teacher_gate_passed:
        if world_size > 1:
            dist.destroy_process_group()
        raise RuntimeError(
            "teacher validation accuracy "
            f"{teacher_accuracy:.6f} did not satisfy the strict "
            f"> {args.teacher_quality_gate:.6f} distillation gate"
        )

    student_epoch_reports: list[dict[str, Any]] = []
    student_model: SeerNetV3 | None = None
    student_config: SeerNetV3Config | None = None
    student_parameter_audit: dict[str, Any] | None = None
    if args.distillation_epochs > 0:
        del teacher_optimizer
        del teacher_train_model
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        student_overrides = S1_OVERRIDES if args.capacity == "T1" else {}
        student_config = SeerNetV3Config.from_registry(
            registry,
            samples["train"][0].features.layout,
            **student_overrides,
        )
        student_model = SeerNetV3(student_config).to(device)
        student_parameter_audit = student_model.configure_trainable_parameters("base")
        student_train_model: torch.nn.Module = student_model
        if world_size > 1:
            student_train_model = DistributedDataParallel(
                student_model, device_ids=[device.index]
            )
        student_optimizer = torch.optim.AdamW(
            student_train_model.parameters(),
            lr=args.student_learning_rate,
            weight_decay=args.weight_decay,
        )
        student_autocast_dtype, student_scaler = _amp_settings(device, args.amp)
        for epoch in range(1, args.distillation_epochs + 1):
            student_train_model.train()
            losses = [
                student_distill_step(
                    student_train_model,
                    teacher_model,
                    batch,
                    student_optimizer,
                    hard_label_weight=args.hard_label_weight,
                    representation_weight=args.representation_distillation_weight,
                    autocast_dtype=student_autocast_dtype,
                    scaler=student_scaler,
                )
                for batch in _rank_batches(
                    samples["train"],
                    batch_size=args.batch_size,
                    seed=args.seed + args.epochs + epoch - 1,
                    rank=rank,
                    world_size=world_size,
                )
            ]
            loss_total = torch.tensor(
                [sum(losses), float(len(losses))], dtype=torch.float64, device=device
            )
            if world_size > 1:
                dist.all_reduce(loss_total)
            predictions = _predict(
                student_model,
                samples["validation"],
                device=device,
                batch_size=args.batch_size,
            )
            validation_score = _regression_validation_score(predictions)
            validation_accuracy = _regression_accuracy(predictions)
            epoch_report = {
                "epoch": epoch,
                "train_loss": float(loss_total[0] / loss_total[1].clamp_min(1)),
                "validation_score": validation_score,
                "validation_accuracy": validation_accuracy,
                "target_accuracy_met": validation_accuracy >= args.target_accuracy,
                "validation_target_mae": _target_mae(predictions),
            }
            if not all(
                math.isfinite(value)
                for value in (
                    epoch_report["train_loss"],
                    validation_score,
                    validation_accuracy,
                )
            ):
                raise RuntimeError(
                    f"distillation epoch {epoch} produced a non-finite metric"
                )
            student_epoch_reports.append(epoch_report)
            if primary:
                print(
                    json.dumps(
                        {"event": "distillation_epoch", **epoch_report}, sort_keys=True
                    ),
                    flush=True,
                )

        if world_size > 1:
            dist.barrier()
        if primary:
            student_accuracy = float(
                student_epoch_reports[-1]["validation_accuracy"]
            )
            torch.save(
                _checkpoint_payload(
                    student_model,
                    student_config,
                    materialized["normalization"],
                    role="student",
                    validation_accuracy=student_accuracy,
                ),
                student_checkpoint_path,
            )
            report.update(
                {
                    "status": "passed",
                    "distillation_epochs_completed": len(student_epoch_reports),
                    "distillation_started": True,
                    "distillation_hard_label_weight": args.hard_label_weight,
                    "distillation_representation_weight": (
                        args.representation_distillation_weight
                    ),
                    "student_capacity": "S1" if args.capacity == "T1" else "smoke",
                    "student_target_accuracy_met": (
                        student_accuracy >= args.target_accuracy
                    ),
                    "student_total_parameter_count": student_parameter_audit[
                        "total_parameter_count"
                    ],
                    "student_trainable_parameter_count": student_parameter_audit[
                        "trainable_parameter_count"
                    ],
                    "peak_vram_allocated_mib": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else None
                    ),
                    "distillation_epochs": student_epoch_reports,
                    "student_checkpoint": str(student_checkpoint_path),
                }
            )
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif primary:
        report["status"] = "passed"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if primary:
        print(
            json.dumps(
                {"event": "complete", "report": str(report_path), "status": "passed"},
                sort_keys=True,
            ),
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
