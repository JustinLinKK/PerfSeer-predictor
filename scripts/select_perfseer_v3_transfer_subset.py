#!/usr/bin/env python3
"""Select a deterministic grouped and latent-diverse target GPU subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.transfer_subset import (
    TransferSubsetRequestV3,
    read_embeddings,
    read_jsonl,
    select_transfer_subset,
)
from perfseer_v3.artifact import load_checkpoint_artifact, sha256_file
from perfseer_v3.graph_ir_v3 import GraphIRV3
from perfseer_v3.hardware_transfer import BaseTransferLineageV3
from perfseer_v3.training_runner import TrainingManifestV3


def _operation_cost_regime(graph: GraphIRV3) -> str:
    if graph.global_features.peak_live_activation_bytes >= 16 * 1024**3:
        return "capacity_bound"
    operation_count = max(1, graph.global_features.operation_nodes)
    if graph.global_features.total_flops / operation_count < 1_000_000.0:
        return "launch_bound"
    traffic = sum(
        node.bytes_read.value + node.bytes_written.value for node in graph.nodes
    )
    if graph.global_features.total_flops / max(1.0, traffic) >= 32.0:
        return "compute_bound"
    return "memory_bound"


def _resource_regime(graph: GraphIRV3) -> str:
    pressure = (
        graph.global_features.peak_live_activation_bytes
        + max((node.estimated_workspace_bytes.value for node in graph.nodes), default=0.0)
    )
    if pressure >= 8 * 1024**3:
        return "heavy"
    if pressure >= 1024**3:
        return "standard"
    return "light"


def _high_cost_operation_family(graph: GraphIRV3) -> str:
    scores: dict[str, float] = defaultdict(float)
    for node in graph.nodes:
        scores[node.family] += (
            node.flops.value
            + node.bytes_read.value
            + node.bytes_written.value
            + node.estimated_workspace_bytes.value
        )
    return max(scores, key=lambda family: (scores[family], family), default="unknown")


def _batch_extremity(row: Mapping[str, Any], graph: GraphIRV3) -> str:
    raw = row.get("final_microbatch_size", graph.metadata.get("batch_size", 1))
    batch_size = int(raw)
    if batch_size <= 2:
        return "very_small"
    if batch_size >= 128:
        return "very_large"
    return "typical"


def enrich_frozen_rows(
    accepted_rows: Sequence[Mapping[str, Any]],
    base_manifest: TrainingManifestV3,
) -> list[dict[str, Any]]:
    """Bind final labels to verified graph paths and production strata."""

    manifest_rows = {row.sample_id: row for row in base_manifest.rows}
    if len(manifest_rows) != len(base_manifest.rows):
        raise ValueError("base training manifest repeats a sample ID")
    accepted_ids = {
        str(row.get("configuration_id", row.get("sample_id", "")))
        for row in accepted_rows
    }
    if accepted_ids != set(manifest_rows):
        raise ValueError(
            "accepted labels and the frozen base training manifest have different IDs"
        )
    enriched: list[dict[str, Any]] = []
    for accepted in accepted_rows:
        configuration_id = str(
            accepted.get("configuration_id", accepted.get("sample_id", ""))
        )
        manifest_row = manifest_rows[configuration_id]
        graph_path = (base_manifest.path.parent / manifest_row.graph_path).resolve()
        graph = GraphIRV3.load(graph_path)
        accepted_signature = str(
            accepted.get("graph_signature", accepted.get("graph_sha256", ""))
        )
        if (
            manifest_row.graph_signature != accepted_signature
            or graph.graph_sha256 != accepted_signature
        ):
            raise ValueError(
                f"accepted row {configuration_id!r} differs from its frozen graph"
            )
        if (
            manifest_row.split != str(accepted.get("split", ""))
            or manifest_row.source_group != str(accepted.get("source_group", ""))
        ):
            raise ValueError(
                f"accepted row {configuration_id!r} differs from its grouped split"
            )
        unknown_fraction = (
            graph.coverage.unknown_operations + graph.coverage.custom_operations
        ) / max(1, len(graph.nodes))
        optimizer = str(
            accepted.get("optimizer", accepted.get("optimizer_id", ""))
        )
        scheduler = str(
            accepted.get("scheduler", accepted.get("scheduler_id", ""))
        )
        if not optimizer or not scheduler or not math.isfinite(unknown_fraction):
            raise ValueError("frozen transfer strata are incomplete")
        enriched.append(
            {
                **dict(accepted),
                "graph_path": str(graph_path),
                "modality": str(
                    accepted.get("modality", accepted.get("quota_modality", ""))
                ),
                "model_family": str(
                    accepted.get("model_family", accepted.get("family_id", ""))
                ),
                "precision_policy": str(
                    accepted.get(
                        "precision_policy", accepted.get("precision_id", "")
                    )
                ),
                "optimizer": optimizer,
                "scheduler": scheduler,
                "optimizer_scheduler": f"{optimizer}|{scheduler}",
                "resource_regime": _resource_regime(graph),
                "operation_cost_regime": _operation_cost_regime(graph),
                "high_cost_operation_family": _high_cost_operation_family(graph),
                "unknown_custom_regime": (
                    "unknown_custom_heavy"
                    if unknown_fraction >= 0.10
                    else "unknown_custom_present"
                    if unknown_fraction > 0
                    else "fully_known"
                ),
                "activation_checkpointing": str(
                    bool(graph.optimizer_config.get("activation_checkpointing", False))
                ).lower(),
                "batch_extremity": _batch_extremity(accepted, graph),
            }
        )
    return enriched


def build_base_transfer_lineage(
    base_manifest: TrainingManifestV3,
    *,
    base_teacher_artifact: Path,
    base_student_artifact: Path,
    embeddings_path: Path,
) -> BaseTransferLineageV3:
    """Verify and bind the exact base artifacts used by target selection."""

    if base_manifest.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
        raise ValueError("transfer selection base manifest must target the frozen A10G")
    if base_manifest.dataset_gate.dataset_fingerprint != base_manifest.dataset_fingerprint:
        raise ValueError("base manifest dataset fingerprint differs from its frozen rows")
    if base_manifest.dataset_gate.split_fingerprint != base_manifest.split_fingerprint:
        raise ValueError("base manifest split fingerprint differs from its frozen rows")

    teacher = load_checkpoint_artifact(base_teacher_artifact)
    student = load_checkpoint_artifact(base_student_artifact)
    failures: list[str] = []
    for name, artifact, expected_release in (
        ("teacher", teacher, "perfseer_v3_teacher"),
        ("student", student, "perfseer_v3_student"),
    ):
        metadata = artifact.metadata
        if metadata.model_release != expected_release:
            failures.append(f"base {name} artifact has the wrong model role")
        if metadata.target_hardware_id != base_manifest.target_hardware_id:
            failures.append(f"base {name} artifact targets another GPU")
        if metadata.adapter_policy != "base":
            failures.append(f"base {name} artifact is already adapted")
        if metadata.dataset_fingerprint != base_manifest.dataset_gate.dataset_fingerprint:
            failures.append(f"base {name} artifact uses another dataset")
        if metadata.split_fingerprint != base_manifest.dataset_gate.split_fingerprint:
            failures.append(f"base {name} artifact uses another grouped split")
        if artifact.normalization is None:
            failures.append(f"base {name} artifact embeds no workload normalization")
        elif (
            metadata.workload_normalization_sha256 != artifact.normalization.sha256
            or metadata.normalization_sha256 != artifact.normalization.sha256
        ):
            failures.append(f"base {name} artifact has inconsistent workload normalization")
    if (
        teacher.normalization is not None
        and student.normalization is not None
        and teacher.normalization.sha256 != student.normalization.sha256
    ):
        failures.append("base teacher and student workload normalizers differ")
    if failures:
        raise ValueError("; ".join(failures))
    assert teacher.normalization is not None
    lineage = BaseTransferLineageV3(
        base_training_manifest_sha256=sha256_file(base_manifest.path),
        base_dataset_fingerprint=base_manifest.dataset_gate.dataset_fingerprint,
        base_split_fingerprint=base_manifest.dataset_gate.split_fingerprint,
        base_teacher_artifact_sha256=teacher.sha256,
        base_student_artifact_sha256=student.sha256,
        base_teacher_embeddings_sha256=sha256_file(embeddings_path),
        workload_normalization_sha256=teacher.normalization.sha256,
    )
    lineage.validate()
    return lineage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-labels", type=Path, required=True)
    parser.add_argument("--base-training-manifest", type=Path, required=True)
    parser.add_argument("--base-teacher-artifact", type=Path, required=True)
    parser.add_argument("--base-student-artifact", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-subset", type=Path)
    parser.add_argument("--active-scores", type=Path)
    args = parser.parse_args(argv)
    target = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    prior = (
        json.loads(args.prior_subset.read_text(encoding="utf-8"))
        if args.prior_subset
        else None
    )
    scores = (
        {
            str(key): float(value)
            for key, value in json.loads(
                args.active_scores.read_text(encoding="utf-8")
            ).items()
        }
        if args.active_scores
        else None
    )
    request = TransferSubsetRequestV3(
        base_hardware_id=str(
            target.get("base_hardware_id", "nvidia_a10g_24gb_aws_g5")
        ),
        target_hardware_id=str(target["target_hardware_id"]),
        label_budget=int(target["label_budget"]),
        active_learning=args.active_scores is not None,
    )
    accepted_rows = read_jsonl(args.accepted_labels)
    if len(accepted_rows) != 18_000:
        raise ValueError("production transfer selection requires exactly 18,000 A10 labels")
    base_manifest = TrainingManifestV3.load(args.base_training_manifest)
    base_lineage = build_base_transfer_lineage(
        base_manifest,
        base_teacher_artifact=args.base_teacher_artifact,
        base_student_artifact=args.base_student_artifact,
        embeddings_path=args.embeddings,
    )
    rows = enrich_frozen_rows(accepted_rows, base_manifest)
    payload = select_transfer_subset(
        rows,
        read_embeddings(args.embeddings),
        request,
        base_lineage=base_lineage,
        prior_manifest=prior,
        active_scores=scores,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "subset_sha256": payload["subset_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
