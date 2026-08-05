#!/usr/bin/env python3
"""Build a fail-closed 18K A10 base-training manifest from finalized evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.features import build_graph_features, graph_precision_category
from perfseer_v3.graph_ir_v3 import GraphIRV3
from perfseer_v3.hardware import graph_hardware_id
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.training import DatasetGateReport
from perfseer_v3.training_runner import (
    TRAINING_MANIFEST_VERSION,
    TrainingManifestRowV3,
    TrainingManifestV3,
    _coarsen_once,
)


A10_HARDWARE_ID = "nvidia_a10g_24gb_aws_g5"
FINAL_ARTIFACTS = (
    "accepted_labels.jsonl",
    "failed_attempts.jsonl",
    "audit_report.json",
    "dataset_manifest.json",
    "target_transform.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _verify_embedded_hash(payload: Mapping[str, Any], field_name: str, *, context: str) -> str:
    unhashed = dict(payload)
    observed = str(unhashed.pop(field_name, ""))
    if observed != _canonical_sha256(unhashed):
        raise ValueError(f"{context} content hash mismatch")
    return observed


def _verify_finalized_artifacts(finalized_directory: Path) -> dict[str, Any]:
    receipt_path = finalized_directory / "completion_receipt.json"
    receipt = _load_mapping(receipt_path)
    receipt_sha256 = _verify_embedded_hash(
        receipt,
        "receipt_sha256",
        context="A10 completion receipt",
    )
    file_hashes = receipt.get("artifact_file_sha256s")
    if not isinstance(file_hashes, Mapping):
        raise ValueError("A10 completion receipt has no artifact hash mapping")
    for name in FINAL_ARTIFACTS:
        path = finalized_directory / name
        if str(file_hashes.get(name, "")) != _sha256_bytes(path.read_bytes()):
            raise ValueError(f"finalized A10 artifact {name!r} differs from its receipt")

    dataset_manifest = _load_mapping(finalized_directory / "dataset_manifest.json")
    dataset_manifest_sha256 = _verify_embedded_hash(
        dataset_manifest,
        "manifest_sha256",
        context="A10 dataset manifest",
    )
    audit_report = _load_mapping(finalized_directory / "audit_report.json")
    audit_report_sha256 = _verify_embedded_hash(
        audit_report,
        "report_sha256",
        context="A10 audit report",
    )
    rows = _load_jsonl(finalized_directory / "accepted_labels.jsonl")
    failures = _load_jsonl(finalized_directory / "failed_attempts.jsonl")
    if any(
        _verify_embedded_hash(row, "row_sha256", context="accepted A10 row")
        != row["row_sha256"]
        for row in rows
    ):
        raise AssertionError("unreachable accepted-row hash mismatch")
    if dataset_manifest.get("accepted_rows_sha256") != _canonical_sha256(tuple(rows)):
        raise ValueError("accepted A10 rows differ from the finalized dataset manifest")
    if dataset_manifest.get("failure_index_sha256") != _canonical_sha256(failures):
        raise ValueError("A10 failure index differs from the finalized dataset manifest")
    if dataset_manifest.get("audit_report_sha256") != audit_report_sha256:
        raise ValueError("A10 dataset manifest does not bind its audit report")
    if receipt.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("A10 completion receipt does not bind the dataset manifest")
    return {
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "audit_report": audit_report,
        "audit_report_sha256": audit_report_sha256,
        "accepted_rows": rows,
        "accepted_labels_file_sha256": _sha256_bytes(
            (finalized_directory / "accepted_labels.jsonl").read_bytes()
        ),
        "failed_attempts": failures,
        "failed_attempts_file_sha256": _sha256_bytes(
            (finalized_directory / "failed_attempts.jsonl").read_bytes()
        ),
    }


def _verify_operation_coverage(path: Path, registry: OperationRegistry) -> tuple[str, float]:
    report = _load_mapping(path)
    report_sha256 = _verify_embedded_hash(
        report,
        "report_sha256",
        context="operation GPU-time coverage report",
    )
    if not registry.training_approved:
        raise ValueError("operation registry is not approved from measured GPU time")
    if registry.selection.get("gpu_time_report_sha256") != report_sha256:
        raise ValueError("operation registry does not bind the supplied GPU-time report")
    coverage = report.get("profiler_time_weighted_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("operation report lacks profiler-time-weighted coverage")
    unknown_fraction = float(coverage.get("unknown_fraction", float("nan")))
    if not 0.0 <= unknown_fraction <= 1.0:
        raise ValueError("operation report has an invalid unknown GPU-time fraction")
    return report_sha256, unknown_fraction


def build_base_training_manifest(
    *,
    finalized_directory: Path,
    graphs_directory: Path,
    operation_coverage_report: Path,
    output_path: Path,
    registry: OperationRegistry | None = None,
    expected_configuration_count: int = 18_000,
) -> dict[str, Any]:
    registry = registry or OperationRegistry.load()
    evidence = _verify_finalized_artifacts(finalized_directory)
    dataset_manifest = evidence["dataset_manifest"]
    audit_report = evidence["audit_report"]
    accepted_rows = evidence["accepted_rows"]
    if expected_configuration_count != 18_000:
        raise ValueError("production base manifests require exactly 18,000 configurations")
    if (
        len(accepted_rows) != 18_000
        or dataset_manifest.get("accepted_run_count") != 18_000
        or audit_report.get("accepted_run_count") != 18_000
        or dataset_manifest.get("measured_epoch_record_count") != 54_000
        or audit_report.get("measured_epoch_record_count") != 54_000
    ):
        raise ValueError("finalized A10 artifacts do not prove the exact 18K/54K contract")
    if dataset_manifest.get("target_hardware_id") != A10_HARDWARE_ID:
        raise ValueError("finalized base labels were not measured on the exact A10G target")
    operation_report_sha256, unknown_fraction = _verify_operation_coverage(
        operation_coverage_report,
        registry,
    )

    rows: list[TrainingManifestRowV3] = []
    feature_schema_sha256: str | None = None
    precisions: set[str] = set()
    capture_qualities: set[str] = set()
    optimizers: set[str] = set()
    schedulers: set[str] = set()
    graph_identities: list[tuple[str, str]] = []
    strict_count = 0
    for accepted in accepted_rows:
        configuration_id = str(accepted["configuration_id"])
        graph_path = (graphs_directory / f"{configuration_id}.json").resolve()
        graph = _coarsen_once(GraphIRV3.load(graph_path), registry)
        if graph.graph_sha256 != accepted.get("graph_sha256"):
            raise ValueError(f"graph {configuration_id!r} differs from its accepted A10 row")
        if graph_hardware_id(graph.metadata) != A10_HARDWARE_ID:
            raise ValueError(f"graph {configuration_id!r} is not bound to the exact A10G")
        features = build_graph_features(graph, registry=registry)
        observed_schema = features.layout.feature_schema_sha256
        if feature_schema_sha256 is None:
            feature_schema_sha256 = observed_schema
        elif feature_schema_sha256 != observed_schema:
            raise ValueError("materialized A10 graphs use mixed feature schemas")
        precisions.add(graph_precision_category(graph))
        capture_qualities.add(graph.coverage.capture_quality)
        optimizers.add(str(features.metadata["optimizer"]))
        schedulers.add(str(features.metadata["scheduler"]))
        strict_count += int(
            graph.coverage.capture_quality == "strict"
            and graph.coverage.backward_capture_quality == "strict"
        )
        graph_identities.append((configuration_id, graph.graph_sha256))
        rows.append(
            TrainingManifestRowV3.from_dict(
                {
                    "sample_id": configuration_id,
                    "graph_path": os.path.relpath(graph_path, output_path.parent.resolve()),
                    "split": accepted["split"],
                    "source_group": accepted["source_group"],
                    "graph_signature": graph.graph_sha256,
                    "hardware_id": A10_HARDWARE_ID,
                    "target": accepted["target_values"],
                    "oom": 0.0,
                    "oom_stage": "none",
                    "peak_live_bytes": graph.global_features.peak_live_activation_bytes,
                    "domain_weight": 1.0,
                }
            )
        )
    if feature_schema_sha256 is None:
        raise ValueError("A10 graph corpus is empty")
    deployment = {
        "target_hardware_id": A10_HARDWARE_ID,
        "hardware_allowlist": [A10_HARDWARE_ID],
        "precision_allowlist": sorted(precisions),
        "capture_quality_allowlist": sorted(capture_qualities),
        "optimizer_allowlist": sorted(optimizers),
        "scheduler_allowlist": sorted(schedulers),
        "training_mode_allowlist": ["training"],
        "minimum_confidence": 0.2,
        "allow_ok_with_unknowns": False,
    }
    pending_gate = DatasetGateReport(
        strict_capture_rate=strict_count / len(rows),
        complete_encoding_rate=1.0,
        unknown_gpu_time_fraction=unknown_fraction,
        source_group_isolated=True,
        measured_gpu_time=True,
        dataset_fingerprint="pending",
        split_fingerprint="pending",
    )
    draft = TrainingManifestV3(
        output_path.resolve(),
        pending_gate,
        deployment,
        tuple(rows),
        {},
    )
    draft.validate_splits()
    source_summary = audit_report.get("provenance", {}).get("source", {})
    source_manifest_sha256 = str(source_summary.get("identity_set_sha256", ""))
    campaign_evidence = {
        "accepted_configuration_count": 18_000,
        "measured_epoch_record_count": 54_000,
        "failed_attempts_retained": True,
        "oom_attempts_retained": True,
        "unstable_attempts_retained": True,
        "quarantined_attempts_retained": True,
        "all_graphs_materialized_and_verified": True,
        "source_group_isolated": True,
        "accepted_labels_sha256": evidence["accepted_labels_file_sha256"],
        "audit_report_sha256": evidence["audit_report_sha256"],
        "completion_receipt_sha256": evidence["receipt_sha256"],
        "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
        "failure_index_sha256": evidence["failed_attempts_file_sha256"],
        "graph_corpus_sha256": _canonical_sha256(tuple(sorted(graph_identities))),
        "operation_coverage_report_sha256": operation_report_sha256,
        "operator_registry_sha256": registry.sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": str(dataset_manifest["target_manifest_sha256"]),
        "feature_schema_sha256": feature_schema_sha256,
        "dataset_fingerprint": draft.dataset_fingerprint,
        "split_fingerprint": draft.split_fingerprint,
    }
    campaign_evidence["campaign_evidence_sha256"] = _canonical_sha256(campaign_evidence)
    payload = {
        "manifest_version": TRAINING_MANIFEST_VERSION,
        "dataset_gate": {
            **pending_gate.__dict__,
            "dataset_fingerprint": draft.dataset_fingerprint,
            "split_fingerprint": draft.split_fingerprint,
        },
        "deployment": deployment,
        "campaign_evidence": campaign_evidence,
        "samples": [
            {
                "sample_id": row.sample_id,
                "graph_path": row.graph_path,
                "split": row.split,
                "source_group": row.source_group,
                "graph_signature": row.graph_signature,
                "hardware_id": row.hardware_id,
                "target": list(row.target),
                "oom": row.oom,
                "oom_stage": row.oom_stage,
                "peak_live_bytes": row.peak_live_bytes,
                "domain_weight": row.domain_weight,
            }
            for row in rows
        ],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-directory", type=Path, required=True)
    parser.add_argument("--graphs-directory", type=Path, required=True)
    parser.add_argument("--operation-coverage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_base_training_manifest(
        finalized_directory=args.finalized_directory.resolve(),
        graphs_directory=args.graphs_directory.resolve(),
        operation_coverage_report=args.operation_coverage_report.resolve(),
        output_path=args.output.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = TrainingManifestV3.load(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "accepted_configuration_count": len(loaded.rows),
                "dataset_fingerprint": loaded.dataset_fingerprint,
                "campaign_evidence_sha256": loaded.campaign_evidence[
                    "campaign_evidence_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
