#!/usr/bin/env python3
"""Generate encoder/model requirement evidence and compatibility matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.features import feature_layout
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.schema import (
    CAPTURE_BACKENDS,
    CAPTURE_MODES,
    DTYPES,
    DYNAMIC_SHAPE_POLICIES,
    EDGE_ALIAS_CLASSES,
    EDGE_DYNAMIC_QUALITIES,
    FEATURE_QUALITIES,
    LAYOUTS,
    OPERATOR_BACKENDS,
    OPTIMIZERS,
    PHASE_TRANSITIONS,
)


DEFAULT_COVERAGE = ROOT / "reports" / "v2_operation_coverage.json"
DEFAULT_GPU_TIME = ROOT / "reports" / "perfseer_v3_supported_corpus_gpu_time.json"
DEFAULT_CAPACITY = ROOT / "reports" / "perfseer_v3_capacity_study.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_rows(
    coverage: dict[str, Any],
    registry: OperationRegistry,
) -> list[dict[str, Any]]:
    rows = []
    for case in coverage["cases"]:
        raw_operations = list(case["raw_operations"])
        resolved = [registry.resolve(raw) for raw in raw_operations]
        exact_count = sum(operation.exact_id > 0 for operation in resolved)
        known_count = sum(operation.is_known for operation in resolved)
        unknown = sorted(
            {operation.raw_target for operation in resolved if not operation.is_known}
        )
        structural = bool(
            case["complete_export"]
            and case["encoded_tensor_node_count"] == case["tensor_node_count"]
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "architecture_source_family": case["family"],
                "modality": case["modality"],
                "capture_quality": case["capture_quality"],
                "strict_capture": bool(case["strict_export_success"]),
                "tensor_operation_nodes": int(case["tensor_node_count"]),
                "structurally_encodable": structural,
                "structural_support_path": (
                    "exact_or_family_or_generic_hash" if structural else "capture_failed"
                ),
                "known_registry_occurrences": known_count,
                "exact_registry_occurrences": exact_count,
                "exact_registry_fraction": exact_count / max(1, len(resolved)),
                "exactly_covered": exact_count == len(resolved),
                "unknown_or_custom_operations": unknown,
                "accuracy_validated": False,
                "accuracy_evidence": "production held-out prediction records unavailable",
            }
        )
    return rows


def _family_rows(
    coverage: dict[str, Any],
    gpu_time: dict[str, Any],
    registry: OperationRegistry,
) -> list[dict[str, Any]]:
    occurrences: Counter[str] = Counter()
    unknown_occurrences: Counter[str] = Counter()
    for raw, count in coverage["raw_operation_counts"].items():
        resolved = registry.resolve(raw)
        occurrences[resolved.family] += int(count)
        if not resolved.is_known:
            unknown_occurrences[resolved.family] += int(count)
    measured_ms: defaultdict[str, float] = defaultdict(float)
    unknown_measured_ms: defaultdict[str, float] = defaultdict(float)
    for raw, duration in gpu_time.get("profiler_time_by_operation", {}).items():
        resolved = registry.resolve(raw)
        measured_ms[resolved.family] += float(duration)
        if not resolved.is_known:
            unknown_measured_ms[resolved.family] += float(duration)
    rules = Counter(rule.family for rule in registry.rules)
    exact_rules = Counter(rule.family for rule in registry.rules if rule.exact_id > 0)
    families = list(registry.families)
    return [
        {
            "family": family,
            "structurally_encodable": True,
            "structural_support_path": (
                "generic_hash" if family == "unknown_or_custom" else "family_embedding"
            ),
            "registry_rule_count": rules[family],
            "exact_registry_rule_count": exact_rules[family],
            "observed_operation_occurrences": occurrences[family],
            "unknown_observed_occurrences": unknown_occurrences[family],
            "measured_profiler_time_ms": measured_ms[family],
            "unknown_measured_profiler_time_ms": unknown_measured_ms[family],
            "accuracy_validated": False,
            "accuracy_evidence": "production family-held-out predictions unavailable",
        }
        for family in families
    ]


def _compatibility_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# PerfSeer v3 model-corpus compatibility matrix",
        "",
        "This report intentionally separates structural encoding from exact registry identity and measured prediction accuracy.",
        "",
        "## Summary",
        "",
        f"- Structurally encodable captured models: {summary['structurally_encodable_models']}/{summary['models']}.",
        f"- Models whose every observed operation has an exact registry ID: {summary['exactly_covered_models']}/{summary['models']}.",
        f"- Accuracy-validated models: {summary['accuracy_validated_models']}/{summary['models']}.",
        f"- Measured profiler-time unknown/custom fraction: {summary['unknown_measured_profiler_time_fraction']:.4%}.",
        "",
        "Accuracy support remains unvalidated until grouped production scheduler labels and held-out predictions exist.",
        "",
        "## Model corpus",
        "",
        "| Case | Source family | Strict | Structural | Exact-only | Exact fraction | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["models"]:
        lines.append(
            f"| {row['case_id']} | {row['architecture_source_family']} | "
            f"{'yes' if row['strict_capture'] else 'no'} | "
            f"{'yes' if row['structurally_encodable'] else 'no'} | "
            f"{'yes' if row['exactly_covered'] else 'no'} | "
            f"{row['exact_registry_fraction']:.1%} | "
            f"{'yes' if row['accuracy_validated'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Operation families",
            "",
            "| Family | Structural path | Rules | Exact IDs | Observed nodes | Measured GPU ms | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["operation_families"]:
        lines.append(
            f"| {row['family']} | {row['structural_support_path']} | "
            f"{row['registry_rule_count']} | {row['exact_registry_rule_count']} | "
            f"{row['observed_operation_occurrences']} | "
            f"{row['measured_profiler_time_ms']:.3f} | no |"
        )
    lines.append("")
    lines.append(f"Report SHA-256: `{payload['report_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def _audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PerfSeer v3 encoder/model requirement audit",
        "",
        "## Outcome",
        "",
        f"Local implementation requirements passed: {payload['summary']['local_passed']}/{payload['summary']['local_requirements']}.",
        f"Production-evidence requirements blocked by missing data/infrastructure: {payload['summary']['production_blocked']}.",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for row in payload["requirements"]:
        evidence = "; ".join(row["evidence"])
        lines.append(f"| {row['requirement']} | {row['status']} | {evidence} |")
    lines.extend(
        [
            "",
            "## Feature contract",
            "",
            f"Schema hash: `{payload['feature_contract']['feature_schema_sha256']}`",
            "",
            f"Ordered widths: `{payload['feature_contract']['ordered_widths']}`",
            "",
            "## Remaining production blockers",
            "",
        ]
    )
    for item in payload["remaining_production_blockers"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Report SHA-256: `{payload['report_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--gpu-time", type=Path, default=DEFAULT_GPU_TIME)
    parser.add_argument("--capacity", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    coverage = _load(args.coverage)
    gpu_time = _load(args.gpu_time)
    capacity = _load(args.capacity)
    registry = OperationRegistry.load()
    layout = feature_layout(registry)
    models = _model_rows(coverage, registry)
    families = _family_rows(coverage, gpu_time, registry)
    total_measured = sum(float(value) for value in gpu_time["profiler_time_by_operation"].values())
    unknown_measured = sum(
        float(value)
        for raw, value in gpu_time["profiler_time_by_operation"].items()
        if not registry.resolve(raw).is_known
    )
    compatibility: dict[str, Any] = {
        "report_version": "perfseer_v3_model_corpus_compatibility_v1",
        "source_reports": {
            str(args.coverage): _sha256(args.coverage),
            str(args.gpu_time): _sha256(args.gpu_time),
        },
        "operator_registry_sha256": registry.sha256,
        "feature_schema_sha256": layout.feature_schema_sha256,
        "summary": {
            "models": len(models),
            "structurally_encodable_models": sum(row["structurally_encodable"] for row in models),
            "exactly_covered_models": sum(row["exactly_covered"] for row in models),
            "accuracy_validated_models": sum(row["accuracy_validated"] for row in models),
            "unknown_measured_profiler_time_fraction": unknown_measured / max(total_measured, 1e-12),
        },
        "models": models,
        "operation_families": families,
    }
    compatibility["report_sha256"] = hashlib.sha256(
        canonical_json(compatibility).encode("utf-8")
    ).hexdigest()

    requirements = [
        {
            "requirement": "versioned canonical graph IR and deterministic hashes",
            "status": "local_pass",
            "evidence": ["graph_ir_v3.py", "build_v3_schema.py", "schema mismatch tests"],
        },
        {
            "requirement": "hierarchical exact/family/hash/phase/dtype/operator encoder",
            "status": "local_pass",
            "evidence": ["model.py HierarchicalNodeEncoder", "identity perturbation tests"],
        },
        {
            "requirement": "typed edge semantics including external tensors, slots, alias and phase",
            "status": "local_pass",
            "evidence": ["features.py typed external self-loops", "edge perturbation tests"],
        },
        {
            "requirement": "hardware/execution global encoder and confidence quality inputs",
            "status": "local_pass",
            "evidence": ["HierarchicalGlobalEncoder", "hardware embedding tests"],
        },
        {
            "requirement": "full training phases, liveness, cost and coarsening representation",
            "status": "local_pass",
            "evidence": ["capture_training.py", "liveness_v3.py", "cost_v3.py", "coarsen_v3.py"],
        },
        {
            "requirement": "existing-trunk control, phase-aware pooling and additive optional heads",
            "status": "local_pass",
            "evidence": ["model.py", "six-output contract and export tests"],
        },
        {
            "requirement": "T0/T1/T2 and S0/S1/S2/S3 capacity sweep",
            "status": "local_pass",
            "evidence": [str(args.capacity), f"{len(capacity['candidates'])} exact parameter counts"],
        },
        {
            "requirement": "gated CUDA/AMP teacher training and representation distillation runner",
            "status": "local_pass",
            "evidence": ["training_runner.py", "run_perfseer_v3_training.py", "smoke verification"],
        },
        {
            "requirement": "versioned artifact, verified CPU export and scheduler fallback",
            "status": "local_pass",
            "evidence": ["artifact.py", "deployment_export.py", "runtime.py"],
        },
        {
            "requirement": "production measured-GPU-time registry approval",
            "status": "production_blocked",
            "evidence": [gpu_time["training_approval_reason"]],
        },
        {
            "requirement": "family-held-out teacher/student prediction and calibration gates",
            "status": "production_blocked",
            "evidence": ["no grouped production scheduler-label prediction corpus"],
        },
        {
            "requirement": "peak predictor training memory/time and matched-v2 deployment ratios",
            "status": "production_blocked",
            "evidence": capacity["production_measurement_status"]["missing"],
        },
    ]
    local = [row for row in requirements if row["status"] == "local_pass"]
    blocked = [row for row in requirements if row["status"] == "production_blocked"]
    audit: dict[str, Any] = {
        "report_version": "perfseer_v3_encoder_model_audit_v1",
        "goal_document": "doc/PerfSeer_v3_encoder_and_model_goal_prompt.md",
        "operator_registry_sha256": registry.sha256,
        "feature_contract": {
            "feature_schema_sha256": layout.feature_schema_sha256,
            "ordered_widths": {
                "node_continuous": len(layout.node_continuous_fields),
                "node_flags": len(layout.node_flag_fields),
                "edge_continuous": len(layout.edge_continuous_fields),
                "edge_flags": len(layout.edge_flag_fields),
                "global_continuous": len(layout.global_continuous_fields),
                "quality": len(layout.quality_fields),
            },
            "categorical_cardinalities": {
                "operator_backends": len(OPERATOR_BACKENDS),
                "feature_qualities": len(FEATURE_QUALITIES),
                "dtypes": len(DTYPES),
                "layouts": len(LAYOUTS),
                "edge_alias_classes": len(EDGE_ALIAS_CLASSES),
                "edge_dynamic_qualities": len(EDGE_DYNAMIC_QUALITIES),
                "phase_transitions": len(PHASE_TRANSITIONS),
                "capture_modes": len(CAPTURE_MODES),
                "capture_backends": len(CAPTURE_BACKENDS),
                "optimizers": len(OPTIMIZERS),
                "dynamic_shape_policies": len(DYNAMIC_SHAPE_POLICIES),
            },
        },
        "capacity_report_sha256": capacity["report_sha256"],
        "compatibility_report_sha256": compatibility["report_sha256"],
        "requirements": requirements,
        "summary": {
            "local_requirements": len(local),
            "local_passed": len(local),
            "production_blocked": len(blocked),
        },
        "remaining_production_blockers": [
            "Authenticate Nautilus and rerun the final CUDA verifier on target infrastructure.",
            "Collect grouped production scheduler labels and measured per-operation GPU time.",
            "Approve a registry revision only from the immutable measured GPU-time report.",
            "Train and select T/S candidates from held-out accuracy, uncertainty, OOM, latency, size, memory, and duration evidence.",
        ],
    }
    audit["report_sha256"] = hashlib.sha256(
        canonical_json(audit).encode("utf-8")
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compatibility_json = args.output_dir / "perfseer_v3_model_corpus_compatibility.json"
    compatibility_md = args.output_dir / "perfseer_v3_model_corpus_compatibility.md"
    audit_json = args.output_dir / "perfseer_v3_encoder_model_audit.json"
    audit_md = args.output_dir / "perfseer_v3_encoder_model_audit.md"
    compatibility_json.write_text(
        json.dumps(compatibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility_md.write_text(_compatibility_markdown(compatibility), encoding="utf-8")
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_md.write_text(_audit_markdown(audit), encoding="utf-8")
    print(
        f"models={len(models)} structural={compatibility['summary']['structurally_encodable_models']} "
        f"accuracy_validated={compatibility['summary']['accuracy_validated_models']} "
        f"audit_sha256={audit['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
