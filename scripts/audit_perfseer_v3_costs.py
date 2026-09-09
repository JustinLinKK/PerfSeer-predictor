"""Audit v3 cost-formula resolution and selective-decomposition evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json, sha256_bytes
from perfseer_v3.capture_export import capture_export
from perfseer_v3.coverage_corpus import representative_source_cases
from perfseer_v3.op_registry import (
    OperationRegistry,
    SEMANTIC_PRESERVE_FAMILIES,
    SUPPORTED_COST_FORMULAS,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_cost_decomposition_audit.json",
    )
    args = parser.parse_args(argv)

    registry = OperationRegistry.load()
    formula_counts = Counter(rule.cost_formula for rule in registry.rules)
    decomposition_counts = Counter(rule.decomposition for rule in registry.rules)
    unsupported_formula_rules = [
        rule.raw
        for rule in registry.rules
        if rule.cost_formula not in SUPPORTED_COST_FORMULAS
    ]
    semantic_preservation_violations = [
        rule.raw
        for rule in registry.rules
        if rule.family in SEMANTIC_PRESERVE_FAMILIES
        and rule.decomposition != "preserve"
    ]

    captures: list[dict[str, Any]] = []
    capture_failures: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    byte_method_counts: Counter[str] = Counter()
    known_nodes_without_flop_formula: list[dict[str, str]] = []
    requested_decompositions: Counter[str] = Counter()
    registered_decompositions: Counter[str] = Counter()
    pre_tensor_nodes = 0
    post_tensor_nodes = 0
    for case in representative_source_cases():
        model, positional, keyword = case.build()
        result = capture_export(model, positional, keyword, registry=registry)
        if not result.success:
            capture_failures.append(
                {
                    "case_id": case.case_id,
                    "failures": [failure.to_dict() for failure in result.failures],
                }
            )
            continue
        assert result.graph is not None
        graph = result.graph
        decomposition = graph.metadata["selective_decomposition"]
        pre_tensor_nodes += int(decomposition["pre_tensor_nodes"])
        post_tensor_nodes += int(decomposition["post_tensor_nodes"])
        requested_decompositions.update(decomposition["requested_targets"])
        registered_decompositions.update(
            decomposition["registered_decomposition_targets"]
        )
        for node in graph.nodes:
            method_counts[node.flops.method] += 1
            byte_method_counts[node.bytes_read.method] += 1
            byte_method_counts[node.bytes_written.method] += 1
            resolved = registry.resolve(node.raw_target)
            if (
                resolved.is_known
                and resolved.cost_formula != "unknown"
                and node.flops.method == "unknown"
            ):
                known_nodes_without_flop_formula.append(
                    {
                        "case_id": case.case_id,
                        "raw_target": node.raw_target,
                        "cost_formula": resolved.cost_formula,
                    }
                )
        captures.append(
            {
                "case_id": case.case_id,
                "capture_quality": graph.coverage.capture_quality,
                "pre_tensor_nodes": decomposition["pre_tensor_nodes"],
                "post_tensor_nodes": decomposition["post_tensor_nodes"],
                "requested_decomposition_targets": decomposition[
                    "requested_targets"
                ],
                "registered_decomposition_targets": decomposition[
                    "registered_decomposition_targets"
                ],
            }
        )

    report: dict[str, Any] = {
        "report_version": "perfseer_v3_cost_decomposition_audit_v1",
        "operator_registry_sha256": registry.sha256,
        "registry_rule_count": len(registry.rules),
        "cost_formula_counts": dict(sorted(formula_counts.items())),
        "decomposition_policy_counts": dict(sorted(decomposition_counts.items())),
        "unsupported_formula_rules": unsupported_formula_rules,
        "semantic_preservation_violations": semantic_preservation_violations,
        "capture_case_count": len(captures),
        "capture_failures": capture_failures,
        "pre_tensor_nodes": pre_tensor_nodes,
        "post_tensor_nodes": post_tensor_nodes,
        "flop_estimate_method_counts": dict(sorted(method_counts.items())),
        "byte_estimate_method_counts": dict(sorted(byte_method_counts.items())),
        "known_nodes_without_flop_formula": sorted(
            known_nodes_without_flop_formula,
            key=lambda row: (row["case_id"], row["raw_target"]),
        ),
        "requested_decomposition_target_counts": dict(
            sorted(requested_decompositions.items())
        ),
        "registered_decomposition_target_counts": dict(
            sorted(registered_decompositions.items())
        ),
        "captures": captures,
        "structural_audit_passed": not (
            unsupported_formula_rules
            or semantic_preservation_violations
            or capture_failures
            or byte_method_counts["unknown"]
        ),
    }
    report["report_sha256"] = sha256_bytes(
        canonical_json(report).encode("utf-8")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"cases={len(captures)} failures={len(capture_failures)} "
        f"structural_passed={report['structural_audit_passed']} "
        f"sha256={report['report_sha256']}"
    )
    return 0 if report["structural_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
