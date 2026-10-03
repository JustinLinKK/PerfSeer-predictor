"""Operation-coverage auditing for legacy FX and strict ``torch.export``."""

from __future__ import annotations

import json
import re
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from .baseline import canonical_json, sha256_bytes
from .capture_export import CaptureOptions, capture_export
from .coverage_corpus import CoverageCase
from .diagnostics import CaptureFailureV3
from .op_registry import OperationRegistry


_FAMILY_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("convolution", "conv"), "convolution"),
    (("scaled_dot_product", "flash_attention", "attention"), "attention"),
    (("addmm", "bmm", "matmul", "mm.", "linear"), "dense_matrix"),
    (("batch_norm", "layer_norm", "group_norm", "instance_norm", "rms_norm"), "normalization"),
    (("softmax", "nll_loss", "cross_entropy", "mse_loss", "binary_cross_entropy"), "loss_probability"),
    (("sum", "mean", "prod", "amax", "amin", "argmax", "argmin", "var", "std", "logsumexp"), "reduction"),
    (("gather", "scatter", "index", "slice", "select", "topk", "sort"), "index_scatter"),
    (("view", "reshape", "permute", "transpose", "clone", "contiguous", "cat", "stack", "split"), "layout_shape"),
    (("relu", "gelu", "silu", "sigmoid", "tanh", "exp", "log", "sqrt", "clamp"), "activation_unary"),
    (("add", "sub", "mul", "div", "pow", "where", "remainder"), "elementwise"),
    (("embedding",), "embedding_sequence"),
    (("dropout", "bernoulli", "rand"), "random_regularization"),
    (("pool", "upsample", "grid_sampler", "pad"), "pool_resample"),
)

# PyG's HashTensor support replaces these two public torch functions at import
# time.  That changes legacy FX diagnostics for unrelated cases according to
# import order, so retain the dispatcher-backed originals for the comparison
# path.  Production v3 export capture is unaffected.
_ORIGINAL_TORCH_INDEX_SELECT = torch._C._VariableFunctions.index_select
_ORIGINAL_TORCH_SELECT = torch._C._VariableFunctions.select


def operation_family(raw_target: str) -> str:
    name = raw_target.lower()
    for hints, family in _FAMILY_HINTS:
        if any(hint in name for hint in hints):
            return family
    return "unknown_or_custom"


@dataclass(frozen=True)
class CaptureFailure:
    case_id: str
    backend: str
    stage: str
    exception_type: str
    message: str
    traceback_tail: tuple[str, ...]


@dataclass(frozen=True)
class CaseAudit:
    case_id: str
    family: str
    modality: str
    eager_success: bool
    strict_export_success: bool
    validated_non_strict_success: bool
    complete_export: bool
    capture_quality: str
    legacy_fx_success: bool
    raw_operations: tuple[str, ...]
    operation_families: tuple[str, ...]
    legacy_operations: tuple[str, ...]
    tensor_node_count: int
    encoded_tensor_node_count: int
    output_bytes_by_operation: dict[str, int]
    flops_by_operation: dict[str, float]
    failure_count: int


def _stable_failure_text(value: str) -> str:
    repo_root = str(Path(__file__).resolve().parents[2])
    value = value.replace(repo_root, "<repo>")
    value = re.sub(
        r"/(?:[^/\s\"']+/)+site-packages/",
        "<site-packages>/",
        value,
    )
    value = re.sub(
        r'File "[^"\n]*/site-packages/',
        'File "<site-packages>/',
        value,
    )
    value = re.sub(
        r'File "/tmp/torch_geometric[^"\n]*\.py"',
        'File "<tmp>/torch_geometric_GENERATED.py"',
        value,
    )
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", value)


def _failure(case_id: str, backend: str, stage: str, exc: BaseException) -> CaptureFailure:
    return CaptureFailure(
        case_id=case_id,
        backend=backend,
        stage=stage,
        exception_type=type(exc).__name__,
        message=_stable_failure_text(str(exc)),
        traceback_tail=tuple(
            _stable_failure_text(line)
            for line in traceback.format_exception(type(exc), exc, exc.__traceback__)[-4:]
        ),
    )


def _capture_failure(case_id: str, failure: CaptureFailureV3) -> CaptureFailure:
    return CaptureFailure(
        case_id=case_id,
        backend=f"{failure.backend}_{failure.mode}",
        stage=failure.stage,
        exception_type=failure.exception_type,
        message=_stable_failure_text(failure.message),
        traceback_tail=tuple(
            _stable_failure_text(line)
            for line in failure.traceback_tail[-4:]
        ),
    )


def _audit_export(
    case: CoverageCase,
    registry: OperationRegistry,
) -> tuple[dict[str, Any], list[CaptureFailure]]:
    failures: list[CaptureFailure] = []
    eager_success = False
    try:
        eager_model, eager_args, eager_kwargs = case.build()
        eager_model.train(case.training)
        with torch.no_grad():
            eager_model(*eager_args, **eager_kwargs)
        eager_success = True
    except Exception as exc:
        failures.append(_failure(case.case_id, "eager", "execute", exc))

    raw_operations: list[str] = []
    families: list[str] = []
    bytes_by_op: Counter[str] = Counter()
    flops_by_op: Counter[str] = Counter()
    tensor_nodes = 0
    encoded_nodes = 0
    strict_success = False
    non_strict_success = False
    capture_quality = "failed"
    if eager_success:
        model, args, kwargs = case.build()
        result = capture_export(
            model,
            args,
            kwargs,
            dynamic_shapes=case.dynamic_shapes,
            registry=registry,
            options=CaptureOptions(training_mode=case.training),
        )
        failures.extend(_capture_failure(case.case_id, failure) for failure in result.failures)
        if result.graph is not None:
            graph = result.graph
            capture_quality = graph.coverage.capture_quality
            strict_success = capture_quality == "strict"
            non_strict_success = capture_quality == "non_strict_validated"
            tensor_nodes = graph.coverage.tensor_nodes_seen
            encoded_nodes = graph.coverage.tensor_nodes_encoded
            for node in graph.nodes:
                raw_operations.append(node.raw_target)
                families.append(node.family)
                bytes_by_op[node.raw_target] += node.output_bytes
                flops_by_op[node.raw_target] += node.flops.value
    return (
        {
            "eager_success": eager_success,
            "strict_export_success": strict_success,
            "validated_non_strict_success": non_strict_success,
            "complete_export": (
                (strict_success or non_strict_success)
                and tensor_nodes == encoded_nodes
            ),
            "capture_quality": capture_quality,
            "raw_operations": tuple(raw_operations),
            "operation_families": tuple(families),
            "tensor_node_count": tensor_nodes,
            "encoded_tensor_node_count": encoded_nodes,
            "output_bytes_by_operation": dict(sorted(bytes_by_op.items())),
            "flops_by_operation": dict(sorted(flops_by_op.items())),
        },
        failures,
    )


def _audit_legacy_fx(case: CoverageCase) -> tuple[tuple[str, ...], list[CaptureFailure]]:
    previous_index_select = torch.index_select
    previous_select = torch.select
    try:
        from perfseer_source_converter.converter import _fx_to_networkx

        model, args, kwargs = case.build()
        torch.index_select = _ORIGINAL_TORCH_INDEX_SELECT
        torch.select = _ORIGINAL_TORCH_SELECT
        if kwargs:
            raise ValueError("legacy FX comparison does not support keyword inputs")
        model.train(case.training)
        traced = symbolic_trace(model)
        with torch.no_grad():
            ShapeProp(traced).propagate(*args)
        graph = _fx_to_networkx(traced)
        operations = tuple(str(data["feature"]["type"]) for _, data in graph.nodes(data=True))
        return operations, []
    except Exception as exc:
        return (), [_failure(case.case_id, "legacy_fx", "capture_or_convert", exc)]
    finally:
        torch.index_select = previous_index_select
        torch.select = previous_select


def audit_case(
    case: CoverageCase,
    *,
    registry: OperationRegistry | None = None,
) -> tuple[CaseAudit, tuple[CaptureFailure, ...]]:
    registry = registry or OperationRegistry.load()
    export_data, failures = _audit_export(case, registry)
    legacy_operations, legacy_failures = _audit_legacy_fx(case)
    failures.extend(legacy_failures)
    audit = CaseAudit(
        case_id=case.case_id,
        family=case.family,
        modality=case.modality,
        eager_success=bool(export_data["eager_success"]),
        strict_export_success=bool(export_data["strict_export_success"]),
        validated_non_strict_success=bool(
            export_data["validated_non_strict_success"]
        ),
        complete_export=bool(export_data["complete_export"]),
        capture_quality=str(export_data["capture_quality"]),
        legacy_fx_success=not legacy_failures,
        raw_operations=export_data["raw_operations"],
        operation_families=export_data["operation_families"],
        legacy_operations=legacy_operations,
        tensor_node_count=int(export_data["tensor_node_count"]),
        encoded_tensor_node_count=int(export_data["encoded_tensor_node_count"]),
        output_bytes_by_operation=export_data["output_bytes_by_operation"],
        flops_by_operation=export_data["flops_by_operation"],
        failure_count=len(failures),
    )
    return audit, tuple(failures)


def smallest_time_vocabulary(
    gpu_time_by_operation: Mapping[str, float],
    *,
    coverage: float = 0.95,
) -> tuple[str, ...]:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    positive = [(str(op), float(value)) for op, value in gpu_time_by_operation.items() if float(value) > 0]
    positive.sort(key=lambda item: (-item[1], item[0]))
    total = sum(value for _, value in positive)
    if total <= 0:
        return ()
    selected: list[str] = []
    cumulative = 0.0
    for operation, value in positive:
        selected.append(operation)
        cumulative += value
        if cumulative / total >= coverage:
            break
    return tuple(selected)


def audit_corpus(
    cases: Iterable[CoverageCase],
    *,
    exact_known: Iterable[str] | None = None,
    gpu_time_by_operation: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], tuple[CaptureFailure, ...]]:
    registry = OperationRegistry.load()
    audits: list[CaseAudit] = []
    failures: list[CaptureFailure] = []
    raw_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    flop_counts: Counter[str] = Counter()
    if exact_known is None:
        known = {
            target
            for rule in registry.rules
            if rule.exact_id > 0
            for target in (rule.raw, *rule.aliases)
        }
    else:
        known = set(exact_known)
    for case in cases:
        audit, case_failures = audit_case(case, registry=registry)
        audits.append(audit)
        failures.extend(case_failures)
        raw_counts.update(audit.raw_operations)
        family_counts.update(audit.operation_families)
        legacy_counts.update(audit.legacy_operations)
        byte_counts.update(audit.output_bytes_by_operation)
        flop_counts.update(audit.flops_by_operation)

    def identity_class(operation: str) -> str:
        if operation in known:
            return "exact_known"
        resolved = registry.resolve(operation)
        family = (
            resolved.family
            if resolved.family != "unknown_or_custom"
            else operation_family(operation)
        )
        if family != "unknown_or_custom":
            return "family_known"
        if resolved.is_custom:
            return "custom"
        return "unknown"

    def weighted_coverage(weights: Mapping[str, float]) -> dict[str, float]:
        result = {
            "total": 0.0,
            "exact_known": 0.0,
            "family_known": 0.0,
            "custom": 0.0,
            "unknown": 0.0,
        }
        for operation, raw_value in weights.items():
            value = max(0.0, float(raw_value))
            result["total"] += value
            result[identity_class(operation)] += value
        total = result["total"]
        result["exact_fraction"] = result["exact_known"] / total if total > 0 else 0.0
        result["family_or_exact_fraction"] = (
            (result["exact_known"] + result["family_known"]) / total if total > 0 else 0.0
        )
        result["custom_fraction"] = result["custom"] / total if total > 0 else 0.0
        result["unknown_fraction"] = result["unknown"] / total if total > 0 else 0.0
        return result

    occurrence_coverage = weighted_coverage(raw_counts)
    occurrence_coverage["occurrence_total"] = occurrence_coverage.pop("total")
    profile_times = {str(key): float(value) for key, value in (gpu_time_by_operation or {}).items()}
    recommended = smallest_time_vocabulary(profile_times) if profile_times else ()

    def sliced_occurrence(field_name: str) -> dict[str, dict[str, float]]:
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for audit in audits:
            grouped[str(getattr(audit, field_name))].update(audit.raw_operations)
        return {
            name: weighted_coverage(counts)
            for name, counts in sorted(grouped.items())
        }

    report: dict[str, Any] = {
        "report_version": "perfseer_v3_operation_coverage_v1",
        "models": len(audits),
        "eager_success_rate": sum(a.eager_success for a in audits) / max(1, len(audits)),
        "strict_export_success_rate": sum(a.strict_export_success for a in audits) / max(1, len(audits)),
        "validated_non_strict_success_rate": (
            sum(a.validated_non_strict_success for a in audits)
            / max(1, len(audits))
        ),
        "complete_graph_success_rate": sum(a.complete_export for a in audits) / max(1, len(audits)),
        "capture_quality_counts": dict(
            sorted(Counter(a.capture_quality for a in audits).items())
        ),
        "legacy_fx_success_rate": sum(a.legacy_fx_success for a in audits) / max(1, len(audits)),
        "tensor_nodes": sum(a.tensor_node_count for a in audits),
        "encoded_tensor_nodes": sum(a.encoded_tensor_node_count for a in audits),
        "unique_raw_operations": len(raw_counts),
        "raw_operation_counts": dict(sorted(raw_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "legacy_operation_counts": dict(sorted(legacy_counts.items())),
        "output_bytes_by_operation": dict(sorted(byte_counts.items())),
        "flops_by_operation": dict(sorted(flop_counts.items())),
        "coverage": occurrence_coverage,
        "flop_weighted_coverage": weighted_coverage(flop_counts),
        "tensor_byte_weighted_coverage": weighted_coverage(byte_counts),
        "profiler_time_weighted_coverage": (
            weighted_coverage(profile_times) if profile_times else None
        ),
        "coverage_by_architecture_family": sliced_occurrence("family"),
        "coverage_by_modality": sliced_occurrence("modality"),
        "profiler_time_by_operation": dict(sorted(profile_times.items())),
        "recommended_exact_vocabulary_95pct_gpu_time": list(recommended),
        "cases": [asdict(audit) for audit in audits],
        "failure_taxonomy": dict(sorted(Counter(f"{f.backend}:{f.exception_type}" for f in failures).items())),
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report).encode("utf-8"))
    return report, tuple(failures)


def coverage_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# PerfSeer v2/v3 Operation Coverage",
        "",
        f"- Cases: {report['models']}",
        f"- Strict export success: {report['strict_export_success_rate']:.1%}",
        f"- Validated non-strict success: {report['validated_non_strict_success_rate']:.1%}",
        f"- Complete graph success: {report['complete_graph_success_rate']:.1%}",
        f"- Legacy FX success: {report['legacy_fx_success_rate']:.1%}",
        f"- Tensor nodes retained: {report['encoded_tensor_nodes']}/{report['tensor_nodes']}",
        f"- Unique raw ATen/custom operations: {report['unique_raw_operations']}",
        f"- Exact/family occurrence coverage: {coverage['family_or_exact_fraction']:.1%}",
        "",
        "## Raw operations",
        "",
        "| Operation | Occurrences | Output bytes |",
        "| --- | ---: | ---: |",
    ]
    byte_counts = report["output_bytes_by_operation"]
    for operation, count in report["raw_operation_counts"].items():
        lines.append(f"| `{operation}` | {count} | {byte_counts.get(operation, 0)} |")
    lines.extend(["", f"Report SHA-256: `{report['report_sha256']}`", ""])
    return "\n".join(lines)


def write_coverage_reports(
    report: Mapping[str, Any],
    failures: Iterable[CaptureFailure],
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "v2_operation_coverage.json"
    markdown_path = root / "v2_operation_coverage.md"
    failure_path = root / "v2_capture_failures.jsonl"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(coverage_markdown(report), encoding="utf-8")
    failure_lines = [json.dumps(asdict(failure), sort_keys=True) for failure in failures]
    failure_path.write_text("\n".join(failure_lines) + ("\n" if failure_lines else ""), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path, "failures": failure_path}
