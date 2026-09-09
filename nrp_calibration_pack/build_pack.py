"""Build the NRP calibration subset and generated model source pack."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch

from perfseer.architecture_schema import FEATURE_SCHEMA_VERSION, NODE_TYPES as ARCH_NODE_TYPES
from nrp_calibration_pack.profile.generated_model_runtime import GraphModel
from nrp_calibration_pack.template_catalog import (
    TE_LOW_PRECISION_TRANSFORMER_FAMILIES,
    build_template_graph,
    iter_template_specs,
    template_family_counts,
)


SEED = 20260617
DEFAULT_TEMPLATE_SEED = SEED
DEFAULT_SUBSET_SIZE = 10000
DEFAULT_PILOT_SUBSET_SIZE = 1000
DEFAULT_PRECISION_SWEEP = ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp", "fp8_te_hybrid")
LOW_PRECISION_FOCUS_CHOICES = ("none", "te_transformer")
LOW_PRECISION_FOCUS_PRECISIONS = ("fp8_te_hybrid", "nvfp4_te")
PRECISION_ALIASES = {
    "fp32": "fp32_ieee",
    "float32": "fp32_ieee",
    "fp32_ieee": "fp32_ieee",
    "tf32": "tf32",
    "bf16": "bf16_amp",
    "bf16_amp": "bf16_amp",
    "fp16": "fp16_amp",
    "float16": "fp16_amp",
    "fp16_amp": "fp16_amp",
    "fp8": "fp8_te_hybrid",
    "fp8_te": "fp8_te_hybrid",
    "fp8_te_hybrid": "fp8_te_hybrid",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "fp4": "nvfp4_te",
    "nvfp4": "nvfp4_te",
    "nvfp4_te": "nvfp4_te",
}
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
NODE_TYPES = ARCH_NODE_TYPES
PURE_FAMILIES = ("mobilenet", "vggnet", "resnext", "densenet", "googlenet")
RESERVE_FRACTION = 0.50
PER_BATCH_RESERVE_FRACTION = 0.75
SIZE_FIELDS = (
    "node_count",
    "edge_count",
    "dag_depth",
    "branch_count",
    "join_count",
    "total_flops",
    "total_memory",
    "total_params",
    "max_tensor_size",
    "train_time",
    "infer_time",
)
SIZE_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
REPORT_SIZE_FIELDS = ("node_count", "total_flops", "total_memory", "total_params", "max_tensor_size")
ARCH_FAMILY_FIELDS = ("family_pattern", "layer_count_bucket")
STRUCTURE_FIELDS = (
    "dag_depth_bucket",
    "branch_count_bucket",
    "join_count_bucket",
    "skip_edge_bucket",
    "op_mix",
    "resource_regime",
    "size_risk",
)


@dataclass(frozen=True)
class GraphRecord:
    stem: str
    graph_path: str
    label_path: str
    batch_size: int
    family_tuple: tuple[str, ...]
    node_count: int
    edge_count: int
    dag_depth: int
    branch_count: int
    join_count: int
    total_flops: float
    total_memory: float
    total_params: float
    max_tensor_size: float
    train_util: float
    train_mem: float
    train_time: float
    infer_util: float
    infer_mem: float
    infer_time: float
    op_counts: tuple[int, ...]

    def vector(self) -> list[float]:
        op_total = max(sum(self.op_counts), 1)
        return [
            math.log1p(self.node_count),
            math.log1p(self.edge_count),
            math.log1p(self.dag_depth),
            math.log1p(self.branch_count),
            math.log1p(self.join_count),
            math.log1p(self.total_flops),
            math.log1p(self.total_memory),
            math.log1p(self.total_params),
            math.log1p(self.max_tensor_size),
            math.log1p(max(self.train_time, 0.0)),
            math.log1p(max(self.infer_time, 0.0)),
            math.log1p(max(self.train_mem, 0.0)),
            math.log1p(max(self.infer_mem, 0.0)),
            *[count / op_total for count in self.op_counts],
        ]


@dataclass(frozen=True)
class GeneratedCandidate:
    record: GraphRecord
    input_shape: tuple[int, ...] | None = None
    input_specs: list[dict[str, Any]] | None = None
    node_specs: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


def default_generation_workers() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_generation_workers(value: int | None) -> int:
    if value is None:
        return 1
    if value < 0:
        raise ValueError("--generation-workers must be >= 0")
    return default_generation_workers() if value == 0 else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NRP calibration model sources.")
    parser.add_argument("--out-dir", default="nrp_calibration_pack")
    parser.add_argument("--catalog-mode", choices=("template",), default="template")
    parser.add_argument(
        "--profile-preset",
        choices=("full", "pilot"),
        default="full",
        help=f"Subset-size preset. 'pilot' uses {DEFAULT_PILOT_SUBSET_SIZE} graphs; 'full' uses {DEFAULT_SUBSET_SIZE}.",
    )
    parser.add_argument("--subset-size", type=int, default=None, help="Override the graph count selected by --profile-preset.")
    parser.add_argument(
        "--precision-sweep",
        default=",".join(DEFAULT_PRECISION_SWEEP),
        help="Comma-separated precision_config values to expand into manifest rows.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-mode", choices=("compile", "construct", "meta", "real", "none"), default="compile")
    parser.add_argument(
        "--low-precision-focus",
        choices=LOW_PRECISION_FOCUS_CHOICES,
        default="none",
        help=(
            "Restrict generated templates for low-precision profiling. "
            "'te_transformer' emits non-embedding transformer rows that pass FP8 and NVFP4 TE gates."
        ),
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=0,
        help="Parallel worker processes for graph loading, source generation, and validation. Use 1 for serial; 0 uses all CPUs.",
    )
    parser.add_argument("--smoke-small", action="store_true", help="Prefer tiny CPU-friendly graphs for local smoke packs.")
    parser.add_argument("--force", action="store_true", help="Regenerate manifest/models/subset/report even if they already exist.")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = DEFAULT_TEMPLATE_SEED
    if args.subset_size is None:
        args.subset_size = DEFAULT_PILOT_SUBSET_SIZE if args.profile_preset == "pilot" else DEFAULT_SUBSET_SIZE
    try:
        args.generation_workers = resolve_generation_workers(args.generation_workers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_precision_sweep(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        values = list(DEFAULT_PRECISION_SWEEP)
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in raw if str(part).strip()]
    if not values:
        raise ValueError("precision sweep cannot be empty")
    out: list[str] = []
    for value in values:
        normalized = normalize_precision_config(value)
        if normalized not in out:
            out.append(normalized)
    return tuple(out)


def normalize_precision_config(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key == "bf32":
        raise ValueError("bf32 is ambiguous; use tf32 or bf16_amp")
    if key == "mxfp8":
        raise ValueError("mxfp8 is out of scope for v1; use fp8_te_hybrid or nvfp4_te")
    if key not in PRECISION_ALIASES:
        allowed = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"unknown precision_config {value!r}; expected one of: {allowed}")
    return PRECISION_ALIASES[key]


def load_records(data_root: Path, generation_workers: int | None = 1) -> list[GraphRecord]:
    graph_dir = data_root / "cg" / "cg"
    label_dir = data_root / "label" / "label"
    if not graph_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"expected dataset under {data_root}/cg/cg and {data_root}/label/label")

    workers = resolve_generation_workers(generation_workers)
    graph_label_paths = [
        (graph_path, label_dir / f"{graph_path.stem}.txt")
        for graph_path in sorted(graph_dir.glob("*.pkl"))
        if (label_dir / f"{graph_path.stem}.txt").exists()
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(load_record_from_paths, graph_label_paths, chunksize=16))

    records: list[GraphRecord] = []
    for graph_path, label_path in graph_label_paths:
        records.append(load_record_from_paths((graph_path, label_path)))
    return records


def materialize_template_records(
    out_dir: Path,
    subset_size: int,
    seed: int,
    *,
    force: bool = False,
    families: Iterable[str] | None = None,
) -> list[GraphRecord]:
    catalog_root = out_dir / "template_catalog"
    graph_dir = catalog_root / "cg" / "cg"
    label_dir = catalog_root / "label" / "label"
    if force and catalog_root.exists():
        shutil.rmtree(catalog_root)
    graph_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    records: list[GraphRecord] = []
    for spec in iter_template_specs(subset_size, seed, families=families):
        graph = build_template_graph(spec)
        graph_path = graph_dir / f"{spec.model_stem}.pkl"
        label_path = label_dir / f"{spec.model_stem}.txt"
        with graph_path.open("wb") as fh:
            pickle.dump(graph, fh)
        label_path.write_text("{'train': '0|0|0|0|0|0|0', 'infer': '0|0|0|0|0|0|0'}\n")
        records.append(record_from_template_graph(graph_path, label_path, graph))
    return records


def load_record_from_paths(paths: tuple[Path, Path]) -> GraphRecord:
    graph_path, label_path = paths
    with graph_path.open("rb") as fh:
        graph = nx.DiGraph(pickle.load(fh))
    labels = parse_label(label_path)
    return record_from_graph(graph_path, label_path, graph, labels)


def record_from_graph(graph_path: Path, label_path: Path, graph: nx.DiGraph, labels: dict[str, list[float]]) -> GraphRecord:
    op_counter = {op: 0 for op in NODE_TYPES}
    total_flops = 0.0
    total_memory = 0.0
    total_params = 0.0
    max_tensor_size = 0.0
    batch_size = batch_size_from_stem(graph_path.stem)
    for _node, data in graph.nodes(data=True):
        feat = data.get("feature", {}) or {}
        op = str(feat.get("type", ""))
        if op in op_counter:
            op_counter[op] += 1
        mem = feat.get("memory_info", {}) or {}
        total_flops += float_or_zero(feat.get("flops"))
        total_memory += float_or_zero(mem.get("bytes"))
        total_params += float_or_zero(mem.get("weight_size"))
        max_tensor_size = max(max_tensor_size, float_or_zero(mem.get("output_size")), float_or_zero(mem.get("input_size")))
        if not batch_size:
            batch_size = int(float_or_zero(mem.get("batch_size")))

    try:
        topo = list(nx.topological_sort(graph))
        longest = nx.dag_longest_path_length(graph) if graph.number_of_nodes() else 0
    except Exception:
        topo = list(graph.nodes())
        longest = 0
    _ = topo
    family = family_from_stem(graph_path.stem)
    train = labels["train"]
    infer = labels["infer"]
    return GraphRecord(
        stem=graph_path.stem,
        graph_path=str(graph_path),
        label_path=str(label_path),
        batch_size=int(batch_size),
        family_tuple=family,
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
        dag_depth=int(longest),
        branch_count=sum(1 for node in graph.nodes if graph.out_degree(node) > 1),
        join_count=sum(1 for node in graph.nodes if graph.in_degree(node) > 1),
        total_flops=total_flops,
        total_memory=total_memory,
        total_params=total_params,
        max_tensor_size=max_tensor_size,
        train_util=train[1],
        train_mem=train[6],
        train_time=train[0],
        infer_util=infer[1],
        infer_mem=infer[6],
        infer_time=infer[0],
        op_counts=tuple(op_counter[op] for op in NODE_TYPES),
    )


def record_from_template_graph(graph_path: Path, label_path: Path, graph: nx.DiGraph) -> GraphRecord:
    op_counter = {op: 0 for op in NODE_TYPES}
    total_flops = 0.0
    total_memory = 0.0
    total_params = 0.0
    max_tensor_size = 0.0
    batch_size = 0
    for _node, data in graph.nodes(data=True):
        feat = data.get("feature", {}) or {}
        op = str(feat.get("type", ""))
        if op in op_counter:
            op_counter[op] += 1
        mem = feat.get("memory_info", {}) or {}
        total_flops += float_or_zero(feat.get("flops"))
        total_memory += float_or_zero(mem.get("bytes"))
        total_params += float_or_zero(mem.get("weight_size"))
        max_tensor_size = max(max_tensor_size, float_or_zero(mem.get("output_size")), float_or_zero(mem.get("input_size")))
        if not batch_size:
            batch_size = int(float_or_zero(mem.get("batch_size")))
    try:
        longest = nx.dag_longest_path_length(graph) if graph.number_of_nodes() else 0
    except Exception:
        longest = 0
    family = str((getattr(graph, "graph", {}) or {}).get("architecture_family", "unknown"))
    return GraphRecord(
        stem=graph_path.stem,
        graph_path=str(graph_path),
        label_path=str(label_path),
        batch_size=max(1, int(batch_size)),
        family_tuple=(family,),
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
        dag_depth=int(longest),
        branch_count=sum(1 for node in graph.nodes if graph.out_degree(node) > 1),
        join_count=sum(1 for node in graph.nodes if graph.in_degree(node) > 1),
        total_flops=total_flops,
        total_memory=total_memory,
        total_params=total_params,
        max_tensor_size=max_tensor_size,
        train_util=0.0,
        train_mem=0.0,
        train_time=max(total_flops / 1e9, 1e-6),
        infer_util=0.0,
        infer_mem=0.0,
        infer_time=max(total_flops / 2e9, 1e-6),
        op_counts=tuple(op_counter[op] for op in NODE_TYPES),
    )


def node_types_for_records(records: Iterable[GraphRecord]) -> tuple[str, ...]:
    return NODE_TYPES


def op_count(record: GraphRecord, op_idx: int) -> int:
    return record.op_counts[op_idx] if op_idx < len(record.op_counts) else 0


def record_op_count_map(record: GraphRecord) -> dict[str, int]:
    return {op: op_count(record, idx) for idx, op in enumerate(NODE_TYPES)}


def parse_label(path: Path) -> dict[str, list[float]]:
    data = ast.literal_eval(path.read_text())
    return {phase: [float_or_zero(part) for part in str(data[phase]).split("|")] for phase in ("train", "infer")}


def batch_size_from_stem(stem: str) -> int:
    match = re.match(r"bs(\d+)_", stem)
    return int(match.group(1)) if match else 0


def family_from_stem(stem: str) -> tuple[str, ...]:
    try:
        raw = stem.split("_bnum", 1)[0].split("_s", 1)[1]
        family = ast.literal_eval(raw)
        return tuple(str(item) for item in family)
    except Exception:
        return ()


def float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def select_subset(records: list[GraphRecord], subset_size: int, seed: int = SEED) -> list[GraphRecord]:
    if subset_size >= len(records):
        return sorted(records, key=lambda rec: rec.stem)
    selected: set[int] = set()
    by_batch = {batch: [idx for idx, rec in enumerate(records) if rec.batch_size == batch] for batch in BATCH_BUCKETS}
    quotas = balanced_quotas(subset_size, BATCH_BUCKETS)

    reserve_mandatory(records, selected, subset_size, quotas)
    matrix = np.asarray([rec.vector() for rec in records], dtype=np.float64)

    for batch in BATCH_BUCKETS:
        batch_indices = by_batch.get(batch, [])
        current = [idx for idx in selected if records[idx].batch_size == batch]
        needed = max(0, quotas[batch] - len(current))
        candidates = [idx for idx in batch_indices if idx not in selected]
        for idx in diverse_indices(matrix, candidates, needed, seed + batch):
            selected.add(idx)

    if len(selected) < subset_size:
        candidates = [idx for idx in range(len(records)) if idx not in selected]
        for idx in diverse_indices(matrix, candidates, subset_size - len(selected), seed + 999):
            selected.add(idx)
    elif len(selected) > subset_size:
        selected = trim_to_size(records, matrix, selected, subset_size, seed)

    return [records[idx] for idx in sorted(selected, key=lambda i: (records[i].batch_size, records[i].stem))]


def select_smoke_subset(records: list[GraphRecord], subset_size: int) -> list[GraphRecord]:
    """Pick tiny generated models that can execute quickly on a local CPU."""

    def smoke_key(record: GraphRecord) -> tuple[float, int, int, int, str]:
        structure_bonus = 0
        op_presence = {op for op, count in zip(NODE_TYPES, record.op_counts) if count > 0}
        if "Add" in op_presence or "Concat" in op_presence:
            structure_bonus = 1
        return (
            math.log1p(record.max_tensor_size) + math.log1p(record.total_params),
            structure_bonus,
            record.batch_size,
            record.node_count,
            record.stem,
        )

    selected: list[GraphRecord] = []
    seen_structures: set[tuple[str, ...]] = set()
    for record in sorted(records, key=smoke_key):
        signature = structure_signature(record)
        if signature in seen_structures and len(selected) + len(seen_structures) >= subset_size:
            continue
        selected.append(record)
        seen_structures.add(signature)
        if len(selected) >= subset_size:
            break
    return sorted(selected, key=lambda rec: (rec.batch_size, rec.stem))


def balanced_quotas(total: int, batches: Iterable[int]) -> dict[int, int]:
    batches = tuple(batches)
    base = total // len(batches)
    rem = total % len(batches)
    return {batch: base + (1 if i < rem else 0) for i, batch in enumerate(batches)}


def reserve_mandatory(records: list[GraphRecord], selected: set[int], subset_size: int, quotas: dict[int, int]) -> None:
    reserve_limit = min(subset_size, max(len(BATCH_BUCKETS), int(math.ceil(subset_size * RESERVE_FRACTION))))
    per_batch_limit = {
        batch: max(1, int(math.ceil(quotas.get(batch, 0) * PER_BATCH_RESERVE_FRACTION)))
        for batch in BATCH_BUCKETS
    }

    for family in PURE_FAMILIES:
        target = (family, family, family, family)
        for batch in BATCH_BUCKETS:
            add_closest_to_median(
                records,
                selected,
                [i for i, rec in enumerate(records) if rec.batch_size == batch and rec.family_tuple == target],
                limit=subset_size,
                per_batch_limit=quotas,
            )

    family_counts: dict[tuple[str, ...], int] = {}
    for rec in records:
        if rec.family_tuple and len(set(rec.family_tuple)) > 1:
            family_counts[rec.family_tuple] = family_counts.get(rec.family_tuple, 0) + 1
    mixed_limit = min(len(family_counts), max(1, subset_size // 5))
    for family, _count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))[:mixed_limit]:
        add_closest_to_median(
            records,
            selected,
            [i for i, rec in enumerate(records) if rec.family_tuple == family],
            limit=reserve_limit,
            per_batch_limit=per_batch_limit,
        )

    for batch in BATCH_BUCKETS:
        for op_idx, _op in enumerate(NODE_TYPES):
            add_closest_to_median(
                records,
                selected,
                [i for i, rec in enumerate(records) if rec.batch_size == batch and rec.op_counts[op_idx] > 0],
                limit=reserve_limit,
                per_batch_limit=per_batch_limit,
            )

    for batch in BATCH_BUCKETS:
        batch_indices = [i for i, rec in enumerate(records) if rec.batch_size == batch]
        for field in SIZE_FIELDS:
            add_quantile_representatives(
                records,
                selected,
                batch_indices,
                field,
                SIZE_QUANTILES,
                limit=reserve_limit,
                per_batch_limit=per_batch_limit,
            )

    structure_groups: dict[tuple[int, tuple[str, ...]], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        structure_groups[(rec.batch_size, structure_signature(rec))].append(idx)
    for _key, candidates in ordered_groups_by_rarity_and_size(structure_groups):
        add_closest_to_median(
            records,
            selected,
            candidates,
            limit=reserve_limit,
            per_batch_limit=per_batch_limit,
        )

    coverage_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        for category, key in coverage_keys(rec):
            coverage_groups[(category, key)].append(idx)
    for _key, candidates in ordered_coverage_groups(coverage_groups):
        add_closest_to_median(
            records,
            selected,
            candidates,
            limit=reserve_limit,
            per_batch_limit=per_batch_limit,
        )

    tail_fields = (
        "node_count",
        "edge_count",
        "total_flops",
        "total_memory",
        "train_time",
        "infer_time",
    )
    tail_count = min(16, max(1, subset_size // (len(tail_fields) * 16)))
    for field in tail_fields:
        ranked = sorted(range(len(records)), key=lambda idx: getattr(records[idx], field))
        for idx in ranked[:tail_count] + ranked[-tail_count:]:
            add_index(records, selected, idx, limit=subset_size, per_batch_limit=quotas)


def add_closest_to_median(
    records: list[GraphRecord],
    selected: set[int],
    candidates: list[int],
    *,
    limit: int | None = None,
    per_batch_limit: dict[int, int] | None = None,
) -> bool:
    if not candidates:
        return False
    vectors = np.asarray([records[idx].vector() for idx in candidates], dtype=np.float64)
    scaled = standardize(vectors)
    median = np.median(scaled, axis=0)
    distances = np.linalg.norm(scaled - median, axis=1)
    for local_idx in np.argsort(distances):
        chosen = candidates[int(local_idx)]
        if add_index(records, selected, chosen, limit=limit, per_batch_limit=per_batch_limit):
            return True
    return False


def add_quantile_representatives(
    records: list[GraphRecord],
    selected: set[int],
    candidates: list[int],
    field: str,
    quantiles: Iterable[float],
    *,
    limit: int | None = None,
    per_batch_limit: dict[int, int] | None = None,
) -> None:
    if not candidates:
        return
    ranked = sorted(candidates, key=lambda idx: (float(getattr(records[idx], field)), records[idx].stem))
    last = len(ranked) - 1
    for quantile in quantiles:
        rank = int(round(max(0.0, min(1.0, quantile)) * last))
        add_index(records, selected, ranked[rank], limit=limit, per_batch_limit=per_batch_limit)


def add_index(
    records: list[GraphRecord],
    selected: set[int],
    idx: int,
    *,
    limit: int | None = None,
    per_batch_limit: dict[int, int] | None = None,
) -> bool:
    if idx in selected:
        return False
    if limit is not None and len(selected) >= limit:
        return False
    if per_batch_limit is not None:
        batch = records[idx].batch_size
        if selected_count_for_batch(records, selected, batch) >= per_batch_limit.get(batch, 0):
            return False
    selected.add(idx)
    return True


def selected_count_for_batch(records: list[GraphRecord], selected: set[int], batch: int) -> int:
    return sum(1 for idx in selected if records[idx].batch_size == batch)


def ordered_groups_by_rarity_and_size(groups: dict[tuple[int, tuple[str, ...]], list[int]]) -> list[tuple[tuple[int, tuple[str, ...]], list[int]]]:
    rare_first = sorted(groups.items(), key=lambda item: (len(item[1]), item[0]))
    common_first = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ordered: list[tuple[tuple[int, tuple[str, ...]], list[int]]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for group_list in (rare_first, common_first):
        for key, candidates in group_list:
            if key not in seen:
                ordered.append((key, candidates))
                seen.add(key)
    return ordered


def ordered_coverage_groups(groups: dict[tuple[str, str], list[int]]) -> list[tuple[tuple[str, str], list[int]]]:
    return sorted(groups.items(), key=lambda item: (len(item[1]), item[0]))


def structure_signature(record: GraphRecord) -> tuple[str, ...]:
    counts = record_op_count_map(record)
    op_presence = {op for op, count in counts.items() if count > 0}
    flags = []
    if "Add" in op_presence:
        flags.append("residual")
    if "Concat" in op_presence:
        flags.append("concat")
    if "BatchNormalization" in op_presence:
        flags.append("batchnorm")
    if "MaxPool" in op_presence:
        flags.append("maxpool")
    if "AveragePool" in op_presence:
        flags.append("avgpool")
    if "Gemm" in op_presence:
        flags.append("linear")
    if "Attention" in op_presence or "MultiHeadAttention" in op_presence:
        flags.append("attention")
    if "GRU" in op_presence or "LSTM" in op_presence or "RNN" in op_presence:
        flags.append("recurrent")
    if "GraphMessage" in op_presence or "GraphAttention" in op_presence:
        flags.append("graph_message")
    if not flags:
        flags.append("plain")
    return (
        f"depth:{bucket_value(record.dag_depth, (32, 96, 192))}",
        f"branches:{bucket_value(record.branch_count, (0, 2, 8, 24))}",
        f"joins:{bucket_value(record.join_count, (0, 2, 8, 24))}",
        f"skip:{skip_edge_bucket(record)}",
        "+".join(flags),
    )


def coverage_keys(record: GraphRecord) -> list[tuple[str, str]]:
    return [
        ("family_pattern", family_pattern_key(record)),
        ("layer_count_bucket", layer_count_bucket(record)),
        ("dag_depth_bucket", f"depth:{bucket_value(record.dag_depth, (8, 32, 96, 192))}"),
        ("branch_count_bucket", f"branches:{bucket_value(record.branch_count, (0, 2, 8, 24))}"),
        ("join_count_bucket", f"joins:{bucket_value(record.join_count, (0, 2, 8, 24))}"),
        ("skip_edge_bucket", f"skip:{skip_edge_bucket(record)}"),
        ("op_mix", op_mix_key(record)),
        ("resource_regime", resource_regime_key(record)),
        ("size_risk", size_risk_key(record)),
    ]


def family_pattern_key(record: GraphRecord) -> str:
    family = record.family_tuple
    if not family:
        return "<unknown>"
    unique = tuple(dict.fromkeys(family))
    if len(unique) == 1:
        return f"pure:{unique[0]}"
    return f"mixed:{len(unique)}:{'+'.join(unique)}"


def layer_count_bucket(record: GraphRecord) -> str:
    count = layer_count_from_stem(record.stem)
    if count <= 0:
        return "<unknown>"
    return f"layers:{bucket_value(count, (4, 8, 16, 32, 64))}"


def layer_count_from_stem(stem: str) -> int:
    match = re.search(r"_bnum(\d+)", stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:layers?|blocks?)(\d+)", stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def skip_edge_bucket(record: GraphRecord) -> str:
    sequential_edges = max(record.node_count - 1, 0)
    extra_edges = max(record.edge_count - sequential_edges, 0)
    if extra_edges <= 0:
        return "none"
    if extra_edges <= 2:
        return "few"
    if extra_edges <= 8:
        return "some"
    return "many"


def op_mix_key(record: GraphRecord) -> str:
    counts = record_op_count_map(record)
    total = max(sum(counts.values()), 1)
    conv = (counts.get("Conv", 0) + counts.get("DepthwiseConv", 0) + counts.get("ConvTranspose", 0)) / total
    gemm = counts.get("Gemm", 0) / total
    attention = (counts.get("Attention", 0) + counts.get("MultiHeadAttention", 0)) / total
    recurrent = (counts.get("RNN", 0) + counts.get("GRU", 0) + counts.get("LSTM", 0)) / total
    graph_ops = (counts.get("GraphMessage", 0) + counts.get("GraphAttention", 0)) / total
    pool = (counts.get("AveragePool", 0) + counts.get("MaxPool", 0) + counts.get("GlobalAveragePool", 0)) / total
    join_ops = (counts.get("Concat", 0) + counts.get("Add", 0)) / total
    if conv >= 0.45:
        return "conv_heavy"
    if attention >= 0.15:
        return "attention_heavy"
    if recurrent >= 0.15:
        return "recurrent_heavy"
    if graph_ops >= 0.15:
        return "graph_message_heavy"
    if gemm >= 0.35:
        return "gemm_heavy"
    if pool >= 0.25:
        return "pooling_heavy"
    if join_ops >= 0.15:
        return "concat_add_heavy"
    return "balanced"


def resource_regime_key(record: GraphRecord) -> str:
    if record.total_flops <= 1e6 and record.total_memory <= 1e6 and record.node_count <= 8:
        return "small_overhead"
    intensity = record.total_flops / max(record.total_memory, 1.0)
    if intensity < 4.0:
        return "memory_bound"
    if intensity > 64.0:
        return "compute_bound"
    return "balanced"


def size_risk_key(record: GraphRecord) -> str:
    if record.batch_size >= 128 and record.max_tensor_size >= 1e8:
        return "near_oom_batch"
    if record.max_tensor_size >= 1e8:
        return "very_large_tensor"
    if record.total_params >= 1e8:
        return "very_large_params"
    if record.batch_size >= 128:
        return "large_batch"
    return "ordinary"


def bucket_value(value: float, thresholds: tuple[float, ...]) -> str:
    for idx, threshold in enumerate(thresholds):
        if value <= threshold:
            return str(idx)
    return str(len(thresholds))


def diverse_indices(matrix: np.ndarray, candidates: list[int], count: int, seed: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)
    data = standardize(matrix[candidates])
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=count, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(data)
        chosen: list[int] = []
        used: set[int] = set()
        for cluster_idx in range(count):
            members = np.where(labels == cluster_idx)[0]
            if members.size == 0:
                continue
            center = kmeans.cluster_centers_[cluster_idx]
            distances = np.linalg.norm(data[members] - center, axis=1)
            order = members[np.argsort(distances)]
            for local_idx in order:
                candidate = candidates[int(local_idx)]
                if candidate not in used:
                    chosen.append(candidate)
                    used.add(candidate)
                    break
        if len(chosen) < count:
            for idx in farthest_point_fill(data, candidates, count - len(chosen), used, seed):
                chosen.append(idx)
        return chosen[:count]
    except Exception:
        return farthest_point_fill(data, candidates, count, set(), seed)


def farthest_point_fill(data: np.ndarray, candidates: list[int], count: int, used: set[int], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    remaining = [idx for idx in range(len(candidates)) if candidates[idx] not in used]
    if not remaining:
        return []
    first = int(rng.choice(remaining))
    chosen_local = [first]
    used.add(candidates[first])
    while len(chosen_local) < count and len(used) < len(candidates):
        chosen_data = data[chosen_local]
        distances = np.min(np.linalg.norm(data[:, None, :] - chosen_data[None, :, :], axis=2), axis=1)
        order = np.argsort(-distances)
        for local_idx in order:
            candidate = candidates[int(local_idx)]
            if candidate not in used:
                used.add(candidate)
                chosen_local.append(int(local_idx))
                break
    return [candidates[idx] for idx in chosen_local[:count]]


def trim_to_size(records: list[GraphRecord], matrix: np.ndarray, selected: set[int], size: int, seed: int) -> set[int]:
    selected_list = sorted(selected)
    keep = set(diverse_indices(matrix, selected_list, size, seed + 12345))
    if len(keep) < size:
        for idx in selected_list:
            keep.add(idx)
            if len(keep) == size:
                break
    return keep


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1.0
    return (matrix - mean) / std


def model_source_parts(graph: nx.DiGraph) -> tuple[list[dict[str, Any]], tuple[int, ...], list[dict[str, Any]]]:
    topo = list(nx.topological_sort(graph))
    id_map = {node: idx for idx, node in enumerate(topo)}
    node_specs: list[dict[str, Any]] = []
    for node in topo:
        feat = graph.nodes[node].get("feature", {}) or {}
        node_specs.append(
            {
                "id": id_map[node],
                "original_id": str(node),
                "type": str(feat.get("type", "")),
                "args": clean_json(feat.get("args", {}) or {}),
                "memory_info": clean_json(feat.get("memory_info", {}) or {}),
                "input_index": int((graph.nodes[node].get("input_index", 0) or 0)),
                "preds": [id_map[pred] for pred in graph.predecessors(node)],
            }
        )
    input_specs = infer_input_specs(graph)
    input_shape = tuple(int(dim) for dim in input_specs[0]["shape"])
    return node_specs, input_shape, input_specs


def generate_model_source(model_id: str, record: GraphRecord, graph: nx.DiGraph) -> str:
    node_specs, input_shape, input_specs = model_source_parts(graph)
    return render_model_source(model_id, record, input_shape, node_specs, input_specs)


def render_model_source(
    model_id: str,
    record: GraphRecord,
    input_shape: tuple[int, ...],
    node_specs: list[dict[str, Any]],
    input_specs: list[dict[str, Any]] | None = None,
) -> str:
    input_specs = input_specs or [{"name": "input0", "shape": list(input_shape), "dtype": "float32", "kind": "float"}]
    forward_args = ", ".join(f"input{idx}: torch.Tensor" for idx, _spec in enumerate(input_specs))
    forward_call = ", ".join(f"input{idx}" for idx, _spec in enumerate(input_specs))
    return "\n".join(
        [
            '"""Generated PerfSeer calibration model source."""',
            "",
            "from __future__ import annotations",
            "",
            "import torch",
            "import torch.nn as nn",
            "",
            "try:",
            "    from nrp_calibration_pack.profile.generated_model_runtime import GraphModel",
            "except ModuleNotFoundError:",
            "    import importlib.util",
            "    import sys",
            "    from pathlib import Path",
            "",
            "    _runtime_path = Path(__file__).resolve().parents[1] / 'profile' / 'generated_model_runtime.py'",
            "    _runtime_spec = importlib.util.spec_from_file_location('_nrp_generated_model_runtime', _runtime_path)",
            "    if _runtime_spec is None or _runtime_spec.loader is None:",
            "        raise",
            "    _runtime_module = importlib.util.module_from_spec(_runtime_spec)",
            "    sys.modules.setdefault(_runtime_spec.name, _runtime_module)",
            "    _runtime_spec.loader.exec_module(_runtime_module)",
            "    GraphModel = _runtime_module.GraphModel",
            "",
            f"MODEL_ID = {model_id!r}",
            f"ORIGINAL_STEM = {record.stem!r}",
            f"INPUT_SHAPE = {tuple(input_shape)!r}",
            f"INPUT_SPECS = {json.dumps(input_specs, sort_keys=True)}",
            f"NODE_SPECS = {json.dumps(node_specs, sort_keys=True)}",
            "",
            "",
            "class GeneratedModel(GraphModel):",
            "    def __init__(self) -> None:",
            "        super().__init__(NODE_SPECS)",
            "",
            f"    def forward(self, {forward_args}) -> torch.Tensor:",
            f"        return super().forward({forward_call})",
            "",
            "",
            "def make_model() -> nn.Module:",
            "    return GeneratedModel()",
            "",
        ]
    )


def infer_input_shape(graph: nx.DiGraph) -> tuple[int, int, int, int]:
    roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
    if not roots:
        raise ValueError("graph has no root")
    feat = graph.nodes[roots[0]].get("feature", {}) or {}
    mem = feat.get("memory_info", {}) or {}
    batch = max(1, int(float_or_zero(mem.get("batch_size"))))
    channels = max(1, int(round(float_or_zero(mem.get("input_channels")))))
    height = max(1, int(round(float_or_zero(mem.get("input_h")))))
    width = max(1, int(round(float_or_zero(mem.get("input_w")))))
    return batch, channels, height, width


def infer_input_specs(graph: nx.DiGraph) -> list[dict[str, Any]]:
    raw_specs = (getattr(graph, "graph", {}) or {}).get("input_specs")
    if isinstance(raw_specs, list) and raw_specs:
        return [normalize_input_spec(spec, idx) for idx, spec in enumerate(raw_specs)]
    return [
        {
            "name": "input0",
            "shape": list(infer_input_shape(graph)),
            "dtype": "float32",
            "kind": "float",
        }
    ]


def normalize_input_spec(spec: dict[str, Any], index: int) -> dict[str, Any]:
    shape = [int(dim) for dim in spec.get("shape", [])]
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid input spec shape at index {index}: {shape!r}")
    return {
        "name": str(spec.get("name", f"input{index}")),
        "shape": shape,
        "dtype": str(spec.get("dtype", "float32")),
        "kind": str(spec.get("kind", "float")),
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_pack(
    records: list[GraphRecord],
    all_records: list[GraphRecord],
    out_dir: Path,
    validation_mode: str,
    precision_sweep: Iterable[str] | str | None = None,
    generation_workers: int | None = 1,
    low_precision_focus: str = "none",
) -> tuple[int, int]:
    precision_configs = parse_precision_sweep(precision_sweep)
    low_precision_focus = normalize_low_precision_focus(low_precision_focus)
    workers = resolve_generation_workers(generation_workers)
    sync_runtime_files(out_dir)
    models_dir = out_dir / "models"
    manifest_dir = out_dir / "manifest"
    subset_graph_dir = out_dir / "subset" / "cg" / "cg"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    subset_graph_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "__init__.py").write_text('"""Generated calibration model modules."""\n')

    manifest_path = manifest_dir / "subset_manifest.jsonl"
    model_rows: list[dict[str, Any]] = []
    validation_failures: list[dict[str, str]] = []
    target_size = len(records)
    selected_stems = {record.stem for record in records}
    candidate_pool = list(records) + [
        record
        for record in sorted(all_records, key=lambda rec: (rec.batch_size, rec.stem))
        if record.stem not in selected_stems
    ]
    attempted: set[str] = set()
    candidate_records: list[GraphRecord] = []

    for record in candidate_pool:
        if record.stem in attempted:
            continue
        attempted.add(record.stem)
        candidate_records.append(record)

    for result in prepare_pack_candidates(candidate_records, validation_mode, workers, low_precision_focus):
        if len(model_rows) >= target_size:
            break
        model_id = f"calib_{len(model_rows):04d}"
        if result.error is not None:
            validation_failures.append({"stem": result.record.stem, "model_id": model_id, "error": result.error})
            continue
        if result.node_specs is None or result.input_shape is None or result.input_specs is None:
            validation_failures.append({"stem": result.record.stem, "model_id": model_id, "error": "missing generated candidate payload"})
            continue
        source = render_model_source(model_id, result.record, result.input_shape, result.node_specs, result.input_specs)
        model_path = models_dir / f"{model_id}.py"
        subset_graph_path = subset_graph_dir / f"{model_id}.pkl"
        model_path.write_text(source)
        shutil.copyfile(result.record.graph_path, subset_graph_path)
        metadata = result.metadata or {}
        row = {
            **asdict(result.record),
            "graph_id": model_id,
            "hardware_id": None,
            "original_stem": result.record.stem,
            "original_graph_path": result.record.graph_path,
            "original_label_path": result.record.label_path,
            "model_id": model_id,
            "model_file": f"models/{model_id}.py",
            "subset_graph_file": f"subset/cg/cg/{model_id}.pkl",
            "base_label_file": f"label/label/{model_id}.txt",
            "input_shape": list(result.input_shape),
            "input_specs": clean_json(result.input_specs),
            "feature_schema_version": str(metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
            "architecture_family": str(metadata.get("architecture_family", family_key(result.record.family_tuple))),
            "variant_kind": str(metadata.get("variant_kind", "source_dataset")),
            "variant_signature": str(metadata.get("variant_signature", result.record.stem)),
            "precision_sweep": list(precision_configs),
            "low_precision_focus": low_precision_focus,
        }
        model_rows.append(clean_json(row))

    if len(model_rows) < target_size:
        raise RuntimeError(
            f"validated only {len(model_rows)} generated models out of requested {target_size}; "
            f"{len(validation_failures)} candidates failed validation"
        )

    with manifest_path.open("w") as fh:
        for row in expand_precision_rows(model_rows, precision_configs):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    write_report(
        out_dir / "selection_report.md",
        all_records,
        [record_by_stem(all_records, row["original_stem"]) for row in model_rows],
        validation_failures,
        precision_configs,
        low_precision_focus=low_precision_focus,
    )
    write_coverage_summary(
        out_dir / "coverage_summary.json",
        all_records,
        [record_by_stem(all_records, row["original_stem"]) for row in model_rows],
        validation_failures,
        precision_configs,
        low_precision_focus=low_precision_focus,
    )
    return len(model_rows), len(validation_failures)


def prepare_pack_candidates(
    records: list[GraphRecord],
    validation_mode: str,
    generation_workers: int,
    low_precision_focus: str = "none",
) -> Iterable[GeneratedCandidate]:
    low_precision_focus = normalize_low_precision_focus(low_precision_focus)
    if generation_workers <= 1:
        for record in records:
            yield prepare_pack_candidate(record, validation_mode, low_precision_focus)
        return

    batch_size = max(32, generation_workers * 4)
    executor = ProcessPoolExecutor(max_workers=generation_workers)
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            yield from executor.map(
                prepare_pack_candidate_for_pool,
                [(record, validation_mode, low_precision_focus) for record in batch],
                chunksize=1,
            )
    finally:
        executor.shutdown(cancel_futures=True)


def prepare_pack_candidate_for_pool(args: tuple[GraphRecord, str, str]) -> GeneratedCandidate:
    record, validation_mode, low_precision_focus = args
    return prepare_pack_candidate(record, validation_mode, low_precision_focus)


def prepare_pack_candidate(record: GraphRecord, validation_mode: str, low_precision_focus: str = "none") -> GeneratedCandidate:
    try:
        with Path(record.graph_path).open("rb") as fh:
            graph = nx.DiGraph(pickle.load(fh))
        unsupported = unsupported_ops(graph)
        if unsupported:
            return GeneratedCandidate(record=record, error=f"unsupported ops: {', '.join(unsupported)}")
        node_specs, input_shape, input_specs = model_source_parts(graph)
        focus_reasons = low_precision_focus_reasons(node_specs, low_precision_focus)
        if focus_reasons:
            preview = "; ".join(focus_reasons[:8])
            if len(focus_reasons) > 8:
                preview += f"; {len(focus_reasons) - 8} more"
            return GeneratedCandidate(record=record, error=f"low_precision_focus {low_precision_focus} rejected: {preview}")
        if validation_mode != "none":
            source = render_model_source("calib_validation", record, input_shape, node_specs, input_specs)
            validate_generated_source(source, input_shape, validation_mode, input_specs)
        metadata = dict((getattr(graph, "graph", {}) or {}))
        return GeneratedCandidate(record=record, input_shape=input_shape, input_specs=input_specs, node_specs=node_specs, metadata=metadata)
    except Exception as exc:
        return GeneratedCandidate(record=record, error=repr(exc))


def normalize_low_precision_focus(value: str | None) -> str:
    focus = (value or "none").strip().lower().replace("-", "_")
    if focus not in LOW_PRECISION_FOCUS_CHOICES:
        allowed = ", ".join(LOW_PRECISION_FOCUS_CHOICES)
        raise ValueError(f"unknown low-precision focus {value!r}; expected one of: {allowed}")
    return focus


def low_precision_focus_families(focus: str) -> tuple[str, ...] | None:
    focus = normalize_low_precision_focus(focus)
    if focus == "te_transformer":
        return TE_LOW_PRECISION_TRANSFORMER_FAMILIES
    return None


def low_precision_focus_reasons(node_specs: list[dict[str, Any]], focus: str) -> list[str]:
    focus = normalize_low_precision_focus(focus)
    if focus == "none":
        return []
    if focus == "te_transformer" and not any(str(spec.get("type")) in {"Attention", "MultiHeadAttention"} for spec in node_specs):
        return ["te_transformer focus requires generated Attention or MultiHeadAttention"]
    model = GraphModel(node_specs)
    reasons: list[str] = []
    for precision_config in LOW_PRECISION_FOCUS_PRECISIONS:
        reasons.extend(f"{precision_config}: {reason}" for reason in model.low_precision_unsupported_reasons(precision_config))
    return reasons


def expand_precision_rows(model_rows: Iterable[dict[str, Any]], precision_configs: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = tuple(precision_configs)
    for row in model_rows:
        for index, precision_config in enumerate(configs):
            expanded = dict(row)
            expanded["precision_config"] = precision_config
            expanded["precision_config_index"] = index
            expanded["label_file"] = f"label/label/{row['model_id']}_{precision_config}.txt"
            expanded["profile_point_id"] = f"{row['model_id']}::{precision_config}"
            rows.append(expanded)
    return rows


def unsupported_ops(graph: nx.DiGraph) -> list[str]:
    supported = set(NODE_TYPES)
    return sorted(
        {
            str((data.get("feature", {}) or {}).get("type", ""))
            for _node, data in graph.nodes(data=True)
            if str((data.get("feature", {}) or {}).get("type", "")) not in supported
        }
    )


def write_coverage_summary(
    path: Path,
    all_records: list[GraphRecord],
    selected: list[GraphRecord],
    validation_failures: list[dict[str, str]] | None = None,
    precision_sweep: Iterable[str] | None = None,
    low_precision_focus: str = "none",
) -> None:
    validation_failures = validation_failures or []
    precision_configs = tuple(precision_sweep or DEFAULT_PRECISION_SWEEP)
    low_precision_focus = normalize_low_precision_focus(low_precision_focus)
    report_node_types = node_types_for_records([*all_records, *selected])
    summary = {
        "full_dataset_graphs": len(all_records),
        "selected_graphs": len(selected),
        "manifest_profile_points": len(selected) * len(precision_configs),
        "seed": SEED,
        "default_subset_size": DEFAULT_SUBSET_SIZE,
        "default_pilot_subset_size": DEFAULT_PILOT_SUBSET_SIZE,
        "precision_sweep": list(precision_configs),
        "low_precision_focus": low_precision_focus,
        "validation_exclusions_replaced": len(validation_failures),
        "batch_size_coverage": {
            str(batch): {
                "full": sum(1 for rec in all_records if rec.batch_size == batch),
                "selected": sum(1 for rec in selected if rec.batch_size == batch),
            }
            for batch in BATCH_BUCKETS
        },
        "operator_coverage": {
            op: {
                "full": sum(op_count(rec, op_idx) for rec in all_records),
                "selected": sum(op_count(rec, op_idx) for rec in selected),
            }
            for op_idx, op in enumerate(report_node_types)
        },
        "family_coverage": coverage_counts(
            (family_key(rec.family_tuple) for rec in all_records),
            (family_key(rec.family_tuple) for rec in selected),
        ),
        "structure_coverage": coverage_counts(
            (structure_key(structure_signature(rec)) for rec in all_records),
            (structure_key(structure_signature(rec)) for rec in selected),
        ),
        "architecture_family_coverage": {
            field: coverage_counts(
                (coverage_value(rec, field) for rec in all_records),
                (coverage_value(rec, field) for rec in selected),
            )
            for field in ARCH_FAMILY_FIELDS
        },
        "model_structure_coverage": {
            field: coverage_counts(
                (coverage_value(rec, field) for rec in all_records),
                (coverage_value(rec, field) for rec in selected),
            )
            for field in STRUCTURE_FIELDS
        },
        "size_quantiles": {
            field: quantile_summary(
                [float(getattr(rec, field)) for rec in all_records],
                [float(getattr(rec, field)) for rec in selected],
            )
            for field in REPORT_SIZE_FIELDS
        },
        "validation_failures": validation_failures[:200],
    }
    path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")


def coverage_value(record: GraphRecord, field: str) -> str:
    values = dict(coverage_keys(record))
    return values.get(field, "<unknown>")


def coverage_counts(full_keys: Iterable[str], selected_keys: Iterable[str]) -> dict[str, dict[str, int]]:
    full: dict[str, int] = {}
    selected: dict[str, int] = {}
    for key in full_keys:
        full[key] = full.get(key, 0) + 1
    for key in selected_keys:
        selected[key] = selected.get(key, 0) + 1
    return {
        key: {"full": full[key], "selected": selected.get(key, 0)}
        for key in sorted(full, key=lambda item: (-selected.get(item, 0), -full[item], item))
    }


def quantile_summary(full_values: list[float], selected_values: list[float]) -> dict[str, dict[str, float]]:
    quantiles = (0, 10, 25, 50, 75, 90, 100)
    full = np.asarray(full_values, dtype=np.float64)
    selected = np.asarray(selected_values, dtype=np.float64)
    return {
        f"p{quantile}": {
            "full": float(np.percentile(full, quantile)),
            "selected": float(np.percentile(selected, quantile)),
        }
        for quantile in quantiles
    }


def family_key(family: tuple[str, ...]) -> str:
    return " / ".join(family) if family else "<unknown>"


def structure_key(signature: tuple[str, ...]) -> str:
    return " / ".join(signature)


def record_by_stem(records: list[GraphRecord], stem: str) -> GraphRecord:
    for record in records:
        if record.stem == stem:
            return record
    raise KeyError(stem)


def validate_generated_model(
    model_path: Path,
    input_shape: tuple[int, ...],
    mode: str,
    input_specs: list[dict[str, Any]] | None = None,
) -> None:
    if mode == "compile":
        compile(model_path.read_text(), str(model_path), "exec")
        return
    module_token = re.sub(r"[^A-Za-z0-9_]+", "_", str(model_path.resolve()))
    module_name = f"_nrp_validate_{model_path.stem}_{os.getpid()}_{module_token[-64:]}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        model = module.make_model()
        if mode == "construct":
            return
        model.eval()
        device = torch.device("meta" if mode == "meta" else "cpu")
        model = model.to(device)
        inputs = synthetic_inputs(input_specs or [{"shape": list(input_shape), "dtype": "float32", "kind": "float"}], device)
        with torch.no_grad():
            out = model(*inputs)
        if len(tuple(out.shape)) == 0:
            raise ValueError(f"{model_path} produced scalar output")
    finally:
        sys.modules.pop(module_name, None)


def validate_generated_source(
    source: str,
    input_shape: tuple[int, ...],
    mode: str,
    input_specs: list[dict[str, Any]] | None = None,
) -> None:
    if mode == "compile":
        compile(source, "<generated calibration model>", "exec")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        sync_runtime_files(tmp_root)
        models_dir = tmp_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "calib_validation.py"
        model_path.write_text(source)
        validate_generated_model(model_path, input_shape, mode, input_specs)


def synthetic_inputs(input_specs: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for spec in input_specs:
        shape = tuple(int(dim) for dim in spec.get("shape", []))
        dtype = str(spec.get("dtype", "float32")).lower()
        kind = str(spec.get("kind", "float")).lower()
        if dtype in {"int64", "long"} or kind in {"tokens", "token_ids"}:
            tensors.append(torch.zeros(shape, dtype=torch.long, device=device))
        elif kind == "adjacency":
            base = torch.eye(shape[-1], dtype=torch.float32, device=device)
            tensors.append(base.expand(shape).clone())
        else:
            tensors.append(torch.zeros(shape, dtype=torch.float32, device=device))
    return tuple(tensors)


def write_report(
    path: Path,
    all_records: list[GraphRecord],
    selected: list[GraphRecord],
    validation_failures: list[dict[str, str]] | None = None,
    precision_sweep: Iterable[str] | None = None,
    low_precision_focus: str = "none",
) -> None:
    validation_failures = validation_failures or []
    precision_configs = tuple(precision_sweep or DEFAULT_PRECISION_SWEEP)
    low_precision_focus = normalize_low_precision_focus(low_precision_focus)
    report_node_types = node_types_for_records([*all_records, *selected])
    lines = [
        "# NRP Calibration Subset Selection Report",
        "",
        f"- Full dataset graphs: {len(all_records)}",
        f"- Selected graphs: {len(selected)}",
        f"- Manifest profile points: {len(selected) * len(precision_configs)}",
        f"- Seed: {SEED}",
        f"- Default target size: {DEFAULT_SUBSET_SIZE}",
        f"- Default pilot target size: {DEFAULT_PILOT_SUBSET_SIZE}",
        f"- Precision sweep: {', '.join(precision_configs)}",
        f"- Low-precision focus: {low_precision_focus}",
        f"- Validation exclusions replaced: {len(validation_failures)}",
        "",
        "## Selection Policy",
        "",
        "- Balance the final subset across batch sizes before filling with feature-space diversity.",
        "- Reserve pure-family examples for every batch size where they exist.",
        "- Reserve one representative for mixed architecture-family tuples when the subset budget allows it.",
        "- Reserve operator-presence, topology-signature, model-structure, resource-regime, and size-quantile anchors before diversity fill.",
        "- Expand each selected graph into one manifest row per precision_config.",
        "- Replace generated-source validation failures with the next best eligible candidates.",
        "",
        "## Batch Size Coverage",
        "",
        "| batch | full | selected |",
        "|---:|---:|---:|",
    ]
    for batch in BATCH_BUCKETS:
        lines.append(f"| {batch} | {sum(1 for r in all_records if r.batch_size == batch)} | {sum(1 for r in selected if r.batch_size == batch)} |")
    lines.extend(["", "## Distribution Summary", "", "| metric | full p50 | selected p50 | full p95 | selected p95 |", "|---|---:|---:|---:|---:|"])
    for field in ("node_count", "edge_count", "dag_depth", "branch_count", "join_count", "total_flops", "total_memory", "train_time", "infer_time"):
        full_vals = np.asarray([float(getattr(r, field)) for r in all_records], dtype=np.float64)
        sel_vals = np.asarray([float(getattr(r, field)) for r in selected], dtype=np.float64)
        lines.append(
            f"| {field} | {np.percentile(full_vals, 50):.4g} | {np.percentile(sel_vals, 50):.4g} | "
            f"{np.percentile(full_vals, 95):.4g} | {np.percentile(sel_vals, 95):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Size Quantile Coverage",
            "",
            "| metric | dataset min | selected min | dataset p10 | selected p10 | dataset p50 | selected p50 | dataset p90 | selected p90 | dataset max | selected max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for field in REPORT_SIZE_FIELDS:
        full_vals = np.asarray([float(getattr(r, field)) for r in all_records], dtype=np.float64)
        sel_vals = np.asarray([float(getattr(r, field)) for r in selected], dtype=np.float64)
        lines.append(
            f"| {field} | {np.percentile(full_vals, 0):.4g} | {np.percentile(sel_vals, 0):.4g} | "
            f"{np.percentile(full_vals, 10):.4g} | {np.percentile(sel_vals, 10):.4g} | "
            f"{np.percentile(full_vals, 50):.4g} | {np.percentile(sel_vals, 50):.4g} | "
            f"{np.percentile(full_vals, 90):.4g} | {np.percentile(sel_vals, 90):.4g} | "
            f"{np.percentile(full_vals, 100):.4g} | {np.percentile(sel_vals, 100):.4g} |"
        )
    lines.extend(["", "## Operator Coverage", "", "| op | full count | selected count |", "|---|---:|---:|"])
    for op_idx, op in enumerate(report_node_types):
        lines.append(f"| {op} | {sum(op_count(r, op_idx) for r in all_records)} | {sum(op_count(r, op_idx) for r in selected)} |")
    lines.extend(["", "## Structure Coverage", "", "| topology signature | full | selected |", "|---|---:|---:|"])
    full_structures: dict[tuple[str, ...], int] = {}
    selected_structures: dict[tuple[str, ...], int] = {}
    for record in all_records:
        signature = structure_signature(record)
        full_structures[signature] = full_structures.get(signature, 0) + 1
    for record in selected:
        signature = structure_signature(record)
        selected_structures[signature] = selected_structures.get(signature, 0) + 1
    structure_keys = sorted(full_structures, key=lambda key: (-selected_structures.get(key, 0), -full_structures[key], key))[:80]
    for signature in structure_keys:
        lines.append(f"| `{' / '.join(signature)}` | {full_structures[signature]} | {selected_structures.get(signature, 0)} |")
    for section, fields in (
        ("Architecture And Family Coverage", ARCH_FAMILY_FIELDS),
        ("Model Structure And Resource Coverage", STRUCTURE_FIELDS),
    ):
        lines.extend(["", f"## {section}", ""])
        for field in fields:
            lines.extend(["", f"### {field}", "", "| bucket | full | selected |", "|---|---:|---:|"])
            full_counts: dict[str, int] = {}
            selected_counts: dict[str, int] = {}
            for record in all_records:
                key = coverage_value(record, field)
                full_counts[key] = full_counts.get(key, 0) + 1
            for record in selected:
                key = coverage_value(record, field)
                selected_counts[key] = selected_counts.get(key, 0) + 1
            for key in sorted(full_counts, key=lambda item: (-selected_counts.get(item, 0), -full_counts[item], item))[:50]:
                lines.append(f"| `{key}` | {full_counts[key]} | {selected_counts.get(key, 0)} |")
    lines.extend(["", "## Family Coverage", "", "| family tuple | full | selected |", "|---|---:|---:|"])
    full_families: dict[tuple[str, ...], int] = {}
    selected_families: dict[tuple[str, ...], int] = {}
    for record in all_records:
        full_families[record.family_tuple] = full_families.get(record.family_tuple, 0) + 1
    for record in selected:
        selected_families[record.family_tuple] = selected_families.get(record.family_tuple, 0) + 1
    family_keys = sorted(full_families, key=lambda key: (-selected_families.get(key, 0), -full_families[key], key))[:50]
    for family in family_keys:
        lines.append(f"| `{family}` | {full_families[family]} | {selected_families.get(family, 0)} |")
    if validation_failures:
        lines.extend(["", "## Validation Exclusions", "", "| model id | stem | error |", "|---|---|---|"])
        for failure in validation_failures[:200]:
            error = failure["error"].replace("|", "\\|")
            lines.append(f"| {failure['model_id']} | `{failure['stem']}` | `{error}` |")
        if len(validation_failures) > 200:
            lines.append(f"| ... | ... | `{len(validation_failures) - 200} additional exclusions omitted` |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    manifest = out_dir / "manifest" / "subset_manifest.jsonl"
    subset_dir = out_dir / "subset"
    if manifest.exists() and models_dir.exists() and subset_dir.exists() and not args.force:
        print(f"pack already exists at {out_dir}; use --force to regenerate", flush=True)
        return
    if models_dir.exists():
        shutil.rmtree(models_dir)
    if (out_dir / "manifest").exists():
        shutil.rmtree(out_dir / "manifest")
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    for generated_file in (out_dir / "selection_report.md", out_dir / "coverage_summary.json"):
        generated_file.unlink(missing_ok=True)

    sync_runtime_files(out_dir)
    print(f"generating with {args.generation_workers} worker(s)", flush=True)
    low_precision_focus = normalize_low_precision_focus(args.low_precision_focus)
    focus_families = low_precision_focus_families(low_precision_focus)
    records = materialize_template_records(out_dir, args.subset_size, args.seed, force=args.force, families=focus_families)
    selected = records
    precision_sweep = parse_precision_sweep(args.precision_sweep)
    valid_count, failure_count = write_pack(
        selected,
        records,
        out_dir,
        args.validation_mode,
        precision_sweep,
        generation_workers=args.generation_workers,
        low_precision_focus=low_precision_focus,
    )
    print(
        f"wrote {valid_count} generated models, {valid_count * len(precision_sweep)} manifest rows, "
        f"subset graphs, and coverage report to {out_dir} "
        f"({failure_count} validation replacements)",
        flush=True,
    )


def sync_runtime_files(out_dir: Path) -> None:
    """Copy runtime files when generating a standalone pack outside this package."""

    package_dir = Path(__file__).resolve().parent
    try:
        if out_dir.resolve() == package_dir.resolve():
            return
    except FileNotFoundError:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_dir / "__init__.py", out_dir / "__init__.py")
    profile_src = package_dir / "profile"
    profile_dst = out_dir / "profile"
    if profile_dst.exists():
        shutil.rmtree(profile_dst)
    profile_dst.mkdir(parents=True, exist_ok=True)
    for source in sorted(profile_src.glob("*.py")):
        shutil.copy2(source, profile_dst / source.name)


if __name__ == "__main__":
    main()
