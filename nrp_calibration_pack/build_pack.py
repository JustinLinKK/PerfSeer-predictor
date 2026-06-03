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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch


SEED = 20260602
DEFAULT_SUBSET_SIZE = 4096
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
NODE_TYPES = (
    "Conv",
    "Relu",
    "BatchNormalization",
    "Concat",
    "AveragePool",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Add",
)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NRP calibration model sources.")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--out-dir", default="nrp_calibration_pack")
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-mode", choices=("compile", "construct", "meta", "real", "none"), default="compile")
    parser.add_argument("--smoke-small", action="store_true", help="Prefer tiny CPU-friendly graphs for local smoke packs.")
    parser.add_argument("--force", action="store_true", help="Regenerate manifest/models/subset/report even if they already exist.")
    return parser.parse_args(argv)


def load_records(data_root: Path) -> list[GraphRecord]:
    graph_dir = data_root / "cg" / "cg"
    label_dir = data_root / "label" / "label"
    if not graph_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"expected dataset under {data_root}/cg/cg and {data_root}/label/label")

    records: list[GraphRecord] = []
    for graph_path in sorted(graph_dir.glob("*.pkl")):
        label_path = label_dir / f"{graph_path.stem}.txt"
        if not label_path.exists():
            continue
        with graph_path.open("rb") as fh:
            graph = nx.DiGraph(pickle.load(fh))
        labels = parse_label(label_path)
        records.append(record_from_graph(graph_path, label_path, graph, labels))
    return records


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


def structure_signature(record: GraphRecord) -> tuple[str, ...]:
    op_presence = {op for op, count in zip(NODE_TYPES, record.op_counts) if count > 0}
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
    if not flags:
        flags.append("plain")
    return (
        f"depth:{bucket_value(record.dag_depth, (32, 96, 192))}",
        f"branches:{bucket_value(record.branch_count, (0, 2, 8, 24))}",
        f"joins:{bucket_value(record.join_count, (0, 2, 8, 24))}",
        "+".join(flags),
    )


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


def generate_model_source(model_id: str, record: GraphRecord, graph: nx.DiGraph) -> str:
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
                "preds": [id_map[pred] for pred in graph.predecessors(node)],
            }
        )
    input_shape = infer_input_shape(graph)
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
            f"NODE_SPECS = {json.dumps(node_specs, sort_keys=True)}",
            "",
            "",
            "class GeneratedModel(GraphModel):",
            "    def __init__(self) -> None:",
            "        super().__init__(NODE_SPECS)",
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


def write_pack(records: list[GraphRecord], all_records: list[GraphRecord], out_dir: Path, validation_mode: str) -> tuple[int, int]:
    sync_runtime_files(out_dir)
    models_dir = out_dir / "models"
    manifest_dir = out_dir / "manifest"
    subset_graph_dir = out_dir / "subset" / "cg" / "cg"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    subset_graph_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "__init__.py").write_text('"""Generated calibration model modules."""\n')

    manifest_path = manifest_dir / "subset_manifest.jsonl"
    valid_rows: list[dict[str, Any]] = []
    validation_failures: list[dict[str, str]] = []
    target_size = len(records)
    selected_stems = {record.stem for record in records}
    candidate_pool = list(records) + [
        record
        for record in sorted(all_records, key=lambda rec: (rec.batch_size, rec.stem))
        if record.stem not in selected_stems
    ]
    attempted: set[str] = set()

    for record in candidate_pool:
        if len(valid_rows) >= target_size:
            break
        if record.stem in attempted:
            continue
        attempted.add(record.stem)
        model_id = f"calib_{len(valid_rows):04d}"
        with Path(record.graph_path).open("rb") as fh:
            graph = nx.DiGraph(pickle.load(fh))
        unsupported = unsupported_ops(graph)
        if unsupported:
            validation_failures.append({"stem": record.stem, "model_id": model_id, "error": f"unsupported ops: {', '.join(unsupported)}"})
            continue
        source = generate_model_source(model_id, record, graph)
        model_path = models_dir / f"{model_id}.py"
        subset_graph_path = subset_graph_dir / f"{model_id}.pkl"
        model_path.write_text(source)
        shutil.copyfile(record.graph_path, subset_graph_path)
        input_shape = infer_input_shape(graph)
        if validation_mode != "none":
            try:
                validate_generated_model(model_path, input_shape, validation_mode)
            except Exception as exc:
                model_path.unlink(missing_ok=True)
                subset_graph_path.unlink(missing_ok=True)
                validation_failures.append({"stem": record.stem, "model_id": model_id, "error": repr(exc)})
                continue
        row = {
            **asdict(record),
            "original_stem": record.stem,
            "original_graph_path": record.graph_path,
            "original_label_path": record.label_path,
            "model_id": model_id,
            "model_file": f"models/{model_id}.py",
            "subset_graph_file": f"subset/cg/cg/{model_id}.pkl",
            "label_file": f"label/label/{model_id}.txt",
            "input_shape": list(input_shape),
        }
        valid_rows.append(clean_json(row))

    if len(valid_rows) < target_size:
        raise RuntimeError(
            f"validated only {len(valid_rows)} generated models out of requested {target_size}; "
            f"{len(validation_failures)} candidates failed validation"
        )

    with manifest_path.open("w") as fh:
        for row in valid_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    write_report(
        out_dir / "selection_report.md",
        all_records,
        [record_by_stem(all_records, row["original_stem"]) for row in valid_rows],
        validation_failures,
    )
    write_coverage_summary(
        out_dir / "coverage_summary.json",
        all_records,
        [record_by_stem(all_records, row["original_stem"]) for row in valid_rows],
        validation_failures,
    )
    return len(valid_rows), len(validation_failures)


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
) -> None:
    validation_failures = validation_failures or []
    summary = {
        "full_dataset_graphs": len(all_records),
        "selected_graphs": len(selected),
        "seed": SEED,
        "default_subset_size": DEFAULT_SUBSET_SIZE,
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
                "full": sum(rec.op_counts[op_idx] for rec in all_records),
                "selected": sum(rec.op_counts[op_idx] for rec in selected),
            }
            for op_idx, op in enumerate(NODE_TYPES)
        },
        "family_coverage": coverage_counts(
            (family_key(rec.family_tuple) for rec in all_records),
            (family_key(rec.family_tuple) for rec in selected),
        ),
        "structure_coverage": coverage_counts(
            (structure_key(structure_signature(rec)) for rec in all_records),
            (structure_key(structure_signature(rec)) for rec in selected),
        ),
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


def validate_generated_model(model_path: Path, input_shape: tuple[int, ...], mode: str) -> None:
    if mode == "compile":
        compile(model_path.read_text(), str(model_path), "exec")
        return
    module_name = f"_nrp_validate_{model_path.stem}_{os.getpid()}"
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
        x = torch.zeros(input_shape, device=device)
        with torch.no_grad():
            out = model(x)
        if len(tuple(out.shape)) == 0:
            raise ValueError(f"{model_path} produced scalar output")
    finally:
        sys.modules.pop(module_name, None)


def write_report(
    path: Path,
    all_records: list[GraphRecord],
    selected: list[GraphRecord],
    validation_failures: list[dict[str, str]] | None = None,
) -> None:
    validation_failures = validation_failures or []
    lines = [
        "# NRP Calibration Subset Selection Report",
        "",
        f"- Full dataset graphs: {len(all_records)}",
        f"- Selected graphs: {len(selected)}",
        f"- Seed: {SEED}",
        f"- Default target size: {DEFAULT_SUBSET_SIZE}",
        f"- Validation exclusions replaced: {len(validation_failures)}",
        "",
        "## Selection Policy",
        "",
        "- Balance the final subset across batch sizes before filling with feature-space diversity.",
        "- Reserve pure-family examples for every batch size where they exist.",
        "- Reserve one representative for mixed architecture-family tuples when the subset budget allows it.",
        "- Reserve operator-presence, topology-signature, and size-quantile anchors before diversity fill.",
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
    for op_idx, op in enumerate(NODE_TYPES):
        lines.append(f"| {op} | {sum(r.op_counts[op_idx] for r in all_records)} | {sum(r.op_counts[op_idx] for r in selected)} |")
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
    records = load_records(Path(args.data_root))
    selected = select_smoke_subset(records, args.subset_size) if args.smoke_small else select_subset(records, args.subset_size, args.seed)
    valid_count, failure_count = write_pack(selected, records, out_dir, args.validation_mode)
    print(
        f"wrote {valid_count} generated models, subset graphs, manifest, and coverage report to {out_dir} "
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
