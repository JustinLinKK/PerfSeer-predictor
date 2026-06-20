from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nrp_calibration_pack.build_pack import (  # noqa: E402
    BATCH_BUCKETS,
    DEFAULT_PILOT_SUBSET_SIZE,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_PRECISION_SWEEP,
    NODE_TYPES,
    GraphRecord,
    generate_model_source,
    parse_args as parse_pack_args,
    parse_precision_sweep,
    select_subset,
    write_pack,
)
from perfseer.data import parse_label as parse_dataset_label  # noqa: E402
from perfseer_optimized.train import apply_overrides as apply_train_overrides  # noqa: E402
from perfseer_optimized.train import parse_args as parse_train_args  # noqa: E402
from perfseer_source_converter import SourceModelSpec, convert_source_to_networkx  # noqa: E402


ARG_DEFAULTS = {
    "conv_kernel_size": 0,
    "conv_stride": 0,
    "conv_padding": 0,
    "conv_dilation": 0,
    "conv_groups": 0,
    "conv_bias": 0,
    "linear_in_features": 0,
    "linear_out_features": 0,
    "linear_bias": 0,
    "pool_kernel_size": 0,
    "pool_stride": 0,
    "pool_padding": 0,
    "pool_ceil_mode": 0,
}


def memory_info(
    *,
    batch: int = 1,
    input_channels: int = 3,
    output_channels: int = 3,
    input_h: int = 8,
    input_w: int = 8,
    output_h: int | None = None,
    output_w: int | None = None,
) -> dict[str, int]:
    output_h = input_h if output_h is None else output_h
    output_w = input_w if output_w is None else output_w
    input_size = batch * input_channels * input_h * input_w
    output_size = batch * output_channels * output_h * output_w
    return {
        "bytes": input_size + output_size,
        "weight_size": 0,
        "batch_size": batch,
        "input_size_with_weight": input_size,
        "input_size": input_size,
        "input_channels": input_channels,
        "input_w": input_w,
        "input_h": input_h,
        "output_size": output_size,
        "output_channels": output_channels,
        "output_w": output_w,
        "output_h": output_h,
    }


def feature(op_type: str, args: dict[str, int] | None = None, mem: dict[str, int] | None = None) -> dict[str, object]:
    merged_args = dict(ARG_DEFAULTS)
    if args:
        merged_args.update(args)
    return {
        "type": op_type,
        "args": merged_args,
        "memory_info": mem or memory_info(),
        "flops": 1,
        "arith_intensity": 1.0,
    }


def eval_result_row(
    run_id: str,
    data_root: str,
    precision_configs: list[str] | None = None,
    split_unit: str | None = "graph_signature",
    test_hash: str | None = "eval-test-hash",
    checkpoint_test_hash: str | None = None,
    checkpoint_split_unit: str | None = None,
    limit_applied: bool = False,
    batch_slices: int = 1,
    resource_slices: int = 1,
    graph_signature_slices: int = 1,
    graph_family_slices: int = 1,
    ckpt_paths: list[str] | None = None,
    mean_mape: float = 1.0,
    label_domains: list[str] | None = None,
    label_domain_counts: dict[str, int] | None = None,
    precision_config_counts: dict[str, int] | None = None,
    hardware_id_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    precision_configs = precision_configs or ["fp32_ieee"]
    if label_domains is None:
        label_domains = ["source"] if "source" in data_root or run_id == "accuracy_baseline" else ["precision_profile"]
    if label_domain_counts is None:
        label_domain_counts = {domain: 8 for domain in label_domains}
    if precision_config_counts is None:
        precision_config_counts = {precision: 8 for precision in precision_configs}
    if hardware_id_counts is None:
        hardware_id_counts = {"test_hardware": 8}
    metric = {"train_time": {"MAPE": 1.0, "RMSPE": 1.0, "Acc10": 100.0}}
    row: dict[str, object] = {
        "event": "eval_complete",
        "run_id": run_id,
        "data_root": data_root,
        "mean_mape": mean_mape,
        "num_test_graphs": 8,
        "ckpt_paths": ckpt_paths or [],
        "metrics": metric,
        "metrics_by_precision": {precision: metric for precision in precision_configs},
        "precision_config_counts": precision_config_counts,
        "metrics_by_label_domain": {domain: metric for domain in label_domains},
        "label_domain_counts": label_domain_counts,
        "hardware_id_counts": hardware_id_counts,
        "metrics_by_batch_size": {f"batch_slice_{idx}": metric for idx in range(max(1, batch_slices))},
        "metrics_by_resource_regime": {f"resource_slice_{idx}": metric for idx in range(max(1, resource_slices))},
        "metrics_by_graph_signature": {f"signature_slice_{idx}": metric for idx in range(max(1, graph_signature_slices))},
        "metrics_by_graph_family": {f"family_slice_{idx}": metric for idx in range(max(1, graph_family_slices))},
    }
    if split_unit:
        row["split_unit"] = split_unit
        if test_hash is not None:
            row["test_hash"] = test_hash
        evaluation_split: dict[str, object] = {"split_unit": split_unit, "test_count": 8}
        if test_hash is not None:
            evaluation_split["test_hash"] = test_hash
        if checkpoint_test_hash is not None:
            evaluation_split["checkpoint_test_hash"] = checkpoint_test_hash
        if checkpoint_split_unit is not None:
            evaluation_split["checkpoint_split_unit"] = checkpoint_split_unit
        if limit_applied:
            evaluation_split["limit_applied"] = True
        row["evaluation_split"] = evaluation_split
    return row


def deploy_result_row(
    run_id: str,
    data_root: str,
    metadata_path: str,
    precision_configs: list[str] | None = None,
    ckpt_paths: list[str] | None = None,
) -> dict[str, object]:
    row = eval_result_row(
        run_id,
        data_root,
        precision_configs or ["fp32_ieee", "bf16_amp"],
        batch_slices=2,
        resource_slices=2,
        graph_signature_slices=2,
        graph_family_slices=2,
        ckpt_paths=ckpt_paths,
    )
    row["event"] = "eval_deploy_complete"
    row["runtime_backend"] = "torchscript"
    row["runtime_backend_actual"] = "torchscript"
    row["runtime_statuses"] = [{"model_idx": 0, "backend": "torchscript", "status": "ok"}]
    row["deployment_metadata"] = metadata_path
    row["latency_forward_ms_p50"] = 1.25
    row["latency_forward_ms_p95"] = 1.75
    row["artifact_size_mb"] = 2.5
    return row


def train_result_row(
    run_id: str,
    data_root: str,
    checkpoints: list[str],
    *,
    limit: int | None = 0,
    precision_config: str = "fp32_ieee",
    source_precision_provenance: str = "",
    source_precision_confirmed: bool | None = None,
    init_checkpoint: str | None = None,
    teacher_ckpt_dir: str | None = None,
    teacher_paths: list[str] | None = None,
    include_checkpoint_metadata: bool = True,
    checkpoint_source_precision: dict[str, object] | None = None,
    checkpoint_initialization: dict[str, object] | None = None,
    checkpoint_distillation_teacher: dict[str, object] | None = None,
    split_label_domain_counts: dict[str, dict[str, int]] | None = None,
    split_precision_config_counts: dict[str, dict[str, int]] | None = None,
    split_hardware_id_counts: dict[str, dict[str, int]] | None = None,
    split_unit: str = "graph_signature",
    train_count: int = 4,
    val_count: int = 1,
    test_count: int = 1,
    test_hash: str = "eval-test-hash",
) -> dict[str, object]:
    source_precision_confirmed = bool(source_precision_provenance) if source_precision_confirmed is None else source_precision_confirmed
    if split_label_domain_counts is None:
        train_counts = {"source": 4} if "source" in data_root else {"source": 2, "precision_profile": 2}
        split_label_domain_counts = {
            "train": train_counts,
            "val": {"source": 1} if "source" in data_root else {"precision_profile": 1},
            "test": {"source": 1} if "source" in data_root else {"precision_profile": 1},
        }
    if split_precision_config_counts is None:
        train_counts = {"fp32_ieee": 4} if "source" in data_root else {"bf16_amp": 2, "fp32_ieee": 2}
        split_precision_config_counts = {
            "train": train_counts,
            "val": {"fp32_ieee": 1} if "source" in data_root else {"bf16_amp": 1},
            "test": {"fp32_ieee": 1} if "source" in data_root else {"bf16_amp": 1},
        }
    if split_hardware_id_counts is None:
        split_hardware_id_counts = {
            "train": {"test_hardware": 4} if "source" in data_root else {"test_hardware": 4},
            "val": {"test_hardware": 1},
            "test": {"test_hardware": 1},
        }
    row: dict[str, object] = {
        "event": "train_complete",
        "run_id": run_id,
        "out_dir": f"runs/optimized/{run_id}",
        "elapsed_sec": 1.0,
        "checkpoints": checkpoints,
        "split": {
            "split_unit": split_unit,
            "train_count": train_count,
            "val_count": val_count,
            "test_hash": test_hash,
            "test_count": test_count,
            "label_domain_counts": split_label_domain_counts,
            "precision_config_counts": split_precision_config_counts,
            "hardware_id_counts": split_hardware_id_counts,
        },
        "config": {
            "data": {
                "root": data_root,
                "limit": limit,
                "source_precision_provenance": source_precision_provenance,
                "source_precision_confirmed": source_precision_confirmed,
            },
            "features": {"precision_config": precision_config},
            "train": {"init_checkpoint": init_checkpoint},
            "distillation": {
                "enabled": bool(teacher_ckpt_dir),
                "teacher_ckpt_dir": teacher_ckpt_dir,
            },
            "run": {"run_id": run_id},
        },
    }
    if include_checkpoint_metadata:
        source_precision = checkpoint_source_precision or {
            "precision_config": precision_config,
            "hardware_id": "test_hardware",
            "provenance": source_precision_provenance,
            "confirmed": source_precision_confirmed,
        }
        initialization = checkpoint_initialization
        if initialization is None and init_checkpoint:
            initialization = {"path": init_checkpoint, "strict": True}
        distillation_teacher = checkpoint_distillation_teacher
        if distillation_teacher is None:
            if teacher_ckpt_dir:
                distillation_teacher = {
                    "kind": "multi",
                    "count": len(teacher_paths or []),
                    "paths": list(teacher_paths or []),
                }
            else:
                distillation_teacher = {"kind": "none", "count": 0, "paths": []}
        row["checkpoint_metadata"] = [
            {
                "path": path,
                "exists": True,
                "model_name": "seernet_multi",
                "metric_idx": None,
                "source_precision": source_precision,
                "precision_hardware_config": {"precision_config": precision_config, "hardware_id": "test_hardware"},
                "initialization": initialization,
                "distillation_teacher": distillation_teacher,
                "split": {
                    "split_unit": split_unit,
                    "train_count": train_count,
                    "val_count": val_count,
                    "test_hash": test_hash,
                    "test_count": test_count,
                    "label_domain_counts": split_label_domain_counts,
                    "precision_config_counts": split_precision_config_counts,
                    "hardware_id_counts": split_hardware_id_counts,
                },
            }
            for path in checkpoints
        ]
    return row


def record(
    stem: str,
    *,
    batch_size: int = 1,
    family_tuple: tuple[str, ...] = ("alpha", "beta", "gamma", "delta"),
    graph_path: str = "",
    label_path: str = "",
    node_count: int = 4,
    edge_count: int = 3,
    train_time: float = 1.0,
    infer_time: float = 0.5,
) -> GraphRecord:
    return GraphRecord(
        stem=stem,
        graph_path=graph_path,
        label_path=label_path,
        batch_size=batch_size,
        family_tuple=family_tuple,
        node_count=node_count,
        edge_count=edge_count,
        dag_depth=max(1, node_count - 1),
        branch_count=0,
        join_count=0,
        total_flops=float(node_count * 10),
        total_memory=float(node_count * 20),
        total_params=float(node_count),
        max_tensor_size=float(node_count * 5),
        train_util=10.0,
        train_mem=100.0,
        train_time=train_time,
        infer_util=20.0,
        infer_mem=80.0,
        infer_time=infer_time,
        op_counts=tuple(1 if i == 0 else 0 for i, _op in enumerate(NODE_TYPES)),
    )


def sequential_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        0,
        feature=feature(
            "Conv",
            {
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "conv_dilation": 1,
                "conv_groups": 1,
                "conv_bias": 1,
            },
            memory_info(input_channels=3, output_channels=4, input_h=8, input_w=8),
        ),
    )
    graph.add_node(1, feature=feature("BatchNormalization", mem=memory_info(input_channels=4, output_channels=4, input_h=8, input_w=8)))
    graph.add_node(2, feature=feature("Relu", mem=memory_info(input_channels=4, output_channels=4, input_h=8, input_w=8)))
    graph.add_node(
        3,
        feature=feature(
            "MaxPool",
            {"pool_kernel_size": 2, "pool_stride": 2, "pool_padding": 0},
            memory_info(input_channels=4, output_channels=4, input_h=8, input_w=8, output_h=4, output_w=4),
        ),
    )
    graph.add_node(
        4,
        feature=feature(
            "AveragePool",
            {"pool_kernel_size": 2, "pool_stride": 2, "pool_padding": 0},
            memory_info(input_channels=4, output_channels=4, input_h=4, input_w=4, output_h=2, output_w=2),
        ),
    )
    graph.add_node(5, feature=feature("GlobalAveragePool", mem=memory_info(input_channels=4, output_channels=4, input_h=2, input_w=2, output_h=1, output_w=1)))
    graph.add_node(6, feature=feature("Flatten", mem=memory_info(input_channels=4, output_channels=4, input_h=1, input_w=1, output_h=1, output_w=1)))
    graph.add_node(
        7,
        feature=feature(
            "Gemm",
            {"linear_in_features": 4, "linear_out_features": 3, "linear_bias": 1},
            memory_info(input_channels=4, output_channels=3, input_h=1, input_w=1, output_h=1, output_w=1),
        ),
    )
    graph.add_edges_from((idx, idx + 1) for idx in range(7))
    return graph


def add_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id in (0, 1):
        graph.add_node(
            node_id,
            feature=feature(
                "Conv",
                {
                    "conv_kernel_size": 1,
                    "conv_stride": 1,
                    "conv_padding": 0,
                    "conv_dilation": 1,
                    "conv_groups": 1,
                    "conv_bias": 0,
                },
                memory_info(input_channels=3, output_channels=3, input_h=4, input_w=4),
            ),
        )
    graph.add_node(2, feature=feature("Add", mem=memory_info(input_channels=3, output_channels=3, input_h=4, input_w=4)))
    graph.add_edges_from([(0, 2), (1, 2)])
    return graph


def concat_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        0,
        feature=feature(
            "Conv",
            {"conv_kernel_size": 1, "conv_stride": 1, "conv_padding": 0, "conv_dilation": 1, "conv_groups": 1, "conv_bias": 0},
            memory_info(input_channels=3, output_channels=2, input_h=4, input_w=4),
        ),
    )
    graph.add_node(
        1,
        feature=feature(
            "Conv",
            {"conv_kernel_size": 1, "conv_stride": 1, "conv_padding": 0, "conv_dilation": 1, "conv_groups": 1, "conv_bias": 0},
            memory_info(input_channels=3, output_channels=3, input_h=4, input_w=4),
        ),
    )
    graph.add_node(2, feature=feature("Concat", mem=memory_info(input_channels=5, output_channels=5, input_h=4, input_w=4)))
    graph.add_edges_from([(0, 2), (1, 2)])
    return graph


def import_generated(source: str, tmp: str):
    path = Path(tmp) / "generated.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("_generated_calibration_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load generated source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def import_run_profile_module():
    path = ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"
    spec = importlib.util.spec_from_file_location("_run_profile_calibration_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load run_profile.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class FakeTransformerEngine:
    def __enter__(self):
        self.old_root = sys.modules.get("transformer_engine")
        self.old_pytorch = sys.modules.get("transformer_engine.pytorch")
        root = types.ModuleType("transformer_engine")
        root.__path__ = []
        pytorch = types.ModuleType("transformer_engine.pytorch")
        pytorch.__package__ = "transformer_engine"
        root.pytorch = pytorch
        sys.modules["transformer_engine"] = root
        sys.modules["transformer_engine.pytorch"] = pytorch
        return pytorch

    def __exit__(self, _exc_type, _exc, _tb):
        if self.old_root is None:
            sys.modules.pop("transformer_engine", None)
        else:
            sys.modules["transformer_engine"] = self.old_root
        if self.old_pytorch is None:
            sys.modules.pop("transformer_engine.pytorch", None)
        else:
            sys.modules["transformer_engine.pytorch"] = self.old_pytorch


class NrpCalibrationPackTests(unittest.TestCase):
    def test_default_subset_size_and_precision_sweep_for_nrp_pack(self) -> None:
        self.assertEqual(DEFAULT_SUBSET_SIZE, 10000)
        self.assertEqual(DEFAULT_PILOT_SUBSET_SIZE, 1000)
        self.assertEqual(DEFAULT_PRECISION_SWEEP, ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp", "fp8_te_hybrid"))

    def test_profile_preset_resolves_pilot_size_and_allows_override(self) -> None:
        self.assertEqual(parse_pack_args([]).subset_size, DEFAULT_SUBSET_SIZE)
        pilot = parse_pack_args(["--profile-preset", "pilot"])
        self.assertEqual(pilot.subset_size, DEFAULT_PILOT_SUBSET_SIZE)
        self.assertEqual(pilot.profile_preset, "pilot")
        override = parse_pack_args(["--profile-preset", "pilot", "--subset-size", "17"])
        self.assertEqual(override.subset_size, 17)

    def test_precision_sweep_rejects_ambiguous_bf32(self) -> None:
        with self.assertRaisesRegex(ValueError, "bf32 is ambiguous"):
            parse_precision_sweep("bf32")

    def test_default_size_selection_returns_exact_count(self) -> None:
        records = [
            record(
                f"bs{BATCH_BUCKETS[idx % len(BATCH_BUCKETS)]}_synthetic_{idx:04d}",
                batch_size=BATCH_BUCKETS[idx % len(BATCH_BUCKETS)],
                node_count=4 + (idx % 17),
                edge_count=3 + (idx % 13),
            )
            for idx in range(DEFAULT_SUBSET_SIZE)
        ]

        selected = select_subset(records, DEFAULT_SUBSET_SIZE, seed=1234)

        self.assertEqual(len(selected), DEFAULT_SUBSET_SIZE)

    def test_subset_selection_is_deterministic_and_batch_covered(self) -> None:
        records: list[GraphRecord] = []
        for batch in BATCH_BUCKETS:
            for idx in range(5):
                records.append(
                    record(
                        f"bs{batch}_synthetic_{idx}",
                        batch_size=batch,
                        node_count=4 + idx,
                        edge_count=3 + idx,
                        train_time=float(batch + idx),
                        infer_time=float(idx + 1) / max(batch, 1),
                    )
                )

        first = select_subset(records, 18, seed=1234)
        second = select_subset(records, 18, seed=1234)

        self.assertEqual([item.stem for item in first], [item.stem for item in second])
        self.assertEqual(len(first), 18)
        self.assertEqual({item.batch_size for item in first}, set(BATCH_BUCKETS))

    def test_generated_sources_execute_supported_ops(self) -> None:
        cases = [
            (sequential_graph(), (1, 3, 8, 8), (1, 3)),
            (add_graph(), (1, 3, 4, 4), (1, 3, 4, 4)),
            (concat_graph(), (1, 3, 4, 4), (1, 5, 4, 4)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, (graph, input_shape, output_shape) in enumerate(cases):
                source = generate_model_source(f"calib_test_{idx}", record(f"stem_{idx}"), graph)
                module, _path = import_generated(source, tmp)
                model = module.make_model().eval()
                with torch.no_grad():
                    output = model(torch.zeros(input_shape))
                self.assertEqual(tuple(output.shape), output_shape)

    def test_generated_source_round_trips_through_converter(self) -> None:
        graph = sequential_graph()
        source = generate_model_source("calib_roundtrip", record("roundtrip"), graph)
        with tempfile.TemporaryDirectory() as tmp:
            module, path = import_generated(source, tmp)
            converted = convert_source_to_networkx(SourceModelSpec(path, "make_model", (module.INPUT_SHAPE,)))

        types = [data["feature"]["type"] for _node, data in converted.nodes(data=True)]
        for expected in ("Conv", "BatchNormalization", "Relu", "MaxPool", "AveragePool", "GlobalAveragePool", "Flatten", "Gemm"):
            self.assertIn(expected, types)

    def test_write_pack_replaces_validation_failures(self) -> None:
        bad = nx.DiGraph()
        bad.add_node(0, feature=feature("Unsupported", mem=memory_info()))
        good = sequential_graph()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bad_path = tmp_path / "bad.pkl"
            good_path = tmp_path / "good.pkl"
            with bad_path.open("wb") as fh:
                pickle.dump(bad, fh)
            with good_path.open("wb") as fh:
                pickle.dump(good, fh)

            bad_record = record("bad", graph_path=str(bad_path), label_path=str(tmp_path / "bad.txt"))
            good_record = record("good", graph_path=str(good_path), label_path=str(tmp_path / "good.txt"))
            written, failures = write_pack([bad_record], [bad_record, good_record], tmp_path / "pack", "real")

            manifest = (tmp_path / "pack" / "manifest" / "subset_manifest.jsonl").read_text()

        self.assertEqual(written, 1)
        self.assertEqual(failures, 1)
        self.assertIn('"stem": "good"', manifest)
        self.assertNotIn('"stem": "bad"', manifest)

    def test_write_pack_manifest_subset_graph_and_coverage_summary(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "graph.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            graph_record = record("original_stem", graph_path=str(graph_path), label_path=str(tmp_path / "original_stem.txt"))

            written, failures = write_pack([graph_record], [graph_record], tmp_path / "pack", "compile")
            manifest_path = tmp_path / "pack" / "manifest" / "subset_manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            row = rows[0]
            coverage = json.loads((tmp_path / "pack" / "coverage_summary.json").read_text())

        self.assertEqual(written, 1)
        self.assertEqual(failures, 0)
        self.assertEqual(len(rows), len(DEFAULT_PRECISION_SWEEP))
        self.assertEqual(row["model_id"], "calib_0000")
        self.assertEqual(row["graph_id"], "calib_0000")
        self.assertEqual(row["precision_config"], "fp32_ieee")
        self.assertEqual(row["original_stem"], "original_stem")
        self.assertEqual(row["model_file"], "models/calib_0000.py")
        self.assertEqual(row["subset_graph_file"], "subset/cg/cg/calib_0000.pkl")
        self.assertEqual(row["label_file"], "label/label/calib_0000_fp32_ieee.txt")
        self.assertEqual(coverage["selected_graphs"], 1)
        self.assertEqual(coverage["manifest_profile_points"], len(DEFAULT_PRECISION_SWEEP))
        self.assertEqual(coverage["precision_sweep"], list(DEFAULT_PRECISION_SWEEP))
        self.assertIn("batch_size_coverage", coverage)
        self.assertIn("operator_coverage", coverage)
        self.assertIn("family_coverage", coverage)
        self.assertIn("structure_coverage", coverage)
        self.assertIn("architecture_family_coverage", coverage)
        self.assertIn("model_structure_coverage", coverage)
        self.assertIn("size_quantiles", coverage)

    def test_profiler_writes_dataset_compatible_label_path(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "graph.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            graph_record = record("original_stem", graph_path=str(graph_path), label_path=str(tmp_path / "original_stem.txt"))
            write_pack([graph_record], [graph_record], tmp_path / "pack", "compile")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
                    "--manifest",
                    str(tmp_path / "pack" / "manifest" / "subset_manifest.jsonl"),
                    "--models-dir",
                    str(tmp_path / "pack" / "models"),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--num-shards",
                    "1",
                    "--precision-config",
                    "fp32_ieee",
                    "--warmup",
                    "1",
                    "--infer-repeats",
                    "1",
                    "--train-repeats",
                    "1",
                    "--device",
                    "cpu",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            label_path = tmp_path / "out" / "label" / "label" / "calib_0000_fp32_ieee.txt"
            label_exists = label_path.exists()
            parsed = parse_dataset_label(str(label_path))

        self.assertTrue(label_exists)
        self.assertEqual(parsed.shape, (6,))

    def test_tf32_controls_prefer_new_fp32_precision_api(self) -> None:
        module = import_run_profile_module()

        try:
            changes = module.set_tf32_controls(True)
        finally:
            module.set_tf32_controls(False)

        self.assertTrue(changes["enabled"])
        if changes["api_style"] == "fp32_precision":
            self.assertIn("torch.backends.fp32_precision", changes["set"])
            self.assertNotIn("torch.backends.cuda.matmul.allow_tf32", changes["set"])
            self.assertNotIn("torch.backends.cudnn.allow_tf32", changes["set"])
            self.assertIn("effective_state", changes)
        else:
            self.assertEqual(changes["api_style"], "legacy_allow_tf32")
            self.assertIn("torch.backends.cuda.matmul.allow_tf32", changes["set"])

    def test_bf16_precision_runtime_records_torch_probe(self) -> None:
        module = import_run_profile_module()
        module.compute_capability_tuple = lambda _device: (8, 0)
        old_probe = getattr(module.torch.cuda, "is_bf16_supported", None)
        module.torch.cuda.is_bf16_supported = lambda: False
        args = types.SimpleNamespace(fp8_backend="transformer_engine")

        try:
            runtime = module.precision_runtime("bf16_amp", torch.device("cuda"), args)
        finally:
            if old_probe is None:
                delattr(module.torch.cuda, "is_bf16_supported")
            else:
                module.torch.cuda.is_bf16_supported = old_probe

        self.assertFalse(runtime.supported)
        self.assertEqual(runtime.details["bf16_probe"]["torch_cuda_is_bf16_supported"], False)
        self.assertTrue(runtime.details["bf16_probe"]["compute_capability_policy_supported"])

    def test_fp8_transformer_engine_policy_allows_ada_compute_capability(self) -> None:
        module = import_run_profile_module()
        module.compute_capability_tuple = lambda _device: (8, 9)
        args = types.SimpleNamespace(fp8_backend="transformer_engine")

        with FakeTransformerEngine():
            runtime = module.precision_runtime("fp8_te_hybrid", torch.device("cuda"), args)

        self.assertTrue(runtime.supported)
        self.assertIsNone(runtime.unsupported_reason)
        self.assertEqual(runtime.details["compute_capability"], "8.9")
        self.assertEqual(runtime.details["fp8_te_min_compute_capability"], "8.9")
        self.assertIn("Ada-or-newer", runtime.details["fp8_te_device_policy"])

    def test_fp8_transformer_engine_policy_rejects_pre_ada_compute_capability(self) -> None:
        module = import_run_profile_module()
        module.compute_capability_tuple = lambda _device: (8, 0)
        args = types.SimpleNamespace(fp8_backend="transformer_engine")

        with FakeTransformerEngine():
            runtime = module.precision_runtime("fp8_te_hybrid", torch.device("cuda"), args)

        self.assertFalse(runtime.supported)
        self.assertIn("Ada-or-newer", runtime.unsupported_reason or "")
        self.assertIn("SM 8.9+", runtime.unsupported_reason or "")
        self.assertEqual(runtime.details["compute_capability"], "8.0")

    def test_materialize_precision_dataset_writes_hardware_metadata(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "graph.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            original_label = tmp_path / "original_stem.txt"
            original_label.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            graph_record = record("original_stem", graph_path=str(graph_path), label_path=str(original_label))
            write_pack([graph_record], [graph_record], tmp_path / "pack", "compile", precision_sweep=("fp32_ieee",))

            results_dir = tmp_path / "results"
            results_dir.mkdir()
            result_row = {
                "status": "ok",
                "model_id": "calib_0000",
                "graph_id": "calib_0000",
                "precision_config": "fp32_ieee",
                "profile_point_id": "calib_0000::fp32_ieee",
                "label": {"train": "1|2|3|4|5|6|7", "infer": "1|2|3|4|5|6|7"},
                "hardware": {
                    "gpu_name": "NVIDIA A100-SXM4-80GB",
                    "compute_capability": "8.0",
                    "multi_processor_count": 108,
                    "total_memory_mib": 81920,
                },
            }
            (results_dir / "results_shard0.jsonl").write_text(json.dumps(result_row) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(tmp_path / "pack"),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--hardware-id",
                    "a100",
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            label_path = out_root / "label" / "label" / "calib_0000_a100_fp32_ieee.txt"
            graph_exists = (out_root / "cg" / "cg" / "calib_0000.pkl").exists()
            label_exists = label_path.exists()
            source_label_exists = (out_root / "label" / "label" / "calib_0000.txt").exists()
            parsed_shape = parse_dataset_label(str(label_path)).shape
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())

        self.assertTrue(graph_exists)
        self.assertTrue(label_exists)
        self.assertEqual(parsed_shape, (6,))
        self.assertEqual(report["precision_labels"], 1)
        self.assertEqual(report["calibration_source_labels"], 1)
        source_rows = [row for row in metadata if row.get("label_domain") == "source"]
        precision_rows = [row for row in metadata if row.get("label_domain") != "source"]
        self.assertEqual(len(source_rows), 1)
        self.assertEqual(len(precision_rows), 1)
        self.assertTrue(source_label_exists)
        self.assertEqual(precision_rows[0]["hardware_id"], "a100")
        self.assertEqual(precision_rows[0]["base_label_file"], "label/label/calib_0000.txt")
        self.assertEqual(precision_rows[0]["hardware_features"]["sm_count"], 108)

    def test_materialize_precision_dataset_tags_source_domain_labels(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_root = tmp_path / "base_dataset"
            (base_root / "cg" / "cg").mkdir(parents=True)
            (base_root / "label" / "label").mkdir(parents=True)
            with (base_root / "cg" / "cg" / "base_0000.pkl").open("wb") as fh:
                pickle.dump(graph, fh)
            (base_root / "label" / "label" / "base_0000.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            pack_dir = tmp_path / "empty_pack"
            pack_dir.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--base-data-root",
                    str(base_root),
                    "--base-mode",
                    "copy",
                    "--source-precision-config",
                    "tf32",
                    "--source-hardware-id",
                    "a100_source",
                    "--source-hardware-features-json",
                    '{"compute_capability": "8.0", "sm_count": 108}',
                    "--source-precision-provenance",
                    "original-profiler-notes.md#tf32",
                    "--require-source-precision-provenance",
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())

        self.assertEqual(report["base_pairs"], 1)
        self.assertEqual(report["source_metadata_labels"], 1)
        self.assertEqual(report["source_precision_config"], "tf32")
        self.assertTrue(report["source_precision_confirmed"])
        self.assertEqual(report["source_precision_provenance"], "original-profiler-notes.md#tf32")
        self.assertEqual(metadata[0]["label_domain"], "source")
        self.assertTrue(metadata[0]["is_base_label"])
        self.assertEqual(metadata[0]["precision_config"], "tf32")
        self.assertEqual(metadata[0]["hardware_id"], "a100_source")
        self.assertTrue(metadata[0]["source_precision_confirmed"])
        self.assertEqual(metadata[0]["source_precision_provenance"], "original-profiler-notes.md#tf32")
        self.assertEqual(metadata[0]["hardware_features"]["sm_count"], 108)

    def test_materialize_precision_dataset_records_unknown_source_precision_as_unconfirmed(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_root = tmp_path / "base_dataset"
            (base_root / "cg" / "cg").mkdir(parents=True)
            (base_root / "label" / "label").mkdir(parents=True)
            with (base_root / "cg" / "cg" / "base_0000.pkl").open("wb") as fh:
                pickle.dump(graph, fh)
            (base_root / "label" / "label" / "base_0000.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            pack_dir = tmp_path / "empty_pack"
            pack_dir.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--base-data-root",
                    str(base_root),
                    "--base-mode",
                    "copy",
                    "--source-precision-config",
                    "source_domain_unknown",
                    "--source-precision-provenance",
                    "https://github.com/upuuuuuu/PerfSeer#dataset-profile",
                    "--require-source-precision-provenance",
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())

        self.assertEqual(report["source_precision_config"], "source_domain_unknown")
        self.assertFalse(report["source_precision_confirmed"])
        self.assertEqual(metadata[0]["precision_config"], "source_domain_unknown")
        self.assertFalse(metadata[0]["source_precision_confirmed"])

    def test_materialize_precision_dataset_requires_source_precision_provenance(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_root = tmp_path / "base_dataset"
            (base_root / "cg" / "cg").mkdir(parents=True)
            (base_root / "label" / "label").mkdir(parents=True)
            with (base_root / "cg" / "cg" / "base_0000.pkl").open("wb") as fh:
                pickle.dump(graph, fh)
            (base_root / "label" / "label" / "base_0000.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            pack_dir = tmp_path / "empty_pack"
            pack_dir.mkdir()

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--base-data-root",
                    str(base_root),
                    "--base-mode",
                    "copy",
                    "--require-source-precision-provenance",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--source-precision-provenance is empty", result.stderr or result.stdout)

    def test_materialize_precision_dataset_can_add_pseudo_precision_rows(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_root = tmp_path / "base_dataset"
            (base_root / "cg" / "cg").mkdir(parents=True)
            (base_root / "label" / "label").mkdir(parents=True)
            with (base_root / "cg" / "cg" / "base_0000.pkl").open("wb") as fh:
                pickle.dump(graph, fh)
            (base_root / "label" / "label" / "base_0000.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            pack_dir = tmp_path / "empty_pack"
            pack_dir.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--base-data-root",
                    str(base_root),
                    "--base-mode",
                    "copy",
                    "--pseudo-precision-sweep",
                    "bf16_amp,fp16_amp",
                    "--pseudo-hardware-id",
                    "a100",
                    "--pseudo-hardware-features-json",
                    '{"compute_capability": "8.0", "sm_count": 108}',
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())
            pseudo_rows = [row for row in metadata if row.get("label_domain") == "pseudo"]
            pseudo_files_exist = [bool((out_root / row["label_file"]).exists()) for row in pseudo_rows]

        self.assertEqual(report["pseudo_labels"], 2)
        self.assertEqual(report["pseudo_precision_sweep"], ["bf16_amp", "fp16_amp"])
        self.assertEqual(report["pseudo_hardware_id"], "a100")
        self.assertEqual(len(pseudo_rows), 2)
        self.assertEqual({row["precision_config"] for row in pseudo_rows}, {"bf16_amp", "fp16_amp"})
        self.assertTrue(all(row["is_pseudo_label"] for row in pseudo_rows))
        self.assertTrue(all(pseudo_files_exist))
        self.assertEqual(pseudo_rows[0]["hardware_features"]["sm_count"], 108)

    def test_materialize_precision_dataset_reports_rejected_fp8_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            results_dir = tmp_path / "results"
            pack_dir.mkdir()
            results_dir.mkdir()
            result_row = {
                "status": "unsupported_precision",
                "model_id": "calib_0000",
                "graph_id": "calib_0000",
                "precision_config": "fp8_te_hybrid",
                "profile_point_id": "calib_0000::fp8_te_hybrid",
                "error": "Generated GraphModel ops are not yet rewritten to Transformer Engine FP8 modules",
                "precision": {
                    "precision_config": "fp8_te_hybrid",
                    "backend": "transformer_engine",
                    "fallback_policy": "record_unsupported_generated_ops",
                },
                "hardware": {"gpu_name": "NVIDIA H100", "compute_capability": "9.0"},
            }
            (results_dir / "results_shard0.jsonl").write_text(json.dumps(result_row) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            report = json.loads((out_root / "precision_materialization_report.json").read_text())
            rejected = [json.loads(line) for line in (out_root / "precision_rejected_rows.jsonl").read_text().splitlines()]

        self.assertEqual(report["precision_labels"], 0)
        self.assertEqual(report["skipped"], {"unsupported_precision": 1})
        self.assertEqual(report["skipped_by_precision"]["fp8_te_hybrid"], {"unsupported_precision": 1})
        self.assertEqual(report["skipped_by_status"]["unsupported_precision"], {"fp8_te_hybrid": 1})
        self.assertEqual(report["fallback_policy_counts"]["record_unsupported_generated_ops"], {"fp8_te_hybrid": 1})
        self.assertEqual(report["unsupported_fp8_rows"], 1)
        self.assertEqual(report["rejected_rows_file"], "precision_rejected_rows.jsonl")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["precision_config"], "fp8_te_hybrid")
        self.assertEqual(rejected[0]["fallback_policy"], "record_unsupported_generated_ops")

    def test_submit_script_renders_indexed_gpu_job(self) -> None:
        result = subprocess.run(
            [
                str(ROOT / "nrp_calibration_pack" / "submit_nrp_calibration.sh"),
                "--namespace",
                "test-ns",
                "--image",
                "example/perfseer:latest",
                "--pvc",
                "calibration-pvc",
                "--gpu-product",
                "NVIDIA-GeForce-RTX-4090",
                "--gpu-resource",
                "nvidia.com/a100",
                "--parallelism",
                "2",
                "--completions",
                "3",
                "--warmup",
                "20",
                "--infer-repeats",
                "50",
                "--train-repeats",
                "50",
                "--sample-interval",
                "0.02",
                "--precision-sweep",
                "fp32_ieee,bf16_amp",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        yaml = result.stdout
        self.assertIn("completionMode: Indexed", yaml)
        self.assertIn("parallelism: 2", yaml)
        self.assertIn("completions: 3", yaml)
        self.assertIn("nvidia.com/a100: \"1\"", yaml)
        self.assertIn("key: nvidia.com/gpu.product", yaml)
        self.assertIn("NVIDIA-GeForce-RTX-4090", yaml)
        self.assertIn("JOB_COMPLETION_INDEX", yaml)
        self.assertIn("batch.kubernetes.io/job-completion-index", yaml)
        self.assertIn("--warmup 20", yaml)
        self.assertIn("--infer-repeats 50", yaml)
        self.assertIn("--train-repeats 50", yaml)
        self.assertIn("--sample-interval 0.02", yaml)
        self.assertIn("--precision-sweep fp32_ieee,bf16_amp", yaml)
        self.assertIn("--fp8-backend transformer_engine", yaml)

    def test_submit_script_uses_stable_default_profile_budget(self) -> None:
        result = subprocess.run(
            [
                str(ROOT / "nrp_calibration_pack" / "submit_nrp_calibration.sh"),
                "--namespace",
                "test-ns",
                "--image",
                "example/perfseer:latest",
                "--pvc",
                "calibration-pvc",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        yaml = result.stdout
        self.assertIn("completions: 64", yaml)
        self.assertIn("--warmup 20", yaml)
        self.assertIn("--infer-repeats 50", yaml)
        self.assertIn("--train-repeats 50", yaml)
        self.assertIn("--sample-interval 0.01", yaml)
        self.assertIn("--fp8-backend transformer_engine", yaml)

    def test_precision_transfer_flow_dry_run_evaluates_source_and_precision_domains(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_precision_transfer_flow.py"),
                "--results-dir",
                "/tmp/profiler_results",
                "--source-data-root",
                "dataset_source",
                "--precision-data-root",
                "dataset_precision",
                "--source-precision-config",
                "tf32",
                "--source-hardware-id",
                "a100_source",
                "--source-precision-provenance",
                "original-profiler-notes.md#tf32",
                "--require-source-precision-provenance",
                "--baseline-run-id",
                "accuracy_baseline",
                "--baseline-data-root",
                "dataset_source",
                "--source-epochs",
                "1",
                "--transfer-epochs",
                "1",
                "--student-epochs",
                "1",
                "--deploy-eval-profile",
                "src/perfseer-optimized/configs/eval_profiles/cpu_torchscript_fp32.yaml",
                "--split-unit",
                "graph_signature",
                "--check-results",
                "--required-precision",
                "bf16_amp",
                "--min-eval-precision-count",
                "bf16_amp=2",
                "--min-eval-hardware-count",
                "test_hardware=2",
                "--required-label-domain",
                "precision_profile",
                "--min-eval-precision-labels",
                "2",
                "--min-precision-slices",
                "2",
                "--min-label-domain-slices",
                "1",
                "--min-batch-size-slices",
                "2",
                "--min-resource-regime-slices",
                "2",
                "--min-graph-signature-slices",
                "2",
                "--min-graph-family-slices",
                "2",
                "--min-materialized-precision-labels",
                "10",
                "--min-materialized-base-pairs",
                "100",
                "--min-materialized-source-labels",
                "100",
                "--min-materialized-pseudo-labels",
                "2",
                "--max-source-baseline-mape-delta",
                "0.25",
                "--max-student-mean-mape",
                "5.0",
                "--max-deploy-mean-mape",
                "6.0",
                "--max-deploy-latency-p50",
                "2.0",
                "--expected-deploy-runtime-backend",
                "torchscript",
                "--expected-deploy-runtime-backend-actual",
                "torchscript",
                "--require-checkpoint-files",
                "--require-train-events",
                "--require-unlimited-train-data",
                "--required-train-label-domain",
                "source,precision_profile",
                "--min-train-precision-count",
                "bf16_amp=2",
                "--min-train-hardware-count",
                "test_hardware=2",
                "--min-train-split-count",
                "2",
                "--min-val-split-count",
                "1",
                "--min-train-test-count",
                "1",
                "--min-train-source-labels",
                "2",
                "--min-train-precision-labels",
                "2",
                "--require-train-checkpoint-metadata",
                "--require-train-lineage",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        stdout = result.stdout
        self.assertIn("# evaluate source teacher", stdout)
        self.assertIn("--ckpt-dir runs/optimized/precision_large_teacher_source", stdout)
        self.assertIn("--data-root dataset_source", stdout)
        self.assertIn("# evaluate precision teacher", stdout)
        self.assertIn("# evaluate precision student", stdout)
        self.assertIn("# evaluate deployment student", stdout)
        self.assertIn("--ckpt-dir runs/optimized/precision_large_teacher_transfer", stdout)
        self.assertIn("--ckpt-dir runs/optimized/precision_distill_student_128", stdout)
        self.assertIn("perfseer_optimized.eval_deploy", stdout)
        self.assertIn("--eval-profile src/perfseer-optimized/configs/eval_profiles/cpu_torchscript_fp32.yaml", stdout)
        self.assertIn("--data-root dataset_precision", stdout)
        self.assertEqual(stdout.count("--split-unit graph_signature"), 3)
        self.assertIn("scripts/check_precision_transfer_results.py", stdout)
        self.assertIn("--required-precision bf16_amp", stdout)
        self.assertIn("--min-eval-precision-count bf16_amp=2", stdout)
        self.assertIn("--min-eval-hardware-count test_hardware=2", stdout)
        self.assertIn("--required-label-domain precision_profile", stdout)
        self.assertIn("--min-eval-precision-labels 2", stdout)
        self.assertIn("--required-split-unit graph_signature", stdout)
        self.assertIn("--materialization-report dataset_precision/precision_materialization_report.json", stdout)
        self.assertIn("--require-source-precision-provenance", stdout)
        self.assertNotIn("--require-source-precision-confirmed", stdout)
        self.assertIn("--source-precision-config tf32", stdout)
        self.assertIn("--source-hardware-id a100_source", stdout)
        self.assertIn("--precision-config tf32", stdout)
        self.assertIn("--hardware-id a100_source", stdout)
        self.assertIn("--expected-source-precision-config tf32", stdout)
        self.assertIn("--expected-source-precision-provenance", stdout)
        self.assertIn("original-profiler-notes.md#tf32", stdout)
        self.assertIn("--baseline-run-id accuracy_baseline", stdout)
        self.assertIn("--baseline-data-root dataset_source", stdout)
        self.assertIn("--max-source-baseline-mape-delta 0.25", stdout)
        self.assertIn("--require-deployment-eval", stdout)
        self.assertIn("--require-deployment-metadata", stdout)
        self.assertIn("--expected-deploy-runtime-backend torchscript", stdout)
        self.assertIn("--expected-deploy-runtime-backend-actual torchscript", stdout)
        self.assertIn("--require-checkpoint-files", stdout)
        self.assertIn("--require-deployment-student-checkpoint", stdout)
        self.assertIn("--require-train-events", stdout)
        self.assertIn("--require-eval-train-checkpoints", stdout)
        self.assertIn("--require-unlimited-train-data", stdout)
        self.assertIn("--required-train-label-domain source,precision_profile", stdout)
        self.assertIn("--min-train-precision-count bf16_amp=2", stdout)
        self.assertIn("--min-train-hardware-count test_hardware=2", stdout)
        self.assertIn("--min-train-split-count 2", stdout)
        self.assertIn("--min-val-split-count 1", stdout)
        self.assertIn("--min-train-test-count 1", stdout)
        self.assertIn("--min-train-source-labels 2", stdout)
        self.assertIn("--min-train-precision-labels 2", stdout)
        self.assertIn("--require-train-checkpoint-metadata", stdout)
        self.assertIn("--require-train-lineage", stdout)
        self.assertIn("--require-source-train-provenance", stdout)
        self.assertIn("--max-deploy-mean-mape 6.0", stdout)
        self.assertIn("--max-deploy-latency-p50 2.0", stdout)
        self.assertIn("--min-precision-slices 2", stdout)
        self.assertIn("--min-label-domain-slices 1", stdout)
        self.assertIn("--min-batch-size-slices 2", stdout)
        self.assertIn("--min-resource-regime-slices 2", stdout)
        self.assertIn("--min-graph-signature-slices 2", stdout)
        self.assertIn("--min-graph-family-slices 2", stdout)
        self.assertIn("--min-materialized-precision-labels 10", stdout)
        self.assertIn("--min-materialized-base-pairs 100", stdout)
        self.assertIn("--min-materialized-source-labels 100", stdout)
        self.assertIn("--min-materialized-pseudo-labels 2", stdout)
        self.assertIn("--max-student-mean-mape 5.0", stdout)
        self.assertIn("--source-precision-provenance", stdout)
        self.assertIn("original-profiler-notes.md#tf32", stdout)
        self.assertIn("--require-source-precision-provenance", stdout)

    def test_precision_transfer_flow_dry_run_emits_structural_validation_suite(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_precision_transfer_flow.py"),
                "--skip-materialize",
                "--skip-source-pretrain",
                "--skip-source-eval",
                "--source-data-root",
                "dataset_source",
                "--precision-data-root",
                "dataset_precision",
                "--check-results",
                "--required-precision",
                "bf16_amp",
                "--required-label-domain",
                "precision_profile",
                "--require-train-events",
                "--min-train-split-count",
                "2",
                "--structural-validation-splits",
                "graph_signature,graph_family",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        stdout = result.stdout
        self.assertIn("# structural validation split graph_signature", stdout)
        self.assertIn("# structural validation split graph_family", stdout)
        self.assertIn("--run-id precision_large_teacher_transfer_graph_signature", stdout)
        self.assertIn("--run-id precision_distill_student_128_graph_signature", stdout)
        self.assertIn("--run-id precision_large_teacher_transfer_graph_family", stdout)
        self.assertIn("--run-id precision_distill_student_128_graph_family", stdout)
        self.assertIn("--split-unit graph_signature", stdout)
        self.assertIn("--split-unit graph_family", stdout)
        self.assertIn("--transfer-run-id precision_large_teacher_transfer_graph_signature", stdout)
        self.assertIn("--student-run-id precision_distill_student_128_graph_signature", stdout)
        self.assertIn("--transfer-run-id precision_large_teacher_transfer_graph_family", stdout)
        self.assertIn("--student-run-id precision_distill_student_128_graph_family", stdout)
        self.assertIn("--required-split-unit graph_signature", stdout)
        self.assertIn("--required-split-unit graph_family", stdout)
        self.assertGreaterEqual(stdout.count("--skip-source"), 3)

    def test_train_cli_overrides_source_precision_feature_identity(self) -> None:
        cfg = {"run": {}, "data": {}, "features": {}, "train": {}}
        args = parse_train_args(["--precision-config", "tf32", "--hardware-id", "a100_source"])
        resolved = apply_train_overrides(cfg, args)

        self.assertEqual(resolved["features"]["precision_config"], "tf32")
        self.assertEqual(resolved["features"]["hardware_id"], "a100_source")

        bad_args = parse_train_args(["--precision-config", "bf32"])
        with self.assertRaises(ValueError):
            apply_train_overrides({"run": {}, "data": {}, "features": {}, "train": {}}, bad_args)

        unknown_args = parse_train_args(
            [
                "--precision-config",
                "source_domain_unknown",
                "--source-precision-provenance",
                "https://github.com/upuuuuuu/PerfSeer#dataset-profile",
                "--require-source-precision-provenance",
            ]
        )
        unknown = apply_train_overrides({"run": {}, "data": {}, "features": {}, "train": {}}, unknown_args)
        self.assertEqual(unknown["features"]["precision_config"], "source_domain_unknown")
        self.assertFalse(unknown["data"]["source_precision_confirmed"])

    def test_check_precision_transfer_results_accepts_complete_eval_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            report = Path(tmp) / "report.json"
            materialization_report = Path(tmp) / "precision_materialization_report.json"
            deployment_metadata = Path(tmp) / "deployment_metadata.json"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                eval_result_row(
                    "accuracy_baseline",
                    "dataset_source",
                    ["fp32_ieee"],
                    mean_mape=1.0,
                ),
                train_result_row(
                    "precision_large_teacher_source",
                    "dataset_source",
                    [str(source_ckpt)],
                    source_precision_provenance="original-profiler-notes.md#tf32",
                ),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                ),
                eval_result_row(
                    "precision_large_teacher_source",
                    "dataset_source",
                    ["fp32_ieee"],
                    batch_slices=2,
                    resource_slices=2,
                    graph_signature_slices=2,
                    graph_family_slices=2,
                    ckpt_paths=[str(source_ckpt)],
                    mean_mape=1.04,
                ),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    batch_slices=2,
                    resource_slices=2,
                    graph_signature_slices=2,
                    graph_family_slices=2,
                    ckpt_paths=[str(transfer_ckpt)],
                ),
                eval_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    batch_slices=2,
                    resource_slices=2,
                    graph_signature_slices=2,
                    graph_family_slices=2,
                    ckpt_paths=[str(student_ckpt)],
                ),
                deploy_result_row("precision_distill_student_128", "dataset_precision", str(deployment_metadata), ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            deployment_metadata.write_text(
                json.dumps(
                    {
                        "feature_config": {"precision_config": "bf16_amp"},
                        "precision_hardware_config": {"precision_config": "bf16_amp", "hardware_id": "a100"},
                        "feature_layout": {"global_dim": 8},
                        "supported_precision_hardware": {
                            "precision_configs": ["bf16_amp", "fp32_ieee"],
                            "hardware_ids": ["a100"],
                        },
                        "required_inputs": {"u": {"dim": 8}},
                    }
                )
                + "\n"
            )
            materialization_report.write_text(
                json.dumps(
                    {
                        "source_precision_config": "fp32_ieee",
                        "source_precision_confirmed": True,
                        "source_precision_provenance": "original-profiler-notes.md#tf32",
                        "base_pairs": 3,
                        "source_metadata_labels": 2,
                        "calibration_source_labels": 1,
                        "precision_labels": 12,
                        "pseudo_labels": 4,
                        "unsupported_fp8_rows": 3,
                    }
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-eval-precision-count",
                    "bf16_amp=2",
                    "--min-eval-hardware-count",
                    "test_hardware=2",
                    "--required-label-domain",
                    "precision_profile",
                    "--min-eval-precision-labels",
                    "2",
                    "--required-split-unit",
                    "graph_signature",
                    "--min-precision-slices",
                    "2",
                    "--min-label-domain-slices",
                    "1",
                    "--min-batch-size-slices",
                    "2",
                    "--min-resource-regime-slices",
                    "2",
                    "--min-graph-signature-slices",
                    "2",
                    "--min-graph-family-slices",
                    "2",
                    "--materialization-report",
                    str(materialization_report),
                    "--require-source-precision-confirmed",
                    "--expected-source-precision-config",
                    "fp32_ieee",
                    "--expected-source-precision-provenance",
                    "original-profiler-notes.md#tf32",
                    "--baseline-run-id",
                    "accuracy_baseline",
                    "--baseline-data-root",
                    "dataset_source",
                    "--max-source-baseline-mape-delta",
                    "0.1",
                    "--min-materialized-precision-labels",
                    "10",
                    "--min-materialized-base-pairs",
                    "3",
                    "--min-materialized-source-labels",
                    "3",
                    "--min-materialized-pseudo-labels",
                    "4",
                    "--require-deployment-eval",
                    "--require-deployment-metadata",
                    "--expected-deploy-runtime-backend",
                    "torchscript",
                    "--expected-deploy-runtime-backend-actual",
                    "torchscript",
                    "--max-deploy-mean-mape",
                    "6.0",
                    "--max-deploy-latency-p50",
                    "2.0",
                    "--require-checkpoint-files",
                    "--require-deployment-student-checkpoint",
                    "--require-train-events",
                    "--require-eval-train-checkpoints",
                    "--required-train-label-domain",
                    "source,precision_profile",
                    "--min-train-precision-count",
                    "bf16_amp=2",
                    "--min-train-hardware-count",
                    "test_hardware=2",
                    "--min-train-split-count",
                    "2",
                    "--min-val-split-count",
                    "1",
                    "--min-train-test-count",
                    "1",
                    "--min-train-source-labels",
                    "2",
                    "--min-train-precision-labels",
                    "2",
                    "--require-unlimited-train-data",
                    "--require-train-checkpoint-metadata",
                    "--require-train-lineage",
                    "--require-source-train-provenance",
                    "--require-source-train-precision-confirmed",
                    "--report-out",
                    str(report),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            saved_report = json.loads(report.read_text())

        self.assertIn("precision transfer result check passed", result.stdout)
        self.assertTrue(saved_report["ok"])
        self.assertEqual(saved_report["required_split_unit"], "graph_signature")
        self.assertTrue(saved_report["checkpoint_files_required"])
        self.assertTrue(saved_report["training_required"])
        self.assertTrue(saved_report["eval_train_checkpoints_required"])
        self.assertTrue(saved_report["unlimited_train_data_required"])
        self.assertEqual(saved_report["required_train_label_domains"], ["source", "precision_profile"])
        self.assertEqual(
            saved_report["min_train_label_counts"],
            {"precision_profile": 2, "pseudo": 0, "source": 2},
        )
        self.assertEqual(saved_report["min_train_precision_counts"], {"bf16_amp": 2})
        self.assertEqual(saved_report["min_train_hardware_counts"], {"test_hardware": 2})
        self.assertEqual(saved_report["min_train_split_counts"], {"train": 2, "val": 1, "test": 1})
        self.assertFalse(saved_report["source_precision_provenance_required"])
        self.assertTrue(saved_report["source_precision_confirmed_required"])
        self.assertTrue(saved_report["source_train_provenance_required"])
        self.assertTrue(saved_report["source_train_precision_confirmed_required"])
        self.assertTrue(saved_report["train_checkpoint_metadata_required"])
        self.assertTrue(saved_report["train_lineage_required"])
        self.assertEqual(saved_report["runs"]["baseline"]["run_id"], "accuracy_baseline")
        self.assertEqual(saved_report["baseline_comparison"]["baseline_mean_mape"], 1.0)
        self.assertEqual(saved_report["baseline_comparison"]["source_mean_mape"], 1.04)
        self.assertAlmostEqual(saved_report["baseline_comparison"]["delta"], 0.04)
        self.assertEqual(saved_report["baseline_comparison"]["max_delta"], 0.1)
        self.assertEqual(
            saved_report["min_slice_counts"],
            {
                "batch_size": 2,
                "graph_family": 2,
                "graph_signature": 2,
                "label_domain": 1,
                "precision": 2,
                "resource_regime": 2,
            },
        )
        self.assertEqual(saved_report["min_eval_precision_counts"], {"bf16_amp": 2})
        self.assertEqual(saved_report["min_eval_hardware_counts"], {"test_hardware": 2})
        self.assertEqual(saved_report["required_label_domains"], ["precision_profile"])
        self.assertEqual(
            saved_report["min_eval_label_counts"],
            {"precision_profile": 2, "pseudo": 0, "source": 0},
        )
        self.assertEqual(saved_report["materialization"]["source_precision_config"], "fp32_ieee")
        self.assertTrue(saved_report["materialization"]["source_precision_confirmed"])
        self.assertEqual(saved_report["materialization"]["base_pairs"], 3)
        self.assertEqual(saved_report["materialization"]["source_metadata_labels"], 2)
        self.assertEqual(saved_report["materialization"]["calibration_source_labels"], 1)
        self.assertEqual(saved_report["materialization"]["source_labels_total"], 3)
        self.assertEqual(saved_report["materialization"]["precision_labels"], 12)
        self.assertEqual(saved_report["materialization"]["pseudo_labels"], 4)
        self.assertTrue(saved_report["deployment_required"])
        self.assertTrue(saved_report["deployment_student_checkpoint_required"])
        self.assertEqual(saved_report["deployment_student_checkpoint"]["student_ckpt_paths"], [str(student_ckpt)])
        self.assertEqual(saved_report["deployment_student_checkpoint"]["deployment_ckpt_paths"], [str(student_ckpt)])
        self.assertEqual(saved_report["deployment"]["runtime_backend"], "torchscript")
        self.assertEqual(saved_report["deployment"]["deployment_metadata"], str(deployment_metadata))
        self.assertEqual(
            saved_report["eval_train_checkpoints"]["source_teacher"]["eval_ckpt_paths"],
            [str(source_ckpt)],
        )
        self.assertEqual(
            saved_report["eval_train_checkpoints"]["precision_teacher"]["train_checkpoints"],
            [str(transfer_ckpt)],
        )
        self.assertEqual(
            saved_report["eval_train_checkpoints"]["precision_student"]["eval_ckpt_paths"],
            [str(student_ckpt)],
        )
        self.assertEqual(
            saved_report["training"]["precision_teacher"]["split_label_domain_counts"]["train"],
            {"precision_profile": 2, "source": 2},
        )
        self.assertEqual(
            saved_report["training"]["precision_student"]["split_label_domain_counts"]["train"],
            {"precision_profile": 2, "source": 2},
        )
        self.assertEqual(
            saved_report["training"]["precision_teacher"]["split_precision_config_counts"]["train"],
            {"bf16_amp": 2, "fp32_ieee": 2},
        )
        self.assertEqual(
            saved_report["training"]["precision_student"]["split_precision_config_counts"]["train"],
            {"bf16_amp": 2, "fp32_ieee": 2},
        )
        self.assertEqual(
            saved_report["training"]["precision_teacher"]["split_hardware_id_counts"]["train"],
            {"test_hardware": 4},
        )
        self.assertEqual(saved_report["training"]["precision_teacher"]["split_unit"], "graph_signature")
        self.assertEqual(saved_report["training"]["precision_teacher"]["train_count"], 4)
        self.assertEqual(saved_report["training"]["precision_teacher"]["val_count"], 1)
        self.assertEqual(saved_report["training"]["precision_teacher"]["test_count"], 1)
        self.assertEqual(saved_report["training"]["precision_teacher"]["test_hash"], "eval-test-hash")
        self.assertEqual(
            saved_report["training"]["precision_student"]["split_hardware_id_counts"]["train"],
            {"test_hardware": 4},
        )
        self.assertEqual(saved_report["training"]["precision_student"]["split_unit"], "graph_signature")
        self.assertEqual(saved_report["training"]["precision_student"]["train_count"], 4)
        self.assertEqual(saved_report["training"]["precision_student"]["val_count"], 1)
        self.assertEqual(saved_report["training"]["precision_student"]["test_count"], 1)
        self.assertEqual(saved_report["training"]["precision_student"]["test_hash"], "eval-test-hash")
        self.assertEqual(saved_report["training"]["source_teacher"]["data_root"], "dataset_source")
        self.assertTrue(saved_report["training"]["source_teacher"]["source_precision_confirmed"])
        self.assertEqual(saved_report["training"]["source_teacher"]["source_precision_provenance"], "original-profiler-notes.md#tf32")
        self.assertEqual(saved_report["training"]["source_teacher"]["checkpoint_metadata_count"], 1)
        self.assertEqual(
            saved_report["training"]["source_teacher"]["checkpoint_source_precision"]["provenance"],
            "original-profiler-notes.md#tf32",
        )
        self.assertEqual(saved_report["training"]["precision_teacher"]["init_checkpoint"], str(source_ckpt))
        self.assertEqual(saved_report["training"]["precision_teacher"]["checkpoint_initialization_paths"], [str(source_ckpt)])
        self.assertEqual(saved_report["training"]["precision_student"]["teacher_ckpt_dir"], str(Path(tmp)))
        self.assertEqual(saved_report["training"]["precision_student"]["checkpoint_distillation_teacher_kinds"], ["multi"])
        self.assertEqual(saved_report["train_lineage"]["transfer_init_checkpoint"], str(source_ckpt))
        self.assertEqual(saved_report["train_lineage"]["student_checkpoint_teacher_paths"], [str(transfer_ckpt)])
        self.assertEqual(saved_report["runs"]["precision_student"]["precision_configs"], ["bf16_amp", "fp32_ieee"])
        self.assertEqual(saved_report["runs"]["precision_student"]["precision_config_counts"], {"bf16_amp": 8, "fp32_ieee": 8})
        self.assertEqual(saved_report["runs"]["precision_student"]["hardware_id_counts"], {"test_hardware": 8})
        self.assertEqual(saved_report["runs"]["precision_student"]["label_domains"], ["precision_profile"])
        self.assertEqual(saved_report["runs"]["precision_student"]["label_domain_counts"], {"precision_profile": 8})
        self.assertEqual(saved_report["runs"]["precision_student"]["split_unit"], "graph_signature")
        self.assertEqual(saved_report["runs"]["precision_student"]["graph_families"], ["family_slice_0", "family_slice_1"])

    def test_check_precision_transfer_results_allows_unknown_source_precision_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            report = Path(tmp) / "report.json"
            materialization_report = Path(tmp) / "precision_materialization_report.json"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            provenance = "SOURCE_PRECISION_PROVENANCE.md#unknown"
            rows = [
                train_result_row(
                    "precision_large_teacher_source",
                    "dataset_source",
                    [str(source_ckpt)],
                    precision_config="source_domain_unknown",
                    source_precision_provenance=provenance,
                    source_precision_confirmed=False,
                ),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            materialization_report.write_text(
                json.dumps(
                    {
                        "source_precision_config": "source_domain_unknown",
                        "source_precision_confirmed": False,
                        "source_precision_provenance": provenance,
                        "base_pairs": 3,
                        "source_metadata_labels": 2,
                        "calibration_source_labels": 1,
                        "precision_labels": 12,
                        "pseudo_labels": 0,
                    }
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--materialization-report",
                    str(materialization_report),
                    "--require-source-precision-provenance",
                    "--expected-source-precision-config",
                    "source_domain_unknown",
                    "--expected-source-precision-provenance",
                    provenance,
                    "--require-train-events",
                    "--require-source-train-provenance",
                    "--require-train-checkpoint-metadata",
                    "--report-out",
                    str(report),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            saved_report = json.loads(report.read_text())

        self.assertIn("precision transfer result check passed", result.stdout)
        self.assertTrue(saved_report["source_precision_provenance_required"])
        self.assertFalse(saved_report["source_precision_confirmed_required"])
        self.assertTrue(saved_report["source_train_provenance_required"])
        self.assertFalse(saved_report["source_train_precision_confirmed_required"])
        self.assertFalse(saved_report["materialization"]["source_precision_confirmed"])
        self.assertFalse(saved_report["training"]["source_teacher"]["source_precision_confirmed"])
        self.assertFalse(saved_report["training"]["source_teacher"]["checkpoint_source_precision"]["confirmed"])
        self.assertEqual(saved_report["training"]["source_teacher"]["source_precision_provenance"], provenance)

    def test_check_precision_transfer_results_rejects_source_baseline_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                {
                    "event": "eval_complete",
                    "run_id": "accuracy_baseline",
                    "data_root": "dataset_source",
                    "mean_mape": 1.0,
                    "num_test_graphs": 8,
                },
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], mean_mape=1.3),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--baseline-run-id",
                    "accuracy_baseline",
                    "--max-source-baseline-mape-delta",
                    "0.1",
                    "--required-precision",
                    "bf16_amp",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_teacher mean_mape delta vs baseline 0.3 exceeds threshold 0.1", result.stdout)
        self.assertNotIn("baseline metrics_by_precision", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_train_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-train-events",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing source_teacher train row", result.stdout)
        self.assertIn("missing precision_teacher train row", result.stdout)
        self.assertIn("missing precision_student train row", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_source_train_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-checkpoint-files",
                    "--require-train-events",
                    "--require-source-train-provenance",
                    "--require-source-train-precision-confirmed",
                    "--expected-source-precision-config",
                    "fp32_ieee",
                    "--expected-source-precision-provenance",
                    "original-profiler-notes.md#tf32",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_teacher train source precision is not confirmed", result.stdout)
        self.assertIn("source_teacher train source precision provenance is empty", result.stdout)
        self.assertIn("source_teacher train source precision provenance does not match expected value", result.stdout)

    def test_check_precision_transfer_results_rejects_bad_train_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                train_result_row(
                    "precision_large_teacher_source",
                    "dataset_source",
                    [str(source_ckpt)],
                    source_precision_provenance="original-profiler-notes.md#tf32",
                    checkpoint_source_precision={
                        "precision_config": "bf16_amp",
                        "hardware_id": "test_hardware",
                        "provenance": "",
                        "confirmed": False,
                    },
                ),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    checkpoint_initialization={"path": str(student_ckpt), "strict": True},
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    checkpoint_distillation_teacher={"kind": "none", "count": 0, "paths": []},
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-checkpoint-files",
                    "--require-train-events",
                    "--require-source-train-provenance",
                    "--require-source-train-precision-confirmed",
                    "--require-train-checkpoint-metadata",
                    "--expected-source-precision-config",
                    "fp32_ieee",
                    "--expected-source-precision-provenance",
                    "original-profiler-notes.md#tf32",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_teacher checkpoint source precision is not confirmed", result.stdout)
        self.assertIn("source_teacher checkpoint source precision provenance is empty", result.stdout)
        self.assertIn("source_teacher checkpoint precision_config 'bf16_amp' does not match expected 'fp32_ieee'", result.stdout)
        self.assertIn("source_teacher checkpoint source precision provenance does not match expected value", result.stdout)
        self.assertIn("precision_teacher checkpoint initialization path", result.stdout)
        self.assertIn("does not match train init_checkpoint", result.stdout)
        self.assertIn("precision_student checkpoint distillation_teacher metadata is missing", result.stdout)

    def test_check_precision_transfer_results_rejects_broken_train_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            unrelated_ckpt = Path(tmp) / "unrelated_seernet_multi.pt"
            wrong_teacher = Path(tmp) / "wrong_teacher.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt, unrelated_ckpt, wrong_teacher):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                train_result_row(
                    "precision_large_teacher_source",
                    "dataset_source",
                    [str(source_ckpt)],
                    source_precision_provenance="original-profiler-notes.md#tf32",
                ),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(unrelated_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp) / "not_transfer_dir"),
                    teacher_paths=[str(wrong_teacher)],
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-train-events",
                    "--require-train-checkpoint-metadata",
                    "--require-train-lineage",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("train lineage precision_teacher init_checkpoint", result.stdout)
        self.assertIn("is not listed in source_teacher train checkpoints", result.stdout)
        self.assertIn("train lineage precision_teacher checkpoint initialization path", result.stdout)
        self.assertIn("train lineage precision_student teacher_ckpt_dir", result.stdout)
        self.assertIn("does not match precision_teacher output/checkpoint directory", result.stdout)
        self.assertIn("train lineage precision_student checkpoint teacher path", result.stdout)
        self.assertIn("is not listed in precision_teacher train checkpoints", result.stdout)

    def test_check_precision_transfer_results_rejects_eval_train_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            transfer_eval_ckpt = Path(tmp) / "transfer_eval_other.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, transfer_eval_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_eval_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-train-events",
                    "--require-eval-train-checkpoints",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("eval/train checkpoint linkage precision_teacher eval checkpoint", result.stdout)
        self.assertIn("is not listed in precision_teacher train checkpoints", result.stdout)
        self.assertIn("eval/train checkpoint linkage precision_teacher train checkpoint", result.stdout)
        self.assertIn("is not listed in precision_teacher eval checkpoints", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_train_label_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            source_only_counts = {"train": {"source": 4}, "val": {"source": 1}, "test": {"source": 1}}
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    split_label_domain_counts=source_only_counts,
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                    split_label_domain_counts=source_only_counts,
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-train-events",
                    "--required-train-label-domain",
                    "source,precision_profile",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("precision_teacher train split missing required label-domain 'precision_profile'", result.stdout)
        self.assertIn("precision_student train split missing required label-domain 'precision_profile'", result.stdout)

    def test_check_precision_transfer_results_rejects_low_train_label_domain_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            low_precision_counts = {
                "train": {"source": 4, "precision_profile": 1},
                "val": {"precision_profile": 1},
                "test": {"precision_profile": 1},
            }
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    split_label_domain_counts=low_precision_counts,
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                    split_label_domain_counts=low_precision_counts,
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-train-precision-labels",
                    "2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher train split label-domain 'precision_profile' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student train split label-domain 'precision_profile' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_low_train_precision_config_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            low_bf16_counts = {
                "train": {"bf16_amp": 1, "fp32_ieee": 4},
                "val": {"bf16_amp": 1},
                "test": {"bf16_amp": 1},
            }
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    split_precision_config_counts=low_bf16_counts,
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                    split_precision_config_counts=low_bf16_counts,
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-train-precision-count",
                    "bf16_amp=2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher train split precision_config 'bf16_amp' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student train split precision_config 'bf16_amp' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_low_train_hardware_id_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            low_hardware_counts = {
                "train": {"a100": 1, "rtx3090": 4},
                "val": {"a100": 1},
                "test": {"a100": 1},
            }
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    split_hardware_id_counts=low_hardware_counts,
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                    split_hardware_id_counts=low_hardware_counts,
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-train-hardware-count",
                    "a100=2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher train split hardware_id 'a100' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student train split hardware_id 'a100' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_train_split_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)]),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    init_checkpoint=str(source_ckpt),
                    split_unit="pair",
                    train_count=1,
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    teacher_ckpt_dir=str(Path(tmp)),
                    teacher_paths=[str(transfer_ckpt)],
                    split_unit="pair",
                    train_count=1,
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-split-unit",
                    "graph_signature",
                    "--require-train-events",
                    "--min-train-split-count",
                    "2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher train split_unit 'pair' does not match required 'graph_signature'",
            result.stdout,
        )
        self.assertIn(
            "precision_teacher train train_count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student train split_unit 'pair' does not match required 'graph_signature'",
            result.stdout,
        )
        self.assertIn(
            "precision_student train train_count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_limited_train_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            source_ckpt = Path(tmp) / "source_seernet_multi.pt"
            transfer_ckpt = Path(tmp) / "transfer_seernet_multi.pt"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            for ckpt in (source_ckpt, transfer_ckpt, student_ckpt):
                ckpt.write_bytes(b"checkpoint")
            rows = [
                train_result_row("precision_large_teacher_source", "dataset_source", [str(source_ckpt)], limit=32),
                train_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    [str(transfer_ckpt)],
                    limit=32,
                    init_checkpoint=str(source_ckpt),
                ),
                train_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    [str(student_ckpt)],
                    limit=32,
                    teacher_ckpt_dir=str(Path(tmp)),
                ),
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(source_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(transfer_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-checkpoint-files",
                    "--require-unlimited-train-data",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_teacher train data.limit 32 is not unlimited", result.stdout)
        self.assertIn("precision_teacher train data.limit 32 is not unlimited", result.stdout)
        self.assertIn("precision_student train data.limit 32 is not unlimited", result.stdout)

    def test_check_precision_transfer_results_rejects_unconfirmed_source_precision_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            materialization_report = Path(tmp) / "precision_materialization_report.json"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            materialization_report.write_text(
                json.dumps(
                    {
                        "source_precision_config": "fp32_ieee",
                        "source_precision_confirmed": False,
                        "source_precision_provenance": "",
                        "base_pairs": 0,
                        "source_metadata_labels": 0,
                        "calibration_source_labels": 0,
                        "precision_labels": 0,
                        "pseudo_labels": 0,
                    }
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--materialization-report",
                    str(materialization_report),
                    "--require-source-precision-confirmed",
                    "--min-materialized-precision-labels",
                    "1",
                    "--min-materialized-base-pairs",
                    "1",
                    "--min-materialized-source-labels",
                    "1",
                    "--min-materialized-pseudo-labels",
                    "1",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("materialization report source precision is not confirmed", result.stdout)
        self.assertIn("materialization report source precision provenance is empty", result.stdout)
        self.assertIn("materialization report precision_labels 0 is below required 1", result.stdout)
        self.assertIn("materialization report base_pairs 0 is below required 1", result.stdout)
        self.assertIn("materialization report source labels 0 is below required 1", result.stdout)
        self.assertIn("materialization report pseudo_labels 0 is below required 1", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_deployment_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            missing_metadata = Path(tmp) / "missing_deployment_metadata.json"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                deploy_result_row("precision_distill_student_128", "dataset_precision", str(missing_metadata)),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-deployment-eval",
                    "--require-deployment-metadata",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deployment metadata file is missing", result.stdout)

    def test_check_precision_transfer_results_rejects_deployment_student_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            student_ckpt = Path(tmp) / "student_seernet_multi.pt"
            deployment_ckpt = Path(tmp) / "other_student_seernet_multi.pt"
            student_ckpt.write_bytes(b"student")
            deployment_ckpt.write_bytes(b"deployment")
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(student_ckpt)]),
                deploy_result_row("precision_distill_student_128", "dataset_precision", "", ckpt_paths=[str(deployment_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-deployment-eval",
                    "--require-deployment-student-checkpoint",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deployment checkpoint linkage deployment_student checkpoint", result.stdout)
        self.assertIn("is not listed in precision_student eval checkpoints", result.stdout)
        self.assertIn("deployment checkpoint linkage precision_student checkpoint", result.stdout)
        self.assertIn("is not listed in deployment_student eval checkpoints", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_precision_label_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    label_domains=["source"],
                ),
                eval_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    label_domains=["source"],
                ),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--required-label-domain",
                    "precision_profile",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("precision_teacher missing required label-domain slice(s): precision_profile", result.stdout)
        self.assertIn("precision_student missing required label-domain slice(s): precision_profile", result.stdout)

    def test_check_precision_transfer_results_rejects_low_eval_label_domain_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    label_domains=["precision_profile"],
                    label_domain_counts={"precision_profile": 1},
                ),
                eval_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    label_domains=["precision_profile"],
                    label_domain_counts={"precision_profile": 1},
                ),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--required-label-domain",
                    "precision_profile",
                    "--min-eval-precision-labels",
                    "2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher eval label-domain 'precision_profile' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student eval label-domain 'precision_profile' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_low_eval_precision_config_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    precision_config_counts={"fp32_ieee": 8, "bf16_amp": 1},
                ),
                eval_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    precision_config_counts={"fp32_ieee": 8, "bf16_amp": 1},
                ),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-eval-precision-count",
                    "bf16_amp=2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher eval precision_config 'bf16_amp' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student eval precision_config 'bf16_amp' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_low_eval_hardware_id_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    hardware_id_counts={"a100": 1, "rtx3090": 8},
                ),
                eval_result_row(
                    "precision_distill_student_128",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    hardware_id_counts={"a100": 1, "rtx3090": 8},
                ),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-eval-hardware-count",
                    "a100=2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "precision_teacher eval hardware_id 'a100' count 1 is below required 2",
            result.stdout,
        )
        self.assertIn(
            "precision_student eval hardware_id 'a100' count 1 is below required 2",
            result.stdout,
        )

    def test_check_precision_transfer_results_rejects_missing_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            missing_ckpt = Path(tmp) / "missing_seernet_multi.pt"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], ckpt_paths=[str(missing_ckpt)]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(missing_ckpt)]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], ckpt_paths=[str(missing_ckpt)]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--require-checkpoint-files",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint file is missing", result.stdout)

    def test_check_precision_transfer_results_rejects_insufficient_slice_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--min-label-domain-slices",
                    "2",
                    "--min-batch-size-slices",
                    "2",
                    "--min-resource-regime-slices",
                    "2",
                    "--min-graph-signature-slices",
                    "2",
                    "--min-graph-family-slices",
                    "2",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metrics_by_label_domain has 1 slice(s), below required 2", result.stdout)
        self.assertIn("metrics_by_batch_size has 1 slice(s), below required 2", result.stdout)
        self.assertIn("metrics_by_resource_regime has 1 slice(s), below required 2", result.stdout)
        self.assertIn("metrics_by_graph_signature has 1 slice(s), below required 2", result.stdout)
        self.assertIn("metrics_by_graph_family has 1 slice(s), below required 2", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_required_split_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], split_unit=None),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], split_unit=None),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], split_unit=None),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--required-split-unit",
                    "graph_signature",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("split_unit None does not match required 'graph_signature'", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_split_hash_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], test_hash=None),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"], test_hash=None),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], test_hash=None),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--required-split-unit",
                    "graph_signature",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_teacher test_hash is missing", result.stdout)
        self.assertIn("precision_teacher test_hash is missing", result.stdout)
        self.assertIn("precision_student test_hash is missing", result.stdout)

    def test_check_precision_transfer_results_rejects_checkpoint_split_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"], checkpoint_test_hash="eval-test-hash"),
                eval_result_row(
                    "precision_large_teacher_transfer",
                    "dataset_precision",
                    ["fp32_ieee", "bf16_amp"],
                    checkpoint_test_hash="different-checkpoint-hash",
                ),
                eval_result_row("precision_distill_student_128", "dataset_precision", ["fp32_ieee", "bf16_amp"], checkpoint_test_hash="eval-test-hash"),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                    "--required-split-unit",
                    "graph_signature",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("precision_teacher test_hash 'eval-test-hash' does not match checkpoint_test_hash 'different-checkpoint-hash'", result.stdout)

    def test_check_precision_transfer_results_rejects_missing_student_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.jsonl"
            rows = [
                eval_result_row("precision_large_teacher_source", "dataset_source", ["fp32_ieee"]),
                eval_result_row("precision_large_teacher_transfer", "dataset_precision", ["fp32_ieee", "bf16_amp"]),
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_precision_transfer_results.py"),
                    "--results",
                    str(results),
                    "--source-data-root",
                    "dataset_source",
                    "--precision-data-root",
                    "dataset_precision",
                    "--required-precision",
                    "bf16_amp",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing precision_student eval row", result.stdout)


if __name__ == "__main__":
    unittest.main()
