from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
import tarfile
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
    DEFAULT_TEMPLATE_SEED,
    DEFAULT_PRECISION_SWEEP,
    NODE_TYPES,
    GraphRecord,
    generate_model_source,
    load_records,
    low_precision_focus_families,
    low_precision_focus_reasons,
    materialize_template_records,
    parse_args as parse_pack_args,
    parse_precision_sweep,
    resolve_generation_workers,
    select_subset,
    write_pack,
)
from nrp_calibration_pack.profile.generated_model_runtime import GraphModel  # noqa: E402
from nrp_calibration_pack.template_catalog import (  # noqa: E402
    TE_LOW_PRECISION_TRANSFORMER_FAMILIES,
    TEMPORAL_SEQUENCE_BUCKETS,
    template_family_counts,
)
from perfseer.architecture_schema import ARCHITECTURE_FAMILY_QUOTAS, FEATURE_SCHEMA_VERSION, NODE_TYPES as ARCH_NODE_TYPES  # noqa: E402
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


def linear_graph(width: int = 16) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph["input_specs"] = [{"name": "input0", "shape": [1, width], "dtype": "float32", "kind": "float"}]
    graph.add_node(
        0,
        feature=feature(
            "Gemm",
            {"linear_in_features": width, "linear_out_features": width, "linear_bias": 1},
            {
                "bytes": width * 2,
                "weight_size": width * width,
                "batch_size": 1,
                "input_size_with_weight": width + width * width,
                "input_size": width,
                "output_size": width,
                "input_features": width,
                "output_features": width,
                "input_channels": width,
                "output_channels": width,
                "input_h": 1,
                "input_w": 1,
                "output_h": 1,
                "output_w": 1,
            },
        ),
    )
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
        self.old_common = sys.modules.get("transformer_engine.common")
        self.old_recipe = sys.modules.get("transformer_engine.common.recipe")
        self.old_fp8 = sys.modules.get("transformer_engine.pytorch.fp8")
        root = types.ModuleType("transformer_engine")
        root.__path__ = []
        pytorch = types.ModuleType("transformer_engine.pytorch")
        pytorch.__package__ = "transformer_engine"
        pytorch.__path__ = []
        pytorch.is_fp8_available = lambda: True
        pytorch.is_nvfp4_available = lambda: True
        fp8 = types.ModuleType("transformer_engine.pytorch.fp8")
        fp8.is_fp8_available = lambda: True
        fp8.is_nvfp4_available = lambda: True
        common = types.ModuleType("transformer_engine.common")
        common.__path__ = []
        recipe = types.ModuleType("transformer_engine.common.recipe")

        class Format:
            HYBRID = "HYBRID"
            E4M3 = "E4M3"
            E5M2 = "E5M2"

        class DelayedScaling:
            def __init__(self, fp8_format=None):
                self.fp8_format = fp8_format

        class NVFP4BlockScaling:
            pass

        recipe.Format = Format
        recipe.DelayedScaling = DelayedScaling
        recipe.NVFP4BlockScaling = NVFP4BlockScaling
        root.pytorch = pytorch
        root.common = common
        common.recipe = recipe
        sys.modules["transformer_engine"] = root
        sys.modules["transformer_engine.pytorch"] = pytorch
        sys.modules["transformer_engine.pytorch.fp8"] = fp8
        sys.modules["transformer_engine.common"] = common
        sys.modules["transformer_engine.common.recipe"] = recipe
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
        if self.old_fp8 is None:
            sys.modules.pop("transformer_engine.pytorch.fp8", None)
        else:
            sys.modules["transformer_engine.pytorch.fp8"] = self.old_fp8
        if self.old_common is None:
            sys.modules.pop("transformer_engine.common", None)
        else:
            sys.modules["transformer_engine.common"] = self.old_common
        if self.old_recipe is None:
            sys.modules.pop("transformer_engine.common.recipe", None)
        else:
            sys.modules["transformer_engine.common.recipe"] = self.old_recipe


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

    def test_template_catalog_mode_uses_canonical_defaults(self) -> None:
        args = parse_pack_args([])

        self.assertEqual(args.catalog_mode, "template")
        self.assertEqual(args.out_dir, "nrp_calibration_pack")
        self.assertEqual(args.seed, DEFAULT_TEMPLATE_SEED)
        self.assertEqual(args.subset_size, DEFAULT_SUBSET_SIZE)
        self.assertEqual(args.low_precision_focus, "none")
        with self.assertRaises(SystemExit):
            parse_pack_args(["--catalog-mode", "dataset"])

    def test_low_precision_focus_parser_selects_te_transformer_families(self) -> None:
        args = parse_pack_args(["--low-precision-focus", "te_transformer"])

        self.assertEqual(args.low_precision_focus, "te_transformer")
        self.assertEqual(low_precision_focus_families("te_transformer"), TE_LOW_PRECISION_TRANSFORMER_FAMILIES)
        with self.assertRaises(SystemExit):
            parse_pack_args(["--low-precision-focus", "cnn"])

    def test_canonical_feature_layout_uses_expanded_schema(self) -> None:
        from perfseer_optimized.data import FeatureConfig, feature_layout

        cfg = FeatureConfig()
        layout = feature_layout(cfg)

        self.assertEqual(cfg.feature_schema_version, FEATURE_SCHEMA_VERSION)
        self.assertIn("type_Attention", layout.node_names)
        self.assertIn("tensor_rank", layout.node_names)
        self.assertIn("architecture_family_bert_encoder", layout.global_names)
        self.assertIn("variant_kind_mixed_stress", layout.global_names)
        self.assertEqual(cfg.signature(), FeatureConfig(feature_schema_version=FEATURE_SCHEMA_VERSION).signature())

    def test_template_full_quota_table_is_exact(self) -> None:
        counts = template_family_counts(DEFAULT_SUBSET_SIZE)

        self.assertEqual(counts, dict(ARCHITECTURE_FAMILY_QUOTAS))
        self.assertEqual(sum(counts.values()), 10000)

    def test_generation_workers_resolve_auto_and_serial_modes(self) -> None:
        self.assertGreaterEqual(parse_pack_args([]).generation_workers, 1)
        self.assertGreaterEqual(resolve_generation_workers(0), 1)
        self.assertEqual(parse_pack_args(["--generation-workers", "1"]).generation_workers, 1)
        with self.assertRaises(SystemExit):
            parse_pack_args(["--generation-workers", "-1"])

    def test_load_records_can_parse_dataset_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_dir = tmp_path / "cg" / "cg"
            label_dir = tmp_path / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for idx in range(2):
                stem = f"bs{idx + 1}_parallel_{idx}"
                with (graph_dir / f"{stem}.pkl").open("wb") as fh:
                    pickle.dump(sequential_graph(), fh)
                (label_dir / f"{stem}.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")

            records = load_records(tmp_path, generation_workers=2)

        self.assertEqual([item.stem for item in records], ["bs1_parallel_0", "bs2_parallel_1"])
        self.assertEqual([item.batch_size for item in records], [1, 2])

    def test_precision_sweep_rejects_ambiguous_bf32(self) -> None:
        with self.assertRaisesRegex(ValueError, "bf32 is ambiguous"):
            parse_precision_sweep("bf32")

    def test_precision_sweep_accepts_nvfp4_aliases_and_rejects_mxfp8(self) -> None:
        self.assertEqual(parse_precision_sweep("fp4,nvfp4,nvfp4_te"), ("nvfp4_te",))
        with self.assertRaisesRegex(ValueError, "mxfp8 is out of scope"):
            parse_precision_sweep("mxfp8")

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

    def test_template_pack_manifest_fields_and_operator_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            records = materialize_template_records(pack_dir, 30, DEFAULT_TEMPLATE_SEED, force=True)
            written, failures = write_pack(records, records, pack_dir, "compile", precision_sweep=("fp32_ieee",), generation_workers=1)
            rows = [
                json.loads(line)
                for line in (pack_dir / "manifest" / "subset_manifest.jsonl").read_text().splitlines()
            ]
            coverage = json.loads((pack_dir / "coverage_summary.json").read_text())

        self.assertEqual(written, 30)
        self.assertEqual(failures, 0)
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        for field in ("architecture_family", "variant_kind", "variant_signature", "input_specs"):
            self.assertIn(field, rows[0])
        family_counts: dict[str, int] = {}
        for row in rows:
            family_counts[row["architecture_family"]] = family_counts.get(row["architecture_family"], 0) + 1
        self.assertTrue(all(count == 2 for count in family_counts.values()))
        self.assertEqual(set(family_counts), set(ARCHITECTURE_FAMILY_QUOTAS))
        self.assertIn("Attention", coverage["operator_coverage"])
        self.assertIn("GRU", coverage["operator_coverage"])
        self.assertIn("GraphMessage", coverage["operator_coverage"])
        self.assertEqual(set(coverage["operator_coverage"]).issubset(set(ARCH_NODE_TYPES)), True)

    def test_temporal_templates_use_dataset_like_sequence_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            records = materialize_template_records(pack_dir, 30, DEFAULT_TEMPLATE_SEED, force=True)
            written, failures = write_pack(records, records, pack_dir, "compile", precision_sweep=("fp32_ieee",), generation_workers=1)
            rows = [
                json.loads(line)
                for line in (pack_dir / "manifest" / "subset_manifest.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(written, 30)
        self.assertEqual(failures, 0)
        temporal_rows = [row for row in rows if row["architecture_family"] in {"gru_temporal", "lstm_temporal"}]
        self.assertTrue(temporal_rows)
        for row in temporal_rows:
            self.assertIn(row["input_specs"][0]["shape"][1], TEMPORAL_SEQUENCE_BUCKETS)
            self.assertGreaterEqual(row["input_specs"][0]["shape"][1], 64)

    def test_te_transformer_focus_pack_generates_fp8_nvfp4_safe_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            records = materialize_template_records(
                pack_dir,
                8,
                DEFAULT_TEMPLATE_SEED,
                force=True,
                families=low_precision_focus_families("te_transformer"),
            )
            written, failures = write_pack(
                records,
                records,
                pack_dir,
                "compile",
                precision_sweep=("fp8_te_hybrid", "nvfp4_te"),
                generation_workers=1,
                low_precision_focus="te_transformer",
            )
            rows = [
                json.loads(line)
                for line in (pack_dir / "manifest" / "subset_manifest.jsonl").read_text().splitlines()
                if '"precision_config": "fp8_te_hybrid"' in line
            ]
            coverage = json.loads((pack_dir / "coverage_summary.json").read_text())
            modules = [import_generated((pack_dir / row["model_file"]).read_text(), tmp)[0] for row in rows]

        self.assertEqual(written, 8)
        self.assertEqual(failures, 0)
        self.assertEqual(coverage["low_precision_focus"], "te_transformer")
        self.assertEqual({row["architecture_family"] for row in rows}.issubset(set(TE_LOW_PRECISION_TRANSFORMER_FAMILIES)), True)
        for module in modules:
            self.assertEqual(low_precision_focus_reasons(module.NODE_SPECS, "te_transformer"), [])
            self.assertEqual(GraphModel(module.NODE_SPECS).low_precision_unsupported_reasons("fp8_te_hybrid"), [])
            self.assertEqual(GraphModel(module.NODE_SPECS).low_precision_unsupported_reasons("nvfp4_te"), [])
            batch, seq, _hidden = module.INPUT_SPECS[0]["shape"]
            self.assertEqual((batch * seq) % 32, 0)

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

    def test_write_pack_parallel_generation_replaces_failures_in_order(self) -> None:
        bad = nx.DiGraph()
        bad.add_node(0, feature=feature("Unsupported", mem=memory_info()))
        first_good = sequential_graph()
        second_good = sequential_graph()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {}
            for name, graph in (("bad", bad), ("first_good", first_good), ("second_good", second_good)):
                path = tmp_path / f"{name}.pkl"
                with path.open("wb") as fh:
                    pickle.dump(graph, fh)
                paths[name] = path

            bad_record = record("bad", graph_path=str(paths["bad"]), label_path=str(tmp_path / "bad.txt"))
            first_good_record = record("first_good", graph_path=str(paths["first_good"]), label_path=str(tmp_path / "first_good.txt"))
            second_good_record = record("second_good", graph_path=str(paths["second_good"]), label_path=str(tmp_path / "second_good.txt"))
            written, failures = write_pack(
                [bad_record, first_good_record],
                [bad_record, first_good_record, second_good_record],
                tmp_path / "pack",
                "compile",
                generation_workers=2,
            )
            rows = [
                json.loads(line)
                for line in (tmp_path / "pack" / "manifest" / "subset_manifest.jsonl").read_text().splitlines()
                if '"precision_config": "fp32_ieee"' in line
            ]

        self.assertEqual(written, 2)
        self.assertEqual(failures, 1)
        self.assertEqual([row["model_id"] for row in rows], ["calib_0000", "calib_0001"])
        self.assertEqual([row["original_stem"] for row in rows], ["first_good", "second_good"])

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
                    "--hardware-id",
                    "rtx4090",
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
            result = json.loads((tmp_path / "out" / "results_shard0.jsonl").read_text().splitlines()[0])
            hardware = json.loads((tmp_path / "out" / "hardware_shard0.json").read_text())

        self.assertTrue(label_exists)
        self.assertEqual(parsed.shape, (6,))
        self.assertEqual(result["hardware_id"], "rtx4090")
        self.assertEqual(result["hardware"]["hardware_id"], "rtx4090")
        self.assertEqual(hardware["hardware_id"], "rtx4090")

    def test_profiler_resumes_completed_labels_without_duplicate_rows(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "graph.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            graph_record = record("original_stem", graph_path=str(graph_path), label_path=str(tmp_path / "original_stem.txt"))
            write_pack([graph_record], [graph_record], tmp_path / "pack", "compile", precision_sweep=("fp32_ieee",))
            cmd = [
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
                "0",
                "--infer-repeats",
                "1",
                "--train-repeats",
                "1",
                "--device",
                "cpu",
            ]

            subprocess.run(cmd, check=True, text=True, capture_output=True)
            second = subprocess.run(cmd, check=True, text=True, capture_output=True)
            result_lines = (tmp_path / "out" / "results_shard0.jsonl").read_text().splitlines()

        self.assertEqual(len(result_lines), 1)
        self.assertIn("resume checkpoint: 1 completed label(s)", second.stdout)
        self.assertIn("calib_0000::fp32_ieee: skip_completed", second.stdout)

    def test_profiler_uses_profile_dataset_specs_for_repeat_timing(self) -> None:
        graph = sequential_graph()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "graph.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            graph_record = record("original_stem", graph_path=str(graph_path), label_path=str(tmp_path / "original_stem.txt"))
            write_pack([graph_record], [graph_record], tmp_path / "pack", "compile", precision_sweep=("fp32_ieee",))
            manifest = tmp_path / "pack" / "manifest" / "subset_manifest.jsonl"
            profile_data_dir = tmp_path / "profile_datasets"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "make_profile_datasets.py"),
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(profile_data_dir),
                    "--train-repeats",
                    "2",
                    "--infer-repeats",
                    "3",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
                    "--manifest",
                    str(manifest),
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
                    "--profile-dataset-dir",
                    str(profile_data_dir),
                    "--device",
                    "cpu",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            label_path = tmp_path / "out" / "label" / "label" / "calib_0000_fp32_ieee.txt"
            parsed = parse_dataset_label(str(label_path))
            result_path = tmp_path / "out" / "results_shard0.jsonl"
            result = json.loads(result_path.read_text().splitlines()[0])

        self.assertEqual(parsed.shape, (6,))
        self.assertEqual(result["profile_dataset"]["source"], "profile_dataset_dir")
        self.assertEqual(result["profile_dataset"]["train_repeats"], 2)
        self.assertEqual(result["profile_dataset"]["infer_repeats"], 3)
        self.assertEqual(len(result["details"]["train"]["raw_iter_ms"]), 2)
        self.assertEqual(len(result["details"]["infer"]["raw_iter_ms"]), 3)
        self.assertNotIn("repeat_unit", result["details"]["train"])

    def test_profiler_handles_template_multi_input_graph_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            records = materialize_template_records(pack_dir, 15, DEFAULT_TEMPLATE_SEED, force=True)
            graph_record = next(item for item in records if item.family_tuple == ("gat_graph",))
            write_pack([graph_record], [graph_record], pack_dir, "compile", precision_sweep=("fp32_ieee",))
            manifest = pack_dir / "manifest" / "subset_manifest.jsonl"
            profile_data_dir = tmp_path / "profile_datasets"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "make_profile_datasets.py"),
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(profile_data_dir),
                    "--train-repeats",
                    "1",
                    "--infer-repeats",
                    "1",
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
                    "--manifest",
                    str(manifest),
                    "--models-dir",
                    str(pack_dir / "models"),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--num-shards",
                    "1",
                    "--precision-config",
                    "fp32_ieee",
                    "--warmup",
                    "1",
                    "--profile-dataset-dir",
                    str(profile_data_dir),
                    "--device",
                    "cpu",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            label_path = tmp_path / "out" / "label" / "label" / "calib_0000_fp32_ieee.txt"
            parsed = parse_dataset_label(str(label_path))
            result = json.loads((tmp_path / "out" / "results_shard0.jsonl").read_text().splitlines()[0])

        self.assertEqual(parsed.shape, (6,))
        self.assertEqual(len(result["input_specs"]), 2)
        self.assertEqual(result["input_specs"][1]["kind"], "adjacency")
        self.assertEqual(result["status"], "ok")

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

    def test_profiler_default_optimizer_is_adam(self) -> None:
        module = import_run_profile_module()

        args = module.parse_args(
            [
                "--manifest",
                "manifest.jsonl",
                "--models-dir",
                "models",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.optimizer, "adam")

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

    def test_precision_selection_auto_and_fp4_aliases(self) -> None:
        module = import_run_profile_module()

        auto = module.precision_selection(types.SimpleNamespace(precision_sweep="auto", precision_config=None))
        explicit = module.precision_selection(types.SimpleNamespace(precision_sweep="fp4,nvfp4,nvfp4_te", precision_config=None))

        self.assertTrue(auto.auto)
        self.assertIsNone(auto.configs)
        self.assertFalse(explicit.auto)
        self.assertEqual(explicit.configs, ["nvfp4_te"])
        with self.assertRaisesRegex(ValueError, "auto cannot be combined"):
            module.precision_selection(types.SimpleNamespace(precision_sweep="auto,fp32_ieee", precision_config=None))
        with self.assertRaisesRegex(ValueError, "mxfp8 is out of scope"):
            module.precision_selection(types.SimpleNamespace(precision_sweep="mxfp8", precision_config=None))

    def test_auto_precision_resolution_tracks_gpu_generation(self) -> None:
        module = import_run_profile_module()
        args = types.SimpleNamespace(fp8_backend="transformer_engine")
        cases = {
            (8, 0): ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp"],
            (8, 9): ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp", "fp8_te_hybrid"],
            (9, 0): ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp", "fp8_te_hybrid"],
            (12, 0): ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp", "fp8_te_hybrid", "nvfp4_te"],
        }
        module.bf16_support_probe = lambda _device, _cc: (True, {"fake": True})

        with FakeTransformerEngine():
            for cc, expected in cases.items():
                module.compute_capability_tuple = lambda _device, cc=cc: cc
                configs, probes = module.resolve_auto_precision_configs(torch.device("cuda"), args)
                self.assertEqual(configs, expected)
                self.assertEqual(probes["nvfp4_te"]["details"]["compute_capability"], f"{cc[0]}.{cc[1]}")

    def test_transformer_engine_cuda_include_auto_discovers_python_wheel_headers(self) -> None:
        module = import_run_profile_module()
        old_env = module.os.environ.get("NVTE_CUDA_INCLUDE_DIR")
        old_find_spec = module.importlib.util.find_spec
        details: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include = root / "nvidia" / "cu13" / "include"
            include.mkdir(parents=True)
            (include / "cuda_runtime.h").write_text("// fake cuda header\n")

            def fake_find_spec(name: str) -> object:
                if name == "nvidia":
                    return types.SimpleNamespace(submodule_search_locations=[str(root / "nvidia")])
                return old_find_spec(name)

            module.os.environ.pop("NVTE_CUDA_INCLUDE_DIR", None)
            module.importlib.util.find_spec = fake_find_spec
            try:
                found = module.configure_transformer_engine_cuda_include(details)
            finally:
                module.importlib.util.find_spec = old_find_spec
                if old_env is None:
                    module.os.environ.pop("NVTE_CUDA_INCLUDE_DIR", None)
                else:
                    module.os.environ["NVTE_CUDA_INCLUDE_DIR"] = old_env

        self.assertEqual(found, str(include))
        self.assertEqual(details["nvte_cuda_include_dir"], str(include))
        self.assertEqual(details["nvte_cuda_include_dir_source"], "python_package:cu13")

    def test_nvfp4_policy_requires_transformer_engine_probe_and_bf16(self) -> None:
        module = import_run_profile_module()
        module.compute_capability_tuple = lambda _device: (12, 0)
        args = types.SimpleNamespace(fp8_backend="transformer_engine")

        with FakeTransformerEngine() as te:
            te.is_nvfp4_available = lambda: False
            module.bf16_support_probe = lambda _device, _cc: (True, {"fake": True})
            unavailable = module.precision_runtime("nvfp4_te", torch.device("cuda"), args)
            te.is_nvfp4_available = lambda: True
            module.bf16_support_probe = lambda _device, _cc: (False, {"fake": False})
            no_bf16 = module.precision_runtime("nvfp4_te", torch.device("cuda"), args)

        self.assertFalse(unavailable.supported)
        self.assertIn("NVFP4", unavailable.unsupported_reason or "")
        self.assertFalse(no_bf16.supported)
        self.assertIn("BF16", no_bf16.unsupported_reason or "")

    def test_generated_low_precision_op_gate_allows_te_safe_rows_only(self) -> None:
        dense_mem = {
            "batch_size": 32,
            "input_size": 1024,
            "output_size": 1024,
            "input_features": 32,
            "output_features": 32,
            "input_channels": 32,
            "output_channels": 32,
        }
        dense = [
            {
                "id": 0,
                "type": "Gemm",
                "args": {"linear_in_features": 32, "linear_out_features": 32, "linear_bias": 1},
                "memory_info": dense_mem,
                "preds": [],
            },
            {"id": 1, "type": "LayerNormalization", "args": {}, "memory_info": dense_mem, "preds": [0]},
        ]
        bad_dim = [
            {
                "id": 0,
                "type": "Gemm",
                "args": {"linear_in_features": 8, "linear_out_features": 16, "linear_bias": 1},
                "memory_info": dense_mem,
                "preds": [],
            }
        ]
        conv = [
            {
                "id": 0,
                "type": "Conv",
                "args": {"conv_kernel_size": 3, "conv_stride": 1, "conv_padding": 1, "conv_groups": 1},
                "memory_info": memory_info(input_channels=3, output_channels=16),
                "preds": [],
            }
        ]
        bad_nvfp4_leading = [
            {
                "id": 0,
                "type": "Gemm",
                "args": {"linear_in_features": 32, "linear_out_features": 32, "linear_bias": 1},
                "memory_info": {
                    "batch_size": 1,
                    "rank": 3,
                    "sequence_length": 48,
                    "input_size": 1 * 48 * 32 * 4,
                    "output_size": 1 * 48 * 32 * 4,
                    "input_features": 32,
                    "output_features": 32,
                    "input_channels": 32,
                    "output_channels": 32,
                },
                "preds": [],
            }
        ]

        self.assertEqual(GraphModel(dense).low_precision_unsupported_reasons("nvfp4_te"), [])
        self.assertIn("dimensions 8->16", GraphModel(bad_dim).low_precision_unsupported_reasons("fp8_te_hybrid")[0])
        self.assertIn("not TE low-precision safe", GraphModel(conv).low_precision_unsupported_reasons("nvfp4_te")[0])
        self.assertEqual(GraphModel(bad_nvfp4_leading).low_precision_unsupported_reasons("fp8_te_hybrid"), [])
        self.assertIn("divisible by 32", GraphModel(bad_nvfp4_leading).low_precision_unsupported_reasons("nvfp4_te")[0])

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
            parsed_shape = parse_dataset_label(str(label_path)).shape
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())

        self.assertTrue(graph_exists)
        self.assertTrue(label_exists)
        self.assertEqual(parsed_shape, (6,))
        self.assertEqual(report["precision_labels"], 1)
        self.assertEqual(report["calibration_source_labels"], 0)
        source_rows = [row for row in metadata if row.get("label_domain") == "source"]
        precision_rows = [row for row in metadata if row.get("label_domain") != "source"]
        self.assertEqual(len(source_rows), 0)
        self.assertEqual(len(precision_rows), 1)
        self.assertEqual(precision_rows[0]["hardware_id"], "a100")
        self.assertEqual(precision_rows[0]["base_label_file"], "")
        self.assertEqual(precision_rows[0]["hardware_features"]["sm_count"], 108)

    def test_materialize_precision_dataset_accepts_repeated_result_dirs(self) -> None:
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

            result_dirs = []
            for hardware_id, sm_count in (("rtx3090", 82), ("rtx4090", 128)):
                results_dir = tmp_path / f"results_{hardware_id}"
                results_dir.mkdir()
                result_dirs.append(results_dir)
                result_row = {
                    "status": "ok",
                    "model_id": "calib_0000",
                    "graph_id": "calib_0000",
                    "hardware_id": hardware_id,
                    "precision_config": "fp32_ieee",
                    "profile_point_id": f"calib_0000::{hardware_id}::fp32_ieee",
                    "label": {"train": "1|2|3|4|5|6|7", "infer": "1|2|3|4|5|6|7"},
                    "hardware": {
                        "hardware_id": hardware_id,
                        "gpu_name": f"NVIDIA {hardware_id.upper()}",
                        "compute_capability": "8.9",
                        "multi_processor_count": sm_count,
                        "total_memory_mib": 24576,
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
                    str(result_dirs[0]),
                    "--results-dir",
                    str(result_dirs[1]),
                    "--out-root",
                    str(tmp_path / "precision_dataset"),
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            out_root = tmp_path / "precision_dataset"
            label_3090 = out_root / "label" / "label" / "calib_0000_rtx3090_fp32_ieee.txt"
            label_4090 = out_root / "label" / "label" / "calib_0000_rtx4090_fp32_ieee.txt"
            label_3090_exists = label_3090.exists()
            label_4090_exists = label_4090.exists()
            metadata = [json.loads(line) for line in (out_root / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            report = json.loads((out_root / "precision_materialization_report.json").read_text())
            precision_rows = [row for row in metadata if row.get("label_domain") == "precision_profile"]
            label_names = {label_3090.name, label_4090.name}

        self.assertTrue(label_3090_exists)
        self.assertTrue(label_4090_exists)
        self.assertEqual(report["precision_labels"], 2)
        self.assertEqual(report["precision_labels_by_hardware"], {"rtx3090": 1, "rtx4090": 1})
        self.assertEqual(report["precision_labels_by_config"], {"fp32_ieee": 2})
        self.assertEqual(report["label_domain_counts"]["precision_profile"], 2)
        self.assertEqual({row["hardware_id"] for row in precision_rows}, {"rtx3090", "rtx4090"})
        self.assertEqual({Path(row["label_file"]).name for row in precision_rows}, label_names)

    def test_materialize_precision_dataset_tags_optional_base_labels(self) -> None:
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

    def test_materialize_precision_dataset_records_unknown_base_precision_as_unconfirmed(self) -> None:
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
                    "unknown",
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

        self.assertEqual(report["source_precision_config"], "unknown")
        self.assertFalse(report["source_precision_confirmed"])
        self.assertEqual(metadata[0]["precision_config"], "unknown")
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
        self.assertEqual(report["unsupported_low_precision_rows"], 1)
        self.assertEqual(report["rejected_rows_file"], "precision_rejected_rows.jsonl")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["precision_config"], "fp8_te_hybrid")
        self.assertEqual(rejected[0]["fallback_policy"], "record_unsupported_generated_ops")

    def test_materialize_precision_dataset_reports_rejected_nvfp4_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_dir = tmp_path / "pack"
            results_dir = tmp_path / "results"
            pack_dir.mkdir()
            results_dir.mkdir()
            result_row = {
                "status": "unsupported_low_precision_op",
                "model_id": "calib_0000",
                "graph_id": "calib_0000",
                "precision_config": "nvfp4_te",
                "profile_point_id": "calib_0000::nvfp4_te",
                "error": "node 0 Conv is not TE low-precision safe",
                "precision": {
                    "precision_config": "nvfp4_te",
                    "backend": "transformer_engine",
                    "fallback_policy": "record_unsupported_low_precision_op",
                },
                "hardware": {"gpu_name": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0"},
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
        self.assertEqual(report["unsupported_low_precision_rows"], 1)
        self.assertEqual(report["skipped_by_precision"]["nvfp4_te"], {"unsupported_low_precision_op": 1})
        self.assertEqual(report["fallback_policy_counts"]["record_unsupported_low_precision_op"], {"nvfp4_te": 1})
        self.assertEqual(rejected[0]["precision_config"], "nvfp4_te")

    def test_source_only_tar_excludes_pkls_and_rebuilds_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_path = tmp_path / "linear.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(linear_graph(), fh)
            graph_record = record("bs1_linear_0", graph_path=str(graph_path), label_path=str(tmp_path / "linear.txt"))
            pack_dir = tmp_path / "pack"
            write_pack([graph_record], [graph_record], pack_dir, "compile", precision_sweep=("fp32_ieee",))

            results_dir = tmp_path / "results_rtx5090"
            results_dir.mkdir()
            result_row = {
                "status": "ok",
                "model_id": "calib_0000",
                "graph_id": "calib_0000",
                "precision_config": "nvfp4_te",
                "profile_point_id": "calib_0000::nvfp4_te",
                "label": {"train": "1|2|3|4|5|6|7", "infer": "1|2|3|4|5|6|7"},
                "precision": {"precision_config": "nvfp4_te", "backend": "transformer_engine"},
                "hardware_id": "rtx5090",
                "hardware": {"hardware_id": "rtx5090", "gpu_name": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0"},
            }
            (results_dir / "results_shard0.jsonl").write_text(json.dumps(result_row) + "\n")
            source_tar = tmp_path / "source_labels.tar.gz"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "package_source_tar.py"),
                    "--pack-dir",
                    str(pack_dir),
                    "--results-dir",
                    str(results_dir),
                    "--out",
                    str(source_tar),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with tarfile.open(source_tar, "r:gz") as tar:
                names = tar.getnames()

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "rebuild_source_tar_dataset.py"),
                    "--source-tar",
                    str(source_tar),
                    "--out-root",
                    str(tmp_path / "rebuilt_dataset"),
                    "--force",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            rebuilt = tmp_path / "rebuilt_dataset"
            metadata = [json.loads(line) for line in (rebuilt / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            label_path = rebuilt / "label" / "label" / "calib_0000_rtx5090_nvfp4_te.txt"
            graph_exists = (rebuilt / "cg" / "cg" / "calib_0000.pkl").exists()
            label_exists = label_path.exists()
            parsed_shape = parse_dataset_label(str(label_path)).shape

        self.assertIn("pack/models/calib_0000.py", names)
        self.assertIn("results/results_rtx5090/results_shard0.jsonl", names)
        self.assertIn("replay/nrp_calibration_pack/profile/run_profile.py", names)
        self.assertIn("replay/scripts/rebuild_source_tar_dataset.py", names)
        self.assertFalse(any(name.endswith(".pkl") for name in names))
        self.assertTrue(graph_exists)
        self.assertTrue(label_exists)
        self.assertEqual(parsed_shape, (6,))
        self.assertEqual(metadata[0]["precision_config"], "nvfp4_te")
        self.assertEqual(metadata[0]["hardware_id"], "rtx5090")

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
                "--hardware-id",
                "rtx4090",
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
        self.assertIn("--hardware-id rtx4090", yaml)
        self.assertIn("--precision-sweep fp32_ieee,bf16_amp", yaml)
        self.assertIn("--fp8-backend transformer_engine", yaml)
        self.assertNotIn("--train-epochs", yaml)

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
        self.assertNotIn("--train-epochs", yaml)

    def test_source_workflow_script_renders_prepare_profile_package_jobs(self) -> None:
        result = subprocess.run(
            [
                str(ROOT / "nrp_calibration_pack" / "submit_nrp_source_workflow.sh"),
                "--namespace",
                "test-ns",
                "--image",
                "example/perfseer-ngc:latest",
                "--pvc",
                "calibration-pvc",
                "--gpu-product",
                "NVIDIA-GeForce-RTX-5090",
                "--hardware-id",
                "rtx5090",
                "--parallelism",
                "2",
                "--completions",
                "3",
                "--subset-size",
                "7",
                "--low-precision-focus",
                "te_transformer",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        yaml = result.stdout
        self.assertIn("name: perfseer-nrp-source-prepare-sources", yaml)
        self.assertIn("name: perfseer-nrp-source-profile-labels", yaml)
        self.assertIn("name: perfseer-nrp-source-package-results", yaml)
        self.assertIn("--subset-size 7", yaml)
        self.assertIn("--low-precision-focus te_transformer", yaml)
        self.assertIn("--precision-sweep fp32_ieee", yaml)
        self.assertIn("--precision-sweep auto", yaml)
        self.assertIn("--optimizer adam", yaml)
        self.assertIn("--hardware-id rtx5090", yaml)
        self.assertIn("completionMode: Indexed", yaml)
        self.assertIn("completions: 3", yaml)
        self.assertIn("parallelism: 2", yaml)
        self.assertIn("NVIDIA-GeForce-RTX-5090", yaml)
        self.assertIn("package_source_tar.py", yaml)
        self.assertIn("perfseer_rtx5090_source_labels.tar.gz", yaml)
        self.assertNotIn("dataset/cg/cg", yaml)

    def test_root_markdown_entrypoint_is_readme_only(self) -> None:
        root_markdown = sorted(path.name for path in ROOT.glob("*.md"))

        self.assertIn("README.md", root_markdown)
        self.assertIn("AGENTS.md", root_markdown)
        self.assertIn("plan.md", root_markdown)
        self.assertEqual([name for name in root_markdown if name not in {"AGENTS.md", "README.md", "plan.md"}], [])

    def test_hardware_filter_keeps_one_gpu_and_multiple_precisions(self) -> None:
        from perfseer_optimized.data import FeatureConfig, feature_config_for_pair, precision_from_label_path, split_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cg" / "cg").mkdir(parents=True)
            (root / "label" / "label").mkdir(parents=True)
            metadata = []
            for graph_idx in range(2):
                graph_id = f"calib_{graph_idx:04d}"
                graph_file = f"cg/cg/{graph_id}.pkl"
                with (root / graph_file).open("wb") as fh:
                    pickle.dump(sequential_graph(), fh)
                for hardware_id, precisions in {"rtx4090": ["fp32_ieee", "tf32"], "rtx3090": ["fp32_ieee"]}.items():
                    for precision_config in precisions:
                        label_name = f"{graph_id}_{hardware_id}_{precision_config}.txt"
                        label_file = f"label/label/{label_name}"
                        (root / label_file).write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
                        metadata.append(
                            {
                                "graph_id": graph_id,
                                "graph_file": graph_file,
                                "label_file": label_file,
                                "hardware_id": hardware_id,
                                "precision_config": precision_config,
                                "label_domain": "precision_profile",
                            }
                        )
            (root / "label" / "precision_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata) + "\n")

            cfg = FeatureConfig(hardware_id="rtx4090", include_precision_features=True)
            train, val, test = split_dataset(str(root), seed=3, split_unit="pair", hardware_id="rtx4090", feature_config=cfg)
            selected = train + val + test

            self.assertEqual(len(selected), 4)
            self.assertEqual({feature_config_for_pair(cfg, gp, lp).hardware_id for gp, lp in selected}, {"rtx4090"})
            self.assertEqual({precision_from_label_path(gp, lp) for gp, lp in selected}, {"fp32_ieee", "tf32"})
            with self.assertRaisesRegex(ValueError, "no labels found for hardware_id='rtx5090'"):
                split_dataset(str(root), seed=3, split_unit="pair", hardware_id="rtx5090", feature_config=FeatureConfig(hardware_id="rtx5090"))

    def test_hardware_distill_flow_dry_run_trains_teacher_from_scratch(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_hardware_distill_flow.py"),
                "--data-root",
                "dataset",
                "--hardware-id",
                "rtx4090",
                "--teacher-epochs",
                "1",
                "--student-epochs",
                "1",
                "--split-unit",
                "graph",
                "--limit",
                "32",
                "--deploy-eval-profile",
                "src/perfseer-optimized/configs/eval_profiles/cpu_torchscript_fp32.yaml",
                "--dry-run",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        stdout = result.stdout
        self.assertIn("# train canonical v2 hardware teacher", stdout)
        self.assertIn("# distill canonical v2 hardware student", stdout)
        self.assertIn("# evaluate hardware teacher", stdout)
        self.assertIn("# evaluate hardware student", stdout)
        self.assertIn("# evaluate deployment student", stdout)
        self.assertIn("src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml", stdout)
        self.assertIn("src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml", stdout)
        self.assertIn("--run-id v2_teacher_rtx4090", stdout)
        self.assertIn("--run-id v2_student_rtx4090", stdout)
        self.assertIn("--teacher-ckpt-dir runs/optimized/v2_teacher_rtx4090", stdout)
        self.assertIn("--hardware-id rtx4090", stdout)
        self.assertIn("--data-root dataset", stdout)
        self.assertIn("--split-unit graph", stdout)
        self.assertNotIn("--init-checkpoint", stdout)
        self.assertNotIn("transfer", stdout)

    def test_train_cli_overrides_hardware_filter_and_rejects_bad_precision(self) -> None:
        cfg = {"run": {}, "data": {}, "features": {}, "train": {}}
        args = parse_train_args(["--precision-config", "tf32", "--hardware-id", "rtx4090"])
        resolved = apply_train_overrides(cfg, args)

        self.assertEqual(resolved["features"]["precision_config"], "tf32")
        self.assertEqual(resolved["features"]["hardware_id"], "rtx4090")
        self.assertEqual(resolved["data"]["hardware_id"], "rtx4090")

        bad_args = parse_train_args(["--precision-config", "bf32"])
        with self.assertRaises(ValueError):
            apply_train_overrides({"run": {}, "data": {}, "features": {}, "train": {}}, bad_args)

    def test_nvfp4_precision_features_use_fp4_encoding(self) -> None:
        from perfseer_optimized.data import FeatureConfig, precision_config_index, precision_hardware_config

        cfg = FeatureConfig(precision_config="fp4", include_precision_features=True)
        resolved = precision_hardware_config(cfg)["resolved_precision"]

        self.assertEqual(precision_config_index("nvfp4"), precision_config_index("nvfp4_te"))
        self.assertEqual(resolved["weight_dtype"], "fp4_e2m1")
        self.assertEqual(resolved["activation_dtype"], "fp4_e2m1")
        self.assertEqual(resolved["grad_dtype"], "fp4_e2m1")
        self.assertEqual(resolved["tensorcore_mode"], "fp4")
        self.assertEqual(resolved["fp4_format"], "nvfp4_e2m1")
        with self.assertRaisesRegex(ValueError, "mxfp8 is out of scope"):
            precision_config_index("mxfp8")


if __name__ == "__main__":
    unittest.main()
