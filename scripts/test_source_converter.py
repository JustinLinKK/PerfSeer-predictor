from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_source_converter import SourceModelSpec, convert_source_to_networkx, convert_source_to_pyg_data
from perfseer_source_converter.convert import main as convert_cli


def ensure_optimized_alias() -> None:
    if "perfseer_optimized" in sys.modules:
        return
    package_dir = SRC / "perfseer-optimized"
    spec = importlib.util.spec_from_file_location(
        "perfseer_optimized",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load perfseer_optimized test alias")
    module = importlib.util.module_from_spec(spec)
    sys.modules["perfseer_optimized"] = module
    spec.loader.exec_module(module)


class SourceConverterTests(unittest.TestCase):
    def write_model(self, tmp: str, source: str) -> Path:
        path = Path(tmp) / "model.py"
        path.write_text(textwrap.dedent(source))
        return path

    def test_sequential_cnn_graph_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class TinyCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 3, padding=1)
                        self.bn = nn.BatchNorm2d(4)
                        self.relu = nn.ReLU()
                        self.pool = nn.MaxPool2d(2)
                        self.flat = nn.Flatten()
                        self.fc = nn.Linear(4 * 4 * 4, 10)

                    def forward(self, x):
                        return self.fc(self.flat(self.pool(self.relu(self.bn(self.conv(x))))))
                """,
            )
            spec = SourceModelSpec(source, "TinyCNN", ((2, 3, 8, 8),))
            graph = convert_source_to_networkx(spec)

        types = [data["feature"]["type"] for _, data in graph.nodes(data=True)]
        self.assertEqual(types, ["Conv", "BatchNormalization", "Relu", "MaxPool", "Flatten", "Gemm"])
        self.assertEqual(list(graph.edges()), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        conv = graph.nodes[0]["feature"]
        self.assertEqual(set(conv), {"type", "args", "memory_info", "flops", "arith_intensity"})
        self.assertEqual(conv["args"]["conv_kernel_size"], 3)
        self.assertEqual(conv["memory_info"]["batch_size"], 2)
        self.assertGreater(conv["flops"], 0)

    def test_residual_add_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class Residual(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 3, 1)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(self.conv(x) + x)
                """,
            )
            graph = convert_source_to_networkx(SourceModelSpec(source, "Residual", ((1, 3, 4, 4),)))

        types = [data["feature"]["type"] for _, data in graph.nodes(data=True)]
        self.assertEqual(types, ["Conv", "Add", "Relu"])
        self.assertIn((0, 1), graph.edges())
        self.assertIn((1, 2), graph.edges())
        self.assertEqual(graph.nodes[1]["feature"]["memory_info"]["output_channels"], 3)

    def test_torch_cat_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch
                import torch.nn as nn

                class CatModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.left = nn.Conv2d(3, 2, 1)
                        self.right = nn.Conv2d(3, 3, 1)

                    def forward(self, x):
                        return torch.cat([self.left(x), self.right(x)], dim=1)
                """,
            )
            graph = convert_source_to_networkx(SourceModelSpec(source, "CatModel", ((1, 3, 5, 5),)))

        types = [data["feature"]["type"] for _, data in graph.nodes(data=True)]
        self.assertEqual(types, ["Conv", "Conv", "Concat"])
        self.assertEqual(set(graph.predecessors(2)), {0, 1})
        self.assertEqual(graph.nodes[2]["feature"]["memory_info"]["output_channels"], 5)

    def test_cli_writes_graph_and_data(self) -> None:
        ensure_optimized_alias()
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class Tiny(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 1)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(self.conv(x))
                """,
            )
            graph_out = Path(tmp) / "graph.pkl"
            data_out = Path(tmp) / "graph.pt"
            convert_cli(
                [
                    "--source",
                    str(source),
                    "--entry",
                    "Tiny",
                    "--input-shape",
                    "1,3,4,4",
                    "--graph-out",
                    str(graph_out),
                    "--out",
                    str(data_out),
                ]
            )
            with graph_out.open("rb") as fh:
                graph = pickle.load(fh)
            data = torch.load(data_out, map_location="cpu", weights_only=False)

        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(data.x.shape[0], 2)
        self.assertNotIn("y", data.keys())

    def test_pyg_data_runs_through_predictor(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, feature_layout
        from perfseer_optimized.model import SeerNetConfig, SeerNetMulti

        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class TinyCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 3, padding=1)
                        self.relu = nn.ReLU()
                        self.pool = nn.AvgPool2d(2)
                        self.flat = nn.Flatten()
                        self.fc = nn.Linear(4 * 4 * 4, 6)

                    def forward(self, x):
                        return self.fc(self.flat(self.pool(self.relu(self.conv(x)))))
                """,
            )
            feature_cfg = FeatureConfig()
            layout = feature_layout(feature_cfg)
            norm_stats = {
                "node_mean": np.zeros(layout.node_dim, dtype=np.float32),
                "node_std": np.ones(layout.node_dim, dtype=np.float32),
                "edge_mean": np.zeros(layout.edge_dim, dtype=np.float32),
                "edge_std": np.ones(layout.edge_dim, dtype=np.float32),
                "global_mean": np.zeros(layout.global_dim, dtype=np.float32),
                "global_std": np.ones(layout.global_dim, dtype=np.float32),
                "y_mean": np.zeros(6, dtype=np.float32),
                "y_std": np.ones(6, dtype=np.float32),
            }
            data = convert_source_to_pyg_data(
                SourceModelSpec(source, "TinyCNN", ((1, 3, 8, 8),)),
                norm_stats=norm_stats,
                feature_config=feature_cfg,
            )

        cfg = SeerNetConfig(
            node_dim=layout.node_dim,
            edge_dim=layout.edge_dim,
            global_dim=layout.global_dim,
            hidden=8,
            num_blocks=1,
            num_outputs=6,
        )
        model = SeerNetMulti(cfg).eval()
        with torch.no_grad():
            pred = model(data)
        self.assertEqual(tuple(pred.shape), (1, 6))
        self.assertNotIn("y", data.keys())

    def test_precision_hardware_features_extend_global_u(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, feature_layout

        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class TinyCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 1)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(self.conv(x))
                """,
            )
            feature_cfg = FeatureConfig(
                include_precision_features=True,
                include_hardware_features=True,
                precision_config="bf16_amp",
                hardware_id="a100_pilot",
                compute_capability=8.0,
                sm_count=108,
                memory_bandwidth_gbps=1555,
                vram_gib=80,
                peak_tf32_tflops=312,
                peak_fp16_bf16_tflops=312,
            )
            layout = feature_layout(feature_cfg)
            data = convert_source_to_pyg_data(
                SourceModelSpec(source, "TinyCNN", ((1, 3, 4, 4),)),
                feature_config=feature_cfg,
            )

        self.assertEqual(tuple(data.u.shape), (1, layout.global_dim))
        self.assertEqual(tuple(data.precision_config_idx.shape), (1,))
        self.assertEqual(tuple(data.batch_size_raw.shape), (1,))
        self.assertEqual(tuple(data.resource_regime_idx.shape), (1,))
        self.assertIn("activation_dtype_bf16", layout.global_names)
        self.assertIn("hardware_sm_count", layout.global_names)
        self.assertGreater(layout.global_dim, feature_layout(FeatureConfig()).global_dim)

    def test_checkpoint_precision_hardware_allowlist_validates_converter_overrides(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, feature_layout

        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class TinyCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 1)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(self.conv(x))
                """,
            )
            checkpoint_cfg = FeatureConfig(
                include_precision_features=True,
                include_hardware_features=True,
                precision_config="fp32_ieee",
                hardware_id="source_a100",
            )
            layout = feature_layout(checkpoint_cfg)
            norm_stats = {
                "node_mean": np.zeros(layout.node_dim, dtype=np.float32),
                "node_std": np.ones(layout.node_dim, dtype=np.float32),
                "edge_mean": np.zeros(layout.edge_dim, dtype=np.float32),
                "edge_std": np.ones(layout.edge_dim, dtype=np.float32),
                "global_mean": np.zeros(layout.global_dim, dtype=np.float32),
                "global_std": np.ones(layout.global_dim, dtype=np.float32),
                "y_mean": np.zeros(6, dtype=np.float32),
                "y_std": np.ones(6, dtype=np.float32),
            }
            ckpt_path = Path(tmp) / "seernet_multi.pt"
            torch.save(
                {
                    "metadata": {
                        "norm_stats": {key: value.tolist() for key, value in norm_stats.items()},
                        "feature_config": checkpoint_cfg.to_dict(),
                        "supported_precision_hardware": {
                            "precision_configs": ["bf16_amp", "fp32_ieee"],
                            "hardware_ids": ["a100", "source_a100"],
                            "precision_hardware_pairs": [
                                {"precision_config": "fp32_ieee", "hardware_id": "source_a100", "count": 1, "label_domains": ["source"]},
                                {"precision_config": "bf16_amp", "hardware_id": "a100", "count": 1, "label_domains": ["precision_profile"]},
                            ],
                        },
                    }
                },
                ckpt_path,
            )

            data = convert_source_to_pyg_data(
                SourceModelSpec(source, "TinyCNN", ((1, 3, 4, 4),)),
                ckpt_path=ckpt_path,
                feature_config={"precision_config": "bf16_amp", "hardware_id": "a100"},
            )

            with self.assertRaisesRegex(ValueError, "unsupported precision/hardware request"):
                convert_source_to_pyg_data(
                    SourceModelSpec(source, "TinyCNN", ((1, 3, 4, 4),)),
                    ckpt_path=ckpt_path,
                    feature_config={"precision_config": "fp16_amp", "hardware_id": "a100"},
                )

        self.assertEqual(tuple(data.precision_config_idx.shape), (1,))

    def test_precision_label_pairs_are_discovered_and_resolved(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import (
            FeatureConfig,
            feature_config_for_pair,
            label_domain_for_pair,
            list_precision_pairs,
            sample_weight_for_pair,
            supported_precision_hardware_summary,
            validate_precision_hardware_request,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            graph_path = graph_dir / "calib_0000.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(nx.DiGraph(), fh)
            (label_dir / "calib_0000.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            precision_label = label_dir / "calib_0000_a100_bf16_amp.txt"
            precision_label.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            (root / "label" / "precision_metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "graph_id": "calib_0000",
                                "graph_file": "cg/cg/calib_0000.pkl",
                                "label_file": "label/label/calib_0000.txt",
                                "label_stem": "calib_0000",
                                "precision_config": "fp32_ieee",
                                "hardware_id": "source_a100",
                                "label_domain": "source",
                                "is_base_label": True,
                                "hardware_features": {"compute_capability": "8.0", "sm_count": 108, "vram_gib": 80},
                            }
                        ),
                        json.dumps(
                            {
                                "graph_id": "calib_0000",
                                "graph_file": "cg/cg/calib_0000.pkl",
                                "label_file": "label/label/calib_0000_a100_bf16_amp.txt",
                                "label_stem": "calib_0000_a100_bf16_amp",
                                "precision_config": "bf16_amp",
                                "hardware_id": "a100",
                                "label_domain": "precision_profile",
                                "hardware_features": {"compute_capability": "8.0", "sm_count": 108, "vram_gib": 80},
                            }
                        ),
                    ]
                )
                + "\n"
            )

            pairs = list_precision_pairs(str(root))
            cfg = FeatureConfig(include_precision_features=True, base_label_weight=0.25, precision_label_weight=3.0)
            pair_cfg = feature_config_for_pair(cfg, str(graph_path), str(precision_label))
            sample_weight = sample_weight_for_pair(cfg, str(graph_path), str(precision_label))
            base_weight = sample_weight_for_pair(cfg, str(graph_path), str(label_dir / "calib_0000.txt"))
            precision_domain = label_domain_for_pair(str(graph_path), str(precision_label))
            base_domain = label_domain_for_pair(str(graph_path), str(label_dir / "calib_0000.txt"))
            supported = supported_precision_hardware_summary(pairs, cfg)

        self.assertEqual([Path(label).name for _graph, label in pairs], ["calib_0000.txt", "calib_0000_a100_bf16_amp.txt"])
        self.assertEqual(pair_cfg.precision_config, "bf16_amp")
        self.assertEqual(pair_cfg.hardware_id, "a100")
        self.assertEqual(pair_cfg.sm_count, 108)
        self.assertEqual(sample_weight, 3.0)
        self.assertEqual(base_weight, 0.25)
        self.assertEqual(precision_domain, "precision_profile")
        self.assertEqual(base_domain, "source")
        self.assertEqual(supported["precision_configs"], ["bf16_amp", "fp32_ieee"])
        self.assertEqual(supported["hardware_ids"], ["a100", "source_a100"])
        validate_precision_hardware_request(FeatureConfig(precision_config="bf16_amp", hardware_id="a100"), supported)
        with self.assertRaisesRegex(ValueError, "unsupported precision/hardware request"):
            validate_precision_hardware_request(FeatureConfig(precision_config="fp16_amp", hardware_id="a100"), supported)

    def test_eval_slice_helpers_group_precision_batch_and_resource(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.eval import rows_by_batch_size, rows_by_graph_family, rows_by_graph_signature, rows_by_index_slice, rows_by_precision
        from perfseer_optimized.data import PRECISION_CONFIG_VOCAB, RESOURCE_REGIME_VOCAB

        y_true = np.asarray([[10, 20, 30, 40, 50, 60], [12, 18, 33, 39, 52, 63]], dtype=np.float64)
        y_pred = y_true * np.asarray([[1.0, 1.1, 0.9, 1.0, 1.0, 1.0], [1.1, 1.0, 1.0, 0.9, 1.0, 1.0]], dtype=np.float64)
        precision_rows = rows_by_precision(y_true, y_pred, np.asarray([0, 2], dtype=np.int64))
        batch_rows = rows_by_batch_size(y_true, y_pred, np.asarray([1, 32], dtype=np.float64))
        resource_rows = rows_by_index_slice(y_true, y_pred, np.asarray([1, 3], dtype=np.int64), RESOURCE_REGIME_VOCAB, "resource")
        graph_rows = rows_by_graph_signature(y_true, y_pred, np.asarray([4, 64], dtype=np.int64), np.asarray([3, 150], dtype=np.int64))
        family_rows = rows_by_graph_family(y_true, y_pred, np.asarray(["pure:resnet", "mixed:bert|gpt"], dtype=object))

        self.assertIn(PRECISION_CONFIG_VOCAB[0], precision_rows)
        self.assertIn("bf16_amp", precision_rows)
        self.assertIn("bs_le_1", batch_rows)
        self.assertIn("bs_le_32", batch_rows)
        self.assertIn("memory_bound", resource_rows)
        self.assertIn("compute_bound", resource_rows)
        self.assertIn("tiny_chain_like", graph_rows)
        self.assertIn("medium_dense", graph_rows)
        self.assertIn("pure:resnet", family_rows)
        self.assertIn("mixed:bert|gpt", family_rows)

    def test_eval_rejects_unsupported_precision_hardware_rows(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig
        from perfseer_optimized.eval import build_test_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            graph_path = graph_dir / "graph_0000.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(nx.DiGraph(), fh)
            label_path = label_dir / "graph_0000_bf16_amp.txt"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            (root / "label" / "precision_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "graph_id": "graph_0000",
                        "graph_file": "cg/cg/graph_0000.pkl",
                        "label_file": "label/label/graph_0000_bf16_amp.txt",
                        "label_stem": "graph_0000_bf16_amp",
                        "precision_config": "bf16_amp",
                        "hardware_id": "a100",
                        "label_domain": "precision_profile",
                    }
                )
                + "\n"
            )

            args = type("Args", (), {"data_root": str(root), "seed": None, "limit": None})()
            ckpt = {
                "metadata": {
                    "feature_config": FeatureConfig(include_precision_features=True, include_hardware_features=True).to_dict(),
                    "split": {
                        "test_pair_ids": [{"graph_stem": "graph_0000", "label_stem": "graph_0000_bf16_amp"}],
                        "supported_precision_hardware": {
                            "precision_configs": ["fp32_ieee"],
                            "hardware_ids": ["source_domain_unknown"],
                            "precision_hardware_pairs": [
                                {"precision_config": "fp32_ieee", "hardware_id": "source_domain_unknown", "count": 1, "label_domains": ["source"]}
                            ],
                        },
                    },
                }
            }

            with self.assertRaisesRegex(ValueError, "unsupported precision/hardware request"):
                build_test_dataset(args, ckpt, {})

    def test_eval_records_split_metadata_for_reconstructed_test_pairs(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, NUM_TARGETS, feature_layout
        from perfseer_optimized.eval import build_test_dataset, split_result_fields

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class Tiny(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc = nn.Linear(4, 2)

                    def forward(self, x):
                        return self.fc(x)
                """,
            )
            graph = convert_source_to_networkx(SourceModelSpec(source, "Tiny", ((1, 4),)))
            graph_path = graph_dir / "graph_0000.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            label_path = label_dir / "graph_0000.txt"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            feature_cfg = FeatureConfig()
            layout = feature_layout(feature_cfg)
            stats = {
                "node_mean": np.zeros(layout.node_dim, dtype=np.float32),
                "node_std": np.ones(layout.node_dim, dtype=np.float32),
                "edge_mean": np.zeros(layout.edge_dim, dtype=np.float32),
                "edge_std": np.ones(layout.edge_dim, dtype=np.float32),
                "global_mean": np.zeros(layout.global_dim, dtype=np.float32),
                "global_std": np.ones(layout.global_dim, dtype=np.float32),
                "y_mean": np.zeros(NUM_TARGETS, dtype=np.float32),
                "y_std": np.ones(NUM_TARGETS, dtype=np.float32),
            }
            args = type("Args", (), {"data_root": str(root), "seed": None, "limit": None})()
            ckpt = {
                "metadata": {
                    "feature_config": feature_cfg.to_dict(),
                    "split": {
                        "seed": 13,
                        "split_unit": "graph_signature",
                        "test_hash": "checkpoint-test-hash",
                        "test_pair_ids": [{"graph_stem": "graph_0000", "label_stem": "graph_0000"}],
                    },
                }
            }

            ds, _cfg, data_root = build_test_dataset(args, ckpt, stats)

        split_meta = args._eval_split_metadata
        result_fields = split_result_fields(args)
        self.assertEqual(len(ds), 1)
        self.assertEqual(data_root, str(root))
        self.assertEqual(split_meta["split_unit"], "graph_signature")
        self.assertEqual(split_meta["checkpoint_test_hash"], "checkpoint-test-hash")
        self.assertEqual(split_meta["source"], "checkpoint_test_pair_ids")
        self.assertTrue(split_meta["test_pair_ids_reconstructed"])
        self.assertEqual(split_meta["test_count"], 1)
        self.assertRegex(split_meta["test_hash"], r"^[0-9a-f]{40}$")
        self.assertEqual(result_fields["split_unit"], "graph_signature")
        self.assertEqual(result_fields["test_hash"], split_meta["test_hash"])
        self.assertEqual(result_fields["evaluation_split"], split_meta)

    def test_deployment_metadata_sidecar_records_precision_schema(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, feature_layout, precision_hardware_config
        from perfseer_optimized.deploy import EvalProfile, PreparedRuntime
        from perfseer_optimized.eval_deploy import write_deployment_metadata

        cfg = FeatureConfig(
            include_precision_features=True,
            include_hardware_features=True,
            precision_config="bf16_amp",
            hardware_id="a100",
            sm_count=108.0,
        )
        supported = {
            "precision_configs": ["bf16_amp", "fp32_ieee"],
            "hardware_ids": ["a100"],
            "precision_hardware_pairs": [{"precision_config": "bf16_amp", "hardware_id": "a100", "count": 4}],
        }
        layout = feature_layout(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model0_torchscript.pt"
            artifact.write_bytes(b"artifact")
            runtime = PreparedRuntime(
                backend="torchscript",
                models=[],
                artifact_paths=[str(artifact)],
                statuses=[{"model_idx": 0, "backend": "torchscript", "status": "ok"}],
            )
            profile = EvalProfile(name="cpu_torchscript_fp32", runtime_backend="torchscript", batch_size=4)
            metadata_path = write_deployment_metadata(
                tmp,
                ckpt={
                    "metadata": {
                        "run_id": "precision_distill_student_128",
                        "precision_hardware_config": precision_hardware_config(cfg),
                        "supported_precision_hardware": supported,
                    }
                },
                ckpt_paths=["runs/optimized/precision_distill_student_128/seernet_multi.pt"],
                runtime=runtime,
                profile=profile,
                feature_cfg=cfg,
                data_root="dataset_precision",
                split_fields={
                    "split_unit": "graph_signature",
                    "test_hash": "abc123",
                    "evaluation_split": {"split_unit": "graph_signature", "test_hash": "abc123", "test_count": 12},
                },
            )
            metadata = json.loads(Path(metadata_path).read_text())

        self.assertIn(metadata_path, runtime.artifact_paths)
        self.assertEqual(metadata["run_id"], "precision_distill_student_128")
        self.assertEqual(metadata["runtime_backend"], "torchscript")
        self.assertEqual(metadata["runtime_backend_actual"], "torchscript")
        self.assertEqual(metadata["feature_config"]["precision_config"], "bf16_amp")
        self.assertEqual(metadata["precision_hardware_config"]["hardware_id"], "a100")
        self.assertEqual(metadata["supported_precision_hardware"], supported)
        self.assertEqual(metadata["required_inputs"]["u"]["dim"], layout.global_dim)
        self.assertEqual(metadata["split_unit"], "graph_signature")
        self.assertEqual(metadata["test_hash"], "abc123")

    def test_graph_split_keeps_precision_variants_together(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import split_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for idx in range(6):
                with (graph_dir / f"graph_{idx}.pkl").open("wb") as fh:
                    pickle.dump(nx.DiGraph(), fh)
                for suffix in ("", "_bf16_amp", "_fp16_amp"):
                    (label_dir / f"graph_{idx}{suffix}.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")

            splits = split_dataset(str(root), seed=7, split_unit="graph")

        seen: dict[str, int] = {}
        for split_idx, split in enumerate(splits):
            for graph_path, _label_path in split:
                graph_stem = Path(graph_path).stem
                if graph_stem in seen:
                    self.assertEqual(seen[graph_stem], split_idx)
                else:
                    seen[graph_stem] = split_idx
        self.assertEqual(set(seen), {f"graph_{idx}" for idx in range(6)})
        self.assertEqual(sum(len(split) for split in splits), 18)

    def test_pseudo_label_rows_are_train_only(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, sample_weight_for_pair, split_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            metadata = []
            for idx in range(6):
                graph_path = graph_dir / f"graph_{idx}.pkl"
                with graph_path.open("wb") as fh:
                    pickle.dump(nx.DiGraph(), fh)
                (label_dir / f"graph_{idx}.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            pseudo_label = label_dir / "graph_0_a100_bf16_amp_pseudo.txt"
            pseudo_label.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            metadata.append(
                {
                    "graph_id": "graph_0",
                    "graph_file": "cg/cg/graph_0.pkl",
                    "label_file": "label/label/graph_0_a100_bf16_amp_pseudo.txt",
                    "label_stem": "graph_0_a100_bf16_amp_pseudo",
                    "precision_config": "bf16_amp",
                    "hardware_id": "a100",
                    "label_domain": "pseudo",
                    "is_pseudo_label": True,
                }
            )
            (root / "label" / "precision_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata) + "\n")

            train, val, test = split_dataset(str(root), seed=11, split_unit="pair")
            pseudo_weight = sample_weight_for_pair(
                FeatureConfig(include_precision_features=True, base_label_weight=0.5, precision_label_weight=3.0, pseudo_label_weight=0.75),
                str(graph_dir / "graph_0.pkl"),
                str(pseudo_label),
            )

        pseudo_name = pseudo_label.name
        self.assertIn(pseudo_name, {Path(label).name for _graph, label in train})
        self.assertNotIn(pseudo_name, {Path(label).name for _graph, label in val})
        self.assertNotIn(pseudo_name, {Path(label).name for _graph, label in test})
        self.assertEqual(pseudo_weight, 0.75)

    def test_pseudo_label_rows_do_not_affect_norm_stats(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, compute_norm_stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            graph_path = graph_dir / "graph_0.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(nx.DiGraph(), fh)
            real_label = label_dir / "graph_0.txt"
            pseudo_label = label_dir / "graph_0_a100_bf16_amp_pseudo.txt"
            real_label.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            pseudo_label.write_text("{'train': '1000|2000|3000|4000|5000|6000|7000', 'infer': '1000|2000|3000|4000|5000|6000|7000'}\n")
            (root / "label" / "precision_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "graph_id": "graph_0",
                        "graph_file": "cg/cg/graph_0.pkl",
                        "label_file": "label/label/graph_0_a100_bf16_amp_pseudo.txt",
                        "label_stem": "graph_0_a100_bf16_amp_pseudo",
                        "precision_config": "bf16_amp",
                        "hardware_id": "a100",
                        "label_domain": "pseudo",
                        "is_pseudo_label": True,
                    }
                )
                + "\n"
            )

            stats = compute_norm_stats([(str(graph_path), str(real_label)), (str(graph_path), str(pseudo_label))], FeatureConfig())

        expected = np.log1p(np.asarray([2, 7, 1, 2, 7, 1], dtype=np.float64))
        self.assertTrue(np.allclose(stats["y_mean"], expected.astype(np.float32), atol=1e-6))
        self.assertTrue(np.allclose(stats["y_std"], np.ones(6, dtype=np.float32), atol=1e-6))

    def test_graph_signature_split_holds_out_structural_clusters(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import graph_signature_for_graph, split_dataset

        def dense_graph(node_count: int, edge_count: int) -> nx.DiGraph:
            graph = nx.DiGraph()
            graph.add_nodes_from(range(node_count))
            for src in range(node_count):
                for dst in range(node_count):
                    if src == dst:
                        continue
                    graph.add_edge(src, dst)
                    if graph.number_of_edges() >= edge_count:
                        return graph
            return graph

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            signatures: dict[str, str] = {}
            for idx in range(6):
                if idx < 3:
                    graph = nx.DiGraph()
                    graph.add_nodes_from(range(4))
                    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
                else:
                    graph = dense_graph(64, 150)
                graph_path = graph_dir / f"graph_{idx}.pkl"
                with graph_path.open("wb") as fh:
                    pickle.dump(graph, fh)
                signatures[graph_path.name] = graph_signature_for_graph(graph)
                for suffix in ("", "_bf16_amp"):
                    (label_dir / f"graph_{idx}{suffix}.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")

            splits = split_dataset(str(root), seed=3, split_unit="graph_signature")

        seen_signature_split: dict[str, int] = {}
        for split_idx, split in enumerate(splits):
            for graph_path, _label_path in split:
                signature = signatures[Path(graph_path).name]
                if signature in seen_signature_split:
                    self.assertEqual(seen_signature_split[signature], split_idx)
                else:
                    seen_signature_split[signature] = split_idx
        self.assertEqual(set(seen_signature_split), {"tiny_chain_like", "medium_dense"})
        self.assertEqual(sum(len(split) for split in splits), 12)

    def test_graph_family_split_holds_out_architecture_families(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import graph_family_for_path, split_dataset

        families = [
            ("resnet", "resnet", "resnet", "resnet"),
            ("resnet", "resnet", "resnet", "resnet"),
            ("resnet", "resnet", "resnet", "resnet"),
            ("bert", "bert", "gpt", "gpt"),
            ("bert", "bert", "gpt", "gpt"),
            ("bert", "bert", "gpt", "gpt"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            family_by_graph: dict[str, str] = {}
            for idx, family in enumerate(families):
                stem = f"bs1_s{family!r}_bnum{idx}"
                graph_path = graph_dir / f"{stem}.pkl"
                graph = nx.DiGraph()
                graph.add_nodes_from(range(4))
                with graph_path.open("wb") as fh:
                    pickle.dump(graph, fh)
                family_by_graph[graph_path.name] = graph_family_for_path(str(graph_path))
                for suffix in ("", "_bf16_amp"):
                    (label_dir / f"{stem}{suffix}.txt").write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")

            splits = split_dataset(str(root), seed=5, split_unit="graph_family")

        seen_family_split: dict[str, int] = {}
        for split_idx, split in enumerate(splits):
            for graph_path, _label_path in split:
                family = family_by_graph[Path(graph_path).name]
                if family in seen_family_split:
                    self.assertEqual(seen_family_split[family], split_idx)
                else:
                    seen_family_split[family] = split_idx
        self.assertEqual(set(seen_family_split), {"pure:resnet", "mixed:bert|gpt"})
        self.assertEqual(sum(len(split) for split in splits), 12)

    def test_log_ratio_target_mode_uses_source_baseline_and_inverts(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig, PerfSeerOptimizedDataset, compute_norm_stats, feature_config_for_pair, invert_targets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "cg" / "cg"
            label_dir = root / "label" / "label"
            graph_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            source = self.write_model(
                tmp,
                """
                import torch.nn as nn

                class TinyCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 4, 1)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(self.conv(x))
                """,
            )
            graph = convert_source_to_networkx(SourceModelSpec(source, "TinyCNN", ((1, 3, 4, 4),)))
            graph_path = graph_dir / "graph_0000.pkl"
            with graph_path.open("wb") as fh:
                pickle.dump(graph, fh)
            base_label = label_dir / "graph_0000.txt"
            precision_label = label_dir / "graph_0000_bf16_amp.txt"
            base_label.write_text("{'train': '2|10|0|0|0|0|100', 'infer': '4|20|0|0|0|0|200'}\n")
            precision_label.write_text("{'train': '4|20|0|0|0|0|200', 'infer': '8|40|0|0|0|0|400'}\n")
            cfg = FeatureConfig(target_mode="log_ratio_to_source", include_precision_features=True)
            pair_cfg = feature_config_for_pair(cfg, str(graph_path), str(precision_label))
            stats = compute_norm_stats([(str(graph_path), str(precision_label))], cfg)
            ds = PerfSeerOptimizedDataset(
                root=str(root),
                file_list=[(str(graph_path), str(precision_label))],
                split="ratio",
                norm_stats=stats,
                feature_config=cfg,
            )
            data = ds[0]
            restored = invert_targets(data.y, stats, pair_cfg, data.y_base_raw).numpy().reshape(-1)

        self.assertTrue(np.allclose(data.y_raw.numpy().reshape(-1), np.log(np.full(6, 2.0)), atol=1e-6))
        self.assertTrue(np.allclose(data.y.numpy().reshape(-1), np.zeros(6), atol=1e-6))
        self.assertTrue(np.allclose(data.y_eval_raw.numpy().reshape(-1), np.asarray([20, 200, 4, 40, 400, 8], dtype=np.float32)))
        self.assertTrue(np.allclose(restored, data.y_eval_raw.numpy().reshape(-1), atol=1e-5))

    def test_distillation_loads_multi_output_teacher_checkpoint(self) -> None:
        ensure_optimized_alias()
        from torch_geometric.data import Data

        from perfseer_optimized.model import SeerNetConfig, SeerNetMulti
        from perfseer_optimized.train import load_teacher_models, teacher_predictions

        cfg = SeerNetConfig(node_dim=3, edge_dim=2, global_dim=4, hidden=6, num_blocks=1, num_outputs=6)
        teacher = SeerNetMulti(cfg).eval()
        batch = Data(
            x=torch.randn(2, 3),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.randn(1, 2),
            u=torch.randn(1, 4),
            batch=torch.zeros(2, dtype=torch.long),
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "seernet_multi.pt"
            torch.save(
                {
                    "model_state_dict": teacher.state_dict(),
                    "model_config": cfg.to_dict(),
                    "model_name": "seernet_multi",
                    "epoch": 3,
                    "val_loss": 0.2,
                    "metadata": {"run_id": "precision_teacher"},
                },
                ckpt_path,
            )
            bundle = load_teacher_models(tmp, torch.device("cpu"))
            pred = teacher_predictions(bundle, batch)

        self.assertEqual(bundle.kind, "multi")
        self.assertEqual(len(bundle.paths), 1)
        self.assertIsNotNone(pred)
        self.assertEqual(tuple(pred.shape), (1, 6))

    def test_teacher_predictions_convert_to_student_target_space(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig
        from perfseer_optimized.train import TeacherBundle, teacher_predictions

        class ConstantTeacher(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros(2, 6)

        teacher_stats = {
            "y_mean": np.full(6, np.log1p(9.0), dtype=np.float32),
            "y_std": np.ones(6, dtype=np.float32),
        }
        student_stats = {
            "y_mean": np.zeros(6, dtype=np.float32),
            "y_std": np.ones(6, dtype=np.float32),
        }
        bundle = TeacherBundle(
            kind="multi",
            models=[ConstantTeacher()],
            paths=["teacher.pt"],
            norm_stats=[teacher_stats],
            feature_configs=[FeatureConfig(target_mode="absolute")],
            metric_indices=[None],
        )
        pred = teacher_predictions(bundle, object(), student_stats, FeatureConfig(target_mode="absolute"))

        self.assertIsNotNone(pred)
        self.assertTrue(torch.allclose(pred, torch.full((2, 6), float(np.log1p(9.0))), atol=1e-6))

    def test_teacher_predictions_convert_absolute_teacher_to_log_ratio_student(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import FeatureConfig
        from perfseer_optimized.train import TeacherBundle, teacher_predictions

        class ConstantTeacher(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros(2, 6)

        teacher_stats = {
            "y_mean": np.full(6, np.log1p(20.0), dtype=np.float32),
            "y_std": np.ones(6, dtype=np.float32),
        }
        student_stats = {
            "y_mean": np.zeros(6, dtype=np.float32),
            "y_std": np.ones(6, dtype=np.float32),
        }
        batch = type("Batch", (), {})()
        batch.y_base_raw = torch.full((2, 6), 10.0)
        bundle = TeacherBundle(
            kind="multi",
            models=[ConstantTeacher()],
            paths=["teacher.pt"],
            norm_stats=[teacher_stats],
            feature_configs=[FeatureConfig(target_mode="absolute")],
            metric_indices=[None],
        )
        pred = teacher_predictions(bundle, batch, student_stats, FeatureConfig(target_mode="log_ratio_to_source"))

        self.assertIsNotNone(pred)
        self.assertTrue(torch.allclose(pred, torch.full((2, 6), float(np.log(2.0))), atol=1e-6))

    def test_distillation_loads_metric_teacher_ensemble(self) -> None:
        ensure_optimized_alias()
        from torch_geometric.data import Data

        from perfseer_optimized.model import SeerNet, SeerNetConfig
        from perfseer_optimized.train import load_teacher_models, teacher_predictions

        cfg = SeerNetConfig(node_dim=3, edge_dim=2, global_dim=4, hidden=6, num_blocks=1, num_outputs=1)
        batch = Data(
            x=torch.randn(2, 3),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.randn(1, 2),
            u=torch.randn(1, 4),
            batch=torch.zeros(2, dtype=torch.long),
        )

        with tempfile.TemporaryDirectory() as tmp:
            for metric_idx in range(6):
                teacher = SeerNet(cfg).eval()
                torch.save(
                    {
                        "model_state_dict": teacher.state_dict(),
                        "model_config": cfg.to_dict(),
                        "model_name": "seernet",
                        "metric_idx": metric_idx,
                    },
                    Path(tmp) / f"seernet_metric{metric_idx}_metric.pt",
                )
            bundle = load_teacher_models(tmp, torch.device("cpu"))
            pred = teacher_predictions(bundle, batch)

        self.assertEqual(bundle.kind, "metric_ensemble")
        self.assertEqual(len(bundle.paths), 6)
        self.assertIsNotNone(pred)
        self.assertEqual(tuple(pred.shape), (1, 6))

    def test_distillation_label_domain_policy_preserves_precision_hard_labels(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.data import LABEL_DOMAIN_VOCAB
        from perfseer_optimized.train import distillation_hard_alphas, weighted_sample_distillation_loss

        batch = type("Batch", (), {})()
        batch.label_domain_idx = torch.tensor(
            [LABEL_DOMAIN_VOCAB.index("source"), LABEL_DOMAIN_VOCAB.index("precision_profile")],
            dtype=torch.long,
        )
        pred = torch.zeros(2, 6)
        hard_target = torch.ones(2, 6)
        teacher_target = torch.zeros(2, 6)
        sample_weight = torch.tensor([1.0, 3.0])
        cfg = {"alpha": 0.5, "source_hard_alpha": 0.0, "precision_hard_alpha": 1.0, "pseudo_hard_alpha": 0.0}

        alphas = distillation_hard_alphas(batch, hard_target, cfg)
        loss, _task_losses = weighted_sample_distillation_loss(
            pred,
            hard_target,
            teacher_target,
            "mse_logstd",
            1.0,
            None,
            sample_weight,
            alphas,
        )

        self.assertTrue(torch.equal(alphas, torch.tensor([0.0, 1.0])))
        self.assertAlmostEqual(float(loss.item()), 4.5, places=6)

    def test_train_initialization_checkpoint_loads_weights(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.model import SeerNetConfig, SeerNetMulti
        from perfseer_optimized.train import load_initial_weights

        cfg = SeerNetConfig(node_dim=3, edge_dim=2, global_dim=4, hidden=6, num_blocks=1, num_outputs=6)
        source = SeerNetMulti(cfg).eval()
        target = SeerNetMulti(cfg).eval()
        with torch.no_grad():
            for param in source.parameters():
                param.fill_(0.25)
            for param in target.parameters():
                param.zero_()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "seernet_multi.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "model_config": cfg.to_dict(),
                    "model_name": "seernet_multi",
                    "epoch": 9,
                    "val_loss": 0.123,
                    "metadata": {"run_id": "source_domain_teacher"},
                },
                ckpt_path,
            )
            info = load_initial_weights(
                target,
                {"train": {"init_checkpoint": str(tmp), "init_strict": True}},
                torch.device("cpu"),
                metric_idx=None,
            )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info["source_run_id"], "source_domain_teacher")
        self.assertEqual(info["source_epoch"], 9)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_train_cli_overrides_split_unit(self) -> None:
        ensure_optimized_alias()
        from perfseer_optimized.train import apply_overrides, load_config, parse_args

        args = parse_args(["--split-unit", "graph_family"])
        cfg = apply_overrides(load_config(None), args)

        self.assertEqual(cfg["data"]["split_unit"], "graph_family")


if __name__ == "__main__":
    unittest.main()
