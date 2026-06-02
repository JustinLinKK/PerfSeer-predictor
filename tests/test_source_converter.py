from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
