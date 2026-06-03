from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
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
    DEFAULT_SUBSET_SIZE,
    NODE_TYPES,
    GraphRecord,
    generate_model_source,
    select_subset,
    write_pack,
)
from perfseer.data import parse_label as parse_dataset_label  # noqa: E402
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


class NrpCalibrationPackTests(unittest.TestCase):
    def test_default_subset_size_is_doubled_for_nrp_pack(self) -> None:
        self.assertEqual(DEFAULT_SUBSET_SIZE, 4096)

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
            row = json.loads(manifest_path.read_text().strip())
            coverage = json.loads((tmp_path / "pack" / "coverage_summary.json").read_text())

        self.assertEqual(written, 1)
        self.assertEqual(failures, 0)
        self.assertEqual(row["model_id"], "calib_0000")
        self.assertEqual(row["original_stem"], "original_stem")
        self.assertEqual(row["model_file"], "models/calib_0000.py")
        self.assertEqual(row["subset_graph_file"], "subset/cg/cg/calib_0000.pkl")
        self.assertEqual(row["label_file"], "label/label/calib_0000.txt")
        self.assertEqual(coverage["selected_graphs"], 1)
        self.assertIn("batch_size_coverage", coverage)
        self.assertIn("operator_coverage", coverage)
        self.assertIn("family_coverage", coverage)
        self.assertIn("structure_coverage", coverage)
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
            label_path = tmp_path / "out" / "label" / "label" / "calib_0000.txt"
            label_exists = label_path.exists()
            parsed = parse_dataset_label(str(label_path))

        self.assertTrue(label_exists)
        self.assertEqual(parsed.shape, (6,))

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


if __name__ == "__main__":
    unittest.main()
