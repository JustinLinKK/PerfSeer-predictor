from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nrp_calibration_pack.profile.generated_model_runtime import GraphModel  # noqa: E402
from nrp_calibration_pack.profile.run_profile import NvmlSampler  # noqa: E402
from nrp_calibration_pack.workload import (  # noqa: E402
    SCHEDULER_RESOURCE_LABEL_VERSION,
    label_v3_from_result,
    manifest_row_from_workload,
    scheduler_resource_label_from_result,
    scheduler_target_vector,
)
from perfseer_optimized.data import (  # noqa: E402
    FeatureConfig,
    SCHEDULER_RESOURCE_TRAIN_TARGET_NAMES,
    SCHEDULER_V2_TRAIN_TARGET_NAMES,
    feature_config_for_pair,
    feature_layout,
    scheduler_resource_target_for_label,
    scheduler_v2_target_for_label,
    target_names_for_config,
)
from scripts.validate_dataset_resource_labels import select_validation_workloads  # noqa: E402


class RealDatasetWorkflowTests(unittest.TestCase):
    def _registry(self, root: Path, status: str = "pending", target_tier: str = "local") -> Path:
        registry = {
            "registry_version": 1,
            "subset_sizes": {"tiny": 2, "small": 3, "full": None},
            "datasets": [
                {
                    "id": "toy_images",
                    "status": status,
                    "target_tier": target_tier,
                    "source_type": "kaggle_competition",
                    "slug": "toy-images",
                    "task_family": "image_classification",
                    "modality": "image",
                    "approval_doc": "candidates/toy_images.md",
                    "model_families": ["resnet_cnn"],
                    "download_command": ["kaggle", "competitions", "download", "-c", "toy-images"],
                }
            ],
        }
        path = root / "registry.json"
        path.write_text(json.dumps(registry, indent=2) + "\n")
        return path

    def test_nvml_sampler_reports_percentiles_and_reliability_flags(self) -> None:
        sampler = object.__new__(NvmlSampler)
        sampler.samples = [
            (0.0, 10.0, 100.0, 10.0, 0.0),
            (50.0, 20.0, 150.0, 15.0, 1.0),
            (100.0, 30.0, 200.0, 20.0, 2.0),
        ]
        sampler._stop = threading.Event()
        sampler._thread = None
        sampler.available = True
        stats = sampler.stop(min_phase_seconds=5.0, min_sampler_samples=4)
        self.assertEqual(stats.sample_count, 3)
        self.assertAlmostEqual(stats.p50_sm_util, 50.0)
        self.assertAlmostEqual(stats.p95_sm_util, 95.0)
        self.assertGreater(stats.sm_util_std, 0.0)
        self.assertIn("short_measurement", stats.reliability_flags)
        self.assertIn("low_sample_count", stats.reliability_flags)
        self.assertIn("high_sm_variance", stats.reliability_flags)

    def test_scheduler_resource_label_version_two_preserves_old_and_new_targets(self) -> None:
        result = {
            "profile_point_id": "m::d::tiny::bs4::adam::fp32::rtx",
            "model_id": "m",
            "status": "ok",
            "batch_size": 4,
            "hardware_id": "rtx",
            "precision_config": "fp32_ieee",
            "resource_profile_mode": "sustained",
            "workload_spec": {
                "dataset": {"dataset_id": "d", "subset_id": "tiny", "num_samples": 10},
                "training": {"batch_size": 4, "grad_accumulation_steps": 1, "optimizer": "adam", "precision": "fp32_ieee"},
            },
            "details": {
                "train": {
                    "mean_wall_iter_ms": 2.5,
                    "mean_iter_ms": 2.0,
                    "total_wall_ms": 25.0,
                    "total_gpu_ms": 20.0,
                    "measurement_steps": 10,
                    "sampler": {
                        "avg_sm_util": 50.0,
                        "sm_util_std": 3.0,
                        "p50_sm_util": 49.0,
                        "p95_sm_util": 65.0,
                        "peak_sm_util": 70.0,
                        "avg_mem_util": 10.0,
                        "mem_util_std": 1.0,
                        "p50_mem_util": 9.0,
                        "p95_mem_util": 15.0,
                        "peak_mem_util": 17.0,
                        "avg_mem_usage": 100.0,
                        "mem_usage_std": 4.0,
                        "p50_mem_usage": 101.0,
                        "p95_mem_usage": 120.0,
                        "peak_mem_usage": 128.0,
                        "sample_count": 101,
                        "measurement_duration_ms": 20000.0,
                        "reliability_flags": [],
                        "source": "nvml",
                    },
                    "peak_torch_allocated_mib": 90.0,
                    "peak_torch_reserved_mib": 96.0,
                    "epoch_time_source": "measured_epochs",
                    "warmup_epochs": 1,
                    "measured_epochs": 2,
                    "steps_per_epoch": 3,
                    "epoch_wall_ms": [8.0, 10.0],
                    "epoch_gpu_ms": [7.0, 9.0],
                    "measured_epoch_wall_mean_ms": 9.0,
                    "measured_epoch_wall_std_ms": 1.0,
                },
                "infer": {"mean_wall_iter_ms": 1.0, "mean_iter_ms": 0.8, "sampler": {"avg_sm_util": 20.0, "peak_mem_usage": 70.0}},
            },
        }
        label = scheduler_resource_label_from_result(result)
        targets = label["targets"]
        self.assertEqual(label["scheduler_resource_label_version"], SCHEDULER_RESOURCE_LABEL_VERSION)
        self.assertEqual(SCHEDULER_RESOURCE_LABEL_VERSION, 2)
        self.assertIn("train_peak_sm_util_percent", targets)
        self.assertEqual(targets["train_p95_sm_util_percent"], 65.0)
        self.assertEqual(targets["train_sm_util_std_percent"], 3.0)
        self.assertEqual(targets["train_measurement_duration_ms"], 20000.0)
        self.assertEqual(targets["train_sampler_samples"], 101.0)
        self.assertEqual(label["resource_quality"]["train"]["sampler_samples"], 101)
        self.assertEqual(label["resource_quality"]["train"]["epoch_time_source"], "measured_epochs")
        self.assertEqual(label["resource_quality"]["train"]["epoch_wall_ms"], [8.0, 10.0])

    def test_generated_embedding_clamps_real_tokenizer_ids(self) -> None:
        model = GraphModel(
            [
                {
                    "id": 0,
                    "type": "Embedding",
                    "args": {"vocab_size": 4},
                    "memory_info": {"output_features": 8, "input_features": 8},
                    "input_index": 0,
                    "preds": [],
                }
            ]
        )
        out = model(torch.tensor([[0, 4, 999]], dtype=torch.long))
        self.assertEqual(tuple(out.shape), (1, 3, 8))

    def test_scheduler_resource_target_source_loads_train_resource_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            label_dir = root / "label" / "label"
            label_dir.mkdir(parents=True)
            label_path = label_dir / "toy.txt"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            targets = {
                "train_avg_sm_util_percent": 11.0,
                "train_p95_sm_util_percent": 22.0,
                "train_peak_vram_used_mib": 33.0,
                "train_peak_torch_reserved_mib": 44.0,
                "train_step_wall_ms": 55.0,
                "train_peak_memory_controller_util_percent": 66.0,
            }
            (root / "label" / "scheduler_resource_label.jsonl").write_text(
                json.dumps({"label_file": "label/label/toy.txt", "profile_point_id": "toy::rtx", "targets": targets}) + "\n"
            )
            vector = scheduler_resource_target_for_label(str(label_path))
            self.assertEqual(vector.tolist(), [targets[name] for name in SCHEDULER_RESOURCE_TRAIN_TARGET_NAMES])
            missing = label_dir / "missing.txt"
            missing.write_text(label_path.read_text())
            with self.assertRaises(FileNotFoundError):
                scheduler_resource_target_for_label(str(missing))

    def test_scheduler_resource_target_names_follow_feature_config(self) -> None:
        self.assertEqual(target_names_for_config(FeatureConfig()), ["train_util", "train_mem", "train_time", "infer_util", "infer_mem", "infer_time"])
        cfg = FeatureConfig(target_source="scheduler_resource_train")
        self.assertEqual(target_names_for_config(cfg), list(SCHEDULER_RESOURCE_TRAIN_TARGET_NAMES))

    def test_scheduler_v2_target_source_combines_epoch_and_resource_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            label_dir = root / "label" / "label"
            label_dir.mkdir(parents=True)
            label_path = label_dir / "toy.txt"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            scheduler_targets = {"train_epoch_ms": 123.0, "train_step_wall_ms": 4.0}
            resource_targets = {
                "train_avg_sm_util_percent": 11.0,
                "train_p95_sm_util_percent": 22.0,
                "train_peak_vram_used_mib": 33.0,
                "train_peak_torch_reserved_mib": 44.0,
                "train_peak_memory_controller_util_percent": 66.0,
            }
            (root / "label" / "scheduler_label_v3.jsonl").write_text(
                json.dumps({"label_file": "label/label/toy.txt", "profile_point_id": "toy::rtx", "targets": scheduler_targets}) + "\n"
            )
            (root / "label" / "scheduler_resource_label.jsonl").write_text(
                json.dumps({"label_file": "label/label/toy.txt", "profile_point_id": "toy::rtx", "targets": resource_targets}) + "\n"
            )

            vector = scheduler_v2_target_for_label(str(label_path))
            expected = {
                "train_epoch_ms": 123.0,
                **resource_targets,
            }
            self.assertEqual(vector.tolist(), [expected[name] for name in SCHEDULER_V2_TRAIN_TARGET_NAMES])
            self.assertEqual(target_names_for_config(FeatureConfig(target_source="scheduler_v2_train")), list(SCHEDULER_V2_TRAIN_TARGET_NAMES))

    def test_validation_selector_uses_all_families_before_round_robin_extras(self) -> None:
        rows = []
        for family in ("resnet_cnn", "efficientnet_cnn", "bert_encoder"):
            for idx in range(3):
                rows.append({"profile_point_id": f"{family}_{idx}", "model": {"architecture_family": family}})
        selected = select_validation_workloads(rows, limit=5)
        self.assertEqual([row["profile_point_id"] for row in selected], ["resnet_cnn_0", "efficientnet_cnn_0", "bert_encoder_0", "resnet_cnn_1", "efficientnet_cnn_1"])

    def test_validation_selector_random_mode_is_seeded_and_stratified(self) -> None:
        rows = []
        for family in ("resnet_cnn", "efficientnet_cnn", "bert_encoder"):
            for idx in range(5):
                rows.append({"profile_point_id": f"{family}_{idx}", "model": {"architecture_family": family}})
        first = select_validation_workloads(rows, limit=5, mode="random", seed=17)
        second = select_validation_workloads(rows, limit=5, mode="random", seed=17)
        other = select_validation_workloads(rows, limit=5, mode="random", seed=18)
        self.assertEqual([row["profile_point_id"] for row in first], [row["profile_point_id"] for row in second])
        self.assertNotEqual([row["profile_point_id"] for row in first], [row["profile_point_id"] for row in other])
        self.assertEqual({row["model"]["architecture_family"] for row in first[:3]}, {"resnet_cnn", "efficientnet_cnn", "bert_encoder"})

    def test_download_refuses_pending_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = self._registry(root, "pending")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry),
                    "download",
                    "toy_images",
                    "--dry-run",
                    "--raw-root",
                    str(root / "raw"),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not approved", proc.stderr + proc.stdout)

    def test_download_refuses_nautilus_only_dataset_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = self._registry(root, "approved", "nautilus")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry),
                    "download",
                    "toy_images",
                    "--dry-run",
                    "--raw-root",
                    str(root / "raw"),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Nautilus-only", proc.stderr + proc.stdout)

            dry_raw = root / "dry" / "raw"
            allowed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry),
                    "download",
                    "toy_images",
                    "--dry-run",
                    "--allow-nautilus-only",
                    "--raw-root",
                    str(dry_raw),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0)
            self.assertIn("kaggle competitions download", allowed.stdout)
            self.assertFalse(dry_raw.exists())

    def test_list_filters_dataset_tier_and_ids_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = {
                "registry_version": 1,
                "datasets": [
                    {"id": "local_ds", "status": "approved", "target_tier": "local", "task_family": "image", "slug": "local"},
                    {
                        "id": "remote_ds",
                        "status": "approved",
                        "target_tier": "nautilus",
                        "task_family": "image",
                        "slug": "remote",
                    },
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry, indent=2) + "\n")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry_path),
                    "list",
                    "--tier",
                    "local",
                    "--ids-only",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertEqual(proc.stdout.strip(), "local_ds")

    def test_clean_local_5090_labeling_archives_guarded_outputs(self) -> None:
        record_root = ROOT / "record"
        record_root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="nrp_results_clean_test_", dir=ROOT))
        record_dir = Path(tempfile.mkdtemp(prefix="clean_test_", dir=record_root))
        archive_id = "unit_archive"
        pid_file = record_dir / "local_5090_balanced_labeling.pid"
        log_file = record_dir / "local_5090_balanced_labeling_test.log"
        try:
            (output_dir / "label" / "label").mkdir(parents=True)
            (output_dir / "results_shard0.jsonl").write_text("{}\n")
            pid_file.write_text("999999\n")
            log_file.write_text("log\n")
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "RECORD_DIR": str(record_dir),
                    "PID_FILE": str(pid_file),
                    "ARCHIVE_ID": archive_id,
                }
            )
            proc = subprocess.run(
                [str(ROOT / "scripts" / "clean_local_5090_labeling.sh")],
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            archive_dir = record_dir / "labeling_archives" / archive_id
            manifest = json.loads((archive_dir / "manifest.json").read_text())
            self.assertFalse(output_dir.exists())
            self.assertIn(output_dir.name, manifest["archived_entries"])
            self.assertIn("cleanup_mode=archive", proc.stdout)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            if record_dir.exists():
                shutil.rmtree(record_dir)

    def test_clean_local_5090_labeling_dry_run_keeps_outputs_and_guards_paths(self) -> None:
        record_root = ROOT / "record"
        record_root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="nrp_results_clean_test_", dir=ROOT))
        record_dir = Path(tempfile.mkdtemp(prefix="clean_test_", dir=record_root))
        try:
            (output_dir / "results_shard0.jsonl").write_text("{}\n")
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "RECORD_DIR": str(record_dir),
                    "PID_FILE": str(record_dir / "local_5090_balanced_labeling.pid"),
                    "DRY_RUN": "1",
                }
            )
            subprocess.run(
                [str(ROOT / "scripts" / "clean_local_5090_labeling.sh")],
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertTrue(output_dir.exists())

            bad_env = os.environ.copy()
            bad_env.update(
                {
                    "OUTPUT_DIR": str(ROOT / "datasets" / "raw"),
                    "RECORD_DIR": str(record_dir),
                    "PID_FILE": str(record_dir / "local_5090_balanced_labeling.pid"),
                    "DRY_RUN": "1",
                }
            )
            bad = subprocess.run(
                [str(ROOT / "scripts" / "clean_local_5090_labeling.sh")],
                env=bad_env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("clean_refused", bad.stderr)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            if record_dir.exists():
                shutil.rmtree(record_dir)

    def test_ogb_dry_run_auto_confirms_and_disables_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = {
                "registry_version": 1,
                "datasets": [
                    {
                        "id": "toy_graph",
                        "status": "approved",
                        "target_tier": "local",
                        "source_type": "pyg_ogb",
                        "slug": "ogbn-products",
                        "task_family": "graph",
                        "modality": "graph",
                        "model_families": ["gat_graph"],
                    }
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry, indent=2) + "\n")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry_path),
                    "download-ogb",
                    "toy_graph",
                    "--raw-root",
                    str(root / "raw"),
                    "--dry-run",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1", proc.stdout)
            self.assertIn("builtins.input", proc.stdout)
            self.assertIn("PygNodePropPredDataset", proc.stdout)

    def test_prepare_writes_deterministic_subsets_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = self._registry(root, "approved")
            raw = root / "raw" / "toy_images"
            raw.mkdir(parents=True)
            for idx in range(5):
                (raw / f"sample_{idx}.bin").write_bytes(bytes([idx]) * (idx + 1))
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "manage_dataset_sources.py"),
                "--registry",
                str(registry),
                "prepare",
                "toy_images",
                "--raw-root",
                str(root / "raw"),
                "--prepared-root",
                str(root / "prepared"),
            ]
            subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
            profile = json.loads((root / "prepared" / "toy_images" / "dataset_profile.json").read_text())
            tiny = json.loads((root / "prepared" / "toy_images" / "subset_masks" / "tiny.json").read_text())
            self.assertEqual(profile["dataset_id"], "toy_images")
            self.assertEqual(profile["sample_count"], 5)
            self.assertEqual(tiny["num_samples"], 2)
            first_keys = tiny["sample_keys"]
            subprocess.run(cmd + ["--force"], check=True, text=True, stdout=subprocess.PIPE)
            tiny_again = json.loads((root / "prepared" / "toy_images" / "subset_masks" / "tiny.json").read_text())
            self.assertEqual(first_keys, tiny_again["sample_keys"])

    def test_prepare_counts_media_records_inside_kaggle_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = {
                "registry_version": 1,
                "subset_sizes": {"tiny": 2, "full": None},
                "datasets": [
                    {
                        "id": "cassava_leaf_disease",
                        "status": "approved",
                        "target_tier": "local",
                        "source_type": "kaggle_competition",
                        "slug": "cassava-leaf-disease-classification",
                        "task_family": "image_classification",
                        "modality": "image",
                        "approval_doc": "candidates/cassava_leaf_disease.md",
                        "model_families": ["resnet_cnn"],
                        "download_command": ["kaggle", "competitions", "download", "-c", "cassava"],
                    }
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry, indent=2) + "\n")
            raw = root / "raw" / "cassava_leaf_disease"
            raw.mkdir(parents=True)
            with zipfile.ZipFile(raw / "cassava.zip", "w") as zf:
                zf.writestr("train_images/a.jpg", b"a")
                zf.writestr("train_images/b.jpg", b"b")
                zf.writestr("test_images/c.jpg", b"c")
                zf.writestr("train.csv", "image_id,label\na.jpg,0\nb.jpg,1\n")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry_path),
                    "prepare",
                    "cassava_leaf_disease",
                    "--raw-root",
                    str(root / "raw"),
                    "--prepared-root",
                    str(root / "prepared"),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            profile = json.loads((root / "prepared" / "cassava_leaf_disease" / "dataset_profile.json").read_text())
            full = json.loads((root / "prepared" / "cassava_leaf_disease" / "subset_masks" / "full.json").read_text())
            self.assertEqual(profile["sample_count"], 2)
            self.assertEqual(full["num_samples"], 2)
            self.assertTrue(all("train_images/" in key for key in full["sample_keys"]))

    def test_prepare_counts_csv_rows_inside_kaggle_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = {
                "registry_version": 1,
                "subset_sizes": {"tiny": 2, "full": None},
                "datasets": [
                    {
                        "id": "store_sales_time_series",
                        "status": "approved",
                        "target_tier": "local",
                        "source_type": "kaggle_competition",
                        "slug": "store-sales-time-series-forecasting",
                        "task_family": "time_series",
                        "modality": "time_series",
                        "approval_doc": "candidates/store_sales_time_series.md",
                        "model_families": ["gru_temporal"],
                        "download_command": ["kaggle", "competitions", "download", "-c", "store"],
                    }
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry, indent=2) + "\n")
            raw = root / "raw" / "store_sales_time_series"
            raw.mkdir(parents=True)
            with zipfile.ZipFile(raw / "store.zip", "w") as zf:
                zf.writestr("train.csv", "id,y\n1,1\n2,2\n3,3\n")
                zf.writestr("test.csv", "id\n4\n")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manage_dataset_sources.py"),
                    "--registry",
                    str(registry_path),
                    "prepare",
                    "store_sales_time_series",
                    "--raw-root",
                    str(root / "raw"),
                    "--prepared-root",
                    str(root / "prepared"),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            profile = json.loads((root / "prepared" / "store_sales_time_series" / "dataset_profile.json").read_text())
            full = json.loads((root / "prepared" / "store_sales_time_series" / "subset_masks" / "full.json").read_text())
            self.assertEqual(profile["sample_count"], 3)
            self.assertEqual(full["num_samples"], 3)
            self.assertTrue(all("train.csv:row:" in key for key in full["sample_keys"]))

    def test_workload_specs_expand_model_dataset_batch_precision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": "calib_0000",
                        "graph_id": "calib_0000",
                        "model_file": "models/calib_0000.py",
                        "architecture_family": "resnet_cnn",
                        "input_shape": [1, 3, 16, 16],
                        "input_specs": [{"name": "image", "shape": [1, 3, 16, 16], "dtype": "float32", "kind": "image"}],
                    }
                )
                + "\n"
            )
            profile_dir = root / "profiles" / "toy_images"
            profile_dir.mkdir(parents=True)
            (profile_dir / "dataset_profile.json").write_text(
                json.dumps(
                    {
                        "dataset_profile_version": 1,
                        "dataset_id": "toy_images",
                        "task_family": "image_classification",
                        "modality": "image",
                        "model_families": ["resnet_cnn"],
                        "sample_count": 5,
                        "sample_bytes_mean": 12.0,
                        "subsets": {"tiny": {"path": "tiny.json", "num_samples": 2}},
                        "metadata_source": "real_downloaded_files",
                    }
                )
                + "\n"
            )
            registry = {
                "registry_version": 1,
                "datasets": [
                    {
                        "id": "toy_images",
                        "status": "prepared",
                        "prepared_profile": str(profile_dir / "dataset_profile.json"),
                        "model_families": ["resnet_cnn"],
                    }
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry) + "\n")
            out = root / "workloads"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "make_workload_specs.py"),
                    "--manifest",
                    str(manifest),
                    "--registry",
                    str(registry_path),
                    "--dataset-profile-root",
                    str(root / "profiles"),
                    "--output-dir",
                    str(out),
                    "--subset-id",
                    "tiny",
                    "--batch-size",
                    "4",
                    "--precision-sweep",
                    "fp32_ieee,bf16_amp",
                    "--hardware-id",
                    "rtx5090",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            rows = [json.loads(line) for line in (out / "workloads.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset"]["dataset_id"], "toy_images")
            self.assertEqual(rows[0]["training"]["optimizer"], "adam")
            self.assertIn("toy_images::tiny::bs4::adam", rows[0]["profile_point_id"])
            self.assertEqual(rows[0]["model"]["input_shape"], [4, 3, 16, 16])
            self.assertEqual(rows[0]["dataset"]["input_shape"], [4, 3, 16, 16])
            self.assertEqual(rows[0]["dataset"]["dataset_input_shape"], [4, 3, 224, 224])
            manifest_row = manifest_row_from_workload(rows[0])
            self.assertEqual(manifest_row["input_shape"], [4, 3, 16, 16])

    def test_workload_specs_marks_real_dataloader_backed_when_local_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": "calib_0000",
                        "model_file": "models/calib_0000.py",
                        "architecture_family": "resnet_cnn",
                        "input_shape": [1, 3, 16, 16],
                    }
                )
                + "\n"
            )
            raw = root / "raw" / "toy_images"
            raw.mkdir(parents=True)
            with zipfile.ZipFile(raw / "toy.zip", "w") as zf:
                zf.writestr("samples/a.bin", b"real-sample")
            profile_dir = root / "profiles" / "toy_images"
            subset_dir = profile_dir / "subset_masks"
            subset_dir.mkdir(parents=True)
            subset = subset_dir / "tiny.json"
            subset.write_text(
                json.dumps({"subset_id": "tiny", "num_samples": 1, "sample_keys": ["toy.zip::samples/a.bin"]}) + "\n"
            )
            (profile_dir / "dataset_profile.json").write_text(
                json.dumps(
                    {
                        "dataset_profile_version": 1,
                        "dataset_id": "toy_images",
                        "task_family": "image_classification",
                        "modality": "image",
                        "model_families": ["resnet_cnn"],
                        "sample_count": 1,
                        "sample_bytes_mean": 11.0,
                        "raw_summary": {"raw_dir": str(raw)},
                        "subsets": {"tiny": {"path": str(subset), "num_samples": 1}},
                        "metadata_source": "real_downloaded_files",
                    }
                )
                + "\n"
            )
            registry = {
                "registry_version": 1,
                "datasets": [
                    {
                        "id": "toy_images",
                        "status": "prepared",
                        "prepared_profile": str(profile_dir / "dataset_profile.json"),
                        "model_families": ["resnet_cnn"],
                    }
                ],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry) + "\n")
            out = root / "workloads"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "make_workload_specs.py"),
                    "--manifest",
                    str(manifest),
                    "--registry",
                    str(registry_path),
                    "--dataset-profile-root",
                    str(root / "profiles"),
                    "--raw-root",
                    str(root / "raw"),
                    "--output-dir",
                    str(out),
                    "--subset-id",
                    "tiny",
                    "--batch-size",
                    "2",
                    "--hardware-id",
                    "rtx5090",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            row = json.loads((out / "workloads.jsonl").read_text().splitlines()[0])
            self.assertTrue(row["dataset"]["real_dataloader_backed"])
            self.assertEqual(row["dataset"]["raw_dir"], str(raw))
            self.assertEqual(row["dataset"]["subset_mask_path"], str(subset))

    def test_label_v3_derives_epoch_from_subset_size(self) -> None:
        result = {
            "profile_point_id": "m::d::tiny::bs4::adam::fp32::rtx",
            "model_id": "m",
            "status": "ok",
            "batch_size": 4,
            "precision_config": "fp32_ieee",
            "workload_spec": {
                "dataset": {"dataset_id": "d", "subset_id": "tiny", "num_samples": 10},
                "training": {"batch_size": 4, "grad_accumulation_steps": 1, "optimizer": "adam", "precision": "fp32_ieee"},
            },
            "details": {
                "train": {
                    "mean_wall_iter_ms": 2.5,
                    "mean_iter_ms": 2.0,
                    "sampler": {"avg_sm_util": 50.0, "peak_mem_usage": 100.0},
                    "peak_torch_allocated_mib": 90.0,
                },
                "infer": {"mean_wall_iter_ms": 1.5, "mean_iter_ms": 1.0, "sampler": {"avg_sm_util": 25.0, "peak_mem_usage": 70.0}},
            },
        }
        label = label_v3_from_result(result)
        self.assertEqual(label["training"]["steps_per_epoch"], 3)
        self.assertAlmostEqual(label["targets"]["train_epoch_ms"], 7.5)
        self.assertAlmostEqual(label["targets"]["train_epoch_ms_step_extrapolated"], 7.5)
        self.assertEqual(label["training"]["epoch_time_source"], "step_extrapolated")
        self.assertEqual(len(scheduler_target_vector(label)), 10)

    def test_label_v3_prefers_measured_epoch_time_when_available(self) -> None:
        result = {
            "profile_point_id": "m::d::tiny::bs4::adam::fp32::rtx",
            "model_id": "m",
            "status": "ok",
            "batch_size": 4,
            "precision_config": "fp32_ieee",
            "workload_spec": {
                "dataset": {"dataset_id": "d", "subset_id": "tiny", "num_samples": 10},
                "training": {"batch_size": 4, "grad_accumulation_steps": 1, "optimizer": "adam", "precision": "fp32_ieee"},
            },
            "details": {
                "train": {
                    "mean_wall_iter_ms": 2.5,
                    "mean_iter_ms": 2.0,
                    "epoch_time_source": "measured_epochs",
                    "warmup_epochs": 1,
                    "measured_epochs": 2,
                    "steps_per_epoch": 3,
                    "epoch_wall_ms": [9.0, 12.0],
                    "measured_epoch_wall_mean_ms": 10.5,
                    "sampler": {"avg_sm_util": 50.0, "peak_mem_usage": 100.0},
                    "peak_torch_allocated_mib": 90.0,
                },
                "infer": {"mean_wall_iter_ms": 1.5, "mean_iter_ms": 1.0, "sampler": {"avg_sm_util": 25.0, "peak_mem_usage": 70.0}},
            },
        }
        label = label_v3_from_result(result)
        self.assertEqual(label["training"]["steps_per_epoch"], 3)
        self.assertAlmostEqual(label["targets"]["train_epoch_ms"], 10.5)
        self.assertAlmostEqual(label["targets"]["train_epoch_ms_step_extrapolated"], 7.5)
        self.assertEqual(label["training"]["epoch_time_source"], "measured_epochs")
        self.assertEqual(label["training"]["warmup_epochs"], 1)
        self.assertEqual(label["training"]["measured_epochs"], 2)
        self.assertEqual(len(scheduler_target_vector(label)), 10)

    def test_workload_precision_expansion_recomputes_profile_point_id(self) -> None:
        from nrp_calibration_pack.profile.run_profile import expand_workload_manifest_precisions

        workload = {
            "workload_spec_version": 1,
            "model": {"model_id": "m", "source_path": "m.py"},
            "dataset": {"dataset_id": "d", "subset_id": "tiny", "num_samples": 1},
            "training": {"batch_size": 2, "optimizer": "adam", "precision": "fp32_ieee"},
            "hardware": {"hardware_id": "rtx5090"},
            "profile_point_id": "m::d::tiny::bs2::adam::fp32_ieee::rtx5090",
        }
        rows = expand_workload_manifest_precisions([{"workload_spec": workload}], ["fp32_ieee", "bf16_amp"])
        profile_ids = [row["profile_point_id"] for row in rows]
        self.assertEqual(len(set(profile_ids)), 2)
        self.assertIn("m::d::tiny::bs2::adam::bf16_amp::rtx5090", profile_ids)

    def test_dataset_metadata_updates_feature_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            label_dir = root / "label" / "label"
            graph_dir = root / "cg" / "cg"
            label_dir.mkdir(parents=True)
            graph_dir.mkdir(parents=True)
            label_path = label_dir / "toy.txt"
            graph_path = graph_dir / "toy.pkl"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            graph_path.write_bytes(b"not-used")
            (root / "label" / "precision_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "label_file": "label/label/toy.txt",
                        "dataset": {"num_samples": 123, "sample_bytes_mean": 456.0},
                        "training": {"batch_size": 8, "grad_accumulation_steps": 2, "optimizer": "adam"},
                    }
                )
                + "\n"
            )
            cfg = feature_config_for_pair(FeatureConfig(), str(graph_path), str(label_path))
            self.assertEqual(cfg.dataset_num_samples, 123)
            self.assertEqual(cfg.training_batch_size, 8)
            self.assertEqual(cfg.training_optimizer, "adam")
            self.assertIn("dataset_num_samples_log1p", feature_layout(cfg).global_names)

    def test_dataset_shape_metadata_updates_feature_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            label_dir = root / "label" / "label"
            graph_dir = root / "cg" / "cg"
            label_dir.mkdir(parents=True)
            graph_dir.mkdir(parents=True)
            label_path = label_dir / "toy.txt"
            graph_path = graph_dir / "toy.pkl"
            label_path.write_text("{'train': '1|2|3|4|5|6|7', 'infer': '1|2|3|4|5|6|7'}\n")
            graph_path.write_bytes(b"not-used")
            (root / "label" / "precision_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "label_file": "label/label/toy.txt",
                        "dataset": {
                            "num_samples": 321,
                            "modality": "image",
                            "task_type": "image_classification",
                            "input_rank": 4,
                            "input_dim0": 8,
                            "input_dim1": 3,
                            "input_dim2": 224,
                            "input_dim3": 224,
                            "input_numel_per_sample": 150528,
                            "input_bytes_per_sample": 602112,
                            "input_bytes_per_batch": 4816896,
                        },
                        "training": {"batch_size": 8, "grad_accumulation_steps": 1, "optimizer": "adam"},
                    }
                )
                + "\n"
            )
            cfg = feature_config_for_pair(FeatureConfig(), str(graph_path), str(label_path))
            layout = feature_layout(cfg)
            self.assertEqual(cfg.dataset_input_rank, 4)
            self.assertEqual(cfg.dataset_input_dim2, 224)
            self.assertEqual(cfg.dataset_input_bytes_per_batch, 4816896)
            self.assertEqual(cfg.dataset_modality, "image")
            self.assertEqual(cfg.dataset_task_type, "image_classification")
            self.assertIn("dataset_input_bytes_per_batch_log1p", layout.global_names)
            self.assertIn("dataset_modality_image", layout.global_names)
            self.assertIn("dataset_task_type_image_classification", layout.global_names)

    def test_materializer_preserves_scheduler_label_v3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            results = root / "results"
            out = root / "out"
            (pack / "manifest").mkdir(parents=True)
            (pack / "subset" / "cg" / "cg").mkdir(parents=True)
            (results).mkdir()
            (pack / "subset" / "cg" / "cg" / "calib_0000.pkl").write_bytes(b"graph")
            (pack / "manifest" / "subset_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "model_id": "calib_0000",
                        "precision_config": "fp32_ieee",
                        "subset_graph_file": "subset/cg/cg/calib_0000.pkl",
                    }
                )
                + "\n"
            )
            label_v3 = {
                "scheduler_label_version": 3,
                "profile_point_id": "calib_0000::toy::tiny::bs4::adam::fp32_ieee::rtx5090",
                "targets": {"train_epoch_ms": 12.0},
                "dataset": {"dataset_id": "toy", "subset_id": "tiny", "num_samples": 8},
                "training": {"batch_size": 4, "optimizer": "adam"},
            }
            scheduler_resource = {
                "scheduler_resource_label_version": SCHEDULER_RESOURCE_LABEL_VERSION,
                "profile_point_id": label_v3["profile_point_id"],
                "targets": {"train_peak_torch_reserved_mib": 5.0},
                "dataset": label_v3["dataset"],
                "training": label_v3["training"],
                "resource_profile_mode": "sustained",
            }
            result = {
                "model_id": "calib_0000",
                "graph_id": "calib_0000",
                "profile_point_id": label_v3["profile_point_id"],
                "status": "ok",
                "precision_config": "fp32_ieee",
                "hardware_id": "rtx5090",
                "label": {"train": "1|2|3|4|5|6|7", "infer": "1|2|3|4|5|6|7"},
                "label_v3": label_v3,
                "scheduler_resource_label": scheduler_resource,
                "workload_spec": {
                    "dataset": {"dataset_id": "toy", "subset_id": "tiny", "num_samples": 8},
                    "training": {"batch_size": 4, "optimizer": "adam"},
                },
            }
            (results / "results_shard0.jsonl").write_text(json.dumps(result) + "\n")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_precision_dataset.py"),
                    "--pack-dir",
                    str(pack),
                    "--results-dir",
                    str(results),
                    "--out-root",
                    str(out),
                    "--force",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            rows = [json.loads(line) for line in (out / "label" / "scheduler_label_v3.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["targets"]["train_epoch_ms"], 12.0)
            resource_rows = [json.loads(line) for line in (out / "label" / "scheduler_resource_label.jsonl").read_text().splitlines()]
            self.assertEqual(len(resource_rows), 1)
            self.assertEqual(resource_rows[0]["targets"]["train_peak_torch_reserved_mib"], 5.0)
            metadata = [json.loads(line) for line in (out / "label" / "precision_metadata.jsonl").read_text().splitlines()]
            self.assertEqual(metadata[0]["dataset_id"], "toy")
            self.assertEqual(metadata[0]["optimizer"], "adam")
            report = json.loads((out / "precision_materialization_report.json").read_text())
            self.assertEqual(report["scheduler_resource_label_rows"], 1)

    def test_workload_profiler_profiles_real_subset_keys_without_synthetic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            out = root / "out"
            raw_zip = root / "raw_zip"
            raw_csv = root / "raw_csv"
            masks = root / "masks"
            models.mkdir()
            raw_zip.mkdir()
            raw_csv.mkdir()
            masks.mkdir()
            model_source = (
                "import torch\n"
                "class M(torch.nn.Module):\n"
                "    def forward(self, x):\n"
                "        return x.float().sum(dim=1, keepdim=True)\n"
                "def make_model():\n"
                "    return M()\n"
            )
            (models / "calib_zip.py").write_text(model_source)
            (models / "calib_csv.py").write_text(model_source)
            with zipfile.ZipFile(raw_zip / "toy.zip", "w") as zf:
                zf.writestr("samples/a.bin", b"zip-backed-sample")
            inner_buffer = io.BytesIO()
            with zipfile.ZipFile(inner_buffer, "w") as inner:
                inner.writestr("train.csv", "text,label\nhello,0\nworld,1\n")
            inner_buffer.seek(0)
            with zipfile.ZipFile(raw_csv / "outer.zip", "w") as outer:
                outer.writestr("train.csv.zip", inner_buffer.read())
            zip_mask = masks / "zip_tiny.json"
            csv_mask = masks / "csv_tiny.json"
            zip_mask.write_text(
                json.dumps({"subset_id": "tiny", "num_samples": 1, "sample_keys": ["toy.zip::samples/a.bin"]}) + "\n"
            )
            csv_mask.write_text(
                json.dumps(
                    {
                        "subset_id": "tiny",
                        "num_samples": 1,
                        "sample_keys": ["outer.zip::train.csv.zip::train.csv:row:000000000001"],
                    }
                )
                + "\n"
            )

            def workload(model_id: str, raw_dir: Path, mask: Path) -> dict[str, object]:
                return {
                    "workload_spec_version": 1,
                    "model": {
                        "model_id": model_id,
                        "source_path": f"{model_id}.py",
                        "entrypoint": "make_model",
                        "architecture_family": "resnet_cnn",
                    },
                    "dataset": {
                        "dataset_id": f"dataset_{model_id}",
                        "subset_id": "tiny",
                        "raw_dir": str(raw_dir),
                        "subset_mask_path": str(mask),
                        "subset_index_path": str(mask),
                        "num_samples": 1,
                        "input_specs": [{"name": "x", "shape": [2, 4], "dtype": "float32", "kind": "float"}],
                        "input_shape": [2, 4],
                        "real_dataloader_backed": True,
                    },
                    "training": {"batch_size": 2, "optimizer": "adam", "precision": "fp32_ieee"},
                    "hardware": {"hardware_id": "cpu"},
                }

            workloads = root / "workloads.jsonl"
            workloads.write_text(
                json.dumps(workload("calib_zip", raw_zip, zip_mask))
                + "\n"
                + json.dumps(workload("calib_csv", raw_csv, csv_mask))
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
                    "--workload-specs",
                    str(workloads),
                    "--models-dir",
                    str(models),
                    "--output-dir",
                    str(out),
                    "--hardware-id",
                    "cpu",
                    "--device",
                    "cpu",
                    "--warmup",
                    "0",
                    "--infer-repeats",
                    "1",
                    "--train-repeats",
                    "1",
                    "--sm-occupancy-source",
                    "nvml_proxy",
                    "--resource-profile-mode",
                    "sustained",
                    "--label-time-mode",
                    "measured_epochs",
                    "--time-label-warmup-epochs",
                    "1",
                    "--time-label-measured-epochs",
                    "2",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            rows = [json.loads(line) for line in (out / "results_shard0.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
            self.assertEqual(rows[0]["profile_dataset"]["adapter"], "local_subset_key_tensor")
            self.assertEqual(rows[0]["resource_profile_mode"], "sustained")
            self.assertEqual(rows[0]["profile_dataset"]["measurement_batches"], {"infer": 1, "train": 2})
            self.assertEqual(rows[0]["profile_dataset"]["label_time_mode"], "measured_epochs")
            self.assertEqual(rows[0]["details"]["train"]["epoch_time_source"], "measured_epochs")
            self.assertEqual(rows[0]["details"]["train"]["warmup_epochs"], 1)
            self.assertEqual(rows[0]["details"]["train"]["measured_epochs"], 2)
            self.assertEqual(len(rows[0]["details"]["train"]["epoch_wall_ms"]), 2)
            self.assertIn("generated_batches", rows[0]["profile_dataset"])
            self.assertGreaterEqual(rows[1]["profile_dataset"]["sample_keys_consumed"], 1)
            self.assertEqual(rows[0]["label_v2"]["train"]["resource_profile_mode"], "sustained")
            self.assertIn("avg_memory_controller_utilization_percent", rows[0]["label_v2"]["train"])
            self.assertIn("peak_torch_reserved_mib", rows[0]["label_v2"]["train"])
            self.assertTrue((out / "label_v3_shard0.jsonl").is_file())
            label_v3_rows = [json.loads(line) for line in (out / "label_v3_shard0.jsonl").read_text().splitlines()]
            self.assertEqual(label_v3_rows[0]["training"]["epoch_time_source"], "measured_epochs")
            self.assertIn("train_epoch_ms_step_extrapolated", label_v3_rows[0]["targets"])
            resource_rows = [json.loads(line) for line in (out / "scheduler_resource_shard0.jsonl").read_text().splitlines()]
            self.assertEqual(len(resource_rows), 2)
            self.assertEqual(resource_rows[0]["resource_profile_mode"], "sustained")
            self.assertEqual(resource_rows[0]["resource_quality"]["train"]["epoch_time_source"], "measured_epochs")

    def test_workload_profiler_rejects_non_real_dataloader_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            out = root / "out"
            models.mkdir()
            (models / "calib_0000.py").write_text(
                "import torch\n"
                "class M(torch.nn.Module):\n"
                "    def forward(self, x):\n"
                "        return x.float().sum(dim=1, keepdim=True)\n"
                "def make_model():\n"
                "    return M()\n"
            )
            workload = {
                "workload_spec_version": 1,
                "model": {
                    "model_id": "calib_0000",
                    "source_path": "calib_0000.py",
                    "entrypoint": "make_model",
                    "architecture_family": "resnet_cnn",
                },
                "dataset": {
                    "dataset_id": "toy",
                    "subset_id": "tiny",
                    "num_samples": 4,
                    "input_specs": [{"name": "x", "shape": [2, 4], "dtype": "float32", "kind": "float"}],
                    "input_shape": [2, 4],
                    "real_dataloader_backed": False,
                },
                "training": {"batch_size": 2, "optimizer": "adam", "precision": "fp32_ieee"},
                "hardware": {"hardware_id": "cpu"},
            }
            workloads = root / "workloads.jsonl"
            workloads.write_text(json.dumps(workload) + "\n")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
                    "--workload-specs",
                    str(workloads),
                    "--models-dir",
                    str(models),
                    "--output-dir",
                    str(out),
                    "--hardware-id",
                    "cpu",
                    "--device",
                    "cpu",
                    "--warmup",
                    "0",
                    "--infer-repeats",
                    "1",
                    "--train-repeats",
                    "1",
                    "--sm-occupancy-source",
                    "nvml_proxy",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            rows = [json.loads(line) for line in (out / "results_shard0.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["status"], "unsupported_real_dataloader")


if __name__ == "__main__":
    unittest.main()
