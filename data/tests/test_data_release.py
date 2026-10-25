"""CPU tests for scientific identity and durable label boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import generate
import join_cached
import label


class GenerationTests(unittest.TestCase):
    def test_budget_covers_all_train_structures_without_labels(self):
        splits = json.loads((HERE / "specs/frozen_group_splits.json").read_text())
        result = list(generate.design("transfer", 42, 16, None, 8192, splits))
        all_train = {generate.digest(c) for c in generate.configurations("pretrain") if splits[generate.group_key(c)] == 0}
        self.assertEqual(len(result), 8192)
        self.assertEqual({generate.digest(c) for c, _ in result}, all_train)
        self.assertEqual(len(all_train), 5397)
        self.assertTrue(all(splits[generate.group_key(c)] == 0 for c, _ in result))
        self.assertEqual(len({(generate.digest(c), variant) for c, variant in result}), 8192)
        self.assertEqual({c["family_id"] for c, _ in result}, set(generate.FAMILY_SPECS))

    def test_manifest_deterministic_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            for path in (a, b):
                generate.generate(path, per_family=1, families=["selu_mlp"])
            self.assertEqual((a / "workloads.jsonl").read_bytes(), (b / "workloads.jsonl").read_bytes())
            rows = label.load_manifest(a)
            self.assertEqual(len(rows), 8)
            source = a / rows[0]["variant_file"]
            source.write_text(source.read_text() + "\n# tampering\n")
            with self.assertRaisesRegex(ValueError, "variant source hash mismatch"):
                label.load_manifest(a)

    def test_duplicate_measurements_are_quarantined_not_chosen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text('{"profile_point_id":"x","targets":{"ms":1}}\n{"profile_point_id":"x","targets":{"ms":9}}\n')
            self.assertEqual(join_cached.read_rows(path, quarantine_duplicates=True)["x"]["status"], "ambiguous_duplicate_quarantined")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                join_cached.read_rows(path)

    def test_original_identity_requires_exact_source_and_precision(self):
        original = {"architecture": {}, "batch_size": 4, "input_specs": [], "precision_config": "fp32_ieee",
                    "optimizer": "adam", "model_id": "x", "source_sha256": "source",
                    "shared_runtime_sha256": "runtime", "shared_catalog_sha256": generate.sha(HERE / "runtime/_model_catalog.py")}
        current = {**original, "runtime_sha256": "runtime", "profile_point_id": "x"}
        join_cached.check_original(current, original)
        for key in ("source_sha256", "precision_config", "batch_size"):
            with self.assertRaises(ValueError):
                join_cached.check_original({**current, key: "changed"}, original)


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_and_hardware_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {"hardware": {"hardware_id": "actual_local_gpu"}, "protocol": label.PROTOCOL}
            label.atomic(root / "run.json", run)
            record = {"run_sha256": generate.digest(run), "hardware": run["hardware"], "status": "oom"}
            label.atomic(root / "records/x.json", record)
            self.assertEqual(label.verify(root)["status_counts"], {"oom": 1})
            before = (root / "records/x.json").read_bytes()
            self.assertEqual(label.verify(root)["records"], 1)
            self.assertEqual(before, (root / "records/x.json").read_bytes())
            record["hardware"] = {"hardware_id": "nvidia_a100"}
            label.atomic(root / "records/x.json", record)
            with self.assertRaisesRegex(ValueError, "hardware mismatch"):
                label.verify(root)

    def test_errors_cannot_publish_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {"hardware": {}, "protocol": label.PROTOCOL}
            label.atomic(root / "run.json", run)
            label.atomic(root / "records/x.json", {"run_sha256": generate.digest(run), "hardware": {}, "status": "timeout", "targets": {"train_step_wall_ms": 1}})
            with self.assertRaisesRegex(ValueError, "failed record"):
                label.verify(root)

    def test_protocol_changes_change_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workloads.jsonl").write_text("{}\n")
            a = label.binding(root, {"gpu_uuid": "one"}, label.PROTOCOL)
            b = label.binding(root, {"gpu_uuid": "two"}, label.PROTOCOL)
            c = label.binding(root, {"gpu_uuid": "one"}, {**label.PROTOCOL, "steps_per_epoch": 2})
            self.assertNotEqual(generate.digest(a), generate.digest(b))
            self.assertNotEqual(generate.digest(a), generate.digest(c))


if __name__ == "__main__":
    unittest.main()
