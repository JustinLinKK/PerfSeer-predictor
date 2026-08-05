from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.transfer_subset import (
    TransferSubsetRequestV3,
    select_transfer_subset,
)
from perfseer_v3.baseline import canonical_json
from perfseer_v3.hardware_transfer import (
    BaseTransferLineageV3,
    transfer_manifest_json_schema,
)
from perfseer_v3.dataset_pack.transfer_labeling import (
    TransferLabelAttemptV3,
    build_target_training_manifest,
    deterministic_memory_probe_batches,
    target_training_manifest_json_schema,
)
from perfseer_v3.dataset_pack.label_worker import (
    LabelWorkerError,
    resolve_transfer_inputs,
)


def _corpus(
    split_counts: tuple[int, int, int] = (210, 45, 45),
) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    rows: list[dict[str, object]] = []
    embeddings: dict[str, list[float]] = {}
    index = 0
    for split, count in zip(("train", "validation", "test"), split_counts):
        for offset in range(count):
            item = f"cfg_{index:04d}"
            rows.append(
                {
                    "configuration_id": item,
                    "split": split,
                    "source_group": f"source_{index:04d}",
                    "graph_signature": f"graph_{index:04d}",
                    "graph_path": f"graphs/cfg_{index:04d}.json",
                    "modality": ("vision", "nlp", "audio", "tabular", "graph", "generated")[
                        offset % 6
                    ],
                    "model_family": f"family_{offset % 12}",
                    "precision_policy": ("fp32", "tf32", "fp16_amp", "bf16_amp")[
                        offset % 4
                    ],
                    "execution_mode": ("eager", "compiled")[offset % 2],
                    "resource_regime": ("light", "standard", "heavy")[offset % 3],
                    "optimizer": ("adamw", "sgd")[offset % 2],
                    "scheduler": ("none", "cosine", "linear")[offset % 3],
                    "operation_cost_regime": (
                        "compute_bound",
                        "memory_bound",
                        "launch_bound",
                        "capacity_bound",
                    )[offset % 4],
                }
            )
            embeddings[item] = [
                float((index * 17) % 101),
                float((index * 29) % 103),
                float((index * 43) % 107),
            ]
            index += 1
    return rows, embeddings


def _base_lineage() -> dict[str, str]:
    return BaseTransferLineageV3(
        base_training_manifest_sha256="1" * 64,
        base_dataset_fingerprint="2" * 64,
        base_split_fingerprint="3" * 64,
        base_teacher_artifact_sha256="4" * 64,
        base_student_artifact_sha256="5" * 64,
        base_teacher_embeddings_sha256="6" * 64,
        workload_normalization_sha256="7" * 64,
    ).to_dict()


class TransferSubsetTests(unittest.TestCase):
    def test_base_lineage_is_canonical_hashed_and_required_across_budgets(self) -> None:
        lineage = _base_lineage()
        self.assertEqual(
            BaseTransferLineageV3.from_dict(dict(reversed(list(lineage.items())))).to_dict(),
            lineage,
        )
        tampered = dict(lineage)
        tampered["workload_normalization_sha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            BaseTransferLineageV3.from_dict(tampered)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            BaseTransferLineageV3(
                **{
                    **BaseTransferLineageV3.from_dict(lineage).__dict__,
                    "base_dataset_fingerprint": "A" * 64,
                }
            ).validate()

        rows, embeddings = _corpus()
        pilot = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=lineage,
        )
        changed = BaseTransferLineageV3(
            **{
                **BaseTransferLineageV3.from_dict(lineage).__dict__,
                "base_teacher_embeddings_sha256": "9" * 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "different frozen A10 base lineage"):
            select_transfer_subset(
                rows,
                embeddings,
                TransferSubsetRequestV3(
                    target_hardware_id="nvidia_h100_sxm_80gb",
                    label_budget=256,
                    active_learning=True,
                ),
                base_lineage=changed,
                prior_manifest=pilot,
                active_scores={
                    str(row["configuration_id"]): float(index)
                    for index, row in enumerate(rows)
                },
            )

    def test_memory_probe_ladder_increases_by_powers_of_two_only(self) -> None:
        self.assertEqual(
            deterministic_memory_probe_batches(
                8,
                (1, 2, 4, 8, 16, 32, 64),
                maximum_microbatch_size=40,
            ),
            (16, 32),
        )
        with self.assertRaisesRegex(ValueError, "power of two"):
            deterministic_memory_probe_batches(6, (2, 4, 6, 8))

    def test_label_worker_binds_exact_frozen_subset_and_base_label(self) -> None:
        rows, embeddings = _corpus()
        subset = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=_base_lineage(),
        )
        selected_id = str(subset["selection"][0]["configuration_id"])
        selection, targets = resolve_transfer_inputs(
            selected_id,
            subset,
            [{"configuration_id": selected_id, "target_values": [1.0] * 6}],
        )
        self.assertEqual(selection["configuration_id"], selected_id)
        self.assertEqual(targets, (1.0,) * 6)
        tampered = dict(subset)
        tampered["target_hardware_id"] = "nvidia_l40s"
        with self.assertRaisesRegex(LabelWorkerError, "content hash"):
            resolve_transfer_inputs(
                selected_id,
                tampered,
                [{"configuration_id": selected_id, "target_values": [1.0] * 6}],
            )

    def test_deterministic_nested_budgets_and_frozen_test_selection(self) -> None:
        rows, embeddings = _corpus()
        pilot_request = TransferSubsetRequestV3(
            target_hardware_id="nvidia_h100_sxm_80gb",
            label_budget=128,
        )
        pilot = select_transfer_subset(
            rows, embeddings, pilot_request, base_lineage=_base_lineage()
        )
        repeated = select_transfer_subset(
            rows, embeddings, pilot_request, base_lineage=_base_lineage()
        )
        self.assertEqual(pilot, repeated)
        jsonschema.validate(pilot, transfer_manifest_json_schema())
        self.assertEqual(len(pilot["selection"]), 128)
        self.assertEqual(pilot["split_counts"], {"train": 96, "validation": 16, "test": 16})
        self.assertFalse(pilot["test_selection_uses_model_errors"])

        scores_a = {str(row["configuration_id"]): float(index % 7) for index, row in enumerate(rows)}
        scores_b = {key: -value for key, value in scores_a.items()}
        expanded_request = TransferSubsetRequestV3(
            target_hardware_id="nvidia_h100_sxm_80gb",
            label_budget=256,
            active_learning=True,
        )
        expanded_a = select_transfer_subset(
            rows,
            embeddings,
            expanded_request,
            base_lineage=_base_lineage(),
            prior_manifest=pilot,
            active_scores=scores_a,
        )
        expanded_b = select_transfer_subset(
            rows,
            embeddings,
            expanded_request,
            base_lineage=_base_lineage(),
            prior_manifest=pilot,
            active_scores=scores_b,
        )
        pilot_ids = {item["configuration_id"] for item in pilot["selection"]}
        expanded_ids = {item["configuration_id"] for item in expanded_a["selection"]}
        self.assertLessEqual(pilot_ids, expanded_ids)
        test_a = [
            item["configuration_id"]
            for item in expanded_a["selection"]
            if item["split"] == "test"
        ]
        test_b = [
            item["configuration_id"]
            for item in expanded_b["selection"]
            if item["split"] == "test"
        ]
        self.assertEqual(test_a, test_b)
        self.assertEqual(len(expanded_a["memory_boundary_probes"]), 32)
        self.assertEqual(
            {
                split: sum(
                    item["grouped_split"] == split
                    for item in expanded_a["memory_boundary_probes"]
                )
                for split in ("train", "validation", "test")
            },
            {"train": 24, "validation": 4, "test": 4},
        )

    def test_group_leakage_is_rejected(self) -> None:
        rows, embeddings = _corpus()
        rows[210]["source_group"] = rows[0]["source_group"]
        with self.assertRaisesRegex(ValueError, "leaks"):
            select_transfer_subset(
                rows,
                embeddings,
                TransferSubsetRequestV3(
                    target_hardware_id="nvidia_h100_sxm_80gb",
                    label_budget=128,
                ),
                base_lineage=_base_lineage(),
            )

    def test_finalized_a10_field_aliases_preserve_stratified_selection(self) -> None:
        rows, embeddings = _corpus()
        aliased = []
        for row in rows:
            converted = dict(row)
            for canonical, production in (
                ("modality", "quota_modality"),
                ("model_family", "family_id"),
                ("precision_policy", "precision_id"),
                ("optimizer", "optimizer_id"),
                ("scheduler", "scheduler_id"),
            ):
                converted[production] = converted.pop(canonical)
            converted["graph_sha256"] = converted.pop("graph_signature")
            aliased.append(converted)
        request = TransferSubsetRequestV3(
            target_hardware_id="nvidia_h100_sxm_80gb",
            label_budget=128,
        )
        expected = select_transfer_subset(
            rows, embeddings, request, base_lineage=_base_lineage()
        )
        observed = select_transfer_subset(
            aliased, embeddings, request, base_lineage=_base_lineage()
        )
        self.assertEqual(expected, observed)

    def test_categorical_coverage_is_global_not_impossible_per_split(self) -> None:
        rows, embeddings = _corpus()
        for index, row in enumerate(rows):
            row["model_family"] = f"production-family-{index % 35:02d}"
        subset = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=_base_lineage(),
        )
        by_id = {str(row["configuration_id"]): row for row in rows}
        observed_families = {
            str(by_id[str(item["configuration_id"])]["model_family"])
            for item in subset["selection"]
        }
        self.assertEqual(len(observed_families), 35)
        self.assertEqual(
            sum(item["split"] == "validation" for item in subset["selection"]),
            16,
        )

    def test_optimizer_scheduler_pairs_are_covered_before_latent_diversity(self) -> None:
        rows, embeddings = _corpus()
        subset = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=_base_lineage(),
        )
        by_id = {str(row["configuration_id"]): row for row in rows}
        expected = {
            (str(row["optimizer"]), str(row["scheduler"])) for row in rows
        }
        observed = {
            (
                str(by_id[str(item["configuration_id"])]["optimizer"]),
                str(by_id[str(item["configuration_id"])]["scheduler"]),
            )
            for item in subset["selection"]
        }
        self.assertEqual(observed, expected)

    def test_nested_subset_and_active_scores_fail_closed(self) -> None:
        rows, embeddings = _corpus()
        pilot = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=_base_lineage(),
        )
        expanded_request = TransferSubsetRequestV3(
            target_hardware_id="nvidia_h100_sxm_80gb",
            label_budget=256,
            active_learning=True,
        )
        all_scores = {
            str(row["configuration_id"]): float(index)
            for index, row in enumerate(rows)
        }
        tampered = json.loads(json.dumps(pilot))
        tampered["selection"][0]["source_group"] = "tampered"
        with self.assertRaisesRegex(ValueError, "content hash"):
            select_transfer_subset(
                rows,
                embeddings,
                expanded_request,
                base_lineage=_base_lineage(),
                prior_manifest=tampered,
                active_scores=all_scores,
            )
        graph_tampered = json.loads(json.dumps(pilot))
        graph_tampered["selection"][0]["graph_path"] = "graphs/other.json"
        unhashed = dict(graph_tampered)
        unhashed.pop("subset_sha256")
        graph_tampered["subset_sha256"] = hashlib.sha256(
            canonical_json(unhashed).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "graph-path lineage"):
            select_transfer_subset(
                rows,
                embeddings,
                expanded_request,
                base_lineage=_base_lineage(),
                prior_manifest=graph_tampered,
                active_scores=all_scores,
            )
        with self.assertRaisesRegex(ValueError, "preceding frozen subset"):
            select_transfer_subset(
                rows,
                embeddings,
                expanded_request,
                base_lineage=_base_lineage(),
                active_scores=all_scores,
            )
        with self.assertRaisesRegex(ValueError, "scores are missing"):
            select_transfer_subset(
                rows,
                embeddings,
                expanded_request,
                base_lineage=_base_lineage(),
                prior_manifest=pilot,
                active_scores={"cfg_0000": 1.0},
            )
        nonfinite_scores = dict(all_scores)
        nonfinite_scores["cfg_0209"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            select_transfer_subset(
                rows,
                embeddings,
                expanded_request,
                base_lineage=_base_lineage(),
                prior_manifest=pilot,
                active_scores=nonfinite_scores,
            )

    def test_all_four_nested_budget_manifests_are_exact_and_deterministic(self) -> None:
        rows, embeddings = _corpus((800, 140, 140))
        prior = None
        all_scores = {
            str(row["configuration_id"]): float(index % 31) / 31.0
            for index, row in enumerate(rows)
        }
        expected_counts = {
            128: {"train": 96, "validation": 16, "test": 16},
            256: {"train": 192, "validation": 32, "test": 32},
            512: {"train": 384, "validation": 64, "test": 64},
            1024: {"train": 768, "validation": 128, "test": 128},
        }
        previous_ids: set[str] = set()
        for budget in (128, 256, 512, 1024):
            request = TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=budget,
                active_learning=budget > 128,
            )
            manifest = select_transfer_subset(
                rows,
                embeddings,
                request,
                base_lineage=_base_lineage(),
                prior_manifest=prior,
                active_scores=all_scores if budget > 128 else None,
            )
            self.assertEqual(manifest["split_counts"], expected_counts[budget])
            self.assertEqual(len(manifest["selection"]), budget)
            current_ids = {
                str(item["configuration_id"]) for item in manifest["selection"]
            }
            self.assertLessEqual(previous_ids, current_ids)
            self.assertEqual(
                manifest,
                select_transfer_subset(
                    rows,
                    embeddings,
                    request,
                    base_lineage=_base_lineage(),
                    prior_manifest=prior,
                    active_scores=all_scores if budget > 128 else None,
                ),
            )
            previous_ids = current_ids
            prior = manifest

    def test_target_manifest_retains_oom_attempts_and_paired_references(self) -> None:
        rows, embeddings = _corpus()
        subset = select_transfer_subset(
            rows,
            embeddings,
            TransferSubsetRequestV3(
                target_hardware_id="nvidia_h100_sxm_80gb",
                label_budget=128,
            ),
            base_lineage=_base_lineage(),
        )
        profile_sha256 = "a" * 64
        attempts = [
            TransferLabelAttemptV3(
                base_configuration_id=str(item["configuration_id"]),
                paired_configuration_id=f"paired_{index:04d}",
                split=str(item["split"]),
                base_hardware_id="nvidia_a10g_24gb_aws_g5",
                target_hardware_id="nvidia_h100_sxm_80gb",
                hardware_profile_sha256=profile_sha256,
                base_targets=(100.0, 40.0, 50.0, 4000.0, 4500.0, 30.0),
                status="accepted",
                failure_stage="none",
                target_targets=(70.0, 60.0, 70.0, 3800.0, 4200.0, 45.0),
                original_microbatch_size=8,
                measured_microbatch_size=8,
                repaired_from_configuration_id=None,
                run_payload={"retained": True},
                graph_path=f"target_graph_{index:04d}.json",
            )
            for index, item in enumerate(subset["selection"])
        ]
        probe_base_id = str(subset["memory_boundary_probes"][0]["configuration_id"])
        attempts.append(
            TransferLabelAttemptV3(
                base_configuration_id=probe_base_id,
                paired_configuration_id=f"{probe_base_id}_memory_probe_h100",
                split="memory_probe",
                base_hardware_id="nvidia_a10g_24gb_aws_g5",
                target_hardware_id="nvidia_h100_sxm_80gb",
                hardware_profile_sha256=profile_sha256,
                base_targets=(100.0, 40.0, 50.0, 4000.0, 4500.0, 30.0),
                status="oom",
                failure_stage="allocator",
                target_targets=None,
                original_microbatch_size=64,
                measured_microbatch_size=64,
                repaired_from_configuration_id=None,
                run_payload=None,
                graph_path="target_probe_graph.json",
                failure_payload={
                    "exception_type": "OutOfMemoryError",
                    "message": "synthetic OOM",
                    "allocator_state": {"memory_reserved_bytes": 1024},
                },
            )
        )
        unsafe_repair = replace(
            attempts[0],
            measured_configuration_id="target-repair",
            measured_microbatch_size=4,
            repaired_from_configuration_id=attempts[0].base_configuration_id,
        )
        with self.assertRaisesRegex(ValueError, "exact A10 configuration"):
            unsafe_repair.validate()
        replace(unsafe_repair, split="memory_probe").validate()
        increased_probe = replace(
            attempts[0],
            split="memory_probe",
            measured_configuration_id="target-larger-batch-probe",
            measured_microbatch_size=16,
            repaired_from_configuration_id=attempts[0].base_configuration_id,
        )
        increased_probe.validate()
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            replace(increased_probe, measured_microbatch_size=24).validate()
        serialized = {**asdict(attempts[0]), "attempt_sha256": attempts[0].sha256}
        self.assertEqual(
            TransferLabelAttemptV3.from_dict(serialized).sha256,
            attempts[0].sha256,
        )
        serialized["base_targets"] = [101.0, *serialized["base_targets"][1:]]
        with self.assertRaisesRegex(ValueError, "content hash"):
            TransferLabelAttemptV3.from_dict(serialized)
        manifest = build_target_training_manifest(subset, attempts)
        jsonschema.validate(manifest, target_training_manifest_json_schema())
        self.assertEqual(len(manifest["samples"]), 128)
        self.assertEqual(len(manifest["oom_attempts"]), 1)
        self.assertEqual(len(manifest["memory_probe_attempts"]), 1)
        self.assertTrue(manifest["failed_attempts_retained"])
        self.assertEqual(manifest["samples"][0]["base_hardware_id"], "nvidia_a10g_24gb_aws_g5")
        self.assertEqual(
            manifest["deployment"]["hardware_allowlist"],
            ["nvidia_h100_sxm_80gb"],
        )
        outside_probe = replace(
            attempts[-1],
            base_configuration_id="outside_frozen_subset",
            paired_configuration_id="outside_frozen_subset_h100",
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen subset"):
            build_target_training_manifest(subset, [*attempts[:-1], outside_probe])


if __name__ == "__main__":
    unittest.main()
