from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from perfseer_v3.dataset_pack.contracts import (
    AttemptStatus,
    EPOCH_MEASUREMENT_VERSION,
    EpochMeasurement,
    FailureStage,
    FingerprintBundle,
    GpuCleanupEvidence,
    LABEL_RUN_RECORD_VERSION,
    LabelRunRecord,
    TARGET_NAMES,
    TelemetrySample,
)
from perfseer_v3.dataset_pack.finalization import (
    CampaignFinalization,
    FINALIZATION_VERSION,
    FinalizationError,
    ResolvedAcceptedSlot,
    SPLIT_COUNTS,
    _fit_target_transform,
    _resolve_workspace_slots,
    _split_fingerprint,
    _verify_provenance_sidecars,
    build_campaign_finalization,
    finalize_workspace,
    publish_campaign_finalization,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.materialization import (
    TASK_COMPLETION_VERSION,
    TASK_LOOP_STATE_VERSION,
    TaskCompletionReceipt,
    TaskLoopState,
    TaskMaterializationError,
)
from perfseer_v3.dataset_pack.repair import (
    make_quota_replacement,
    next_oom_repair,
    quarantine_candidate,
)
from perfseer_v3.dataset_pack.sampler import TargetCandidate, build_target_manifest
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.dataset_pack.task_registry import TaskRegistryEntry, load_task_registry
from perfseer_v3.dataset_pack.workflow import SlotState, _save_slot, run_task_workflow


def _cleanup() -> GpuCleanupEvidence:
    return GpuCleanupEvidence(
        pre_sample_device_used_vram_mib=100.0,
        post_cleanup_device_used_vram_mib=100.0,
        release_tolerance_mib=64.0,
        cleanup_seconds=1.0,
        stable_dwell_seconds=0.5,
        child_process_tree_exited=True,
        owned_gpu_processes_remaining=0,
        passed=True,
    )


def _fingerprints(candidate: TargetCandidate) -> FingerprintBundle:
    return FingerprintBundle(
        source_sha256=candidate.source_sha256,
        graph_sha256=canonical_sha256(
            {
                "source_sha256": candidate.source_sha256,
                "factory_id": candidate.factory_id,
                "architecture_parameters": candidate.architecture_parameters,
                "input_signature": candidate.input_signature,
                "training_step_id": candidate.training_step_id,
            }
        ),
        environment_sha256=canonical_sha256({"environment": "fixture"}),
        hardware_sha256=canonical_sha256({"hardware": "fixture-a10g"}),
        support_contract_sha256=canonical_sha256({"support": "fixture"}),
        dataset_sha256=canonical_sha256({"task_id": candidate.task_id}),
    )


def _failed_oom(candidate: TargetCandidate, task: TaskRegistryEntry) -> LabelRunRecord:
    record = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id=canonical_sha256({"configuration_id": candidate.candidate_id, "status": "oom"}),
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.OOM,
        failure_stage=FailureStage.ALLOCATOR,
        fingerprints=_fingerprints(candidate),
        gpu_uuid="GPU-fixture-a10g",
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=str(candidate.execution["mode"]),
        compile_completed_before_epoch_1=False,
        expected_train_examples=task.expected_train_examples,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=(),
        finite_loss_epochs=(),
        finite_gradient_epochs=(),
        epoch_measurements=(),
        cleanup=_cleanup(),
        targets=None,
    )
    record.validate_against_configuration(candidate, task)
    return record


def _accepted(candidate: TargetCandidate, task: TaskRegistryEntry) -> LabelRunRecord:
    expected_batches = math.ceil(task.expected_train_examples / candidate.microbatch_size)
    expected_steps = math.ceil(expected_batches / candidate.gradient_accumulation_steps)
    base = 800.0 + float(candidate.ordinal % 101)
    sm = 45.0 + float(candidate.ordinal % 31)
    memory = 30.0 + float(candidate.ordinal % 41)
    vram = 4_000.0 + float(candidate.ordinal % 2_000)
    measurements = tuple(
        EpochMeasurement(
            version=EPOCH_MEASUREMENT_VERSION,
            epoch=epoch,
            epoch_completed=True,
            epoch_ms=base * (0.99 if epoch == 3 else 1.0 if epoch == 4 else 1.01),
            examples_seen=task.expected_train_examples,
            batches_seen=expected_batches,
            microsteps=expected_batches,
            optimizer_steps=expected_steps,
            loss_finite=True,
            gradients_finite=True,
            telemetry_complete=True,
            telemetry_samples=(
                TelemetrySample(
                    timestamp_offset_s=float(epoch - 3),
                    duration_s=float(epoch - 1),
                    sm_util_percent=min(100.0, sm + epoch - 4),
                    memory_controller_util_percent=min(100.0, memory + epoch - 4),
                    device_used_vram_mib=vram + 10.0 * (epoch - 3),
                    throttle_reason_bits=0,
                ),
            ),
            peak_torch_reserved_mib=vram - 100.0 + 20.0 * (epoch - 3),
            requested_backend_id=str(candidate.execution["backend_id"]),
            observed_backend_id=str(candidate.execution["backend_id"]),
            foreign_process_detected=False,
        )
        for epoch in (3, 4, 5)
    )
    draft = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id=canonical_sha256(
            {"configuration_id": candidate.candidate_id, "status": "accepted"}
        ),
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.ACCEPTED,
        failure_stage=FailureStage.NONE,
        fingerprints=_fingerprints(candidate),
        gpu_uuid="GPU-fixture-a10g",
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=str(candidate.execution["mode"]),
        compile_completed_before_epoch_1=candidate.execution["mode"] == "compiled",
        expected_train_examples=task.expected_train_examples,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=(1, 2, 3, 4, 5),
        finite_loss_epochs=(1, 2, 3, 4, 5),
        finite_gradient_epochs=(1, 2, 3, 4, 5),
        epoch_measurements=measurements,
        cleanup=_cleanup(),
        targets=None,
    )
    record = replace(draft, targets=draft.aggregate_targets())
    record.validate_against_configuration(candidate, task)
    return record


class ExactCampaignFinalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_target_manifest()
        cls.task_registry = load_task_registry()
        cls.tasks = {row.task_id: row for row in cls.task_registry.entries}
        slots = []
        failures = []
        for index, root in enumerate(cls.manifest.candidates):
            task = cls.tasks[root.task_id]
            if index == 0:
                failure = _failed_oom(root, task)
                repair = next_oom_repair(
                    root,
                    failure,
                    repair_index=0,
                    root_candidate_id=root.candidate_id,
                )
                candidate = repair.candidate
                record = _accepted(candidate, task)
                lineage_payload = {
                    "root_candidate_id": root.candidate_id,
                    "attempted_candidate_ids": (root.candidate_id, candidate.candidate_id),
                    "oom_repair_attempt_ids": (repair.attempt_id,),
                    "quarantine_ids": (),
                    "substitution_ids": (),
                }
                slots.append(
                    ResolvedAcceptedSlot(
                        root,
                        (root, candidate),
                        record,
                        (repair.attempt_id,),
                        (),
                        (),
                        canonical_sha256(lineage_payload),
                    )
                )
                cls.repair = repair
                failures.append(failure)
            else:
                slots.append(ResolvedAcceptedSlot.direct(root, _accepted(root, task)))
        cls.slots = tuple(slots)
        cls.failures = tuple(failures)
        cls.accepted_by_id = {
            row.accepted_record.configuration_id: row.accepted_record for row in cls.slots
        }
        cls.manifest_sha256 = cls.manifest.sha256
        cls.finalization = build_campaign_finalization(
            cls.manifest,
            cls.slots,
            cls.failures,
        )
        record_sha256s = {
            configuration_id: canonical_sha256(asdict(record))
            for configuration_id, record in cls.accepted_by_id.items()
        }
        receipts = []
        for task in cls.task_registry.entries:
            task_slots = tuple(
                row for row in cls.slots if row.root_candidate.task_id == task.task_id
            )
            resolutions = tuple(
                (row.root_candidate.candidate_id, row.accepted_candidate.candidate_id)
                for row in task_slots
            )
            accepted_record_sha256s = tuple(
                record_sha256s[row.accepted_candidate.candidate_id] for row in task_slots
            )
            records_manifest_sha256 = canonical_sha256(
                {
                    "resolutions": resolutions,
                    "accepted_record_sha256s": accepted_record_sha256s,
                }
            )
            draft = TaskCompletionReceipt(
                version=TASK_COMPLETION_VERSION,
                task_id=task.task_id,
                target_manifest_sha256=cls.manifest_sha256,
                dataset_fingerprint=task_slots[0].accepted_record.fingerprints.dataset_sha256,
                resolutions=resolutions,
                accepted_record_sha256s=accepted_record_sha256s,
                records_manifest_sha256=records_manifest_sha256,
                receipt_sha256="",
            )
            receipt = replace(
                draft,
                receipt_sha256=canonical_sha256(draft.unhashed_payload()),
            )
            receipt.validate()
            receipts.append(receipt)
        cls.completed_receipts = tuple(receipts)
        cls.completed_loop = TaskLoopState(
            version=TASK_LOOP_STATE_VERSION,
            target_manifest_sha256=cls.manifest_sha256,
            task_registry_sha256=cls.task_registry.sha256,
            completed_task_ids=tuple(row.task_id for row in cls.task_registry.entries),
            completed_receipt_sha256s=tuple(row.receipt_sha256 for row in receipts),
            active_task_id=None,
        )
        cls.completed_loop.validate(cls.completed_loop.completed_task_ids)

    def _completed_task_evidence(
        self,
        workspace: Path,
    ) -> tuple[TaskLoopState, tuple[TaskCompletionReceipt, ...]]:
        for receipt in self.completed_receipts:
            atomic_write_json(
                workspace
                / "state"
                / "completed_materializations"
                / f"{receipt.task_id}.json",
                receipt.to_dict(),
            )
        return self.completed_loop, self.completed_receipts

    def test_exact_counts_quotas_repair_and_splits(self) -> None:
        result = self.finalization
        result.validate()
        self.assertEqual(len(result.rows), 18_000)
        self.assertEqual(
            sum(len(row.accepted_record.epoch_measurements) for row in self.slots),
            54_000,
        )
        self.assertEqual(
            {measurement.epoch for row in self.slots for measurement in row.accepted_record.epoch_measurements},
            {3, 4, 5},
        )
        self.assertEqual(Counter(row.split for row in result.rows), Counter(SPLIT_COUNTS))
        self.assertEqual(len({row.configuration_id for row in result.rows}), 18_000)
        self.assertEqual(len({row.run_id for row in result.rows}), 18_000)
        self.assertEqual(result.audit_report["failure"]["attempt_count"], 1)
        self.assertEqual(result.audit_report["batch_repair"]["oom_repair_attempt_count"], 1)
        self.assertEqual(result.audit_report["batch_repair"]["batch_changed_row_count"], 1)
        self.assertEqual(result.dataset_manifest["artifact_scope"], "a10g_label_pack")
        self.assertFalse(result.dataset_manifest["graph_ir_materialized"])

    def test_source_and_graph_groups_never_leak(self) -> None:
        source_splits: dict[str, str] = {}
        graph_splits: dict[str, str] = {}
        for row in self.finalization.rows:
            self.assertEqual(source_splits.setdefault(row.source_group, row.split), row.split)
            self.assertEqual(graph_splits.setdefault(row.graph_sha256, row.split), row.split)
        held_out = self.finalization.audit_report["split"]["held_out_generated_lineages"]
        self.assertGreaterEqual(len(held_out), 10)
        for lineage in held_out:
            self.assertEqual(
                {row.split for row in self.finalization.rows if row.source_lineage == lineage},
                {"test"},
            )

    def test_target_transform_reads_training_rows_only(self) -> None:
        rows = self.finalization.rows
        split_hash = _split_fingerprint(rows)
        baseline = _fit_target_transform(rows, split_hash)
        poisoned_holdout = tuple(
            row
            if row.split == "train"
            else replace(
                row,
                target_values=tuple(value + 1_000_000.0 for value in row.target_values),
                row_sha256="",
            )
            for row in rows
        )
        poisoned_holdout = tuple(
            row
            if row.row_sha256
            else replace(row, row_sha256=canonical_sha256(row.unhashed_payload()))
            for row in poisoned_holdout
        )
        self.assertEqual(
            _fit_target_transform(poisoned_holdout, split_hash),
            baseline,
        )
        first_train = next(index for index, row in enumerate(rows) if row.split == "train")
        changed = list(rows)
        draft = replace(
            changed[first_train],
            target_values=tuple(value + 10.0 for value in changed[first_train].target_values),
            row_sha256="",
        )
        changed[first_train] = replace(
            draft,
            row_sha256=canonical_sha256(draft.unhashed_payload()),
        )
        self.assertNotEqual(
            _fit_target_transform(tuple(changed), split_hash).transform_sha256,
            baseline.transform_sha256,
        )

    def test_publication_is_byte_deterministic_and_verify_only_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = publish_campaign_finalization(self.finalization, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            second = publish_campaign_finalization(self.finalization, output, verify_only=True)
            after = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["accepted_run_count"], 18_000)
            (output / "audit_report.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "differs"):
                publish_campaign_finalization(self.finalization, output, verify_only=True)

    def test_duplicate_terminal_run_id_fails(self) -> None:
        rows = list(self.finalization.rows)
        draft = replace(rows[1], run_id=rows[0].run_id, row_sha256="")
        rows[1] = replace(draft, row_sha256=canonical_sha256(draft.unhashed_payload()))
        changed = CampaignFinalization(
            tuple(rows),
            self.finalization.failure_index,
            self.finalization.target_transform,
            self.finalization.audit_report,
            self.finalization.dataset_manifest,
        )
        with self.assertRaisesRegex(FinalizationError, "run IDs"):
            changed.validate()

    def test_missing_frozen_slot_fails_closed(self) -> None:
        with self.assertRaisesRegex(FinalizationError, "18,000 frozen roots"):
            build_campaign_finalization(
                self.manifest,
                self.slots[:-1],
                self.failures,
            )

    def test_generated_replacement_stays_in_its_source_group(self) -> None:
        root = next(
            row for row in self.manifest.candidates if row.family_id == "independent_generated"
        )
        failed = replace(
            _failed_oom(root, self.tasks[root.task_id]),
            status=AttemptStatus.FAILED,
            failure_stage=FailureStage.FORWARD,
        )
        quarantine = quarantine_candidate(
            root,
            failed,
            failure_stage=FailureStage.FORWARD,
            reason_code="fixture_forward_failure",
        )
        replacement = make_quota_replacement(
            root,
            quarantine,
            replacement_index=0,
        ).candidate
        self.assertEqual(
            (replacement.source_lineage, replacement.source_sha256, replacement.source_split),
            (root.source_lineage, root.source_sha256, root.source_split),
        )
        self.assertNotEqual(replacement.candidate_id, root.candidate_id)

    def test_workspace_reconstructs_quarantine_and_substitution_lineage(self) -> None:
        root = self.slots[1].root_candidate
        task = self.tasks[root.task_id]
        failure = replace(
            _failed_oom(root, task),
            status=AttemptStatus.FAILED,
            failure_stage=FailureStage.FORWARD,
        )
        quarantine = quarantine_candidate(
            root,
            failure,
            failure_stage=FailureStage.FORWARD,
            reason_code="fixture_forward_failure",
        )
        substitution = make_quota_replacement(
            root,
            quarantine,
            replacement_index=0,
        )
        replacement = substitution.candidate
        replacement_record = _accepted(replacement, task)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            atomic_write_json(
                workspace / "attempts" / "repairs" / "oom" / f"{self.repair.attempt_id}.json",
                asdict(self.repair),
            )
            first = self.slots[0]
            _save_slot(
                workspace / "state" / "slots" / f"{first.root_candidate.candidate_id}.json",
                SlotState(
                    version="perfseer_v3_a10g_18k_workflow_v2",
                    root_candidate_id=first.root_candidate.candidate_id,
                    chain_root_candidate=first.root_candidate,
                    current_candidate=first.accepted_candidate,
                    oom_attempts=(self.repair,),
                    repair_index=1,
                    replacement_index=0,
                    state_sha256="",
                ),
            )
            atomic_write_json(
                workspace
                / "attempts"
                / "repairs"
                / "quarantine"
                / f"{quarantine.quarantine_id}.json",
                asdict(quarantine),
            )
            atomic_write_json(
                workspace
                / "attempts"
                / "repairs"
                / "substitution"
                / f"{substitution.substitution_id}.json",
                asdict(substitution),
            )
            _save_slot(
                workspace / "state" / "slots" / f"{root.candidate_id}.json",
                SlotState(
                    version="perfseer_v3_a10g_18k_workflow_v2",
                    root_candidate_id=root.candidate_id,
                    chain_root_candidate=replacement,
                    current_candidate=replacement,
                    oom_attempts=(),
                    repair_index=0,
                    replacement_index=1,
                    state_sha256="",
                ),
            )
            resolutions = {
                row.root_candidate.candidate_id: row.accepted_candidate.candidate_id
                for row in self.slots
            }
            resolutions[root.candidate_id] = replacement.candidate_id
            accepted = dict(self.accepted_by_id)
            del accepted[root.candidate_id]
            accepted[replacement.candidate_id] = replacement_record
            reconstructed = _resolve_workspace_slots(
                workspace,
                self.manifest,
                resolutions,
                accepted,
                (*self.failures, failure),
            )
        resolved = reconstructed[root.ordinal]
        self.assertEqual(resolved.accepted_candidate, replacement)
        self.assertEqual(resolved.quarantine_ids, (quarantine.quarantine_id,))
        self.assertEqual(resolved.substitution_ids, (substitution.substitution_id,))
        self.assertEqual(resolved.attempted_candidates, (root, replacement))

    def test_verify_only_missing_workspace_is_read_only_in_api_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "missing"
            with self.assertRaisesRegex(TaskMaterializationError, "manifest is missing"):
                finalize_workspace(workspace, verify_only=True)
            self.assertFalse(workspace.exists())
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "finalize_a10g_18k_pack.py"),
                    "--workspace",
                    str(workspace),
                    "--verify-only",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(workspace.exists())

    def test_finalize_workspace_validates_all_22_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            loop, receipts = self._completed_task_evidence(workspace)
            expected_receipt = {"receipt_sha256": canonical_sha256({"final": True})}
            with (
                patch(
                    "perfseer_v3.dataset_pack.finalization.freeze_initial_target_manifest",
                    return_value=(self.manifest, workspace / "manifests" / "initial_targets.jsonl"),
                ) as freeze,
                patch(
                    "perfseer_v3.dataset_pack.finalization.load_task_loop_state",
                    return_value=loop,
                ) as load_loop,
                patch(
                    "perfseer_v3.dataset_pack.finalization._load_records",
                    return_value=(self.accepted_by_id, self.failures),
                ),
                patch(
                    "perfseer_v3.dataset_pack.finalization._resolve_workspace_slots",
                    return_value=self.slots,
                ),
                patch(
                    "perfseer_v3.dataset_pack.finalization._verify_provenance_sidecars",
                    return_value={"payload_sidecars_verified": True},
                ),
                patch(
                    "perfseer_v3.dataset_pack.finalization.build_campaign_finalization",
                    return_value=self.finalization,
                ) as build,
                patch(
                    "perfseer_v3.dataset_pack.finalization.publish_campaign_finalization",
                    return_value=expected_receipt,
                ) as publish,
            ):
                actual = finalize_workspace(workspace, verify_only=True)
            self.assertEqual(actual, expected_receipt)
            self.assertEqual(len(receipts), 22)
            freeze.assert_called_once_with(workspace.resolve(), create_if_missing=False)
            self.assertFalse(load_loop.call_args.kwargs["create_if_missing"])
            build.assert_called_once()
            self.assertTrue(publish.call_args.kwargs["verify_only"])

    def test_completed_workflow_invokes_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            loop, _ = self._completed_task_evidence(workspace)
            with (
                patch(
                    "perfseer_v3.dataset_pack.workflow.freeze_initial_target_manifest",
                    return_value=(self.manifest, workspace / "manifests" / "initial_targets.jsonl"),
                ),
                patch(
                    "perfseer_v3.dataset_pack.workflow.load_task_loop_state",
                    return_value=loop,
                ),
                patch("perfseer_v3.dataset_pack.workflow.KaggleCliClient"),
                patch("perfseer_v3.dataset_pack.workflow.PinnedMleBenchPreparer"),
                patch("perfseer_v3.dataset_pack.workflow.TaskMaterializer"),
                patch("perfseer_v3.dataset_pack.workflow._verify_completed_prefix"),
                patch("perfseer_v3.dataset_pack.workflow.discover_a10g_probes", return_value=()),
                patch("perfseer_v3.dataset_pack.finalization.finalize_workspace") as finalize,
            ):
                run_task_workflow(
                    workspace=workspace,
                    repository_root=ROOT,
                    mlebench_checkout=ROOT,
                )
            finalize.assert_called_once_with(workspace.resolve())

    def test_workspace_reconstructs_direct_and_oom_lineages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repair_path = (
                workspace
                / "attempts"
                / "repairs"
                / "oom"
                / f"{self.repair.attempt_id}.json"
            )
            atomic_write_json(repair_path, asdict(self.repair))
            first = self.slots[0]
            state = SlotState(
                version="perfseer_v3_a10g_18k_workflow_v2",
                root_candidate_id=first.root_candidate.candidate_id,
                chain_root_candidate=first.root_candidate,
                current_candidate=first.accepted_candidate,
                oom_attempts=(self.repair,),
                repair_index=1,
                replacement_index=0,
                state_sha256="",
            )
            _save_slot(
                workspace / "state" / "slots" / f"{first.root_candidate.candidate_id}.json",
                state,
            )
            resolutions = {
                row.root_candidate.candidate_id: row.accepted_candidate.candidate_id
                for row in self.slots
            }
            reconstructed = _resolve_workspace_slots(
                workspace,
                self.manifest,
                resolutions,
                self.accepted_by_id,
                self.failures,
            )
        self.assertEqual(len(reconstructed), 18_000)
        self.assertEqual(reconstructed[0].repair_lineage_sha256, self.slots[0].repair_lineage_sha256)
        self.assertEqual(reconstructed[-1].accepted_candidate, self.slots[-1].accepted_candidate)

    def test_provenance_sidecars_are_hash_bound(self) -> None:
        records = tuple(self.accepted_by_id.values())
        environment_payload = {"environment": "fixture"}
        hardware_payload = {"hardware": "fixture-a10g"}
        environment_sha = canonical_sha256(environment_payload)
        hardware_sha = canonical_sha256(hardware_payload)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            atomic_write_json(
                workspace / "provenance" / "environment" / f"{environment_sha}.json",
                environment_payload,
            )
            hardware_path = (
                workspace / "provenance" / "hardware" / f"{hardware_sha}.json"
            )
            atomic_write_json(hardware_path, hardware_payload)
            report = _verify_provenance_sidecars(workspace, records)
            self.assertTrue(report["payload_sidecars_verified"])
            hardware_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "hash differs"):
                _verify_provenance_sidecars(workspace, records)


if __name__ == "__main__":
    unittest.main()
