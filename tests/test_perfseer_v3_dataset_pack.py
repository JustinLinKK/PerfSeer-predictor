from __future__ import annotations

import math
import json
import importlib
import csv
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest import mock
import wave
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.dataset_pack import TARGET_HARDWARE_ID
from perfseer_v3.dataset_pack.fingerprints import (
    canonical_sha256,
    canonical_value,
    file_sha256,
    joined_sha256,
)
from perfseer_v3.dataset_pack.quota import DEFAULT_QUOTA_CONFIG_PATH, load_quota_plan
from perfseer_v3.dataset_pack.contracts import (
    AttemptStatus,
    CorpusLayer,
    EPOCH_MEASUREMENT_VERSION,
    EpochMeasurement,
    FailureStage,
    FingerprintBundle,
    GpuCleanupEvidence,
    LABEL_RUN_RECORD_VERSION,
    LabelRunRecord,
    TelemetrySample,
    TargetVector,
    WORKLOAD_CONFIG_VERSION,
    WorkloadConfiguration,
)
from perfseer_v3.dataset_pack.operation_support import (
    DEFAULT_GENERATED_CONTRACT_PATH,
    DEFAULT_SUPPORT_POLICY_PATH,
    OperationSupportError,
    build_operation_support_contract,
)
from perfseer_v3.dataset_pack.coverage_config import (
    DEFAULT_COVERAGE_CONFIG_PATH,
    CoverageConfigError,
    load_operation_coverage_config,
)
from perfseer_v3.dataset_pack.model_registry import (
    DEFAULT_MODEL_REGISTRY_PATH,
    ModelRegistry,
    ModelRegistryError,
    load_model_registry,
)
from perfseer_v3.dataset_pack.task_registry import (
    DEFAULT_TASK_REGISTRY_PATH,
    FROZEN_TASKS,
    TaskRegistry,
    TaskRegistryError,
    load_task_registry,
)
from perfseer_v3.dataset_pack.composite_sampler import (
    CompositePlanningError,
    build_composite_block_registry,
    plan_composite_corpus,
    select_composite_gap_wave,
)
from perfseer_v3.dataset_pack.composite_blocks import (
    build_composite_block,
    verify_composite_block,
)
from perfseer_v3.dataset_pack.dispatcher_verify import (
    DISPATCH_EVIDENCE_VERSION,
    DISPATCH_REQUEST_VERSION,
    DispatchEvidence,
    DispatchRequest,
    DispatchVerification,
    DispatchVerificationError,
    verify_dispatch,
)
from perfseer_v3.dataset_pack.operation_coverage import (
    OperationCoverageError,
    build_operation_coverage_report,
    freeze_runtime_allowlist,
    make_coverage_observation,
    make_structural_coverage_observation,
)
from perfseer_v3.dataset_pack.operation_sampler import (
    OperationPlanningError,
    build_operation_generator_registry,
    plan_operation_corpus,
    select_operation_gap_wave,
)
from perfseer_v3.dataset_pack.operation_benchmarks import run_p1_fixture_suite
from perfseer_v3.dataset_pack.adapters import (
    AdapterError,
    adapter_for_task,
    all_task_adapters,
)
from perfseer_v3.dataset_pack.model_golden import run_model_factory_audit
from perfseer_v3.dataset_pack.models import ModelFactoryError, golden_validate_model
from perfseer_v3.dataset_pack.models.base import tensor_outputs
from perfseer_v3.dataset_pack.models.factory import default_adapter_id
from perfseer_v3.dataset_pack.compatibility import (
    DEPLOYMENT_SCHEDULERS,
    MUON_MATRIX_FREE_FAMILIES,
    CompatibilityRequest,
    evaluate_compatibility,
)
from perfseer_v3.dataset_pack.generated_lineages import (
    build_generated_lineage_registry,
)
from perfseer_v3.dataset_pack.planning_bundle import (
    DatasetPlanningBundle,
    PLANNING_BUNDLE_VERSION,
    build_dataset_planning_bundle,
)
from perfseer_v3.dataset_pack.repair import (
    OOM_REPAIR_ACTION,
    RepairError,
    fill_to_quota,
    make_quota_replacement,
    next_oom_repair,
    quarantine_candidate,
)
from perfseer_v3.dataset_pack.sampler import (
    BATCH_LADDERS,
    CandidatePlanningError,
    MINIMUM_TERMINAL_LEARNING_RATE_RATIO,
    TERMINAL_DECAY_SCHEDULERS,
    _batch_classification,
    build_target_manifest,
    scheduler_minimum_lr_ratio,
)
from perfseer_v3.dataset_pack.local_validation import (
    SIGNATURE_VERSION,
    LocalValidationError,
    ValidationSignature,
    _coverage_tokens,
    build_local_validation_plan,
    signature_factors,
    validation_signature_from_dict,
)
from perfseer_v3.dataset_pack.local_runtime import (
    _build_model,
    _build_optimizer,
    _build_scheduler,
    bind_candidate_batch_shape,
    five_epoch_optimizer_steps,
    validate_candidate_execution,
)
from perfseer_v3.dataset_pack.local_task_fixture import build_local_real_format_batch
from perfseer_v3.dataset_pack.kaggle import (
    KaggleCliClient,
    KaggleCompetitionProbe,
    KaggleMaterializationError,
    KaggleRemoteFile,
    extract_zip_archive,
    inspect_zip_archive,
    validate_external_credentials,
)
from perfseer_v3.dataset_pack.materialization import (
    TASK_COMPLETION_VERSION,
    TASK_LOOP_STATE_VERSION,
    TaskCompletionReceipt,
    TaskLoopState,
    TaskMaterializer,
    freeze_initial_target_manifest,
    initial_task_loop_state,
)
from perfseer_v3.dataset_pack.mlebench_bridge import PinnedMleBenchPreparer
from perfseer_v3.dataset_pack.label_worker import WORKER_ENVELOPE_VERSION
from perfseer_v3.dataset_pack.prepared_view import (
    PREPARED_EXAMPLE_COUNT,
    _JIGSAW_TARGETS,
    _RANZCR_TARGETS,
    build_shared_prepared_view,
    load_and_verify_prepared_view,
)
from perfseer_v3.dataset_pack.real_data import RealPreparedDataError, VerifiedPreparedDataset
from perfseer_v3.dataset_pack.a10g_runner import (
    A10G_RUN_RESULT_VERSION,
    FiveEpochRunResult,
    TelemetryReading,
    run_five_epoch_training,
)
from perfseer_v3.dataset_pack.supervisor import (
    AttemptSupervisor,
    SupervisorError,
    build_accepted_label_record,
    lock_campaign_environment,
    wait_for_gpu_cleanup,
)
from perfseer_v3.dataset_pack.workflow import (
    _advance_failed_slot,
    _load_slot,
    _new_slot,
    _save_slot,
    _verify_completed_prefix,
)
from perfseer_v3.dataset_pack.storage import FreeSpaceGuard, GIB
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.op_registry import OperationRegistry


class FingerprintTests(unittest.TestCase):
    def test_mapping_and_set_order_do_not_change_hash(self) -> None:
        left = {"beta": {3, 1, 2}, "alpha": [True, None]}
        right = {"alpha": [True, None], "beta": {2, 3, 1}}
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        self.assertEqual(canonical_value(left), canonical_value(right))

    def test_mapping_key_collisions_and_nonfinite_floats_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "string keys"):
            canonical_sha256({1: "integer", "1": "string"})
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    canonical_sha256({"value": value})

    def test_hash_is_sensitive_and_dataclasses_are_supported(self) -> None:
        @dataclass(frozen=True)
        class Payload:
            name: str
            values: tuple[int, ...]

        self.assertNotEqual(
            canonical_sha256(Payload("a", (1, 2))),
            canonical_sha256(Payload("a", (1, 3))),
        )

    def test_file_and_joined_hashes_are_full_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"perfseer-a10g")
            digest = file_sha256(path, block_bytes=3)
        self.assertEqual(len(digest), 64)
        self.assertEqual(joined_sha256(file=digest), canonical_sha256({"file": digest}))
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            joined_sha256(file="not-a-digest")


class QuotaPlanTests(unittest.TestCase):
    @staticmethod
    def _load_mutation(mutator) -> None:
        raw = yaml.safe_load(DEFAULT_QUOTA_CONFIG_PATH.read_text(encoding="utf-8"))
        mutator(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quota.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            load_quota_plan(path)

    def test_frozen_quota_plan_is_exact(self) -> None:
        plan = load_quota_plan()
        self.assertEqual(plan.target_hardware_id, TARGET_HARDWARE_ID)
        self.assertEqual(plan.accepted_successful_configurations, 18_000)
        self.assertEqual(plan.accepted_runs_per_configuration, 1)
        self.assertEqual(plan.expected_accepted_run_records, 18_000)
        self.assertEqual(plan.expected_measured_epoch_records, 54_000)
        self.assertEqual(plan.protocol.warmup_epochs, (1, 2))
        self.assertEqual(plan.protocol.measured_epochs, (3, 4, 5))
        self.assertEqual(len(plan.cells), 35)
        self.assertEqual(
            plan.modality_totals,
            {
                "audio": 1_300,
                "generated": 3_250,
                "graph": 1_300,
                "nlp": 5_050,
                "tabular": 950,
                "vision": 6_150,
            },
        )
        self.assertEqual(sum(plan.family_totals.values()), 18_000)
        self.assertEqual(len(plan.sha256), 64)

    def test_quota_hash_is_deterministic(self) -> None:
        self.assertEqual(load_quota_plan().sha256, load_quota_plan().sha256)

    def test_balanced_family_drift_and_reordering_fail_closed(self) -> None:
        def balanced(raw) -> None:
            raw["end_to_end"]["family_quotas"][0]["accepted_configurations"] += 1
            raw["end_to_end"]["family_quotas"][1]["accepted_configurations"] -= 1

        with self.assertRaisesRegex(ValueError, "frozen ordered 35-cell"):
            self._load_mutation(balanced)

        def reorder(raw) -> None:
            cells = raw["end_to_end"]["family_quotas"]
            cells[0], cells[1] = cells[1], cells[0]

        with self.assertRaisesRegex(ValueError, "frozen ordered 35-cell"):
            self._load_mutation(reorder)

    def test_single_run_epoch_protocol_safety_fields_fail_closed(self) -> None:
        mutations = {
            "direct epoch": lambda raw: raw["protocol"].__setitem__(
                "derive_end_to_end_epoch_time_from_step_time", True
            ),
            "run count": lambda raw: raw["end_to_end"].__setitem__(
                "expected_accepted_run_records", 1
            ),
            "measured epochs": lambda raw: raw["protocol"].__setitem__(
                "measured_epochs", [2, 3, 4]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._load_mutation(mutate)

    def test_missing_unknown_duplicate_and_coerced_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*allow_mps"):
            self._load_mutation(lambda raw: raw["protocol"].pop("allow_mps"))
        with self.assertRaisesRegex(ValueError, "unknown=.*extra"):
            self._load_mutation(lambda raw: raw.__setitem__("extra", True))
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self._load_mutation(
                lambda raw: raw["end_to_end"].__setitem__(
                    "accepted_runs_per_configuration", 1.0
                )
            )
        duplicate = DEFAULT_QUOTA_CONFIG_PATH.read_text(encoding="utf-8") + "version: duplicate\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate YAML key"):
                load_quota_plan(path)

    def test_generated_lineage_maximum_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum must be positive"):
            self._load_mutation(
                lambda raw: raw["generated_lineages"].__setitem__(
                    "maximum_accepted_per_lineage", 0
                )
            )

    def test_a10_is_not_accepted_as_a10g(self) -> None:
        raw = yaml.safe_load(DEFAULT_QUOTA_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["target_hardware_id"] = "nvidia_a10"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quota.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen AWS A10G"):
                load_quota_plan(path)


def _workload_configuration() -> WorkloadConfiguration:
    return WorkloadConfiguration(
        version=WORKLOAD_CONFIG_VERSION,
        source_lineage="test_lineage",
        source_sha256="a" * 64,
        generator_version="generator_v1",
        mutation_specification={"depth": 2},
        modality="vision",
        architecture_family="resnet50",
        architecture_parameters={"width": 16},
        task_id="histopathologic-cancer",
        dataset_revision="fixture_v1",
        input_signature={"shape": [2, 3, 8, 8]},
        training_step_id="classification_v1",
        microbatch_size=64,
        gradient_accumulation_steps=1,
        precision_policy={"parameter": "float32", "autocast": False},
        optimizer={"name": "adamw", "learning_rate": 0.001},
        scheduler={"name": "none", "progress": 0.0, "step_unit": "none"},
        activation_checkpointing={"enabled": False},
        execution={"mode": "compiled"},
        seed_policy={"seed": 7},
        dependency_revision="lock-v1",
        container_digest="sha256:" + "b" * 64,
        cuda_revision="12.4",
        driver_revision="550",
        pytorch_revision="2.6",
        kernel_library_revisions={"cudnn": "9"},
        target_hardware_id="nvidia_a10g_24gb_aws_g5",
        observation_protocol_sha256=load_quota_plan().protocol.sha256,
    )


def _cleanup(*, passed: bool = True) -> GpuCleanupEvidence:
    return GpuCleanupEvidence(
        pre_sample_device_used_vram_mib=100.0,
        post_cleanup_device_used_vram_mib=110.0 if passed else 500.0,
        release_tolerance_mib=256.0,
        cleanup_seconds=1.0,
        stable_dwell_seconds=0.5,
        child_process_tree_exited=True,
        owned_gpu_processes_remaining=0,
        passed=passed,
    )


def _fingerprints(layer: CorpusLayer) -> FingerprintBundle:
    return FingerprintBundle(
        source_sha256="1" * 64,
        graph_sha256="2" * 64,
        environment_sha256="3" * 64,
        hardware_sha256="4" * 64,
        support_contract_sha256="5" * 64,
        dataset_sha256="6" * 64 if layer == CorpusLayer.END_TO_END else None,
        synthetic_input_spec_sha256=(
            "7" * 64 if layer in {CorpusLayer.OPERATION, CorpusLayer.COMPOSITE} else None
        ),
    )


def _oom_failure_record(candidate) -> LabelRunRecord:
    return LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id=f"oom-{candidate.candidate_id[:12]}",
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.OOM,
        failure_stage=FailureStage.ALLOCATOR,
        fingerprints=_fingerprints(CorpusLayer.END_TO_END),
        gpu_uuid="GPU-test",
        target_hardware_id=TARGET_HARDWARE_ID,
        capture_workload_sha256="8" * 64,
        profile_workload_sha256="8" * 64,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=candidate.execution["mode"],
        compile_completed_before_epoch_1=False,
        expected_train_examples=candidate.batch_plan.expected_train_examples,
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
    )


class ContractTests(unittest.TestCase):
    @staticmethod
    def _epoch(
        epoch: int,
        epoch_ms: float,
        *,
        sm: float,
        duration: float,
        memory_controller: float,
        vram: float,
        reserved: float,
    ) -> EpochMeasurement:
        return EpochMeasurement(
            version=EPOCH_MEASUREMENT_VERSION,
            epoch=epoch,
            epoch_completed=True,
            epoch_ms=epoch_ms,
            examples_seen=4_096,
            batches_seen=64,
            microsteps=64,
            optimizer_steps=64,
            loss_finite=True,
            gradients_finite=True,
            telemetry_complete=True,
            telemetry_samples=(
                TelemetrySample(
                    timestamp_offset_s=float(epoch),
                    duration_s=duration,
                    sm_util_percent=sm,
                    memory_controller_util_percent=memory_controller,
                    device_used_vram_mib=vram,
                ),
            ),
            peak_torch_reserved_mib=reserved,
            requested_backend_id="inductor_cuda",
            observed_backend_id="inductor_cuda",
            foreign_process_detected=False,
        )

    @classmethod
    def _accepted_run(cls) -> LabelRunRecord:
        record = LabelRunRecord(
            version=LABEL_RUN_RECORD_VERSION,
            run_id="run-1",
            configuration_id=_workload_configuration().configuration_id,
            status=AttemptStatus.ACCEPTED,
            failure_stage=FailureStage.NONE,
            fingerprints=_fingerprints(CorpusLayer.END_TO_END),
            gpu_uuid="GPU-test",
            target_hardware_id=TARGET_HARDWARE_ID,
            capture_workload_sha256="8" * 64,
            profile_workload_sha256="8" * 64,
            coverage_cell_ids=("vision:resnet50",),
            task_id="histopathologic-cancer",
            execution_mode="compiled",
            compile_completed_before_epoch_1=True,
            expected_train_examples=4_096,
            microbatch_size=64,
            gradient_accumulation_steps=1,
            total_epochs=5,
            warmup_epochs=(1, 2),
            measured_epochs=(3, 4, 5),
            completed_epochs=(1, 2, 3, 4, 5),
            finite_loss_epochs=(1, 2, 3, 4, 5),
            finite_gradient_epochs=(1, 2, 3, 4, 5),
            epoch_measurements=(
                cls._epoch(3, 1_000.0, sm=80.0, duration=1.0, memory_controller=60.0, vram=10_000.0, reserved=9_000.0),
                cls._epoch(4, 1_020.0, sm=100.0, duration=3.0, memory_controller=90.0, vram=12_000.0, reserved=11_800.0),
                cls._epoch(5, 990.0, sm=90.0, duration=2.0, memory_controller=70.0, vram=11_000.0, reserved=10_000.0),
            ),
            cleanup=_cleanup(),
        )
        return replace(record, targets=record.aggregate_targets())

    def test_configuration_id_is_stable_and_protocol_sensitive(self) -> None:
        config = _workload_configuration()
        self.assertEqual(config.configuration_id, _workload_configuration().configuration_id)
        changed = replace(config, observation_protocol_sha256="f" * 64)
        self.assertNotEqual(config.configuration_id, changed.configuration_id)
        with self.assertRaisesRegex(ValueError, "AWS A10G"):
            replace(config, target_hardware_id="nvidia_a10").validate()

    def test_target_masks_require_null_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be null"):
            TargetVector(
                values=(10.0, 0.0, None, None, None, None),
                validity=(True, False, False, False, False, False),
            ).validate()

    def test_label_run_aggregates_only_epochs_three_to_five(self) -> None:
        run = self._accepted_run()
        run.validate()
        self.assertAlmostEqual(run.targets.values[0], (1_000.0 + 1_020.0 + 990.0) / 3)
        self.assertAlmostEqual(run.targets.values[1], (80.0 + 300.0 + 180.0) / 6.0)
        self.assertEqual(run.targets.values[2:], (100.0, 12_000.0, 11_800.0, 90.0))
        self.assertLessEqual(run.epoch_time_relative_spread, 0.10)
        task = next(
            row
            for row in load_task_registry().entries
            if row.task_id == run.task_id
        )
        self.assertEqual(
            len(run.bound_record_sha256(_workload_configuration(), task)),
            64,
        )
        self.assertFalse(hasattr(run, "record_sha256"))

    def test_label_run_rejects_instability_throttle_contamination_and_cleanup(self) -> None:
        run = self._accepted_run()
        unstable_epoch = replace(run.epoch_measurements[-1], epoch_ms=700.0)
        unstable = replace(run, epoch_measurements=(*run.epoch_measurements[:-1], unstable_epoch))
        unstable = replace(unstable, targets=unstable.aggregate_targets())
        with self.assertRaisesRegex(ValueError, "10% epoch stability"):
            unstable.validate()
        throttled_sample = replace(
            run.epoch_measurements[0].telemetry_samples[0], throttle_reason_bits=1
        )
        throttled_epoch = replace(
            run.epoch_measurements[0], telemetry_samples=(throttled_sample,)
        )
        with self.assertRaisesRegex(ValueError, "throttle"):
            replace(run, epoch_measurements=(throttled_epoch, *run.epoch_measurements[1:])).validate()
        with self.assertRaisesRegex(ValueError, "foreign GPU process"):
            replace(
                run,
                epoch_measurements=(replace(run.epoch_measurements[0], foreign_process_detected=True), *run.epoch_measurements[1:]),
            ).validate()
        with self.assertRaisesRegex(ValueError, "passing cleanup"):
            replace(run, cleanup=_cleanup(passed=False)).validate()

    def test_rejected_attempt_never_retains_supervised_targets(self) -> None:
        run = self._accepted_run()
        failed = replace(
            run,
            status=AttemptStatus.FAILED,
            failure_stage=FailureStage.FORWARD,
            targets=None,
            completed_epochs=(1, 2),
            finite_loss_epochs=(1, 2),
            finite_gradient_epochs=(1, 2),
            epoch_measurements=(),
        )
        failed.validate()
        with self.assertRaisesRegex(ValueError, "must not retain"):
            replace(failed, targets=run.targets).validate()

    def test_accepted_run_requires_exact_prepared_traversal(self) -> None:
        run = self._accepted_run()
        task = next(
            row
            for row in load_task_registry().entries
            if row.task_id == "histopathologic-cancer"
        )
        run.validate_against_configuration(_workload_configuration(), task)
        wrong = replace(run.epoch_measurements[0], examples_seen=1, batches_seen=1, microsteps=1, optimizer_steps=1)
        changed = replace(run, epoch_measurements=(wrong, *run.epoch_measurements[1:]))
        changed = replace(changed, targets=changed.aggregate_targets())
        with self.assertRaisesRegex(ValueError, "epoch traversal"):
            changed.validate()

        coordinated = replace(
            run,
            expected_train_examples=8,
            microbatch_size=8,
            epoch_measurements=tuple(
                replace(
                    row,
                    examples_seen=8,
                    batches_seen=1,
                    microsteps=1,
                    optimizer_steps=1,
                )
                for row in run.epoch_measurements
            ),
        )
        coordinated = replace(coordinated, targets=coordinated.aggregate_targets())
        coordinated.validate()
        with self.assertRaisesRegex(ValueError, "not bound"):
            coordinated.validate_against_configuration(_workload_configuration(), task)


class TaskAndModelRegistryTests(unittest.TestCase):
    @staticmethod
    def _mutated_registry(path: Path, mutator) -> Path:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutator(raw)
        directory = tempfile.mkdtemp()
        output = Path(directory) / path.name
        output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return output

    def test_task_registry_is_pinned_complete_unmaterialized_and_round_trips(self) -> None:
        registry = load_task_registry()
        self.assertEqual(len(registry.entries), 22)
        self.assertEqual(
            tuple((row.task_id, row.kaggle_slug, row.modality) for row in registry.entries),
            FROZEN_TASKS,
        )
        self.assertEqual(
            registry.metadata_revision,
            "507f92e1138bb6e40dac5c6ee7a6758e6424bf97",
        )
        self.assertFalse(registry.training_approved)
        self.assertEqual(registry.archive_checksum_state, "materialization_required")
        self.assertTrue(all(not row.source_checksums for row in registry.entries))
        self.assertEqual(
            {row.expected_train_examples for row in registry.entries},
            {4_096},
        )
        self.assertTrue(
            all(
                row.validation_rules["archive_checksum_state"] == "materialization_required"
                for row in registry.entries
            )
        )
        self.assertEqual(TaskRegistry.from_dict(registry.to_dict()), registry)
        self.assertEqual(len(registry.sha256), 64)

    def test_task_registry_rejects_guessed_reordered_unknown_and_duplicate_metadata(self) -> None:
        mutations = {
            "slug": lambda raw: raw["entries"][0].__setitem__("kaggle_slug", "guessed"),
            "order": lambda raw: raw["entries"].__setitem__(
                slice(0, 2), [raw["entries"][1], raw["entries"][0]]
            ),
            "unknown": lambda raw: raw["entries"][0].__setitem__("extra", True),
            "approval": lambda raw: raw.__setitem__("training_approved", True),
            "size": lambda raw: raw["entries"][0].__setitem__(
                "compressed_size_hint_bytes", 1
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                path = self._mutated_registry(DEFAULT_TASK_REGISTRY_PATH, mutate)
                with self.assertRaises(TaskRegistryError):
                    load_task_registry(path)
        duplicate = DEFAULT_TASK_REGISTRY_PATH.read_text(encoding="utf-8") + "version: duplicate\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.yaml"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(TaskRegistryError, "duplicate YAML key"):
                load_task_registry(path)

    def test_task_registry_serialized_hash_and_schema_fail_closed(self) -> None:
        payload = load_task_registry().to_dict()
        payload["registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(TaskRegistryError, "hash mismatch"):
            TaskRegistry.from_dict(payload)
        payload = load_task_registry().to_dict()
        payload["entries"][0]["unknown"] = True
        with self.assertRaisesRegex(TaskRegistryError, "schema differs"):
            TaskRegistry.from_dict(payload)
        payload = load_task_registry().to_dict()
        payload["size_hint_source_url"] = (
            "https://evil.invalid/"
            "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
        )
        payload["registry_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "registry_sha256"}
        )
        with self.assertRaisesRegex(TaskRegistryError, "pinned MLE-bench Lite table"):
            TaskRegistry.from_dict(payload)
        payload = load_task_registry().to_dict()
        payload["entries"][0]["prepared_view_recipe"]["require_exact_prepared_count"] = False
        payload["registry_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "registry_sha256"}
        )
        with self.assertRaisesRegex(TaskRegistryError, "exact-count contract"):
            TaskRegistry.from_dict(payload)

    def test_model_registry_matches_all_quota_cells_and_round_trips(self) -> None:
        tasks = load_task_registry()
        quota = load_quota_plan()
        registry = load_model_registry(tasks=tasks, quota=quota)
        self.assertEqual(len(registry.entries), 35)
        self.assertEqual(
            tuple((row.modality, row.family_id) for row in registry.entries),
            tuple((modality, family_id) for modality, family_id, _, _ in quota.cells_as_tuples)
            if hasattr(quota, "cells_as_tuples")
            else tuple((row.modality, row.family_id) for row in quota.cells),
        )
        self.assertEqual(registry.factory_status, "implemented_phase3_factories_v1")
        self.assertFalse(registry.training_approved)
        self.assertEqual(registry.task_registry_sha256, tasks.sha256)
        self.assertEqual(registry.quota_plan_sha256, quota.sha256)
        self.assertEqual(ModelRegistry.from_dict(registry.to_dict()), registry)
        self.assertEqual(len(registry.sha256), 64)

    def test_model_registry_rejects_missing_family_unknown_adapter_and_completion_claim(self) -> None:
        mutations = {
            "missing": lambda raw: raw["entries"].pop(),
            "adapter": lambda raw: raw["entries"][0].__setitem__(
                "adapter_ids", ["not-a-task"]
            ),
            "implemented": lambda raw: raw.__setitem__("factory_status", "implemented"),
            "approval": lambda raw: raw.__setitem__("training_approved", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                path = self._mutated_registry(DEFAULT_MODEL_REGISTRY_PATH, mutate)
                with self.assertRaises(ModelRegistryError):
                    load_model_registry(path)

    def test_model_registry_serialized_hash_and_schema_fail_closed(self) -> None:
        mutations = {
            "adapter": lambda row: row.__setitem__("adapter_ids", ["not-a-task"]),
            "factory": lambda row: row.__setitem__(
                "factory_id", "perfseer_v3.dataset_pack.models.wrong"
            ),
            "lineage": lambda row: row.__setitem__(
                "source_lineage", "planning:model_family:wrong:v1"
            ),
            "source": lambda row: row.__setitem__("source_sha256", "0" * 64),
            "fields": lambda row: row.__setitem__(
                "adjustable_architecture_fields", ["unbound_change"]
            ),
        }
        for name, mutate in mutations.items():
            payload = load_model_registry().to_dict()
            mutate(payload["entries"][0])
            payload["registry_sha256"] = canonical_sha256(
                {key: value for key, value in payload.items() if key != "registry_sha256"}
            )
            with self.subTest(name=name):
                with self.assertRaises(ModelRegistryError):
                    ModelRegistry.from_dict(payload)
        payload = load_model_registry().to_dict()
        payload["unknown"] = True
        with self.assertRaisesRegex(ModelRegistryError, "schema differs"):
            ModelRegistry.from_dict(payload)


def _write_csv_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_zip_fixture(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _touch_fixture(path: Path, payload: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_public_task_fixture(task_id: str, public: Path) -> None:
    public.mkdir(parents=True)
    if task_id == "histopathologic-cancer":
        _write_csv_fixture(public / "train_labels.csv", [{"id": "a", "label": 1}])
        _touch_fixture(public / "train/a.tif")
    elif task_id == "dogs-vs-cats":
        _write_zip_fixture(public / "train.zip", {"train/cat.0.jpg": b"cat"})
    elif task_id == "dog-breed":
        _write_csv_fixture(public / "labels.csv", [{"id": "a", "breed": "beagle"}])
        _touch_fixture(public / "train/a.jpg")
    elif task_id == "siim-isic-melanoma":
        _write_csv_fixture(public / "train.csv", [{"image_name": "a", "target": 0}])
        _touch_fixture(public / "jpeg/train/a.jpg")
    elif task_id == "aptos2019":
        _write_csv_fixture(public / "train.csv", [{"id_code": "a", "diagnosis": 2}])
        _touch_fixture(public / "train_images/a.png")
    elif task_id == "aerial-cactus":
        _write_csv_fixture(public / "train.csv", [{"id": "a.jpg", "has_cactus": 1}])
        _write_zip_fixture(public / "train.zip", {"train/a.jpg": b"cactus"})
    elif task_id == "plant-pathology":
        _write_csv_fixture(
            public / "train.csv",
            [{"image_id": "a", "healthy": 1, "multiple_diseases": 0, "rust": 0, "scab": 0}],
        )
        _touch_fixture(public / "images/a.jpg")
    elif task_id == "ranzcr-clip":
        row: dict[str, object] = {"StudyInstanceUID": "a"}
        row.update({name: int(index == 0) for index, name in enumerate(_RANZCR_TARGETS)})
        _write_csv_fixture(public / "train.csv", [row])
        _touch_fixture(public / "train/a.jpg")
    elif task_id == "leaf-classification":
        _write_csv_fixture(public / "train.csv", [{"id": 1, "species": "Acer", "margin1": 0.5}])
        _touch_fixture(public / "images/1.jpg")
    elif task_id == "denoising-dirty-documents":
        _touch_fixture(public / "train/a.png", b"dirty")
        _touch_fixture(public / "train_cleaned/a.png", b"clean")
    elif task_id == "jigsaw-toxic":
        row = {"id": "a", "comment_text": "hello"}
        row.update({name: int(index == 0) for index, name in enumerate(_JIGSAW_TARGETS)})
        _write_csv_fixture(public / "train.csv", [row])
    elif task_id == "detecting-insults":
        _write_csv_fixture(public / "train.csv", [{"Insult": 0, "Date": "x", "Comment": "hello"}])
    elif task_id == "spooky-author":
        _write_csv_fixture(public / "train.csv", [{"id": "a", "text": "hello", "author": "EAP"}])
    elif task_id == "random-acts-of-pizza":
        (public / "train.json").write_text(
            json.dumps([{"request_id": "a", "request_text": "pizza", "requester_received_pizza": True}]),
            encoding="utf-8",
        )
    elif task_id in {"text-normalization-english", "text-normalization-russian"}:
        prefix = "en" if task_id.endswith("english") else "ru"
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["sentence_id", "token_id", "class", "before", "after"])
        writer.writeheader()
        writer.writerow({"sentence_id": 0, "token_id": 0, "class": "PLAIN", "before": "one", "after": "one"})
        _write_zip_fixture(public / f"{prefix}_train.csv.zip", {f"{prefix}_train.csv": buffer.getvalue().encode()})
    elif task_id == "mlsp-2013-birds":
        _write_csv_fixture(public / "essential_data/CVfolds_2.txt", [{"rec_id": 0, "fold": 0}])
        _write_csv_fixture(public / "essential_data/rec_id2filename.txt", [{"rec_id": 0, "filename": "bird"}])
        (public / "essential_data/rec_labels_test_hidden.txt").write_text("rec_id,[labels]\n0,1,3\n", encoding="utf-8")
        _touch_fixture(public / "essential_data/src_wavs/bird.wav")
    elif task_id == "icml-2013-whale":
        _write_zip_fixture(public / "train2.zip", {"train2/20130101_x_TRAIN0_1.aif": b"whale"})
    elif task_id == "nyc-taxi-fare":
        _write_csv_fixture(public / "labels.csv", [{"key": "a", "fare_amount": 4.5, "passenger_count": 1}])
    elif task_id == "nomad2018":
        _write_csv_fixture(public / "train.csv", [{"id": 1, "spacegroup": 2, "formation_energy_ev_natom": 0.2, "bandgap_energy_ev": 1.2}])
        _touch_fixture(public / "train/1/geometry.xyz", b"1\nfixture\nH 0 0 0\n")
    elif task_id == "tabular-playground-dec-2021":
        _write_csv_fixture(public / "train.csv", [{"Id": 1, "feature": 0.5, "Cover_Type": 2}])
    elif task_id == "tabular-playground-may-2022":
        _write_csv_fixture(public / "train.csv", [{"id": 1, "f_00": 0.5, "target": 1}])
    else:
        raise AssertionError(f"missing synthetic fixture for {task_id}")


class _FakeKaggleClient:
    def __init__(self) -> None:
        self.authentication_count = 0
        self.download_count = 0

    def authenticate(self) -> None:
        self.authentication_count += 1

    def probe_competition(self, slug: str) -> KaggleCompetitionProbe:
        return KaggleCompetitionProbe(
            slug,
            (KaggleRemoteFile(f"{slug}-fixture.bin", 7, "fixture"),),
        )

    def download_competition(self, slug: str, destination_archive: str | Path) -> Path:
        self.download_count += 1
        destination = Path(destination_archive)
        _write_zip_fixture(destination, {"raw/fixture.txt": b"fixture"})
        return destination


class _FakeMleBenchPreparer:
    def __init__(self) -> None:
        self.prepare_count = 0

    def prepare(self, entry, raw, public, private) -> None:
        self.prepare_count += 1
        self.assert_raw = Path(raw) / "raw/fixture.txt"
        if not self.assert_raw.is_file():
            raise AssertionError("fake preparer did not receive safely extracted raw data")
        _make_public_task_fixture(entry.task_id, Path(public))
        _touch_fixture(Path(private) / "answers.txt")


class _MutatingOncePreparer(_FakeMleBenchPreparer):
    def prepare(self, entry, raw, public, private) -> None:
        self.prepare_count += 1
        raw_fixture = Path(raw) / "raw/fixture.txt"
        if self.prepare_count == 1:
            raw_fixture.unlink()
            raise RuntimeError("simulated interrupted mutating preparer")
        if not raw_fixture.is_file():
            raise AssertionError("preparer retry did not rebuild immutable raw input")
        _make_public_task_fixture(entry.task_id, Path(public))
        _touch_fixture(Path(private) / "answers.txt")


class MinimalMaterializationTests(unittest.TestCase):
    def test_production_entrypoints_bootstrap_fresh_source_checkout(self) -> None:
        environment = os.environ.copy()
        # -S skips editable-install .pth files. Supplying only the dependency
        # directory proves each entrypoint adds this checkout's src tree itself.
        environment["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
        for script in (
            ROOT / "scripts" / "run_a10g_18k_pack.py",
            ROOT / "scripts" / "finalize_a10g_18k_pack.py",
        ):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, "-S", str(script), "--help"],
                    cwd=Path(tempfile.gettempdir()),
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

        specification = importlib.util.spec_from_file_location(
            "perfseer_a10g_entrypoint_test",
            ROOT / "scripts" / "run_a10g_18k_pack.py",
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        calls: list[str] = []
        with (
            mock.patch.object(
                module,
                "validate_external_credentials",
                side_effect=lambda *_: calls.append("credentials"),
            ),
            mock.patch.object(
                module,
                "verify_runtime_dependencies",
                side_effect=lambda: calls.append("runtime"),
            ),
            mock.patch.object(
                module,
                "verify_frozen_preparers",
                side_effect=lambda *_: calls.append("preflight"),
            ) as preflight,
            mock.patch.object(
                module,
                "verify_all_kaggle_access",
                side_effect=lambda *_: calls.append("kaggle"),
            ),
            mock.patch.object(
                module,
                "run_task_workflow",
                side_effect=lambda **_: calls.append("workflow"),
            ) as workflow,
            mock.patch.object(
                sys,
                "argv",
                [
                    str(ROOT / "scripts" / "run_a10g_18k_pack.py"),
                    "--workspace",
                    str(Path(tempfile.gettempdir()) / "perfseer-a10g-entrypoint-test"),
                    "--mlebench-checkout",
                    "/opt/mle-bench",
                ],
            ),
        ):
            module.main()
        self.assertEqual(
            calls,
            ["credentials", "runtime", "preflight", "kaggle", "workflow"],
        )
        self.assertEqual(len(preflight.call_args.args[1]), 22)
        self.assertEqual(
            workflow.call_args.kwargs["repository_root"],
            ROOT,
        )

    def test_runtime_preflight_imports_dependencies_and_requires_cuda(self) -> None:
        specification = importlib.util.spec_from_file_location(
            "perfseer_a10g_runtime_preflight_test",
            ROOT / "scripts" / "run_a10g_18k_pack.py",
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        versions = module.FROZEN_RUNTIME_VERSIONS

        def unavailable_cuda(command, **_kwargs):
            output = b'{"cuda": "fixture", "available": false}' if command[-1] == "torch" else b""
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

        with (
            mock.patch.object(module.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(
                module.importlib.metadata,
                "version",
                side_effect=lambda name: versions[name],
            ),
            mock.patch.object(
                module.subprocess,
                "run",
                side_effect=unavailable_cuda,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "working CUDA"):
                module.verify_runtime_dependencies()

        probe_environments = []

        def broken_import(command, **kwargs):
            probe_environments.append(kwargs["env"])
            return subprocess.CompletedProcess(
                command,
                1 if command[-1] == "tensorflow" else 0,
                stdout=b"secret-output",
                stderr=b"secret-error",
            )

        with (
            mock.patch.dict(
                module.os.environ,
                {
                    "KAGGLE_USERNAME": "secret-user",
                    "KAGGLE_KEY": "secret-key",
                    "AWS_SECRET_ACCESS_KEY": "secret-cloud",
                },
            ),
            mock.patch.object(module.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(
                module.importlib.metadata,
                "version",
                side_effect=lambda name: versions[name],
            ),
            mock.patch.object(module.subprocess, "run", side_effect=broken_import),
        ):
            with self.assertRaisesRegex(RuntimeError, "TensorFlow cannot be imported") as raised:
                module.verify_runtime_dependencies()
        self.assertNotIn("secret", str(raised.exception))
        self.assertTrue(probe_environments)
        for environment in probe_environments:
            self.assertEqual(environment["KAGGLE_USERNAME"], "runtime-probe")
            self.assertEqual(environment["KAGGLE_KEY"], "runtime-probe")
            self.assertIn(".perfseer-runtime-probe-", environment["KAGGLE_CONFIG_DIR"])
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
            self.assertNotIn("secret-user", environment.values())
            self.assertNotIn("secret-key", environment.values())
            self.assertNotIn("secret-cloud", environment.values())

        with (
            mock.patch.object(module.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(
                module.importlib.metadata,
                "version",
                side_effect=lambda name: "wrong" if name == "torch" else versions[name],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "torch==wrong"):
                module.verify_runtime_dependencies()

    def test_disk_guard_uses_only_peak_formula_and_free_headroom(self) -> None:
        guard = FreeSpaceGuard()
        accepted = guard.assess(
            task_id="fixture",
            stage="before_download",
            current_non_task_bytes=1 * GIB,
            task_archive_bytes=2 * GIB,
            extracted_bytes=4 * GIB,
            extraction_temporary_bytes=1 * GIB,
            observed_free_bytes=100 * GIB,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.projected_peak_bytes, 48 * GIB)
        refused = guard.assess(
            task_id="fixture",
            stage="before_extraction",
            current_non_task_bytes=1 * GIB,
            task_archive_bytes=2 * GIB,
            extracted_bytes=4 * GIB,
            extraction_temporary_bytes=1 * GIB,
            observed_free_bytes=40 * GIB,
            already_present_task_bytes=2 * GIB,
        )
        self.assertFalse(refused.accepted)
        with self.assertRaisesRegex(Exception, "cannot enter"):
            refused.require()

    def test_credentials_inside_checkout_are_rejected_without_reading_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "secrets/kaggle.json"
            _touch_fixture(config, b"not-json-and-must-not-be-read")
            os.chmod(config, 0o600)
            with self.assertRaisesRegex(KaggleMaterializationError, "outside"):
                validate_external_credentials(root, environment={"KAGGLE_CONFIG_DIR": str(config.parent)})
            evidence = validate_external_credentials(root, environment={"KAGGLE_API_TOKEN": "secret"})
            self.assertEqual(evidence.source, "api_token_environment")
            self.assertIsNone(evidence.location)

    def test_pinned_preparer_masks_external_file_credentials_from_child(self) -> None:
        entry = load_task_registry().entries[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            source = (
                checkout
                / "mlebench/competitions"
                / entry.kaggle_slug
                / "prepare.py"
            )
            _touch_fixture(source, b"# fixture")
            raw = root / "task-cache/raw"
            raw.mkdir(parents=True)
            operator_config = root / "operator-kaggle"
            operator_config.mkdir()
            _touch_fixture(operator_config / "kaggle.json", b"operator-secret")
            preparer = object.__new__(PinnedMleBenchPreparer)
            object.__setattr__(preparer, "checkout", checkout)
            object.__setattr__(preparer, "timeout_seconds", 60)
            captured = {}

            class Completed:
                returncode = 0

            def fake_run(command, **kwargs):
                captured.update(kwargs["env"])
                public = Path(command[command.index("--public") + 1])
                private = Path(command[command.index("--private") + 1])
                _touch_fixture(public / "data.txt")
                _touch_fixture(private / "answers.txt")
                return Completed()

            with mock.patch.dict(
                os.environ,
                {
                    "KAGGLE_CONFIG_DIR": str(operator_config),
                    "KAGGLE_API_TOKEN": "operator-token",
                },
                clear=False,
            ), mock.patch(
                "perfseer_v3.dataset_pack.mlebench_bridge.subprocess.run",
                side_effect=fake_run,
            ):
                preparer.prepare(entry, raw, root / "public", root / "private")
            isolated_config = Path(captured["KAGGLE_CONFIG_DIR"])
            self.assertNotEqual(isolated_config, operator_config)
            self.assertFalse(isolated_config.exists())
            self.assertNotIn("KAGGLE_API_TOKEN", captured)
            self.assertNotIn("KAGGLE_USERNAME", captured)
            self.assertNotIn("KAGGLE_KEY", captured)
            self.assertTrue((operator_config / "kaggle.json").is_file())

    def test_safe_zip_extracts_and_rejects_traversal_duplicate_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.zip"
            _write_zip_fixture(valid, {"train/a.csv": b"x,y\n1,2\n"})
            inventory = inspect_zip_archive(valid, maximum_uncompressed_bytes=1024)
            extracted = extract_zip_archive(valid, root / "raw", inventory)
            self.assertEqual((extracted / "train/a.csv").read_bytes(), b"x,y\n1,2\n")
            late_directory = root / "late-directory.zip"
            with zipfile.ZipFile(late_directory, "w") as archive:
                archive.writestr("nested/value.txt", b"value")
                archive.writestr("nested/", b"")
            late_inventory = inspect_zip_archive(
                late_directory, maximum_uncompressed_bytes=1024
            )
            late_extracted = extract_zip_archive(
                late_directory, root / "late-directory", late_inventory
            )
            self.assertEqual((late_extracted / "nested/value.txt").read_bytes(), b"value")
            traversal = root / "traversal.zip"
            _write_zip_fixture(traversal, {"../escape": b"x"})
            with self.assertRaises(KaggleMaterializationError):
                inspect_zip_archive(traversal, maximum_uncompressed_bytes=1024)
            duplicate = root / "duplicate.zip"
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("A", b"x")
                archive.writestr("a", b"y")
            with self.assertRaisesRegex(KaggleMaterializationError, "duplicate"):
                inspect_zip_archive(duplicate, maximum_uncompressed_bytes=1024)
            symlink = root / "symlink.zip"
            with zipfile.ZipFile(symlink, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "target")
            with self.assertRaisesRegex(KaggleMaterializationError, "symlink"):
                inspect_zip_archive(symlink, maximum_uncompressed_bytes=1024)

    def test_cli_failure_does_not_echo_secret_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "kaggle-fixture"
            executable.write_text("#!/bin/sh\necho token-super-secret >&2\nexit 1\n", encoding="utf-8")
            os.chmod(executable, 0o700)
            client = KaggleCliClient(executable=str(executable), timeout_seconds=2)
            with self.assertRaises(KaggleMaterializationError) as caught:
                client.authenticate()
            self.assertNotIn("token-super-secret", str(caught.exception))

    def test_kaggle_inventory_uses_supported_pagination(self) -> None:
        client = KaggleCliClient(executable=sys.executable, timeout_seconds=2)
        pages = (
            subprocess.CompletedProcess(
                (),
                0,
                "Next Page Token = second-page\nname,size,creationDate\nb.bin,2,now\n",
                "",
            ),
            subprocess.CompletedProcess(
                (),
                0,
                "name,size,creationDate\na.bin,1,now\n",
                "",
            ),
        )
        with mock.patch.object(client, "_run", side_effect=pages) as run:
            probe = client.probe_competition("fixture")
        self.assertEqual(tuple(row.name for row in probe.files), ("a.bin", "b.bin"))
        self.assertIn("200", run.call_args_list[0].args[0])
        self.assertNotIn("--page-token", run.call_args_list[0].args[0])
        self.assertEqual(
            run.call_args_list[1].args[0][-2:],
            ["--page-token", "second-page"],
        )

    def test_all_22_shared_view_recipes_emit_exactly_4096_records(self) -> None:
        for entry in load_task_registry().entries:
            with self.subTest(task_id=entry.task_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                public = root / "public"
                _make_public_task_fixture(entry.task_id, public)
                manifest = build_shared_prepared_view(
                    entry,
                    public,
                    root / "prepared",
                    archive_sha256="a" * 64,
                )
                self.assertEqual(manifest.prepared_example_count, PREPARED_EXAMPLE_COUNT)
                self.assertEqual(manifest.source_example_count, 1)
                self.assertTrue(manifest.sampling_with_replacement)
                rows = (root / "prepared/samples.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), PREPARED_EXAMPLE_COUNT)
                self.assertEqual(json.loads(rows[-1])["view_index"], PREPARED_EXAMPLE_COUNT - 1)

    def test_prepared_view_rejects_jsonl_and_selected_source_tampering(self) -> None:
        entry = next(row for row in load_task_registry().entries if row.task_id == "jigsaw-toxic")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            build_shared_prepared_view(entry, public, prepared, archive_sha256="a" * 64)
            rows = (prepared / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0])
            row["target"][0] = 0
            rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
            (prepared / "samples.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "hash changed"):
                load_and_verify_prepared_view(
                    entry, public, prepared, archive_sha256="a" * 64
                )

        entry = next(
            row for row in load_task_registry().entries if row.task_id == "histopathologic-cancer"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            build_shared_prepared_view(entry, public, prepared, archive_sha256="b" * 64)
            (public / "train/a.tif").write_bytes(b"changed")
            with self.assertRaisesRegex(Exception, "source bytes changed"):
                load_and_verify_prepared_view(
                    entry, public, prepared, archive_sha256="b" * 64
                )

    def test_frozen_manifest_rejects_non_identity_row_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, path = freeze_initial_target_manifest(directory)
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["seed_policy"]["seed"] += 1
            rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "rows changed"):
                freeze_initial_target_manifest(directory)

    def test_verified_real_loader_decodes_each_modality(self) -> None:
        task_ids = (
            "histopathologic-cancer",
            "jigsaw-toxic",
            "mlsp-2013-birds",
            "nyc-taxi-fare",
            "nomad2018",
        )
        entries = {row.task_id: row for row in load_task_registry().entries}
        for task_id in task_ids:
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                public = root / "public"
                _make_public_task_fixture(task_id, public)
                if task_id == "histopathologic-cancer":
                    from PIL import Image

                    Image.new("RGB", (8, 8), (10, 20, 30)).save(public / "train/a.tif")
                elif task_id == "mlsp-2013-birds":
                    with wave.open(str(public / "essential_data/src_wavs/bird.wav"), "wb") as stream:
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.setframerate(8_000)
                        stream.writeframes(b"\x00\x00" * 64)
                entry = entries[task_id]
                prepared = root / "prepared"
                build_shared_prepared_view(
                    entry, public, prepared, archive_sha256="c" * 64
                )
                dataset = VerifiedPreparedDataset(
                    entry, public, prepared, archive_sha256="c" * 64
                )
                batch = dataset.build_batch([0, 1])
                self.assertEqual(len(dataset), PREPARED_EXAMPLE_COUNT)
                self.assertIsInstance(batch["target"], torch.Tensor)
                self.assertTrue(batch["inputs"])
                if task_id == "nomad2018":
                    self.assertIsNotNone(batch["inputs"]["pyg_batch"])
                    self.assertEqual(batch["inputs"]["pyg_batch"].num_graphs, 2)

    def test_real_loader_rejects_rows_mutated_after_manifest_verification(self) -> None:
        entry = next(
            row for row in load_task_registry().entries if row.task_id == "jigsaw-toxic"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            prepared = root / "prepared"
            _make_public_task_fixture(entry.task_id, public)
            build_shared_prepared_view(
                entry, public, prepared, archive_sha256="d" * 64
            )
            original = load_and_verify_prepared_view

            def verify_then_mutate(*args, **kwargs):
                manifest = original(*args, **kwargs)
                path = prepared / "samples.jsonl"
                rows = path.read_text(encoding="utf-8").splitlines()
                first = json.loads(rows[0])
                first["source_occurrence"] += 1
                rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
                path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                return manifest

            with mock.patch(
                "perfseer_v3.dataset_pack.real_data.load_and_verify_prepared_view",
                side_effect=verify_then_mutate,
            ), self.assertRaisesRegex(RealPreparedDataError, "verified view"):
                VerifiedPreparedDataset(
                    entry, public, prepared, archive_sha256="d" * 64
                )

    def test_all_22_local_gate_fixtures_use_the_verified_real_loader(self) -> None:
        fingerprints = set()
        for adapter in all_task_adapters():
            with self.subTest(task_id=adapter.task_id):
                fingerprint, batch = build_local_real_format_batch(adapter)
                self.assertEqual(len(fingerprint), 64)
                self.assertTrue(batch["inputs"])
                self.assertIsInstance(batch["target"], torch.Tensor)
                fingerprints.add(fingerprint)
        self.assertEqual(len(fingerprints), 22)

    def test_five_epoch_engine_keeps_only_epoch_3_to_5_measurements(self) -> None:
        class FakeTelemetry:
            gpu_uuid = "GPU-fixture"
            hardware_fingerprint = "d" * 64

            def read(self):
                return TelemetryReading(50.0, 25.0, 100.0, 0, (os.getpid(),))

        manifest = build_target_manifest()
        candidate = max(
            (
                row
                for row in manifest.candidates
                if row.task_id == "nyc-taxi-fare"
                and row.execution["mode"] == "eager"
                and row.precision_policy["policy_id"] == "fp32_tf32"
                and row.optimizer["name"] == "adam"
                and row.scheduler["name"] == "none"
            ),
            key=lambda row: row.microbatch_size,
        )
        entry = next(row for row in load_task_registry().entries if row.task_id == candidate.task_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            build_shared_prepared_view(entry, public, prepared, archive_sha256="e" * 64)
            result = run_five_epoch_training(
                candidate,
                entry,
                public_directory=public,
                prepared_directory=prepared,
                archive_sha256="e" * 64,
                telemetry_backend=FakeTelemetry(),
                telemetry_interval_seconds=0.02,
                device="cpu",
                allow_non_a10g_test_device=True,
            )
        self.assertEqual(result.completed_epochs, (1, 2, 3, 4, 5))
        self.assertEqual(tuple(row.epoch for row in result.epoch_measurements), (3, 4, 5))
        self.assertEqual(
            sum(row.examples_seen for row in result.epoch_measurements),
            3 * PREPARED_EXAMPLE_COUNT,
        )
        self.assertFalse(result.compile_completed_before_epoch_1)

    def test_supervisor_builds_record_and_proves_cleanup(self) -> None:
        candidate = build_target_manifest().candidates[0]
        entry = next(row for row in load_task_registry().entries if row.task_id == candidate.task_id)
        expected_batches = math.ceil(entry.expected_train_examples / candidate.microbatch_size)
        expected_steps = math.ceil(expected_batches / candidate.gradient_accumulation_steps)
        measurements = tuple(
            EpochMeasurement(
                version=EPOCH_MEASUREMENT_VERSION,
                epoch=epoch,
                epoch_completed=True,
                epoch_ms=1000.0 + (epoch - 4) * 10.0,
                examples_seen=entry.expected_train_examples,
                batches_seen=expected_batches,
                microsteps=expected_batches,
                optimizer_steps=expected_steps,
                loss_finite=True,
                gradients_finite=True,
                telemetry_complete=True,
                telemetry_samples=(TelemetrySample(0.0, 1.0, 50.0, 20.0, 100.0),),
                peak_torch_reserved_mib=80.0,
                requested_backend_id=candidate.execution["backend_id"],
                observed_backend_id=candidate.execution["backend_id"],
                foreign_process_detected=False,
            )
            for epoch in (3, 4, 5)
        )
        run = FiveEpochRunResult(
            version=A10G_RUN_RESULT_VERSION,
            gpu_uuid="GPU-fixture",
            hardware_sha256="f" * 64,
            requested_backend_id=candidate.execution["backend_id"],
            observed_backend_id=candidate.execution["backend_id"],
            compile_completed_before_epoch_1=candidate.execution["mode"] == "compiled",
            completed_epochs=(1, 2, 3, 4, 5),
            finite_loss_epochs=(1, 2, 3, 4, 5),
            finite_gradient_epochs=(1, 2, 3, 4, 5),
            epoch_measurements=measurements,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            view = build_shared_prepared_view(
                entry, public, root / "prepared", archive_sha256="a" * 64
            )
            record = build_accepted_label_record(
                candidate, entry, view, run, _cleanup(), attempt_index=0
            )
        record.validate_against_configuration(candidate, entry)
        self.assertEqual(record.targets.values[0], 1000.0)

        class CleanProbe:
            physical_index = 0
            gpu_uuid = "GPU-fixture"
            hardware_fingerprint = "f" * 64

            def read(self):
                return TelemetryReading(0.0, 0.0, 100.0, 0, ())

        cleanup = wait_for_gpu_cleanup(
            CleanProbe(),
            TelemetryReading(0.0, 0.0, 100.0, 0, ()),
            stable_dwell_seconds=0.02,
            timeout_seconds=1.0,
            poll_seconds=0.005,
        )
        self.assertTrue(cleanup.passed)

    def test_supervisor_aborts_hard_child_failure_and_cleans_sandboxes(self) -> None:
        candidate = build_target_manifest().candidates[0]
        entry = next(
            row for row in load_task_registry().entries
            if row.task_id == candidate.task_id
        )

        class CleanProbe:
            physical_index = 0
            gpu_uuid = "GPU-fixture"
            hardware_fingerprint = "f" * 64

            def read(self):
                return TelemetryReading(0.0, 0.0, 100.0, 0, ())

        class CrashedProcess:
            pid = 2_000_000_000
            returncode = 77

            @staticmethod
            def wait(timeout=None):
                return 77

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            view = build_shared_prepared_view(
                entry, public, prepared, archive_sha256="a" * 64
            )
            operator_config = root / "operator-kaggle"
            operator_config.mkdir()
            (operator_config / "kaggle.json").write_text(
                '{"username":"operator","key":"secret"}', encoding="utf-8"
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "KAGGLE_CONFIG_DIR": str(operator_config),
                        "KAGGLE_API_TOKEN": "operator-token",
                    },
                    clear=False,
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.subprocess.Popen",
                    return_value=CrashedProcess(),
                ) as popen,
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.os.killpg",
                    side_effect=ProcessLookupError,
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.wait_for_gpu_cleanup",
                    return_value=_cleanup(),
                ),
            ):
                supervisor = AttemptSupervisor(root)
                for _ in range(2):
                    with self.assertRaisesRegex(SupervisorError, "diagnostic envelope"):
                        supervisor.run(
                            candidate,
                            entry,
                            view,
                            public_directory=public,
                            prepared_directory=prepared,
                            archive_sha256="a" * 64,
                            probe=CleanProbe(),
                            attempt_index=0,
                        )
            child_environment = popen.call_args.kwargs["env"]
            isolated_config = Path(child_environment["KAGGLE_CONFIG_DIR"])
            inductor_cache = Path(child_environment["TORCHINDUCTOR_CACHE_DIR"])
            triton_cache = Path(child_environment["TRITON_CACHE_DIR"])
            self.assertNotEqual(isolated_config, operator_config)
            self.assertNotIn("KAGGLE_API_TOKEN", child_environment)
            self.assertNotIn("KAGGLE_USERNAME", child_environment)
            self.assertNotIn("KAGGLE_KEY", child_environment)
            self.assertFalse(isolated_config.exists())
            self.assertTrue(inductor_cache.is_relative_to(root))
            self.assertTrue(triton_cache.is_relative_to(root))
            self.assertFalse(inductor_cache.parent.exists())
            self.assertFalse(triton_cache.parent.exists())
            self.assertTrue((operator_config / "kaggle.json").is_file())
            self.assertFalse((root / "attempts" / "failed").exists())
            logs = tuple((root / "attempts/logs").glob("*.log"))
            self.assertEqual(len(logs), 2)
            self.assertEqual(len({row.name for row in logs}), 2)
            self.assertTrue(all(row.stat().st_mode & 0o777 == 0o600 for row in logs))
            self.assertIsNot(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)

    def test_campaign_environment_lock_rejects_resume_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = lock_campaign_environment(root)
            self.assertEqual(lock_campaign_environment(root), first)
            path = root / "state/campaign_environment.json"
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["environment"]["torch"] = "different"
            atomic_write_json(path, changed)
            with self.assertRaisesRegex(SupervisorError, "environment changed"):
                lock_campaign_environment(root)

            atomic_write_json(path, first)
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["environment"]["packages"]["torchvision"] = "different"
            atomic_write_json(path, changed)
            with self.assertRaisesRegex(SupervisorError, "environment changed"):
                lock_campaign_environment(root)

    def test_supervisor_repairs_only_explicit_oom_child_envelopes(self) -> None:
        candidate = build_target_manifest().candidates[0]
        entry = next(
            row for row in load_task_registry().entries
            if row.task_id == candidate.task_id
        )

        class CleanProbe:
            physical_index = 0
            gpu_uuid = "GPU-fixture"
            hardware_fingerprint = "f" * 64

            def read(self):
                return TelemetryReading(0.0, 0.0, 100.0, 0, ())

        class FinishedProcess:
            pid = 2_000_000_000

            def __init__(self, returncode):
                self.returncode = returncode

            def wait(self, timeout=None):
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            view = build_shared_prepared_view(
                entry, public, prepared, archive_sha256="a" * 64
            )
            requested = {"status": "failed"}

            def fake_popen(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                status = requested["status"]
                payload = (
                    {"failure_stage": "allocator", "reason_code": "cuda_out_of_memory"}
                    if status == "oom"
                    else {
                        "failure_stage": "forward",
                        "reason_code": "builtins.FileNotFoundError",
                    }
                )
                envelope = {
                    "version": WORKER_ENVELOPE_VERSION,
                    "candidate_id": candidate.candidate_id,
                    "status": status,
                    "payload": payload,
                }
                atomic_write_json(
                    output,
                    {**envelope, "envelope_sha256": canonical_sha256(envelope)},
                )
                return FinishedProcess(20 if status == "oom" else 21)

            with (
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.subprocess.Popen",
                    side_effect=fake_popen,
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.os.killpg",
                    side_effect=ProcessLookupError,
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.wait_for_gpu_cleanup",
                    return_value=_cleanup(),
                ),
            ):
                with self.assertRaisesRegex(SupervisorError, "FileNotFoundError"):
                    AttemptSupervisor(root).run(
                        candidate,
                        entry,
                        view,
                        public_directory=public,
                        prepared_directory=prepared,
                        archive_sha256="a" * 64,
                        probe=CleanProbe(),
                        attempt_index=0,
                    )
                self.assertFalse((root / "attempts/failed").exists())
                requested["status"] = "oom"
                record = AttemptSupervisor(root).run(
                    candidate,
                    entry,
                    view,
                    public_directory=public,
                    prepared_directory=prepared,
                    archive_sha256="a" * 64,
                    probe=CleanProbe(),
                    attempt_index=1,
                )
            self.assertEqual(record.status, AttemptStatus.OOM)
            self.assertEqual(record.failure_stage, FailureStage.ALLOCATOR)
            self.assertTrue(
                (root / "attempts/failed" / f"{record.run_id}.json").is_file()
            )

    def test_supervisor_timeout_aborts_without_consuming_replacement(self) -> None:
        candidate = build_target_manifest().candidates[0]
        entry = next(
            row for row in load_task_registry().entries
            if row.task_id == candidate.task_id
        )

        class CleanProbe:
            physical_index = 0
            gpu_uuid = "GPU-fixture"
            hardware_fingerprint = "f" * 64

            def read(self):
                return TelemetryReading(0.0, 0.0, 100.0, 0, ())

        class TimedOutProcess:
            pid = 2_000_000_000
            returncode = -15
            wait_count = 0

            def wait(self, timeout=None):
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired("fixture", timeout)
                return self.returncode

        def kill_process_group(pid, signal_number):
            if signal_number == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            _make_public_task_fixture(entry.task_id, public)
            prepared = root / "prepared"
            view = build_shared_prepared_view(
                entry, public, prepared, archive_sha256="a" * 64
            )
            with (
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.subprocess.Popen",
                    return_value=TimedOutProcess(),
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.os.killpg",
                    side_effect=kill_process_group,
                ),
                mock.patch(
                    "perfseer_v3.dataset_pack.supervisor.wait_for_gpu_cleanup",
                    return_value=_cleanup(),
                ),
            ):
                with self.assertRaisesRegex(SupervisorError, "timed out"):
                    AttemptSupervisor(root, timeout_seconds=1).run(
                        candidate,
                        entry,
                        view,
                        public_directory=public,
                        prepared_directory=prepared,
                        archive_sha256="a" * 64,
                        probe=CleanProbe(),
                        attempt_index=0,
                    )
            self.assertFalse((root / "attempts/failed").exists())
            self.assertEqual(len(tuple((root / "attempts/logs").glob("*.log"))), 1)

    def test_all_22_tasks_materialize_resume_and_cleanup_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"KAGGLE_API_TOKEN": "fixture-only"},
            clear=False,
        ):
            workspace = Path(directory) / "external-workspace"
            client = _FakeKaggleClient()
            preparer = _FakeMleBenchPreparer()
            materializer = TaskMaterializer(
                workspace=workspace,
                repository_root=ROOT,
                kaggle=client,
                preparer=preparer,
            )
            entries = load_task_registry().entries
            for entry in entries:
                with self.subTest(task_id=entry.task_id):
                    materialized = materializer.materialize(entry)
                    self.assertEqual(materialized.state.stage, "view_ready")
                    resumed = materializer.materialize(entry)
                    self.assertEqual(
                        resumed.view_manifest.dataset_fingerprint,
                        materialized.view_manifest.dataset_fingerprint,
                    )
                    # Production cleanup is intentionally unavailable until all
                    # task label records produce a verified completion receipt.
                    with self.assertRaises(TypeError):
                        materializer.cleanup_completed_task(
                            materialized, durable_outputs_sha256="a" * 64
                        )
                    shutil.rmtree(materializer.task_cache)
                    self.assertFalse(materializer.task_cache.exists())
            self.assertEqual(client.download_count, len(entries))
            self.assertEqual(preparer.prepare_count, len(entries))
            self.assertEqual(client.authentication_count, len(entries) * 2)

    def test_interrupted_mutating_preparer_reextracts_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"KAGGLE_API_TOKEN": "fixture-only"}, clear=False
        ):
            materializer = TaskMaterializer(
                workspace=Path(directory) / "external-workspace",
                repository_root=ROOT,
                kaggle=_FakeKaggleClient(),
                preparer=_MutatingOncePreparer(),
            )
            entry = load_task_registry().entries[0]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                materializer.materialize(entry)
            state = json.loads(materializer.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "preparing")
            completed = materializer.materialize(entry)
            self.assertEqual(completed.state.stage, "view_ready")
            self.assertEqual(materializer.preparer.prepare_count, 2)

    def test_receipt_first_recovery_deletes_cache_before_next_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"KAGGLE_API_TOKEN": "fixture-only"}, clear=False
        ):
            materializer = TaskMaterializer(
                workspace=Path(directory) / "external-workspace",
                repository_root=ROOT,
                kaggle=_FakeKaggleClient(),
                preparer=_FakeMleBenchPreparer(),
            )
            first, second = load_task_registry().entries[:2]
            completed = materializer.materialize(first)
            resolutions = (("1" * 64, "2" * 64),)
            record_hashes = ("3" * 64,)
            records_manifest = canonical_sha256(
                {
                    "resolutions": resolutions,
                    "accepted_record_sha256s": record_hashes,
                }
            )
            draft = TaskCompletionReceipt(
                version=TASK_COMPLETION_VERSION,
                task_id=first.task_id,
                target_manifest_sha256=build_target_manifest().sha256,
                dataset_fingerprint=completed.view_manifest.dataset_fingerprint,
                resolutions=resolutions,
                accepted_record_sha256s=record_hashes,
                records_manifest_sha256=records_manifest,
                receipt_sha256="",
            )
            receipt = replace(
                draft,
                receipt_sha256=canonical_sha256(draft.unhashed_payload()),
            )
            receipt_path = (
                materializer.workspace
                / "state/completed_materializations"
                / f"{first.task_id}.json"
            )
            atomic_write_json(receipt_path, receipt.to_dict())
            with mock.patch.object(materializer, "_verify_cleanup_durability") as verify:
                self.assertEqual(
                    materializer.cleanup_recovered_task(receipt), receipt_path
                )
                verify.assert_called_once_with(receipt, task_id=first.task_id)
            self.assertFalse(materializer.task_cache.exists())
            next_task = materializer.materialize(second)
            self.assertEqual(next_task.entry.task_id, second.task_id)

    def test_task_loop_is_frozen_order_and_hash_checked(self) -> None:
        order = ["a", "b"]
        state = TaskLoopState(
            version=TASK_LOOP_STATE_VERSION,
            target_manifest_sha256="a" * 64,
            task_registry_sha256="b" * 64,
            completed_task_ids=(),
            completed_receipt_sha256s=(),
            active_task_id=None,
        )
        state.validate(order)
        state = state.begin("a", order)
        self.assertEqual(state.select_next(order), "a")
        resolutions = (("1" * 64, "2" * 64),)
        record_hashes = ("3" * 64,)
        records_manifest = canonical_sha256(
            {"resolutions": resolutions, "accepted_record_sha256s": record_hashes}
        )
        draft = TaskCompletionReceipt(
            version=TASK_COMPLETION_VERSION,
            task_id="a",
            target_manifest_sha256="a" * 64,
            dataset_fingerprint="b" * 64,
            resolutions=resolutions,
            accepted_record_sha256s=record_hashes,
            records_manifest_sha256=records_manifest,
            receipt_sha256="",
        )
        receipt = replace(
            draft, receipt_sha256=canonical_sha256(draft.unhashed_payload())
        )
        state = state.complete("a", receipt, order)
        self.assertEqual(state.select_next(order), "b")
        with self.assertRaisesRegex(Exception, "next frozen task"):
            state.begin("a", order)
        payload = state.to_dict(order)
        self.assertEqual(TaskLoopState.from_dict(payload, order), state)
        payload["state_sha256"] = "0" * 64
        with self.assertRaisesRegex(Exception, "hash mismatch"):
            TaskLoopState.from_dict(payload, order)

    def test_resume_revalidates_every_completed_receipt_before_returning(self) -> None:
        resolutions = (("1" * 64, "2" * 64),)
        record_hashes = ("3" * 64,)
        records_manifest = canonical_sha256(
            {"resolutions": resolutions, "accepted_record_sha256s": record_hashes}
        )
        draft = TaskCompletionReceipt(
            version=TASK_COMPLETION_VERSION,
            task_id="completed-task",
            target_manifest_sha256="4" * 64,
            dataset_fingerprint="5" * 64,
            resolutions=resolutions,
            accepted_record_sha256s=record_hashes,
            records_manifest_sha256=records_manifest,
            receipt_sha256="",
        )
        receipt = replace(
            draft, receipt_sha256=canonical_sha256(draft.unhashed_payload())
        )
        loop = TaskLoopState(
            version=TASK_LOOP_STATE_VERSION,
            target_manifest_sha256="4" * 64,
            task_registry_sha256="6" * 64,
            completed_task_ids=(receipt.task_id,),
            completed_receipt_sha256s=(receipt.receipt_sha256,),
            active_task_id=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = (
                workspace
                / "state/completed_materializations"
                / f"{receipt.task_id}.json"
            )
            atomic_write_json(path, receipt.to_dict())
            with mock.patch(
                "perfseer_v3.dataset_pack.workflow._verify_recovered_receipt"
            ) as verify:
                _verify_completed_prefix(workspace, loop, object())
                verify.assert_called_once()
            path.unlink()
            with self.assertRaisesRegex(Exception, "missing or invalid"):
                _verify_completed_prefix(workspace, loop, object())


class Phase3ModelAndAdapterTests(unittest.TestCase):
    @staticmethod
    def _mutate_architecture_value(field, value):
        replacements = {
            "activation": "prelu",
            "architecture_specification": "branching_conv_silu_v1",
            "atom_features": 4,
            "capacity_factor": 1.5,
            "clip_samples": 128,
            "components": 4,
            "feature_count": 6,
            "heads": 1,
            "input_resolution": 12,
            "kernel_size": 5,
            "kv_heads": 2,
            "mel_bins": 8,
            "modality": "audio",
            "neighbor_count": 1,
            "neighbor_limit": 4,
            "ngram_buckets": 32,
            "node_count": 6,
            "patch_size": 2,
            "sample_rate": 8_000,
            "sequence_length": 6,
            "source_lineage": "generated_branch_lineage_001",
            "temperature": 3.0,
            "top_k": 2,
            "variant": "b4",
            "width_multiplier": 0.5,
            "depth_multiplier": 1.0,
            "window_size": 1,
        }
        if field in replacements:
            replacement = replacements[field]
            if replacement == value:
                raise AssertionError(f"architecture replacement for {field} is unchanged")
            return replacement
        if type(value) is bool:
            return not value
        if type(value) is int:
            return value + 1
        if type(value) is float:
            return value + 0.5
        if type(value) is str:
            return value + "_mutation"
        if isinstance(value, list):
            return [value[0] + 1, *value[1:]]
        raise AssertionError(f"unsupported architecture value {value!r}")

    @staticmethod
    def _tensor_outputs_differ(left, right) -> bool:
        left_tensors, right_tensors = tensor_outputs(left), tensor_outputs(right)
        if len(left_tensors) != len(right_tensors):
            return True
        return any(
            a.shape != b.shape or not torch.equal(a, b)
            for a, b in zip(left_tensors, right_tensors)
        )

    def test_all_22_adapters_expose_fixture_only_contracts(self) -> None:
        adapters = all_task_adapters()
        self.assertEqual(len(adapters), 22)
        self.assertEqual(len({adapter.task_id for adapter in adapters}), 22)
        fingerprints = set()
        for adapter in adapters:
            prepared = adapter.prepare_dataset()
            self.assertTrue(prepared.fixture_only)
            fingerprints.add(adapter.dataset_fingerprint())
            self.assertEqual(adapter.examples_per_epoch(), 4 if adapter.task_id != "nomad2018" else 2)
            with self.assertRaisesRegex(AdapterError, "local fixtures only"):
                adapter.prepare_dataset(fixture=False)
        self.assertEqual(len(fingerprints), 22)

    def test_real_task_output_widths_match_pinned_training_schemas(self) -> None:
        expected = {
            "dog-breed": 120,
            "ranzcr-clip": 9,
            "leaf-classification": 99,
            "jigsaw-toxic": 6,
            "mlsp-2013-birds": 19,
            "tabular-playground-dec-2021": 7,
        }
        self.assertEqual(
            {task_id: adapter_for_task(task_id).target_width for task_id in expected},
            expected,
        )

    def test_seq2seq_families_preserve_independent_decoder_length(self) -> None:
        adapter = adapter_for_task("text-normalization-english")
        batch = {
            "inputs": {
                "token_ids": torch.arange(14).reshape(2, 7),
                "attention_mask": torch.ones((2, 7), dtype=torch.long),
                "decoder_ids": torch.arange(10).reshape(2, 5),
            },
            "target": torch.arange(10).reshape(2, 5).remainder(
                adapter.target_width
            ),
        }
        registry = load_model_registry()
        for family_id in ("t5_small", "kimi_delta_attention", "gru_rnn_seq2seq"):
            entry = next(
                row for row in registry.entries if row.family_id == family_id
            )
            model = importlib.import_module(entry.factory_id).build_model(
                output_width=adapter.target_width,
                task_kind=adapter.task_kind,
                seed=0,
            )
            output = model(batch["inputs"])
            with self.subTest(family=family_id):
                self.assertEqual(tuple(output.shape), (2, 5, adapter.target_width))
                adapter.validate_output_shape(output, batch["target"])
                self.assertTrue(torch.isfinite(model.compute_loss(output, batch, adapter)))

    def test_all_35_implemented_factory_modules_are_importable_and_bound(self) -> None:
        registry = load_model_registry()
        self.assertEqual(registry.factory_status, "implemented_phase3_factories_v1")
        self.assertEqual(len(registry.entries), 35)
        self.assertEqual(len({entry.source_sha256 for entry in registry.entries}), 35)
        for entry in registry.entries:
            module = importlib.import_module(entry.factory_id)
            self.assertTrue(callable(getattr(module, "build_model", None)))
            self.assertEqual(entry.source_lineage, f"model_family:{entry.family_id}:v1")

    def test_all_109_declared_architecture_fields_are_execution_sensitive(self) -> None:
        registry = load_model_registry()
        field_count = 0
        for entry in registry.entries:
            adapter = adapter_for_task(
                default_adapter_id(entry.family_id, entry.adapter_ids)
            )
            builder = importlib.import_module(entry.factory_id).build_model
            baseline = builder(
                output_width=adapter.target_width,
                task_kind=adapter.task_kind,
                seed=0,
            )
            batch = adapter.build_collator()(adapter.build_train_dataset())
            inputs = adapter.build_model_inputs(batch)
            baseline.eval()
            with torch.no_grad():
                baseline_output = baseline(inputs)
                baseline_loss = baseline.compute_loss(baseline_output, batch, adapter)
            baseline_state_schema = tuple(
                (name, tuple(value.shape))
                for name, value in baseline.state_dict().items()
            )
            self.assertEqual(
                set(baseline.architecture_parameters),
                set(entry.adjustable_architecture_fields),
            )
            for field in entry.adjustable_architecture_fields:
                field_count += 1
                mutated = dict(baseline.architecture_parameters)
                mutated[field] = self._mutate_architecture_value(field, mutated[field])
                try:
                    variant = builder(
                        output_width=adapter.target_width,
                        task_kind=adapter.task_kind,
                        seed=0,
                        architecture_parameters=mutated,
                    )
                except ModelFactoryError:
                    with self.subTest(family=entry.family_id, field=field):
                        self.assertEqual(entry.family_id, "independent_generated")
                        self.assertIn(
                            field,
                            {
                                "source_lineage",
                                "architecture_specification",
                                "modality",
                                "depth",
                                "width",
                            },
                        )
                    continue
                variant.eval()
                with torch.no_grad():
                    variant_output = variant(inputs)
                variant_state_schema = tuple(
                    (name, tuple(value.shape))
                    for name, value in variant.state_dict().items()
                )
                loss_changed = False
                try:
                    with torch.no_grad():
                        variant_loss = variant.compute_loss(
                            variant_output, batch, adapter
                        )
                    loss_changed = not torch.equal(baseline_loss, variant_loss)
                except (RuntimeError, ValueError, ModelFactoryError):
                    loss_changed = True
                with self.subTest(family=entry.family_id, field=field):
                    self.assertNotEqual(
                        baseline.architecture_execution_sha256,
                        variant.architecture_execution_sha256,
                    )
                    self.assertNotEqual(
                        baseline.architecture_field_evidence[field]["value_sha256"],
                        variant.architecture_field_evidence[field]["value_sha256"],
                    )
                    self.assertTrue(
                        variant.architecture_field_evidence[field]["semantic_role"]
                    )
                    self.assertTrue(
                        baseline_state_schema != variant_state_schema
                        or self._tensor_outputs_differ(
                            baseline_output, variant_output
                        )
                        or loss_changed
                        or (entry.family_id == "gcn" and field == "node_count"),
                        # GCN node_count is a per-graph input contract. Its
                        # exact value is enforced against the candidate input
                        # signature rather than mutating an already-built
                        # fixture inside the model.
                        f"{entry.family_id}.{field} did not change backbone, execution, or loss",
                    )
        self.assertEqual(field_count, 109)

    def test_backend_identity_cannot_be_supplied_as_observed_evidence(self) -> None:
        entry = load_model_registry().entries[0]
        adapter = adapter_for_task(default_adapter_id(entry.family_id, entry.adapter_ids))
        model = importlib.import_module(entry.factory_id).build_model(
            output_width=adapter.target_width,
            task_kind=adapter.task_kind,
            seed=0,
        )
        with self.assertRaisesRegex(ModelFactoryError, "execution-derived"):
            golden_validate_model(
                model=model,
                adapter=adapter,
                factory_id=entry.factory_id,
                assertions=entry.faithful_operation_assertions,
                backend_id="nonexistent_spoofed_backend",
                optimization_steps=2,
            )

    def test_stage_window_token_mixing_and_moe_capacity_semantics(self) -> None:
        registry = load_model_registry()

        for family_id, field in (
            ("resnet50", "stage_depths"),
            ("mish_resnet", "stage_depths"),
            ("swin_t", "depths"),
            ("restormer", "depths"),
        ):
            entry = next(row for row in registry.entries if row.family_id == family_id)
            adapter = adapter_for_task(default_adapter_id(family_id, entry.adapter_ids))
            batch = adapter.build_collator()(adapter.build_train_dataset())
            inputs = adapter.build_model_inputs(batch)
            builder = importlib.import_module(entry.factory_id).build_model
            baseline = builder(output_width=adapter.target_width, task_kind=adapter.task_kind)
            outputs = []
            schemas = []
            for distribution in ([2, 1], [1, 2]):
                parameters = dict(baseline.architecture_parameters)
                parameters[field] = distribution
                model = builder(
                    output_width=adapter.target_width,
                    task_kind=adapter.task_kind,
                    seed=0,
                    architecture_parameters=parameters,
                )
                model.eval()
                with torch.no_grad():
                    outputs.append(model(inputs))
                schemas.append(tuple(model.state_dict()))
            with self.subTest(family=family_id, field=field):
                self.assertTrue(
                    schemas[0] != schemas[1]
                    or self._tensor_outputs_differ(outputs[0], outputs[1])
                )

        mixer_entry = next(row for row in registry.entries if row.family_id == "mlp_mixer_s")
        mixer_adapter = adapter_for_task(
            default_adapter_id(mixer_entry.family_id, mixer_entry.adapter_ids)
        )
        mixer = importlib.import_module(mixer_entry.factory_id).build_model(
            output_width=mixer_adapter.target_width,
            task_kind=mixer_adapter.task_kind,
        )
        mixer_batch = mixer_adapter.build_collator()(mixer_adapter.build_train_dataset())
        mixer_tokens = mixer.patch(
            mixer_adapter.build_model_inputs(mixer_batch)["image"]
        ).flatten(2).transpose(1, 2)
        self.assertEqual(mixer.token_in[0].weight.shape[1], mixer_tokens.shape[1])

        swin_entry = next(row for row in registry.entries if row.family_id == "swin_t")
        swin_adapter = adapter_for_task(
            default_adapter_id(swin_entry.family_id, swin_entry.adapter_ids)
        )
        swin = importlib.import_module(swin_entry.factory_id).build_model(
            output_width=swin_adapter.target_width,
            task_kind=swin_adapter.task_kind,
        )
        swin_batch = swin_adapter.build_collator()(swin_adapter.build_train_dataset())
        swin_inputs = swin_adapter.build_model_inputs(swin_batch)
        swin_graph = make_fx(lambda value: swin(value))(swin_inputs)
        attention_lengths = []
        for node in swin_graph.graph.nodes:
            target = str(node.target)
            if "scaled_dot_product" in target:
                query = node.args[0].meta.get("val")
                attention_lengths.append(query.shape[-2])
            elif "_safe_softmax" in target:
                attention = node.meta.get("val")
                self.assertEqual(attention.shape[-2], attention.shape[-1])
                attention_lengths.append(attention.shape[-1])
        self.assertTrue(attention_lengths)
        self.assertTrue(
            all(length <= swin.window_size**2 for length in attention_lengths)
        )

        moe_entry = next(row for row in registry.entries if row.family_id == "switch_moe")
        moe_adapter = adapter_for_task(
            default_adapter_id(moe_entry.family_id, moe_entry.adapter_ids)
        )
        moe_batch = moe_adapter.build_collator()(moe_adapter.build_train_dataset())
        moe_inputs = moe_adapter.build_model_inputs(moe_batch)
        moe_builder = importlib.import_module(moe_entry.factory_id).build_model
        capacity_outputs = []
        for capacity in (0.5, 1.5):
            parameters = {
                "expert_count": 3,
                "top_k": 1,
                "capacity_factor": capacity,
            }
            model = moe_builder(
                output_width=moe_adapter.target_width,
                task_kind=moe_adapter.task_kind,
                seed=0,
                architecture_parameters=parameters,
            )
            model.eval()
            with torch.no_grad():
                capacity_outputs.append(model(moe_inputs))
        self.assertTrue(
            self._tensor_outputs_differ(
                capacity_outputs[0], capacity_outputs[1]
            )
        )

    def test_complete_model_and_adapter_golden_matrix_is_fail_closed(self) -> None:
        audit = run_model_factory_audit(optimization_steps=3)
        self.assertEqual(len(audit.model_results), 35)
        self.assertEqual(len(audit.adapter_results), 22)
        self.assertFalse(audit.training_approved)
        self.assertEqual(audit.observed_hardware_id, "local_cpu_fixture")
        self.assertTrue(all(result.best_loss < result.initial_loss for result in audit.model_results))
        self.assertTrue(all(result.changed_parameter_count > 0 for result in audit.model_results))
        self.assertEqual(
            [
                result.family_id
                for result in audit.model_results
                if result.mixed_precision_status == "unsupported_local_cpu"
            ],
            [],
        )
        self.assertTrue(
            all(result.verified_assertions == result.faithful_operation_assertions for result in audit.model_results)
        )
        self.assertTrue(
            all(
                result.capture_raw_targets
                and result.capture_canonical_operations
                and result.capture_eager_equivalent
                for result in audit.model_results
            )
        )
        self.assertTrue(audit.observed_operation_cells)
        self.assertTrue(
            all(not row["accepted_a10g_measurement"] for row in audit.observed_operation_cells)
        )
        graph_adapter = next(row for row in audit.adapter_results if row.task_id == "nomad2018")
        self.assertTrue(graph_adapter.pyg_batch_verified)
        self.assertEqual(len(audit.specialized_step_results), 2)
        self.assertTrue(
            all(result.final_loss < result.initial_loss for result in audit.specialized_step_results)
        )
        self.assertTrue(all(audit.specialized_contract_checks.values()))
        self.assertEqual(len(audit.sha256), 64)


class Phase4PlanningContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_target_manifest()
        cls.operation_plan = plan_operation_corpus()
        cls.composite_plan = plan_composite_corpus()
        cls.bundle = build_dataset_planning_bundle()
        root = cls.manifest.candidates[0]
        parent = root
        attempts = []
        index = 0
        while parent.microbatch_size > 1:
            attempt = next_oom_repair(
                parent,
                _oom_failure_record(parent),
                repair_index=index,
                reason_code="cuda_out_of_memory",
                root_candidate_id=root.candidate_id,
            )
            attempts.append(attempt)
            parent = attempt.candidate
            index += 1
        cls.repair_root = root
        cls.repair_attempts = tuple(attempts)
        cls.terminal_failure = _oom_failure_record(parent)
        cls.quarantine = quarantine_candidate(
            root,
            cls.terminal_failure,
            cls.repair_attempts,
            failure_stage=FailureStage.ALLOCATOR,
            reason_code="cuda_out_of_memory_at_batch_one",
        )
        cls.replacement = make_quota_replacement(
            root, cls.quarantine, replacement_index=0
        )

    def test_exact_target_manifest_is_deterministic_and_complete(self) -> None:
        summary = self.manifest.to_summary()
        self.assertEqual(summary["candidate_count"], 18_000)
        self.assertEqual(len({row.candidate_id for row in self.manifest.candidates}), 18_000)
        self.assertEqual(summary["generated_lineage_count"], 50)
        self.assertEqual(summary["generated_held_out_lineage_count"], 10)
        self.assertEqual(summary["generated_lineage_maximum"], 65)
        self.assertEqual(sum(summary["family_counts"].values()), 18_000)
        self.assertEqual(sum(summary["modality_counts"].values()), 18_000)
        self.assertEqual(
            self.manifest.sha256,
            build_target_manifest().sha256,
        )
        optimizer_families = {
            name: {
                row.family_id
                for row in self.manifest.candidates
                if row.optimizer["name"] == name
            }
            for name in summary["optimizer_ids"]
        }
        self.assertTrue(all(len(families) >= 2 for families in optimizer_families.values()))

    def test_compatibility_is_fail_closed_and_candidate_rechecks_it(self) -> None:
        row = self.manifest.candidates[0]
        request = CompatibilityRequest(
            family_id=row.family_id,
            modality=row.source_modality,
            architecture_parameters=row.architecture_parameters,
            input_signature=row.input_signature,
            precision_id=row.precision_policy["policy_id"],
            optimizer_id=row.optimizer["name"],
            scheduler_id=row.scheduler["name"],
            execution_mode=row.execution["mode"],
            backend_id=row.execution["backend_id"],
        )
        cases = {
            "unknown_family": (
                replace(request, family_id="not_registered"),
                "family_not_registered",
            ),
            "unsupported_precision": (
                replace(request, precision_id="fp8"),
                "precision_not_supported_on_a10g",
            ),
            "negative_dimension": (
                replace(
                    request,
                    architecture_parameters={"input_resolution": -16},
                ),
                "architecture_value_invalid",
            ),
            "window_divisibility": (
                replace(
                    request,
                    family_id="swin_t",
                    architecture_parameters={
                        "heads": 2,
                        "hidden_size": 8,
                        "input_resolution": 24,
                        "patch_size": 4,
                        "window_size": 4,
                    },
                    input_signature={"resolution": 24},
                ),
                "window_does_not_tile_patch_grid",
            ),
        }
        for name, (changed, reason) in cases.items():
            with self.subTest(name=name):
                self.assertIn(reason, evaluate_compatibility(changed).reason_codes)
        tampered = replace(
            row,
            precision_policy={**row.precision_policy, "policy_id": "fp8"},
        )
        tampered = replace(
            tampered,
            candidate_id=canonical_sha256(tampered.unhashed_payload()),
        )
        with self.assertRaises(CandidatePlanningError):
            tampered.validate()
        undeclared = replace(
            row,
            architecture_parameters={
                **row.architecture_parameters,
                "not_a_declared_field": 1,
            },
        )
        undeclared_request = replace(
            request,
            architecture_parameters=undeclared.architecture_parameters,
        )
        undeclared = replace(
            undeclared,
            compatibility_request_sha256=undeclared_request.sha256,
        )
        invalid_candidates = {
            "undeclared_architecture": undeclared,
            "negative_learning_rate": replace(
                row,
                optimizer={**row.optimizer, "learning_rate": -1.0},
            ),
            "zero_gradient_accumulation": replace(
                row, gradient_accumulation_steps=0
            ),
        }
        fake_specification = {
            **row.coverage_cell_specs[0],
            "operation": "not.a.registered.operation",
        }
        invalid_candidates["fabricated_coverage_operation"] = replace(
            row,
            coverage_cell_specs=(
                fake_specification,
                *row.coverage_cell_specs[1:],
            ),
            coverage_cell_ids=(
                canonical_sha256(fake_specification),
                *row.coverage_cell_ids[1:],
            ),
        )
        for name, candidate in invalid_candidates.items():
            candidate = replace(
                candidate,
                candidate_id=canonical_sha256(candidate.unhashed_payload()),
            )
            with self.subTest(candidate_mutation=name):
                with self.assertRaises(CandidatePlanningError):
                    candidate.validate()
                changed_manifest = replace(
                    self.manifest,
                    candidates=(candidate, *self.manifest.candidates[1:]),
                )
                with self.assertRaises(CandidatePlanningError):
                    changed_manifest.validate()

    def test_full_batch_ladders_and_training_controls_match_policy(self) -> None:
        observed = {tier: set() for tier in BATCH_LADDERS}
        for row in self.manifest.candidates:
            row.batch_plan.validate()
            self.assertEqual(row.batch_plan.full_ladder, BATCH_LADDERS[row.batch_plan.effective_tier])
            self.assertGreaterEqual(row.batch_plan.expected_train_examples // row.microbatch_size, 8)
            observed[row.batch_plan.effective_tier].add(row.microbatch_size)
        self.assertEqual(observed, {tier: set(ladder) for tier, ladder in BATCH_LADDERS.items()})
        for family in ("vit_s16", "swin_t"):
            self.assertEqual(
                {row.batch_plan.base_tier for row in self.manifest.candidates if row.family_id == family},
                {"heavy"},
            )
        efficient = [row for row in self.manifest.candidates if row.family_id == "efficientnet_b0_b4"]
        self.assertEqual(
            {(row.architecture_parameters["variant"], row.batch_plan.base_tier) for row in efficient},
            {("b0", "standard"), ("b4", "heavy")},
        )
        generated = [
            row
            for row in self.manifest.candidates
            if row.family_id == "independent_generated"
        ]
        self.assertEqual(
            {row.batch_plan.base_tier for row in generated},
            {"light", "standard", "heavy"},
        )
        self.assertTrue(
            all(
                len(
                    {
                        row.batch_plan.base_tier
                        for row in generated
                        if row.source_lineage == lineage
                    }
                )
                == 1
                for lineage in {row.source_lineage for row in generated}
            )
        )
        one_edge_axis = _batch_classification(
            "gcn",
            {"hidden_size": 8, "layers": 2},
            {"node_count": 8, "edge_count": 512},
            "bf16",
            True,
            "adamw",
        )
        self.assertEqual(one_edge_axis[:2], ("standard", "standard"))
        two_graph_axes = _batch_classification(
            "gcn",
            {"hidden_size": 8, "layers": 2},
            {"node_count": 128, "edge_count": 512},
            "bf16",
            True,
            "adamw",
        )
        self.assertEqual(two_graph_axes[:2], ("standard", "heavy"))
        self.assertIn("multiple_top_quartile_activation_axes", two_graph_axes[2])
        for family in {row.family_id for row in self.manifest.candidates}:
            for tier, ladder in BATCH_LADDERS.items():
                counts = [
                    sum(
                        row.family_id == family
                        and row.batch_plan.effective_tier == tier
                        and row.microbatch_size == batch
                        for row in self.manifest.candidates
                    )
                    for batch in ladder
                ]
                if any(counts):
                    self.assertLessEqual(max(counts) - min(counts), 1)
        checkpointed = sum(
            row.activation_checkpointing["enabled"]
            for row in self.manifest.candidates
        )
        self.assertEqual(checkpointed, 3_600)
        self.assertEqual(
            {row.gradient_accumulation_steps for row in self.manifest.candidates},
            {1, 2, 4, 8},
        )
        self.assertEqual(
            {row.scheduler["progress"] for row in self.manifest.candidates},
            {0.0, 0.1, 0.5, 0.9, 1.0},
        )

    def test_fifty_generated_source_structures_execute_all_modalities(self) -> None:
        registry = build_generated_lineage_registry()
        self.assertEqual(len(registry.lineages), 50)
        self.assertEqual(len({row.source_sha256 for row in registry.lineages}), 50)
        self.assertEqual(len({row.structure_sha256 for row in registry.lineages}), 50)
        tasks = {
            "vision": "histopathologic-cancer",
            "nlp": "jigsaw-toxic",
            "audio": "mlsp-2013-birds",
            "tabular": "nyc-taxi-fare",
            "graph": "nomad2018",
        }
        module = importlib.import_module(
            "perfseer_v3.dataset_pack.models.independent_generated"
        )
        for lineage in registry.lineages:
            adapter = adapter_for_task(tasks[lineage.modality])
            batch = adapter.build_train_dataset()[0]
            model = module.build_model(
                output_width=adapter.target_width,
                task_kind=adapter.task_kind,
                seed=0,
                architecture_parameters={
                    "source_lineage": lineage.lineage_id,
                    "architecture_specification": lineage.architecture_specification,
                    "modality": lineage.modality,
                    "depth": lineage.depth,
                    "width": lineage.width,
                },
            )
            output = model(batch["inputs"])
            adapter.validate_output_shape(output, batch["target"])
            loss = model.compute_loss(output, batch, adapter)
            loss.backward()
            parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if lineage.topology == "sparse_embedding":
                self.assertTrue(all(parameter.grad.is_sparse for parameter in parameters))
                optimizer = torch.optim.SparseAdam(parameters, lr=1e-2)
            else:
                optimizer = torch.optim.SGD(parameters, lr=1e-2)
            optimizer.step()

    def test_fasttext_sparseadam_route_has_only_real_sparse_trainable_gradients(self) -> None:
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.family_id == "fasttext_embeddingbag"
            and row.optimizer["name"] == "sparse_adam"
        )
        adapter = adapter_for_task(candidate.task_id)
        module = importlib.import_module(candidate.factory_id)
        model = module.build_model(
            output_width=adapter.target_width,
            task_kind=adapter.task_kind,
            architecture_parameters=candidate.architecture_parameters,
        )
        batch = adapter.build_train_dataset()[0]
        loss = model.compute_loss(model(batch["inputs"]), batch, adapter)
        loss.backward()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.grad is not None and parameter.grad.is_sparse for parameter in parameters))
        torch.optim.SparseAdam(parameters, lr=1e-2).step()

    def test_production_bundle_excludes_local_qa_corpora(self) -> None:
        self.bundle.validate()
        self.assertEqual(self.bundle.production_corpus, "end_to_end_18k")
        self.assertTrue(self.bundle.local_qa_corpora_excluded)
        self.assertFalse(hasattr(self.bundle, "operation_manifest"))
        self.assertFalse(hasattr(self.bundle, "composite_manifest"))

    def test_oom_repairs_quarantine_and_replacement_are_immutable(self) -> None:
        self.assertTrue(all(row.action == OOM_REPAIR_ACTION for row in self.repair_attempts))
        chain = (self.repair_root, *(row.candidate for row in self.repair_attempts))
        self.assertEqual(len({row.candidate_id for row in chain}), len(chain))
        self.assertEqual(chain[-1].microbatch_size, 1)
        self.assertTrue(self.quarantine.batch_one_exhausted)
        self.assertNotEqual(
            self.replacement.candidate.candidate_id,
            self.repair_root.candidate_id,
        )
        self.assertEqual(
            self.replacement.candidate.coverage_cell_ids,
            self.repair_root.coverage_cell_ids,
        )
        self.assertEqual(self.replacement.candidate.task_id, self.repair_root.task_id)
        self.assertNotEqual(
            (
                self.replacement.candidate.architecture_parameters,
                self.replacement.candidate.input_signature,
            ),
            (
                self.repair_root.architecture_parameters,
                self.repair_root.input_signature,
            ),
        )
        with self.assertRaisesRegex(RepairError, "batch 1"):
            quarantine_candidate(
                self.repair_root,
                _oom_failure_record(self.repair_attempts[-1].parent_candidate),
                self.repair_attempts[:-1],
                failure_stage=FailureStage.ALLOCATOR,
                reason_code="premature",
            )
        non_oom = replace(
            _oom_failure_record(self.repair_root),
            status=AttemptStatus.FAILED,
            failure_stage=FailureStage.FORWARD,
        )
        with self.assertRaisesRegex(RepairError, "persisted allocator-OOM"):
            next_oom_repair(
                self.repair_root,
                non_oom,
                repair_index=0,
                reason_code="not_an_oom",
            )

    def test_resumable_slot_preserves_complete_oom_chain_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state_path = workspace / "state/slots/root.json"
            slot = _new_slot(self.repair_root)
            expected_attempt_ids = []
            while slot.current_candidate.microbatch_size > 1:
                slot = _advance_failed_slot(
                    workspace,
                    slot,
                    _oom_failure_record(slot.current_candidate),
                    self.repair_root,
                )
                expected_attempt_ids.append(slot.oom_attempts[-1].attempt_id)
                slot = _save_slot(state_path, slot)
                slot = _load_slot(state_path, self.repair_root)
            replacement_slot = _advance_failed_slot(
                workspace,
                slot,
                _oom_failure_record(slot.current_candidate),
                self.repair_root,
            )
            quarantine_path = next(
                (workspace / "attempts/repairs/quarantine").glob("*.json")
            )
            quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
            self.assertEqual(
                quarantine["root_candidate_id"], self.repair_root.candidate_id
            )
            self.assertEqual(
                quarantine["terminal_candidate_id"], slot.current_candidate.candidate_id
            )
            self.assertEqual(quarantine["oom_repair_attempt_ids"], expected_attempt_ids)
            self.assertEqual(replacement_slot.oom_attempts, ())
            self.assertEqual(replacement_slot.repair_index, 0)
            self.assertEqual(replacement_slot.replacement_index, 1)
            self.assertNotEqual(
                replacement_slot.chain_root_candidate.candidate_id,
                self.repair_root.candidate_id,
            )

    def test_fill_to_quota_accepts_one_exact_replacement(self) -> None:
        accepted = [
            row.candidate_id
            for row in self.manifest.candidates
            if row.candidate_id != self.repair_root.candidate_id
        ]
        accepted.append(self.replacement.candidate.candidate_id)
        result = fill_to_quota(
            self.manifest, accepted, (self.replacement,)
        )
        self.assertEqual(len(result.accepted_candidates), 18_000)
        self.assertEqual(result.substitution_ids, (self.replacement.substitution_id,))
        self.assertTrue(result.exact_quota_filled)


class LocalValidationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_target_manifest()
        cls.plan = build_local_validation_plan(cls.manifest)

    def test_all_18k_rows_map_to_pairwise_representatives_without_a10g_claims(self) -> None:
        self.plan.validate(self.manifest)
        self.assertEqual(self.plan.static_row_count, 18_000)
        self.assertEqual(len(self.plan.mappings), 18_000)
        self.assertGreater(len(self.plan.representative_signatures), 0)
        self.assertLess(len(self.plan.representative_signatures), 18_000)
        self.assertFalse(self.plan.accepted_a10g_measurement)
        self.assertEqual(self.plan.validation_hardware_id, "nvidia_rtx_5090_32gb")
        selected = {
            row.signature_id for row in self.plan.representative_signatures
        }
        self.assertTrue(
            all(
                set(row.validation_evidence_signature_ids) <= selected
                for row in self.plan.mappings
            )
        )
        candidates = {row.candidate_id: row for row in self.manifest.candidates}
        self.assertTrue(
            all(
                row.execution_signature_id
                == canonical_sha256(signature_factors(candidates[row.configuration_id]))
                for row in self.plan.mappings
            )
        )
        families = {
            row.factors["family"] for row in self.plan.representative_signatures
        }
        lineages = {
            row.factors["source_route"]
            for row in self.plan.representative_signatures
            if row.factors["family"] == "independent_generated"
        }
        tasks = {
            row.factors["task_adapter"]
            for row in self.plan.representative_signatures
        }
        self.assertEqual(len(families), 35)
        self.assertEqual(len(lineages), 50)
        self.assertEqual(len(tasks), 22)
        signature = self.plan.representative_signatures[0]
        self.assertEqual(validation_signature_from_dict(signature.to_dict()), signature)
        with self.assertRaises(LocalValidationError):
            validation_signature_from_dict({**signature.to_dict(), "unknown": True})

    def test_compiled_sgd_row_has_factor_complete_evidence_not_route_aliasing(self) -> None:
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.execution["mode"] == "compiled" and row.optimizer["name"] == "sgd"
        )
        mapping = next(
            row for row in self.plan.mappings if row.configuration_id == candidate.candidate_id
        )
        selected = {
            row.signature_id: set(row.coverage_tokens)
            for row in self.plan.representative_signatures
        }
        covered = set().union(
            *(selected[value] for value in mapping.validation_evidence_signature_ids)
        )
        factors = signature_factors(candidate)
        execution = factors["execution"]
        optimizer = factors["optimizer"]
        self.assertIn(f"value:execution={execution}", covered)
        self.assertIn(f"value:optimizer={optimizer}", covered)
        self.assertIn(
            f"pair:execution={execution}|optimizer={optimizer}",
            covered,
        )
        self.assertEqual(mapping.execution_signature_id, canonical_sha256(factors))

    def test_short_compiler_harness_runs_eager_and_two_compiled_updates(self) -> None:
        signature = self.plan.representative_signatures[0]
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.candidate_id == signature.representative_candidate_id
        )
        result = validate_candidate_execution(
            candidate,
            signature,
            device="cpu",
            compile_backend="eager",
        )
        result.validate()
        self.assertEqual(result.compiled_update_count, 2)
        self.assertFalse(result.accepted_a10g_measurement)
        self.assertEqual(result.observed_hardware_id, "local_cpu_test_fixture")

    def test_cgcnn_real_batch_retains_required_pyg_batch(self) -> None:
        signature = next(
            row
            for row in self.plan.representative_signatures
            if row.factors["family"] == "cgcnn"
        )
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.candidate_id == signature.representative_candidate_id
        )
        result = validate_candidate_execution(
            candidate,
            signature,
            device="cpu",
            compile_backend="eager",
        )
        result.validate()
        self.assertEqual(result.family_id, "cgcnn")
        self.assertTrue(result.output_shape_verified)

    def test_terminal_scheduler_floor_is_explicit_and_scoped(self) -> None:
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.family_id == "panns_cnn14"
            and row.scheduler["name"] == "polynomial"
            and row.scheduler["progress"] == 1.0
            and row.optimizer["name"] == "sgd"
        )
        adapter = adapter_for_task(candidate.task_id)
        for scheduler_name in DEPLOYMENT_SCHEDULERS:
            expected_ratio = (
                MINIMUM_TERMINAL_LEARNING_RATE_RATIO
                if scheduler_name in TERMINAL_DECAY_SCHEDULERS
                else 0.0
            )
            self.assertEqual(
                scheduler_minimum_lr_ratio(scheduler_name), expected_ratio
            )
            for progress in (0.0, 0.9, 1.0):
                scheduled = replace(
                    candidate,
                    scheduler={
                        "name": scheduler_name,
                        "step_unit": (
                            "none"
                            if scheduler_name == "none"
                            else "epoch"
                            if scheduler_name
                            in {
                                "step",
                                "multi_step",
                                "exponential",
                                "cosine",
                                "cosine_warm_restarts",
                                "reduce_on_plateau",
                            }
                            else "optimizer_step"
                        ),
                        "minimum_lr_ratio": expected_ratio,
                        "progress": progress,
                    },
                )
                model = _build_model(scheduled, adapter, torch.device("cpu"))
                optimizer = _build_optimizer(scheduled, model)
                total_steps = five_epoch_optimizer_steps(
                    scheduled, expected_train_examples=4_096
                )
                scheduler = _build_scheduler(
                    scheduled,
                    optimizer,
                    total_optimizer_steps=total_steps,
                )
                with self.subTest(scheduler=scheduler_name, progress=progress):
                    self.assertEqual(
                        scheduled.scheduler["minimum_lr_ratio"], expected_ratio
                    )
                    if expected_ratio:
                        self.assertTrue(
                            all(
                                group["lr"]
                                >= scheduled.optimizer["learning_rate"]
                                * expected_ratio
                                for group in optimizer.optimizers[0].param_groups
                            )
                        )
                    else:
                        self.assertEqual(scheduler.minimum_learning_rates, ())

    def test_signature_includes_scheduler_progress_and_floor(self) -> None:
        candidate = self.manifest.candidates[0]
        baseline = signature_factors(candidate)
        changed_progress = replace(
            candidate,
            scheduler={**candidate.scheduler, "progress": 0.123},
        )
        changed_floor = replace(
            candidate,
            scheduler={**candidate.scheduler, "minimum_lr_ratio": 0.123},
        )
        progress_factors = signature_factors(changed_progress)
        floor_factors = signature_factors(changed_floor)
        self.assertNotEqual(progress_factors, baseline)
        self.assertEqual(progress_factors["scheduler"], baseline["scheduler"])
        self.assertNotEqual(
            progress_factors["scheduler_progress"], baseline["scheduler_progress"]
        )
        self.assertNotEqual(floor_factors["scheduler"], baseline["scheduler"])

    def test_muon_is_never_assigned_without_matrix_parameters(self) -> None:
        self.assertFalse(
            any(
                row.family_id in MUON_MATRIX_FREE_FAMILIES
                and row.optimizer["name"] == "muon"
                for row in self.manifest.candidates
            )
        )
        for family_id in MUON_MATRIX_FREE_FAMILIES:
            candidate = next(
                row for row in self.manifest.candidates if row.family_id == family_id
            )
            request = CompatibilityRequest(
                family_id=candidate.family_id,
                modality=candidate.source_modality,
                architecture_parameters=candidate.architecture_parameters,
                input_signature=candidate.input_signature,
                precision_id="fp32_tf32",
                optimizer_id="muon",
                scheduler_id="none",
                execution_mode="eager",
                backend_id="cuda_eager",
            )
            with self.subTest(family=family_id):
                self.assertIn(
                    "optimizer_parameter_contract_invalid",
                    evaluate_compatibility(request).reason_codes,
                )

    def test_gcn_node_contract_is_per_graph_and_statically_bound(self) -> None:
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.family_id == "gcn" and row.microbatch_size > 1
        )
        self.assertEqual(
            candidate.architecture_parameters["node_count"],
            candidate.input_signature["node_count"],
        )
        request = CompatibilityRequest(
            family_id=candidate.family_id,
            modality=candidate.source_modality,
            architecture_parameters=candidate.architecture_parameters,
            input_signature={
                **candidate.input_signature,
                "node_count": candidate.input_signature["node_count"] + 1,
            },
            precision_id=str(candidate.precision_policy["policy_id"]),
            optimizer_id=str(candidate.optimizer["name"]),
            scheduler_id=str(candidate.scheduler["name"]),
            execution_mode=str(candidate.execution["mode"]),
            backend_id=str(candidate.execution["backend_id"]),
        )
        self.assertIn(
            "input_signature_invalid",
            evaluate_compatibility(request).reason_codes,
        )

        adapter = adapter_for_task(candidate.task_id)
        _, batch = build_local_real_format_batch(adapter)
        batch = bind_candidate_batch_shape(
            candidate,
            adapter,
            batch,
        )
        model = _build_model(candidate, adapter, torch.device("cpu"))
        output = model(adapter.build_model_inputs(batch))
        self.assertEqual(output.shape[0], candidate.microbatch_size)
        self.assertTrue(
            torch.all(output.abs().sum(dim=1) > 0),
            "GCN must retain every graph in the configured microbatch",
        )

    def test_terminal_scheduler_progress_interactions_are_required(self) -> None:
        for scheduler_name in TERMINAL_DECAY_SCHEDULERS:
            for progress in (0.9, 1.0):
                candidate = next(
                    row
                    for row in self.manifest.candidates
                    if row.scheduler["name"] == scheduler_name
                    and row.scheduler["progress"] == progress
                )
                factors = signature_factors(candidate)
                tokens = _coverage_tokens(factors)
                with self.subTest(scheduler=scheduler_name, progress=progress):
                    self.assertIn(
                        "pair:scheduler="
                        f"{factors['scheduler']}|scheduler_progress={progress}",
                        tokens,
                    )
                    self.assertIn(
                        "pair:optimizer="
                        f"{factors['optimizer']}|scheduler_progress={progress}",
                        tokens,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA compiler regression requires a GPU")
    def test_swin_window_attention_compiles_with_valid_cuda_layout(self) -> None:
        signature = next(
            row
            for row in self.plan.representative_signatures
            if row.factors["family"] == "swin_t"
            and row.factors["task_adapter"] == "ranzcr-clip"
            and row.factors["precision"].startswith("fp16_grad_scaler:")
        )
        candidate = next(
            row
            for row in self.manifest.candidates
            if row.candidate_id == signature.representative_candidate_id
        )
        result = validate_candidate_execution(
            candidate,
            signature,
            device="cuda",
            compile_backend="inductor",
        )
        result.validate()
        self.assertTrue(result.compile_passed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA compiler regression requires a GPU")
    def test_fp16_checkpoint_and_restormer_compile_regressions(self) -> None:
        for family_id, ordinal in (("inception_v3", 1_335), ("restormer", 5_640)):
            candidate = next(
                row
                for row in self.manifest.candidates
                if row.ordinal == ordinal
            )
            self.assertEqual(candidate.family_id, family_id)
            factors = signature_factors(candidate)
            signature = ValidationSignature(
                version=SIGNATURE_VERSION,
                signature_id=canonical_sha256(factors),
                representative_candidate_id=candidate.candidate_id,
                factors=factors,
                coverage_tokens=tuple(sorted(_coverage_tokens(factors))),
                forced_route_representative=False,
            )
            result = validate_candidate_execution(
                candidate,
                signature,
                device="cuda",
                compile_backend="inductor",
            )
            with self.subTest(family=family_id):
                result.validate()
                self.assertTrue(result.gradients_finite)
                self.assertTrue(result.compile_passed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA compiler regression requires a GPU")
    def test_graph_reduction_and_vit_compile_regressions(self) -> None:
        exact_graph_ordinals = (
            ("gcn_large_batch", "gcn", 13_969),
            ("gcn_small_batch", "gcn", 14_069),
            ("generated_graph", "independent_generated", 16_969),
        )
        for case_name, family_id, ordinal in exact_graph_ordinals:
            candidate = next(
                row for row in self.manifest.candidates if row.ordinal == ordinal
            )
            self.assertEqual(candidate.family_id, family_id)
            self.assertEqual(candidate.precision_policy["policy_id"], "bf16")
            factors = signature_factors(candidate)
            signature = ValidationSignature(
                version=SIGNATURE_VERSION,
                signature_id=canonical_sha256(factors),
                representative_candidate_id=candidate.candidate_id,
                factors=factors,
                coverage_tokens=tuple(sorted(_coverage_tokens(factors))),
                forced_route_representative=False,
            )
            result = validate_candidate_execution(
                candidate,
                signature,
                device="cuda",
                compile_backend="inductor",
            )
            with self.subTest(case=case_name):
                result.validate()
                self.assertTrue(result.eager_compiled_equivalent)
                self.assertTrue(result.compile_passed)

        predicates = (
            (
                "vit_s16",
                lambda row: row.factors["precision"].startswith("bf16:"),
            ),
        )
        for family_id, predicate in predicates:
            signature = next(
                row
                for row in self.plan.representative_signatures
                if row.factors["family"] == family_id and predicate(row)
            )
            candidate = next(
                row
                for row in self.manifest.candidates
                if row.candidate_id == signature.representative_candidate_id
            )
            result = validate_candidate_execution(
                candidate,
                signature,
                device="cuda",
                compile_backend="inductor",
            )
            with self.subTest(family=family_id):
                result.validate()
                self.assertTrue(result.eager_compiled_equivalent)
                self.assertTrue(result.compile_passed)
        self.assertEqual(result.compiled_update_count, 2)

class Phase2PlanningContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generators = build_operation_generator_registry()
        cls.operation_plan = plan_operation_corpus(generator_registry=cls.generators)
        cls.blocks = build_composite_block_registry()
        cls.composite_plan = plan_composite_corpus(registry=cls.blocks)

    def test_generator_registry_exactly_implements_declared_mappings(self) -> None:
        support = build_operation_support_contract()
        required = [row for row in support.operations if row.support_state == "required_measured"]
        self.assertEqual(len(self.generators.generators), len(required))
        self.assertEqual(len(self.generators.generators), 179)
        self.assertEqual(
            {row.canonical_operation_id for row in self.generators.generators},
            {row.canonical_id for row in required},
        )
        self.assertEqual(
            {row.generator_id for row in self.generators.generators},
            {row.microbenchmark_generator_ids[0] for row in required},
        )
        self.assertTrue(all(row.golden_test_id for row in self.generators.generators))

    def test_operation_plan_is_constrained_deterministic_and_has_five_shapes(self) -> None:
        plan = self.operation_plan
        self.assertEqual(len(plan.candidates), 8_000)
        self.assertEqual(plan.normal_repetitions, 1)
        self.assertEqual(len({row.candidate_id for row in plan.candidates}), 8_000)
        shapes = {}
        for row in plan.candidates:
            shapes.setdefault(row.generator_id, set()).add(row.shape_regime)
        self.assertTrue(
            all(values == {"tiny", "small", "medium", "large", "boundary"} for values in shapes.values())
        )
        self.assertEqual(
            plan.sha256,
            plan_operation_corpus(generator_registry=self.generators).sha256,
        )
        first = plan.candidates[0]
        with self.assertRaisesRegex(OperationPlanningError, "candidate ID"):
            replace(first, variant_seed=first.variant_seed + 1).validate(
                self.generators.generators[0]
            )

    def test_operation_gap_wave_prioritizes_missing_and_high_error_cells(self) -> None:
        candidates = self.operation_plan.candidates[:40]
        observed = {row.coverage_cell_id for row in candidates[:20]}
        target = candidates[30]
        wave = select_operation_gap_wave(
            candidates,
            observed,
            limit=5,
            error_by_coverage_cell={target.coverage_cell_id: 100.0},
        )
        self.assertEqual(wave[0].coverage_cell_id, target.coverage_cell_id)
        self.assertTrue(all(row.coverage_cell_id not in observed for row in wave))

    def test_operation_registry_plan_and_gap_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(OperationPlanningError, "order or exact mapping"):
            replace(
                self.generators,
                generators=(
                    self.generators.generators[1],
                    self.generators.generators[0],
                    *self.generators.generators[2:],
                ),
            ).validate()
        first = self.operation_plan.candidates[0]
        tampered = replace(first, variant_seed=-1)
        tampered = replace(tampered, candidate_id=canonical_sha256(tampered.unhashed_payload()))
        with self.assertRaisesRegex(OperationPlanningError, "variant seed"):
            tampered.validate(self.generators.generators[0])
        with self.assertRaisesRegex(OperationPlanningError, "order or deterministic mapping"):
            replace(
                self.operation_plan,
                candidates=(
                    self.operation_plan.candidates[1],
                    first,
                    *self.operation_plan.candidates[2:],
                ),
            ).validate(self.generators)
        with self.assertRaisesRegex(OperationPlanningError, "nonnegative numbers"):
            select_operation_gap_wave(
                (first,),
                (),
                limit=1,
                error_by_coverage_cell={first.coverage_cell_id: math.nan},
            )
        with self.assertRaisesRegex(OperationPlanningError, "duplicate IDs"):
            select_operation_gap_wave((first, first), (), limit=1)

    def test_composite_registry_and_plan_cover_three_contexts_and_interactions(self) -> None:
        self.assertEqual(len(self.blocks.blocks), 45)
        self.assertEqual(len(self.composite_plan.candidates), 2_000)
        self.assertEqual(self.composite_plan.normal_repetitions, 1)
        support = build_operation_support_contract()
        blocks = {row.block_id: row for row in self.blocks.blocks}
        for operation in support.operations:
            if operation.support_state != "required_measured":
                continue
            contexts = {
                blocks[block_id].context for block_id in operation.composite_block_ids
            }
            self.assertEqual(
                contexts,
                {"sequential", "residual_or_branch", "saved_activation_or_alias"},
            )
        interactions = {
            value
            for block in self.blocks.blocks
            for value in block.interaction_requirements
        }
        self.assertTrue(
            {
                "branch",
                "join",
                "residual",
                "alias",
                "materialization",
                "saved_activation",
                "liveness",
                "backward",
                "optimizer_interaction",
            }
            <= interactions
        )
        self.assertEqual(
            self.composite_plan.sha256,
            plan_composite_corpus(registry=self.blocks).sha256,
        )

    def test_generated_operation_and_composite_registries_match_live_contracts(self) -> None:
        registry_root = SRC / "perfseer_v3" / "registries"
        operation_payload = yaml.safe_load(
            (registry_root / "operation_benchmark_registry.yaml").read_text(encoding="utf-8")
        )
        composite_payload = yaml.safe_load(
            (registry_root / "composite_block_registry.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(operation_payload, self.generators.to_dict())
        self.assertEqual(composite_payload, self.blocks.to_dict())

    def test_explicit_p1_fixtures_execute_or_retain_a10g_gate(self) -> None:
        results = run_p1_fixture_suite()
        self.assertEqual(len(results), 6)
        local = [row for row in results if row.status == "local_structural_verified"]
        gated = [
            row
            for row in results
            if row.status == "a10g_environment_qualification_required"
        ]
        self.assertEqual(len(local), 5)
        self.assertTrue(all(row.observed_targets and row.reason is None for row in local))
        self.assertEqual([row.fixture_id for row in gated], ["p1:triton_cuda_fused_training"])
        self.assertTrue(gated[0].reason)

    def test_composite_factories_execute_all_three_interaction_contexts(self) -> None:
        for context in ("sequential", "residual_or_branch", "saved_activation_or_alias"):
            block = next(
                row
                for row in self.blocks.blocks
                if row.family == "training" and row.context == context
            )
            candidate = next(
                row
                for row in self.composite_plan.candidates
                if row.block_id == block.block_id and row.dtype == "float32"
            )
            result = verify_composite_block(build_composite_block(block, candidate))
            with self.subTest(context=context):
                self.assertEqual(result.block_id, block.block_id)
                self.assertEqual(
                    result.verified_operation_ids,
                    block.canonical_operation_ids,
                )
                self.assertEqual(result.interaction_requirements, block.interaction_requirements)

    def test_composite_gap_wave_prioritizes_unseen_cells(self) -> None:
        candidates = self.composite_plan.candidates[:20]
        observed = set(candidates[0].coverage_cell_ids)
        wave = select_composite_gap_wave(candidates, observed, limit=3)
        self.assertEqual(len(wave), 3)
        self.assertTrue(
            all(not all(cell in observed for cell in row.coverage_cell_ids) for row in wave)
        )

    def test_composite_registry_plan_and_gap_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(CompositePlanningError, "order or exact mapping"):
            replace(
                self.blocks,
                blocks=(self.blocks.blocks[1], self.blocks.blocks[0], *self.blocks.blocks[2:]),
            ).validate()
        first = self.composite_plan.candidates[0]
        tampered = replace(first, variant_seed="0")
        tampered = replace(tampered, candidate_id=canonical_sha256(tampered.unhashed_payload()))
        with self.assertRaisesRegex(CompositePlanningError, "optimizer/batch"):
            tampered.validate(self.blocks.blocks[0])
        with self.assertRaisesRegex(CompositePlanningError, "order or deterministic mapping"):
            replace(
                self.composite_plan,
                candidates=(
                    self.composite_plan.candidates[1],
                    first,
                    *self.composite_plan.candidates[2:],
                ),
            ).validate(self.blocks)
        with self.assertRaisesRegex(CompositePlanningError, "nonnegative numbers"):
            select_composite_gap_wave(
                (first,),
                (),
                limit=1,
                error_by_coverage_cell={first.coverage_cell_ids[0]: math.nan},
            )
        with self.assertRaisesRegex(CompositePlanningError, "duplicate IDs"):
            select_composite_gap_wave((first, first), (), limit=1)

    @staticmethod
    def _dispatch_request(operation_index: int = 0, *, backend: str = "cpu_eager"):
        generators = build_operation_generator_registry()
        generator = generators.generators[operation_index]
        identity_source = {
            "optimizer_step_annotation": "perfseer_phase_annotation",
            "training_graph_annotation": "perfseer_phase_annotation",
            "capture_observed_python_semantics": "capture_semantic_summary",
        }.get(generator.execution_route, "dispatcher_trace")
        return DispatchRequest(
            version=DISPATCH_REQUEST_VERSION,
            canonical_operation_id=generator.canonical_operation_id,
            raw_target=generator.raw_target,
            aliases=generator.aliases,
            generator_registry_sha256=generators.sha256,
            measurement_scope="local_smoke" if backend == "cpu_eager" else "a10g_measurement",
            requested_backend_id=backend,
            identity_source=identity_source,
            environment_gated=generator.execution_route == "a10g_environment_gated_dispatcher",
            specialized_backend=generator.execution_route == "a10g_environment_gated_dispatcher",
        )

    def test_dispatch_verification_binds_operation_backend_and_fingerprints(self) -> None:
        request = self._dispatch_request()
        evidence = DispatchEvidence(
            version=DISPATCH_EVIDENCE_VERSION,
            request_sha256=request.sha256,
            target_hardware_id="local_cpu_fixture",
            measurement_occurrence_id="local:test_dispatch",
            accepted_measurement=False,
            supported=True,
            unsupported_reason=None,
            capture_workload_sha256="a" * 64,
            profile_workload_sha256="a" * 64,
            identity_source="dispatcher_trace",
            observed_raw_targets=(request.raw_target,),
            observed_backend_ids=("cpu_eager",),
        )
        result = verify_dispatch(request, evidence)
        self.assertEqual(result.status, "verified")
        self.assertFalse(result.accepted_measurement)
        with self.assertRaisesRegex(DispatchVerificationError, "fingerprints do not match"):
            verify_dispatch(
                request,
                replace(evidence, profile_workload_sha256="b" * 64),
            )
        with self.assertRaisesRegex(DispatchVerificationError, "backend.*substituted"):
            verify_dispatch(
                request,
                replace(evidence, observed_backend_ids=("different_backend",)),
            )
        with self.assertRaisesRegex(DispatchVerificationError, "was not observed"):
            verify_dispatch(
                request,
                replace(evidence, observed_raw_targets=("aten::relu",)),
            )
        with self.assertRaisesRegex(DispatchVerificationError, "substituted or ambiguous"):
            verify_dispatch(
                request,
                replace(evidence, observed_backend_ids=("cpu_eager", "fallback")),
            )
        with self.assertRaisesRegex(DispatchVerificationError, "must be a tuple"):
            replace(evidence, observed_backend_ids=["cpu_eager"]).validate_shape()
        with self.assertRaisesRegex(DispatchVerificationError, "local smoke"):
            replace(request, requested_backend_id="arbitrary_backend").validate()
        with self.assertRaisesRegex(DispatchVerificationError, "verification status"):
            replace(result, status="forged").validate()

    def test_environment_gated_dispatch_has_explicit_unsupported_state(self) -> None:
        index = next(
            index
            for index, row in enumerate(self.generators.generators)
            if row.execution_route == "a10g_environment_gated_dispatcher"
        )
        request = self._dispatch_request(index)
        evidence = DispatchEvidence(
            version=DISPATCH_EVIDENCE_VERSION,
            request_sha256=request.sha256,
            target_hardware_id="local_cpu_fixture",
            measurement_occurrence_id="local:test_unsupported",
            accepted_measurement=False,
            supported=False,
            unsupported_reason="CUDA A10G backend qualification is unavailable in local CPU smoke",
            capture_workload_sha256="c" * 64,
            profile_workload_sha256="c" * 64,
            identity_source="dispatcher_trace",
            observed_raw_targets=(),
            observed_backend_ids=(),
        )
        self.assertEqual(verify_dispatch(request, evidence).status, "environment_unsupported")
        with self.assertRaisesRegex(DispatchVerificationError, "gating flags"):
            replace(request, environment_gated=False, specialized_backend=False).validate()
        with self.assertRaisesRegex(DispatchVerificationError, "AWS A10G"):
            replace(
                evidence,
                supported=True,
                unsupported_reason=None,
                accepted_measurement=True,
                observed_raw_targets=(request.raw_target,),
                observed_backend_ids=(request.requested_backend_id,),
            ).validate_shape()

    def test_coverage_report_and_allowlist_reject_fixture_or_partial_evidence(self) -> None:
        empty = build_operation_coverage_report(())
        self.assertFalse(empty["gates_passed"])
        self.assertEqual(empty["measurement_source"], "no_accepted_measurements")
        with self.assertRaisesRegex(OperationCoverageError, "accepted AWS A10G"):
            freeze_runtime_allowlist(empty)

        candidate = self.operation_plan.candidates[0]
        generator_index = next(
            index
            for index, generator in enumerate(self.generators.generators)
            if generator.generator_id == candidate.generator_id
        )
        dispatch_request = self._dispatch_request(generator_index, backend=candidate.backend)
        dispatch_evidence = DispatchEvidence(
            version=DISPATCH_EVIDENCE_VERSION,
            request_sha256=dispatch_request.sha256,
            target_hardware_id="nvidia_a10g_24gb_aws_g5",
            measurement_occurrence_id="a10g:partial:0",
            accepted_measurement=True,
            supported=True,
            unsupported_reason=None,
            capture_workload_sha256="d" * 64,
            profile_workload_sha256="d" * 64,
            identity_source=dispatch_request.identity_source,
            observed_raw_targets=(dispatch_request.raw_target,),
            observed_backend_ids=(dispatch_request.requested_backend_id,),
        )
        with self.assertRaisesRegex(OperationCoverageError, "dispatcher evidence"):
            make_coverage_observation(
                dispatch_request=dispatch_request,
                dispatch_evidence=dispatch_evidence,
                canonical_operation_id=candidate.canonical_operation_id,
                coverage_cell_id=candidate.coverage_cell_id,
                generator_id=candidate.generator_id,
                family=candidate.family,
                shape_regime=candidate.shape_regime,
                dtype=candidate.dtype,
                accumulation_dtype=candidate.accumulation_dtype,
                phase=candidate.phase,
                backend=candidate.backend,
                layout=candidate.layout,
                optimizer_context=candidate.optimizer_context,
                architecture_context=candidate.architecture_context,
                target_hardware_id="local_cpu_fixture",
                accepted=True,
                complete_capture=True,
                strict_capture=True,
                tensor_producing_nodes=1,
                structurally_encoded_nodes=1,
                silently_dropped_tensor_nodes=0,
                capture_profile_fingerprint_match=True,
                gpu_time_us=10.0,
                capture_artifact_sha256="1" * 64,
                generators=self.generators,
            )

        observation = make_coverage_observation(
            dispatch_request=dispatch_request,
            dispatch_evidence=dispatch_evidence,
            canonical_operation_id=candidate.canonical_operation_id,
            coverage_cell_id=candidate.coverage_cell_id,
            generator_id=candidate.generator_id,
            family=candidate.family,
            shape_regime=candidate.shape_regime,
            dtype=candidate.dtype,
            accumulation_dtype=candidate.accumulation_dtype,
            phase=candidate.phase,
            backend=candidate.backend,
            layout=candidate.layout,
            optimizer_context=candidate.optimizer_context,
            architecture_context=candidate.architecture_context,
            target_hardware_id="nvidia_a10g_24gb_aws_g5",
            accepted=True,
            complete_capture=True,
            strict_capture=True,
            tensor_producing_nodes=1,
            structurally_encoded_nodes=1,
            silently_dropped_tensor_nodes=0,
            capture_profile_fingerprint_match=True,
            gpu_time_us=10.0,
            capture_artifact_sha256="1" * 64,
            generators=self.generators,
        )
        partial = build_operation_coverage_report((observation,))
        self.assertFalse(partial["gates_passed"])
        self.assertGreater(len(partial["required_operation_gaps"]), 0)
        with self.assertRaisesRegex(OperationCoverageError, "accepted AWS A10G"):
            freeze_runtime_allowlist(partial)

        replay_cell = canonical_sha256(
            {
                "operation": candidate.canonical_operation_id,
                "shape_regime": "boundary",
                "dtype": candidate.dtype,
                "accumulation_dtype": candidate.accumulation_dtype,
                "phase": candidate.phase,
                "backend": candidate.backend,
                "layout": candidate.layout,
                "optimizer": candidate.optimizer_context,
                "architecture_context": candidate.architecture_context,
            }
        )
        replayed_dispatch_evidence = replace(
            dispatch_evidence,
            measurement_occurrence_id="a10g:partial:rehashed",
        )
        replay = make_coverage_observation(
            dispatch_request=dispatch_request,
            dispatch_evidence=replayed_dispatch_evidence,
            canonical_operation_id=candidate.canonical_operation_id,
            coverage_cell_id=replay_cell,
            generator_id=candidate.generator_id,
            family=candidate.family,
            shape_regime="boundary",
            dtype=candidate.dtype,
            accumulation_dtype=candidate.accumulation_dtype,
            phase=candidate.phase,
            backend=candidate.backend,
            layout=candidate.layout,
            optimizer_context=candidate.optimizer_context,
            architecture_context=candidate.architecture_context,
            target_hardware_id="nvidia_a10g_24gb_aws_g5",
            accepted=True,
            complete_capture=True,
            strict_capture=True,
            tensor_producing_nodes=1,
            structurally_encoded_nodes=1,
            silently_dropped_tensor_nodes=0,
            capture_profile_fingerprint_match=True,
            gpu_time_us=1_000_000_000.0,
            capture_artifact_sha256="1" * 64,
            generators=self.generators,
        )
        replayed_measurement = replace(
            replay.measurement_evidence,
            profiler_artifact_sha256=observation.measurement_evidence.profiler_artifact_sha256,
        )
        replayed_measurement = replace(
            replayed_measurement,
            measurement_id=canonical_sha256(replayed_measurement.unhashed_payload()),
        )
        replayed_observation = replace(replay, measurement_evidence=replayed_measurement)
        replayed_observation = replace(
            replayed_observation,
            observation_id=canonical_sha256(replayed_observation.unhashed_payload()),
        )
        with self.assertRaisesRegex(OperationCoverageError, "content-addressed profiler"):
            replayed_observation.validate(self.generators)

    def test_failed_a10g_capture_counts_against_encoding_and_strict_gates(self) -> None:
        observations = []
        for index, candidate in enumerate(self.operation_plan.candidates[:20]):
            generator_index = next(
                position
                for position, generator in enumerate(self.generators.generators)
                if generator.generator_id == candidate.generator_id
            )
            request = self._dispatch_request(generator_index, backend=candidate.backend)
            evidence = DispatchEvidence(
                version=DISPATCH_EVIDENCE_VERSION,
                request_sha256=request.sha256,
                target_hardware_id="nvidia_a10g_24gb_aws_g5",
                measurement_occurrence_id=f"a10g:denominator:{index}",
                accepted_measurement=True,
                supported=True,
                unsupported_reason=None,
                capture_workload_sha256=f"{index + 1:064x}",
                profile_workload_sha256=f"{index + 1:064x}",
                identity_source=request.identity_source,
                observed_raw_targets=(request.raw_target,),
                observed_backend_ids=(request.requested_backend_id,),
            )
            failed = index == 19
            observations.append(
                make_coverage_observation(
                    dispatch_request=request,
                    dispatch_evidence=evidence,
                    canonical_operation_id=candidate.canonical_operation_id,
                    coverage_cell_id=candidate.coverage_cell_id,
                    generator_id=candidate.generator_id,
                    family=candidate.family,
                    shape_regime=candidate.shape_regime,
                    dtype=candidate.dtype,
                    accumulation_dtype=candidate.accumulation_dtype,
                    phase=candidate.phase,
                    backend=candidate.backend,
                    layout=candidate.layout,
                    optimizer_context=candidate.optimizer_context,
                    architecture_context=candidate.architecture_context,
                    target_hardware_id="nvidia_a10g_24gb_aws_g5",
                    accepted=not failed,
                    complete_capture=not failed,
                    strict_capture=not failed,
                    tensor_producing_nodes=1_000_000_000 if failed else 1,
                    structurally_encoded_nodes=0 if failed else 1,
                    silently_dropped_tensor_nodes=0,
                    capture_profile_fingerprint_match=True,
                    gpu_time_us=0.0 if failed else 1.0,
                    capture_artifact_sha256=f"{index + 101:064x}",
                    generators=self.generators,
                )
            )
        report = build_operation_coverage_report(observations)
        self.assertEqual(report["gate_values"]["strict_complete_capture_rate"], 0.95)
        self.assertLess(
            report["gate_values"]["complete_tensor_node_encoding_rate"],
            0.000001,
        )

    def test_unknown_custom_time_is_structurally_representable(self) -> None:
        observation = make_structural_coverage_observation(
            raw_target="vendor::fused_kernel",
            workload_sha256="e" * 64,
            shape_regime="small",
            dtype="float16",
            accumulation_dtype="float32",
            phase="forward",
            layout="contiguous",
            optimizer_context="none",
            architecture_context="sequential",
            accepted=True,
            complete_capture=True,
            strict_capture=True,
            tensor_producing_nodes=1,
            structurally_encoded_nodes=1,
            silently_dropped_tensor_nodes=0,
            capture_profile_fingerprint_match=True,
            gpu_time_us=5.0,
            capture_artifact_sha256="3" * 64,
        )
        report = build_operation_coverage_report((observation,))
        self.assertEqual(report["gpu_time_us_by_operation"], {"UNK": 5.0})
        self.assertEqual(
            report["gate_values"]["measured_unknown_or_custom_gpu_time_fraction"],
            1.0,
        )

    def test_allowlist_freeze_recomputes_report_from_embedded_evidence(self) -> None:
        malicious = build_operation_coverage_report(())
        malicious["measurement_source"] = "accepted_a10g"
        malicious["training_approved"] = True
        malicious["gates_passed"] = True
        malicious["provisional_exact_vocabulary"] = ["evil::operation"]
        unhashed = dict(malicious)
        unhashed.pop("report_sha256")
        malicious["report_sha256"] = canonical_sha256(unhashed)
        with self.assertRaisesRegex(OperationCoverageError, "does not reproduce"):
            freeze_runtime_allowlist(malicious)


class OperationSupportTests(unittest.TestCase):
    def test_contract_is_registry_derived_complete_and_unapproved(self) -> None:
        registry = OperationRegistry.load()
        contract = build_operation_support_contract(registry=registry)
        self.assertEqual(len(contract.operations), len(registry.rules))
        self.assertEqual(contract.registry_sha256, registry.sha256)
        self.assertFalse(contract.training_approved)
        self.assertEqual(contract.contract_status, "planning")
        self.assertEqual(
            [entry.canonical_id for entry in contract.operations],
            [rule.canonical_id for rule in registry.rules],
        )
        self.assertTrue(all(entry.structurally_encodable for entry in contract.operations))
        self.assertTrue(
            all(
                entry.microbenchmark_generator_ids and entry.golden_test_ids
                for entry in contract.operations
                if entry.support_state == "required_measured"
            )
        )
        self.assertEqual(sum(entry.structural_path == "exact" for entry in contract.operations), 15)
        self.assertEqual(
            sum(entry.structural_path == "family_hash_custom" for entry in contract.operations),
            164,
        )
        first = contract.operations[0]
        tampered_entries = (replace(first, exact_id=999), *contract.operations[1:])
        with self.assertRaisesRegex(OperationSupportError, "aliases/exact ID"):
            replace(contract, operations=tampered_entries).validate(registry)
        tampered_entries = (
            replace(first, required_shape_regimes=("tiny",)),
            *contract.operations[1:],
        )
        with self.assertRaisesRegex(OperationSupportError, "frozen mappings/dimensions"):
            replace(contract, operations=tampered_entries).validate(registry)
        tampered_entries = (
            replace(first, composite_block_ids=(first.composite_block_ids[0],)),
            *contract.operations[1:],
        )
        with self.assertRaisesRegex(OperationSupportError, "frozen mappings/dimensions"):
            replace(contract, operations=tampered_entries).validate(registry)

    def test_empty_families_have_explicit_non_measured_reasons(self) -> None:
        contract = build_operation_support_contract()
        policies = {entry.family: entry for entry in contract.families}
        for family in (
            "unknown_or_custom",
            "sparse_graph",
            "quantized_low_precision",
            "spectral_linalg",
            "custom_fused",
        ):
            self.assertEqual(policies[family].support_state, "structural_only")
            self.assertTrue(policies[family].reason)

    def test_stale_registry_or_policy_fails_closed(self) -> None:
        raw = yaml.safe_load(DEFAULT_SUPPORT_POLICY_PATH.read_text(encoding="utf-8"))
        raw["registry_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OperationSupportError, "stale"):
                build_operation_support_contract(policy_path=path)

        raw = yaml.safe_load(DEFAULT_SUPPORT_POLICY_PATH.read_text(encoding="utf-8"))
        raw["generator_mapping_status"] = "implemented"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(OperationSupportError, "cannot claim implementation"):
                build_operation_support_contract(policy_path=path)

    def test_generated_contract_matches_live_materialization(self) -> None:
        checked_in = json.loads(DEFAULT_GENERATED_CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, build_operation_support_contract().to_dict())

    def test_coverage_config_is_strict_and_bound_to_support_contract(self) -> None:
        config = load_operation_coverage_config()
        contract = build_operation_support_contract()
        self.assertEqual(contract.coverage_config_sha256, config.sha256)
        raw = yaml.safe_load(DEFAULT_COVERAGE_CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = {
            "target": lambda value: value.__setitem__("target_hardware_id", "nvidia_a10"),
            "approval": lambda value: value.__setitem__("training_approved", True),
            "shape": lambda value: value.__setitem__("required_shape_regimes", ["tiny"]),
            "context": lambda value: value.__setitem__(
                "minimum_composite_contexts_when_semantically_possible", 1
            ),
            "gate": lambda value: value["gates"].__setitem__(
                "minimum_exact_vocabulary_gpu_time_fraction", 0.5
            ),
        }
        for name, mutate in mutations.items():
            changed = json.loads(json.dumps(raw))
            mutate(changed)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "coverage.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaises(CoverageConfigError):
                        load_operation_coverage_config(path)

    def test_one_shape_or_context_policy_fails_closed(self) -> None:
        raw = yaml.safe_load(DEFAULT_SUPPORT_POLICY_PATH.read_text(encoding="utf-8"))
        for key in ("required_shape_regimes", "composite_contexts"):
            changed = json.loads(json.dumps(raw))
            changed["defaults"][key] = [changed["defaults"][key][0]]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.subTest(key=key):
                    with self.assertRaises(OperationSupportError):
                        build_operation_support_contract(policy_path=path)


if __name__ == "__main__":
    unittest.main()
