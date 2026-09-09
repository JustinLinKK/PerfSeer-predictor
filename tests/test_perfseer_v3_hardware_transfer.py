from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROFILE_SCRIPT = ROOT / "scripts/profile_perfseer_v3_target_hardware.py"
PROFILE_SPEC = importlib.util.spec_from_file_location(
    "profile_perfseer_v3_target_hardware", PROFILE_SCRIPT
)
if PROFILE_SPEC is None or PROFILE_SPEC.loader is None:
    raise RuntimeError("unable to load target hardware profiler")
PROFILE_MODULE = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = PROFILE_MODULE
PROFILE_SPEC.loader.exec_module(PROFILE_MODULE)

from perfseer_v3.capture_export import CaptureOptions, capture_export
from perfseer_v3.features import build_graph_features, fit_normalization
from perfseer_v3.hardware import (
    HARDWARE_MICROBENCHMARK_FIELDS,
    HardwareNormalizationPolicyV3,
    HardwareProfileV3,
    assert_physical_hardware_identity,
    graph_hardware_id,
)
from perfseer_v3.dataset_pack.a10g_runner import (
    A10G_RUN_RESULT_VERSION,
    FiveEpochRunResult,
)
from perfseer_v3.dataset_pack.contracts import (
    EPOCH_MEASUREMENT_VERSION,
    EpochMeasurement,
    TelemetrySample,
)
from perfseer_v3.dataset_pack.transfer_labeling import (
    TransferLabelAttemptV3,
    aggregate_target_run,
    materialize_target_conditioned_graph,
)
from perfseer_v3.hardware_transfer import (
    BaseTransferLineageV3,
    PairedResidualTransformV3,
    TransferConfigV3,
    adapter_identity_regularization_loss,
    configure_transfer_trainable_parameters,
    parameter_checksum,
)
from perfseer_v3.model import SeerNetV3, SeerNetV3Config
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.training import (
    PairedTransferSampleV3,
    target_student_adapter_step,
    target_teacher_adapter_step,
)
from perfseer_v3.training_runner import (
    TargetTrainingManifestRowV3,
    TargetTrainingManifestV3,
    materialize_target_training_samples,
)


class _Tiny(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ x.transpose(0, 1))


class HardwareTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = capture_export(
            _Tiny(),
            (torch.randn(4, 4),),
            options=CaptureOptions(
                target_hardware_id="nvidia_h100_sxm_80gb",
                hardware_features={
                    "memory_bytes": 80 * 1024**3,
                    "sm_count": 120,
                    "compute_capability": 9.0,
                },
            ),
        )
        if not result.success or result.graph is None:
            raise AssertionError(result.failures)
        cls.features = build_graph_features(result.graph)

    def test_fixed_hardware_normalization_distinguishes_profiles(self) -> None:
        a10 = HardwareProfileV3(
            "nvidia_a10g_24gb_aws_g5",
            {"memory_bytes": 24 * 1024**3, "sm_count": 72, "compute_capability": 8.6},
            {},
            {"cuda_version": "12.8"},
        )
        h100 = HardwareProfileV3(
            "nvidia_h100_sxm_80gb",
            {"memory_bytes": 80 * 1024**3, "sm_count": 120, "compute_capability": 9.0},
            {},
            {"cuda_version": "12.8"},
        )
        policy = HardwareNormalizationPolicyV3()
        a10_values = policy.normalize(a10)
        h100_values = policy.normalize(h100)
        self.assertNotEqual(a10_values.values[:3], h100_values.values[:3])
        self.assertEqual(a10_values.policy_sha256, h100_values.policy_sha256)
        self.assertEqual(len(a10_values.values), len(a10_values.missing_mask))

    def test_supported_nvidia_generations_remain_distinct_after_normalization(self) -> None:
        specifications = (
            ("nvidia_a10g_24gb_aws_g5", 24, 72, 8.6),
            ("nvidia_a100_sxm_80gb", 80, 108, 8.0),
            ("nvidia_l4_24gb", 24, 60, 8.9),
            ("nvidia_l40s_48gb", 48, 142, 8.9),
            ("nvidia_h100_sxm_80gb", 80, 120, 9.0),
            ("nvidia_rtx_5090_32gb", 32, 170, 12.0),
            ("nvidia_future_arch_192gb", 192, 256, 13.0),
        )
        policy = HardwareNormalizationPolicyV3()
        normalized = {
            policy.normalize(
                HardwareProfileV3(
                    hardware_id,
                    {
                        "memory_bytes": memory_gib * 1024**3,
                        "sm_count": sm_count,
                        "compute_capability": compute_capability,
                    },
                    {},
                    {"cuda_version": "13.0"},
                )
            ).values
            for hardware_id, memory_gib, sm_count, compute_capability in specifications
        }
        self.assertEqual(len(normalized), len(specifications))

    def test_hardware_profile_canonicalization_and_hashing(self) -> None:
        first = HardwareProfileV3(
            "NVIDIA H100 SXM 80GB",
            {"sm_count": 120, "memory_bytes": 80 * 1024**3, "compute_capability": 9.0},
            {"pointwise_gbps_small": 1.5, "gemm_fp32_tflops_small": 2.5},
            {"pytorch_version": "2.11.0", "cuda_version": "12.8"},
        )
        second = HardwareProfileV3.from_dict(
            {
                "environment": {"cuda_version": "12.8", "pytorch_version": "2.11.0"},
                "microbenchmarks": {
                    "gemm_fp32_tflops_small": 2.5,
                    "pointwise_gbps_small": 1.5,
                },
                "static": {
                    "compute_capability": 9.0,
                    "memory_bytes": 80 * 1024**3,
                    "sm_count": 120,
                },
                "hardware_id": "nvidia_h100_sxm_80gb",
            }
        )
        self.assertEqual(first.canonical_payload, second.canonical_payload)
        self.assertEqual(first.sha256, second.sha256)
        changed = HardwareProfileV3(
            second.hardware_id,
            second.static,
            second.microbenchmarks,
            {**second.environment, "cuda_version": "13.0"},
        )
        self.assertNotEqual(first.sha256, changed.sha256)

    def test_profiler_preserves_missing_optional_static_fields(self) -> None:
        self.assertIsNone(PROFILE_MODULE._positive_or_none(None))
        self.assertIsNone(PROFILE_MODULE._positive_or_none(0))
        self.assertIsNone(PROFILE_MODULE._positive_or_none(-1))
        self.assertEqual(
            PROFILE_MODULE._positive_or_none(1234, scale=1000.0),
            1_234_000.0,
        )

    def test_complete_profile_requires_core_specs_and_environment_identity(self) -> None:
        benchmarks = {name: 1.0 for name in HARDWARE_MICROBENCHMARK_FIELDS}
        incomplete = HardwareProfileV3(
            "nvidia_h100_sxm_80gb",
            {"memory_bytes": 80 * 1024**3, "sm_count": 120, "compute_capability": 9.0},
            benchmarks,
            {"cuda_version": "12.8", "pytorch_version": "2.11.0"},
        )
        with self.assertRaisesRegex(ValueError, "environment identity"):
            incomplete.validate(require_complete_signature=True)
        complete = HardwareProfileV3(
            incomplete.hardware_id,
            incomplete.static,
            benchmarks,
            {
                "cuda_version": "12.8",
                "cudnn_version": "9.10.2",
                "pytorch_version": "2.11.0+cu130",
                "driver_version": "580.65.06",
            },
        )
        complete.validate(require_complete_signature=True)
        with self.assertRaisesRegex(ValueError, "compute_capability"):
            HardwareProfileV3(
                complete.hardware_id,
                {**complete.static, "compute_capability": 0.0},
                benchmarks,
                complete.environment,
            ).validate()

    def test_declared_target_model_must_match_physical_device(self) -> None:
        for declared, physical in (
            ("nvidia_a10g_24gb_aws_g5", "NVIDIA A10G"),
            ("nvidia_h100_sxm_80gb", "NVIDIA H100 80GB HBM3"),
            ("nvidia_l40s", "NVIDIA L40S"),
            ("nvidia_rtx_5090", "NVIDIA GeForce RTX 5090"),
        ):
            assert_physical_hardware_identity(declared, physical)
        with self.assertRaisesRegex(ValueError, "does not match"):
            assert_physical_hardware_identity(
                "nvidia_h100_sxm_80gb",
                "NVIDIA A100-SXM4-80GB",
            )

    def test_successful_target_run_aggregates_epochs_three_through_five(self) -> None:
        measurements = tuple(
            EpochMeasurement(
                version=EPOCH_MEASUREMENT_VERSION,
                epoch=epoch,
                epoch_completed=True,
                epoch_ms=float(epoch * 10 - 20),
                examples_seen=32,
                batches_seen=4,
                microsteps=4,
                optimizer_steps=1,
                loss_finite=True,
                gradients_finite=True,
                telemetry_complete=True,
                telemetry_samples=(
                    TelemetrySample(
                        timestamp_offset_s=0.0,
                        duration_s=1.0,
                        sm_util_percent=float(epoch * 10 - 20),
                        memory_controller_util_percent=float(epoch * 20 - 40),
                        device_used_vram_mib=float(epoch * 100 - 200),
                    ),
                ),
                peak_torch_reserved_mib=float(epoch * 100 - 150),
                requested_backend_id="eager",
                observed_backend_id="eager",
                foreign_process_detected=False,
            )
            for epoch in (3, 4, 5)
        )
        run = FiveEpochRunResult(
            version=A10G_RUN_RESULT_VERSION,
            gpu_uuid="GPU-test",
            hardware_sha256="a" * 64,
            requested_backend_id="eager",
            observed_backend_id="eager",
            compile_completed_before_epoch_1=True,
            completed_epochs=(1, 2, 3, 4, 5),
            finite_loss_epochs=(1, 2, 3, 4, 5),
            finite_gradient_epochs=(1, 2, 3, 4, 5),
            epoch_measurements=measurements,
        )
        self.assertEqual(
            aggregate_target_run(run),
            (20.0, 20.0, 30.0, 300.0, 350.0, 60.0),
        )

    def test_target_conditioned_graph_preserves_workload_and_paired_signature(self) -> None:
        capture = capture_export(
            _Tiny(),
            (torch.randn(4, 4),),
            options=CaptureOptions(
                target_hardware_id="nvidia_a10g_24gb_aws_g5",
                hardware_features={
                    "memory_bytes": 24 * 1024**3,
                    "sm_count": 72,
                    "compute_capability": 8.6,
                },
            ),
        )
        self.assertTrue(capture.success, capture.failures)
        assert capture.graph is not None
        target_profile = HardwareProfileV3(
            "nvidia_h100_sxm_80gb",
            {"memory_bytes": 80 * 1024**3, "sm_count": 120, "compute_capability": 9.0},
            {name: 1.0 for name in HARDWARE_MICROBENCHMARK_FIELDS},
            {
                "cuda_version": "12.8",
                "cudnn_version": "9.10.2",
                "pytorch_version": torch.__version__,
                "driver_version": "580.65.06",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = capture.graph.save(root / "base.json")
            target_path = materialize_target_conditioned_graph(
                base_path,
                root / "target.json",
                target_profile=target_profile,
                paired_graph_signature=capture.graph.graph_sha256,
            )
            with self.assertRaisesRegex(ValueError, "cannot reuse"):
                materialize_target_conditioned_graph(
                    base_path,
                    root / "invalid_changed_probe.json",
                    target_profile=target_profile,
                    paired_graph_signature=capture.graph.graph_sha256,
                    measured_configuration_id="changed-batch-candidate",
                )
            from perfseer_v3.graph_ir_v3 import GraphIRV3

            target_graph = GraphIRV3.load(target_path)
            self.assertEqual(graph_hardware_id(target_graph.metadata), target_profile.hardware_id)
            self.assertEqual(
                target_graph.metadata["paired_base_graph_signature"],
                capture.graph.graph_sha256,
            )
            base_features = build_graph_features(capture.graph)
            target_features = build_graph_features(target_graph)
            torch.testing.assert_close(base_features.x_cont, target_features.x_cont)
            torch.testing.assert_close(base_features.edge_cont, target_features.edge_cont)
            torch.testing.assert_close(base_features.u_cont, target_features.u_cont)
            self.assertFalse(torch.equal(base_features.hardware_cont, target_features.hardware_cont))

    def test_paired_residual_round_trip_is_stable_at_bounds(self) -> None:
        base = torch.tensor(
            [[100.0, 0.0, 100.0, 4096.0, 4608.0, 0.5], [0.0, 50.0, 99.5, 0.0, 1.0, 100.0]]
        )
        target = torch.tensor(
            [[50.0, 1.0, 99.0, 2048.0, 2304.0, 25.0], [1e-5, 75.0, 1.0, 1e-5, 2.0, 98.0]]
        )
        transform = PairedResidualTransformV3()
        residual = transform.encode(base, target)
        decoded = transform.decode(base, residual)
        self.assertTrue(torch.isfinite(residual).all())
        torch.testing.assert_close(decoded, target, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(
            transform.decode(base, torch.zeros_like(base)),
            base,
            rtol=0.0,
            atol=0.0,
        )

    def test_adapter_step_changes_only_allowlisted_parameters(self) -> None:
        config = SeerNetV3Config.from_registry(
            OperationRegistry.load(),
            self.features.layout,
            hidden=32,
            num_blocks=2,
            adapter_rank=8,
            adapter_policy="film_low_rank",
            dropout=0.0,
        )
        model = SeerNetV3(config)
        audit = configure_transfer_trainable_parameters(model, "film_low_rank")
        frozen_before = parameter_checksum(model, audit["frozen_names"])
        trainable_before = parameter_checksum(model, audit["trainable_names"])
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        sample = PairedTransferSampleV3(
            features=self.features,
            base_target=torch.tensor([100.0, 50.0, 60.0, 4000.0, 4500.0, 40.0]),
            target=torch.tensor([70.0, 65.0, 75.0, 3800.0, 4200.0, 55.0]),
            peak_live_bytes=1024.0,
        )
        loss = target_teacher_adapter_step(model, [sample], optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(frozen_before, parameter_checksum(model, audit["frozen_names"]))
        self.assertNotEqual(trainable_before, parameter_checksum(model, audit["trainable_names"]))

    def test_retained_oom_probe_trains_classification_without_regression(self) -> None:
        capture = capture_export(
            _Tiny(),
            (torch.randn(4, 4),),
            options=CaptureOptions(
                target_hardware_id="nvidia_a10g_24gb_aws_g5",
                hardware_features={
                    "memory_bytes": 24 * 1024**3,
                    "sm_count": 72,
                    "compute_capability": 8.6,
                },
            ),
        )
        self.assertTrue(capture.success, capture.failures)
        assert capture.graph is not None
        target_profile = HardwareProfileV3(
            "nvidia_h100_sxm_80gb",
            {"memory_bytes": 80 * 1024**3, "sm_count": 120, "compute_capability": 9.0},
            {name: 1.0 for name in HARDWARE_MICROBENCHMARK_FIELDS},
            {
                "cuda_version": "12.8",
                "cudnn_version": "9.10.2",
                "pytorch_version": torch.__version__,
                "driver_version": "580.65.06",
            },
        )
        base_targets = (100.0, 40.0, 50.0, 4000.0, 4500.0, 30.0)
        target_targets = (70.0, 60.0, 70.0, 3800.0, 4200.0, 45.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = capture.graph.save(root / "base.json")
            target_path = materialize_target_conditioned_graph(
                base_path,
                root / "target.json",
                target_profile=target_profile,
                paired_graph_signature=capture.graph.graph_sha256,
            )
            normalization = fit_normalization(
                [build_graph_features(capture.graph)],
                split_name="train",
                split_fingerprint="frozen-base-split",
            )
            row = TargetTrainingManifestRowV3(
                sample_id="paired-success",
                base_configuration_id="base-config",
                graph_path=target_path.name,
                split="train",
                source_group="source-group",
                graph_signature=capture.graph.graph_sha256,
                base_target=base_targets,
                target=target_targets,
            )
            probe = TransferLabelAttemptV3(
                base_configuration_id="base-config",
                paired_configuration_id="paired-oom-probe",
                split="memory_probe",
                base_hardware_id="nvidia_a10g_24gb_aws_g5",
                target_hardware_id=target_profile.hardware_id,
                hardware_profile_sha256=target_profile.sha256,
                base_targets=base_targets,
                status="oom",
                failure_stage="allocator",
                target_targets=None,
                original_microbatch_size=8,
                measured_microbatch_size=8,
                repaired_from_configuration_id=None,
                run_payload=None,
                graph_path=target_path.name,
                failure_payload={
                    "exception_type": "OutOfMemoryError",
                    "message": "synthetic OOM",
                    "allocator_state": {"memory_reserved_bytes": 1024},
                },
            )
            manifest = TargetTrainingManifestV3(
                path=root / "target-manifest.json",
                source_subset_sha256="a" * 64,
                base_lineage=BaseTransferLineageV3(
                    base_training_manifest_sha256="1" * 64,
                    base_dataset_fingerprint="2" * 64,
                    base_split_fingerprint="3" * 64,
                    base_teacher_artifact_sha256="4" * 64,
                    base_student_artifact_sha256="5" * 64,
                    base_teacher_embeddings_sha256="6" * 64,
                    workload_normalization_sha256="7" * 64,
                ),
                label_budget=128,
                base_hardware_id="nvidia_a10g_24gb_aws_g5",
                target_hardware_id=target_profile.hardware_id,
                hardware_profile_sha256=target_profile.sha256,
                deployment={},
                rows=(row,),
                oom_attempts=(probe,),
                memory_probe_attempts=(probe,),
                manifest_sha256="b" * 64,
            )
            samples = materialize_target_training_samples(
                manifest,
                registry=OperationRegistry.load(),
                base_normalization=normalization,
            )["train"]
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            [(sample.evidence_kind, sample.regression_available, sample.oom) for sample in samples],
            [("paired_label", True, 0.0), ("memory_probe", False, 1.0)],
        )

        config = SeerNetV3Config.from_registry(
            OperationRegistry.load(),
            samples[0].features.layout,
            hidden=32,
            num_blocks=2,
            adapter_rank=8,
            adapter_policy="film_low_rank",
            dropout=0.0,
        )
        model = SeerNetV3(config)
        audit = configure_transfer_trainable_parameters(model, "film_low_rank")
        trainable_before = parameter_checksum(model, audit["trainable_names"])
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss = target_teacher_adapter_step(model, [samples[1]], optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertNotEqual(
            trainable_before,
            parameter_checksum(model, audit["trainable_names"]),
        )

    def test_identity_regularization_and_last_block_policy_are_separate(self) -> None:
        config = SeerNetV3Config.from_registry(
            OperationRegistry.load(),
            self.features.layout,
            hidden=32,
            num_blocks=2,
            adapter_rank=8,
            adapter_policy="film_low_rank",
            dropout=0.0,
        )
        model = SeerNetV3(config)
        configure_transfer_trainable_parameters(model, "film_low_rank")
        self.assertEqual(
            adapter_identity_regularization_loss(model).detach().item(),
            0.0,
        )
        with torch.no_grad():
            model.hardware_adapter.film.bias[0] = 1.0
        self.assertGreater(
            adapter_identity_regularization_loss(model).detach().item(),
            0.0,
        )

        groups = model.named_parameter_groups()
        self.assertTrue(groups["linear_only"])
        self.assertTrue(
            all(name.startswith("target_residual_head.") for name in groups["linear_only"])
        )
        self.assertTrue(groups["film_only"])
        self.assertTrue(groups["low_rank_only"])
        self.assertFalse(
            any("hardware_adapter.down." in name for name in groups["film_only"])
        )
        self.assertFalse(
            any("hardware_adapter.film." in name for name in groups["low_rank_only"])
        )
        adapter = set(groups["adapter_only"])
        heads_only = set(groups["adapter_heads"]) - adapter
        last_block_only = set(groups["adapter_last_block"]) - adapter
        self.assertTrue(heads_only)
        self.assertTrue(last_block_only)
        self.assertTrue(heads_only.isdisjoint(last_block_only))
        self.assertTrue(
            all(name.startswith("blocks.1.") for name in last_block_only)
        )

    def test_transfer_budget_and_full_backbone_gates(self) -> None:
        config = TransferConfigV3(
            base_hardware_id="nvidia_a10g_24gb_aws_g5",
            target_hardware_id="nvidia_h100_sxm_80gb",
            label_budget=256,
            adapter_policy="film_low_rank",
            adapter_rank=32,
        )
        self.assertEqual(config.split_counts, (192, 32, 32))
        with self.assertRaisesRegex(ValueError, "label budget"):
            TransferConfigV3(
                base_hardware_id="a10",
                target_hardware_id="h100",
                label_budget=200,
                adapter_policy="film_low_rank",
                adapter_rank=32,
            ).validate()

    def test_synthetic_base_to_target_teacher_and_student_flow(self) -> None:
        registry = OperationRegistry.load()
        teacher_config = SeerNetV3Config.from_registry(
            registry,
            self.features.layout,
            hidden=32,
            num_blocks=2,
            adapter_rank=8,
            adapter_policy="film_low_rank",
            dropout=0.0,
        )
        student_config = SeerNetV3Config.from_registry(
            registry,
            self.features.layout,
            hidden=24,
            num_blocks=1,
            adapter_rank=8,
            adapter_policy="film_low_rank",
            dropout=0.0,
        )
        teacher = SeerNetV3(teacher_config)
        student = SeerNetV3(student_config)
        teacher_audit = configure_transfer_trainable_parameters(teacher, "film_low_rank")
        student_audit = configure_transfer_trainable_parameters(student, "film_low_rank")
        teacher_frozen = parameter_checksum(teacher, teacher_audit["frozen_names"])
        student_frozen = parameter_checksum(student, student_audit["frozen_names"])
        sample = PairedTransferSampleV3(
            features=self.features,
            base_target=torch.tensor([100.0, 50.0, 60.0, 4000.0, 4500.0, 40.0]),
            target=torch.tensor([60.0, 70.0, 80.0, 3500.0, 4000.0, 60.0]),
            peak_live_bytes=1024.0,
        )
        teacher_optimizer = torch.optim.AdamW(
            [parameter for parameter in teacher.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        student_optimizer = torch.optim.AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        teacher_loss = target_teacher_adapter_step(
            teacher, [sample], teacher_optimizer
        )
        student_loss = target_student_adapter_step(
            student, teacher, [sample], student_optimizer
        )
        self.assertTrue(torch.isfinite(torch.tensor([teacher_loss, student_loss])).all())
        self.assertEqual(
            teacher_frozen,
            parameter_checksum(teacher, teacher_audit["frozen_names"]),
        )
        self.assertEqual(
            student_frozen,
            parameter_checksum(student, student_audit["frozen_names"]),
        )


if __name__ == "__main__":
    unittest.main()
