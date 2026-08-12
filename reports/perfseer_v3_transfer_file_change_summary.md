# PerfSeer v3 transfer-refactor file-by-file change summary

Date: 2026-08-05  
Baseline: `ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95`

The source plan itself is an input document and is not described as an
implementation edit below. Frozen A10 configuration-identity files were not
changed.

## Documentation and evidence

- `docs/perfseer_v3_encoder_model_design.md` updates the encoder contract to
  describe separated workload and hardware conditioning.
- `docs/perfseer_v3_transfer_learning_runbook.md` provides base training,
  profiling, exact-lineage subset selection, paired labeling, target training,
  evaluation, Nautilus verification, and rollback commands.
- `plan.md` records the implementation phases, verification discipline, and
  external evidence boundary.
- `reports/perfseer_v3_transfer_capacity_study.json` records recomputed T0/T1
  and S0/S1 parameter counts, adapter fractions, student TorchScript sizes,
  local CPU latency, and unavailable production measurements.
- `reports/perfseer_v3_transfer_implementation_status.md` separates completed
  source work from blocked production/CUDA evidence.
- `reports/perfseer_v3_transfer_completion_audit.md` maps every plan section,
  required test, experiment, and deliverable to evidence or an explicit
  blocker.
- `record/perfseer_v3_transfer_local_verification.md` records exact local test,
  schema, build, install, capacity, and prepared-CUDA verification results.
- `record/perfseer_v3_transfer_cuda_verifier_job.yaml` embeds the final source
  overlay in a finite one-GPU PyTorch CUDA ConfigMap+Job manifest.
- `record/monitor_perfseer_v3_transfer_cuda_verifier.sh` records Job/Pod state,
  events, logs, process state, persistent verifier-log tails, and outputs on the
  required cadence after submission.

## Operator scripts

- `scripts/benchmark_perfseer_v3_capacity.py` benchmarks the revised adapter
  candidates and student deployment artifacts with graph-bucket latency.
- `scripts/build_perfseer_v3_base_training_manifest.py` turns frozen production
  A10 outputs into an integrity-checked 18K/54K base-training manifest.
- `scripts/build_perfseer_v3_memory_probe_ladders.py` creates deterministic
  power-of-two target memory-probe plans from a frozen subset.
- `scripts/build_perfseer_v3_target_training_manifest.py` verifies worker
  envelopes and aggregates accepted, OOM, and memory-probe attempts.
- `scripts/build_perfseer_v3_transfer_schemas.py` regenerates the hardware,
  subset, target-manifest, and artifact schemas from source contracts.
- `scripts/evaluate_perfseer_v3_transfer.py` enforces the complete experiment
  matrix, grouped frozen test identity, metric slices, accuracy gates,
  scheduler outcomes, and conservative claims.
- `scripts/prepare_perfseer_v3_nautilus_verifier.py` creates a deterministic
  compressed source overlay and the finite PyTorch CUDA verification Job.
- `scripts/profile_perfseer_v3_target_hardware.py` captures physical identity,
  environment versions, nullable static fields, and 40 CUDA microbenchmarks.
- `scripts/run_perfseer_v3_cuda_verifier.py` checks CUDA execution, adapter
  training, freeze checksums, identity behavior, and the output contract.
- `scripts/run_perfseer_v3_training.py` exposes explicit base teacher/student
  and target teacher/student adapter stages and their required artifacts.
- `scripts/select_perfseer_v3_transfer_subset.py` verifies all 18K base rows and
  exact base artifact/normalizer lineage, derives production strata, and runs
  deterministic latent-diverse or active selection.

## Core feature, hardware, and model contracts

- `src/perfseer_v3/__init__.py` exports the new hardware, transfer, and
  normalization public contracts.
- `src/perfseer_v3/version.py` bumps feature and normalization semantics and
  defines versioned hardware/transfer manifest identifiers.
- `src/perfseer_v3/schema.py` hashes the split workload/hardware layout,
  hardware normalization policy, missing masks, and microbenchmark fields.
- `src/perfseer_v3/features.py` separates workload and hardware tensors,
  preserves train-only workload normalization, applies fixed hardware
  normalization, and reports clipping independently.
- `src/perfseer_v3/hardware.py` defines canonical profile records, static and
  40-point signature fields, environment compatibility values, missing masks,
  physical GPU checks, fixed reference transforms, and profile/policy hashes.
- `src/perfseer_v3/model.py` removes categorical GPU identity from the reusable
  trunk; adds the hardware-profile encoder, identity FiLM/low-rank adapter,
  paired residual head, named trainable groups, and direct peak-live plus
  predicted-VRAM/capacity OOM features.
- `src/perfseer_v3/hardware_transfer.py` defines base and adapted lineage,
  transfer budgets/configs, paired residual transforms, parameter freeze
  checksums, identity/L2-SP regularization, and subset schema generation.
- `src/perfseer_v3/transfer_subset.py` implements deterministic grouped coverage,
  latent farthest-point selection, nested active expansion, frozen-test
  isolation, exact base lineage, and memory-probe allocation.

## Training and labeling

- `src/perfseer_v3/training.py` adds hardware-independent Stage A objectives,
  transformed six-target losses, paired transfer samples, target teacher and
  student steps, relational distillation, and calibration fitting.
- `src/perfseer_v3/training_runner.py` adds production base campaign gates,
  explicit four-stage orchestration, target manifest materialization, early
  stopping/best-state restore, exact base artifact/normalizer binding, resume
  gates, teacher/student lineage checks, and complete run reports.
- `src/perfseer_v3/dataset_pack/a10g_runner.py` retains structured CUDA OOM and
  allocator evidence needed by target transfer labeling.
- `src/perfseer_v3/dataset_pack/label_worker.py` accepts a frozen transfer subset
  and target profile while reusing the existing exact five-epoch execution
  path, including repair/probe envelopes.
- `src/perfseer_v3/dataset_pack/transfer_labeling.py` defines paired target
  attempts, target-conditioned graph materialization, deterministic probe
  ladders, OOM/repair retention, and the target-training manifest/schema.
- `src/perfseer_v3/training_manifest.example.json` demonstrates the revised
  versioned feature and hardware metadata contract.

## Artifacts, runtime, export, and evaluation

- `src/perfseer_v3/artifact.py` adds GPU-specific transfer metadata, exact base
  artifact/subset/profile/normalizer/calibration hashes, policy/rank/count
  checks, and tamper rejection.
- `src/perfseer_v3/runtime.py` reports base, linear-only, or learned-adapter
  provenance while preserving exact-GPU, schema, OOD, confidence, and fallback
  behavior.
- `src/perfseer_v3/deployment_export.py` exports the selected adapted student,
  verifies eager/reloaded TorchScript equality, and reports adapter/artifact
  sizes and graph-bucket CPU latency.
- `src/perfseer_v3/evaluation.py` adds transfer-relevant slice fields used by
  the strict evidence-matrix evaluator.
- `src/perfseer_v3/README.md` documents the refactored architecture, supported
  workflow, and evidence boundary.

## Capacity and training configs

- `src/perfseer_v3/configs/capacity_sweep/capacity_candidates.yaml` adds
  adapter-aware T0/T1 and S0/S1 structural candidates and target variants.
- `src/perfseer_v3/configs/train_hardware_teacher/v3_teacher.yaml` selects the
  recommended T1 base start and adds pretraining, transformed-loss, validation,
  and early-stopping settings.
- `src/perfseer_v3/configs/train_deploy_model/v3_student.yaml` selects the S1
  deployment start and adds complete distillation/validation settings.
- `src/perfseer_v3/configs/transfer/a10_t0_base_teacher.yaml` freezes the T0
  teacher control.
- `src/perfseer_v3/configs/transfer/a10_t1_base_teacher.yaml` defines the T1
  recommended base teacher.
- `src/perfseer_v3/configs/transfer/a10_s0_base_student.yaml` freezes the S0
  student control.
- `src/perfseer_v3/configs/transfer/a10_s1_base_student.yaml` defines the S1
  recommended base student.
- `src/perfseer_v3/configs/transfer/target_teacher_rank32.yaml` defines the
  default rank-32 FiLM plus low-rank target teacher.
- `src/perfseer_v3/configs/transfer/target_student_rank16.yaml` defines the
  default rank-16 target student and adapted-teacher distillation.
- `src/perfseer_v3/configs/transfer/target_teacher_linear_only_control.yaml`
  defines the output-only calibration control.
- `src/perfseer_v3/configs/transfer/target_teacher_film_only_control.yaml`
  defines the FiLM-only control.
- `src/perfseer_v3/configs/transfer/target_teacher_low_rank_only_control.yaml`
  defines the low-rank-only control.
- `src/perfseer_v3/configs/transfer/target_teacher_adapter_heads_ablation.yaml`
  defines the disabled adapter-plus-heads extension.
- `src/perfseer_v3/configs/transfer/target_teacher_last_block_ablation.yaml`
  defines the disabled adapter-plus-last-block extension.

## Generated schemas and registry-derived hashes

- `src/perfseer_v3/schemas/perfseer_graph_v3.json` is regenerated for the new
  feature/hardware contract.
- `src/perfseer_v3/schemas/perfseer_hardware_profile_v1.json` defines profile
  identity, nullable static values, environment, and 40 signature fields.
- `src/perfseer_v3/schemas/perfseer_transfer_subset_manifest_v1.json` defines
  exact base lineage, nested grouped selection, reasons, and memory probes.
- `src/perfseer_v3/schemas/perfseer_target_training_manifest_v1.json` defines
  base lineage, paired successful rows, retained OOMs, and probe evidence.
- `src/perfseer_v3/schemas/perfseer_transfer_artifact_metadata_v1.json` defines
  complete GPU-specific adapter and transfer lineage.
- `src/perfseer_v3/registries/composite_block_registry.yaml` updates the
  registry-declared schema hash after the semantic version bump.
- `src/perfseer_v3/registries/operation_benchmark_registry.yaml` updates the
  registry-declared schema hash after the semantic version bump.
- `src/perfseer_v3/registries/operation_support_a10g.generated.json` regenerates
  the support snapshot against the new registry/schema contract without
  claiming measured training approval.

## Tests

- `tests/test_perfseer_v3_features_coarsen.py` verifies paired workload
  invariance and independent workload/hardware normalization behavior.
- `tests/test_perfseer_v3_model.py` verifies no random hardware identity,
  adapter identity, parameter groups, explicit OOM features, and scriptability.
- `tests/test_perfseer_v3_training.py` verifies Stage A exclusion, campaign
  gates, early stopping, exact target resume/base lineage, and refit rejection.
- `tests/test_perfseer_v3_artifact_runtime.py` verifies every adapter policy,
  artifact tampering, exact-GPU runtime behavior, calibration, and export.
- `tests/test_perfseer_v3_hardware_transfer.py` verifies profile
  canonicalization/hashing, normalization/missingness, target graph identity,
  residuals, freeze checksums, OOM-only learning, policies, and synthetic
  teacher/student transfer.
- `tests/test_perfseer_v3_transfer_subset.py` verifies deterministic nested
  budgets, exact base lineage, all categorical and optimizer/scheduler pair
  coverage, split isolation, active-test freezing, probes, and target manifests.
- `tests/test_perfseer_v3_transfer_evaluation.py` verifies that the full matrix,
  slices, gates, curves, scheduler outcomes, and conservative claim policy are
  mandatory.
