# PerfSeer v3 A10G-to-NVIDIA transfer runbook

## Scope and evidence boundary

This repository implements an A10G-pretrained workload backbone and a
parameter-efficient, small-label adaptation workflow for one concrete target
NVIDIA GPU. It does not implement an unmeasured zero-shot "any NVIDIA GPU"
predictor. Each deployed artifact remains bound to one exact hardware ID and
one hardware-profile hash.

The source baseline is branch `v2` at
`ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95`; it was unchanged when this
refactor began. The frozen A10G configuration identities were not edited.

Production accuracy is intentionally unavailable until all of the following
exist:

- the measured 18,000-row A10G corpus and 54,000 measured epochs;
- a GPU-time-derived operation registry with `training_approved: true`;
- paired target labels on a frozen grouped subset;
- a frozen grouped target test set with both OOM classes where feasible;
- the complete ablation and 128/256/512/1,024-label evidence matrix.

No source-level, synthetic, CPU, or CUDA smoke result substitutes for those
measurements. The evaluator always records
`broad_any_nvidia_claim_authorized: false`.

## Versioned contracts

The transfer refactor uses these independent namespaces:

- graph IR: `perfseer_ir_v3`;
- split feature schema: `perfseer_graph_v3_transfer_v1`;
- output contract: `perfseer_v3_outputs_v3_transfer`;
- workload normalizer: `perfseer_v3_workload_normalization_v2`;
- hardware profile: `perfseer_v3_hardware_profile_v1`;
- hardware normalizer: `perfseer_v3_hardware_normalization_v1`;
- transfer subset: `perfseer_v3_transfer_manifest_v1`;
- target training manifest: `perfseer_v3_target_training_manifest_v1`;
- transfer artifact metadata: `perfseer_v3_transfer_artifact_metadata_v1`.

Generated schemas live in `src/perfseer_v3/schemas/`. Regenerate and validate
them before a run:

```bash
.venv/bin/python scripts/build_v3_schema.py
.venv/bin/python scripts/build_perfseer_v3_transfer_schemas.py
```

The current feature schema SHA-256 is
`eb2c59805c04deeb32785699be387144b419a03764369454a9ac0b4259fe2e8a`.
Any old artifact with the previous combined workload/hardware tensor contract
fails closed.

## Architecture and freeze boundary

The reusable node, edge, workload-global, phase, and SeerBlock path receives
no categorical GPU identity. After graph and phase pooling, a 62-value
hardware block (22 fixed/static/environment values plus 40 microbenchmarks)
and a 62-value missing mask enter `HardwareProfileEncoderV3`.

The default adapter applies identity-initialized FiLM followed by an
identity-initialized low-rank residual. Teacher rank is 32; student rank is 16.
Target adaptation trains only the hardware encoder output, FiLM/low-rank
adapter, separate paired-residual projection, and target calibration. Frozen
parameter bytes are hashed before and after every target run.

The six residual transforms are output-specific: positive metrics use log
ratios and bounded utilization metrics use clipped-logit residuals. A zero
residual decodes exactly to the A10 base prediction. OOM remains a separate
head conditioned directly on predicted peak-live bytes and the decoded
predicted-VRAM/physical-capacity ratio. OOM-only and memory-probe examples train
the OOM, failure-stage, and confidence outputs, but are masked out of exact
paired-regression losses.

Stage A base pretraining learns the hardware-independent workload path. Its
optimizer is released before Stage B fits the complete A10 model. Base and
target stages validate at the configured interval, use an output-appropriate
composite validation score, stop after the configured patience, and restore the
best state before writing an artifact.

## Base model commands

The production gate will reject these commands while the checked-in bootstrap
registry is unapproved. That rejection is deliberate.

```bash
# Build and integrity-check the exact 18K/54K production manifest first.
.venv/bin/python scripts/build_perfseer_v3_base_training_manifest.py \
  --finalized-directory /data/a10/finalized \
  --graphs-directory /data/a10/graphs \
  --operation-coverage-report /data/a10/operation_coverage.json \
  --output /data/a10/perfseer_v3_training_manifest.json

# T0 control
.venv/bin/python scripts/run_perfseer_v3_training.py base_teacher \
  --config src/perfseer_v3/configs/transfer/a10_t0_base_teacher.yaml \
  --manifest /data/a10/perfseer_v3_training_manifest.json \
  --output /artifacts/a10_t0.pt --device cuda --amp bfloat16

# Recommended T1 candidate
.venv/bin/python scripts/run_perfseer_v3_training.py base_teacher \
  --config src/perfseer_v3/configs/transfer/a10_t1_base_teacher.yaml \
  --manifest /data/a10/perfseer_v3_training_manifest.json \
  --output /artifacts/a10_t1.pt --device cuda --amp bfloat16

# S0 control; substitute a10_s1_base_student.yaml for S1
.venv/bin/python scripts/run_perfseer_v3_training.py base_student \
  --config src/perfseer_v3/configs/transfer/a10_s0_base_student.yaml \
  --manifest /data/a10/perfseer_v3_training_manifest.json \
  --teacher-artifact /artifacts/a10_t1.pt \
  --output /artifacts/a10_s0.pt --device cuda --amp bfloat16
```

Select T1 only from grouped held-out significance, and select the smallest
student that passes accuracy, CPU latency, artifact size, memory, and
scheduler gates.

## Target profiling and subset selection

Run the profiler inside a PyTorch CUDA container on the exact target GPU:

```bash
.venv/bin/python scripts/profile_perfseer_v3_target_hardware.py \
  --hardware-id nvidia_h100_sxm_80gb \
  --output /data/target/hardware_profile.json
```

Create a small target request file containing `target_hardware_id` and one of
the four permitted `label_budget` values. Select the 128-label pilot from
final A10 labels and frozen base-teacher embeddings:

```bash
.venv/bin/python scripts/select_perfseer_v3_transfer_subset.py \
  --accepted-labels /data/a10/accepted_labels.jsonl \
  --base-training-manifest /data/a10/perfseer_v3_training_manifest.json \
  --base-teacher-artifact /data/a10/perfseer_v3_teacher.pt \
  --base-student-artifact /data/a10/perfseer_v3_student.pt \
  --embeddings /data/a10/t1_embeddings.npz \
  --target-manifest /data/target/request_128.json \
  --output /data/target/subset_128.json
```

The selector verifies that all 18,000 accepted IDs, grouped splits, graph
signatures, and graph paths match the frozen base training manifest. It also
binds the exact manifest, dataset/split fingerprints, teacher/student artifact
hashes, base-teacher embedding file, and shared workload normalizer into the
hashed subset lineage. It derives
resource, operation-cost, unknown/custom, checkpointing, batch-extremity, and
high-cost-operation strata from those verified graphs and covers exact
optimizer/scheduler pairs before latent diversity; the final labels alone do
not contain every field needed by the coverage contract.

For 256, 512, or 1,024, pass the preceding subset with `--prior-subset`.
After the pilot, also pass `--active-scores`; only the train split consumes
those scores. Validation and test extensions stay deterministic and the test
split never consumes model errors.

## Paired target labeling

The existing fresh-process label worker accepts the frozen subset rather than
duplicating model execution. For each selected candidate JSON:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m perfseer_v3.dataset_pack.label_worker \
  --candidate /data/target/candidates/CONFIG.json \
  --public /data/public --prepared /data/prepared \
  --archive-sha256 ARCHIVE_SHA256 \
  --target-hardware-profile /data/target/hardware_profile.json \
  --transfer-subset /data/target/subset_128.json \
  --base-labels /data/a10/accepted_labels.jsonl \
  --target-graph-output /data/target/graphs/CONFIG.json \
  --output /data/target/attempts/CONFIG.json
```

The worker executes the unchanged five-epoch protocol, retains the exact A10
pair, binds the target profile, and returns accepted or OOM attempt evidence.
Generate the larger-batch power-of-two probe candidates from the frozen
candidate files first:

```bash
.venv/bin/python scripts/build_perfseer_v3_memory_probe_ladders.py \
  --subset /data/target/subset_128.json \
  --candidate-directory /data/a10/final_candidates \
  --maximum-microbatch-size 512 \
  --output /data/target/memory_probe_plan.json
```

Run the exact paired batch first, then each listed candidate in order and stop
after the first OOM. For a changed-batch probe, additionally pass
`--memory-probe`, the original `--base-configuration-id`,
`--original-microbatch-size`, and `--memory-probe-graph` pointing to the
candidate-specific graph produced by the existing capture path. A changed
batch is rejected if it reuses the original graph. Preserve every OOM envelope
with its allocator state as well as a smaller repaired success. Probe/repair
evidence trains OOM and failure-stage outputs, but cannot count as an exact
paired residual label unless that measured configuration has its own frozen
A10 label pair.

Aggregate worker envelopes only after every selected configuration has one
successful target measurement:

```bash
.venv/bin/python scripts/build_perfseer_v3_target_training_manifest.py \
  --subset /data/target/subset_128.json \
  --attempt /data/target/attempts/accepted.jsonl \
  --attempt /data/target/attempts/memory_probes.jsonl \
  --output /data/target/perfseer_v3_target_training_manifest.json
```

The aggregator rejects tampered subsets/envelopes, duplicate pairs, missing
successful labels, split changes, mixed profiles, and missing
target-conditioned graphs. OOM attempts remain outside the successful-label
budget and are retained in the manifest.

## Target teacher and student commands

```bash
.venv/bin/python scripts/run_perfseer_v3_training.py target_teacher_adapter \
  --config src/perfseer_v3/configs/transfer/target_teacher_rank32.yaml \
  --manifest /data/target/perfseer_v3_target_training_manifest.json \
  --base-artifact /artifacts/a10_t1.pt \
  --output /artifacts/target_t1_rank32.pt --device cuda --amp bfloat16

.venv/bin/python scripts/run_perfseer_v3_training.py target_student_adapter \
  --config src/perfseer_v3/configs/transfer/target_student_rank16.yaml \
  --manifest /data/target/perfseer_v3_target_training_manifest.json \
  --base-artifact /artifacts/a10_s1.pt \
  --teacher-artifact /artifacts/target_t1_rank32.pt \
  --output /artifacts/target_s1_rank16.pt --device cuda --amp bfloat16
```

Use `--resume-artifact` to resume a target adapter. Resume is rejected unless
role, base artifact SHA, base/target GPU IDs, subset, profile, workload
normalizer, adapter policy, and rank all match. The target student likewise
rejects a teacher trained on another GPU, subset, or hardware profile.

The adapter+heads and last-SeerBlock configs are disabled by default. Enable
one controlled ablation only after rank-32 adapter underfitting at 512 labels
is measured; do not weaken the default frozen-trunk policy.

The required distinct controls are configured separately under
`src/perfseer_v3/configs/transfer/`:

- `target_teacher_linear_only_control.yaml` trains only the target residual
  projection;
- `target_teacher_film_only_control.yaml` trains hardware encoding plus FiLM;
- `target_teacher_low_rank_only_control.yaml` trains hardware encoding plus the
  low-rank residual;
- `target_teacher_rank32.yaml` is the default combined FiLM/low-rank policy.

## Export, capacity, and evaluation

```bash
.venv/bin/python scripts/export_perfseer_v3_student.py \
  --artifact /artifacts/target_s1_rank16.pt \
  --graph /data/target/graphs/HELD_OUT.json \
  --output /artifacts/target_s1_rank16.torchscript.pt

.venv/bin/python scripts/benchmark_perfseer_v3_capacity.py \
  --benchmark-students \
  --output reports/perfseer_v3_transfer_capacity_study.json

.venv/bin/python scripts/evaluate_perfseer_v3_transfer.py \
  --evidence /data/target/complete_transfer_evidence.json \
  --output reports/perfseer_v3_transfer_evaluation.json
```

The evaluator requires the complete ablation matrix, exactly four FiLM label
budgets, one frozen grouped-test fingerprint, test-only prediction rows,
explicit family/modality/precision/optimizer/scheduler/execution/resource
slices, and all scheduler outcomes. It emits per-slice metrics, OOM recall and
AUROC availability, suggested gates, and refuses a broad hardware claim.

The checked-in local capacity report records exact structural counts:

| Candidate | Parameters | Adapter parameters | Adapter fraction | TorchScript bytes |
|---|---:|---:|---:|---:|
| T0 | 137,187,705 | 220,199 | 0.1605% | not exported |
| T1 | 260,081,601 | 272,167 | 0.1046% | not exported |
| S0 | 1,845,997 | 45,143 | 2.4455% | 7,530,857 |
| S1 | 2,485,993 | 50,615 | 2.0360% | 10,102,137 |

On the recorded one-thread, ten-iteration local CPU smoke, S1 p95 was 1.42,
2.39, and 5.04 ms for small, median, and large graph buckets. These are
implementation measurements, not production acceptance results; the report
leaves production validation, calibration, peak training memory, and matched
v2 comparisons explicitly null.

## Verification and rollback

Local verification:

```bash
.venv/bin/python -m pytest -q tests/test_perfseer_v3_*.py
.venv/bin/python -m compileall -q src scripts tests
uv build
git diff --check
```

CUDA verification uses the finite Kubernetes Job generated by
`scripts/prepare_perfseer_v3_nautilus_verifier.py`. The Job uses a pinned
PyTorch CUDA image, requests one GPU, has an active deadline, and writes its
monitor under `record/`.

Rollback is additive and requires no checkpoint conversion:

1. Stop selecting the transfer artifact in the artifact registry.
2. Restore the last A10-only v3 artifact for A10 workloads, or keep scheduler
   fallback enabled for other hardware.
3. For a full source rollback, deploy branch `v2` at
   `ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95` with its matching v2 artifacts.
4. Never load a transfer-schema state dict into the old feature layout; the
   version/hash checks are designed to reject it.

No rollback step deletes labels, attempts, target profiles, manifests, or
artifacts. Preserve them as immutable evidence for diagnosis or a later
corrected adapter run.
