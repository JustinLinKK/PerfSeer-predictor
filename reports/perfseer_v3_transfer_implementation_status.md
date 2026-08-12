# PerfSeer v3 transfer-refactor implementation status

Date: 2026-08-05

## Outcome

The source refactor in
`PerfSeer_v3_A10G_to_NVIDIA_transfer_learning_refactor_plan.md` is implemented
against branch `v2` at audited commit
`ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95`. The frozen A10G configuration
identities and A10-only controls were not changed.

The implementation is locally verified. CUDA execution on Nautilus remains
prepared but not submitted because this environment has neither
`/home/justin/.kube/config` nor a `KUBECONFIG` value. This is a credential gate,
not evidence of successful GPU execution.

## Implemented contracts and behavior

- Workload and hardware features, normalizers, and missingness are independently
  versioned. GPU categorical identity no longer enters the reusable graph trunk.
- A hardware-profile encoder and identity-initialized FiLM plus low-rank adapters
  support rank-32 teacher and rank-16 student target adaptation.
- Target training uses output-specific paired residuals and named parameter
  allowlists; frozen parameter bytes are checked before and after adaptation.
  OOM-only and deliberate memory-probe evidence train classification/stage
  outputs without being miscounted as an exact paired regression label.
- Deterministic grouped 128/256/512/1,024 target subsets preserve frozen test
  isolation. The selector binds all 18,000 IDs, splits, graph signatures, and
  graph paths to the production base manifest before applying global coverage
  quotas, including exact optimizer/scheduler pairs. It also binds the exact
  base manifest, dataset/split fingerprints, teacher and student checkpoints,
  embedding file, and frozen workload normalizer. Active scores affect only the
  training split, and every nested expansion retains that lineage exactly.
- Target profiling supplies 40 CUDA microbenchmarks. Target label envelopes bind
  the exact profile, original A10 pair, target graph, OOM/probe evidence, and
  repair lineage. Memory probes use deterministic power-of-two batch ladders,
  candidate-specific graphs, and retained CUDA allocator evidence.
- Base Stage A pretraining excludes hardware conditioning; Stage B and target
  stages use output-appropriate transformed auxiliary losses, composite
  validation scores, early stopping, and restoration of the best state.
- Transfer controls have distinct `linear_only`, `film_only`, `low_rank_only`,
  and default `film_low_rank` trainable policies; the broader heads and
  last-block ablations remain disabled by default.
- The OOM classifier receives both predicted peak-live memory and the decoded
  predicted-VRAM/physical-capacity ratio directly. Production-complete target
  profiles require core physical fields, all 40 benchmarks, and exact CUDA,
  cuDNN, PyTorch, and driver identities.
- Training, resume, artifact, runtime, export, and evaluation paths fail closed on
  profile, normalizer, hardware, policy, rank, or lineage mismatches.
- The evaluator requires the complete ablation and label-budget matrix and always
  emits `broad_any_nvidia_claim_authorized: false`.

The current semantic feature-schema SHA-256 is
`eb2c59805c04deeb32785699be387144b419a03764369454a9ac0b4259fe2e8a`.

## Verification evidence

The exhaustive aggregate command
`.venv/bin/python -m pytest -q tests/test_perfseer_v3_*.py` completed in
48 minutes 29 seconds with **259 tests passed and 336 parameterized subtests
passed**. This includes the exhaustive 18,000-configuration/54,000-epoch
three-shard finalize, interruption, merge, and verification workflow, plus the
strict transfer audit for OOM-only training, deterministic nested subsets,
memory-probe ladders, freeze policies, early stopping, artifact lineage,
lowercase hash canonicality, and the explicit
A10/A100/L4/L40S/H100/RTX 5090/future-GPU normalization matrix.

Additional checks passed:

- Python compilation for `src`, `scripts`, and `tests`;
- generated registry and JSON-schema metadata validation;
- `git diff --check`;
- source and wheel builds with `uv build`;
- isolated wheel install and import from a temporary target directory, with
  versioned schemas and all new transfer-control configs resolved as package
  data.

The exact command ledger is
`record/perfseer_v3_transfer_local_verification.md`; the requirement audit and
file-level summary are in
`reports/perfseer_v3_transfer_completion_audit.md` and
`reports/perfseer_v3_transfer_file_change_summary.md`.

## Structural capacity evidence

The local structural report is
`reports/perfseer_v3_transfer_capacity_study.json`.

| Candidate | Parameters | Adapter parameters | Adapter fraction | TorchScript bytes |
|---|---:|---:|---:|---:|
| T0 | 137,187,705 | 220,199 | 0.1605% | not exported |
| T1 | 260,081,601 | 272,167 | 0.1046% | not exported |
| S0 | 1,845,997 | 45,143 | 2.4455% | 7,530,857 |
| S1 | 2,485,993 | 50,615 | 2.0360% | 10,102,137 |

On the recorded one-thread, ten-iteration CPU smoke, S1 p95 latency was 1.42,
2.39, and 5.04 ms for small, median, and large graph buckets. Its
content-declared SHA-256 is
`2b6def85705ad3a3800806de7600ef35e4fbc1cfca7cad617fdfc38370214ecd`;
the serialized-file SHA-256 is
`cddeefdc3ef236135290efb02ea9d5f9003afce64c451182d51bbcfcd1e3c896`.
These numbers are local implementation measurements, not production acceptance
evidence.

## Nautilus CUDA gate

The finite verifier is ready at
`record/perfseer_v3_transfer_cuda_verifier_job.yaml`. It uses
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, requests one NVIDIA GPU, and has
a 900-second active deadline. Its 176-file source overlay SHA-256 is
`3494f7d61bf78da111e548a2c45362173b68ed56d566791553fe207a612ccefc`.
Its verifier checks CUDA execution, base identity behavior, one target adapter
step, frozen/trainable checksums, transfer policy, rank, finite loss, and the
output contract.

The required local monitor is ready at
`record/monitor_perfseer_v3_transfer_cuda_verifier.sh`. Once the Job is submitted,
it will write
`record/perfseer_v3_transfer_cuda_verifier_monitor.log` and record Job/Pod status,
events, process status, training-log tail, and output/checkpoint files every 45
seconds for the first five minutes and every 20 minutes thereafter.

No Job has been submitted and no monitor has been started. Cluster authentication
must be provisioned first. Immediately after a future submission, the operator
must collect Job status, Pod status, Pod description, Pod logs, and recent events
before doing other work, then start the monitor and report progress at least once
per minute until the Job is stable or failed.

## Production evidence boundary

The checked-in operation registry remains `training_approved: false`. Production
accuracy, calibration, OOM, scheduler-utility, matched latency, training resource,
and label-budget curves remain unavailable until real A10G and paired target-GPU
labels exist and the complete frozen-test matrix passes. Local CPU, synthetic,
packaging, and CUDA smoke results cannot authorize those claims.

## Rollback

1. Stop selecting a target transfer artifact in the artifact registry.
2. Restore the last A10-only v3 artifact for A10 workloads and retain fail-closed
   scheduler fallback for other hardware.
3. For a full source rollback, deploy branch `v2` at
   `ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95` with matching v2 artifacts.
4. Do not load transfer-schema weights into the previous combined feature layout.

Rollback does not delete profiles, labels, attempts, manifests, checkpoints, or
artifacts; retain them as immutable diagnostic evidence.
