# PerfSeer V3 Native Nautilus A10 18K Labeler Branch

## Objective

Preserve the completed V100 work and implement a separate native Nautilus A10
labeling workflow on `feature/perfseer-v3-nautilus-a10-18k-labeler`. Reuse the
immutable-image, unified-CLI, digest-only Kubernetes Job, monitoring, and
verification design. Build and test locally only: do not publish an image or
create, modify, or submit any Nautilus resource.

## Frozen reference and native contract

- Freeze source commit `d7abb69b3c79e65e2f3834065ce284484a020ad4`,
  manifest hash `bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e`,
  and task-registry hash
  `781b93ddc020d7cbc77d816458fc213065088a27b9e72d42290ddd8e656de049`.
- Preserve exactly 18,000 rows, 22 tasks, 35 families, and 54,000 retained
  measured epochs, including 5,193 TF32, 5,232 BF16, 4,171 FP16 AMP, and
  3,404 mixed-structured candidates.
- Introduce hardware `nvidia_a10_24gb_nrp` and family
  `nvidia_ampere_a10_24gb_nrp_v1`. Require one visible `NVIDIA A10`, compute
  capability 8.6, and 22--26 GiB of memory.
- Enable TF32 only for TF32 candidates. Run BF16 autocast without scaling and
  FP16 autocast with gradient scaling. Preserve the original mixed-structured
  semantics.
- Keep the V100 implementation intact.

## Crosswalk and execution

- Generate an immutable 18,000-row crosswalk containing ordinal, original A10G
  candidate ID, native NRP-A10 candidate ID, and semantic distribution
  signature.
- Prove a bijection and equality of the task, family, modality, architecture,
  precision, optimizer, scheduler, execution, batch, seed, and coverage
  distributions. Keep the AWS A10G and NRP A10 corpora separate.
- Include the original candidate ID and reference-manifest hash in every native
  result.
- Execute exactly one worker. Use a CephFS-compatible exclusive workspace lock,
  atomic state/results, fail-closed task-cache cleanup, GPU cleanup checks, and
  deterministic OOM repair.
- Select a canonical 96-label pilot covering all 22 tasks, 35 families, four
  precision modes, eager/compiled execution, regimes, checkpoint settings, and
  memory tiers. Resume the same pilot after interruption.
- Preserve pilot records in the production workspace. Complete the remaining
  17,904 labels through 70 ordered chunks of at most 256 new accepted labels;
  reject out-of-order or concurrent chunks and resume an interrupted chunk.

## Container and Nautilus handoff

- Build `linux/amd64` from
  `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`.
- Pin and hash the full 18K runtime, including TorchAudio, TorchVision, PyG,
  Transformers, TensorFlow CPU, Kaggle, audio/archive, NLP, graph, and tabular
  dependencies. Embed clean PerfSeer source and MLE-bench revision
  `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`.
- Perform no Git, apt, pip, or conda operation when a Job starts, and never put
  credentials or datasets in image layers.
- Provide one CLI with `analyze`, `image-preflight`, `smoke-local`,
  `run-campaign --pilot`, `run-campaign --chunk-index N
  --max-new-accepted 256`, and `verify --partial|--complete`.
- Render digest-only one-Pod Jobs requesting and limiting one `NVIDIA-A10`,
  8 CPUs, 32 GiB RAM, and 16 GiB `/dev/shm`, with `backoffLimit: 0`, a
  read-only Kaggle Secret, and a 700 GiB RWX PVC.
- Document all 22 Kaggle download gates, NRP GitLab publication, pilot and 70
  chunks, immediate diagnostics, durable monitoring, and failure handling.

## Verification

- Verify the frozen hashes/counts, native-ID uniqueness, complete crosswalk,
  hardware acceptance/rejection, precision behavior, one-worker enforcement,
  pilot coverage, ordered chunks, interruption resume, exclusive locking, OOM
  repair, atomic writes, and cleanup failure isolation.
- Build the final image and verify pinned imports, `sm_86` plus `sm_120`, source
  and MLE-bench hashes, and absence of credentials.
- On the RTX 5090, run one-batch fixtures for all 35 families and all precision
  modes, plus five real epochs for PANNs/MLSP Birds with TF32 and CGCNN/NOMAD
  with BF16. Mark every RTX result `production_eligible: false`.
- Require actual smallest-file downloads from all 22 Kaggle competitions with
  bounded retry/backoff. If an agreement blocks a test, retain an explicit
  external-blocker record and do not substitute synthetic production evidence.
- Validate pilot, production-chunk, and finalization YAML offline. Actual A10
  concurrency and long-run stability remain future pilot acceptance gates.

## Scope controls

- Commit only concise V100 evidence before branching; keep bulky local datasets
  ignored.
- Keep `backup/pre-v100-labeler-20260811` unchanged.
- Perform no legacy cleanup on this branch.
