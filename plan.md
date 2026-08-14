# PerfSeer V3 Non-Vision Four-A10 Labeler

## Objective

Create `feature/perfseer-v3-nautilus-a10-nonvision-4gpu-labeler` from the
clean Speech V2 branch. Preserve the parent commit as the historical 18K
reference, remove all vision-sourced work from the active campaign, build one
immutable image that runs four independent A10 workers in production and one
RTX 5090 worker during local validation, and test the complete workflow
locally. Do not publish an image or modify Nautilus.

## Corpus and lineage contract

- Exclude all candidates whose `source_modality` is `vision`, including the
  generated-from-vision slice. Keep the teammate's vision corpus separate.
- Freeze exactly 11,200 candidates, 12 Kaggle tasks, 22 model families, and
  33,600 retained measured epochs: audio 1,300; graph 1,300; NLP 5,050;
  tabular 950; and non-vision generated 2,600.
- Preserve precision totals of 3,232 TF32, 3,183 BF16, 2,541 FP16 AMP, and
  2,244 mixed-structured candidates.
- Generate an immutable crosswalk with 11,200 retained Speech V2 rows and
  6,800 excluded rows. Record old/new IDs, semantic signatures, predecessor
  hashes, and exclusion reasons. All retained IDs change under the new batch
  contract.
- Keep historical V1/V2 contracts available through the parent branch and a
  concise active lineage record, but remove vision tasks, families, gates,
  runtime allowlists, and active documentation from this image/profile.

## Batch and repair contract

- Requested effective batch is exactly one of `32, 64, 128, 256, 512`, with
  2,240 candidates assigned to each size.
- Record requested effective batch, runtime microbatch, and gradient
  accumulation separately. Initial microbatch caps are 512 for light, 256 for
  standard, and 64 for heavy families; accumulation preserves effective batch.
- On CUDA OOM, halve microbatch and double accumulation down to microbatch 1.
  Persist every attempt and verify GPU cleanup before continuing.
- If microbatch 1 fails, quarantine the configuration and generate a
  deterministic replacement in the same task, family, precision, execution,
  batch, and coverage cell. Allow no more than three replacements per slot.

## Four-worker execution

- Run one controller and four independent child-process worker slots in one
  Pod. Do not use DDP or NCCL.
- Production requires four unique homogeneous NVIDIA A10 GPUs, compute
  capability 8.6, 22--26 GiB each, and unique UUIDs. Pin one visible GPU to
  each child through `CUDA_VISIBLE_DEVICES`; the child uses logical device 0.
- Materialize and hash-lock each task once, expose prepared data read-only,
  use a CephFS-compatible exclusive campaign lock, and write unique atomic
  state, log, and result files.
- Require worker heartbeats, a ten-minute no-progress timeout, and a two-hour
  absolute attempt timeout. Retry a cleanly recoverable transient timeout or
  process crash once.
- Persist and isolate candidate-local errors so sibling workers and later
  candidates continue. Abort immediately only for global hardware, cleanup,
  credential, dataset, archive, workspace, hash, or atomic-state failures.
- Drain schedulable work before reporting failure. If quota remains incomplete,
  write a partial receipt and failure ledger, then exit nonzero.

## Image, CLI, and downloadable artifacts

- Build a pinned `linux/amd64` PyTorch 2.10/CUDA 12.8 image supporting
  `sm_86` and `sm_120`, the 22 non-vision families, embedded clean source, and
  pinned MLE-bench. Perform no Git, apt, pip, conda, or dataset installation at
  startup, and never put credentials or datasets in image layers.
- Provide `analyze`, `image-preflight`, `smoke-local`, `run-campaign`, `verify`,
  and `export` interfaces. Production supports the pilot and ordered 256-label
  chunks; local validation uses one RTX 5090 and a separate non-production
  workspace.
- Export no trained weights or checkpoints. Produce a verified `.tar.zst`
  containing labels, exact per-candidate training configurations,
  content-addressed model/runtime source, a candidate-to-artifact index,
  failure/quarantine ledgers, manifests, receipts, and SHA-256 sums.
- Verify offline that every exported representative configuration reconstructs
  its model from the bundled source.

## Local and offline acceptance

- Verify corpus counts, distributions, lineage, batch totals, four-worker
  assignment, hardware rejection, locking, atomic resume, OOM descent,
  timeout handling, failure isolation, and incomplete final status.
- Build the final image and verify dependencies, embedded hashes, CUDA
  architectures, read-only startup, credential absence, and absence of active
  vision tasks/families.
- Audit model construction across all 11,200 candidates and run boundary
  one-step tests across every family, precision, execution, batch, and
  generated-model structural path.
- Run 32 real five-epoch labels inside the final image on the local RTX 5090,
  sequentially with one worker: audio 5, graph 5, tabular 5, NLP 11, generated
  6. The frozen selection covers all 22 families, all 12 tasks, all precision
  and execution modes, all regimes, checkpointing on/off, and at least five
  candidates at each requested effective batch. Retain epochs 3--5 and mark
  every record `production_eligible: false`.
- Inject OOM, timeout, deterministic child error, and crash cases and prove the
  queue continues. Rebuild after source changes and repeat final preflight,
  real workflow, export reconstruction, secret scan, and offline Kubernetes
  validation.
- Require real access to all 12 Kaggle competitions. If an agreement or archive
  blocks a dataset, preserve the implementation and record the external
  blocker; never substitute a mirror or synthetic production evidence.

## Nautilus handoff

- Render digest-only Jobs requesting four `nvidia.com/gpu` A10s, 32 CPU,
  128 GiB RAM, 32 GiB `/dev/shm`, 32 GiB temporary storage, a read-only Kaggle
  Secret, a dedicated 700 GiB RWX CephFS PVC, `backoffLimit: 0`, and a 48-hour
  deadline. Requests and limits are equal.
- Use the 32 local validation IDs as the canonical A10 pilot. Process the
  remaining 11,168 rows in 44 sequential chunks: 43 chunks of 256 and one of
  160. Never allow overlapping writers.
- Provide a command-by-command runbook for Kaggle gates, public NRP GitLab
  project creation, local build/test, registry login and push, digest
  resolution, Secret/PVC creation, offline rendering, pilot submission,
  immediate diagnostics, tracked monitoring, sequential chunks, verification,
  S3/rclone export, download, and final hash verification.
- During implementation, perform no registry push or Nautilus mutation. Only
  read-only cluster queries such as `kubectl get`, `describe`, `logs`, and
  event inspection are permitted.

## Limitations

The RTX 5090 test validates the immutable container, task preparation, model
factories, five-epoch protocol, recovery, and export paths. It cannot prove A10
memory fit or four-A10 scheduling. The future operator-run four-A10 pilot is the
production acceptance gate.
