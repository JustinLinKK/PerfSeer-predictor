# PerfSeer V3 A10G 18K dataset coding plan

Date: 2026-07-30

Implementation root: `src/perfseer_v3/dataset_pack`

This is the decision-complete plan for one 18,000-row, end-to-end dataset. It
does not claim that production A10G labels have been collected.

## 1. Goal and boundaries

The production target is exactly `nvidia_a10g_24gb_aws_g5`. Every accepted
label must come from a qualified AWS A10G 24 GiB. The local RTX 5090 is only a
source, task, shape, optimizer, scheduler, and compiler verifier. Measurements
from the RTX 5090 never become A10G labels and cannot prove A10G memory fit or
kernel compatibility.

The frozen contract is:

```json
{
  "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
  "accepted_configurations": 18000,
  "accepted_runs_per_configuration": 1,
  "epochs_per_run": 5,
  "warmup_epochs": [1, 2],
  "measured_epochs": [3, 4, 5],
  "accepted_run_records": 18000,
  "measured_epoch_records": 54000
}
```

The exact 35-family quotas and modality totals remain frozen in
`src/perfseer_v3/configs/a10g_18k_dataset_pack.yaml`. The generated quota uses
50 independent source lineages. The initial manifest and every repaired final
manifest must preserve those quota cells and total exactly 18,000 successful
configurations.

Only end-to-end configurations enter the AWS production dataset. Existing
operation and composite generators remain local QA utilities; they do not
create additional AWS corpora or regression rows.

The workflow is instance-local. It requires no S3, SQS, DynamoDB, Spot
manager, remote lease service, storage controller, or other AWS API. The
operator clones the repository, supplies Kaggle credentials outside the
checkout, and runs one resumable command.

## 2. Exact observation contract

One accepted configuration means one fresh child process and one complete
five-epoch training run. Epochs 1 and 2 update the model normally but are
warmup. Epochs 3, 4, and 5 are the only measured epochs. A successful
configuration is never repeated.

Each accepted `LabelRunRecord` contains exactly three `EpochMeasurement`
objects for epochs 3, 4, and 5. A measured epoch stores only:

- synchronized complete-epoch wall time;
- examples, batches, microsteps, and optimizer-step counts;
- finite-loss and finite-gradient flags;
- timestamped SM utilization, memory-controller utilization, used device
  memory, sample duration, and throttle-reason bits;
- peak PyTorch reserved memory; and
- requested/observed backend, telemetry-completeness, and contamination flags.

The run envelope stores source, graph, dataset, environment, hardware, and
configuration identities plus process/GPU cleanup evidence. These are
provenance or acceptance evidence, not predictor inputs or additional labels.

The six V3 supervised outputs are unchanged:

| Target | Epochs 3–5 aggregate |
| --- | --- |
| `train_epoch_ms` | Arithmetic mean of the three direct complete-epoch times |
| `train_avg_sm_util_percent` | Time-weighted mean over all valid SM samples |
| `train_p95_sm_util_percent` | Nearest-rank P95 over all valid SM samples |
| `train_peak_vram_used_mib` | Maximum sampled used device memory |
| `train_peak_torch_reserved_mib` | Maximum PyTorch reserved-memory peak |
| `train_peak_memory_controller_util_percent` | Maximum valid memory-controller sample |

The three peak values are required outputs, not `GraphFeaturesV3` inputs. The
live `perfseer_v3.training.TARGET_NAMES` and dataset-pack `TARGET_NAMES`
contracts both contain them. Removing them would change the V3 model, artifact,
runtime, and scheduler interfaces; it would not merely simplify collection.

Routine accepted records do not collect these unused measurements:

- PyTorch allocated-memory peak;
- step-time extrapolations or separate CUDA-active time;
- forward/backward/optimizer phase timings;
- operator or profiler traces;
- power, temperature, or clock histories; or
- separate data-stall telemetry.

The source-level `AUXILIARY_TARGET_NAMES` tuple is retained only for local
operation/composite QA records. None of its ten fields is part of an AWS
`EpochMeasurement`, `LabelRunRecord`, or successful regression row.

Loss/gradient flags, traversal counts, throttle bits, process contamination,
backend identity, and cleanup evidence remain because they decide whether a
run is valid. They are not additional regression targets.

The separate OOM probability/stage, uncertainty, confidence, and peak-live-byte
heads do not expand this successful-run measurement contract. OOM identity and
stage come from retained failed attempts; uncertainty and confidence are learned;
and optional peak-live bytes come from graph-liveness data rather than another
NVML measurement. When peak-live data is absent, its auxiliary loss is masked.

The canonical epoch label is direct wall time for a complete epoch; multiplying
a sampled step time is forbidden. Acceptance requires all five epochs to
finish, finite loss and gradients in every epoch, complete telemetry in epochs
3–5, no foreign GPU process, no harmful throttle reason, matching backend
identity, and successful cleanup. The three measured epoch times must satisfy:

```text
(max_epoch_ms - min_epoch_ms) / mean_epoch_ms <= 0.10
```

Partial, failed, OOM, timed-out, contaminated, unstable, or cleanup-failed
attempts retain null targets and do not count toward 18,000.

## 3. Lightweight local RTX 5090 verification

Local verification must not train all 18,000 configurations for five epochs.
It uses two gates.

First, statically validate every manifest row:

- canonical configuration ID and all source/task/protocol hashes;
- exact registered factory and generated lineage;
- task adapter, task kind, output width, and specialized training step;
- architecture fields, input shapes, precision, checkpoint, and batch policy;
- optimizer implementation and parameter-group compatibility;
- scheduler implementation, progress, and step unit;
- eager/compiled request and backend compatibility; and
- exact family, modality, task, regime, and generated-lineage quotas.

Second, derive an immutable execution-signature hash from fields that choose
different executable behavior. Group equivalent rows and deterministically
select a pairwise covering set containing every:

- model family and all 50 generated lineages;
- task adapter and task kind;
- precision, gradient-scaler, and checkpoint policy;
- optimizer, parameter-group, scheduler, and scheduler-step implementation;
- eager and compiled path;
- architecture/shape regime; and
- specialized training step.

For each selected signature, the RTX 5090 gate builds its tiny real-format
task fixture, runs one eager reference update, compiles the training path, and
runs two compiled forward/loss/backward/optimizer/scheduler updates. It checks
finite output, loss, gradients, optimizer/scheduler state, parameter changes,
output shape, and eager/compiled equivalence.

Each manifest row retains its own exact execution-signature hash. It also maps
to a deterministic, factor-complete evidence bundle from the selected set: the
bundle includes the row's exact source/task route and collectively covers every
required executable factor and pair for that row. All signatures in that
bundle must pass before the row is locally cleared. This pairwise gate verifies
the selected implementation paths and interactions; it does not falsely claim
that the row's entire unique cross-product was executed locally.

The frozen source statically validates exactly 18,000 rows and maps every row
to factor-complete evidence drawn from 522 planned representatives. The final
local identities are:

```text
target manifest:        bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e
local plan:             25e2c6946d91620f028acbfea3962d7ca7498bfa601a05e44ba14b7562783a36
validation harness:     cd54878db34da69c061b85136eff6316225a4091bf7a9547626880b5409dc52b
validation environment: 9f9393d5395136a2b045b98f42c6af240892c191d6e57a572ac80b386109ac24
gate summary:           267ac8e527156d86a88083013d4282075309ff23438ff61ad6bef537eba16fd4
```

Tiny real-data decoders for vision, NLP, audio, tabular, and graph tasks pass
locally. Cross-modality short update checks pass, including unequal
encoder/decoder lengths for T5, Kimi, and GRU seq2seq paths, and the corrected
CGCNN/PyG source path passes its focused regression. Generated configurations
bind both the immutable lineage DSL and the executable interpreter source.
Graph reductions use a sorted segment reduction with FP32 accumulation while
retaining the declared BF16 autocast behavior for linear layers and
activations.

All 522 representatives passed in independent fresh processes on the local
RTX 5090. The gate reports zero unresolved failures, covers all 18,000 row
mappings, spans all 35 families and 22 task adapters, and passes an independent
`--verify-only` reconstruction. Every durable pass is bound to the complete
dataset-pack Python source closure, gate driver, and
Torch/CUDA/driver/backend environment. Older superseded passes are preserved
only as diagnostics. The fail-closed gate validates every result and hash
against the current source-built plan, and every record says
`accepted_a10g_measurement: false`.

## 4. Smart batch policy and OOM repair

Initial planning uses full power-of-two ladders:

| Tier | Batch ladder | Initial families |
| --- | --- | --- |
| Light | 16, 32, 64, 128, 256, 512 | MobileNetV3, PReLU/ELU CNN, fastText, SELU MLP, MDN, lightweight generated models |
| Standard | 8, 16, 32, 64, 128, 256 | ResNet, EfficientNet-B0, Inception, ordinary transformers/RNNs, audio CNNs, TabTransformer, GCN/GAT/GraphSAGE, most generated models |
| Heavy | 1, 2, 4, 8, 16, 32, 64 | EfficientNet-B4, high-shape ViT/Swin, U-Net, pix2pix, Restormer, T5, Llama, MoE, DistilBERT joint step, PANNs, CGCNN |

Every one of the 35 registry families has an initial tier.
`independent_generated` is classified dynamically for each of its 50 lineages.
A configuration is promoted for high resolution, sequence length, depth,
width, audio duration, graph size, expert count, dual-model training, FP32,
LBFGS, or disabled checkpointing when those choices raise memory pressure.

The prepared training view contains 4,096 examples. The selected batch is
capped so every epoch contains at least eight batches, and planned rows are
distributed over the complete eligible ladder.

An OOM authorizes exactly one kind of retry: a new configuration ID using the
next lower power of two. A Light configuration that fails at 16 may continue
through 8, 4, 2, and 1. Each failure is retained diagnostically; only the first
successful child in the repair chain fills the quota slot. Batch-1 OOM, other
failures, or instability quarantine the candidate and create a deterministic
same-cell replacement with a genuinely changed architecture/input signature.

The initial manifest remains immutable. Repair and replacement records link
the frozen quota slot to the accepted final configuration.

## 5. Simple dataset-by-dataset AWS workflow

The frozen production environment requires Python 3.11, 3.12, or 3.13. From a
fresh clone, install exactly the checked-in lock:

```bash
uv sync --frozen --extra a10g-dataset-pack
```

Clone MLE-bench separately, leave it clean, and check out exactly revision
`507f92e1138bb6e40dac5c6ee7a6758e6424bf97`. Kaggle credentials must remain
outside the repository, and the production account must manually accept the
rules for all 22 frozen competitions before collection.

The implemented operator command is:

```bash
uv run --frozen --extra a10g-dataset-pack python scripts/run_a10g_18k_pack.py \
  --workspace /mnt/perfseer-a10g-18k \
  --mlebench-checkout /opt/mle-bench
```

The entrypoint adds this checkout's `src` directory to the parent and child
import paths. Before changing workspace state, it checks the exact frozen
Torch 2.11.0, PyG 2.7.0, torchvision 0.26.0, torchaudio 2.11.0, Transformers
5.7.0, and Triton 3.6.0 versions; imports every required runtime dependency in
an isolated child; requires a working CUDA PyTorch build; validates and imports
every pinned MLE-bench preparer; authenticates the Kaggle account; and retrieves
a complete paginated inventory for all 22 tasks. Missing rules acceptance
therefore fails before a task is downloaded or labeled.

Kaggle credentials are supplied through the normal external Kaggle credential
mechanism. Dependency probes use a private temporary `KAGGLE_CONFIG_DIR` and
inert import-only credentials, so they cannot inspect the operator's real
Kaggle file. Token variables are removed from model/preparation child
environments, and each child receives a private empty `KAGGLE_CONFIG_DIR`
instead of the operator's external credential path. Credentials must not be
copied into this repository or the workspace records.

The first labeling invocation writes one hash-bound campaign-environment lock
covering Python, PyTorch, CUDA, cuDNN, NVIDIA driver, and the optional
`PERFSEER_CONTAINER_DIGEST`. Every resume must match it exactly.

The command performs one resumable loop:

1. Build, validate, and freeze the exact 18K manifest before any download.
2. Select the next incomplete task in the frozen 22-task order.
3. Check conservative archive/extraction/preparation estimates against the
   600 GiB workspace limit while retaining 40 GiB safety headroom.
4. Download only that Kaggle task. An unverified stale or partial download is
   discarded and downloaded again rather than trusted.
5. Hash and safely inspect the archive, reject traversal/symlink/duplicate
   entries, extract it, and run the pinned MLE-bench preparation boundary.
6. Build one shared 4,096-example prepared view whose selected files or ZIP
   members are size- and SHA-bound; revalidate all view and source bytes on
   resume.
7. Run every frozen quota slot assigned to the task, with one isolated worker
   per visible qualified A10G.
8. Atomically save accepted records, failed attempts, repair links, hashes,
   slot state, and task-loop state.
9. Require a completion receipt proving a one-to-one mapping from every task
   quota slot to a validated durable accepted record.
10. Recheck the receipt and record hashes, then delete only that task's
    archive, extracted tree, and reconstructible prepared cache. Resume
    completes an interrupted receipt-first deletion before advancing, and
    revalidates every previously completed receipt and accepted record.
11. Continue to the next task.

Disk admission uses one free-space calculation, not a persistent storage
ledger:

```text
current durable bytes
+ task archive estimate or exact size
+ extracted/nested/prepared-view bound
+ extraction temporary bound
+ 40 GiB safety margin
< 600 GiB
```

The materializer and resumable production call site are implemented. All 22
task layouts pass tiny local archive/schema/resume/integrity/cleanup tests; no
real Kaggle task data was downloaded during local verification.

## 6. AWS labeling behavior

Eager/compiled mode remains part of the immutable configuration ID. A compiled
row completes compilation and autotuning before epoch 1. Immediately before
epoch 3 the child synchronizes CUDA, resets peak memory counters, and starts
asynchronous target-producing telemetry. Telemetry runs continuously through
epochs 3–5.

Each physical GPU has one parent supervisor. Each attempt runs in a fresh child
whose `CUDA_VISIBLE_DEVICES` exposes exactly that GPU. The child must see one
qualified A10G. The supervisor rejects a busy GPU, enforces a timeout, kills
the complete process group when needed, and requires process exit plus a stable
return to the pre-run NVML memory baseline before reusing the GPU.

Every child writes stdout/stderr to a private `0600` attempt log under the
external workspace. Successful logs are removed; failed logs remain and the
parent error reports their path. A missing or corrupt envelope, unexpected
child exception, dependency/data/compiler failure, or unsafe cleanup aborts
the campaign instead of consuming deterministic quota replacements. Only an
explicit CUDA OOM enters batch descent; a completed but unstable measurement
enters the same-cell quarantine/replacement path.

Compiled attempts use per-attempt `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` directories inside the guarded workspace. The supervisor
deletes these reconstructible caches after the child exits, preventing
compiler artifacts from filling an unguarded instance-root `/tmp`.

The runner preserves only the samples needed to derive the six outputs and
the integrity flags needed for acceptance. It never copies measured VRAM or
utilization into the model's input feature graph.

The finalized regression row stores the six aggregate labels and hashes its
accepted evidence record; it does not duplicate raw telemetry into the model's
feature tensor. Raw samples remain audit evidence for recomputing aggregates and
checking throttling; the accepted evidence separately retains the contamination
flag.

An unchanged unstable run is not repeated. It is quarantined and replaced in
the same quota cell. OOM descent is allowed only because each smaller batch is
a new configuration. Failed/OOM attempts remain outside the successful
regression manifest.

The fresh-process supervisor, five-epoch runner, mixed eager/compiled dispatch,
minimal telemetry, aggregation, stability gate, repair handoff, atomic output,
and cleanup proof are implemented and pass CPU-only protocol fixtures. There is
no separate A10G smoke corpus or preliminary campaign. The parent qualifies
every visible GPU as an A10G before the first attempt, and every production row
then passes or fails the same fail-closed runner; the task loop resumes and
repairs failures normally.

## 7. Finalization and completion gates

Finalization may run only over validated accepted A10G records. It must prove:

- exactly 18,000 unique accepted configuration IDs and run IDs;
- exactly 54,000 ordered epoch measurements, always epochs 3, 4, and 5;
- exact family, modality, task, regime, precision, optimizer, scheduler,
  execution, and generated-lineage quotas after repairs;
- every batch or effective-tier change is a legal, linked OOM repair and is
  reported as a delta from the frozen manifest rather than treated as quota
  drift;
- no accepted record with a failed stability, telemetry, contamination,
  backend, traversal, finite-state, or cleanup gate;
- canonical source and graph identities that keep every related row in one
  leakage group;
- complete repair/quarantine lineage and no failed attempt in the successful
  manifest; and
- hash-bound raw hardware and environment sidecars plus source, task, dataset,
  receipt, and support-contract provenance.

Create deterministic 80%/10%/10% train/validation/test splits by whole source
group. Equivalent graphs and mutations of one source lineage remain in one
split, and at least 10 generated lineages are held out completely. Fit the six
target transforms using training rows only. Feature normalization remains a
later GraphIR/training-manifest concern because this label pack deliberately
does not materialize `GraphIRV3` files.

Produce concise quota, failure, batch-repair, stability, execution-signature,
provenance, deduplication, split-leakage, and normalization reports. Preserve
failed/OOM attempts outside the successful training manifest.

The deterministic finalizer is implemented in
`src/perfseer_v3/dataset_pack/finalization.py` and is invoked automatically
after the 22nd task receipt. It writes canonical `accepted_labels.jsonl`,
`failed_attempts.jsonl`, `target_transform.json`, `audit_report.json`, and
`dataset_manifest.json`, then writes `completion_receipt.json` last. An
explicit `scripts/finalize_a10g_18k_pack.py --verify-only` recomputes every
artifact byte without creating or changing workspace state.

Exact fixtures cover 18,000 accepted rows, 54,000 actual epoch subrecords, all
22 completion receipts, direct/OOM/quarantine/substitution lineages, exact
grouped splits, train-only target transforms, provenance tampering, deterministic
publication, CLI read-only verification, and workflow auto-finalization. The
output declares `artifact_scope: a10g_label_pack`,
`accepted_a10g_measurement: true`, and `graph_ir_materialized: false`; it does
not claim to be a complete V3 training manifest.

On the final source snapshot, all 13 exact finalization tests and all 99
dataset-pack tests pass. An independent verifier separately reproduced all 112
tests and 294 subtests, the read-only failure behavior, actual epoch counts, all
receipts, workflow integration, and both repair-lineage forms.

Remaining production completion gates are:

1. Collect task by task until exactly 18,000 A10G records are accepted.
2. Run the implemented finalizer and its independent `--verify-only` pass over
   those real accepted records.

Documentation and source verification must parse every JSON example; confirm
18,000/54,000 and warmup/measured epoch counts; reject stale repetition,
multi-run aggregation, step-extrapolation, production operation/composite, storage
ledger, or AWS-service requirements; map all 35 families and 50 generated
lineages to batch policies; map all 18K rows to factor-complete local evidence;
distinguish RTX validation from A10G labels; and pass focused tests,
Markdown/link checks, `git diff --check`, and independent review.
