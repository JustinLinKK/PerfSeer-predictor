# PerfSeer v3.2

The installed package is `perfseer_v32`. Both the T1 teacher and S1 student now
consume paired training/inference graphs and predict the dataset's twelve native
A10 labels. This contract requires fresh training. Existing v3, v3.1, and earlier
v3.2 checkpoints and prepared datasets remain separate.

## Dataset and paired graphs

The two ZIP archives in `dataset_with_label/raw_source` contain 74,704 source rows:
34,684 exact duplicates and 40,020 unique measurements. Preparation preserves
native labels, provenance, and the existing 32,108 / 4,100 / 3,812 train,
validation, and test assignments. It writes `dataset_with_label/ready_for_train_12`;
the legacy `ready_for_train` directory is retained as the split reference.

```bash
python -m perfseer_v32.dataset prepare --workers 4
python -m perfseer_v32.dataset verify --workers 4
```

Use `--limit 24` for a partial capture preflight covering six modalities and four
precision settings. Partial preparation cannot be used for training. Matching
strict training captures are reused by default from the legacy v3.2 dataset;
`--reuse` can name another verified training-capture directory. Every inference
graph is captured in eval mode under no-grad and the configured autocast policy,
with model state and RNG restored and eager/replay outputs checked under the same
backend settings. Training retains forward, loss, backward, and optimizer phases
and recorded parameter-gradient parity evidence. Capture failures are explicit.

These are logical execution graphs, not measured CUDA kernel schedules. CPU
capture does not validate A10 CUDA execution or TF32 numerical behavior. Inference
features retain batch shape, precision, and backend while neutralizing training-only
optimizer, scheduler, accumulation, and epoch settings. Graph preparation does not
alter labels. The separately applied shorter-time policy below changes active
targets while retaining original measurements. Extrapolated epoch time remains
auxiliary metadata for the original measurement.

## Model and output contract

T1 retains width 1280 and ten backbone blocks (274,598,156 parameters); S1 retains
width 224 and two blocks (2,888,744 parameters).
Within each model, the two graph passes share node/edge/global encoders and
message-passing weights. Separate mode readouts feed six MLP prediction heads:

| Head | Ordered targets |
| --- | --- |
| Training timing | `train_step_wall_ms`, `train_step_gpu_ms`, `train_epoch_ms` |
| Training SM | `train_avg_sm_util_percent` |
| Training memory | `train_avg_vram_mib`, `train_peak_vram_mib`, `train_peak_torch_allocated_mib` |
| Inference timing | `infer_step_wall_ms`, `infer_step_gpu_ms` |
| Inference SM | `infer_avg_sm_util_percent` |
| Inference memory | `infer_avg_vram_mib`, `infer_peak_vram_mib` |

The prediction tensor has shape `[batch, 12]` in this exact order. Timing is in
milliseconds, SM in percent (0–100), and memory in MiB. Timing/memory use positive
softplus outputs with training-median scales; SM uses `100 * sigmoid`. Output biases
start at training medians. Each mode has normalization fitted only on training rows.
The backbone supports BF16 autocast; heads, output transforms, and losses use FP32.

Inputs use `perfseer_v32_paired_graphs_v1`; outputs use
`perfseer_v32_twelve_outputs_v1`. Dataset, features, normalization, and checkpoints
have v2 identities. Prediction/export consumes paired designs and serialized
per-mode normalization. Old three-output artifacts are rejected, including when a
resume code-fingerprint override is supplied.

## Losses, metrics, and gates

Each of the six heads receives one sixth of the supervised loss, averaging
normalized absolute error within that head. Timing/memory divide by the positive
true target; SM divides by `max(target, 1.0)` to prevent zero-SM loss explosions.
Distillation uses `0.6 * hard + 0.4 * soft + 0.05 * relation`, with ground-truth
denominators for both prediction losses and masked, per-mode relational matching.
Muon updates backbone/readout linear matrices; AdamW updates prediction heads,
embeddings, biases, and normalization parameters.

Evaluation uses `abs(prediction - target) / max(abs(target), 1e-6)` for **every**
label, including SM. The 5% threshold is relative error, not five percentage points
of utilization. Arithmetic is float64 over exported FP32 tensors. Reports include
exact 5%/10% hit counts and rates, physical MAE, relative mean errors, stabilized
training loss under its own name, and zero-SM diagnostics.

Checkpoint selection maximizes worst-label 5% accuracy, then mean 5%, worst-label
10%, mean 10%, and finally favors the earlier epoch. The gate requires at least
95% within 5% for every label. The selected teacher must pass validation and its
final test before student training starts. The same acceptance rule applies to the
student. Gate records bind checkpoint hash, dataset, normalization, metric/selection
versions, and evaluation configuration; stale records are rejected on resume.

The existing defaults remain: teacher 600 epochs / 10 warmup epochs; student 100 /
5; effective predictor batch 256; maximum microbatch 64. Teacher early stopping
starts at epoch 30 with patience six. Whole-batch OOM retry retains every row and
restores RNG. Resume requires matching contracts, schedule, normalization, and code;
student resume also requires the same teacher checkpoint.

```bash
# Starts a full campaign only when explicitly invoked.
python -m perfseer_v32.runner --output record/perfseer-v32/twelve-run

# Independent verification of an epoch's exported values and hit counts.
python -m perfseer_v32.verification --predictions \
  record/perfseer-v32/twelve-run/teacher-epoch-0001-predictions.json.gz

# Diagnostic ceilings and constant-training-median baselines; excludes test metrics.
python -m perfseer_v32.verification --audit-dataset \
  --dataset src/perfseer_v3.2/dataset_with_label/ready_for_train_12 \
  --output record/perfseer-v32/twelve-audit
```

`--local-validation --teacher-epochs 1` explicitly runs a complete teacher epoch
and export/reload check without test evaluation or student distillation. It is
execution validation, not an accuracy claim. The original labels contain
conflicting measurements for identical inputs; separate graphs and task heads do
not guarantee that the strict accuracy gate becomes attainable.

## Implementation verification (2026-09-07)

The [verification report](../../record/perfseer-v32/native-twelve-20260907/verification-report.json)
records 87 passing focused/regression tests, all 2,352 verified graph pairs, unchanged
40,020 measurements and split assignments, full-training normalization, and its
independent global-moment check. Both full-capacity models passed bounded synthetic
CPU training, checkpoint/export reload, and CPU BF16 forward checks. Both also
passed forward checks on 24 real validation cases spanning six modalities and four
workload precision settings. Test inputs were validated without scoring test labels.

The paired-input audit gives optimistic validation 5%-tolerance ceilings of 52.61%
for training epoch time, 51.98% for training SM, and 70.24% for inference SM. These
are input-ambiguity diagnostics, not trained-model scores. The strict 95% gate
remains unattainable with these inputs and preserved labels. No full training
campaign, CUDA forward/training validation, new image publication, or cluster job
was performed for this refactor. The native CPU RNG helper avoids initializing
CUDA during capture or CPU tests.

## Local input and head experiment (2026-09-07)

The [experiment report](../../record/perfseer-v32/local-ambiguity-experiment-20260907/README.md)
traces all 36,208 training/validation rows through 2,096 paired artifacts. Neither
feature extraction nor normalization introduced additional identical-input groups.
Six source-identical pairs, covering all modalities, already have conflicting
timing and SM labels. Original template batch sizes, operator aliases, and input
names explain representative source-to-capture merges; three alias pairs passed
exact output/gradient checks in the pinned local runtime. Original A10 runtime
and measurement conditions still need source evidence.

A CPU head-only comparison used a frozen, randomly initialized native S1 encoder,
48 real paired inputs, 1,926 training rows, 1,183 validation rows, three seeds,
and 600 updates per variant. Twelve independent heads did not consistently improve
over six grouped heads. This is a limited diagnostic, not end-to-end predictor
validation or a production gate result. The six-head production design and every
source label remain unchanged; the report includes a specific teammate request.

## Similarity-guided training estimates (2026-09-08)

The separate [similarity-cleaning experiment](../../record/perfseer-v32/similarity-cleaning-20260908/README.md)
produces explicitly estimated training labels with preserved native targets and
donor provenance. It adjusted 51 values in 26 training records using strong
exact-repeat consensus; broader conflicts remain unresolved. Original dataset
artifacts and validation/test measurements are unchanged. The estimates have a
different schema and are not a native dataset or production gate reference.

## Active shorter-time reference labels (2026-09-08)

Following the user's explicit choice, `ready_for_train_12/dataset_manifest.json`
now selects revised split files under `label-policies/perfseer_v32_shorter_timing_v1/`.
The original `train/`, `validation/`, and `test/samples.json.gz` files remain intact;
their manifest is saved as `original-dataset-manifest.json` in the policy directory.
Every original measurement remains in `native_targets`. The production loader
reads the selected `targets` through the active manifest.

The [update report](../../record/perfseer-v32/shorter-timing-20260908/README.md)
records 39,085 rows with revised targets and 177,425 changed numeric values,
including the six earlier consensus SM edits. All 40,020 records remain. The
630.496 ms example now uses the observed 277.016 ms timing tuple.

The verified consensus training edits are applied first. Remaining timing groups
are compared only within the same split, dataset/subset, hardware, modality,
recorded mode settings, and identical mode input tensors. A conflict means the
measurements have no common interval within 5% for at least one timing output.
Conflicting training groups adopt the remaining observation with the shortest
epoch time, copying its wall/GPU/epoch tuple together; inference groups adopt the
shortest wall step and its corresponding GPU time. Ties use the other timing
values and then sample ID. No minima are copied between unrelated structures or
between splits. SM and memory retain their values after the earlier consensus edits.

```bash
python -m perfseer_v32.label_policy \
  --consensus record/perfseer-v32/similarity-cleaning-20260908/estimated-training \
  --workers 4
python -m perfseer_v32.dataset verify --workers 4
```

The command verifies an already active policy on rerun. The policy directory
contains the original manifest, consensus provenance, input identities, all donor
decisions, and every replacement. Dataset verification reconstructs revised rows
from their original donors and rejects mismatches. The new dataset fingerprint
invalidates previous normalization, checkpoints, and gate records; fresh training
is required. Prediction exports and gates identify the label policy explicitly.

**These are user-selected shorter-time reference labels, not remeasurements.**
Validation and test target values also follow this policy, while original measured
evaluation labels remain available in the preserved files. Scores against revised
references are not evidence of improved accuracy against the original profiler
measurements. The 95%-within-5% gate is unchanged; SM conflicts remain and no
training campaign or accuracy claim accompanies this dataset update.

## Image and deployment

The refreshed `dist/perfseer-v32-full-training-20260908.zip` contains the complete
current design and revised dataset. Its root `train.sh` verifies the package and
builds from those exact files before starting the A100 training workflow. See the
[handoff instructions](containers/training/HANDOFF.md). The historical published
image below remains an older three-output release.

```bash
python -m pytest -q tests/test_perfseer_v32.py tests/test_perfseer_v31.py tests/test_perfseer_v3_combined_training.py
python scripts/build_perfseer_v32_image.py --tag perfseer-v32:release-candidate
```

The image uses a digest-pinned PyTorch CUDA 12.8 base and contains the complete
verified train-ready dataset. Raw archives, source extraction directories,
credentials, checkpoints, and older prepared datasets are excluded.
It runs as UID/GID 1000 and supports a read-only root filesystem.
The build script stages an explicit file set into a fresh context and verifies
every copied file against its source hash before invoking Docker.

Nautilus preparation follows its [GPU job guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/).
Historical image publication, immutable digest, local full-epoch evidence, and the
prepared submission manifest are recorded under `record/perfseer-v32/`. No cluster job is
submitted by dataset preparation, training, or image building.

## Historical three-output image

The following immutable image and its commands preserve the earlier three-output
release. They do not contain this native twelve-output refactor. Build a new image
with the command above for the new contract; no new image was published as part of
this change. Use a compatible Linux or Windows WSL2 Docker GPU environment. The image includes the code and all 40,020 prepared measurements; no
repository checkout, ZIP download, or host Python installation is needed. It
contains no trained checkpoints, so a fresh output directory starts new training.

```bash
PERFSEER_IMAGE=gitlab-registry.nrp-nautilus.io/justinlinkk/prefseer-predictor-labeling@sha256:5a5778ca5ad287c85f3b75361b3ecde110c7133ac2424009fb083299a85d85a4
docker pull "$PERFSEER_IMAGE"
mkdir -p perfseer-v32-output

docker run --rm --name perfseer-v32-training \
  --gpus device=0 --shm-size=2g --stop-timeout=300 \
  --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache \
  -v "$PWD/perfseer-v32-output:/outputs" \
  "$PERFSEER_IMAGE" \
  --dataset /opt/perfseer/src/perfseer_v3.2/dataset_with_label/ready_for_train \
  --output /outputs/run --resume \
  --teacher-epochs 600 --student-epochs 100 \
  --effective-batch 256 --microbatch 4 \
  --teacher-early-stopping-patience 6 \
  --teacher-early-stopping-min-epochs 30 \
  --no-teacher-stop-on-validation-gate
```

Logs, feature cache, checkpoints, and exports remain under
`perfseer-v32-output/run`. Rerun the same command after an interruption to resume
the latest checkpoint. Microbatch 4 was verified on an RTX 5090; OOM retries reduce
it without changing the effective batch. The student stage starts only after the
selected teacher passes both validation and test quality gates.

GPU setup: [NVIDIA Container Toolkit for Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
or [Docker Desktop GPU support through WSL2](https://docs.docker.com/desktop/features/gpu/).

## RTX 5090 transfer-labeling campaign

The transfer labeler retains all **12 native profiling targets** (seven training
and five inference metrics), including average and peak device VRAM for both
phases. The native predictor now uses the same ordered twelve-label contract.
This prepares data for later adaptation; it does not train or modify the predictor.

Run the launcher from the repository. It loads the local v3, v3.1, and v3.2 package
aliases directly from this checkout. If the active interpreter lacks PyTorch, it
re-executes the prepared Conda interpreter; override that path with
`PERFSEER_PYTHON` when needed:

```bash
python scripts/run_perfseer_v32_transfer_labeling.py --help
```

Worker subprocesses use the same bootstrap launcher, so neither the parent nor the
workers depend on editable package registration. On this workstation the fallback
interpreter is `/home/justin/miniconda3/envs/perfseer/bin/python`.

```bash
python scripts/run_perfseer_v32_transfer_labeling.py prepare
python scripts/run_perfseer_v32_transfer_labeling.py prepare-data
python scripts/run_perfseer_v32_transfer_labeling.py verify
```

All three commands run on CPU. `prepare-data` acquires the original source datasets
and prepares their deterministic tiny masks at
`record/perfseer-v32/transfer-source-data`. It verifies existing downloads on rerun.
Kaggle downloads use the installed CLI and the user's existing external credentials;
competition access must already be available. OGBN-Products comes from the official
[OGB archive](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products), read without
constructing its full PyG graph. No additional Python dependencies are required in
the current environment.

The source catalog was recovered from the original v3 real-A10 workflow's
`dataset_sources/registry.json` and `scripts/manage_dataset_sources.py`, then
reconciled against every native source label:

| Dataset | Original source | Full entries | Tiny entries |
| --- | --- | ---: | ---: |
| Cassava | Kaggle competition `cassava-leaf-disease-classification` | 21,397 | 1,024 |
| Pothole segmentation | `farzadnekouei/pothole-image-segmentation-dataset` | 780 | 780 |
| TACO detection | `vencerlanz09/taco-dataset-yolo-format` | 6,004 | 1,024 |
| Jigsaw | Kaggle competition `jigsaw-toxic-comment-classification-challenge` | 159,571 | 1,024 |
| CNN/DailyMail | `gowrishankarp/newspaper-text-summarization-cnn-dailymail` | 287,113 | 1,024 |
| Animal audio | `warcoder/cats-vs-dogs-vs-birds-audio-classification` | 610 | 610 |
| Store Sales | Kaggle competition `store-sales-time-series-forecasting` | 3,000,888 | 1,024 |
| Credit-card default | `uciml/default-of-credit-card-clients-dataset` | 30,000 | 1,024 |
| OGBN-Products | `ogbn-products` | 2,449,029 | 1,024 |

The original selector orders archive-relative media keys, CSV record keys, or
graph node keys by SHA256 of `dataset_id + ':' + key`, then keeps the first
`min(1024, full_count)`. TACO and Pothole retain the original media-entry convention.
Hashes, source provenance, masks, and input fingerprints are stored alongside the
raw archives. `verify` rebuilds all nine masks and checks every archive hash.
To reuse another existing download cache during preparation, add
`--reuse-data-root /actual/cache/root`; files are copied only after verification.

The default campaign output is
`record/perfseer-v32/transfer-rtx5090`, containing the frozen 10,240 configurations,
all source aliases, coverage, pilot/repeat panels, and adaptation/evaluation masks.
The split is 7,456 training, 1,408 validation, and 1,376 test configurations;
all variants retain their A10 architecture split. Batches span 1 through 128,
with four precisions and additional optimizer/accumulation comparisons.

After the current GPU experiment finishes, explicitly start labeling:

```bash
python scripts/run_perfseer_v32_transfer_labeling.py label \
  --output record/perfseer-v32/transfer-rtx5090 --resume
```

For an approximately 24-hour campaign, use the frozen reduced scope:

```bash
python scripts/run_perfseer_v32_transfer_labeling.py label \
  --campaign-profile 24h \
  --output record/perfseer-v32/transfer-rtx5090 --resume
```

The reduced scope contains 4,128 labels and 4,256 measurements including repeats.
It retains all 248 anchors, all four precisions, batches 1/8/128, all 992 A10-matched
cases, and reduced optimizer and accumulation comparisons. Its estimate is 23.0
hours at the observed 185 measurements/hour. It reuses compatible measurements from
the full campaign and writes `labels-24h.json.gz` and `labeling-report-24h.json`
without replacing the full-campaign artifacts. Verify completion with
`verify --campaign-profile 24h --require-complete`.

Press `Ctrl+C` once to pause labeling. The launcher terminates the active worker,
preserves its partial evidence, refreshes the scoped report, records a
`campaign_paused` event, and exits without a traceback. Run the same command again
with `--resume`; every verified configuration is skipped and only the interrupted
configuration is measured again. Partial phase timing is never accepted as a label.

The source root defaults to the prepared directory above, so no placeholder path
is needed. Override it with `--source-data-root` only for another verified root.
`label` verifies `source-data-manifest.json`, all nine raw archives and masks, and
their input fingerprints before checking the GPU. It performs no downloads.
Missing data or competing GPU compute processes stop the launch. The recovered
adapter generates tensors from real subset keys and file fingerprints; it does
not decode conventional training examples or silently substitute another input
protocol.

Each configuration measures inference before training and retains at least five
seconds per phase. The 48-case pilot runs first; its failure prevents the main
campaign. A fixed panel of 512 configurations gets two additional repetitions.
This means at least 31 hours of measured phases for a complete campaign, before
warmup, graph capture, and process overhead; the pilot provides actual timing.

Resume reuses verified successful attempts and archives interrupted attempts
before retrying them. OOM and other completed failures remain explicit; batches
are never silently reduced. Code, source inputs, hardware/software environment,
and protocol fingerprints must match, including the source-data manifest.
Progress is written to `progress.jsonl`,
worker output to `worker.log`, and raw phase measurements to `attempts/`.
`labels.json.gz` contains verified primary measurements with repetition statistics.
Use `verify --require-complete` to require all 10,240 records, each with all
12 targets and both completed phases. Ordinary `verify` also accepts a correctly
prepared campaign with zero GPU measurements.

## Optional training-time and VRAM calibration

See [CALIBRATION.md](CALIBRATION.md) for the frozen-source two-adapter path,
native-label audit, CPU experiment commands, deployment gates and rollback.
The existing twelve-output predictor remains the default.
