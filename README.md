# PerfSeer Predictor

PerfSeer predicts training and inference performance from compute graphs. This
expanded-catalog branch keeps SeerNet as a graph predictor while extending the
dataset, feature schema, profiler, and training flow to convolutional,
transformer, recurrent, graph, audio, detector, segmentation, and tabular model
families. Git tracks source, tests, configs, and this README; generated packs,
profiler results, datasets, checkpoints, and smoke outputs are ignored.

## Student Predictor Deployment

`src/perfseer_student` owns the production student graph featurizer, source
encoder, model definition, export tool, registry loader, and strict CPU
TorchScript runtime. Source conversion reuses `perfseer_source_converter`.

The deployment contract is:

- raw node features: 53 (`23` operation categories + `30` continuous values)
- raw edge features: 3
- raw global features: 40 (`36` graph values + `4` precision categories)
- outputs: six training/inference targets; `train_mem` at index `1` is average
  used GPU VRAM in MiB
- artifacts:
  - `models/nvidia_a10/student_a10_cpu.torchscript.pt`
  - `models/nvidia_rtx_pro_6000_blackwell/student_rtx_pro_6000_blackwell_cpu.torchscript.pt`
- device: CPU only

Each artifact embeds its own input normalization and output de-normalization.
Hardware aliases, compute capability, VRAM range, schema, output index, path,
and SHA-256 are recorded in `models/registry.json`.

### RTX PRO 6000 Blackwell compatibility

The trusted `student_RTX_6000_Blackwell.pt` checkpoint was trained with a
legacy `53 / 3 / 14` input contract. Its 14 global fields are the first ten
structural graph features followed by the four precision fields. The deployed
TorchScript wrapper accepts the current `53 / 3 / 40` scheduler contract and
selects those original fields before normalization and inference; it does not
pretend the checkpoint learned the omitted branch/join/depth and operation
histogram fields.

The registry supports the
[NVIDIA RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)
Server, Workstation, and Max-Q Workstation editions. These variants have 96 GB
VRAM and CUDA compute capability 12.0. The scheduler still falls back to
branch profiling when the reported name, compute capability, or VRAM does not
match.

Run the focused deployment verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Terminology:

- NRP = National Research Platform; NGC = NVIDIA GPU Cloud.
- GPU = Graphics Processing Unit; CPU = Central Processing Unit.
- CUDA = Compute Unified Device Architecture; PVC = Persistent Volume Claim.
- SM = Streaming Multiprocessor; GNN = Graph Neural Network.
- CNN = Convolutional Neural Network; RNN = Recurrent Neural Network.
- GRU = Gated Recurrent Unit; LSTM = Long Short-Term Memory.
- FP32 = 32-bit floating point; FP8 = 8-bit floating point.
- NVFP4 = NVIDIA 4-bit floating point; JSONL = JSON Lines; PKL = Python pickle.

## Design

The baseline was Convolutional Neural Network-heavy: convolution, batch norm,
pooling, `Gemm`, `Add`, and `Concat`. Current schema: `perfseer_graph_v1`.

- Operators: convolution, depthwise/transpose convolution, normalization,
  embedding, matmul/bmm, attention, recurrent ops, graph-message ops,
  activations, pooling, upsample, detector heads, segmentation heads, tabular
  ops.
- Features: tensor rank/shape, compute, memory, topology, destination tensors,
  architecture family, modality, variant, depth/width buckets, precision recipe,
  label domain.
- Hardware policy: one teacher/student pair per hardware class, not one
  cross-hardware predictor.

Because feature dimensions changed, train this branch from scratch. Do not reuse
baseline checkpoints.

## Environment

Use the exported `perfseer` conda environment for local development and
transfer to other machines:

```bash
conda env create -f environment.yml
conda activate perfseer
```

`environment.yml` is exported without build strings or a local `prefix:` so it
can be recreated on another compatible Linux/CUDA platform.

## Model Input And Output

Training object: PyTorch Geometric `Data`.

- `data.x`: node features; `data.edge_index`: directed graph edges.
- `data.edge_attr`: edge tensor summaries, optional destination tensor,
  edge-topology features.
- `data.u`: graph-level aggregate, architecture, precision, and label-domain
  features.
- `data.y`: standardized six-target label; `data.y_raw`: raw six-target label.
  The six target names depend on `features.target_source`.

Canonical v2 model outputs, in order:

```text
train_epoch_ms, train_avg_sm_util_percent, train_p95_sm_util_percent,
train_peak_vram_used_mib, train_peak_torch_reserved_mib,
train_peak_memory_controller_util_percent
```

The legacy compatibility output order is:

```text
train_util, train_mem, train_time, infer_util, infer_mem, infer_time
```

Profiler label files remain PerfSeer-compatible:

```text
{'train': '<7 pipe-separated fields>', 'infer': '<7 pipe-separated fields>'}
```

Each phase string is:

```text
time|average_sm_util|average_memory_util|average_memory_usage|peak_sm_util|peak_memory_util|peak_memory_usage
```

`parse_label()` maps the two phase strings into the six model targets.

Scheduler-grade runs additionally write `label/scheduler_label_v3.jsonl` and
`label/scheduler_resource_label.jsonl`. The canonical v2 target source
`scheduler_v2_train` combines measured scheduler epoch time from
`scheduler_label_v3.targets.train_epoch_ms` with sustained utilization and memory
targets from `scheduler_resource_label.targets`. The legacy six-target files
remain a compatibility projection while the optimized data path learns extra
dataset, dataloader, optimizer, precision, and hardware features from
`precision_metadata.jsonl`.

For training-time labels, read the field carefully: the legacy
`label/label/*.txt` `train.time` value is the compatibility timing field used by
the original six-target parser, not a measured full epoch. For scheduler
packing, use `scheduler_label_v3.targets.train_epoch_ms`. In the measured-epoch
scheduler workflow this is the mean wall time of the measured epochs after one
warmup epoch. In legacy runs it falls back to the step-extrapolated one-epoch
estimate.

## Repository Layout

- `src/perfseer/`: shared schema plus original parser/model utilities.
- `src/perfseer-optimized/`: training, evaluation, distillation, and deployment
  package, imported as `perfseer_optimized`.
- `src/perfseer_source_converter/`: source-to-graph conversion.
- `nrp_calibration_pack/`: template catalog generator, generated-model runtime,
  profiler, profile-dataset builder, Dockerfile, and National Research Platform
  submit wrapper.
- `dataset_sources/`: approval-first real dataset registry and one Markdown
  review card per candidate dataset.
- `scripts/manage_dataset_sources.py`: approves, dry-runs, downloads, and
  prepares real dataset sources after manual review.
- `nrp_calibration_pack/profile/make_workload_specs.py`: combines model
  sources, approved dataset profiles, subset masks, batch sizes, optimizer,
  precision, and hardware into scheduler `WorkloadSpec` rows.
- `scripts/rebuild_source_tar_dataset.py`: rebuilds `dataset/cg/cg` and
  `dataset/label/label` from a source-label package.
- `scripts/run_nrp_source_workflow_local.py`: submits the source-first Nautilus
  workflow, waits for stages, and downloads source-label and dataset packages.
- `scripts/run_hardware_distill_flow.py`: canonical v2 teacher training followed
  by same-hardware student distillation.

## Dataset

Canonical generated pack: `nrp_calibration_pack/`. Canonical materialized
training dataset: `dataset/`.

```text
dataset/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  label/scheduler_label_v3.jsonl
  label/scheduler_resource_label.jsonl
  precision_materialization_report.json
  precision_rejected_rows.jsonl
```

The balanced local scheduler catalog uses 10,005 base templates and no
heavyweight model-library imports such as torchvision, timm, or transformers.
That size is intentional: 10,005 is divisible by the 15 architecture families,
so every family contributes exactly 667 base models. A four-precision expansion
of this balanced set has 40,020 profile points.

| Family | Count |
| --- | ---: |
| `ast_audio_transformer` | 667 |
| `bert_encoder` | 667 |
| `efficientnet_cnn` | 667 |
| `ft_transformer_tabular` | 667 |
| `gat_graph` | 667 |
| `gru_temporal` | 667 |
| `lstm_temporal` | 667 |
| `mpnn_graph` | 667 |
| `resnet_cnn` | 667 |
| `t5_encoder_decoder` | 667 |
| `unet_encoder_decoder` | 667 |
| `vgg_cnn` | 667 |
| `vit_encoder` | 667 |
| `wav2vec2_audio` | 667 |
| `yolo_detector` | 667 |

Variant mix per family: 10% canonical anchors, 30% added-depth, 30%
dropped-depth, 20% width/shape/hyperparameter changes, 10% mixed stress.
Manifest fields: `model_id`, `architecture_family`, `variant_kind`,
`variant_signature`, `input_specs`, `feature_schema_version`, `model_file`,
`subset_graph_file`, `precision_config`, `profile_point_id`, label paths.

Acceptance gates for the balanced local scheduler workflow: 10,005 rows in
`nrp_calibration_pack/manifest/subset_manifest.jsonl`; each architecture family
has 667 base models; every row has architecture, variant, input spec, schema,
precision, and model path metadata; unsupported operator coverage is zero.

## Real Dataset Scheduler Workflow

The scheduler predictor should be trained from real task datasets, not random
tensor-only workloads. Use one approved dataset per task family and create
deterministic subset masks (`tiny`, `small`, `medium`, `large`, `full`) to
represent different dataset sizes. The registry has a default `local` tier for
this RTX 5090 workstation and a `nautilus` tier for PVC-backed large/reference
datasets.

Review candidate sources first:

```bash
python scripts/manage_dataset_sources.py list --tier local
python scripts/manage_dataset_sources.py show cassava_leaf_disease
```

Approve a source only after reading its Markdown card and accepting external
terms:

```bash
python scripts/manage_dataset_sources.py approve cassava_leaf_disease
```

Download and prepare approved data:

```bash
python scripts/manage_dataset_sources.py download cassava_leaf_disease \
  --raw-root datasets/raw

python scripts/manage_dataset_sources.py prepare cassava_leaf_disease \
  --raw-root datasets/raw \
  --prepared-root datasets/prepared
```

Raw and prepared dataset bodies are ignored by git. The tracked artifacts are
the approval docs, registry status, subset masks, checksums, and metadata
summaries.

Nautilus-only candidates such as `imagenet_object_localization`,
`carvana_image_masking`, and `birdclef_2024` are refused by the local download
command unless `--allow-nautilus-only` is passed on a PVC-backed Nautilus job.

After generating model sources, create scheduler workload specs:

```bash
python nrp_calibration_pack/profile/make_workload_specs.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --registry dataset_sources/registry.json \
  --dataset-profile-root datasets/prepared \
  --output-dir nrp_calibration_pack/workload_specs \
  --subset-id tiny \
  --subset-id small \
  --batch-size 8 \
  --batch-size 32 \
  --precision-sweep fp32_ieee,bf16_amp \
  --optimizer adam \
  --hardware-id rtx5090 \
  --force
```

Profile those workloads with scheduler labels:

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --workload-specs nrp_calibration_pack/workload_specs/workloads.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090_scheduler \
  --hardware-id rtx5090 \
  --device cuda \
  --sm-occupancy-source nvml_proxy \
  --resource-profile-mode sustained \
  --min-phase-seconds 20 \
  --min-sampler-samples 100 \
  --label-time-mode measured_epochs \
  --time-label-warmup-epochs 1 \
  --time-label-measured-epochs 2 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

The profiler refuses scheduler workload rows whose dataset profile does not
declare a real dataloader adapter. Use `--allow-synthetic-workload-inputs` only
for smoke tests that intentionally exercise the schema without production
labels.

`label_v3` stores both the scheduler epoch label and the legacy step-derived
estimate. In measured-epoch mode, the scheduler label is:

```text
train_epoch_ms = mean(measured_epoch_wall_ms after warmup epochs)
```

The compatibility estimate remains available as:

```text
train_epoch_ms_step_extrapolated =
  train_step_wall_ms * ceil(num_samples / effective_batch_size)
```

Use `train_epoch_ms` when comparing labels against a real multi-epoch training
run. The legacy six-target `train_time`/`train.time` field is retained for
PerfSeer compatibility and should not be interpreted as a full epoch label.

Validate label reliability against 5-epoch golden runs on a local RTX 5090:

```bash
python scripts/validate_dataset_resource_labels.py \
  --workloads nrp_calibration_pack/workload_specs_balanced_local/workloads.jsonl \
  --models-dir nrp_calibration_pack/models \
  --hardware-id rtx5090 \
  --device cuda
```

The validator selects 20 workload rows across architecture families, profiles
them with sustained labels, runs matching 5-epoch golden training phases, and
writes comparison CSV, JSON, and Markdown summaries under
`record/resource_label_validation_<timestamp>/`.

## Generate Pack

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 10005 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack \
  --precision-sweep fp32_ieee \
  --validation-mode compile \
  --generation-workers "$(nproc)" \
  --force
```

Before a large scheduler-label run, validate the generated model structures:

```bash
python scripts/validate_generated_model_structures.py \
  --output-dir record/generated_model_structure_validation_smoke \
  --force
```

The verifier generates a representative pack, checks coverage across all 15
architecture families and variant kinds, then runs one forward/backward profile
smoke per generated model. Add `--cuda-if-available` for an RTX 5090 smoke or
`--full` for the heavier 10,005-base-model local sweep.

For local RTX 5090 FP8/NVFP4 label generation, make a transformer-focused pack
instead of starting with the broad CNN-first catalog:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 256 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack_te_transformer \
  --precision-sweep fp32_ieee \
  --validation-mode compile \
  --generation-workers "$(nproc)" \
  --low-precision-focus te_transformer \
  --force
```

`te_transformer` emits non-embedding transformer template families
(`vit_encoder`, `ast_audio_transformer`, `wav2vec2_audio`, and
`ft_transformer_tabular`) and verifies that each generated source passes both
the FP8 and NVFP4 Transformer Engine shape gates. The full catalog still
contains CNN, recurrent, graph, and message-passing models for baseline
precisions; those operator families are intentionally recorded as
`unsupported_low_precision_op` for FP8/NVFP4 in v1.

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force
```

## Create Hardware Labels

Profile the same pack once per hardware class; each result root has one hardware
ID. `--precision-sweep auto` resolves supported precisions after Compute Unified
Device Architecture device selection. Base precisions run wherever supported;
8-bit floating point requires Transformer Engine plus Ada/Hopper/Blackwell
probes; `nvfp4_te` requires Transformer Engine NVIDIA 4-bit floating point on
Blackwell-class hardware. `fp4` and `nvfp4` alias to `nvfp4_te`; `mxfp8` is out
of scope for v1.

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090 \
  --hardware-id rtx5090 \
  --precision-sweep auto \
  --profile-dataset-dir nrp_calibration_pack/profile_datasets \
  --device cuda \
  --sm-occupancy-source nvml_proxy \
  --resource-profile-mode sustained \
  --min-phase-seconds 20 \
  --min-sampler-samples 100 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --optimizer adam \
  --num-shards <N> \
  --shard-index <I>
```

Resume is default. Same `--output-dir`, `--num-shards`, and `--shard-index`
skip completed profile points by scanning `results_shard<I>.jsonl` and labels.
Use `--no-resume` to reprofile.

Low precision is explicit. Transformer Engine rewrites apply only to dense,
norm, and generated attention rows that pass shape gates: 8-bit floating point
needs 16-wide feature and leading-dimension alignment; NVIDIA 4-bit floating
point needs 32-wide feature alignment and leading dimension at least 32.
Convolutional, recurrent, graph, message-passing, embedding-heavy token
transformer, undersized, and unsupported mixes become
`unsupported_low_precision_op`, not silent 32-bit fallback.

Change only `--output-dir`, `--hardware-id`, and hardware/node affinity per
Graphics Processing Unit:

```text
nrp_results_rtx3090  -> --hardware-id rtx3090
nrp_results_rtx4090  -> --hardware-id rtx4090
nrp_results_rtx5090  -> --hardware-id rtx5090
```

Only `status == "ok"` rows become training labels. Unsupported, out-of-memory
and error rows go to `precision_rejected_rows.jsonl`.

## Source-First Nautilus Workflow

Build and push the profiling image:

```bash
docker build -f nrp_calibration_pack/Dockerfile -t <registry>/perfseer-ngc:latest .
docker push <registry>/perfseer-ngc:latest
```

Render Persistent Volume Claim-backed prepare, profile, and package jobs:

```bash
./nrp_calibration_pack/submit_nrp_source_workflow.sh \
  --namespace <namespace> \
  --image <registry>/perfseer-ngc:latest \
  --pvc <output-pvc> \
  --gpu-product NVIDIA-GeForce-RTX-5090 \
  --hardware-id rtx5090 \
  --parallelism 4 \
  --completions 64 \
  --dry-run
```

Submit one stage at a time: `--stage prepare`, wait, `--stage profile`, wait,
then `--stage package`. Package outputs:

For a 5090 FP8/NVFP4 transformer-focused Nautilus run, add
`--low-precision-focus te_transformer` to the prepare/source workflow command.

- `perfseer_<hardware_id>_source_labels.tar.gz`: `models/*.py`, manifests,
  profile specs, labels, hardware JSON, result JSON Lines, rejected rows,
  coverage reports, provenance, and `replay/` profiler/runtime scripts.
- `perfseer_<hardware_id>_dataset.tar.gz`: rebuilt `cg/cg/*.pkl`,
  `label/label/*.txt`, `label/precision_metadata.jsonl`, and rejected-row
  metadata.

One-command local runner:

```bash
python3 scripts/run_nrp_source_workflow_local.py \
  --namespace <namespace> \
  --image <registry>/perfseer-ngc:latest \
  --allow-mutable-image-tag \
  --pvc <output-pvc> \
  --gpus a100,a40,l4,rtx_a4000 \
  --hardware-id mixed4 \
  --completions 64 \
  --local-output-dir nrp_downloads
```

The runner is the recommended interface when the repository tree must be staged
into a Nautilus Persistent Volume Claim, when multiple Graphics Processing Unit
types should run in parallel, or when source-label and materialized dataset
tarballs should be copied back automatically.

Full dataset run from an Omen backend shell:

```bash
RUN_ID="perfseer-full-omen-$(date +%m%d%H%M%S)"
LOG="record/${RUN_ID}_driver.log"

nohup python3 -u scripts/run_nrp_source_workflow_local.py \
  --namespace ecepxie \
  --image pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel \
  --allow-mutable-image-tag \
  --utility-image alpine:3.20 \
  --pvc test-pvc \
  --job-prefix "${RUN_ID}" \
  --workflow-dir "/mnt/output/${RUN_ID}" \
  --hardware-id mixed4_full_omen \
  --stage-local-repo \
  --subset-size 10005 \
  --completions 64 \
  --parallelism 4 \
  --profile-scheduling-mode shard-switcher \
  --gpus a100,a40,l4,rtx_a4000 \
  --active-gpus 4 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --sample-interval 0.01 \
  --optimizer adam \
  --sm-occupancy-source nvml_proxy \
  --profile-precision-sweep auto \
  --bootstrap-command 'python -m pip install --no-cache-dir torch_geometric ogb networkx scikit-learn tqdm nvidia-ml-py pyyaml' \
  --local-output-dir "nrp_downloads/${RUN_ID}" \
  --timeout-seconds 604800 \
  --stage-timeout-seconds 1800 \
  --poll-seconds 60 \
  --kubectl-request-timeout 30s \
  --kubectl-hard-timeout-seconds 180 \
  > "${LOG}" 2>&1 &

echo "$!" > "record/${RUN_ID}.pid"
```

That command runs the complete 10,005-model balanced scheduler catalog. `nohup` means no hangup,
`python3 -u` means unbuffered output, `2>&1` redirects standard error to
standard output, and the trailing `&` backgrounds the process. The runner
creates prepare, profile, package, and download stages; stages the current Omen
repository into the Persistent Volume Claim; keeps the Omen process alive after
the Secure Shell session exits; and writes the source-label and dataset tarballs
under `nrp_downloads/${RUN_ID}` when the workflow finishes.

The same command can be used for a small timing smoke by changing these options:

```text
--subset-size 100
--completions 1
--parallelism 1
--gpus rtx_a6000
--active-gpus 1
--profile-precision-sweep fp32_ieee
--warmup 1
--infer-repeats 1
--train-repeats 1
```

The default `gpu-partition` scheduler creates exactly four profile Kubernetes
Jobs, one per `--gpus` preset. Each job requests one Graphics Processing Unit
kind, uses `parallelism: 1`, and receives a disjoint contiguous shard range; the
union of those ranges is `0..completions-1`. The `shard-switcher` scheduler
submits one shard job at a time up to `--active-gpus`, picks from the `--gpus`
preset list, and retries a shard on another Graphics Processing Unit preset when
the current job stays pending too long or fails for a retryable node reason.

If the image does not already contain this repository, stage the current local
tree into the Persistent Volume Claim first:

```bash
python3 scripts/run_nrp_source_workflow_local.py \
  --namespace <namespace> \
  --image pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel \
  --pvc <output-pvc> \
  --stage-local-repo \
  --gpus a100,a40,l4,rtx_a4000 \
  --hardware-id <hardware-id> \
  --local-output-dir nrp_downloads
```

The runner uses a short-lived download pod for copy-back because `kubectl cp`
requires a running container and cannot copy from a completed package pod.

Rebuild a dataset locally from a source-label tarball:

```bash
python scripts/rebuild_source_tar_dataset.py \
  --source-tar perfseer_rtx5090_source_labels.tar.gz \
  --out-root dataset_rtx5090_rebuilt \
  --force
```

Training path: `generate_model_sources.py` writes sources/manifests/coverage and
optional graph PKLs; `run_profile.py` profiles each source model on the target
Graphics Processing Unit with `--precision-sweep auto`; `package_source_tar.py`
writes the source-label package; `rebuild_source_tar_dataset.py` rebuilds
`dataset/cg/cg/*.pkl`; `run_nrp_source_workflow_local.py` copies both tarballs;
hardware-filtered teacher/student training reads `precision_metadata.jsonl` and
splits labels matching `--hardware-id`.

## Train Per Hardware

Train one v2 teacher/student pair per hardware ID. The active model configs are
only:

- `src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml`
- `src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml`

Older architecture configs are under `src/perfseer-optimized/configs/legacy/`
and should not be used for new v2 runs.
See `doc/v2_teacher_student_model_pair.md` for the current architecture summary.

```bash
python scripts/run_hardware_distill_flow.py \
  --data-root dataset \
  --hardware-id rtx5090 \
  --split-unit graph
```

Useful individual commands:

```bash
python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml \
  --data-root dataset \
  --hardware-id rtx5090 \
  --run-id v2_teacher_rtx5090

python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml \
  --data-root dataset \
  --hardware-id rtx5090 \
  --teacher-ckpt-dir runs/optimized/v2_teacher_rtx5090 \
  --run-id v2_student_rtx5090
```

Both configs set `features.target_source: scheduler_v2_train` and
`features.target_mode: absolute`. Rows missing from either
`label/scheduler_label_v3.jsonl` or `label/scheduler_resource_label.jsonl` fail
loudly so the v2 pair cannot silently train on stale legacy labels.

Rerun with `--hardware-id rtx3090` and `--hardware-id rtx5090` for other GPUs.
Each hardware model can learn from every accepted precision recipe for that
hardware.

## Folder Label Sampling

Use this auxiliary path for a local folder of PyTorch model source files instead
of the generated calibration pack. It copies labels back, not a full materialized
dataset. Each `.py` file must define `make_model()`; `MODEL_ID` and
`INPUT_SHAPE` are optional. The namespace and Persistent Volume Claim below are
examples.

```text
local model folder
-> build manifest
-> upload models, manifest, profiler, and verifier to Nautilus Persistent Volume Claim
-> submit one-Graphics Processing Unit jobs with switching, up to --active-gpus concurrent jobs
-> generate labels
-> verify labels
-> copy remote labels back to local labels/<run_id>/
```

```bash
python3 scripts/run_nautilus_folder_label_sampling.py \
  --models-dir /path/to/pytorch_model_files \
  --local-labels-dir labels \
  --namespace ecepxie \
  --pvc test-pvc \
  --gpus all-readme \
  --active-gpus 4 \
  --pending-timeout-seconds 300 \
  --min-successful-gpus 1
```

```bash
python3 scripts/verify_sampled_labels.py labels/<run_id>
```

## Validation

```bash
python -m py_compile \
  nrp_calibration_pack/build_pack.py \
  nrp_calibration_pack/generate_model_sources.py \
  nrp_calibration_pack/profile/generated_model_runtime.py \
  nrp_calibration_pack/profile/make_profile_datasets.py \
  nrp_calibration_pack/profile/run_profile.py \
  nrp_calibration_pack/workload.py \
  nrp_calibration_pack/package_source_tar.py \
  nrp_calibration_pack/template_catalog.py \
  scripts/validate_dataset_resource_labels.py \
  scripts/rebuild_source_tar_dataset.py \
  scripts/run_nrp_source_workflow_local.py \
  scripts/run_hardware_distill_flow.py \
  src/perfseer/architecture_schema.py \
  src/perfseer-optimized/data.py \
  src/perfseer-optimized/train.py \
  src/perfseer-optimized/eval.py \
  src/perfseer_source_converter/converter.py

python -m unittest scripts.test_nrp_calibration_pack scripts.test_source_converter -v
git diff --check
git ls-files -ci --exclude-standard
```

Tiny Central Processing Unit smoke:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 15 \
  --seed 20260617 \
  --out-dir /tmp/perfseer_smoke_pack \
  --precision-sweep fp32_ieee \
  --validation-mode compile \
  --force

python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest /tmp/perfseer_smoke_pack/manifest/subset_manifest.jsonl \
  --output-dir /tmp/perfseer_smoke_pack/profile_datasets \
  --train-repeats 1 \
  --infer-repeats 1 \
  --force

python /tmp/perfseer_smoke_pack/profile/run_profile.py \
  --manifest /tmp/perfseer_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/perfseer_smoke_pack/models \
  --output-dir /tmp/perfseer_smoke_results \
  --hardware-id cpu_smoke \
  --precision-sweep fp32_ieee \
  --profile-dataset-dir /tmp/perfseer_smoke_pack/profile_datasets \
  --device cpu \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1
```
