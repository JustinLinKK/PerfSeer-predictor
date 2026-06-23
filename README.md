# PerfSeer Predictor

PerfSeer predicts training and inference performance from compute graphs. This
expanded-catalog branch keeps SeerNet as a graph predictor while extending the
dataset, feature schema, profiler, and training flow to convolutional,
transformer, recurrent, graph, audio, detector, segmentation, and tabular model
families. Git tracks source, tests, configs, and this README; generated packs,
profiler results, datasets, checkpoints, and smoke outputs are ignored.

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

## Model Input And Output

Training object: PyTorch Geometric `Data`.

- `data.x`: node features; `data.edge_index`: directed graph edges.
- `data.edge_attr`: edge tensor summaries, optional destination tensor,
  edge-topology features.
- `data.u`: graph-level aggregate, architecture, precision, and label-domain
  features.
- `data.y`: standardized six-target label; `data.y_raw`: raw six-target label.

Model outputs, in order:

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

## Repository Layout

- `src/perfseer/`: shared schema plus original parser/model utilities.
- `src/perfseer-optimized/`: training, evaluation, distillation, and deployment
  package, imported as `perfseer_optimized`.
- `src/perfseer_source_converter/`: source-to-graph conversion.
- `nrp_calibration_pack/`: template catalog generator, generated-model runtime,
  profiler, profile-dataset builder, Dockerfile, and National Research Platform
  submit wrapper.
- `scripts/rebuild_source_tar_dataset.py`: rebuilds `dataset/cg/cg` and
  `dataset/label/label` from a source-label package.
- `scripts/run_nrp_source_workflow_local.py`: submits the source-first Nautilus
  workflow, waits for stages, and downloads source-label and dataset packages.
- `scripts/run_hardware_distill_flow.py`: scratch teacher training followed by
  same-hardware student distillation.

## Dataset

Canonical generated pack: `nrp_calibration_pack/`. Canonical materialized
training dataset: `dataset/`.

```text
dataset/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  precision_materialization_report.json
  precision_rejected_rows.jsonl
```

The deterministic repo-local catalog has 10,000 templates and no heavyweight
model-library imports such as torchvision, timm, or transformers.

| Family | Count |
| --- | ---: |
| `resnet_cnn` | 2160 |
| `efficientnet_cnn` | 1520 |
| `bert_encoder` | 1200 |
| `vit_encoder` | 1200 |
| `yolo_detector` | 560 |
| `unet_encoder_decoder` | 480 |
| `ast_audio_transformer` | 480 |
| `gru_temporal` | 480 |
| `lstm_temporal` | 320 |
| `vgg_cnn` | 400 |
| `gat_graph` | 240 |
| `mpnn_graph` | 240 |
| `t5_encoder_decoder` | 240 |
| `wav2vec2_audio` | 240 |
| `ft_transformer_tabular` | 240 |

Variant mix per family: 10% canonical anchors, 30% added-depth, 30%
dropped-depth, 20% width/shape/hyperparameter changes, 10% mixed stress.
Manifest fields: `model_id`, `architecture_family`, `variant_kind`,
`variant_signature`, `input_specs`, `feature_schema_version`, `model_file`,
`subset_graph_file`, `precision_config`, `profile_point_id`, label paths.

Acceptance gates: 10,000 rows in
`nrp_calibration_pack/manifest/subset_manifest.jsonl`; family quotas match the
table; every row has architecture, variant, input spec, schema, precision, and
model path metadata; unsupported operator coverage is zero.

## Generate Pack

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 10000 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack \
  --precision-sweep fp32_ieee \
  --validation-mode compile \
  --generation-workers "$(nproc)" \
  --force
```

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
  --subset-size 10000 \
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
  --bootstrap-command 'python -m pip install --no-cache-dir torch_geometric networkx scikit-learn tqdm nvidia-ml-py pyyaml' \
  --local-output-dir "nrp_downloads/${RUN_ID}" \
  --timeout-seconds 604800 \
  --stage-timeout-seconds 1800 \
  --poll-seconds 60 \
  --kubectl-request-timeout 30s \
  --kubectl-hard-timeout-seconds 180 \
  > "${LOG}" 2>&1 &

echo "$!" > "record/${RUN_ID}.pid"
```

That command runs the complete 10,000-model catalog. `nohup` means no hangup,
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

Train a large teacher from scratch and distill the matching student once per
hardware ID.

```bash
python scripts/run_hardware_distill_flow.py \
  --data-root dataset \
  --hardware-id rtx4090 \
  --teacher-epochs 600 \
  --student-epochs 500 \
  --split-unit graph
```

Useful individual commands:

```bash
python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_hardware_teacher/large_teacher.yaml \
  --data-root dataset \
  --hardware-id rtx4090 \
  --run-id hardware_large_teacher_rtx4090

python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/distill_student_128.yaml \
  --data-root dataset \
  --hardware-id rtx4090 \
  --teacher-ckpt-dir runs/optimized/hardware_large_teacher_rtx4090 \
  --run-id hardware_distill_student_128_rtx4090
```

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
  nrp_calibration_pack/package_source_tar.py \
  nrp_calibration_pack/template_catalog.py \
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
