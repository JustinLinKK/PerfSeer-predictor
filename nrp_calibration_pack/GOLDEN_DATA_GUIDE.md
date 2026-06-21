# Calibration Pack Guide

This package generates the canonical PerfSeer calibration pack and profiles it
on one hardware class at a time.

V1 keeps the pack source-first. The prepare stage writes deterministic Python
model sources and manifests, the profile stage expands `--precision-sweep auto`
on the selected GPU, and the package stage downloads only sources, labels, and
metadata. Generated PKLs are rebuilt later from the source tarball for audit.

## Generate Sources

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

## Build Profile Specs

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force
```

## Profile A Hardware Shard

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090 \
  --hardware-id rtx5090 \
  --precision-sweep auto \
  --profile-dataset-dir nrp_calibration_pack/profile_datasets \
  --device cuda \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

Profiling has a built-in checkpoint: rerun the same command with the same
`--output-dir`, `--num-shards`, and `--shard-index` to continue after a pause,
manual interrupt, or job eviction. The profiler skips rows only when both the
completed result row and label file already exist; any half-finished row is
retried. Use `--no-resume` to start a shard over.

Use a separate result directory for each hardware ID, such as
`nrp_results_rtx3090`, `nrp_results_rtx4090`, and `nrp_results_rtx5090`.

`auto` is resolved inside the profiler after CUDA device selection. It includes
base precisions wherever supported, FP8 only when Transformer Engine reports FP8
on Ada/Hopper/Blackwell-class hardware, and canonical `nvfp4_te` only when
Transformer Engine reports NVFP4 on Blackwell-class hardware. `fp4` and `nvfp4`
are accepted aliases. `mxfp8` remains out of scope for v1.

Low precision is enabled only for generated dense, norm, and attention rows that
pass the TE rewrite and shape gates. FP8 requires 16-wide feature and leading
alignment; NVFP4 requires 32-wide feature alignment and a leading dimension of
at least 32. Unsupported conv/RNN/graph/message-passing mixes and undersized
low-precision shapes are recorded as `unsupported_low_precision_op`, never
silently profiled as FP32.

## Nautilus Source Workflow

Build the NGC-based image from the repo root:

```bash
docker build -f nrp_calibration_pack/Dockerfile -t <registry>/perfseer-ngc:latest .
docker push <registry>/perfseer-ngc:latest
```

Render the three jobs:

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

Submit one stage at a time: `--stage prepare`, wait for completion, then
`--stage profile`, then `--stage package`. The package tarball includes
`models/*.py`, manifests, profile specs, labels, hardware JSON, results JSONL,
rejected rows, reports, provenance, and a `replay/` copy of the profiler/runtime
scripts. It excludes `subset/cg/cg/*.pkl`, generated PyG PKLs, caches, and
checkpoints.

## Rebuild Or Materialize Labels

For source-tar audit rebuild:

```bash
python scripts/rebuild_source_tar_dataset.py \
  --source-tar perfseer_rtx5090_source_labels.tar.gz \
  --out-root dataset_rtx5090_rebuilt \
  --force
```

For local result directories:

```bash
python scripts/materialize_precision_dataset.py \
  --pack-dir nrp_calibration_pack \
  --results-dir nrp_results_rtx3090 \
  --results-dir nrp_results_rtx4090 \
  --results-dir nrp_results_rtx5090 \
  --out-root dataset \
  --force
```

Only successful profiler rows become labels. Rejected precision rows, OOMs, and
errors are retained in `precision_rejected_rows.jsonl`.

Training then uses the existing hardware-filtered flow. The input to model
design is the Python source pack; `rebuild_source_tar_dataset.py` regenerates
`dataset/cg/cg/*.pkl` from those sources, and training reads
`label/precision_metadata.jsonl` with `--hardware-id` to keep teacher/student
runs scoped to one measured hardware class.
