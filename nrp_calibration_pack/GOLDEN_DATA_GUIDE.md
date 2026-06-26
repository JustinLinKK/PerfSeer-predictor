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

For a local RTX 5090 FP8/NVFP4 transformer sweep, generate a TE-focused source
pack:

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

This focus mode keeps v1 low precision on non-embedding transformer rows whose
dense, norm, and generated attention operators satisfy both FP8 and NVFP4
Transformer Engine gates. The broad catalog still includes CNN, recurrent,
graph, message-passing, and token-embedding transformer rows for baseline
precisions, but those are explicit `unsupported_low_precision_op` rows for
FP8/NVFP4 until separate runtime rewrites are implemented.

## Build Profile Specs

Legacy synthetic/repeat specs remain available for compatibility:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force
```

For scheduler-grade labels, use approved real dataset profiles instead of random
input specs. Review the local tier first, then approve each dataset source
before downloading:

```bash
python scripts/manage_dataset_sources.py list --tier local
python scripts/manage_dataset_sources.py show cassava_leaf_disease
python scripts/manage_dataset_sources.py approve cassava_leaf_disease
```

Download and prepare deterministic real-data subsets:

```bash
python scripts/manage_dataset_sources.py download cassava_leaf_disease \
  --raw-root datasets/raw

python scripts/manage_dataset_sources.py prepare cassava_leaf_disease \
  --raw-root datasets/raw \
  --prepared-root datasets/prepared
```

Large/reference sources are in the `nautilus` tier. The local download path
refuses them unless `--allow-nautilus-only` is passed inside a PVC-backed
Nautilus job.

Build scheduler workload specs:

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

## Profile A Hardware Shard

Legacy profile-spec path:

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

Scheduler workload path:

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --workload-specs nrp_calibration_pack/workload_specs/workloads.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090_scheduler \
  --hardware-id rtx5090 \
  --precision-sweep auto \
  --device cuda \
  --sm-occupancy-source nvml_proxy \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

Scheduler runs write both legacy PerfSeer-compatible label files and
`label_v3_shard<I>.jsonl`. `label_v3` derives the default epoch label from the
real subset size:

```text
train_epoch_ms = train_step_wall_ms * ceil(num_samples / effective_batch_size)
```

The profiler rejects scheduler workload rows that do not declare a real
dataloader adapter. `--allow-synthetic-workload-inputs` exists only for schema
smoke tests, not production labels.

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
at least 32. Unsupported conv/RNN/graph/message-passing mixes,
embedding-heavy token transformers, and undersized low-precision shapes are
recorded as `unsupported_low_precision_op`, never silently profiled as FP32.

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

For a 5090 FP8/NVFP4 transformer-focused Nautilus run, add
`--low-precision-focus te_transformer` to the source workflow command.

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
