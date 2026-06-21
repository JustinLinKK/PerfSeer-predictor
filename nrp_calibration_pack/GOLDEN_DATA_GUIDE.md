# Calibration Pack Guide

This package generates the canonical PerfSeer calibration pack and profiles it
on one hardware class at a time.

## Generate Sources

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 10000 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
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
  --output-dir nrp_results_rtx4090 \
  --hardware-id rtx4090 \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
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

## Materialize Labels

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
