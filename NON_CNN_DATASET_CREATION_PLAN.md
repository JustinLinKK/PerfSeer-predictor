# Non-CNN PerfSeer Dataset Creation Plan

## Summary

PerfSeer keeps SeerNet as the predictor architecture because it already predicts from compute graphs. The non-CNN limitation is addressed by adding a v2 graph feature schema, a deterministic repo-local model-template catalog, and multi-input profiling support.

The new catalog mode is:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template_v2 \
  --subset-size 10000 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack_noncnn \
  --precision-sweep fp32_ieee \
  --generation-workers "$(nproc)" \
  --force
```

The profiled dataset root should be materialized separately as `dataset_noncnn_10000/` from successful profiler outputs so the original source dataset remains unchanged.

## Model And Feature Changes

- Use `FeatureConfig(feature_schema_version="template_v2_noncnn")` for the non-CNN dataset; legacy configs keep the old CNN-compatible dimensions.
- The v2 schema expands operator one-hot coverage to CNNs, transformers, recurrent models, graph message-passing models, detector/segmentation heads, audio models, and tabular transformers.
- V2 node features add generic tensor-shape channels: rank, input/output size, feature dimensions, sequence length, spatial area, and graph node count.
- V2 global features add architecture family, modality, variant kind, and depth/width buckets from graph metadata.
- V2 checkpoints are intentionally incompatible with legacy checkpoints and should be retrained from scratch.

## Catalog Quotas

The 10,000 base-model catalog uses these exact quotas:

| family | count |
|---|---:|
| resnet_cnn | 2160 |
| efficientnet_cnn | 1520 |
| bert_encoder | 1200 |
| vit_encoder | 1200 |
| yolo_detector | 560 |
| unet_encoder_decoder | 480 |
| ast_audio_transformer | 480 |
| gru_temporal | 480 |
| lstm_temporal | 320 |
| vgg_cnn | 400 |
| gat_graph | 240 |
| mpnn_graph | 240 |
| t5_encoder_decoder | 240 |
| wav2vec2_audio | 240 |
| ft_transformer_tabular | 240 |

Each family uses the fixed mutation mix: 10% canonical, 30% added depth, 30% dropped depth, 20% width/shape, and 10% mixed stress.

## Validation

Run a small local real-mode smoke first:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template_v2 \
  --subset-size 30 \
  --out-dir /tmp/perfseer_noncnn_smoke_pack \
  --precision-sweep fp32_ieee \
  --validation-mode real \
  --generation-workers 1 \
  --force

python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest /tmp/perfseer_noncnn_smoke_pack/manifest/subset_manifest.jsonl \
  --output-dir /tmp/perfseer_noncnn_smoke_pack/profile_datasets \
  --train-repeats 1 \
  --infer-repeats 1 \
  --force

python /tmp/perfseer_noncnn_smoke_pack/profile/run_profile.py \
  --manifest /tmp/perfseer_noncnn_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/perfseer_noncnn_smoke_pack/models \
  --output-dir /tmp/perfseer_noncnn_smoke_results \
  --num-shards 1 \
  --precision-config fp32_ieee \
  --profile-dataset-dir /tmp/perfseer_noncnn_smoke_pack/profile_datasets \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1 \
  --device cpu
```

Acceptance gates:

- Manifest includes `input_specs`, `feature_schema_version`, `architecture_family`, `variant_kind`, and `variant_signature`.
- Coverage summary reports the requested architecture families and v2 operators.
- CPU smoke labels parse through the existing dataset label parser into six targets.
- A v2 `FeatureConfig` can build normalized PyG tensors and run a SeerNet forward/backward pass.
