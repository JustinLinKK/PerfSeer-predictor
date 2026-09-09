# Candidate Dataset: ImageNet Object Localization

- Dataset id: `imagenet_object_localization`
- Target tier: Nautilus only
- Local replacement: `cassava_leaf_disease`
- Task family: image classification
- Covered model families: `resnet_cnn`, `efficientnet_cnn`, `vgg_cnn`, `vit_encoder`
- Source: https://www.kaggle.com/c/imagenet-object-localization-challenge
- Kaggle slug: `imagenet-object-localization-challenge`
- Source type: Kaggle competition
- Approval status in registry: `approved`

## Why This Dataset

This is the PVC-backed large image-classification reference candidate. It lets
multiple CNN and ViT-like model structures share one real task dataset while
dataset-size variation is represented by deterministic subset masks, but the
download/extracted footprint is too large for the default local workflow.

Use `cassava_leaf_disease` for local RTX 5090 development.

## Download Command

```bash
kaggle competitions download -c imagenet-object-localization-challenge -p datasets/raw/imagenet_object_localization
```

## Preparation Plan

- Extract train images and labels after you accept the Kaggle competition terms.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve class balance across subset masks.
- Emit metadata summary with sample count, class count, image-size statistics,
  sample-byte statistics, preprocessing settings, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm Nautilus PVC storage budget is acceptable.
- Mark this dataset `approved` in `dataset_sources/registry.json` or run
  `python scripts/manage_dataset_sources.py approve imagenet_object_localization`.
