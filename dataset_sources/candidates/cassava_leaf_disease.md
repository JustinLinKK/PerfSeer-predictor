# Candidate Dataset: Cassava Leaf Disease

- Dataset id: `cassava_leaf_disease`
- Target tier: local RTX 5090
- Task family: image classification
- Covered model families: `resnet_cnn`, `efficientnet_cnn`, `vgg_cnn`, `vit_encoder`
- Source: https://www.kaggle.com/c/cassava-leaf-disease-classification
- Kaggle slug: `cassava-leaf-disease-classification`
- Source type: Kaggle competition
- Expected storage: under 10 GB compressed; extracted image cache may be larger
- Approval status in registry: `pending`

## Why This Dataset

This is the local image-classification replacement for ImageNet Object
Localization. It is large enough to exercise real image decoding, augmentation,
classification labels, and CNN/ViT batch behavior, while staying inside a
workstation storage budget.

## Download Command

```bash
kaggle competitions download -c cassava-leaf-disease-classification -p datasets/raw/cassava_leaf_disease
```

## Preparation Plan

- Extract train images and `train.csv` after accepting Kaggle competition terms.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve disease-label class balance across subset masks.
- Emit metadata summary with sample count, class count, image-size statistics,
  sample-byte statistics, preprocessing settings, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm local storage budget is acceptable.
- Mark this dataset `approved` in `dataset_sources/registry.json` or run
  `python scripts/manage_dataset_sources.py approve cassava_leaf_disease`.
