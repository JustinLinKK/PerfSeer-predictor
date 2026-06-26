# Candidate Dataset: Carvana Image Masking

- Dataset id: `carvana_image_masking`
- Target tier: Nautilus only
- Local replacement: `siim_acr_pneumothorax`
- Task family: image segmentation
- Covered model families: `unet_encoder_decoder`
- Source: https://www.kaggle.com/c/carvana-image-masking-challenge
- Kaggle slug: `carvana-image-masking-challenge`
- Source type: Kaggle competition
- Approval status in registry: `approved`

## Why This Dataset

This remains a strong segmentation reference candidate for U-Net-style
workloads, but the full download/extracted footprint is better suited to a
Nautilus PVC than the default local workflow.

Use `siim_acr_pneumothorax` for local RTX 5090 development.

## Download Command

```bash
kaggle competitions download -c carvana-image-masking-challenge -p datasets/raw/carvana_image_masking
```

## Preparation Plan

- Extract train images and mask files after you accept the Kaggle rules.
- Pair every image with its mask.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve image-resolution and mask-size distributions.
- Emit metadata summary with image-size statistics, mask shape, bytes, and
  subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm Nautilus PVC storage budget is acceptable.
- Mark this dataset `approved` before running any download.
