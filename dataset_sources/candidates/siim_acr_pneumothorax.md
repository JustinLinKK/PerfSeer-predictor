# Candidate Dataset: SIIM-ACR Pneumothorax Segmentation

- Dataset id: `siim_acr_pneumothorax`
- Target tier: local RTX 5090
- Task family: image segmentation
- Covered model families: `unet_encoder_decoder`
- Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
- Kaggle slug: `siim-acr-pneumothorax-segmentation`
- Source type: Kaggle competition
- Expected storage: low single-digit GB compressed; DICOM preprocessing cache may be larger
- Approval status in registry: `pending`

## Why This Dataset

This is the local segmentation replacement for Carvana. It provides real
medical images and mask targets for U-Net-style workloads without requiring the
large extracted image/mask footprint of the Carvana challenge on a workstation.

## Download Command

```bash
kaggle competitions download -c siim-acr-pneumothorax-segmentation -p datasets/raw/siim_acr_pneumothorax
```

## Preparation Plan

- Extract DICOM images and mask annotations after accepting Kaggle rules.
- Pair each image with its encoded segmentation mask or no-mask target.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve positive/negative mask balance and image-resolution buckets.
- Emit metadata summary with image-size statistics, mask density statistics,
  bytes, preprocessing policy, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm local storage and medical-data handling expectations are acceptable.
- Mark this dataset `approved` before running any download.
