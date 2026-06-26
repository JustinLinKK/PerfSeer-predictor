# Candidate Dataset: Global Wheat Detection

- Dataset id: `global_wheat_detection`
- Target tier: Nautilus/reference
- Local replacement: `great_barrier_reef`
- Task family: object detection
- Covered model families: `yolo_detector`
- Source: https://www.kaggle.com/c/global-wheat-detection
- Kaggle slug: `global-wheat-detection`
- Source type: Kaggle competition
- Approval status in registry: `approved`

## Why This Dataset

This remains a useful object-detection reference candidate for YOLO-style
detector workloads, but the local primary is now `great_barrier_reef` because it
is closer to the desired local storage/stress profile.

## Download Command

```bash
kaggle competitions download -c global-wheat-detection -p datasets/raw/global_wheat_detection
```

## Preparation Plan

- Extract images and annotations after Kaggle terms are accepted.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve image-resolution and object-count buckets.
- Emit metadata summary with object-count percentiles, image-size statistics,
  target bytes, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm Nautilus/reference-run storage budget is acceptable.
- Mark this dataset `approved` before running any download.
