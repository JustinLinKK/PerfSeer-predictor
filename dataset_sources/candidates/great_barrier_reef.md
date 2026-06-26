# Candidate Dataset: TensorFlow Great Barrier Reef

- Dataset id: `great_barrier_reef`
- Target tier: local RTX 5090
- Task family: object detection
- Covered model families: `yolo_detector`
- Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef
- Kaggle slug: `tensorflow-great-barrier-reef`
- Source type: Kaggle competition
- Expected storage: around 20-30 GB download/extract budget
- Approval status in registry: `pending`

## Why This Dataset

This is the local object-detection primary candidate. It is closer to the
requested 20-30 GB local budget than Global Wheat while still offering real
images, annotations, object-count variation, and YOLO-style target tensors.

## Download Command

```bash
kaggle competitions download -c tensorflow-great-barrier-reef -p datasets/raw/great_barrier_reef
```

## Preparation Plan

- Extract video-frame images and training annotations after accepting Kaggle
  competition terms.
- Convert bounding boxes to the detector adapter format.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve frame/video distribution, image-resolution buckets, and object-count
  buckets.
- Emit metadata summary with object-count percentiles, image-size statistics,
  target bytes, preprocessing settings, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm local storage budget is acceptable.
- Mark this dataset `approved` before running any download.
