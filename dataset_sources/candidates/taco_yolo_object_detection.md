# Candidate Dataset: TACO YOLO Object Detection

- Source: Kaggle dataset `vencerlanz09/taco-dataset-yolo-format`
- Source type: Kaggle dataset
- Task family: object detection
- Modality: image
- Model families: `yolo_detector`
- Selected replacement for: Kaggle competition `tensorflow-great-barrier-reef`

## Reason

This dataset is a public Kaggle dataset packaged for You Only Look Once object
detection. Its archive contains image files, label text files, and `data.yaml`,
so it is a better fit for the `yolo_detector` model family than a generic image
object-detection archive. It replaces the gated Great Barrier Reef competition
source because the current Kaggle credentials returned Hypertext Transfer
Protocol `403 Forbidden` for that competition during the Nautilus prepare job.

## Verification

The dataset was verified by listing files and downloading the archive through the
same Kaggle Application Programming Interface credential path used by the
workflow:

```bash
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets files vencerlanz09/taco-dataset-yolo-format
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets download -d vencerlanz09/taco-dataset-yolo-format -p /tmp/perfseer_kaggle_check_taco --force
```

Observed archive:

```text
/tmp/perfseer_kaggle_check_taco/taco-dataset-yolo-format.zip
```

The listed files include `data.yaml`, `train`, `valid`, and `test` image and
label directories.

## Preparation Notes

The generic preparation path in `scripts/manage_dataset_sources.py` samples
image members from the downloaded ZIP archive for subset masks. It does not
validate bounding-box label pairing; that should be added later if the scheduler
training path needs class-level detector supervision. The current label-sampling
workflow only needs a real dataset profile, subset masks, and image-backed input
metadata for `yolo_detector` workloads.
