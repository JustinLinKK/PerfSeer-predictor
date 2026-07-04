# Pothole Image Segmentation Dataset

- Source: Kaggle dataset `farzadnekouei/pothole-image-segmentation-dataset`
- Source type: Kaggle dataset
- Task family: image segmentation
- Modality: image
- Target tier: local
- Estimated storage: under 1 GB compressed
- Model families: `unet_encoder_decoder`
- Reason selected: public Kaggle dataset download succeeded through Kaggle Application Programming Interface credentials, unlike gated competition segmentation datasets that returned Hypertext Transfer Protocol `403 Forbidden`.

## Verification

```bash
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets files farzadnekouei/pothole-image-segmentation-dataset
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets download -d farzadnekouei/pothole-image-segmentation-dataset -p /tmp/perfseer_kaggle_check_pothole --force
```
