# Candidate Dataset: Animal Audio Classification

- Source: Kaggle dataset `warcoder/cats-vs-dogs-vs-birds-audio-classification`
- Source type: Kaggle dataset
- Task family: audio classification
- Modality: audio
- Model families: `ast_audio_transformer`, `wav2vec2_audio`
- Selected replacement for: Kaggle competition `birdclef-2023`

## Reason

This dataset is a public Kaggle audio classification dataset with Waveform Audio
File Format files. It replaces the gated BirdCLEF 2023 competition source
because the current Kaggle credentials returned Hypertext Transfer Protocol
`403 Forbidden` for that competition during the Nautilus prepare job.

## Verification

The dataset was verified by listing files and downloading the archive through the
same Kaggle Application Programming Interface credential path used by the
workflow:

```bash
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets files warcoder/cats-vs-dogs-vs-birds-audio-classification
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets download -d warcoder/cats-vs-dogs-vs-birds-audio-classification -p /tmp/perfseer_kaggle_check_audio --force
```

Observed archive:

```text
/tmp/perfseer_kaggle_check_audio/cats-vs-dogs-vs-birds-audio-classification.zip
```

The archive contains 610 `.wav` files under animal class folders, which the
generic preparation path can discover with its audio suffix filter.

## Preparation Notes

The generic preparation path in `scripts/manage_dataset_sources.py` samples
audio members from the ZIP archive for subset masks. It does not validate class
balance from directory names. The current label-sampling workflow needs
real-file-backed audio input metadata and subset masks for audio model families.
