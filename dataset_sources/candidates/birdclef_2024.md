# Candidate Dataset: BirdCLEF 2024

- Dataset id: `birdclef_2024`
- Target tier: Nautilus only
- Local replacement: `birdclef_2023`
- Task family: audio classification
- Covered model families: `ast_audio_transformer`, `wav2vec2_audio`
- Source: https://www.kaggle.com/c/birdclef-2024
- Kaggle slug: `birdclef-2024`
- Source type: Kaggle competition
- Approval status in registry: `approved`

## Why This Dataset

This is the PVC-backed audio reference candidate for AST and wav2vec-style
workloads. It lets the profiler capture real audio duration and
spectrogram/tokenization costs, but the footprint is too large for the default
local workflow.

Use `birdclef_2023` for local RTX 5090 development.

## Download Command

```bash
kaggle competitions download -c birdclef-2024 -p datasets/raw/birdclef_2024
```

## Preparation Plan

- Extract audio and labels after Kaggle terms are accepted.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve class balance and audio-duration percentiles.
- Emit metadata summary with clip-duration statistics, decoded sample-rate
  policy, spectrogram settings, bytes, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm Nautilus PVC storage and preprocessing cost are acceptable.
- Mark this dataset `approved` before running any download.
