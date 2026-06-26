# Candidate Dataset: BirdCLEF 2023

- Dataset id: `birdclef_2023`
- Target tier: local RTX 5090
- Task family: audio classification
- Covered model families: `ast_audio_transformer`, `wav2vec2_audio`
- Source: https://www.kaggle.com/c/birdclef-2023
- Kaggle slug: `birdclef-2023`
- Source type: Kaggle competition
- Expected storage: single-digit GB download; prepared spectrogram/token caches may be larger
- Approval status in registry: `pending`

## Why This Dataset

This is the local audio-classification replacement for BirdCLEF 2024. It keeps
real audio duration, decoding, and class-distribution behavior in the profiling
loop while avoiding the larger 2024 footprint for local development.

## Download Command

```bash
kaggle competitions download -c birdclef-2023 -p datasets/raw/birdclef_2023
```

## Preparation Plan

- Extract audio and labels after accepting Kaggle competition terms.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve class balance and audio-duration percentiles.
- Emit metadata summary with clip-duration statistics, decoded sample-rate
  policy, spectrogram/tokenizer settings, bytes, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm storage and preprocessing cost are acceptable.
- Mark this dataset `approved` before running any download.
