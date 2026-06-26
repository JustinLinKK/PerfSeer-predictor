# Candidate Dataset: CNN/DailyMail Summarization

- Dataset id: `cnn_dailymail_summarization`
- Target tier: local RTX 5090
- Task family: seq2seq text
- Covered model families: `t5_encoder_decoder`
- Source: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
- Kaggle slug: `gowrishankarp/newspaper-text-summarization-cnn-dailymail`
- Source type: Kaggle dataset
- Expected storage: well under 10 GB
- Approval status in registry: `approved`

## Why This Dataset

This is the primary seq2seq candidate for T5-style encoder-decoder workloads.
It gives the predictor real input and target sequence-length distributions.

## Download Command

```bash
kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail -p datasets/raw/cnn_dailymail_summarization
```

## Preparation Plan

- Download after reviewing the Kaggle dataset terms and license metadata.
- Tokenize articles and summaries with the chosen seq2seq tokenizer.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve source and target sequence-length percentiles.
- Emit metadata summary with source/target token statistics, bytes, and subset
  checksums.

## Approval Checklist

- Confirm Kaggle dataset terms/license are acceptable.
- Confirm tokenizer choice is acceptable for scheduler profiling.
- Mark this dataset `approved` before running any download.
