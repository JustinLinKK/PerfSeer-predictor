# Candidate Dataset: Jigsaw Toxic Comment Classification

- Dataset id: `jigsaw_toxic_comment`
- Target tier: local RTX 5090
- Task family: text classification
- Covered model families: `bert_encoder`
- Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Kaggle slug: `jigsaw-toxic-comment-classification-challenge`
- Source type: Kaggle competition
- Expected storage: well under 10 GB
- Approval status in registry: `approved`

## Why This Dataset

This is the primary BERT-style text-classification candidate. It gives the
predictor real token-length distributions and multi-label target structure.

## Download Command

```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p datasets/raw/jigsaw_toxic_comment
```

## Preparation Plan

- Extract CSV files after Kaggle terms are accepted.
- Tokenize with the chosen scheduler benchmark tokenizer during preparation.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve label prevalence and sequence-length percentiles.
- Emit metadata summary with token-length statistics, label dimensions, bytes,
  and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm tokenizer choice is acceptable for scheduler profiling.
- Mark this dataset `approved` before running any download.
