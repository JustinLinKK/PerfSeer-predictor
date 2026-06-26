# Candidate Dataset: Home Credit Default Risk

- Dataset id: `home_credit_default_risk`
- Target tier: local RTX 5090
- Task family: tabular
- Covered model families: `ft_transformer_tabular`
- Source: https://www.kaggle.com/c/home-credit-default-risk
- Kaggle slug: `home-credit-default-risk`
- Source type: Kaggle competition
- Expected storage: well under 10 GB
- Approval status in registry: `approved`

## Why This Dataset

This is the primary tabular candidate for FT-Transformer-style workloads. It
provides real numeric/categorical feature distributions and missing-value
patterns.

## Download Command

```bash
kaggle competitions download -c home-credit-default-risk -p datasets/raw/home_credit_default_risk
```

## Preparation Plan

- Extract CSV files after Kaggle terms are accepted.
- Build a stable feature table with numeric and categorical columns.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve class balance and categorical-cardinality buckets.
- Emit metadata summary with row count, numeric/categorical feature counts,
  target shape, bytes, and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm preprocessing policy is acceptable.
- Mark this dataset `approved` before running any download.
