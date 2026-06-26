# Candidate Dataset: Store Sales Time Series Forecasting

- Dataset id: `store_sales_time_series`
- Target tier: local RTX 5090
- Task family: time series
- Covered model families: `gru_temporal`, `lstm_temporal`
- Source: https://www.kaggle.com/c/store-sales-time-series-forecasting
- Kaggle slug: `store-sales-time-series-forecasting`
- Source type: Kaggle competition
- Expected storage: well under 10 GB
- Approval status in registry: `approved`

## Why This Dataset

This is the primary time-series candidate for GRU/LSTM workloads. It gives the
predictor real time-window, item, store, and covariate distributions.

## Download Command

```bash
kaggle competitions download -c store-sales-time-series-forecasting -p datasets/raw/store_sales_time_series
```

## Preparation Plan

- Extract CSV files after Kaggle terms are accepted.
- Build rolling windows for the selected forecast horizon.
- Build deterministic subset masks: `tiny`, `small`, `medium`, `large`, `full`.
- Preserve store/item/date-window distribution.
- Emit metadata summary with window length, feature count, target shape, bytes,
  and subset checksums.

## Approval Checklist

- Confirm Kaggle competition terms are acceptable.
- Confirm windowing/horizon policy is acceptable.
- Mark this dataset `approved` before running any download.
