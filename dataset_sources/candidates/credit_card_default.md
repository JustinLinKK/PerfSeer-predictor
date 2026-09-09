# Candidate Dataset: Credit Card Default

- Source: Kaggle dataset `uciml/default-of-credit-card-clients-dataset`
- Source type: Kaggle dataset
- Task family: tabular
- Modality: tabular
- Model families: `ft_transformer_tabular`
- Selected replacement for: Kaggle competition `home-credit-default-risk`

## Reason

This dataset is a public Kaggle tabular credit-default dataset. It replaces the
gated Home Credit Default Risk competition source because the current Kaggle
credentials returned Hypertext Transfer Protocol `403 Forbidden` for that
competition during local verification.

## Verification

The dataset was verified by listing files and downloading the archive through the
same Kaggle Application Programming Interface credential path used by the
workflow:

```bash
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets files uciml/default-of-credit-card-clients-dataset
env -u PIP_PREFIX .venv/bin/python -m kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p /tmp/perfseer_kaggle_check_credit --force
```

Observed archive:

```text
/tmp/perfseer_kaggle_check_credit/default-of-credit-card-clients-dataset.zip
```

The archive contains `UCI_Credit_Card.csv`.

## Preparation Notes

`scripts/manage_dataset_sources.py` has a dataset-specific branch for
`credit_card_default` so subset masks are based on CSV row count rather than the
single archive member.
