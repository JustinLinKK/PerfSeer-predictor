# Candidate Dataset: OGBN-Products

- Dataset id: `ogbn_products`
- Target tier: local RTX 5090
- Task family: graph
- Covered model families: `gat_graph`, `mpnn_graph`
- Source: https://snap-stanford.github.io/ogb-web/
- OGB slug: `ogbn-products`
- Source type: PyG/OGB
- Expected storage: single-digit GB download
- Approval status in registry: `pending`

## Why This Dataset

This is the primary large graph candidate. OGB provides standardized loaders
that are compatible with PyTorch Geometric and are a better fit for graph
workloads than most Kaggle competition datasets.

## Download Command

```bash
python scripts/manage_dataset_sources.py download-ogb ogbn_products --raw-root datasets/raw
```

## Preparation Plan

- Download through the OGB loader after approval.
- Build deterministic node/edge or subgraph subset masks: `tiny`, `small`,
  `medium`, `large`, `full`.
- Preserve node-degree, edge-count, and class-label buckets.
- Emit metadata summary with node/edge-count statistics, feature dimensions,
  label dimensions, bytes, and subset checksums.

## Approval Checklist

- Confirm OGB terms and storage budget are acceptable.
- Confirm graph task framing is node classification for v1.
- Mark this dataset `approved` before running any download.
