# PerfSeer Predictor

PerfSeer is a graph neural network based performance predictor for deep learning models. It represents a model as a compute graph with node, edge, and graph-level features, then predicts hardware/performance metrics such as execution time, memory usage, and SM utilization for both training and inference.

This project is based on the paper **"PerfSeer: An Efficient and Accurate Deep Learning Models Performance Predictor"** by Xinlong Zhao, Jiande Sun, Jia Zhang, Sujuan Hou, Shuai Li, Tong Liu, and Ke Liu. The paper introduces the SeerNet model, including SynMM aggregation and Global-Node Perspective Boost, plus the multi-output SeerNet-Multi variant with PCGrad.

## Project Direction

The first step of this repository is a reimplementation of the PerfSeer/SeerNet predictor from the paper and the open-sourced dataset. The public source material provides the dataset and dataset utilities, so the model, data pipeline, training loop, and evaluation flow are implemented here from the paper description.

After reproducing the baseline predictor, the next step is optimization for practical CPU inference. The optimized workflow is intended for scheduler-style use cases where predictions need to be accurate enough to guide GPU job placement while remaining lightweight enough to run on CPU.

## Repository Layout

- `src/perfseer/`: baseline PerfSeer/SeerNet reproduction.
- `src/perfseer-optimized/`: optimized package, imported as `perfseer_optimized`.
- `src/perfseer-optimized/configs/`: experiment configs for baseline, regularized, gated, topology, multi-task, and distillation variants.
- `runs/full/`: existing full-run baseline checkpoints and curves.
- `runs/optimized/`: optimized experiment outputs.
- `MODEL_STRUCTURE_REPORT.md`: detailed baseline model structure notes.
- `record.md`: reproduction record and dataset notes.

## Dataset

The dataset comes from the PerfSeer open-source release and contains 53k+ compute graphs with matching labels.

Dataset link:

```text
https://drive.google.com/drive/folders/1T7DNKUyjIdnLIMTL4IZA67ynvhA75Pdt
```

Expected local layout:

```text
dataset/
  cg/cg/*.pkl
  label/label/*.txt
```

## Quick Start

Install the package in the `perfseer` conda environment:

```bash
conda run -n perfseer python -m pip install -e .
```

Train the baseline reproduction:

```bash
conda run -n perfseer python -m perfseer.train --metric all --epochs 500 --patience 30 --out runs/full --data-root dataset
```

Evaluate the baseline checkpoints:

```bash
conda run -n perfseer python -m perfseer.eval --ckpt-dir runs/full --data-root dataset --batch-size 128
```

Run an optimized smoke test:

```bash
conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/baseline.yaml \
  --limit 200 \
  --epochs 2
```

Evaluate optimized checkpoints with CPU benchmarking:

```bash
conda run -n perfseer python -m perfseer_optimized.eval \
  --ckpt-dir runs/optimized/baseline \
  --data-root dataset \
  --limit 200 \
  --bench-cpu \
  --batch-size 1
```

## Current Goals

- Preserve a faithful baseline implementation for comparison.
- Improve robustness and accuracy with training, feature, and architecture variants.
- Add multi-output and distilled models for faster CPU deployment.
- Track per-metric accuracy and CPU inference latency in a reproducible result ledger.
