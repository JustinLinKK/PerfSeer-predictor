# PerfSeer Predictor

PerfSeer is a graph neural network based performance predictor for deep learning models. It represents a model as a compute graph with node, edge, and graph-level features, then predicts hardware/performance metrics such as execution time, memory usage, and SM utilization for both training and inference.

This project is based on the paper **"PerfSeer: An Efficient and Accurate Deep Learning Models Performance Predictor"** by Xinlong Zhao, Jiande Sun, Jia Zhang, Sujuan Hou, Shuai Li, Tong Liu, and Ke Liu. The paper introduces the SeerNet model, including SynMM aggregation and Global-Node Perspective Boost, plus the multi-output SeerNet-Multi variant with PCGrad.

## Project Direction

The first step of this repository is a reimplementation of the PerfSeer/SeerNet predictor from the paper and the open-sourced dataset. The public source material provides the dataset and dataset utilities, so the model, data pipeline, training loop, and evaluation flow are implemented here from the paper description.

After reproducing the baseline predictor, the next step is optimization for practical CPU inference. The optimized workflow is intended for scheduler-style use cases where predictions need to be accurate enough to guide GPU job placement while remaining lightweight enough to run on CPU.

## What Is Optimized

The baseline package in `src/perfseer/` preserves the reproduction setup: six independent single-output SeerNet models, one SeerBlock, Adam training, standardized log-space MSE, and the original handcrafted graph features.

The optimized package in `src/perfseer-optimized/` keeps that baseline available, then adds configurable improvements for accuracy, robustness, and CPU deployment:

- Training improvements: YAML configs, AdamW, weight decay, Huber/log-cosh losses, gradient clipping, optional EMA, richer checkpoint metadata, and a result ledger.
- Architecture variants: LayerNorm encoders, pre/post-normalized SeerBlocks, gated residual updates, two-block models, `SynMMPlus`, optional attention pooling, and paper/code ambiguity flags for ablation.
- Feature variants: optional topology, critical-path, edge-topology, destination tensor, operator one-hot, and raw versus batch-scaled time target modes.
- Deployment variants: `SeerNetMulti` with six output heads, optional PCGrad, metric loss weighting, teacher-student distillation, validation-only calibration, and CPU latency benchmarking.
- Evaluation tooling: metadata-based test reconstruction, per-metric MAPE/RMSPE/5%Acc/10%Acc, prediction export, error bucket reports, worst-case prediction tables, and CPU p50/p95 timing.

## Repository Layout

- `src/perfseer/`: baseline PerfSeer/SeerNet reproduction.
- `src/perfseer-optimized/`: optimized package, imported as `perfseer_optimized`.
- `src/perfseer-optimized/configs/`: backward-compatible experiment configs plus the remapped `train_accuracy/`, `train_deploy_model/`, and `eval_profiles/` config groups.
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

Training can use GPU. Both the baseline and optimized training CLIs select CUDA automatically when it is available. CPU is used for deployment-style inference tests by running optimized evaluation with `--bench-cpu`.

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

## Evaluation Procedure

Use a three-stage evaluation flow:

1. GPU inference evaluation for accuracy.
2. CPU inference evaluation for deployment latency and accuracy parity.
3. CPU-oriented optimization evaluation, which may be combined with step 2 when the model/config itself is designed for CPU inference.

The first stage answers: "Is the model accurate?" Run inference on GPU or the default device for fast metric computation:

```bash
conda run -n perfseer python -m perfseer_optimized.eval \
  --eval-profile src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml \
  --ckpt-dir runs/optimized/gated_2block_256 \
  --data-root dataset
```

The second stage answers: "How does the same checkpoint behave on CPU?" Run CPU evaluation with latency benchmarking:

```bash
conda run -n perfseer python -m perfseer_optimized.eval \
  --eval-profile src/perfseer-optimized/configs/eval_profiles/cpu_pytorch_fp32.yaml \
  --ckpt-dir runs/optimized/gated_2block_256 \
  --data-root dataset
```

The third stage applies CPU-inference-oriented choices, such as a single multi-output model, a distilled student, smaller hidden size, fewer blocks, dynamic quantization, TorchScript, ONNX Runtime, OpenVINO, and CPU thread tuning. For configs under `train_deploy_model/`, evaluate directly with deployment profiles because the model structure is part of the CPU deployment strategy:

```bash
conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/shared_multitask_192_pcgrad.yaml \
  --data-root dataset

conda run -n perfseer python scripts/run_deploy_matrix.py \
  --ckpt-dir runs/optimized/deploy_shared_multitask_192_pcgrad \
  --data-root dataset \
  --num-bench-graphs 1000
```

For CPU timing, use `--batch-size 1` to approximate scheduler-style single-graph prediction. Use `--batch-size 128` for throughput-style accuracy evaluation. Compare both per-metric accuracy and CPU p50/p95 latency before selecting a deployment candidate.

For baseline or accuracy-track checkpoints, CPU deployment cost should be measured as the total cost of all six single-output models. For deployment-track checkpoints such as `SeerNetMulti` or distilled students, CPU deployment cost is measured as one shared multi-output forward. `eval_deploy` records the requested backend and the actual backend used, so a failed export fallback is visible in `runs/results.jsonl`.

## Optimization Experiment Procedure

Each YAML file in `src/perfseer-optimized/configs/` is one experiment recipe. Existing top-level configs remain usable, but new experiments should use the remapped folders:

- `configs/train_accuracy/`: model/training configs whose main target is prediction accuracy.
- `configs/train_deploy_model/`: model/training configs whose structure is meant for CPU deployment.
- `configs/eval_profiles/`: runtime/evaluation profiles, not training configs.

Config meanings:

- `train_accuracy/baseline.yaml`: baseline-compatible reproduction through the optimized package.
- `train_accuracy/reg_layernorm_adamw.yaml`: training regularization with LayerNorm, dropout, AdamW, weight decay, Huber loss, gradient clipping, and calibration.
- `train_accuracy/gated_2block_256.yaml`: main accuracy structural candidate with two SeerBlocks, gated residuals, pre-norm, and hidden size `256`.
- `train_accuracy/topo_features.yaml`: accuracy candidate with topology, critical-path, edge-topology, destination tensor features, and `SynMMPlus`.
- `train_deploy_model/shared_multitask_256_pcgrad.yaml`: one shared multi-output model with six metric heads, PCGrad, and weighted metric losses.
- `train_deploy_model/shared_multitask_192_pcgrad.yaml`: compact shared multi-output deployment candidate.
- `train_deploy_model/distill_student_192.yaml`: compact distilled student for CPU deployment; set `distillation.teacher_ckpt_dir` before a full run.
- `train_deploy_model/distill_student_128.yaml`: smaller distilled CPU student candidate.
- `eval_profiles/*.yaml`: GPU accuracy, CPU PyTorch fp32, dynamic INT8, TorchScript, ONNX Runtime, and OpenVINO runtime profiles.

Run the baseline-compatible optimized package:

```bash
conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_accuracy/baseline.yaml \
  --data-root dataset

conda run -n perfseer python -m perfseer_optimized.eval \
  --eval-profile src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml \
  --ckpt-dir runs/optimized/accuracy_baseline \
  --data-root dataset
```

Run the main optimized structural candidate:

```bash
conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_accuracy/gated_2block_256.yaml \
  --data-root dataset

conda run -n perfseer python -m perfseer_optimized.eval \
  --eval-profile src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml \
  --ckpt-dir runs/optimized/accuracy_gated_2block_256 \
  --data-root dataset
```

Run several configs sequentially:

```bash
conda run -n perfseer python scripts/run_sweep.py \
  src/perfseer-optimized/configs/train_accuracy/reg_layernorm_adamw.yaml \
  src/perfseer-optimized/configs/train_accuracy/gated_2block_256.yaml \
  src/perfseer-optimized/configs/train_accuracy/topo_features.yaml
```

For a quick smoke test, add `--limit 200 --epochs 2`:

```bash
conda run -n perfseer python scripts/run_sweep.py \
  src/perfseer-optimized/configs/train_accuracy/baseline.yaml \
  src/perfseer-optimized/configs/train_deploy_model/shared_multitask_192_pcgrad.yaml \
  --limit 200 \
  --epochs 2
```

Run the CPU deployment runtime matrix for a trained checkpoint directory:

```bash
conda run -n perfseer python scripts/run_deploy_matrix.py \
  --ckpt-dir runs/optimized/deploy_shared_multitask_192_pcgrad \
  --data-root dataset \
  --batch-size 1 \
  --num-bench-graphs 1000
```

Compare completed evaluations:

```bash
conda run -n perfseer python scripts/summarize_results.py --results runs/results.jsonl
```

Run the full unattended experiment flow:

```bash
nohup bash run_full_experiment.sh --name full_$(date +%Y%m%d_%H%M%S) \
  > full_experiment.nohup.log 2>&1 &
```

The full-flow script trains the accuracy-track configs, evaluates GPU accuracy, records CPU fp32 latency, trains deployment-track configs, runs the CPU runtime matrix, and writes plots under `runs/experiments/<name>/plots/`. Use `--smoke` for a quick limited run, for example:

```bash
bash run_full_experiment.sh --name smoke --smoke --skip-distill
```

For a fair comparison, keep the same dataset root, split seed, training batch size, and epoch budget across runs. Do not select models by test results during tuning; use validation behavior for model selection and reserve test metrics for final reporting.

Training runs may use GPU for speed. Final accuracy can be checked on GPU, but deployment latency comparisons should use CPU with `--bench-cpu`, because the optimized deployment target is CPU-side scheduler inference.

## Current Goals

- Preserve a faithful baseline implementation for comparison.
- Improve robustness and accuracy with training, feature, and architecture variants.
- Add multi-output and distilled models for faster CPU deployment.
- Track per-metric accuracy and CPU inference latency in a reproducible result ledger.
