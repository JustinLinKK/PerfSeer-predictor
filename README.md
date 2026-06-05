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
- `train_precision_teacher/large_teacher.yaml`: large precision/hardware-aware multi-output teacher for source-domain pretraining and precision transfer fine-tuning. For transfer fine-tuning, set `train.init_checkpoint` to the source-domain `seernet_multi.pt` or its checkpoint directory.
- `train_deploy_model/precision_distill_student_192.yaml`: precision/hardware-aware 192-wide distilled student; set `distillation.teacher_ckpt_dir` to either a large teacher run directory containing `seernet_multi.pt` or a six-checkpoint `seernet_metric*.pt` ensemble.
- `train_deploy_model/precision_distill_student_128.yaml`: precision/hardware-aware 128-wide distilled student for scheduler deployment; uses the same teacher checkpoint conventions.
- `eval_profiles/*.yaml`: GPU accuracy, CPU PyTorch fp32, dynamic INT8, TorchScript, ONNX Runtime, and OpenVINO runtime profiles.

Precision calibration runs can be converted into a training-ready dataset with
`scripts/materialize_precision_dataset.py`. It writes precision-specific label
files plus `label/precision_metadata.jsonl`, which the optimized data loader
uses to set per-sample precision/hardware global features. When including the
original dataset with `--base-data-root`, pass `--source-precision-config` and
`--source-hardware-id` so source-domain labels are tagged explicitly without
being upweighted as newly profiled precision rows. For final runs, also pass
`--source-precision-provenance` plus `--require-source-precision-provenance` so
the exact original-label precision setup is recorded in metadata and the
materialization report. If the original recipe is still unknown, keep
`--source-precision-config source_domain_unknown`; provenance will be required
but `source_precision_confirmed` remains false. Add
`--require-source-precision-confirmed` only after the original profiler recipe
is truly known. The precision-transfer flow also forwards the same provenance
into source-teacher pretraining metadata so the source checkpoint can be
checked before transfer. The materializer also writes
`precision_rejected_rows.jsonl` and status/precision/fallback counts in
`precision_materialization_report.json`, keeping unsupported FP8 rows auditable.
Optimized evaluation records aggregate metrics plus `metrics_by_precision`,
`metrics_by_batch_size`, `metrics_by_resource_regime`, and
`metrics_by_graph_signature` in `runs/results.jsonl`, along with
`precision_config_counts`, `hardware_id_counts`, and `label_domain_counts` so
held-out recipe, hardware, and source/precision-profile/pseudo-label coverage
can be checked directly.
The training CLI stores initialization and distillation-teacher metadata in each
checkpoint so source-domain pretraining, transfer fine-tuning, and student
distillation remain auditable. Train rows and checkpoint metadata also carry
the split unit, split hash, and train/val/test counts, which lets the final
result checker require `graph_signature` or `graph_family` training evidence
with minimum split sizes before accepting a deployment ledger. Precision student
distillation uses label-domain hard-alpha settings so measured precision labels
can stay hard-label dominated while source or pseudo rows can blend toward
teacher soft targets. The
precision materializer can create opt-in pseudo rows with
`--pseudo-precision-sweep`; pseudo rows are kept in the train split only and
excluded from normalization stats, and use `features.pseudo_label_weight`
rather than the measured precision weight.
Precision training checkpoints also store `supported_precision_hardware` allow-lists for
clear deployment-time validation of requested precision/hardware domains; the
source converter accepts `--precision-config`, `--hardware-id`, and JSON feature
overrides and checks them against that allow-list. Optimized evaluation also
rejects test rows outside the checkpoint's supported precision/hardware pairs
before scoring. Deployment evaluation writes `deployment_metadata.json` beside
exported runtime artifacts with the feature layout, precision/hardware config,
supported allow-list, runtime backend, and held-out split evidence.
Precision teacher/student configs set
`data.split_unit: graph`, which keeps all precision variants for a graph in the
same train/val/test split to avoid graph leakage across precision rows. For a
structural robustness run, pass `--split-unit graph_signature` to the training
CLI or flow runner to hold out whole graph-signature clusters. Their
default `features.target_mode` is `absolute`; set it to `log_ratio_to_source`
for a residual transfer experiment that predicts log(label/source_label) and is
converted back to absolute metrics during evaluation.

Typical precision-transfer sequence:

```bash
python scripts/run_precision_transfer_flow.py \
  --results-dir /mnt/output/nrp_calibration \
  --precision-data-root dataset_precision_a100 \
  --hardware-id a100 \
  --dry-run
```

The flow runner prints or executes the materialization, source-teacher
pretraining, source-domain evaluation, precision transfer, student distillation,
and held-out precision evaluation commands. Use `--skip-source-eval` only when
you already have a separate baseline-preservation check for the source teacher.
Add `--structural-validation-splits graph_signature,graph_family` to run
additional split-suffixed transfer/student validation flows that reuse the
source checkpoint and require matching structural split evidence in the checker.
For final source pretraining gates, pass `--baseline-run-id` and
`--max-source-baseline-mape-delta` with `--check-results` so the source teacher
must stay within the accepted MAPE delta from the current baseline eval row.
Its default `--source-precision-config` is `source_domain_unknown`; replace it
with `fp32_ieee`, `tf32`, or another concrete recipe only after the original
label collection setup is confirmed. The current evidence snapshot is recorded
in `SOURCE_PRECISION_PROVENANCE.md`.
Add `--check-results` after a real run to validate that `runs/results.jsonl`
contains source-teacher, precision-teacher, and precision-student eval rows with
precision, label-domain, batch-size, resource-regime, and graph-signature slices
before using the student for deployment. Pass `--required-label-domain
precision_profile` for final transfer/student gates so held-out accuracy must
come from real precision labels rather than only source or pseudo rows; add
`--min-eval-precision-labels` to require a concrete held-out precision-profile
count in precision teacher/student and deployment eval rows. Add
`--min-eval-precision-count bf16_amp=20` or repeat the flag for each required
recipe when a final run must prove enough held-out examples for specific
precision configs. Add `--min-eval-hardware-count a100=20` when a final run
must prove enough held-out examples for a specific hardware domain. When the
flow is run with `--split-unit
graph_signature`, the checker also requires eval rows to report that same
split-unit evidence and test hash; deployment eval rows record the same split
metadata for later artifact comparison. If a checkpoint test hash is present,
the checker rejects eval rows that silently score a different full held-out
split. Add `--min-batch-size-slices`,
`--min-resource-regime-slices`, `--min-label-domain-slices`, or
`--min-graph-signature-slices` when a run must prove broader held-out coverage.
When source precision provenance is required, the checker reads
`precision_materialization_report.json` to verify that provenance and the
accepted precision-label count, and requires the source-teacher
`train_complete` row to record the same provenance. Add
`--require-source-precision-confirmed` only when the source recipe has been
confirmed and both the materialization report and source train checkpoint should
mark it confirmed. Pass
`--deploy-eval-profile` to add a deployment runtime evaluation for the student;
with `--check-results`, the checker requires the deployment row and its
`deployment_metadata.json` sidecar. Add `--require-checkpoint-files` when the
final gate should also prove that every eval ledger row still points at a real
checkpoint file; with deployment eval enabled, the flow also requires the
deployment eval checkpoint paths to match the held-out precision-student eval
checkpoint paths. With `--require-train-events` and `--require-checkpoint-files`,
the flow additionally requires source/transfer/student eval checkpoint paths to
match their corresponding `train_complete` checkpoint paths. Add
`--require-train-events` to require source, transfer, and
student `train_complete` rows in the same ledger; pass
`--required-train-label-domain source,precision_profile` so precision
teacher/student training rows must prove the train split included both original
source-domain and measured precision-label rows; add
`--min-train-source-labels` and `--min-train-precision-labels` to enforce final
train-split count floors for the original 53k source rows and real precision
profiles; add `--min-train-precision-count bf16_amp=20` for per-recipe
train-split floors; add `--min-train-hardware-count a100=20` for per-hardware
train-split floors; add `--min-train-split-count`, `--min-val-split-count`, and
`--min-train-test-count` when final ledgers must prove structural-holdout
training used enough rows in every split; add
`--require-unlimited-train-data` for final non-smoke runs so accidental
`--limit` training cannot pass the gate. The underlying
training commands are:

```bash
conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_precision_teacher/large_teacher.yaml \
  --run-id precision_large_teacher_source \
  --data-root dataset \
  --precision-config fp32_ieee \
  --hardware-id source_domain_unknown \
  --source-precision-provenance original-profiler-notes.md#fp32 \
  --require-source-precision-provenance

conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_precision_teacher/large_teacher.yaml \
  --run-id precision_large_teacher_transfer \
  --data-root dataset_precision_a100 \
  --init-checkpoint runs/optimized/precision_large_teacher_source/seernet_multi.pt

conda run -n perfseer python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/precision_distill_student_128.yaml \
  --data-root dataset_precision_a100 \
  --teacher-ckpt-dir runs/optimized/precision_large_teacher_transfer
```

### Optimized Model Differences and Expected Performance

The baseline comparison point is `train_accuracy/baseline.yaml`: six independent single-output SeerNet models, hidden size `256`, one SeerBlock, direct residual updates, `SynMM`, operator one-hot features, Adam, standardized log-space MSE, and no validation calibration. CPU deployment latency for this baseline is the combined cost of running all six single-output models.

The following table explains how each optimized training config differs from that baseline and what change it is expected to make. The measured deltas are from the completed full run `runs/experiments/full_20260601_122347/results.jsonl` against the optimized baseline result in the same ledger: GPU mean MAPE `3.891%` and CPU PyTorch fp32 p50 `6.934 ms`. Negative MAPE delta is better; CPU speedup is higher-is-better.

| Config | Main difference from baseline | Expected performance change | Current measured change |
| --- | --- | --- | --- |
| `accuracy_reg_layernorm_adamw` | Keeps six single-output `SeerNet` models, one block, and hidden `256`, but adds LayerNorm, dropout `0.05`, AdamW, weight decay, Huber log-std loss, gradient clipping, and linear calibration. | Better training robustness and less outlier sensitivity, with about the same CPU cost. It can improve individual noisy metrics but is not guaranteed to improve mean MAPE. | GPU mean MAPE `3.981%`, `+0.090 pp` worse than baseline. CPU PyTorch p50 `6.732 ms`, about `1.03x` baseline speed. |
| `accuracy_gated_2block_256` | Keeps six single-output models and hidden `256`, but uses two SeerBlocks, pre-norm blocks, gated residual updates, LayerNorm, dropout, AdamW, Huber loss, gradient clipping, and calibration. | Modest accuracy gain from more graph message-passing capacity, with a clear CPU latency cost because six deeper models must run. | GPU mean MAPE `3.827%`, `-0.064 pp` better than baseline. CPU PyTorch p50 `14.127 ms`, about `0.49x` baseline speed; TorchScript p50 `11.084 ms`. |
| `accuracy_topo_features` | Starts from the gated two-block model, then adds topology, critical-path, edge-topology, destination tensor features, and `SynMMPlus`. | More graph-structure signal, especially for time/utilization metrics, but more input features and a deeper model make CPU inference slower and can overfit if the added features do not help the split. | GPU mean MAPE `3.935%`, `+0.044 pp` worse than baseline. CPU PyTorch p50 `16.727 ms`, about `0.41x` baseline speed. |
| `deploy_shared_multitask_256_pcgrad` | Replaces six single-output models with one `SeerNetMulti` model, hidden `256`, two blocks, separate metric heads, topology features, `SynMMPlus`, weighted losses, and PCGrad. | Large CPU speedup because one shared trunk predicts all six metrics. Accuracy should stay near baseline if PCGrad handles task conflicts. | GPU mean MAPE `3.923%`, `+0.032 pp` worse than baseline. CPU PyTorch p50 `2.418 ms`, `2.87x` faster; best measured backend is TorchScript p50 `1.764 ms`, `3.93x` faster. |
| `deploy_shared_multitask_192_pcgrad` | Same shared multi-output idea as the 256-wide deployment model, but hidden/head size is reduced to `192` and aggregation uses `SynMM`. | Smaller and faster than the 256-wide multi-output model, with a likely small accuracy loss from reduced capacity. | GPU mean MAPE `4.005%`, `+0.115 pp` worse than baseline. CPU PyTorch p50 `1.936 ms`, `3.58x` faster; best measured backend is TorchScript p50 `1.327 ms`, `5.23x` faster. |
| `deploy_distill_student_192` | Uses the same compact 192-wide `SeerNetMulti` shape, but trains with teacher distillation from `accuracy_topo_features` instead of PCGrad weighted losses. | Recovers accuracy lost by the compact multi-output model while keeping the CPU latency benefit of a single shared trunk. | GPU mean MAPE `3.877%`, `-0.014 pp` better than baseline. CPU PyTorch p50 `1.803 ms`, `3.85x` faster; best measured backend is TorchScript p50 `1.340 ms`, `5.17x` faster. |
| `deploy_distill_student_128` | Shrinks the distilled shared multi-output student to hidden/head size `128` with the same topology features and teacher-distillation path. | Smallest and fastest deployment candidate. Distillation keeps accuracy in the second tier while the smaller shared trunk minimizes scheduler-side CPU cost. | GPU/CPU mean MAPE `3.861%`, `-0.030 pp` better than baseline. CPU PyTorch p50 `1.602 ms`, `4.33x` faster; best measured backend is TorchScript p50 `1.056 ms`, `6.57x` faster. |

The accuracy-track models are useful for understanding which modeling changes improve prediction quality, but they are not necessarily good CPU deployment choices because they still run six separate single-output models. The deployment-track models trade a small amount of accuracy budget for a much lower scheduler-side CPU cost by sharing one graph encoder across all six metric heads.

### Final CPU Deployment Choice

Use `deploy_distill_student_128` with the `cpu_torchscript_fp32` runtime profile as the final CPU-side scheduler predictor. In the completed full run it is the fastest measured CPU candidate: p50 `1.056 ms`, p95 `1.529 ms`, throughput `923.1 graphs/s`, artifact size `2.90 MB`, and `707,089` parameters.

This is the best latency/accuracy trade-off in the run. Its CPU mean MAPE is `3.861%`, which is not the absolute MAPE champion, but it is in the second accuracy tier: only `0.034 pp` worse than `accuracy_gated_2block_256` at `3.827%`, while being about `13.4x` faster by p50 CPU latency than that accuracy champion's PyTorch CPU path. It is also slightly more accurate than the optimized baseline (`3.891%`) while being `6.57x` faster by p50 latency and about `10x` smaller by artifact size.

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

The full-flow script trains the accuracy-track configs, evaluates GPU accuracy, records CPU fp32 latency, trains deployment-track configs, runs the CPU runtime matrix, and writes plots under `runs/experiments/<name>/plots/`. The plot set includes `tradeoff_3d_cpu_size_speed_mape.html`, an interactive CPU trade-off view over model artifact size, CPU throughput, and CPU mean MAPE. Use `--smoke` for a quick limited run, for example:

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
