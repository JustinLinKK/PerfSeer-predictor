# PerfSeer Model Improvement Plan for Coding Agent

## Objective

Improve prediction accuracy for the current PerfSeer/SeerNet reproduction while keeping CPU inference practical for a GPU training job scheduler. The scheduler will use predictions to reduce GPU interference, so the priority is not only mean MAPE but also reliable time/utilization/memory predictions and low CPU inference overhead.

## Hard Constraints

- Use the same dataset and labels. Do not add new profiling data.
- CPU inference is required for deployment.
- Preserve the current baseline exactly before changing the model.
- Do not tune on the test set. Use validation for model selection and keep test for final reporting.
- Every experiment must report mean MAPE plus per-metric MAPE for:
  - `train_util`
  - `train_mem`
  - `train_time`
  - `infer_util`
  - `infer_mem`
  - `infer_time`

## Current Baseline to Preserve

Current implementation:

- Six independent single-output SeerNet models, one per metric.
- `hidden=256`, `num_blocks=1`.
- `1,263,107` trainable parameters per single-metric model.
- Current full-run mean MAPE: `3.898%`.
- Weakest metrics:
  - `train_time`: `5.606%` MAPE.
  - `infer_time`: `5.521%` MAPE.
- Strongest metrics:
  - `train_mem`: `2.008%` MAPE.
  - `infer_mem`: `2.131%` MAPE.

Paper/reference context:

- PerfSeer/SeerNet relies on node, edge, and global graph features.
- The paper shows SynMM and GNPB are high-value components, so keep them as the default unless explicitly ablated.
- The paper's SeerNet-Multi with PCGrad is useful for multi-metric CPU deployment, but single-metric SeerNet is more accurate in the reported results. Treat multi-task as a deployment and possible pretraining/distillation path, not as an automatic accuracy replacement.

## Recommended Priority Order

1. Add a reproducible experiment and CPU-benchmark harness.
2. Add low-risk regularization and training improvements.
3. Add residual gating and normalization so deeper SeerBlocks can be tested safely.
4. Add topology/critical-path features aimed at improving time metrics.
5. Implement multi-task PCGrad and teacher-student distillation for CPU deployment.
6. Run structured sweeps and select a Pareto-optimal model by MAPE, per-metric regressions, parameter count, and CPU latency.

---

# P0: Reproducibility and Benchmark Harness

## Changes

### 1. Persist full run metadata

Modify checkpoint saving so every run stores:

- Full model config.
- Full feature normalization stats.
- Target normalization stats.
- Dataset root.
- Split seed.
- Exact train/val/test file stems or a hash of each split.
- Git commit hash if available.
- PyTorch/PyG versions.
- CPU information and thread count for inference benchmarks.

Files likely involved:

- `src/perfseer/train.py`
- `src/perfseer/eval.py`
- `src/perfseer/data.py`

### 2. Add experiment config files

Create:

```text
configs/
  baseline.yaml
  reg_layernorm_adamw.yaml
  gated_2block_192.yaml
  gated_2block_256.yaml
  topo_features.yaml
  multitask_pcgrad.yaml
  distill_student.yaml
```

Each config should include model, feature, optimizer, loss, seed, data, and evaluation settings.

### 3. Add a result ledger

Create `runs/results.csv` or `runs/results.jsonl` with one row per experiment:

```text
run_id, seed, model_name, hidden, num_blocks, params, cpu_pred_ms_p50, cpu_pred_ms_p95,
mean_mape, train_util_mape, train_mem_mape, train_time_mape,
infer_util_mape, infer_mem_mape, infer_time_mape,
rmspe_mean, acc5_mean, acc10_mean, notes
```

### 4. Add CPU inference benchmark

Add an evaluation mode:

```bash
python -m perfseer.eval \
  --ckpt-dir runs/<run_id> \
  --data-root dataset \
  --batch-size 1 \
  --device cpu \
  --bench-cpu \
  --num-bench-graphs 1000
```

Measure separately:

- Graph representation/extraction latency if available.
- Model forward latency only.
- End-to-end prediction latency.
- p50, p95, and mean latency.

Use a fixed CPU thread setting for comparable results:

```python
torch.set_num_threads(args.cpu_threads)
torch.set_num_interop_threads(max(1, args.cpu_interop_threads))
```

## Acceptance Criteria

- Baseline run can be reproduced and produces metrics close to the recorded baseline.
- Evaluation no longer depends on recomputing hidden normalization state from a matching seed only.
- Every future experiment can be ranked without manual log reading.

---

# P1: Low-Risk Training Improvements

These changes should be implemented first because they are likely to improve generalization without changing graph semantics.

## 1. Add AdamW and weight decay

Current baseline uses Adam. Add CLI/config support for:

```yaml
optimizer: adamw
lr: 0.001
weight_decay: 0.00001  # also test 0.0001
```

Sweep:

```text
optimizer: Adam, AdamW
weight_decay: 0, 1e-5, 1e-4
lr: 1e-3, 3e-4
```

Recommended first full candidate:

```yaml
optimizer: adamw
lr: 0.001
weight_decay: 0.00001
```

## 2. Add LayerNorm

Add LayerNorm after the stream encoders and optionally inside block MLPs.

Recommended default:

```text
node_enc: Linear -> LayerNorm
edge_enc: Linear -> LayerNorm
global_enc: Linear -> LayerNorm
block outputs: optional LayerNorm before residual add
```

Prefer `LayerNorm(hidden)` over BatchNorm because graph batches have variable node counts and CPU inference must be stable for batch size 1.

## 3. Add small dropout

Add dropout in MLP hidden layers and before prediction heads.

Sweep:

```text
dropout: 0.0, 0.05, 0.10
```

Recommended first candidate: `dropout=0.05`.

Avoid high dropout initially because the dataset is not tiny and the current model already performs well.

## 4. Add gradient clipping

Add:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Make it configurable:

```yaml
grad_clip_norm: 1.0
```

## 5. Add robust loss options

Keep current standardized log-space MSE as the baseline. Add:

- Huber loss in standardized log space.
- Log-cosh loss in standardized log space.
- Optional relative-error validation objective for checkpoint selection.

Recommended sweep:

```text
loss: mse_logstd, huber_logstd, logcosh_logstd
```

For Huber, test:

```text
huber_delta: 0.5, 1.0
```

## 6. Add checkpoint averaging or EMA

Add optional exponential moving average of weights:

```yaml
ema_decay: 0.999
```

Evaluate both raw best checkpoint and EMA checkpoint. Keep the better validation MAPE checkpoint.

## Acceptance Criteria

Promote this phase only if at least one config improves mean MAPE or improves time MAPE without hurting memory metrics materially.

Suggested reject rules:

- Reject if mean MAPE worsens by more than `0.10` percentage points.
- Reject if either time metric worsens by more than `0.25` percentage points unless CPU latency or parameter count improves substantially.
- Reject if memory MAPE worsens by more than `0.20` percentage points.

---

# P2: Model Structure Improvements

## 1. Add gated residual updates

Current model applies direct residual updates after each SeerBlock. Replace or augment this with configurable gated residuals.

Implement a reusable module:

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden: int, init_value: float = 0.1, mode: str = "scalar"):
        ...

    def forward(self, old, update):
        # scalar or vector gate, initialized small
        return old + gate * update
```

Recommended default:

```text
mode: scalar_per_stream
init_value: 0.1
streams: node, edge, global, z
```

Alternative:

```python
old + torch.sigmoid(gate) * dropout(update)
```

Initialize `gate` so the first value is about `0.1`, not `0.5`, to avoid destabilizing deeper blocks.

## 2. Add pre-norm SeerBlock option

For deeper models, add:

```text
v_in = LayerNorm(v)
e_in = LayerNorm(e)
u_in = LayerNorm(u)
z_in = LayerNorm(z)
```

Then feed normalized streams into MLP updates, but keep residual path on the original streams.

Recommended defaults for deeper variants:

```yaml
block_norm: prenorm
residual: gated
dropout: 0.05
```

## 3. Test two-block models before going deeper

The current model has one SeerBlock. Time prediction may benefit from more message passing, but deeper blocks increase CPU cost and overfit risk.

Recommended sweep order:

```text
hidden=192, num_blocks=2, LayerNorm, gated residual, dropout=0.05
hidden=256, num_blocks=2, LayerNorm, gated residual, dropout=0.05
hidden=320, num_blocks=1, LayerNorm, gated residual, dropout=0.05
hidden=128, num_blocks=3, LayerNorm, gated residual, dropout=0.05
```

Do not start with `hidden=256, num_blocks=3`; it may be too expensive for CPU and may overfit.

## 4. Keep SynMM and GNPB by default

Do not remove SynMM or GNPB in the main candidate. They are core accuracy components in the paper.

Only ablate them in explicit experiments:

```yaml
use_synmm: true
use_gnpb: true
```

## 5. Add `SynMMPlus` as an optional aggregation

Current SynMM uses max and mean. Add a configurable aggregation variant:

```text
SynMM:      concat(max, mean) -> Linear
SynMMPlus:  concat(max, mean, sum, std) -> Linear
```

Use `SynMMPlus` only for node-to-global aggregation first. Do not add it to all edge/node aggregations until benchmarked.

Sweep:

```text
global_agg: synmm, synmm_plus
```

## 6. Add lightweight attention pooling variant

Add a graph-level attention pooling option for `v -> u` only:

```text
attention_pool: score = Linear(tanh(Linear(v)))
pooled = scatter_softmax(score, batch) weighted sum v
output = Linear(concat(attn_pool, mean, max))
```

Because CPU inference matters, keep it single-head first. Compare against SynMM/SynMMPlus.

## 7. Make paper-vs-code ambiguous choices configurable

Add flags rather than hard-coding one interpretation:

```yaml
include_u_in_edge_update: true | false
mlp_z_num_linear_layers: 2 | 3
residual_mode: none | direct | gated
use_operator_type_onehot: true | false
softmax_agg_mode: learned_score | feature_softmax
```

Run a small calibration matrix before full training:

```text
A: current behavior
B: no operator one-hot
C: no u in edge update
D: MLP_z with 2 linear layers total
E: gated residual instead of direct residual
F: combinations of B+C+D+E
```

Do this on `--limit 5000` or `--limit 10000` for screening, then full dataset only for top candidates.

## Acceptance Criteria

A structural change should only be promoted if:

- Full-test mean MAPE improves, or
- Full-test time MAPE improves significantly with no major regression in memory/utilization, and
- CPU prediction p95 remains acceptable for scheduler use.

Recommended CPU forward target:

```text
single graph model-forward p95 <= 10 ms
```

If graph extraction still dominates end-to-end latency, keep model forward p95 low enough that future extraction optimizations are not blocked.

---

# P3: Feature Improvements for Time Accuracy

Current memory metrics are already strong. Most feature work should target `train_time` and `infer_time`.

## 1. Add topological node features

Add optional node features derived only from the existing compute graph:

```text
in_degree
out_degree
normalized_topological_index
normalized_forward_depth
normalized_reverse_depth
is_source_node
is_sink_node
is_branch_node          # out_degree > 1
is_join_node            # in_degree > 1
num_ancestors_log1p     # if cheap enough
num_descendants_log1p   # if cheap enough
```

Normalize scalar topology features using train-only stats where appropriate.

## 2. Add critical-path proxy features

Runtime is often closer to critical-path behavior than total FLOPs alone. Add graph-level and node-level proxies:

Node-level:

```text
longest_path_depth_from_input
longest_path_depth_to_output
flop_weighted_depth_from_input
flop_weighted_depth_to_output
is_on_unweighted_longest_path
is_on_flop_weighted_longest_path
```

Graph-level:

```text
max_topological_depth
mean_topological_depth
max_flop_weighted_path
critical_path_flops_ratio = max_flop_weighted_path / total_flops
num_branch_nodes
num_join_nodes
branch_join_ratio
```

Keep these behind a feature flag:

```yaml
features:
  topology: true
  critical_path: true
```

## 3. Enrich edge features

Current edge features come from source output tensor metadata. Add optional edge features:

```text
source_out_degree
target_in_degree
source_topological_depth
target_topological_depth
depth_delta
is_skip_like_edge        # depth_delta > 1
edge_tensor_bytes_log1p  # existing size, ensure consistently named
```

If destination input metadata is available, add:

```text
destination_input_size
destination_input_shape_channels
destination_input_shape_height
destination_input_shape_width
```

## 4. Keep operator-type one-hot as an ablation, not a default removal

The paper says node categories reduced accuracy, but the current reproduction includes a 10-channel operator one-hot and already beats the paper target. Therefore:

- Do not remove operator type unconditionally.
- Add `use_operator_type_onehot` flag.
- Compare `true` vs `false` under the exact same architecture and seed.
- If removing it improves validation and test MAPE, promote removal.

## 5. Update cache keys

Any feature dimension or normalization change must invalidate caches. Include feature flags and feature dimensions in the cache hash.

## Acceptance Criteria

Feature changes are successful if they improve either:

- `train_time` and `infer_time` MAPE by at least `0.25` percentage points each, or
- mean MAPE by at least `0.10` percentage points,

without causing memory metrics to regress by more than `0.20` percentage points.

---

# P4: Multi-Task, PCGrad, and Distillation

The final scheduler probably needs multiple metrics at once. Loading six separate single-output models on CPU may be unnecessary if a multi-output model is accurate enough.

## 1. Implement SeerNetMulti

Architecture:

```text
shared encoders
shared SeerBlock trunk
six metric-specific MLP heads
```

Each head:

```text
Linear(hidden, hidden) -> activation -> dropout -> Linear(hidden, 1)
```

Output shape:

```text
[B, 6]
```

Config:

```yaml
model: seernet_multi
num_outputs: 6
head_hidden: 256
metric_heads: separate
```

## 2. Implement PCGrad

Add PCGrad support for multi-task training. Requirements:

- Compute one loss per metric.
- Compute one gradient vector per task for shared trunk parameters.
- If cosine similarity between task gradients is negative, project away the conflicting component.
- Keep head-specific gradients task-specific.
- Make PCGrad optional.

Config:

```yaml
multi_task:
  enabled: true
  loss_reduction: pcgrad
```

Compare against:

```text
plain_sum
weighted_sum
uncertainty_weighted
pcgrad
```

## 3. Add metric loss weighting

Time metrics are currently weaker. Add configurable weights:

```yaml
loss_weights:
  train_util: 1.0
  train_mem: 0.75
  train_time: 1.5
  infer_util: 1.0
  infer_mem: 0.75
  infer_time: 1.5
```

Sweep:

```text
time_weight: 1.0, 1.5, 2.0
mem_weight: 0.5, 0.75, 1.0
```

## 4. Use multi-task as pretraining, then fine-tune single-metric heads

For best accuracy, try:

1. Train `SeerNetMulti` shared trunk with PCGrad.
2. Initialize six single-output models from the shared trunk.
3. Fine-tune each metric independently with low LR, e.g. `1e-4` or `3e-5`.
4. Compare against training six independent models from scratch.

This may improve time metrics without sacrificing the already strong memory metrics.

## 5. Add teacher-student distillation for CPU deployment

For scheduler deployment, train a compact multi-output student using the same training graphs:

Teacher:

- Best six single-metric models, possibly averaged across seeds.

Student:

- `hidden=128` or `hidden=192`.
- `num_blocks=1` or `2`.
- Six heads.

Student loss:

```text
L = alpha * label_loss + (1 - alpha) * teacher_distillation_loss
```

Recommended:

```yaml
alpha: 0.5
teacher_loss: mse_logstd
label_loss: huber_logstd
```

This uses the same dataset graphs and labels; it does not require new profiling data.

## Deployment Decision Rule

Maintain two model classes:

1. `accuracy_champion`: best per-metric accuracy, can be six single-metric models.
2. `scheduler_deploy`: best CPU Pareto model, preferably one multi-output model.

Promote `scheduler_deploy` if:

```text
mean MAPE <= accuracy_champion_mean_mape + 0.30 percentage points
AND train_time/infer_time MAPE <= accuracy_champion_time_mape + 0.30 percentage points
AND CPU p95 forward latency is at least 2x faster than running six independent models
```

---

# P5: Time Target Audit

The report notes uncertainty about whether `time` should be raw per-sample time or `time * batch_size` per-iteration time. For a GPU training job scheduler, per-iteration or job-level runtime may be more relevant than per-sample time.

## Changes

Add configurable label parsing:

```yaml
time_target_mode: raw | batch_scaled
```

Implementation:

```python
if time_target_mode == "raw":
    train_time = raw_train_time
    infer_time = raw_infer_time
elif time_target_mode == "batch_scaled":
    train_time = raw_train_time * batch_size
    infer_time = raw_infer_time * batch_size
```

## Evaluation

Report both:

- MAPE on raw per-sample time.
- MAPE on batch-scaled per-iteration time.

Select the target that matches scheduler objective:

- If scheduling decisions are based on step duration and occupancy, prefer batch-scaled time.
- If scheduling decisions normalize by sample throughput, raw time may be acceptable.

## Acceptance Criteria

Do not switch the production target until the downstream scheduler objective is clear. Keep both modes available.

---

# P6: Validation Calibration

Add a post-training calibration option using validation predictions only. This is low-cost at CPU inference and may reduce systematic bias.

## 1. Linear calibration in log target space

For each metric, fit on validation set:

```text
y_true_log = a * y_pred_log + b
```

At inference:

```text
y_pred_calibrated_log = a * y_pred_log + b
```

Store `a` and `b` in checkpoint metadata.

## 2. Optional isotonic calibration

Try isotonic regression only if monotonic bias is visible and there is no overfitting. Do not use by default.

## Acceptance Criteria

Promote calibration if it improves validation and test MAPE without hurting RMSPE or 5%Acc.

---

# Experiment Matrix

## Stage 0: Baseline Reproduction

```bash
python -m perfseer.train --metric all --epochs 500 --patience 30 --out runs/baseline_repro --data-root dataset
python -m perfseer.eval --ckpt-dir runs/baseline_repro --data-root dataset --batch-size 128 --bench-cpu
```

Expected: close to current `3.898%` mean MAPE.

## Stage 1: Low-Risk Training Sweep

Run full dataset if feasible; otherwise screen with `--limit 10000` and then full-run top candidates.

```text
A1: baseline + AdamW wd=1e-5
A2: baseline + AdamW wd=1e-4
A3: baseline + LayerNorm + AdamW wd=1e-5
A4: baseline + LayerNorm + dropout=0.05 + AdamW wd=1e-5
A5: A4 + Huber loss delta=1.0
A6: A4 + log-cosh loss
A7: A4 + EMA
```

Promote best A-candidate as `regularized_baseline`.

## Stage 2: Structural Sweep

```text
B1: hidden=192, blocks=2, LayerNorm, gated residual, dropout=0.05
B2: hidden=256, blocks=2, LayerNorm, gated residual, dropout=0.05
B3: hidden=320, blocks=1, LayerNorm, gated residual, dropout=0.05
B4: hidden=128, blocks=3, LayerNorm, gated residual, dropout=0.05
B5: B1 + SynMMPlus
B6: B2 + SynMMPlus
B7: B1 + single-head attention graph pooling
B8: B2 + single-head attention graph pooling
```

Promote the model that improves time MAPE without CPU latency regression.

## Stage 3: Feature Sweep

```text
C1: best_B + topology node features
C2: best_B + critical-path graph/node features
C3: best_B + edge topology features
C4: best_B + topology + critical-path + edge topology
C5: C4 with operator one-hot disabled
C6: C4 with no u in edge update
C7: C4 with MLP_z using 2 linear layers total
```

Promote best C-candidate as `accuracy_champion_candidate`.

## Stage 4: Multi-Task and Distillation

```text
D1: SeerNetMulti, hidden=256, blocks=1, plain summed loss
D2: SeerNetMulti, hidden=256, blocks=1, PCGrad
D3: SeerNetMulti, hidden=256, blocks=2, gated residual, PCGrad
D4: D3 + time-weighted loss
D5: Distilled student, hidden=192, blocks=2, labels + teacher predictions
D6: Distilled student, hidden=128, blocks=2, labels + teacher predictions
```

Promote best D-candidate as `scheduler_deploy_candidate` if it satisfies the deployment decision rule.

## Stage 5: Seed Robustness

Run top 3 candidates with at least these seeds:

```text
42, 43, 44
```

Optional broader seed set:

```text
42, 43, 44, 45, 46
```

Report mean and standard deviation of all metrics. Do not declare a winner from one lucky split.

---

# Model Selection Metrics

Use this ranking order:

1. Lower `mean_mape`.
2. Lower `train_time_mape` and `infer_time_mape`.
3. No regression in `train_mem_mape` and `infer_mem_mape`.
4. Lower RMSPE.
5. Higher 5%Acc and 10%Acc.
6. Lower CPU p95 inference latency.
7. Fewer parameters.

Suggested Pareto score for quick sorting only:

```text
score = mean_mape
      + 0.20 * max(0, train_time_mape - baseline_train_time_mape)
      + 0.20 * max(0, infer_time_mape - baseline_infer_time_mape)
      + 0.05 * max(0, cpu_p95_ms - 10)
```

Do not use this score as the only decision metric. Always inspect per-metric regressions.

---

# Implementation Checklist

## `src/perfseer/model.py`

- [ ] Add configurable MLP builder with activation, dropout, LayerNorm options.
- [ ] Add stream encoder LayerNorm.
- [ ] Add `ResidualGate`.
- [ ] Add `block_norm` option: `none`, `prenorm`, `postnorm`.
- [ ] Add `SynMMPlus`.
- [ ] Add optional attention graph pooling.
- [ ] Add `include_u_in_edge_update` flag.
- [ ] Add `mlp_z_num_linear_layers` flag.
- [ ] Add `SeerNetMulti` with six heads.
- [ ] Ensure `count_parameters` supports all variants.

## `src/perfseer/data.py`

- [ ] Add `use_operator_type_onehot` flag.
- [ ] Add topology node features.
- [ ] Add critical-path node/global features.
- [ ] Add edge topology features.
- [ ] Add destination tensor metadata if available.
- [ ] Add `time_target_mode` flag.
- [ ] Include feature flags and dimensions in dataset cache key.
- [ ] Persist full feature normalization stats.

## `src/perfseer/train.py`

- [ ] Add YAML/JSON config loading.
- [ ] Add AdamW.
- [ ] Add weight decay.
- [ ] Add gradient clipping.
- [ ] Add Huber and log-cosh losses.
- [ ] Add EMA/checkpoint averaging option.
- [ ] Add multi-task training loop.
- [ ] Add PCGrad.
- [ ] Add metric loss weights.
- [ ] Save full run metadata.
- [ ] Write result row to `runs/results.csv` or `.jsonl`.

## `src/perfseer/eval.py`

- [ ] Evaluate single-output and multi-output checkpoints.
- [ ] Add CPU benchmark mode.
- [ ] Add calibration application.
- [ ] Add batch-scaled and raw time reporting when relevant.
- [ ] Load normalization stats from checkpoint metadata.
- [ ] Export per-sample predictions to CSV for error analysis.

## New files

- [ ] `src/perfseer/losses.py`
- [ ] `src/perfseer/pcgrad.py`
- [ ] `src/perfseer/calibration.py`
- [ ] `src/perfseer/bench.py`
- [ ] `scripts/run_sweep.py`
- [ ] `scripts/summarize_results.py`
- [ ] `configs/*.yaml`

---

# Error Analysis Required Before Final Selection

For top candidates, generate plots/tables grouped by:

- Architecture family if available from filename/metadata.
- Batch size.
- FLOP range buckets.
- Node count buckets.
- Edge count buckets.
- Memory usage buckets.
- Time target magnitude buckets.

Save:

```text
runs/<run_id>/error_by_bucket.csv
runs/<run_id>/worst_100_predictions.csv
runs/<run_id>/prediction_scatter_<metric>.png
runs/<run_id>/residual_by_bucket_<metric>.png
```

Focus inspection on high-error `train_time` and `infer_time` samples.

---

# Final Deliverables

The coding agent should produce:

1. A reproducible baseline run.
2. A regularized single-metric model candidate.
3. A structural/topology-enhanced accuracy candidate.
4. A multi-output CPU deployment candidate.
5. A result ledger comparing all experiments.
6. CPU latency benchmark results.
7. A final recommendation with:
   - `accuracy_champion` checkpoint path.
   - `scheduler_deploy` checkpoint path.
   - Full per-metric MAPE/RMSPE/5%Acc/10%Acc.
   - Parameter counts.
   - CPU p50/p95 inference latency.
   - Known regressions and risks.

---

# Default First Implementation Path

Start with this exact path:

```text
1. Implement P0 metadata + CPU benchmark.
2. Implement LayerNorm/dropout/AdamW/Huber/grad clipping.
3. Run A1-A7.
4. Implement gated residual + two-block variants.
5. Run B1-B8.
6. Implement topology and critical-path features.
7. Run C1-C7.
8. Implement SeerNetMulti + PCGrad.
9. Run D1-D6.
10. Run seed robustness on top 3 candidates.
11. Select accuracy_champion and scheduler_deploy.
```

Do not begin with a large architecture rewrite. The current model is already strong, so the highest-value path is controlled changes with reproducible measurement.
