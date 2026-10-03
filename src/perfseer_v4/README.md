# PerfSeer v4

v4.0 predicts seven training metrics from one training execution graph. The three
heads predict wall/GPU step time and epoch time (ms), average SM utilization (%),
and average/peak NVML VRAM and peak Torch allocated memory (MiB), in that order.
The training graph includes forward, loss, backward, and optimizer operations.
No inference execution graph is required. Historical v3.2 remains separate.

The package deliberately reuses v3.2's numerical training-feature layout for
checkpoint conversion parity. Its training-only feature wrapper, normalization,
dataset, input, output, loss, metrics, and checkpoints have v4 identities. The
teacher retains width 1280 and ten blocks; the student width 224 and two blocks.
Each of the three task heads receives one-third of the loss. Timing/memory use
scaled positive softplus; SM uses 100 times sigmoid. Distillation retains
`0.6 * hard + 0.4 * soft + 0.05 * training_phase_relation`.

Use the repository launcher, which bootstraps the existing package aliases:

```bash
python scripts/run_perfseer_v4.py prepare \
  --source src/perfseer_v3.2/dataset_with_label/ready_for_train_12 \
  --output src/perfseer_v4/dataset_with_label/ready_for_train
python scripts/run_perfseer_v4.py verify \
  --dataset src/perfseer_v4/dataset_with_label/ready_for_train
python scripts/run_perfseer_v4.py convert \
  --source predictor_v3.2/weights/predictor_v3.2_student.pt --output record/perfseer-v4/converted.pt
python scripts/run_perfseer_v4.py train --help
# Explicitly launches a full training campaign; not part of installation/tests:
python scripts/run_perfseer_v4.py train --output record/perfseer-v4/fresh-training
```

Pass the actual local checkpoint path to conversion. Conversion accepts compatible
v3.2 teacher or student checkpoints/exports, checks contracts, and retains only
training weights/scales and the training normalization. It records source hashes
and drops optimizer state, training progress, and prior accuracy gates. Ordinary
v4 loading rejects v3.2 checkpoints. Converted weights still reflect multitask
training; only a fresh controlled campaign can measure the effect of removing
inference objectives. Source dataset and hardware identity remain bound.

Dataset projection preserves active training targets, every row/split/group, and
original native measurements and label-policy provenance. Revised shorter-time
references are not new measurements. Source files are not rewritten. Generated
datasets/checkpoints/logs are local artifacts excluded from Git.

`export` consumes `--checkpoint` and `--output`. `predict` also takes `--designs`,
a JSON list of v4 training-only designs, and writes seven-key prediction objects.
The Python interface is `perfseer_v4.inference.predict(artifact, designs, ...)`.
`verify --predictions` independently reconstructs exported metric counts. The
production gate remains at least 95% within 5% relative error on every target;
validation and test must pass before teacher-to-student distillation proceeds.
Training keeps existing T1/S1 schedules, BF16-backbone/FP32-head execution,
whole-batch OOM retries, and strict resume checks. Fewer outputs are not an
accuracy claim.

## Baseline implementation verification

The initial v4.0 implementation passed 37 focused tests and 41 v3.2/calibration
regression tests. The shipped student converted with exact FP32 prediction parity
on 24 saved validation graphs spanning six modalities and four precisions. Its
parameter count changes from 2,888,744 to 2,281,251. A bounded single-thread CPU
forward check (three repeats per case) measured aggregate v4 time at 75.2% of
v3.2 time; this excludes capture, training, serving overhead, and GPU execution.
Local details are in `record/perfseer-v4/baseline-verification.json`. Neither
conversion nor these software checks establish improved prediction accuracy.

## Runnable transfer candidates

| Variant | Source model | Target adaptation |
| --- | --- | --- |
| v4.0 | T1/S1 training graph predictor | Unchanged source or full fine-tuning control |
| v4.1 | Frozen trained v4.0 | Seven independent regularized residual regressions |
| v4.2 | Frozen trained v4.0 backbone and heads | Three hardware-conditioned rank-eight adapters |
| v4.3 | Resource MLP, two width-128 GELU layers | Seven independent RBF Gaussian-process residuals |

The S1 v4.2 adapters add 67,104 trainable parameters. The default v4.3 source MLP
has 35,591 parameters. v4.3 inputs are the ordered union of existing static
calibration descriptors, without intercept or source prediction, plus epoch
length and its missing flag: 17 descriptors followed by 62 fixed-reference
hardware values and 62 missing masks. `resource_model.RESOURCE_NAMES` fixes the
order. No message-passing backbone is used by this model. Workloads with different
graphs can share resource summaries; these collisions limit its expressiveness.

### Residual equations and fitting

For the actual frozen source prediction `b`, target observation `y`, and fitted
correction `delta`, each output uses its own residual and feature map:

| Outputs | Residual | Corrected prediction |
| --- | --- | --- |
| Three times | `log(y) - log(b)` | `b * exp(delta)` |
| Three memories | `(y-b) / max(b, 64 MiB)` | `max(0, b + max(b, 64 MiB) * delta)` |
| SM | `(y-b) / 100` | `clip(b + 100 * delta, 0, 100)` |

Memory residual arithmetic uses bytes internally; public values remain MiB.
Residuals never substitute measured source labels for source predictions. SM
uses timing descriptors; epoch time additionally uses epoch length and missingness.
Each task fits its own scaler on its valid fit rows. Every independent architecture
group receives equal total weight; positive row weights sum to one. v4.1 limits
the feature count, including its intercept, to `max(1, floor(fit_groups / 2))`.
Small budgets can therefore reduce it to a constant correction.

With design matrix Phi and W the diagonal matrix of normalized weights,

```text
J(w) = (r - Phi w)^T W (r - Phi w) + lambda w^T w
gradient J = 2 [(Phi^T W Phi + lambda I)w - Phi^T W r]
Hessian J = 2 (Phi^T W Phi + lambda I)
u^T Hessian J u = 2 ||sqrt(W) Phi u||^2 + 2 lambda ||u||^2 > 0
```

The last inequality holds for nonzero u and positive lambda, even with deficient
rank. The implementation solves the augmented system `[sqrt(W)Phi; sqrt(lambda)I]`
against `[sqrt(W)r; 0]` in float64. All coefficients, including the intercept, are
regularized. Validation chooses lambda from `1e-4, 1e-3, 1e-2, 0.1, 1, 10`.
Constant residual and physical additive affine corrections are separate controls.

### Hardware adapters

For each of the three heads, independently,

```text
z'_g = (1 + gamma_g(h)) * z + beta_g(h) + B_g A_g z / 8
```

The hardware network is Linear → GELU → Linear with hidden width 32. Its final
layer and B start at zero; A starts randomly. This guarantees `z'_g = z` initially.
At initialization A's gradient is zero, while B and the final hardware layer can
receive gradients. The backbone and heads stay in evaluation mode and retain
their exact weights; gradients still pass through the frozen heads to adapters.
This combines [FiLM conditioning](https://arxiv.org/abs/1709.07871) with a
[low-rank update](https://arxiv.org/abs/2106.09685); transfer improvement is a
testable hypothesis rather than a consequence of either construction.

Training minimizes the seven-target loss plus
`1e-3 * ||theta - theta_initial||^2 / adapted_parameter_count`.
Defaults are AdamW with zero weight decay, learning rate 1e-4, effective batch 64,
microbatch four, at most 100 epochs, and validation early stopping after epoch 20
with patience ten. The fine-tuning control uses the same schedule and updates
all v4.0 weights. An untrained, median-only graph model is rejected for v4.2.
Single-GPU source data cannot establish zero-shot hardware generalization.

### Resource model and Gaussian processes

The v4.3 source MLP uses source-training-only resource standardization and the
same three physical-output heads and equal-head loss. Source training defaults:
AdamW, learning rate 1e-3, weight decay 1e-4, batch 256, at most 100 epochs, and
the same early-stopping rule. Target fitting freezes this separately trained MLP.
The GP sees only the 17 resource descriptors, standardized on target-fit rows.

For a unit-amplitude RBF kernel and normalized group weights W, the working
observation-noise covariance is D = lambda W^-1. The joint Gaussian of observed
residuals and a latent query residual has covariance blocks `[C, k; k^T, 1]`, where
`C = K + D + 1e-8 I`. Conditioning gives

```text
posterior mean = k^T C^-1 r
posterior variance = 1 - k^T C^-1 k
```

These are the [GPML Chapter 2 conditional-Gaussian equations](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
The implementation uses float64 Cholesky solves and validates duplicate inputs
and constant features. Validation chooses length scale from `0.5, 1, 2, 4` and
lambda from the ridge grid. `b * exp(posterior_mean)` is the modeled timing
posterior median; its mean would additionally contain `exp(variance / 2)`.
The reported variance describes latent residual uncertainty, not a validated
deployment confidence interval. Query noise is not added to it.

## Commands and comparison protocol

The following commands explicitly run experiments; implementation verification
uses only bounded synthetic updates, never a full campaign:

```bash
# First train the independent v4.3 source on the source GPU dataset.
python scripts/run_perfseer_v4.py train-resource \
  --dataset /path/to/source-v4-dataset --output record/v4-resource-source

# Fit and select on target train/validation rows; use a trained/converted v4.0 source.
python scripts/run_perfseer_v4.py transfer \
  --dataset /path/to/target-v4-dataset --source /path/to/v4-source.pt \
  --resource-source record/v4-resource-source/resource-best.pt \
  --environment /path/to/target-environment.json --output record/v4-transfer

# Separately open the held-out test labels after selection is sealed.
python scripts/run_perfseer_v4.py evaluate \
  --selection record/v4-transfer/selection.json --output record/v4-evaluation.json

# Example residual prediction; use the resource source checkpoint for a v4.3 GP.
python scripts/run_perfseer_v4.py predict --checkpoint /path/to/v4-source.pt \
  --designs /path/to/training-designs.json \
  --calibration record/v4-transfer/budget-32-seed-11-v4.1.json \
  --environment /path/to/target-environment.json --output record/v4-predictions.json
```

`--variants source constant affine v4.1 v4.0-finetune v4.2` omits the separately
trained resource model. Neural candidates also support `export` and `predict`;
adapted prediction requires its exact target environment. The environment uses
the existing `perfseer_v32.calibration_contracts.ENVIRONMENT_VERSION` schema:
hardware_id, gpu_model, capacity_bytes, partition, driver, cuda, framework,
libraries, allocator, backend_policy, and host_context. Supply actual measurement
context. Source identities bind weights, model, normalization, preprocessing,
PyTorch version, CPU threads, and CPU/FP32/microbatch-one residual predictions.
Changing these invalidates calibration. Unknown target environments are rejected;
the Python residual API supports explicit `unknown_domain="source_fallback"`.

Default fit budgets are 32/64/128/256 recorded sample IDs, with seeds 11/29/47.
Sampling is target-blind, nested by budget, and balanced across available groups.
All candidates share fit, validation, and test IDs. Repeated training-input hashes
can appear under distinct recorded sample IDs; reports separately count distinct
inputs and independent groups rather than asserting that every row is independent.
Validation-label costs are reported separately. Prepared data cannot recover
repeat-profiling cost; no new measurements are collected by these commands.

Reports include all seven outputs, exact inclusive 5%/10% hit counts and rates,
physical MAE, median/p95 relative errors, memory underprediction, fitting cost,
and forward latency with its scope. Native measurements and active revised labels
receive separate metrics. Wall-step time and peak VRAM also receive group-bootstrap
paired error intervals against validation-selected controls. Validation selects
candidates per target before test access; dataset verification audits test integrity
but never uses test accuracy for selection. Reports preserve inconclusive outcomes
and cannot authorize deployment. Existing production accuracy gates are unchanged.

Run focused algebra verification with `python scripts/verify_perfseer_v4_math.py`.
Tests cover conversion/export parity, dataset lineage, independent loss/metric
reconstruction, adapter identity/gradients, frozen weights, GP equations, serialized
predictions, split leakage, and source/environment tampering. No implementation
check establishes improved transfer accuracy.

Final implementation checks: 228 repository tests passed; three existing A10
profile-dependent modules were skipped. All 25 v4 and 15 existing mathematical
checks passed. A synthetic three-row dataset exercised the real launcher through
preparation, verification, one-epoch resource training, all seven comparison
candidates, held-out evaluation, export, and prediction with each adapted variant.
Public prediction replay independently reproduced active/native hit counts, MAE,
and error tails. The wheel built and imported outside the checkout. Local evidence
is under `record/perfseer-v4/`, including `regressions.log`, `math-verification.json`,
`dataset-verification.json`, and `cli-smoke/`. The full dataset projection preserves
40,020 rows and 119 groups while reducing 2,352 paired inputs to 936 training-only
inputs. These checks ran no full training, GPU profiling, or Nautilus campaign.
