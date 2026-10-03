# Frozen-source training time and VRAM calibration

This implements the first, reversible experiment in
[`PerfSeer_Training_Time_VRAM_Refactor_Plan.md`](../../PerfSeer_Training_Time_VRAM_Refactor_Plan.md).
The deployed source checkpoint and its normalization remain frozen. Independent
float64 weighted ridge models correct `train_step_wall_ms` multiplicatively and
`train_peak_vram_mib` additively. Memory calculations use bytes internally and
convert to MiB at the existing API boundary. All twelve named outputs remain
present. The other ten outputs retain their source values and are explicitly
listed as unadapted by `predict_calibrated`.

The primary duration is mean end-to-end **logical optimizer update** time,
including accumulation, input tensor generation and transfer. It is distinct
from CUDA-event elapsed time and active kernel duration. Native repeat run means
are averaged before fitting log residuals. The primary memory quantity is the
arithmetic mean of run peaks sampled by whole-device NVML. Allocated memory,
reserved memory and per-process memory are different quantities. No missing
label or OOM is converted to zero. Native archives and historical label policies
are never rewritten.

`calibration_contracts.py` defines versioned workload, environment, observation,
measurement and target contracts. Workload identity includes the model source
specification, executed tensor shapes, complete training configuration and
semantic dataset identity. Audit rejects cross-split groups, anchors, workloads,
copied measurements and undeclared duplicate configurations. Aliases reference
one observation and carry no additional fit weight. Repeatability reports count
actual saved GPU runs separately from unique configurations and groups.

`calibration.py` fits train-only feature scalers and regularizes every coefficient,
including the intercept. Default static task maps each have twelve features;
capacity is reduced to at most half the independent fit-group count. Selection
uses the fixed lambda grid on validation labels, separately for each primary
objective. Source-normalized features are not refitted. Artifacts bind source
weights, target scales, normalization, model configuration, preprocessing code,
feature order, CPU inference settings and the complete target environment.
Teacher coefficients cannot be applied to a different student.

The existing `predict` function accepts optional `calibration` and `environment`
arguments and retains its list-of-twelve-output-dictionaries return value. Its
default path is unchanged. Adapted inference currently requires CPU, FP32 and
predictor microbatch one, matching calibration preparation. To receive statuses,
including an explicitly requested source fallback, use:

```python
from perfseer_v32.calibration import load_adapter
from perfseer_v32.inference import predict, predict_calibrated

adapter = load_adapter("adapter.json")
result = predict_calibrated(source_artifact, designs, adapter, query_environment,
                            unknown_domain="source_fallback")
predictions = result["predictions"]
print(result["status"], result["unadapted_targets"], result["negative_raw_memory_count"])

# Rollback uses the unchanged source checkpoint.
predictions = predict(source_artifact, designs, calibration=None)
```

The default unknown-environment policy rejects the request. An environment
fingerprint includes GPU identity/capacity, partition, driver, CUDA/framework and
library versions, allocator, backend policy and host context. Historical missing
fields are explicitly `unknown_not_recorded`; they do not match newly supplied
values automatically. Preserve the actual query environment rather than copying
an adapter's environment to bypass this check.

`train_epoch_ms` remains an unadapted auxiliary output for exact zero-adapter API
identity. For a declared fixed-shape schedule, use
`calibration_contracts.schedule_duration` to derive epoch and total duration from
the calibrated step time and known dataset/batch/accumulation counts. It rejects
variable final batches and partial accumulation, which need separate step-shape
classes. Counts never come from held-out timing labels.

## Commands

The launcher uses the checkout package aliases and the existing
`PERFSEER_PYTHON` fallback. It runs on CPU and submits no cluster job. The frozen
`predictor_v3.2/source` snapshot is preserved; use this checkout's runtime for the
new opt-in path.

```bash
python scripts/run_perfseer_v32_calibration.py provenance \
  --output record/time-memory-calibration/provenance.json
python scripts/run_perfseer_v32_calibration.py prepare \
  --output record/time-memory-calibration/data --workers 4
python scripts/run_perfseer_v32_calibration.py audit \
  --data record/time-memory-calibration/data
```

Defaults use the local deployed student, native `labels-24h.json.gz` campaign,
and verified paired transfer dataset. Override `--source`, `--campaign` and
`--dataset` for other verified artifacts. `--limit-per-split 4` is a bounded smoke
check, not an evaluation cohort. Workers are independent CPU processes, each
with one Torch thread; the maximum is eight. Preparation verifies raw attempts,
paired graphs, actual batch and precision, repeat values and epoch/step identity,
then stores frozen source predictions and static features. It records native
attempt hashes, repeat variability and source-training-panel uncertainty.
Outputs refuse to replace different existing artifacts; choose a fresh directory
for a new preparation. The original checkpoints and datasets remain intact.

Create a margins JSON **before evaluating test labels**. This is an illustrative
contract, not approved product tolerances:

```json
{
  "time_mae_ms": 0.1,
  "memory_mae_mib": 8.0,
  "relative_error": 0.01,
  "near_zero_relative_floor": 0.000001,
  "memory_underprediction_mib": 8.0,
  "hit_rate": 0.02,
  "warm_p95_fraction": 0.1,
  "minimum_test_groups": 10
}
```

Margins for relative-error metrics are absolute differences in relative error;
hit-rate margins are fractions (0.02 means two percentage points). The near-zero
floor is the minimum relative-error improvement required for superiority, never
a denominator used to invent percentage errors for zero targets.

```bash
python scripts/run_perfseer_v32_calibration.py select \
  --data record/time-memory-calibration/data \
  --margins path/to/predeclared-margins.json \
  --output record/time-memory-calibration/selection/selection.json
python scripts/run_perfseer_v32_calibration.py evaluate \
  --data record/time-memory-calibration/data \
  --selection record/time-memory-calibration/selection/selection.json \
  --output record/time-memory-calibration/test-report.json
```

Selection defaults to budgets 32/64/128/256 and seeds 11/29/47. Fit rows are chosen
using group, modality, precision, batch and static feature coverage, without
target magnitudes. Baselines are unchanged source, constant correction, affine
calibration, independent linear adapters and linear adapters augmented with a
static memory descriptor. All use the same fit rows, validation rows and test
rows. Validation and repeat costs are disclosed separately. The strongest
control is selected separately for each primary objective; averaging the two
objectives cannot hide a regression. Candidate and comparison choices and all
margins are sealed before the separate test command.

Reports include exact original-unit 5%/10% hits, median/p95 relative error,
absolute memory error and p95 underprediction, family/precision/batch/panel
breakdowns, grouped bootstrap intervals, and negative raw memory predictions.
Paired error differences use **candidate minus baseline**, so negative is
better. Fewer than the declared independent test groups, missing evidence, or
confidence intervals crossing a noninferiority margin produce `inconclusive`.

To include the existing transfer procedure, pass `--current-transfer` to both
selection and evaluation with a JSON list of trials. Each trial supplies
`budget`, `seed`, exact `fit_ids`, `preprocessing_fit_ids`, `distillation_fit_ids`,
`source_identity`, `domain_fingerprint`, and `checkpoint_sha256`. The selection
file supplies `validation_predictions`; the evaluation file supplies
`test_predictions`, each mapping sample IDs to the twelve named outputs. Query
coverage must match exactly, preprocessing/distillation IDs must be permitted
fit IDs, and checkpoint/provenance must match across stages. Reports from a
full-label teacher transfer do not qualify as a 32-label control.

```bash
python scripts/run_perfseer_v32_calibration.py benchmark \
  --adapter record/time-memory-calibration/selection/budget-32-seed-29-linear.json \
  --design path/to/paired-design.json.gz \
  --output record/time-memory-calibration/benchmark.json --intended-host
```

Benchmarking measures checkpoint loading, first query, paired warm query p95,
artifact sizes, CPU threads and concurrency. It explicitly reports graph
extraction as unmeasured when given a precomputed graph; measure workload capture
on the intended serving host separately. Supply a JSON list of benchmark reports
using evaluation's `--benchmarks` option. Reports bind the adapter fingerprint.
The public API benchmark includes model restoration and source-binding checks;
it does not claim an adapter-only latency represents complete serving cost.

Deployment packaging is gated:

```bash
python scripts/run_perfseer_v32_calibration.py package \
  --selection record/time-memory-calibration/selection/selection.json \
  --report record/time-memory-calibration/test-report.json \
  --budget 32 --seed 29 --output record/time-memory-calibration/deployment
```

This verifies the selected candidate and promotion evidence, then emits
`adapter.json` and `deployment.json` with hashes and an explicit rollback switch.
It references and rechecks the original checkpoint rather than replacing it.

## Memory baseline and conditional phases

`memory_baseline.storage_peak` accepts versioned storage traces with full backing
extents and inclusive lifetimes. Aliases count once; input/output coexistence and
temporary workspaces count during the operation. Persistent states use actual
bytes and declared lazy-initialization lifetimes. `process_peak` takes coincident
reserved and external samples and computes the peak of their sum.

Current GraphIR records tensor-view bytes, not complete backing-storage extents.
`graph_baseline` therefore returns an **approximation**, with unsupported workspace
operators and missing lifetime/allocator information visible. It is neither an
exact training-memory simulator nor a guaranteed lower bound. The analytic
ablation adds this static descriptor to the memory adapter while retaining the
same frozen source prediction and residual scale. It does not replace the source
memory estimate with a purported process-memory baseline. A candidate selected
from this ablation cannot pass promotion while its empirical trace gate remains
unverified.

Source branch retraining, source analytic-residual training, final-student
distillation, nonlinear adapters and deployment promotion are conditional on the
plan's empirical gates. They are not enabled by this first implementation. The
reviewed September 24 transferred pair and teammate failing-run artifacts are
not present in this checkout; the provenance command names the precise gaps.
The available local student/native RTX 5090 corpus supports independent
experiments, but does not reproduce that different failing run. Source-known
and source-unseen panels remain `unknown` until source-training membership is
verified. No accuracy improvement is established by algebra or software tests.

## Focused verification

```bash
python scripts/verify_transfer_math.py
PYTHONPATH="$PWD/scripts:$PWD/src" /home/justin/miniconda3/envs/perfseer/bin/python -c \
  'import run_perfseer_v32_transfer_labeling; import pytest; raise SystemExit(pytest.main(["-q", "tests/test_perfseer_v32_calibration.py", "tests/test_perfseer_v32.py"]))'
```

The mathematical verifier is the plan's original fifteen synthetic checks. The
regression suite additionally exercises actual ridge fitting, train-only scaling,
group weighting, missing/zero validity masks, split and alias leakage, frozen
checkpoint identity, real graph-capture/inference integration, serialization,
unknown-environment fallback, equal-budget selection and promotion guards.
