# A100 transfer: training-time and training-SM diagnosis

## Scope

Read-only diagnosis of the completed A100 teacher-transfer checkpoint
`transfer-best.pt` (selected epoch 14) on the fixed 25% dataset. No weight,
label, or source edit was made.

## Observations

| Metric | Train within 5% | Held-out test within 5% | Test R2 |
|---|---:|---:|---:|
| Training step wall time | 36.18% | 12.48% | -0.260 |
| Training step GPU time | 36.58% | 13.23% | -0.261 |
| Training epoch time | 33.18% | 6.94% | 0.198 |
| Training average SM utilization | 46.50% | 13.79% | 0.258 |
| Inference average SM utilization | 75.62% | 52.16% | 0.941 |

The held-out test target marginals are covered by the train 1st--99th
percentile interval: 0.00% of test step-time targets and 0.38% of test
training-SM targets are outside it. Therefore the large drop is not explained
by target-range extrapolation.

There are no target collisions: every repeated `input_path` maps to an
identical 12-element Ground Truth vector. Thus conflicting labels for an
identical feature input are not the source.

The group-wise split has zero shared `group_id` and zero shared `input_path`
between train and test (93 train groups versus 13 test groups). The test
failure therefore measures unseen-architecture generalization, not retrieval
of a previously observed input.

## Causal conclusion before repeatability testing

The immediate model-side failure mode is a generalization gap to unseen
training graphs. The A100 transfer freezes the node, edge, and global encoders
plus all ten message-passing blocks; it trains only phase adapters and the six
heads (39,385,612 of 274,598,156 parameters, 14.34%).  That made frozen
representation adaptation a testable hypothesis, rather than an established
root cause.

This does **not** establish that the six-head topology is itself defective:
training and inference SM use separate singleton heads and receive equal loss
weight, yet inference-SM generalizes much better. Nor is a global output
collapse present: test prediction standard deviations are 1.005x and 1.015x
the Ground Truth standard deviations for training step time and training SM,
respectively.

## Measurement and optimization facts

Training labels are not one-shot timings: each phase runs for at least five
seconds and is sampled every 10 ms. This rules out a trivial one/few-sample
telemetry explanation, but it does not by itself quantify NVIDIA Management
Library (NVML) sampling-window noise or runtime-state variability.

The loss minimizes group-balanced mean relative error, not within-5% hit rate.
This can worsen threshold hit rates but cannot explain the train/inference-SM
asymmetry by output weighting: each is its own one-output head group with the
same one-sixth loss weight.

## Completed Experiment A — fixed-configuration repeatability

On 2026-09-21, the pre-registered 12-configuration held-out panel (two
configurations from each of six modalities) was measured three times each in
fresh GPU-0 worker processes.  The hardware, executable configuration, seed,
warmup, phase duration, and sampling code were unchanged.  Thus each
configuration produces three measurements of the same intended target.

| Target | Median pairwise disagreement | 95th percentile | Pairs within 5% | Pairs within 10% |
|---|---:|---:|---:|---:|
| Training step wall time | 1.50% | 54.67% | 55.56% | 55.56% |
| Training step GPU time | 1.50% | 54.67% | 55.56% | 55.56% |
| Training epoch time | 1.50% | 54.67% | 55.56% | 55.56% |
| Training average SM | 3.44% | 52.88% | 55.56% | 55.56% |
| Inference step wall time | 0.94% | 1.89% | 100.00% | 100.00% |
| Inference step GPU time | 0.97% | 1.92% | 100.00% | 100.00% |
| Inference average SM | 3.33% | 9.04% | 72.22% | 100.00% |

The original test-set label also differs from the new three-run median by a
median 15.54% for all three training-time targets and 11.05% for training
average SM (six of twelve configurations are within either 5% or 10%).  By
contrast, original versus remeasured inference time has median discrepancy
2.13% (12/12 within 5%).

This is direct evidence that the training-time and training-SM labels are
heteroscedastic: some configurations are repeatable, but a substantial subset
is not reproducible at either requested threshold.  Therefore measurement
variability is a proven contributor to the low held-out 5% and 10% rates for
these outputs.  It cannot be the whole explanation, because the median
same-configuration disagreement is below 5%; the final-block ablation is
required to quantify the remaining representation-capacity contribution.

The repeat traces also rule out an accidentally changed captured graph.  Each
of the three captures has the same audit; a recursive comparison found exactly
one difference, `graph.metadata.configuration_id`, which is an identifier and
not a graph tensor or operation.  For the bf16 text configuration `calib_8360`,
the otherwise identical repeats produced mean per-step CUDA-event times of
4.131, 2.228, and 2.233 ms and training-SM means of 10.83%, 20.15%, and
19.76%.  CUDA-event time is measured on the device, so CPU scheduling alone
cannot produce that timing discrepancy.  The experiment establishes
process-level device execution variability, while leaving its mechanism
(kernel selection versus an unrecorded device state) open.

## Linking repeatability to the frozen-model errors

The fixed epoch-14 teacher-transfer prediction was rescored only on the same
12 held-out configurations.  This does not retrain or select a checkpoint.

| Target | Model vs. original stored label: median error / hits at 5%, 10% | Model vs. three-run median: median error / hits at 5%, 10% |
|---|---:|---:|
| Training step wall time | 18.35% / 1, 3 | 9.20% / 1, 6 |
| Training step GPU time | 18.43% / 1, 3 | 9.04% / 1, 6 |
| Training epoch time | 18.96% / 0, 4 | 7.34% / 2, 8 |
| Training average SM | 19.79% / 3, 3 | 21.40% / 2, 3 |
| Inference step wall time | 2.36% / 9, 10 | 2.13% / 9, 9 |
| Inference step GPU time | 2.80% / 9, 10 | 2.27% / 9, 9 |
| Inference average SM | 4.05% / 7, 10 | 4.75% / 6, 11 |

For the three training-time outputs, replacing the one-run stored value with
the repeat median nearly halves median error and materially raises 10% hits.
The effect does **not** occur for training SM: its error is unchanged within
the small panel's resolution.  Therefore the evidence separates the causes:
unstable timing labels materially depress timing accuracy, whereas training-SM
error remains after this noise correction and needs the capacity ablation.

## Completed Experiment B — final-block adaptation

The frozen parent, split, labels, seed, effective batch 64, microbatch 16,
learning rate 2e-4, 30-epoch ceiling, and early-stopping rule are held fixed.
Only message-passing `blocks.9` was unfrozen.  This changes trainable capacity
from 39,385,612 / 274,598,156 (14.34%) to 62,346,252 / 274,598,156 (22.71%).
The run at `last-block-unfreeze-v1` selected epoch 14 by the unchanged
validation rule, then performed its one permitted held-out-test evaluation:

| Target | Frozen held-out test, within 5% / 10% | Final-block held-out test, within 5% / 10% |
|---|---:|---:|
| Training step wall time | 12.48% / 24.48% | 20.92% / 27.20% |
| Training step GPU time | 13.23% / 24.48% | 20.92% / 27.20% |
| Training epoch time | 6.94% / 16.51% | 12.66% / 20.08% |
| Training average SM | 13.79% / 30.39% | 7.41% / 21.48% |
| Inference average SM | 52.16% / 87.24% | 64.82% / 90.71% |

The additional capacity improves all three structural timing outputs and
inference SM, but lowers training-SM accuracy at both requested thresholds.
It therefore rejects the claim that one frozen final graph block is the main
cause of low training-SM accuracy.  It supports a narrower claim: frozen
representation capacity contributes to timing error, while the training-SM
failure has another primary cause.

## Completed input and deterministic controls

For two high-variance configurations, five fresh executions were repeated
under the original policy, then after preloading every deterministic GPU input
outside the timed phase.  Training-step GPU-time disagreement did not fall:
it changed from 22.32% to 39.72% (time series) and from 29.59% to 31.88%
(text).  Training-SM disagreement also did not fall: 20.67% to 34.80% and
28.55% to 39.26%, respectively.  This rejects the input-generation/copy path
as the dominant mechanism.

The same two configurations were then repeated five times with cuDNN
determinism, PyTorch deterministic algorithms, and the required deterministic
CuBLAS workspace configuration.  Training-step GPU disagreement remained
23.00% (time series) and 34.63% (text); training-SM remained 22.11% and
37.37%.  Inference step time remained below 1.19% median disagreement.  Thus
kernel nondeterminism is not the dominant mechanism either.

## Completed hardware-state control

The telemetry panel completed ten fresh default-policy executions (five per
configuration), preserving one semantic graph hash per configuration.  In all
training samples, both configurations had fixed 1410-MHz SM clock, 1512-MHz
memory clock, P0 performance state, and 42--43 C temperature.  Mean power
occupied only 70.27--71.23 W.  During this particular stable interval,
training-step GPU disagreement was 4.02% for text and 1.59% for time series;
training-SM disagreement was 2.32% and 2.29%.

Therefore clocks, power, thermal state, and P-state cannot explain the earlier
large swings: they did not change inside this panel, and the same immutable
graph has been both stable and unstable at different times.  The experiment
does not claim that every unobserved execution detail is harmless; it removes
the four directly measured hardware-state candidates.

## Supported root cause

The benchmark label called `train_step_gpu_ms` is an elapsed CUDA-stream
interval, not active-kernel time.  Its start event is recorded immediately
before the Python `train()` closure, while the end event is recorded only after
that closure returns.  The closure includes input construction/copies,
optimizer zeroing and stepping, autocast, forward computation, backward
computation, and Python launch orchestration.  Thus a GPU idle gap after the
start event but before a later kernel launch is part of the target.  The
near-equality of reported training wall and CUDA-event times, together with
low average SM utilization, is consistent with this mixed pipeline interval;
it is not evidence of an intrinsic, graph-only device-runtime function.

The static predictor is given graph/configuration features but not worker
scheduling, stream-queue timing, or the per-run duty cycle that the interval
and 10-ms SM sampling observe.  Direct repeats prove that this omitted state
can change labels for the same graph.  Preloading inputs proves that input
creation alone is not dominant; deterministic-kernel controls prove kernel
algorithm choice is not dominant; hardware telemetry removes clock/power/
thermal/P-state changes.  The remaining supported cause is therefore
unobserved host-to-device launch/queue execution state within the benchmark
definition, rather than the six-head architecture or a one-block frozen
transfer bottleneck.
