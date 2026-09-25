# A100 transfer: cause of low training-epoch-time accuracy

## Question

Why does `train_epoch_ms` have low held-out accuracy when training step wall
time is already difficult to predict?

## Metric trace and hard label identity

The labeler executes each training profiling block with
`block_steps = steps_per_epoch`.  Each such block is exactly one training
epoch.  It stores `wall_ms` per block, then defines

\[
  y_{\mathrm{epoch}} = \operatorname{mean}_{b}(b.\mathrm{wall\_ms}),
  \qquad
  y_{\mathrm{step}} =
  \frac{\sum_b b.\mathrm{wall\_ms}}{\sum_b b.\mathrm{steps}}.
\]

Because every block has the same `steps_per_epoch = n`, this is the exact
constraint

\[
  y_{\mathrm{epoch}} = n\, y_{\mathrm{step}}.
\]

No independently measured epoch-duration target exists.  The stored auxiliary
field `train_epoch_ms_step_extrapolated` is the same relation explicitly.

## Held-out projection experiment

For every row, derive the known integer \(n\) from the Ground Truth relation
and replace only the epoch prediction with
\(\hat y_{\mathrm{epoch}} = n\hat y_{\mathrm{step}}\).  This is a
post-processing constraint, not a fitted parameter or a test-set-selected
model.  Ground Truth label residuals from the identity were at most
\(7.74\times10^{-6}\%\) on the 1,066-row test split and
\(6.82\times10^{-6}\%\) on the 1,136-row validation split (float32
rounding only).

| Model / split | Raw epoch within 5% / 10% | Constraint-projected epoch within 5% / 10% | Mean / median / 95th-percentile prediction constraint violation |
|---|---:|---:|---:|
| Frozen transfer, validation (1,136) | 13.29% / 22.18% | 14.26% / 24.91% | 12.53% / 2.95% / 63.48% |
| Frozen transfer, held-out test (1,066) | 6.94% / 16.51% | 12.48% / 24.48% | 14.94% / 2.98% / 83.71% |
| Final-block transfer, validation (1,136) | 15.58% / 26.50% | 18.66% / 28.17% | 13.04% / 2.11% / 117.04% |
| Final-block transfer, held-out test (1,066) | 12.66% / 20.08% | 20.92% / 27.20% | 11.01% / 2.24% / 60.55% |

The projected held-out rates equal the corresponding training-step-wall rates,
as required by the exact relation.  No model can use an independently learned
epoch head to improve on a correct step-wall estimate for this target.

## Cause and boundary of the claim

The model uses one three-output time head and a per-output relative-error loss;
it has no structural constraint tying outputs 0 (step wall time) and 2 (epoch
time).  Consequently it emits inconsistent pairs, with 14.94% mean epoch/step
contract violation for the frozen model on held-out data.  That avoidable
inconsistency costs 5.53 percentage points within 5% and 7.97 points within
10% on held-out epoch accuracy.

The remaining projected epoch error is exactly the step-wall error.  Its cause
is therefore the previously established mixed host-to-device training-interval
label variability; multiplying by a known step count preserves relative error.
This result does not claim that a new model was trained or that the projected
rate fixes the stochastic runtime-label component.
