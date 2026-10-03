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
