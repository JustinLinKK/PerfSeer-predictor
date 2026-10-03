# PerfSeer v3.1

The directory is deliberately named `perfseer_v3.1`; its installed Python package
is `perfseer_v31` (`python -m pip install -e . --no-deps --no-build-isolation`).
Existing v3 predictor code and checkpoints are not modified.

## Contract

Both T1 (1280 wide, 10 blocks) and S1 (224 wide, 2 blocks) predict, in order:

1. `train_epoch_ms`
2. `train_avg_sm_util_percent`
3. `train_peak_vram_mib`

Predictions describe A10 workloads, irrespective of the A100 used to train the
predictor. No inference, allocator, uncertainty, OOM, or transfer heads remain.
Time and VRAM decode with positive softplus times training medians; SM uses
100 times sigmoid. Six-output checkpoints are rejected, not migrated.

## Dataset preparation

```bash
python -m perfseer_v31.dataset migrate
python -m perfseer_v31.dataset prepare --workers 6
python -m perfseer_v31.dataset verify
```

The legacy snapshot is `dataset_with_label/legacy_v3`; the v3 dataset path is a
compatibility symlink. Original archives and source measurements remain there.
The new records live separately in `dataset_with_label/ready_for_train`.
All 36,409 source rows must convert, with no microbatch filter or fallback graph.
An incomplete or failed conversion cannot produce a training manifest.

`generated_model_runtime.py` is the byte-pinned calibration runtime recovered from
the other local checkout (SHA-256
`c05fb507cabb8efd0c050a37f01bb58460d19e785872d5b3bf604187117a27fb`).
`recovered_contracts.py` and `recovered_generated_lineages.py` restore dependencies
omitted from the small archive. They are loaded into its archived namespace;
the archived factories, task registry, target width, and architecture parameters
remain authoritative. Legacy hardware naming occurs only in archived source and
provenance, not active prediction identity.

Each versioned input contains `INPUT_SPECS`, `NODE_SPECS`, `EDGE_SPECS`, and
`TRAINING_CONFIG`. The shared compiler traces actual eager autograd (including
custom losses and unused parameters), checks loss and every participating
parameter gradient, and preserves shapes, argument values, topology, and aliases.
CPU oneDNN is disabled so recurrent operators use native PyTorch decomposition.
The same backend context covers eager backward, tracing, and replay. Both
collections' FP16/BF16/TF32 policy aliases and the small collection's
`mixed_structured` BF16 policy are explicit; unknown precision policies fail.
Calibration SGD uses momentum zero and calibration AdamW uses decay 0.01, as
specified by its profiler's constructor defaults; small-collection settings are
recovered independently from its profiling runtime.
Optimizer resource summaries remain explicitly estimated; backward graphs do not.
These are logical model-design graphs, not measured CUDA kernel schedules.

Workload batch size, accumulation, precision, optimizer, scheduler, loss, backend,
and epoch length are inputs. Source/model/team/modality strings are not learned.
Seed-42 groups co-locate configuration variants and conservative normalized
architecture matches, including cross-source duplicates, before an approximately
80/10/10 split. Full-training streaming moments fit normalization with min/max
clipping; validation and test never fit normalization.

## Training

```bash
python -m perfseer_v31.runner --output /outputs/perfseer-v31-a100 --resume
```

Production requires an A100. Muon handles hidden linear matrices; AdamW handles
embeddings, the prediction head, biases, and normalization. The teacher runs all
600 epochs with 10-epoch warmup; student distillation runs 100 epochs with 5-epoch
warmup. Both use cosine decay to 1% of peak LR, BF16 autocast, FP32 equally weighted
time/VRAM relative errors and SM absolute error/100, and norm-1 gradient clipping.
Effective predictor batches are 256 (the last batch can be smaller); a memory
probe chooses a microbatch up to 64 and allocation retries never drop examples or
change the recorded workload batch size.

Validation runs each epoch. Best selection minimizes the worst output error,
then mean error, then epoch. SIGTERM/SIGINT save an optimizer-step boundary and
exit 75; latest checkpoints include RNG, both optimizers/schedulers, normalization,
data identity, and the deterministic shuffle cursor. Best and latest are separate.

After epoch 600, restore the best teacher. Each validation output must have error
strictly below 0.10: time MAPE, SM MAE/100, and VRAM MAPE. Only then evaluate the
untouched test partition once. Failure stops distillation; test never chooses a
different teacher. A passing teacher is frozen for 100 student epochs with 0.6
hard-label, 0.4 normalized teacher-prediction, and 0.05 relational representation
loss. Relations compare global and phase tokens within each graph so accumulation
does not change the objective. Student selection uses validation and reports test
separately. Passing these accuracy gates is an experiment, not an implementation
guarantee.

## Verification and deployment

```bash
python -m pytest -q tests/test_perfseer_v31.py tests/test_perfseer_v3_combined_training.py
python -m perfseer_v31.verification --dataset src/perfseer_v3.1/dataset_with_label/ready_for_train --output record/perfseer-v31-verification --mode features
docker build -f src/perfseer_v3.1/containers/training/Dockerfile -t perfseer-v31:verified .
```

Full conversion, bounded overfit, exact-image CUDA checks, and regression tests
must pass before an immutable image is published and an A100 Job is submitted.
Check quota and persistent capacity immediately before submission. Deployment
must use `nvidia.com/a100: 1` in `ecepxie`, separate v3.1 output paths, immediate
status/log/event collection, and a durable monitor under `record/`.
