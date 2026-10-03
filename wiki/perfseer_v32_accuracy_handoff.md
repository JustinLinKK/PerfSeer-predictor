**PerfSeer v3.2 accuracy investigation — teammate handoff, 2026-09-06**

The local audit found an input/label ambiguity that limits timing and SM accuracy.
It does not establish the cause of the supplied screenshot: the screenshot's
twelve-output model and evaluation have not been provided. No predictor, loss,
checkpoint, dataset, or quality gate was changed, and no training job was started.

The local [v3.2 model](../src/perfseer_v3.2/model.py) inherits the v3.1 model and
predicts only `train_epoch_ms`, `train_avg_sm_util_percent`, and
`train_peak_vram_mib`. Twelve native measurement labels are retained in the dataset
and used by the separate transfer-labeling workflow; that does not create twelve
prediction heads. Please provide the actual twelve-output implementation.

The local [loss and evaluator](../src/perfseer_v3.1/training.py) use relative error
for time/VRAM, absolute SM error divided by 100, and `score = 1 - mean(error)`.
These scores are different from the fraction of samples within a tolerance.
For example, SM truth 10% and prediction 11% has 10% relative error, one percentage
point absolute error, and local normalized error 0.01. Thus a 99% local score does
not imply that 99% of predictions meet a 5% relative tolerance.

**Verified local evidence**

The audit checked all 32,108 training and 4,100 validation rows against the split
hashes and verified all 2,096 referenced input artifact paths. These partitions
contain 700 and 120 distinct input hashes respectively, with no hash overlap.
Repeated identical inputs often carry different timing and SM labels. For example,
these validation samples use the same input artifact, and their original generated
sources have identical `INPUT_SPECS` and `NODE_SPECS`:

| Model ID; same store-sales workload, batch 8, Adam, BF16 | Train epoch | Train SM average | Train VRAM peak |
| --- | ---: | ---: | ---: |
| `calib_4679` | 277.016 ms | 15.982% | 806.5625 MiB |
| `calib_4979` | 630.496 ms | 7.107% | 806.5625 MiB |

A deterministic predictor receiving the same input cannot return both labels.
This verifies a limitation in the current input/label contract. It does not yet
identify whether the original measurements differ because of execution settings
missing from the inputs, profiling conditions, source reconstruction, or another
measurement problem. Changing the number of layers cannot recover absent input
information. Model IDs are provenance, not a justified replacement for that information.

For each group of identical input bytes, the audit allowed an oracle to choose the
best prediction independently for each target and tolerance using that split's
labels. Under `abs(prediction - truth) <= tolerance * truth`, its maximum fraction
of hits on the **local validation set** is:

| Target | Maximum within 5% | Maximum within 10% |
| --- | ---: | ---: |
| Train SM average | 54.37% | 74.73% |
| Train epoch | 55.37% | 76.27% |
| Train step GPU | 55.37% | 76.22% |
| Train step wall | 55.37% | 76.27% |
| Inference SM average | 76.76% | 93.32% |
| Inference step GPU | 80.93% | 95.34% |
| Inference step wall | 80.98% | 95.34% |

These are optimistic per-output upper bounds for predictors using the current
inputs, not trained-model results, independent-test estimates, or ceilings for a
model with additional execution features. They are not asserted to describe the
screenshot's run. Both its split identity and its metric formula must be confirmed.
They nevertheless show a concrete reason the local training timing/SM targets are
harder than the inference targets.

Memory performance also needs a baseline: on this validation set, predicting the
training-set median for every row already puts 99.05% of inference-average VRAM
labels and 97.78% of training-peak VRAM labels within 5%. Near-perfect VRAM hit rates
alone therefore provide limited evidence about the model's predictive quality.

**Please send these existing artifacts, in priority order**

1. **Exact run identity and executable code.** Git commit plus local changes and
   untracked source, or the exact source bundle; immutable container digest if used;
   training/evaluation commands and resolved configuration; whether the screenshot
   evaluates teacher, student, ensemble, or a transferred model; source and target
   GPU identities. Include the actual model, feature extraction, loss,
   normalization, inference postprocessing, and evaluation/table-generation code.
   A version label or model diagram alone cannot identify the implementation.

2. **Checkpoint and output contract.** The exact selected checkpoint/export used
   for the screenshot, its SHA256 and selected epoch, ordered output names and
   units, model configuration, target transforms/scales, and fitted input
   normalization. Confirm whether every displayed output is predicted directly
   or calculated from another prediction. Include latest checkpoint metadata and
   the best-selection/early-stopping rule; optimizer state is useful if already
   packaged, but is not required to reproduce inference.

3. **Unrounded per-sample predictions and evaluation details.** CSV, Parquet, or
   JSONL containing `sample_id`, `split`, `input_sha256` or feature hash,
   `target_name`, `y_true`, `y_pred` in physical units, and label-validity mask.
   Include all evaluated rows, counts/exclusions per output, model/architecture
   group, modality, GPU, batch size, precision, optimizer, and steps per epoch, or
   a joinable metadata file. Provide the exact 5%/10% formulas, denominator,
   zero/near-zero policy, strict/inclusive boundary, clipping, inverse transforms,
   missing-label handling, and aggregation weights. Confirm SM relative percent
   versus absolute percentage points. Send existing train/validation predictions;
   include existing test predictions if the screenshot used test. Do not retune
   against the test set or collect only the worst examples.

4. **Training trajectory and effective settings.** `training.log`, all available
   `teacher-epoch-*.json` / `student-epoch-*.json`, final reports, and configuration
   overrides. Provide per-output training/validation losses and metrics by epoch,
   actual learning rates for each optimizer, loss weights, batch/accumulation,
   precision, seed, scheduler, clipping, best epoch, stop reason, and resume/OOM
   history. If available, include head/trunk gradient norms and clipping frequency.
   Mark unrecorded quantities as unavailable; do not rerun full training merely
   to manufacture a history. Aggregate loss alone cannot separate output-specific
   underfitting, overfitting, optimization trouble, and multi-task interference.

5. **Dataset and input lineage.** Dataset manifest/fingerprint, exact split sample
   lists and architecture groups, original labels/provenance, preparation code,
   and feature-cache version/fingerprint. If the fingerprint is
   `25d701408a189b2547a6a5f537e1a5425b36afc57cb9ecef95423f2c81ece4e7`, confirm it;
   the local data need not be transferred again. Otherwise send the changed data
   or the reproducible replacement. Include source models and actual feature
   tensors for the two example IDs above, plus a few typical and high-error
   workloads for each affected family. We need to distinguish identical workloads
   from genuinely different workloads collapsed by capture or preprocessing.

6. **Original profiling protocol and repeat evidence.** For the example pair and
   representative failing workloads, send raw per-step/per-epoch wall and CUDA
   event times, timestamped SM/VRAM samples, and any existing independent repeats.
   Include actual GPU model/UUID, driver/CUDA/cuDNN/PyTorch versions, clocks and
   power limits, throttling/concurrent processes, CPU/loader settings, warmup and
   measured duration, synchronization and sampling rules, train/eval mode,
   optimizer/gradient accumulation, AMP/TF32 and backend flags. State whether
   epoch time is directly measured or extrapolated, how GPU versus wall time is
   defined, and whether Torch peak means allocated or reserved memory. The original
   profiler source and runtime are needed to validate reconstructed workloads.

The smallest useful first delivery is items 1–3 plus the dataset fingerprint and
split lists. Existing logs and profiling records can follow; no new full training
run is needed to begin diagnosis.

**How the evidence determines the correction**

First reproduce the table from the supplied prediction rows and verify checkpoint,
target order, transforms, masks, and split identity. Correct a reproducible mapping
or evaluation defect locally, with a focused regression check, if one is found.
Then repeat the identical-input analysis on the actual model features. If distinct
execution settings were lost, repair capture/features and version the regenerated
artifacts. If matched workloads have unstable measurements, repair and standardize
profiling, retain the raw measurements, and quantify repeat variation before
choosing a defensible target aggregation. Do not silently average labels, delete
hard cases, or weaken the metric to make the table improve.

After the data and metric contracts are verified, compare per-output train and
validation curves and residuals by family, precision, size, and GPU. Use those
results to choose bounded validation experiments for loss weighting, output
parameterization, shared versus separate heads, or capacity. Such architecture
changes remain hypotheses until they improve the same held-out validation metrics
across matched runs. A final test evaluation must remain separate from tuning.

**Verification and reproducibility**

Run `python record/perfseer-v32/accuracy-audit-20260906.py` from the repository root.
The [audit source](../record/perfseer-v32/accuracy-audit-20260906.py) verifies the
interval-overlap calculation independently by exhaustive endpoint evaluation for
every group, target, and tolerance, and checks split/input/source hashes. The
[JSON evidence](../record/perfseer-v32/accuracy-audit-20260906.json) records all twelve
targets, baselines, fingerprints, and the contradictory pair. Test labels were not
read. These two evidence files reside under the repository's ignored `record/`
directory; attach them explicitly when forwarding the report.

`python -m pytest -q tests/test_perfseer_v32.py tests/test_perfseer_v31.py` passed:
29 tests, with existing oneDNN/TF32 and mocked-scheduler warnings. These verify
implementation contracts, not the screenshot's accuracy or a successful repair.
The only additions are this handoff, the audit source/evidence, and the task plan
in `plan.md`; production model code remains unchanged pending the actual run.
