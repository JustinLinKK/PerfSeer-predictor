# Golden GPU Data Guide

This pack converts selected PerfSeer compute graphs into executable PyTorch
modules so the same graph shapes can be profiled on NRP GPUs. These generated
modules are "reverse-engineered" workload models, not accuracy models.

## What Gets Trained

The generated models do not need semantic training or convergence. Their random
weights are enough because the golden labels measure hardware behavior: forward
latency, backward/update latency, utilization, and memory. The profiler runs:

- inference: `model.eval()`, random input, `torch.no_grad()`, timed forward.
- training: `model.train()`, random input, MSE against zeros, backward, and one
  SGD step per timed repeat.

This mirrors the original dataset label contract: `train.time` and `infer.time`
are step/forward timings in the 7-field `train` and `infer` strings, not epoch
durations.

## Profiling Budget

Use this only for plumbing smoke tests:

```bash
--warmup 1 --infer-repeats 1 --train-repeats 1 --sample-interval 0.01
```

The default Nautilus submit settings are the recommended golden-data budget:

```bash
--warmup 20 --train-repeats 50 --infer-repeats 50 --sample-interval 0.01
```

Run a second pass on the same GPU type if a model's repeated timing is unstable
or if the selected GPU nodes are heterogeneous. Keep only rows from matching GPU
product, driver, CUDA, and PyTorch versions when fitting hardware calibration.

## Build The Source Pack

From the repository root:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir nrp_calibration_pack \
  --profile-preset full \
  --subset-size 10000 \
  --generation-workers "$(nproc)" \
  --force
```

The default subset is `10000` graphs. It covers all batch buckets, pure and mixed
architecture families, operator-presence cases, topology signatures, and model
structure/resource/size quantiles before using diversity fill.
Pack generation validates candidate model sources in parallel by default. Use
`--generation-workers 1` for serial/debug runs, or pass an explicit worker count
to match the local machine.

Run the first precision pilot with `--profile-preset pilot`, which selects
1000 graphs before expanding to the full 10000-graph sweep. `--subset-size`
still overrides either preset for smoke/debug runs.

The default manifest expands each selected graph across:

```text
fp32_ieee, tf32, bf16_amp, fp16_amp, fp8_te_hybrid
```

Use `--profile-preset pilot --precision-sweep fp32_ieee,bf16_amp` for a smaller
pilot sweep. Use `--precision-sweep fp32_ieee` for local CPU smoke tests.
Profiler rows record the actual precision recipe metadata, including the TF32
control API family/effective state, BF16 support probe, FP16 GradScaler state,
FP8 backend policy, and unsupported/fallback status where applicable.

Generation writes:

- `subset/cg/cg/calib_XXXX.pkl`: filtered graph subset with model-id filenames.
- `models/calib_XXXX.py`: reverse-engineered executable PyTorch model source.
- `manifest/subset_manifest.jsonl`: mapping to original dataset stems and
  expected `label/label/calib_XXXX_<precision_config>.txt` output files.

Create one profile dataset spec per model before building the image or tarball:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --force
```

These JSON specs keep the pack small. The profiler allocates random tensors with
the manifest input shape on the target GPU when generating labels.

## Submit A Golden Run

```bash
./nrp_calibration_pack/submit_nrp_calibration.sh \
  --namespace <namespace> \
  --image <your-registry>/perfseer-calibration:latest \
  --pvc <output-pvc> \
  --gpu-product NVIDIA-GeForce-RTX-4090 \
  --parallelism 4 \
  --completions 64 \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --profile-dataset-dir /workspace/nrp_calibration_pack/profile_datasets
```

Increase `--completions` for more shards when the PVC and cluster allow it. The
job writes dataset-compatible labels to
`label/label/<model_id>_<precision_config>.txt` and detailed profiling rows to
`results_shard*.jsonl`. FP8 rows currently record unsupported/fallback details
unless the generated workloads are rewritten for the selected Transformer Engine
FP8 recipe.

## Accepting Golden Rows

Use a row as golden data only when:

- `status` is `ok` in the matching `results_shard*.jsonl` row.
- the hardware metadata matches the target GPU product and software stack.
- both `train` and `infer` labels are present.
- NVML sampling is available, or memory-only fallback is acceptable for the
  target metric.

Rows marked `oom` or `error` should be kept in the detailed results for audit,
but they should not replace valid label files.

## Materialize Precision Training Data

After a profiling run, convert accepted `results_shard*.jsonl` rows into a
PerfSeer dataset with precision/hardware metadata:

```bash
python scripts/materialize_precision_dataset.py \
  --pack-dir nrp_calibration_pack \
  --results-dir /mnt/output/nrp_calibration \
  --out-root dataset_precision_a100 \
  --hardware-id a100 \
  --base-data-root dataset \
  --base-mode symlink \
  --source-precision-config fp32_ieee \
  --source-hardware-id source_domain_unknown \
  --source-precision-provenance original-profiler-notes.md#fp32 \
  --require-source-precision-provenance \
  --force
```

The output keeps the normal dataset shape:

```text
dataset_precision_a100/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  precision_materialization_report.json
  precision_rejected_rows.jsonl
```

`precision_metadata.jsonl` maps each precision label back to its graph,
`hardware_id`, `precision_config`, and numeric hardware features. When
`--base-data-root` is used, copied or symlinked source-domain labels also get
metadata rows with `label_domain: source`, so their precision/hardware domain is
auditable but they still use the base sample weight. For final runs, keep
`--require-source-precision-provenance` enabled so the materializer fails unless
the original source-label precision setup is documented through
`--source-precision-provenance`; the precision-transfer flow also forwards that
provenance into source-teacher pretraining metadata so the source checkpoint can
be checked before transfer. The optimized data loader uses this sidecar to
set per-sample global precision/hardware features and to upweight real precision
labels during transfer or distillation.
Evaluation results include per-precision, per-batch-size, graph-signature, and
memory-bound/compute-bound resource-regime slices in `runs/results.jsonl`.
Accuracy and deployment eval rows also record the eval split unit and test hash
so structural holdout runs can be checked after the fact.
Precision teacher/student configs use `data.split_unit: graph`, so all profiled
precision labels for the same graph stay in the same split. Switch it to
`graph_signature`, or pass `--split-unit graph_signature` to the flow runner,
for a structural robustness run that holds out whole graph-signature clusters.
They default to `features.target_mode: absolute`;
switch to `log_ratio_to_source` to run a residual transfer experiment using the
materialized source labels as baselines.
Precision student distillation keeps measured precision rows hard-label
dominated while source or pseudo rows can blend toward teacher soft targets via
the `distillation.*_hard_alpha` settings. Add pseudo rows with
`--pseudo-precision-sweep` when a student should learn sparse graph/dtype pairs
from the teacher; pseudo rows are kept in the train split only, excluded from
normalization stats, and use `features.pseudo_label_weight` rather than the
measured precision weight.
Training checkpoints store `supported_precision_hardware` allow-lists so
deployment inference can reject unseen precision/hardware requests clearly. The
source converter's precision/hardware override flags are validated against this
allow-list before predictor inputs are built, and optimized evaluation rejects
test rows outside the checkpoint allow-list before scoring. Deployment
evaluation writes `deployment_metadata.json` beside exported runtime artifacts
with the feature layout, precision/hardware config, supported allow-list,
runtime backend, and held-out split evidence.
Rows rejected during materialization are written to
`precision_rejected_rows.jsonl`; the report also aggregates skipped rows by
status, precision, and fallback policy so unsupported FP8 cases remain separate
from accepted labels.

The intended training handoff is:

```bash
python scripts/run_precision_transfer_flow.py \
  --results-dir /mnt/output/nrp_calibration \
  --precision-data-root dataset_precision_a100 \
  --hardware-id a100 \
  --dry-run
```

Remove `--dry-run` after inspecting the command plan. The runner materializes
accepted precision rows, pretrains the source teacher, evaluates that source
teacher on the original source-domain held-out split, fine-tunes the precision
teacher, distills the student, and evaluates the precision checkpoints. Use
`--skip-source-eval` only when that baseline-preservation check is handled by a
separate run. Add `--check-results` after a real run to verify that the result
ledger contains source-teacher, precision-teacher, and precision-student eval
rows with precision, batch-size, resource-regime, and graph-signature slices.
The runner defaults source labels to `source_domain_unknown`; replace that with
`fp32_ieee`, `tf32`, or another concrete recipe only after the original label
collection precision setup is confirmed.
When `--required-split-unit graph_signature` is used, those rows must also carry
test-hash evidence, and any checkpoint test hash must match the evaluated full
held-out split.
Use the `--min-*-slices` checker flags when the acceptance gate should require
more than one batch, resource, or graph-signature slice. With source precision
provenance enabled, the checker also validates
`precision_materialization_report.json` so prebuilt datasets cannot skip the
source-precision evidence gate silently, and requires the source-teacher
`train_complete` row to record the same provenance. Pass
`--deploy-eval-profile` to run a
deployment runtime profile for the student; the checker then requires the
`eval_deploy_complete` row and its `deployment_metadata.json` sidecar. Add
`--require-checkpoint-files` when the final gate should also prove that every
eval ledger row still points at a real checkpoint file. Add
`--require-train-events` to require source, transfer, and student
`train_complete` rows in the same ledger; add
`--require-unlimited-train-data` for final non-smoke runs so accidental
`--limit` training cannot pass the gate.
The underlying training commands are:

```bash
python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_precision_teacher/large_teacher.yaml \
  --run-id precision_large_teacher_source \
  --data-root dataset \
  --precision-config fp32_ieee \
  --hardware-id source_domain_unknown \
  --source-precision-provenance original-profiler-notes.md#fp32 \
  --require-source-precision-provenance

python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_precision_teacher/large_teacher.yaml \
  --run-id precision_large_teacher_transfer \
  --data-root dataset_precision_a100 \
  --init-checkpoint runs/optimized/precision_large_teacher_source/seernet_multi.pt

python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/precision_distill_student_128.yaml \
  --data-root dataset_precision_a100 \
  --teacher-ckpt-dir runs/optimized/precision_large_teacher_transfer
```
