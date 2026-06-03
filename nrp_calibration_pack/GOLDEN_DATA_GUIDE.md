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

If using epoch language, treat the job as one profiling epoch over every selected
model on a fixed GPU type. The minimum useful train/infer work is iteration
count, not dataset epochs.

## Profiling Budget

Use this only for plumbing smoke tests:

```bash
--warmup 1 --infer-repeats 1 --train-repeats 1 --sample-interval 0.01
```

The default Nautilus submit settings are the recommended golden-data budget:

```bash
--warmup 20 --infer-repeats 50 --train-repeats 50 --sample-interval 0.01
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
  --subset-size 4096 \
  --force
```

The default subset is `4096` graphs. It covers all batch buckets, pure and mixed
architecture families, operator-presence cases, topology signatures, and model
size quantiles before using diversity fill.

Generation writes:

- `subset/cg/cg/calib_XXXX.pkl`: filtered graph subset with model-id filenames.
- `models/calib_XXXX.py`: reverse-engineered executable PyTorch model source.
- `manifest/subset_manifest.jsonl`: mapping to original dataset stems and
  expected `label/label/calib_XXXX.txt` output files.

## Submit A Golden Run

```bash
./nrp_calibration_pack/submit_nrp_calibration.sh \
  --namespace <namespace> \
  --image <your-registry>/perfseer-calibration:latest \
  --pvc <output-pvc> \
  --gpu-product NVIDIA-GeForce-RTX-4090 \
  --parallelism 4 \
  --completions 64 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50
```

Increase `--completions` for more shards when the PVC and cluster allow it. The
job writes dataset-compatible labels to `label/label/<model_id>.txt` and
detailed profiling rows to `results_shard*.jsonl`.

## Accepting Golden Rows

Use a row as golden data only when:

- `status` is `ok` in the matching `results_shard*.jsonl` row.
- the hardware metadata matches the target GPU product and software stack.
- both `train` and `infer` labels are present.
- NVML sampling is available, or memory-only fallback is acceptable for the
  target metric.

Rows marked `oom` or `error` should be kept in the detailed results for audit,
but they should not replace valid label files.
