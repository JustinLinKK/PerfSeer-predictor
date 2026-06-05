# NRP Calibration Pack

This folder contains a representative PerfSeer calibration subset for profiling
on NRP Nautilus GPUs. The pack is designed to collect new ground-truth labels on
target hardware while preserving the current dataset label format.

## Contents

- `generate_model_sources.py`: local CLI for subset selection and reverse-engineered source-model generation.
- `build_pack.py`: reusable implementation used by the generator CLI.
- `GOLDEN_DATA_GUIDE.md`: how to profile the generated models and produce GPU golden labels.
- `subset/cg/cg/`: generated, model-id-named graph subset for transfer-learning inputs.
- `models/`: generated PyTorch source modules, one per selected graph.
- `manifest/subset_manifest.jsonl`: selected graph metadata and model-file map.
- `coverage_summary.json`: machine-readable selected-vs-full coverage summary.
- `selection_report.md`: selected-vs-full dataset coverage report.
- `profile/run_profile.py`: NRP runtime profiler for train and inference labels.
- `submit_nrp_calibration.sh`: one-click Kubernetes Indexed Job launcher.
- `Dockerfile`: optional image recipe for shipping the pack.

## Generate The Local Source Pack

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir nrp_calibration_pack \
  --profile-preset full \
  --subset-size 10000 \
  --force
```

Default behavior selects the full `10000` graphs with seed `20260602` and
validates every generated source with Python compilation only. Use
`--profile-preset pilot` for the chosen 1000-graph precision pilot before the
full sweep; `--subset-size` can still override either preset. The selector
balances batch sizes, reserves pure and mixed architecture-family coverage,
anchors rare operator and topology signatures, anchors model-structure,
resource, and size coverage, anchors model-size quantiles, then fills the
remaining slots with feature-space diversity.

The generated manifest expands every selected graph across the default
precision sweep:

```text
fp32_ieee, tf32, bf16_amp, fp16_amp, fp8_te_hybrid
```

Use `--profile-preset pilot --precision-sweep fp32_ieee,bf16_amp` for a smaller
pilot pack, or `--precision-sweep fp32_ieee` for local CPU smoke tests. `bf32`
is intentionally rejected because it is ambiguous; choose `tf32` or `bf16_amp`.
Each profiler result row records the actual precision recipe metadata, including
the TF32 control API family/effective state, BF16 support probe, FP16 GradScaler
state, FP8 backend policy, and unsupported/fallback status where applicable.

The generator writes three handoff artifacts:

- `subset/cg/cg/calib_XXXX.pkl`: filtered graph subset with model-id filenames.
- `models/calib_XXXX.py`: reverse-engineered executable PyTorch workload model.
- `manifest/subset_manifest.jsonl`: mapping between model ids, precision configs, original dataset stems, subset graph files, model files, and expected label files.

Use `--smoke-small --subset-size 2 --validation-mode real` for a small local CPU
forward-check pack. Do not commit generated `calib_*.py`, `manifest/`,
`subset/`, `selection_report.md`, or `coverage_summary.json`; they are ignored
and should be regenerated locally before building the cluster image.

## Local Smoke Test

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir /tmp/nrp_calibration_smoke_pack \
  --subset-size 2 \
  --smoke-small \
  --validation-mode real \
  --force

python /tmp/nrp_calibration_smoke_pack/profile/run_profile.py \
  --manifest /tmp/nrp_calibration_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/nrp_calibration_smoke_pack/models \
  --output-dir /tmp/perfseer_calibration_smoke \
  --num-shards 1 \
  --precision-config fp32_ieee \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1 \
  --device cpu
```

This profiles two tiny selected models on CPU and writes labels under
`/tmp/perfseer_calibration_smoke/label/label/`.

## Build Image

From the repository root:

```bash
python nrp_calibration_pack/generate_model_sources.py --force
docker build -f nrp_calibration_pack/Dockerfile -t <your-registry>/perfseer-calibration:latest .
docker push <your-registry>/perfseer-calibration:latest
```

## Generate the dataset

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir nrp_calibration_pack \
  --subset-size 10000 \
  --force
```

## Submit To NRP Nautilus

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
  --train-repeats 50
```

For generic GPUs the default resource is `nvidia.com/gpu`. For special NRP GPU
resources, pass `--gpu-resource`, for example `--gpu-resource nvidia.com/a100`.
The `--gpu-product` argument is rendered as node affinity on
`nvidia.com/gpu.product`.

Use `--dry-run` to print the rendered YAML before submission.

For the full golden-data procedure, including how the reverse-engineered source
models are trained and inferred during profiling, see `GOLDEN_DATA_GUIDE.md`.

## Output

The job writes:

- `label/label/<model_id>_<precision_config>.txt`: dataset-compatible label dict for a precision-specific profile point.
- `results_shard*.jsonl`: detailed hardware, timing, memory, and status rows.
- `hardware_shard*.json`: detected CUDA/GPU metadata for each shard.

The label format is:

```text
time|average_sm_util|average_memory_util|average_memory_usuage|peak_sm_util|peak_memory_util|peak_memory_usuage
```

`time` is mean per-sample milliseconds. Memory usage is reported in MiB.
