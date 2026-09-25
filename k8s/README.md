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
- `profile/make_profile_datasets.py`: creates per-model input/repeat specs for label generation.
- `submit_nrp_calibration.sh`: one-click Kubernetes Indexed Job launcher.
- `submit_nrp_source_workflow.sh`: three-stage source-first Nautilus workflow renderer.
- `package_source_tar.py`: packages sources, labels, results, and metadata without large PKLs.
- `Dockerfile`: NGC PyTorch/Transformer Engine image recipe for profiling the repo.

## Generate The Local Source Pack

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir nrp_calibration_pack \
  --profile-preset full \
  --subset-size 10000 \
  --generation-workers "$(nproc)" \
  --force
```

Default behavior selects the full `10000` graphs with seed `20260602` and
validates every generated source with Python compilation only. Source pack
generation runs in parallel by default; pass `--generation-workers 1` for
serial/debug runs, or an explicit worker count to tune CPU and disk pressure. Use
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

For the source-first NRP workflow, generate only `--precision-sweep fp32_ieee`
and let `profile/run_profile.py --precision-sweep auto` expand precisions on the
actual GPU. `bf32` is intentionally rejected because it is ambiguous; choose
`tf32` or `bf16_amp`. Canonical FP4 is stored as `nvfp4_te`; `fp4` and `nvfp4`
are accepted CLI aliases, and `mxfp8` is intentionally out of scope for v1.
Each profiler result row records the actual precision recipe metadata, including
the TF32 control API family/effective state, BF16 support probe, FP16 GradScaler
state, Transformer Engine FP8/NVFP4 policy, and unsupported/fallback status.

The generator writes three handoff artifacts:

- `subset/cg/cg/calib_XXXX.pkl`: filtered graph subset with model-id filenames.
- `models/calib_XXXX.py`: reverse-engineered executable PyTorch workload model.
- `manifest/subset_manifest.jsonl`: mapping between model ids, precision configs, original dataset stems, subset graph files, model files, and expected label files.

Use `--smoke-small --subset-size 2 --validation-mode real` for a small local CPU
forward-check pack. Do not commit generated `calib_*.py`, `manifest/`,
`subset/`, `selection_report.md`, or `coverage_summary.json`; they are ignored
and should be regenerated locally before building the cluster image.

## Make Profile Dataset Specs

The generated workload models do not need real accuracy data. For label
generation, create one lightweight synthetic input spec per model; the profiler
uses each spec to allocate a correctly shaped random input on the target device
while preserving the original `train`/`infer` label semantics:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --force
```

## Local Smoke Test

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --data-root dataset \
  --out-dir /tmp/nrp_calibration_smoke_pack \
  --subset-size 2 \
  --smoke-small \
  --validation-mode real \
  --force

python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest /tmp/nrp_calibration_smoke_pack/manifest/subset_manifest.jsonl \
  --output-dir /tmp/nrp_calibration_smoke_pack/profile_datasets \
  --train-repeats 2 \
  --infer-repeats 2 \
  --force

python /tmp/nrp_calibration_smoke_pack/profile/run_profile.py \
  --manifest /tmp/nrp_calibration_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/nrp_calibration_smoke_pack/models \
  --output-dir /tmp/perfseer_calibration_smoke \
  --hardware-id cpu_smoke \
  --num-shards 1 \
  --precision-config fp32_ieee \
  --profile-dataset-dir /tmp/nrp_calibration_smoke_pack/profile_datasets \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1 \
  --device cpu
```

This profiles two tiny selected models on CPU and writes labels under
`/tmp/perfseer_calibration_smoke/label/label/`.
Rerunning the same profiler command resumes by default: completed labels listed
in `results_shard<I>.jsonl` are skipped, and incomplete interrupted profile
points are retried. Pass `--no-resume` to intentionally regenerate a shard.

## Build Image

From the repository root:

```bash
docker build -f nrp_calibration_pack/Dockerfile -t <your-registry>/perfseer-ngc:latest .
docker push <your-registry>/perfseer-ngc:latest
```

The Dockerfile defaults to the verified NGC PyTorch 26.03 image with
Transformer Engine 2.13. The Nautilus prepare job generates the source pack on
the PVC, so the image carries repo code rather than a prebuilt local pack.

## Generate The Source Pack Locally

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --out-dir nrp_calibration_pack \
  --subset-size 10000 \
  --precision-sweep fp32_ieee \
  --generation-workers "$(nproc)" \
  --force
```

## Submit To NRP Nautilus

```bash
./nrp_calibration_pack/submit_nrp_source_workflow.sh \
  --namespace <namespace> \
  --image <your-registry>/perfseer-ngc:latest \
  --pvc <output-pvc> \
  --gpu-product NVIDIA-GeForce-RTX-5090 \
  --hardware-id rtx5090 \
  --parallelism 4 \
  --completions 64 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --dry-run
```

For generic GPUs the default resource is `nvidia.com/gpu`. For special NRP GPU
resources, pass `--gpu-resource`, for example `--gpu-resource nvidia.com/a100`.
The `--gpu-product` argument is rendered as node affinity on
`nvidia.com/gpu.product`.

Use `--dry-run` to print the prepare/profile/package YAML. For real submission,
run one stage at a time with `--stage prepare`, wait for completion, then
`--stage profile`, then `--stage package`. The profile stage uses
`--precision-sweep auto`: base precisions run wherever supported, FP8 requires
Transformer Engine plus Ada/Hopper/Blackwell-class hardware, and `nvfp4_te`
requires Transformer Engine NVFP4 on Blackwell-class hardware. The generated
runtime only enables TE low precision for dense, norm, and attention rows that
also satisfy the shape gate: FP8 needs 16-wide feature/leading alignment, and
NVFP4 needs 32-wide feature alignment with leading dimension at least 32.

For the full golden-data procedure, including how the reverse-engineered source
models are trained and inferred during profiling, see `GOLDEN_DATA_GUIDE.md`.

## Output

The job writes:

- `label/label/<model_id>_<precision_config>.txt`: dataset-compatible label dict for a precision-specific profile point.
- `results_shard*.jsonl`: detailed hardware, timing, memory, and status rows.
- `hardware_shard*.json`: detected CUDA/GPU metadata for each shard.
- `perfseer_<hardware_id>_source_labels.tar.gz`: source-only package from the package stage, including a small `replay/` script bundle.

Pass `--hardware-id` during profiling or submission to store stable hardware
labels such as `rtx3090`, `rtx4090`, or `rtx5090` in those outputs.
If a profiling job is interrupted or restarted, use the same output directory
and shard arguments; the profiler will continue from the last completed label.

For reproducibility audits, rebuild `dataset/cg/cg/*.pkl` from the source-only
tarball instead of downloading generated PKLs:

```bash
python scripts/rebuild_source_tar_dataset.py \
  --source-tar perfseer_rtx5090_source_labels.tar.gz \
  --out-root dataset_rtx5090_rebuilt \
  --force
```

The label format is:

```text
time|average_sm_util|average_memory_util|average_memory_usuage|peak_sm_util|peak_memory_util|peak_memory_usuage
```

`time` is mean per-sample milliseconds. Memory usage is reported in MiB.
