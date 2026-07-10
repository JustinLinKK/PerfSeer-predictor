# Label Generation With And Without Dataset Inputs

## Short Answer

PerfSeer currently has two practical label-generation styles:

| Method | What it profiles | Best use | Reliability for realistic training-resource prediction |
| --- | --- | --- | --- |
| Without dataset | Generated model source plus synthetic tensors that match the model input shape | Fast compatibility labels, smoke tests, rough model-only GPU cost | Low to medium |
| With dataset/workload specs | Generated model source plus dataset/profile metadata, subset size, batch settings, optimizer, precision, hardware ID, and deterministic batches tied to real dataset sample keys | Scheduler/resource predictor labels | Medium to high, but only inside the same hardware/workload envelope |

The with-dataset method is more reliable for predicting realistic training memory and resource usage because it records the workload context that actually changes training cost: dataset size, input shape, batch size, optimizer, precision, dataloader settings, and hardware. The without-dataset method can still measure model compute, but it does not know what real task, dataset, subset size, dataloader, or training setup the model will run with.

Important limitation: in the current code, the with-dataset path checks real dataset files and sample keys, then creates deterministic tensors from those sample keys. It is dataset-backed for shape, size, identity, metadata, and repeatable batch selection, but it is not a full semantic replay of real image/text/audio contents.

## What A Label Means In This Repository

The original PerfSeer-compatible label file is:

```text
{'train': '<7 fields>', 'infer': '<7 fields>'}
```

Each phase string has this order:

```text
time|average_sm_util|average_memory_util|average_memory_usage|peak_sm_util|peak_memory_util|peak_memory_usage
```

The baseline parser in `src/perfseer/data.py` turns that into six targets:

```text
train_util, train_mem, train_time, infer_util, infer_mem, infer_time
```

That means the classic six-target model learns:

- average SM utilization for train and inference;
- peak memory usage for train and inference;
- time for train and inference.

It does not learn every field in the seven-field string. For example, average memory-controller utilization and peak SM utilization are present in the raw label string, but the original six-target parser does not keep them as separate targets.

The newer scheduler/resource path adds richer files:

- `label_v3_shard<N>.jsonl`: step time, GPU time, derived epoch time, SM utilization, peak VRAM, dataset/training context.
- `scheduler_resource_shard<N>.jsonl`: fuller resource labels, including average/peak SM utilization, memory-controller utilization, average/peak VRAM, PyTorch allocated memory, and PyTorch reserved memory.
- after materialization, these become `label/scheduler_label_v3.jsonl` and `label/scheduler_resource_label.jsonl`.

## Method 1: Label Generation Without A Dataset

This is the older compatibility-style path.

The flow is:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force

python nrp_calibration_pack/profile/run_profile.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090 \
  --hardware-id rtx5090 \
  --precision-sweep auto \
  --profile-dataset-dir nrp_calibration_pack/profile_datasets \
  --device cuda \
  --sm-occupancy-source nvml_proxy \
  --resource-profile-mode sustained \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --optimizer adam \
  --num-shards <N> \
  --shard-index <I>
```

What happens:

- `make_profile_datasets.py` writes one JSON spec per model with input shape, batch size, repeat counts, and seed.
- `run_profile.py` loads the generated model source.
- It creates synthetic inputs with the right shape and dtype.
- It runs inference and train phases.
- It writes `label/label/*.txt`, `results_shard<N>.jsonl`, `label_v3_shard<N>.jsonl`, and `scheduler_resource_shard<N>.jsonl`.

How memory labels are collected:

- NVML samples GPU memory while the phase runs.
- `average_memory_usage` is average VRAM used in MiB.
- `peak_memory_usage` is peak VRAM used in MiB.
- `average_memory_util` and `peak_memory_util` are memory-controller activity percentages, not the same thing as allocated memory.
- PyTorch-specific peaks are also recorded in richer JSON labels as `peak_torch_allocated_mib` and `peak_torch_reserved_mib`.

How SM labels are collected:

- `average_sm_util` and `peak_sm_util` come from NVML GPU utilization samples.
- If `--sm-occupancy-source nvml_proxy` is used, SM occupancy fields in `label_v2` are only a proxy based on utilization.
- If `--sm-occupancy-source ncu` is used, the profiler tries Nsight Compute and the metric `sm__warps_active.avg.pct_of_peak_sustained_active`. This is closer to true SM occupancy, but it is slower and can fail without counter permissions.

Reliability:

- Good for checking that model source code runs.
- Good for approximate model-only GPU timing and memory pressure.
- Weak for realistic training prediction because real dataset size, dataloader behavior, task modality, and training workload shape are mostly absent.
- If trained only on these labels, the predictor may learn graph shape and operator cost, but it can miss important real-world effects such as sequence length, image size, dataloader workers, precision, optimizer state, and hardware differences.

## Method 2: Label Generation With Dataset/Workload Specs

This is the newer scheduler-style path.

The flow is:

```bash
python nrp_calibration_pack/profile/make_workload_specs.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --registry dataset_sources/registry.json \
  --dataset-profile-root datasets/prepared \
  --output-dir nrp_calibration_pack/workload_specs \
  --subset-id tiny \
  --batch-size 8 \
  --precision-sweep fp32_ieee,bf16_amp \
  --optimizer adam \
  --hardware-id rtx5090 \
  --force

python nrp_calibration_pack/profile/run_profile.py \
  --workload-specs nrp_calibration_pack/workload_specs/workloads.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090_scheduler \
  --hardware-id rtx5090 \
  --device cuda \
  --sm-occupancy-source nvml_proxy \
  --resource-profile-mode sustained \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

Then materialize the dataset:

```bash
python scripts/materialize_precision_dataset.py \
  --pack-dir nrp_calibration_pack \
  --results-dir nrp_results_rtx5090_scheduler \
  --out-root dataset_rtx5090_scheduler \
  --hardware-id rtx5090 \
  --force
```

What happens:

- `make_workload_specs.py` maps each model family to an approved prepared dataset profile.
- It records dataset ID, subset ID, sample count, input shape, modality, task type, batch size, optimizer, precision, dataloader settings, and hardware ID.
- `run_profile.py` rejects workload rows without a real dataloader adapter by default.
- In sustained mode, it measures a whole multi-step train or inference phase instead of synchronizing every repeat separately.
- The optimized data path reads `precision_metadata.jsonl`, so dataset, training, precision, and hardware metadata become predictor input features.

How memory labels are collected:

- The legacy label string still records average and peak VRAM used through NVML.
- `label_v2` records `avg_vram_used_mib`, `peak_vram_used_mib`, `peak_torch_allocated_mib`, and `peak_torch_reserved_mib`.
- `scheduler_resource_label` records both train and inference targets:
  - `train_avg_vram_used_mib`
  - `train_peak_vram_used_mib`
  - `train_peak_torch_allocated_mib`
  - `train_peak_torch_reserved_mib`
  - matching inference fields

How SM labels are collected:

- NVML provides `avg_sm_util` and `peak_sm_util`.
- These become `train_avg_sm_util_percent`, `train_peak_sm_util_percent`, `infer_avg_sm_util_percent`, and `infer_peak_sm_util_percent` in `scheduler_resource_label`.
- If Nsight Compute is requested, `label_v2` can include a more exact occupancy metric. For large runs, NVML proxy mode is usually the practical default.

Reliability:

- Better than the no-dataset path because the predictor can see the workload context that affects real training.
- Better for scheduler planning because it includes hardware, precision, optimizer, batch size, dataset size, and dataloader metadata.
- Still not perfect: the current real-dataset adapter uses sample keys and fingerprints to seed deterministic tensors. It does not fully decode and train on the original data values.
- The labels measure short repeated phases, not full end-to-end convergence. `train_epoch_ms` is derived from step time and dataset size, not always measured as a full real epoch.

## Which Method Is More Reliable?

Use the with-dataset/workload-spec method for predictor training if your goal is to predict realistic training resource and memory usage.

Use the without-dataset method only when:

- you need quick compatibility labels;
- you are testing whether generated models run;
- you only need rough model-graph cost;
- no real dataset profile exists yet.

For realistic scheduling, the ranking is:

1. Best: with-dataset workload specs, sustained profiling, same hardware, same precision, same optimizer, same batch-size range, and enough repeated labels.
2. Acceptable for early development: with-dataset workload specs on smaller subsets, then validate on larger subsets.
3. Weak: synthetic no-dataset labels used directly to predict production training resources.

## Can We Train A Predictor Using These Output Labels?

Yes, but choose the target carefully.

For the old six-target PerfSeer model, training on `label/label/*.txt` is valid because the parser expects that exact format. This is useful when you want:

- train/infer average SM utilization;
- train/infer peak memory usage;
- train/infer time.

For realistic training-resource prediction, prefer the canonical v2 target source `scheduler_v2_train` because it preserves the measured epoch-time label and the richer resource labels:

- use `label/scheduler_label_v3.jsonl` for scheduler-grade time and epoch estimates;
- use `label/scheduler_resource_label.jsonl` for memory and utilization targets;
- keep `label/precision_metadata.jsonl` so dataset/training/hardware features are available to the model.

If you train only from the legacy six-target file, the model may hide different reasons for the same label value. For example, two runs can have similar peak VRAM but different PyTorch reserved memory, memory-controller pressure, precision, or dataloader behavior. The richer JSON labels keep those differences visible.

## Practical Reliability Checklist

Before trusting a trained predictor for realistic model-resource planning, check these conditions:

- The training labels were collected on the same hardware class you want to predict, such as `rtx5090`.
- The labels include the same precision modes you plan to predict, such as `fp32_ieee`, `tf32`, `bf16_amp`, or `fp16_amp`.
- The labels cover the batch-size range you care about.
- The labels cover the same model families, not just CNNs if you plan to predict transformers, RNNs, graph models, audio models, or tabular models.
- The materialized dataset has `precision_metadata.jsonl` and scheduler/resource JSONL files.
- Repeated runs on a small validation subset produce similar labels.
- You reserve a few real end-to-end training runs as golden checks against the predictor.

## Recommended Workflow

For a realistic predictor, use this rule:

```text
Train one teacher/student pair per hardware on real-dataset `scheduler_v2_train` labels.
Use synthetic labels only as warm-up, smoke-test, or fallback data.
Validate against real training runs before trusting predictions for scheduling.
```

Recommended target files:

```text
dataset_rtx5090_scheduler/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  label/scheduler_label_v3.jsonl
  label/scheduler_resource_label.jsonl
```

Recommended command style:

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --workload-specs nrp_calibration_pack/workload_specs/workloads.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx5090_scheduler \
  --hardware-id rtx5090 \
  --device cuda \
  --resource-profile-mode sustained \
  --sm-occupancy-source nvml_proxy \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50
```

Use Nsight Compute only for audit rows:

```bash
python nrp_calibration_pack/profile/run_profile.py \
  ... \
  --resource-audit-source ncu
```

That gives more exact SM occupancy evidence for a smaller subset, without making the full labeling run too slow or dependent on GPU counter permissions.

## Bottom Line

The no-dataset method measures "how expensive this generated model is with synthetic inputs."

The with-dataset method measures "how expensive this model/workload/hardware/precision setup is under a repeatable dataset-backed training phase."

For a predictor that should estimate a model's realistic training resource and memory usage, use the with-dataset method, keep the richer scheduler/resource labels, and validate with a small number of full real training runs before making scheduling decisions from the predictor alone.
