# PerfSeer Predictor

PerfSeer predicts deep learning model performance from compute graphs. The
predictor consumes graph tensors, not raw model source code, and produces six
hardware/performance targets for training and inference.

This repository has two compatible purposes:

- `main` preserves the baseline PerfSeer/SeerNet reproduction and the legacy
  CNN-oriented dataset path.
- `v2` adds an opt-in non-CNN calibration catalog, expanded graph features,
  multi-input profiling, precision/hardware metadata, and a 10,000-model
  dataset workflow for CNN, transformer, recurrent, graph, audio, detector,
  segmentation, and tabular workloads.

Generated packs, profiling results, and materialized datasets are ignored by
git. Source code, tests, configs, and this README are the committed interface.

## Baseline Versus V2 Design

The baseline `src/perfseer/` implementation follows the original PerfSeer
contract:

- Input graph: `dataset/cg/cg/*.pkl` paired with `dataset/label/label/*.txt`.
- Model family coverage: the released/generated calibration pack is effectively
  CNN-family image workloads.
- Node vocabulary: fixed CNN-heavy operator types such as `Conv`, `Relu`,
  `BatchNormalization`, pooling, `Gemm`, `Add`, and `Concat`.
- Feature dimensions: `NODE_DIM=30`, `EDGE_DIM=5`, `GLOBAL_DIM=18`.
- Prediction shape: one six-target vector, or six single-output SeerNet models
  in the faithful baseline training path.

The v2 path keeps SeerNet as a graph predictor but changes the graph encoding
and dataset source:

- Feature schema: `template_v2_noncnn`, selected with
  `FeatureConfig(feature_schema_version="template_v2_noncnn")`.
- Operator coverage: CNNs, depthwise convolution, transposed convolution,
  normalization, embeddings, matmul/bmm, attention, RNN/GRU/LSTM, graph message
  passing, audio/sequence ops, detector heads, segmentation heads, and tabular
  feature operators.
- Shape coverage: tensor rank, input/output size, channels/features, sequence
  length, spatial area, and graph node count.
- Graph metadata: architecture family, modality, variant kind, depth bucket,
  width bucket, precision config, hardware id, hardware features, and label
  domain.
- Checkpoints: v2 feature dimensions differ from the legacy schema, so v2
  checkpoints must be trained from scratch.

The optimized package in `src/perfseer-optimized/` provides the practical
training/deployment path. It includes configurable feature layouts,
multi-output `SeerNetMulti`, PCGrad, distillation, precision/hardware features,
dataset metadata handling, and CPU deployment evaluation.

## Model Input And Output

The model input is a PyTorch Geometric `Data` object. The important fields are:

- `data.x`: node feature matrix. In v2 this includes the expanded operator
  one-hot, numeric op arguments, compute/memory statistics, tensor-shape
  summaries, and node proportions.
- `data.edge_index`: directed compute-graph edges.
- `data.edge_attr`: edge features derived from source and destination tensor
  summaries.
- `data.u`: graph-level features. In v2 this includes topology, aggregate
  compute/memory statistics, architecture family, modality, variant kind,
  precision recipe, hardware id, hardware numeric features, and label domain.
- `data.y`: standardized training target.
- `data.y_raw`: raw six-target label in the original metric space.

The output is always six targets in this order:

```text
train_util, train_mem, train_time, infer_util, infer_mem, infer_time
```

Profiler label files keep the original dataset-compatible format:

```text
{'train': '<7 pipe-separated fields>', 'infer': '<7 pipe-separated fields>'}
```

Each phase string is:

```text
time|average_sm_util|average_memory_util|average_memory_usage|peak_sm_util|peak_memory_util|peak_memory_usage
```

`parse_label()` maps those two seven-field strings into the six PerfSeer
targets. The first field of each phase is the train or inference time target.

## Repository Layout

- `src/perfseer/`: faithful baseline data/model/train/eval implementation.
- `src/perfseer-optimized/`: optimized and v2-aware training, evaluation, and
  deployment package, imported as `perfseer_optimized`.
- `src/perfseer/architecture_schema.py`: legacy and v2 feature schema constants.
- `src/perfseer_source_converter/`: source-to-graph conversion utilities.
- `nrp_calibration_pack/`: calibration pack generator, v2 template catalog,
  profiler, profile dataset generator, Dockerfile, and NRP submit wrapper.
- `scripts/materialize_precision_dataset.py`: converts profiler outputs into a
  training-ready `dataset/cg/cg` and `dataset/label/label` layout.
- `tests/`: unit and smoke tests for pack generation, profiling,
  materialization, source conversion, and precision transfer plumbing.

## Dataset Layouts

The original PerfSeer dataset uses this layout:

```text
dataset/
  cg/cg/*.pkl
  label/label/*.txt
```

The v2 materialized dataset uses the same graph/label layout and adds metadata:

```text
dataset_noncnn_10000/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  precision_materialization_report.json
  precision_rejected_rows.jsonl
```

`precision_metadata.jsonl` maps every label file to its graph, hardware id,
precision config, label domain, and hardware/precision details. Accepted
profiled labels use `label_domain: precision_profile`. Unsupported rows, OOMs,
and profiler errors are not training labels; they are written to
`precision_rejected_rows.jsonl`.

## The New 10,000-Model V2 Catalog

The v2 catalog is generated from deterministic repo-local templates. It does
not import torchvision, timm, transformers, or other heavyweight model
libraries.

Exact family quotas:

| Family | Count |
| --- | ---: |
| `resnet_cnn` | 2160 |
| `efficientnet_cnn` | 1520 |
| `bert_encoder` | 1200 |
| `vit_encoder` | 1200 |
| `yolo_detector` | 560 |
| `unet_encoder_decoder` | 480 |
| `ast_audio_transformer` | 480 |
| `gru_temporal` | 480 |
| `lstm_temporal` | 320 |
| `vgg_cnn` | 400 |
| `gat_graph` | 240 |
| `mpnn_graph` | 240 |
| `t5_encoder_decoder` | 240 |
| `wav2vec2_audio` | 240 |
| `ft_transformer_tabular` | 240 |

Each family uses this variant mix:

- 10% canonical anchors
- 30% added-depth variants
- 30% dropped-depth variants
- 20% width, shape, or hyperparameter variants
- 10% mixed stress variants

The full precision manifest expands every base model across:

```text
fp32_ieee, tf32, bf16_amp, fp16_amp, fp8_te_hybrid
```

For 10,000 base models this produces 50,000 manifest rows. FP8 rows are included
for audit coverage, but current generated GraphModel ops are not rewritten to
Transformer Engine FP8 modules. They should appear as rejected
`unsupported_precision` rows until real FP8 GraphModel support is implemented.

Every manifest row should include:

- `model_id`
- `architecture_family`
- `variant_kind`
- `variant_signature`
- `input_specs`
- `feature_schema_version`
- `model_file`
- `subset_graph_file`
- `label_file`
- `precision_config`

## Create The V2 Source Pack

Create the shared 10,000-model pack locally:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template_v2 \
  --subset-size 10000 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack_noncnn \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
  --validation-mode compile \
  --generation-workers "$(nproc)" \
  --force
```

Create the per-model synthetic input/repeat specs used by the profiler:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack_noncnn/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack_noncnn/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force
```

The generated pack should contain:

```text
nrp_calibration_pack_noncnn/
  models/*.py
  subset/cg/cg/*.pkl
  manifest/subset_manifest.jsonl
  profile_datasets/*.json
  coverage_summary.json
  selection_report.md
```

Do not commit this directory.

## Create Correct Label Data On Hardware

Correct labels require profiling the exact same pack on the target hardware.
For this project, run separate jobs for RTX 3090, RTX 4090, and RTX 5090. Do not
mix hardware classes in the same profiler output directory.

For RTX 4090, the direct profiler command shape is:

```bash
python nrp_calibration_pack_noncnn/profile/run_profile.py \
  --manifest nrp_calibration_pack_noncnn/manifest/subset_manifest.jsonl \
  --models-dir nrp_calibration_pack_noncnn/models \
  --output-dir nrp_noncnn_results_rtx4090 \
  --hardware-id rtx4090 \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
  --profile-dataset-dir nrp_calibration_pack_noncnn/profile_datasets \
  --device cuda \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

Repeat for RTX 3090 and RTX 5090 by changing:

- `--output-dir nrp_noncnn_results_rtx3090` and `--hardware-id rtx3090`
- `--output-dir nrp_noncnn_results_rtx5090` and `--hardware-id rtx5090`
- node affinity or job placement so the requested GPU is actually used

The NRP wrapper renders an indexed Kubernetes job:

```bash
./nrp_calibration_pack/submit_nrp_calibration.sh \
  --namespace <namespace> \
  --image <your-registry>/perfseer-calibration:latest \
  --pvc <output-pvc> \
  --gpu-product NVIDIA-GeForce-RTX-4090 \
  --output-dir /mnt/output/nrp_noncnn_results_rtx4090 \
  --hardware-id rtx4090 \
  --parallelism 4 \
  --completions 64 \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
  --profile-dataset-dir /workspace/nrp_calibration_pack_noncnn/profile_datasets \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50
```

Before accepting labels, check each result root:

- `hardware_shard*.json` contains the expected `hardware_id` and GPU product.
- `results_shard*.jsonl` contains only one hardware class.
- `fp32_ieee` rows are `status: ok` for all 10,000 models.
- TF32, BF16, and FP16 support matches the card and PyTorch/CUDA stack.
- FP8 rows may be `unsupported_precision`; that is expected with the current
  GraphModel runtime.
- OOM or error rows are investigated before training on that hardware class.

## Materialize The Combined Dataset

After the three hardware runs finish, combine their successful labels into one
dataset:

```bash
python scripts/materialize_precision_dataset.py \
  --pack-dir nrp_calibration_pack_noncnn \
  --results-dir nrp_noncnn_results_rtx3090 \
  --results-dir nrp_noncnn_results_rtx4090 \
  --results-dir nrp_noncnn_results_rtx5090 \
  --out-root dataset_noncnn_10000 \
  --force
```

Expected acceptance checks:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = Path("nrp_calibration_pack_noncnn/manifest/subset_manifest.jsonl")
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
print("manifest rows:", len(rows))
print("base models:", len({row["model_id"] for row in rows}))
print("families:", {k: sum(row.get("architecture_family") == k for row in rows) // 5 for k in sorted({row.get("architecture_family") for row in rows})})

report = json.loads(Path("dataset_noncnn_10000/precision_materialization_report.json").read_text())
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

The manifest should have 50,000 rows and 10,000 unique base models. The final
dataset should contain accepted labels, `label/precision_metadata.jsonl`, and a
rejection report for unsupported or failed rows.

## Training And Evaluation

Install editable package dependencies in the project environment:

```bash
python -m pip install -e .
```

Run a quick optimized smoke test:

```bash
python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/baseline.yaml \
  --limit 200 \
  --epochs 2
```

Train on the v2 dataset with a v2 feature config by using a config whose
`features.feature_schema_version` is `template_v2_noncnn`, or by adding the same
field to an existing optimized config. Because feature dimensions changed, do
not reuse legacy checkpoints.

Useful evaluation commands:

```bash
python -m perfseer_optimized.eval \
  --ckpt-dir runs/optimized/<run-id> \
  --data-root dataset_noncnn_10000 \
  --batch-size 128

python -m perfseer_optimized.eval \
  --ckpt-dir runs/optimized/<run-id> \
  --data-root dataset_noncnn_10000 \
  --bench-cpu \
  --batch-size 1
```

For deployment candidates, use the multi-output or distilled student configs in
`src/perfseer-optimized/configs/train_deploy_model/`, then run the deployment
matrix for CPU latency and backend export evidence.

## Local Validation

Run the standard checks after changing code:

```bash
python -m py_compile \
  nrp_calibration_pack/build_pack.py \
  nrp_calibration_pack/generate_model_sources.py \
  nrp_calibration_pack/profile/generated_model_runtime.py \
  nrp_calibration_pack/profile/make_profile_datasets.py \
  nrp_calibration_pack/profile/run_profile.py \
  nrp_calibration_pack/template_catalog.py \
  scripts/materialize_precision_dataset.py \
  src/perfseer/architecture_schema.py \
  src/perfseer-optimized/data.py \
  src/perfseer_source_converter/converter.py

python -m unittest tests.test_nrp_calibration_pack tests.test_source_converter -v
git diff --check
git ls-files -ci --exclude-standard
```

Run a tiny v2 CPU smoke before any full hardware job:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template_v2 \
  --subset-size 15 \
  --out-dir /tmp/perfseer_noncnn_smoke_pack \
  --precision-sweep fp32_ieee \
  --validation-mode real \
  --generation-workers 1 \
  --force

python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest /tmp/perfseer_noncnn_smoke_pack/manifest/subset_manifest.jsonl \
  --output-dir /tmp/perfseer_noncnn_smoke_pack/profile_datasets \
  --train-repeats 1 \
  --infer-repeats 1 \
  --force

python /tmp/perfseer_noncnn_smoke_pack/profile/run_profile.py \
  --manifest /tmp/perfseer_noncnn_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/perfseer_noncnn_smoke_pack/models \
  --output-dir /tmp/perfseer_noncnn_smoke_results \
  --hardware-id cpu_smoke \
  --num-shards 1 \
  --precision-config fp32_ieee \
  --profile-dataset-dir /tmp/perfseer_noncnn_smoke_pack/profile_datasets \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1 \
  --device cpu
```

## Notes And Constraints

- Root documentation is intentionally consolidated into this README. Detailed
  pack-specific notes remain under `nrp_calibration_pack/`.
- Generated datasets are large and hardware-specific; keep them out of git.
- `bf32` is not a valid precision config. Use `tf32` or `bf16_amp`.
- `fp8_te_hybrid` is currently an audited unsupported path for generated
  GraphModel ops, not a source of accepted training labels.
- For final reported results, record the exact commit, CUDA version, PyTorch
  version, GPU product, driver, precision sweep, warmup, repeat counts, and
  rejection counts.
