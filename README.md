# PerfSeer Predictor

PerfSeer predicts training and inference performance from compute graphs. This
branch is the canonical expanded-catalog workflow: SeerNet remains a graph
predictor, but the dataset, feature schema, profiler, and training flow are
built for CNN, transformer, recurrent, graph, audio, detector, segmentation, and
tabular model families.

Generated packs, profiler results, datasets, checkpoints, and smoke outputs are
ignored by git. Source code, tests, configs, and this README are the committed
interface.

## Model Design

The original baseline used a fixed CNN-heavy graph schema. Its generated pack
covered image-style DAGs with operators such as convolution, batch norm, pooling,
`Gemm`, `Add`, and `Concat`. That was enough for the original dataset, but it
did not encode the operators and input shapes needed by modern text, sequence,
audio, graph, detector, segmentation, or tabular workloads.

The current design keeps the same core idea, a SeerNet GNN over compute graphs,
and changes the representation around it:

- `perfseer_graph_v1` is the single canonical feature schema.
- The operator vocabulary includes convolution, depthwise convolution, transpose
  convolution, normalization, embedding, matmul/bmm, attention, RNN/GRU/LSTM,
  graph-message operators, activations, pooling, upsample, detector heads,
  segmentation heads, and tabular feature operators.
- Node and edge features include tensor-rank and tensor-shape summaries in
  addition to compute, memory, topology, and destination-tensor summaries.
- Graph-level features include architecture family, modality, variant kind,
  depth bucket, width bucket, precision recipe, and label domain.
- Hardware is handled as a dataset filter. We train one teacher/student pair per
  hardware class instead of one cross-hardware model, which reduces the feature
  burden and keeps each model specialized to a single GPU family.

Because the feature dimensions changed, train this branch from scratch. Do not
reuse baseline checkpoints.

## Model Input And Output

Training uses PyTorch Geometric `Data` objects:

- `data.x`: node features with expanded operator one-hot values, operation
  arguments, compute/memory statistics, topology, and tensor-shape summaries.
- `data.edge_index`: directed compute-graph edges.
- `data.edge_attr`: edge tensor summaries, with optional destination tensor and
  edge-topology features.
- `data.u`: graph-level features for aggregate resources, architecture metadata,
  precision recipe, and label domain.
- `data.y`: standardized six-target training label.
- `data.y_raw`: raw six-target label in the original metric space.

The model output is always six targets in this fixed order:

```text
train_util, train_mem, train_time, infer_util, infer_mem, infer_time
```

Profiler label files keep the original PerfSeer-compatible dict format:

```text
{'train': '<7 pipe-separated fields>', 'infer': '<7 pipe-separated fields>'}
```

Each phase string is:

```text
time|average_sm_util|average_memory_util|average_memory_usage|peak_sm_util|peak_memory_util|peak_memory_usage
```

`parse_label()` maps those two seven-field strings into the six model targets.

## Repository Layout

- `src/perfseer/`: shared schema and original parser/model utilities still used
  by the optimized pipeline.
- `src/perfseer-optimized/`: canonical training, evaluation, distillation, and
  deployment package, imported as `perfseer_optimized`.
- `src/perfseer_source_converter/`: source-to-graph conversion utilities.
- `nrp_calibration_pack/`: template catalog generator, generated-model runtime,
  profiler, profile-dataset builder, Dockerfile, and NRP submit wrapper.
- `scripts/materialize_precision_dataset.py`: converts profiler results into
  `dataset/cg/cg` and `dataset/label/label`.
- `scripts/run_hardware_distill_flow.py`: scratch teacher training followed by
  same-hardware student distillation.

## Dataset Contents

The canonical generated pack is `nrp_calibration_pack/`; the canonical
materialized training dataset is `dataset/`.

```text
dataset/
  cg/cg/*.pkl
  label/label/*.txt
  label/precision_metadata.jsonl
  precision_materialization_report.json
  precision_rejected_rows.jsonl
```

The 10,000-template catalog is deterministic and repo-local. It does not import
torchvision, timm, transformers, or other heavyweight model libraries.

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

Each family uses this variant mix: 10% canonical anchors, 30% added-depth,
30% dropped-depth, 20% width/shape/hyperparameter changes, and 10% mixed stress.

Every manifest row includes `model_id`, `architecture_family`, `variant_kind`,
`variant_signature`, `input_specs`, `feature_schema_version`, `model_file`,
`subset_graph_file`, `precision_config`, `profile_point_id`, and label paths.

## Create The Pack

Generate the shared 10,000-model pack once:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 10000 \
  --seed 20260617 \
  --out-dir nrp_calibration_pack \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
  --validation-mode compile \
  --generation-workers "$(nproc)" \
  --force
```

Create profile dataset specs:

```bash
python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --output-dir nrp_calibration_pack/profile_datasets \
  --train-repeats 50 \
  --infer-repeats 50 \
  --seed 20260617 \
  --force
```

Acceptance gates:

- `nrp_calibration_pack/manifest/subset_manifest.jsonl` has 50,000 rows.
- Family quotas match the table above.
- Every row has architecture, variant, input spec, schema, precision, and model
  path metadata.
- Unsupported operator coverage is zero.

## Create Hardware Labels

Profile the exact same pack once per hardware class. Each result root must
contain only one hardware ID.

Example for RTX 4090:

```bash
python nrp_calibration_pack/profile/run_profile.py \
  --manifest nrp_calibration_pack/manifest/subset_manifest.jsonl \
  --models-dir nrp_calibration_pack/models \
  --output-dir nrp_results_rtx4090 \
  --hardware-id rtx4090 \
  --precision-sweep fp32_ieee,tf32,bf16_amp,fp16_amp,fp8_te_hybrid \
  --profile-dataset-dir nrp_calibration_pack/profile_datasets \
  --device cuda \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --num-shards <N> \
  --shard-index <I>
```

Repeat on RTX 3090 and RTX 5090 by changing only `--output-dir`,
`--hardware-id`, and the actual hardware/node affinity:

```text
nrp_results_rtx3090  -> --hardware-id rtx3090
nrp_results_rtx4090  -> --hardware-id rtx4090
nrp_results_rtx5090  -> --hardware-id rtx5090
```

Materialize one combined dataset:

```bash
python scripts/materialize_precision_dataset.py \
  --pack-dir nrp_calibration_pack \
  --results-dir nrp_results_rtx3090 \
  --results-dir nrp_results_rtx4090 \
  --results-dir nrp_results_rtx5090 \
  --out-root dataset \
  --force
```

Only `status == "ok"` rows become training labels. Unsupported, OOM, and error
rows are written to `precision_rejected_rows.jsonl`. FP8 rows are expected there
until real FP8 GraphModel support is implemented.

## Train One Model Per Hardware

Train a large teacher from scratch and distill the matching student. Use one run
per hardware ID:

```bash
python scripts/run_hardware_distill_flow.py \
  --data-root dataset \
  --hardware-id rtx4090 \
  --teacher-epochs 600 \
  --student-epochs 500 \
  --split-unit graph
```

For the other GPUs, rerun with `--hardware-id rtx3090` and `--hardware-id
rtx5090`. The runner passes the hardware ID to training, and training filters
`precision_metadata.jsonl` before the split. Each hardware model can still learn
from every accepted precision recipe for that hardware.

Useful individual commands:

```bash
python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_hardware_teacher/large_teacher.yaml \
  --data-root dataset \
  --hardware-id rtx4090 \
  --run-id hardware_large_teacher_rtx4090

python -m perfseer_optimized.train \
  --config src/perfseer-optimized/configs/train_deploy_model/distill_student_128.yaml \
  --data-root dataset \
  --hardware-id rtx4090 \
  --teacher-ckpt-dir runs/optimized/hardware_large_teacher_rtx4090 \
  --run-id hardware_distill_student_128_rtx4090
```

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
  scripts/run_hardware_distill_flow.py \
  src/perfseer/architecture_schema.py \
  src/perfseer-optimized/data.py \
  src/perfseer-optimized/train.py \
  src/perfseer-optimized/eval.py \
  src/perfseer_source_converter/converter.py

python -m unittest tests.test_nrp_calibration_pack tests.test_source_converter -v
git diff --check
git ls-files -ci --exclude-standard
```

Tiny CPU smoke:

```bash
python nrp_calibration_pack/generate_model_sources.py \
  --catalog-mode template \
  --subset-size 15 \
  --seed 20260617 \
  --out-dir /tmp/perfseer_smoke_pack \
  --precision-sweep fp32_ieee \
  --validation-mode compile \
  --force

python nrp_calibration_pack/profile/make_profile_datasets.py \
  --manifest /tmp/perfseer_smoke_pack/manifest/subset_manifest.jsonl \
  --output-dir /tmp/perfseer_smoke_pack/profile_datasets \
  --train-repeats 1 \
  --infer-repeats 1 \
  --force

python /tmp/perfseer_smoke_pack/profile/run_profile.py \
  --manifest /tmp/perfseer_smoke_pack/manifest/subset_manifest.jsonl \
  --models-dir /tmp/perfseer_smoke_pack/models \
  --output-dir /tmp/perfseer_smoke_results \
  --hardware-id cpu_smoke \
  --precision-sweep fp32_ieee \
  --profile-dataset-dir /tmp/perfseer_smoke_pack/profile_datasets \
  --device cpu \
  --warmup 1 \
  --infer-repeats 1 \
  --train-repeats 1
```
