# Integration Guide

This guide documents the input and output contract for the current PerfSeer v2
model pair. The active pair is one large hardware teacher and one compact
distilled student per hardware ID.

## Active Model Pair

Use these configs for the current v2 path:

- Teacher: `src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml`
- Student: `src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml`

Both configs use:

- `model.name: seernet_multi`
- `features.feature_schema_version: perfseer_graph_v1`
- `features.target_source: scheduler_v2_train`
- `features.target_mode: absolute`
- `features.include_precision_features: true`
- `features.include_dataset_features: true`
- `features.include_hardware_features: false`

The canonical wrapper is:

```bash
python scripts/run_hardware_distill_flow.py --hardware-id rtx5090
```

It defaults to `v2_teacher_<hardware-id>` and `v2_student_<hardware-id>` run
IDs.

## Model Input

The runtime input is a PyTorch Geometric `Data` object or batched `Batch` object.
`SeerNetMulti.forward(data)` consumes graph tensors directly from the object.

With the active v2 configs, the feature layout is:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `data.x` | `[num_nodes, 78]` | `float32` | Node features from the compute graph, including operator type, argument values, FLOPs, memory/weight sizes, tensor shape summaries, topology, and critical-path features. |
| `data.edge_index` | `[2, num_edges]` | `int64` | Directed compute-graph edges, source row first and destination row second. |
| `data.edge_attr` | `[num_edges, 16]` | `float32` | Edge tensor summaries, edge-topology features, and destination tensor summaries. |
| `data.u` | `[num_graphs, 164]` | `float32` | Graph-level features, including graph totals, architecture family/modality/variant, precision recipe, dataset metadata, dataloader metadata, and training metadata. |
| `data.batch` | `[num_nodes]` | `int64` | Optional graph index per node for batched inference. If missing, the model treats the input as one graph. |

Training and evaluation examples also carry label and metadata tensors:

| Field | Shape | Meaning |
| --- | --- | --- |
| `data.y` | `[num_graphs, 6]` | Standardized training target used for loss computation. |
| `data.y_raw` | `[num_graphs, 6]` | Raw target values before standardization. |
| `data.y_eval_raw` | `[num_graphs, 6]` | Raw values used by evaluation. |
| `data.y_base_raw` | `[num_graphs, 6]` | Base/source-label raw values when a target mode needs them. |
| `data.sample_weight` | `[num_graphs]` | Per-example weight from source, precision, or pseudo-label domain. |
| `data.precision_config_idx` | `[num_graphs]` | Encoded precision recipe. |
| `data.resource_regime_idx` | `[num_graphs]` | Encoded resource regime. |

Build these objects through the optimized data path, not by manually assembling
partial tensors:

- `perfseer_optimized.data.PerfSeerOptimizedDataset`
- `perfseer_optimized.data.build_pyg_data()`
- `perfseer_optimized.data.feature_config_for_pair()`

## Dataset Files

The materialized dataset root is normally `dataset/`.

Required graph and compatibility-label inputs:

```text
dataset/cg/cg/*.pkl
dataset/label/label/*.txt
```

The v2 target source also requires scheduler sidecar rows. For a label file such
as `dataset/label/label/<id>.txt`, the loader searches these sidecars in the
label directory, the label parent, and the dataset root:

```text
scheduler_label_v3.jsonl
scheduler_resource_label.jsonl
precision_metadata.jsonl
```

Rows are matched by `profile_point_id`, `label_file`, label basename, or label
stem. If `features.target_source: scheduler_v2_train` is active, missing
`scheduler_label_v3.jsonl` or `scheduler_resource_label.jsonl` rows fail loudly
instead of silently falling back to legacy labels.

## Model Output

`SeerNetMulti.forward(data)` returns a tensor shaped:

```text
[num_graphs, 6]
```

The returned values are in the standardized target space used during training.
Use the checkpoint `metadata.norm_stats` and
`perfseer_optimized.data.invert_targets()` to convert predictions back to raw
units. Checkpoints also store `metadata.target_names`, `metadata.feature_config`,
and `metadata.feature_layout`; downstream code should read those fields instead
of hard-coding dimensions.

For the active v2 configs, the raw output order is:

| Index | Name | Raw Unit |
| --- | --- | --- |
| 0 | `train_epoch_ms` | Milliseconds per training epoch. |
| 1 | `train_avg_sm_util_percent` | Average Streaming Multiprocessor utilization percent. |
| 2 | `train_p95_sm_util_percent` | 95th percentile Streaming Multiprocessor utilization percent. |
| 3 | `train_peak_vram_used_mib` | Peak used GPU memory in MiB. |
| 4 | `train_peak_torch_reserved_mib` | Peak PyTorch reserved GPU memory in MiB. |
| 5 | `train_peak_memory_controller_util_percent` | Peak memory-controller utilization percent. |

The student has the same input and output contract as the teacher. Distillation
changes training, not the integration-facing prediction shape or target order.

## Legacy Output Warning

Do not interpret v2 predictions as the legacy six-target order:

```text
train_util, train_mem, train_time, infer_util, infer_mem, infer_time
```

That order is only for compatibility with older label parsing. Current v2
teacher/student runs use `scheduler_v2_train` and the target order listed above.

## Integration Checklist

1. Load the checkpoint and read `metadata.feature_config`, `metadata.feature_layout`, `metadata.target_names`, and `metadata.norm_stats`.
2. Build graph inputs with the same `FeatureConfig` used by the checkpoint.
3. Keep `scheduler_label_v3.jsonl`, `scheduler_resource_label.jsonl`, and `precision_metadata.jsonl` beside the materialized labels or at the dataset root.
4. Run the teacher and student per hardware ID. Do not mix hardware IDs inside one run ID.
5. Convert model outputs back to raw units before reporting or comparing metrics.
