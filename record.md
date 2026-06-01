# PerfSeer Replication Record

## Setup Investigation (2026-05-29)

### Remote host

- Host alias `Justin-Linux` -> hostname `Justin-WS`
- CPU: AMD Ryzen 9 9950X3D, 32 threads
- RAM: 45 GiB; Disk: 785 GiB free on `/`
- GPU training is available. The `perfseer` conda environment reports one CUDA device:
  `NVIDIA GeForce RTX 5090`.
- Use GPU for model training when available. Use CPU for inference latency tests and
  scheduler-deployment benchmarks.
- Python 3.12.3 (system); `uv` 0.11.17 installed to `~/.local/bin/uv`
- Project path: `~/PerfSeer-predictor` (git repo `git@github.com:JustinLinKK/PerfSeer-predictor.git`, was empty except LICENSE/.gitignore/README)
- venv: `~/PerfSeer-predictor/.venv` (created via `uv venv --system-site-packages`)

### Source material

- Paper: arXiv:2502.01206v1 "PerfSeer" (Shandong Normal Univ.)
- Official repo `github.com/upuuuuuu/PerfSeer`: contains ONLY dataset utilities
  (`util_dataset/features_define.py`, `example.py`) + model figure. **No model/training code.**
  => SeerNet / SeerNet-Multi must be implemented from scratch per paper.

### Dataset (downloaded via gdown to remote `~/PerfSeer-predictor/dataset/`)

- Google Drive folder `1T7DNKUyjIdnLIMTL4IZA67ynvhA75Pdt`
- `cg.zip` 239 MB -> `dataset/cg/cg/*.pkl` = 53407 compute graphs (networkx DiGraph, pickled)
- `label.zip` 22 MB -> `dataset/label/label/*.txt` = matching labels (same filename stem)
- Batch-size distribution (filename prefix): bs1..bs256, ~6.4k each (bs256 only 3334 due to OOM)

### Real data format (verified by loading a sample)

Node feature dict keys: `type`, `args`, `memory_info`, `flops`, `arith_intensity`.

- `type` (str): one of {Conv, Relu, BatchNormalization, Concat, AveragePool,
  GlobalAveragePool, Flatten, Gemm, MaxPool, Add} (~10 types observed)
- `args` (13 ints): conv_{kernel_size,stride,padding,dilation,groups,bias},
  linear_{in_features,out_features,bias}, pool_{kernel_size,stride,padding,ceil_mode}
- `memory_info` (12 fields): bytes, weight_size, batch_size, input_size_with_weight,
  input_size, input_channels, input_w, input_h, output_size, output_channels,
  output_w, output_h. **NOTE: several values are stored as STRINGS** (e.g. "282594816")
  -> must cast to float.
- `flops` (int), `arith_intensity` (float = FLOPs/MAC)

Edges: `compute_graph.edges()` carry **NO attributes** (`{}`). Edge features (tensor size
`e^sz`, shape `e^sp` = batch,channel,h,w) must be DERIVED from the SOURCE node's
`memory_info` output_{size,channels,w,h} + batch_size.

Label format (eval'd dict): `{'train': '<7 fields>', 'infer': '<7 fields>'}`
fields = `time|average_sm_util|average_memory_util|average_memory_usuage|peak_sm_util|peak_memory_util|peak_memory_usuage`
- time = per-sample execution time (ms); multiply by batch_size for per-iteration time.

### 6 target metrics (paper Tables 1-3)

| Paper column | Phase | Label field |
|---|---|---|
| Training Util | train | average_sm_util |
| Training Mem  | train | peak_memory_usuage |
| Training Time | train | time |
| Inference Util| infer | average_sm_util |
| Inference Mem | infer | peak_memory_usuage |
| Inference Time| infer | time |

### Reproduction targets (paper)

- SeerNet (single-metric): mean MAPE 5.14% across 6 metrics, params 1.02 M
- SeerNet-Multi (+PCGrad): mean MAPE 7.75%, params 1.15 M
- Training: split 2:1:1 train/val/test, batch 128, Adam, lr 1e-3 halved after 5 epochs
  w/o improvement down to 1e-6, up to 500 epochs, MSE loss
- Metrics: MAPE, RMSPE, x%Acc (within relative error x%)

## Implementation

- Code under `src/perfseer/` (data.py, model.py, train.py, eval.py, metrics.py), `pyproject.toml` (src layout, `uv pip install -e .`)
- Feature dims (computed): NODE_DIM=30, EDGE_DIM=5, GLOBAL_DIM=18, NUM_TARGETS=6
- SeerNet params at hidden=256, num_blocks=1: **1,263,107** (paper target ~1.02M; faithful to eqs 1-8, not trimmed)
- Pipeline: log1p + train-only z-score; targets predicted in std-log space, inverted (expm1) before metrics
- Dataset processed/cached via PyG InMemoryDataset; cache keyed by split name + hash(pairs + norm_stats)

## Experiment 1: SeerNet single-metric, full dataset (started 2026-05-29)

### Setting

- `python -m perfseer.train --metric all --epochs 500 --patience 30 --out runs/full --data-root dataset`
- 6 independent single-metric models; split 2:1:1 (seed 42); batch 128; Adam lr 1e-3;
  ReduceLROnPlateau halve/patience5/min1e-6; early stop patience 30 epochs; MSE in std-log space; GPU training when available
- Remote Justin-WS, nohup PID 47138, log `logs/train_full.log`

### Smoke test (pre-run, 200 train graphs / 2 epochs, metric 0)

- Pipeline runs end-to-end on real data; train loss 0.96 -> 0.16; eval table renders.
  (undertrained sanity only; not a result)

### Result

- (pending)
