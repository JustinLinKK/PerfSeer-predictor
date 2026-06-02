# Hardware Generalization TODO

## Current Interpretation

- Current PerfSeer predictions should be treated as labels for the dataset hardware, not universal GPU predictions.
- The six predicted metrics are:
  - `train_util`: average SM utilization, percent-like `0-100`.
  - `train_mem`: peak training memory usage, dataset raw memory unit.
  - `train_time`: training time, per-sample milliseconds; multiply by batch size for per-iteration time.
  - `infer_util`: average SM utilization, percent-like `0-100`.
  - `infer_mem`: peak inference memory usage, dataset raw memory unit.
  - `infer_time`: inference time, per-sample milliseconds.
- Existing graph/model features do not include target GPU specs, so time/utilization extrapolation to new hardware is not reliable as-is.

## Recommended Path

1. Add per-GPU calibration first.
   - Keep the current trained predictor.
   - Profile a stratified subset of graphs on each target GPU.
   - Fit per-metric calibration in standardized log space, starting with affine calibration.
   - Track calibration quality separately for time, memory, and utilization.

2. Build a hardware-conditioned predictor next.
   - Add hardware metadata as graph/global features:
     - GPU architecture or compute capability.
     - SM count.
     - Memory bandwidth.
     - VRAM size.
     - L2 cache size.
     - Peak FP32/Tensor Core throughput if available.
     - CUDA/cuDNN/runtime versions when controlled.
   - Train one shared model across GPUs instead of one full model per GPU.

3. Use targeted profiling, not full cross-product profiling.
   - Do not profile all 53k graphs on every GPU initially.
   - Start with roughly `1k-5k` stratified graphs per GPU.
   - Sample across batch sizes, FLOP ranges, memory ranges, operator mixes, graph depth, and branchiness.
   - Add more profiling data only where validation errors remain high.

## Implementation Notes

- Add a hardware-profile file format, likely YAML or JSON, for GPU specs.
- Extend `FeatureConfig` with an optional hardware feature block.
- Store hardware profile identity in checkpoint metadata and result ledgers.
- Add calibration/evaluation reports grouped by hardware.
- Keep the current source-code converter useful by allowing it to accept a target hardware profile at inference time.

## Avoid For Now

- Do not rely on zero-shot extrapolation from RTX 3090-style labels to very different GPUs.
- Do not train completely separate full models per GPU unless the deployment scope is only one or two fixed GPUs.
- Do not profile the full dataset on every GPU before first testing whether calibration plus hardware features is enough.
