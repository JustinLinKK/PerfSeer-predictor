# PerfSeer v3 Operation-Coverage and Encoder Redesign Plan

## Purpose

Implement a v3 hardware-performance predictor that can encode and predict resource behavior for most practical PyTorch neural-network workloads, including models that use operations outside the current hand-written v2 classifier.

This plan is intentionally implementation-oriented. It is organized as a sequence of small pull requests with explicit files, tests, metrics, and stop gates. Do not begin v3 teacher/student training until the graph-capture, schema, and dataset-validity gates pass.

Repository baseline reviewed:

- Repository: [`JustinLinKK/PerfSeer-predictor`](https://github.com/JustinLinKK/PerfSeer-predictor)
- Branch: `v2`
- Reviewed head: `a24e13979906f8ea8242ce45408ee7b4d1202f4d`
- v2 teacher: [`src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml)
- v2 student: [`src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml)

## Executive decision

Do not solve v3 by adding more `isinstance(...)` branches and more one-hot columns.

The correct v3 direction is:

1. Replace `torch.fx.symbolic_trace` as the primary frontend with `torch.export`.
2. Normalize source models into a version-pinned ATen graph.
3. Preserve every tensor-producing operation, including unknown and custom operations.
4. Encode operations hierarchically:
   - stable broad operation family;
   - exact common operation identity;
   - hashed long-tail/custom identity;
   - semantic and implementation flags;
   - shape, dtype, byte, FLOP, topology, and liveness features.
5. Represent forward, backward, and optimizer work instead of predicting training behavior from a forward-only graph.
6. Profile the exact callable that produced the graph. Never profile a substitute implementation under another operation label.
7. Select v3 training workloads from measured operation-coverage gaps, not only from fixed architecture quotas.
8. Train a new v3 teacher from the rebuilt dataset, then distill a v3 student. Do not load the v2 encoder weights as if the feature semantics were unchanged.

`torch.export` is a better primary frontend because it produces a sound, normalized, flattened ATen graph with tensor shape/dtype metadata and supports substantially more Python programs than symbolic FX. PyTorch explicitly distinguishes full-graph `torch.export` from partial `torch.compile` capture and documents the lower-level ATen result: [PyTorch `torch.export` overview](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export.html).

## 1. Findings from the current v2 branch

### 1.1 The attached 23-operation report is not the current v2 schema

The attached report correctly explains why an operation vocabulary and a trained model artifact must evolve together. However, its `23 / 53 / 3 / 40` artifact description is older than the current `v2` branch.

Current `src/perfseer/architecture_schema.py` declares 35 node types:

`Conv`, `DepthwiseConv`, `ConvTranspose`, `BatchNormalization`, `LayerNormalization`, `GroupNormalization`, `Embedding`, `Gemm`, `MatMul`, `Bmm`, `Attention`, `MultiHeadAttention`, `RNN`, `GRU`, `LSTM`, `GraphMessage`, `GraphAttention`, `Relu`, `Gelu`, `Silu`, `Softmax`, `Sigmoid`, `Mul`, `Add`, `Concat`, `Flatten`, `Reshape`, `Transpose`, `AveragePool`, `MaxPool`, `GlobalAveragePool`, `Upsample`, `DetectorHead`, `SegmentationHead`, and `TabularFeature`.

See [`src/perfseer/architecture_schema.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer/architecture_schema.py).

The two v2 YAML files both use `feature_schema_version: perfseer_graph_v1`, so “model release v2” and “feature schema v1” are currently different version namespaces. v3 must make this distinction explicit in metadata.

### 1.2 The source converter still covers only a narrow subset

The current converter in [`src/perfseer_source_converter/converter.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer_source_converter/converter.py):

- uses `torch.fx.symbolic_trace`;
- runs `ShapeProp` on CPU;
- forces the model into `.eval()`;
- specializes to one concrete example-input set;
- manually classifies `call_module`, `call_function`, and `call_method`;
- raises `UnsupportedOpError` on an unclassified tensor operation;
- emits a forward-only `networkx.DiGraph`.

The normal source path can currently emit approximately 28 of the 35 schema labels. The following schema labels are not produced by the normal classifier:

- `Attention`
- `GraphMessage`
- `GraphAttention`
- `Sigmoid`
- `DetectorHead`
- `SegmentationHead`
- `TabularFeature`

Most of these appear only in hand-generated graph records. `Sigmoid` is present in the schema and runtime but is not classified in the reviewed converter.

The remaining unsupported source operations include the report’s missing operations plus many common cases:

- 1-D/3-D convolutions, transposed variants, and function-form convolution;
- instance norm, RMS norm, sync batch norm, and additional normalization forms;
- subtraction, division, power, comparisons, clamp, min/max, and most unary math;
- `sum`, `prod`, min/max, variance, standard deviation, norms, and log-sum-exp;
- function-form reshape/transpose and most indexing/scatter operations;
- `stack`, `repeat`, `expand`, `tile`, `split`, `chunk`, and `unbind`;
- losses and optimizer operations;
- scaled-dot-product/flash attention;
- sparse, quantized, FFT, and custom operators.

### 1.3 The current graph loses important tensor semantics

The converter’s graph format has structural limitations beyond operation names:

- `_tensor_meta` returns the first tensor metadata object for tuple/list outputs.
- Tuple outputs from recurrent or attention operations are not represented completely.
- `operator.getitem` is aliased to one predecessor rather than represented as an output slot.
- A `networkx.DiGraph` cannot retain multiple tensor edges between the same producer and consumer.
- Input position, output position, dtype, stride, layout, aliasing, and view/materialization semantics are not stored on edges.
- Parameters lifted or referenced through `get_attr` are excluded from topology.

This is why adding classifier cases alone would still leave inaccurate graphs for attention, recurrent networks, branching, indexing, and custom operators.

### 1.4 Current cost features contain unit and formula problems

Audit and correct these before generating any v3 labels:

- Convolution FLOPs assume a square 2-D kernel and use only the first kernel scalar.
- `Gemm` FLOPs multiply only by `output_shape[0]`, so sequence-leading dimensions can be omitted.
- `weight_size` is generally an element count, while several downstream names imply bytes.
- `input_size_with_weight` adds `input_size / element_size` to a parameter element count, mixing a byte-derived term with elements.
- `total_bytes` multiplies parameter elements by the output element size, which can be wrong under mixed dtypes.
- arithmetic intensity is calculated using `total_bytes / element_size`, producing FLOPs per element rather than FLOPs per byte.
- layout, indexing, copy, padding, and upsample operations can have substantial memory cost even if their arithmetic FLOPs are zero.
- attention assumes a particular output layout and a simplified formula.
- backward activation storage, optimizer state, gradient tensors, workspaces, and allocator lifetime are absent.

Add unit names such as `_numel` and `_bytes` to all v3 fields. Never use an ambiguous `_size` suffix.

### 1.5 The generated workload can label one operation while executing another

The v2 generated runtime in [`nrp_calibration_pack/profile/generated_model_runtime.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/nrp_calibration_pack/profile/generated_model_runtime.py) reconstructs graph specifications as executable modules, but some mappings are not semantically equivalent:

- `MatMul` and `Bmm` are implemented as `nn.Linear`;
- `Attention` is implemented as a simplified self-attention block;
- `Reshape` always becomes `reshape(batch, -1)`;
- `Transpose` always swaps dimensions 1 and 2;
- graph attention and graph message use the same simplified message layer;
- domain heads are approximations.

This invalidates operation-specific learning: the graph says one operation while the GPU measures another.

For v3, the same model instance/callable must be used for:

1. graph capture;
2. correctness validation;
3. warm-up;
4. profiling;
5. label creation.

### 1.6 The v2 model predicts training targets from a forward-only graph

The v2 targets include epoch time, SM utilization, peak VRAM, reserved memory, and memory-controller utilization. Those measurements include:

- forward;
- loss;
- backward;
- gradient handling;
- optimizer update;
- optionally AMP/gradient scaling.

The encoder sees only the evaluation-mode forward graph. This is a fundamental information gap, especially for custom autograd, activation checkpointing, optimizer selection, fused/foreach optimizers, recurrent models, and attention.

PyTorch’s AOT Autograd and Compiled Autograd paths can expose backward computation. Compiled Autograd is still evolving, so v3 should isolate it behind a version-pinned adapter and retain a safe analytical fallback: [Compiled Autograd tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html).

### 1.7 Existing tests are useful but too small

[`scripts/test_source_converter.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/scripts/test_source_converter.py) covers a small CNN, residual add, concatenation, a GRU/attention-like example, and deployment integration.

It does not currently enforce:

- source-model capture rate across a real corpus;
- unique/weighted operation coverage;
- multi-output edge correctness;
- dynamic shapes;
- backward/optimizer coverage;
- unknown/custom-op behavior;
- alias/view behavior;
- analytical byte/FLOP accuracy;
- graph/profile semantic identity.

## 2. v3 target architecture

### 2.1 Version namespaces

Use separate versions:

- model release: `perfseer_v3_teacher`, `perfseer_v3_student`;
- raw graph IR: `perfseer_ir_v3`;
- feature schema: `perfseer_graph_v3`;
- label schema: `scheduler_resource_label_v3` or a new v4 only if fields change;
- operator registry: `perfseer_aten_ops_v3`.

Every checkpoint and deployment artifact must store all applicable IDs and hashes.

### 2.2 Capture pipeline

Use this ordered policy:

1. `torch.export.export(..., strict=True)` with explicit example args/kwargs and dynamic-shape constraints.
2. If strict export fails, optionally try `strict=False`.
3. Validate a non-strict result by replaying it against eager mode on multiple legal input samples.
4. Keep legacy symbolic FX only as a diagnostic comparison path, not as the production v3 graph.
5. If all capture paths fail, return a structured failure and let the scheduler use its branch-profile fallback.

Never silently drop a node to make capture succeed.

`torch.export` graphs contain normalized ATen/custom operators, lifted parameters, and per-node tensor shape/dtype metadata. It is therefore unnecessary and fragile to enumerate every `nn.Module`, functional alias, and tensor method separately.

### 2.3 Functionalization and selective decomposition

Do not indiscriminately decompose every operation to tiny primitives. That would erase fused-kernel semantics that matter for runtime prediction.

Produce:

1. a functional ATen graph with mutation and aliasing normalized;
2. a selective decomposition pass for unsupported composite operations;
3. a semantic summary from the pre-decomposition graph.

Preserve expensive/fused semantic operations where possible:

- convolution;
- linear/addmm/mm/bmm;
- native normalization operations;
- scaled-dot-product attention and backend-specific attention;
- recurrent primitives;
- sparse matrix operations;
- quantize/dequantize;
- fused optimizer operations;
- registered custom/fused kernels.

Decompose inexpensive aliases and wrapper operations when this improves canonicalization.

### 2.4 v3 graph IR

Create `GraphIRV3` dataclasses under `src/perfseer/graph_ir_v3.py`. Persist deterministic JSON during development; compression can be added later.

Each graph stores:

- schema IDs/hashes;
- capture backend/mode and PyTorch version;
- source/model fingerprint;
- input signature and dynamic constraints;
- training/inference mode;
- precision and optimizer configuration;
- operation nodes;
- tensor edges;
- graph-global features;
- coverage/quality metadata;
- capture warnings/failures.

Each operation node stores at least:

- `node_id`;
- raw target, e.g. `aten.add.Tensor` or `transformer_engine::...`;
- canonical operation ID;
- broad family ID;
- phase: `forward`, `loss`, `backward`, or `optimizer`;
- exact/common-op ID or `UNK`;
- stable long-tail hash bucket;
- source module stack/path where available;
- flags: in-place, view-only, materializing, reduction, broadcast, random, sparse, quantized, fused, custom, transposed, grouped, adaptive, bidirectional;
- normalized operation arguments;
- input/output tensor counts;
- parameter/buffer element and byte counts;
- FLOPs/MACs;
- bytes read/written;
- workspace estimate and quality flag;
- saved-for-backward bytes;
- topological and liveness features.

Each tensor edge stores:

- producer and consumer node IDs;
- producer output index and consumer input index;
- tensor role: activation, parameter, buffer, gradient, optimizer state, constant, or model input/output;
- shape or symbolic-shape expression;
- rank, dtype, element width, numel, and bytes;
- stride and memory-format class;
- alias group;
- view/materialization status;
- first-use/last-use distance;
- dynamic-shape quality.

Use a multiedge representation or a typed tensor-edge list. Do not use a plain `DiGraph` as the authoritative v3 representation.

### 2.5 Hierarchical operation encoding

Replace the unbounded one-hot operation block with:

- family embedding;
- exact common-operation embedding;
- long-tail hash embedding;
- phase embedding;
- dtype and accumulation-dtype embeddings;
- rank/layout embeddings;
- explicit semantic flags;
- normalized continuous cost/shape features.

Recommended starting dimensions:

- operation family embedding: 12–16;
- exact operation embedding: 24–32;
- long-tail hash embedding: 8–16;
- phase embedding: 4;
- dtype/layout/rank embeddings: 4–8 each.

Select the exact-operation vocabulary from measured corpus coverage, not a hand-written target count. A practical initial cap is the smallest vocabulary that covers at least 95% of cumulative operator GPU time in the training corpus. Everything else remains encodable through family + hash + generic features.

Unknown and custom operators must never receive an all-zero identity. Use:

- `family=unknown_or_custom`;
- `exact_op=UNK`;
- stable namespace/name hash bucket;
- all available tensor and cost features;
- `is_custom=1`;
- a graph-level unknown-cost/unknown-byte fraction.

### 2.6 Raw and deployment graph views

ATen graphs can be much larger than current generated graphs. Keep:

- raw IR for audit, reproducibility, and coverage;
- deterministic deployment view for GNN training/inference.

Safe deployment coarsening rules:

- collapse a chain only when it is single-entry/single-exit;
- preserve total FLOPs, bytes, saved bytes, and operation histogram;
- allow view-only chains to collapse while retaining alias/liveness effects;
- optionally collapse same-shape pointwise chains;
- never collapse convolution, dense/matmul, attention, normalization, reduction, indexing/scatter, sparse, random, or optimizer boundaries;
- never merge across a branch, join, mutation, dtype conversion, or materialization boundary.

Store coarsening ratio and member-operation histograms. Add an ablation that proves coarsening does not materially reduce prediction accuracy.

## 3. Operation coverage targets

The v3 registry should target ATen/custom operations, not Python spelling aliases.

### P0: required before v3 training

| Family | Required semantic coverage | Important features |
| --- | --- | --- |
| Dense/matrix | linear, addmm, mm, matmul, bmm, einsum, dot/mv, bilinear | M/N/K, batch product, transpose flags, bias, dtype/accumulation |
| Convolution | 1-D/2-D/3-D, transposed, grouped, depthwise | full kernel/stride/padding/dilation vectors, groups, channels, layout |
| Attention | scaled-dot-product attention, flash/efficient/math backends, MHA decompositions | batch, heads, sequence lengths, head dim, causal/mask/dropout, backend |
| Normalization | batch, layer, group, instance, RMS, sync/native forms | reduction axes, normalized size, affine params, epsilon, training mode |
| Elementwise arithmetic | add, sub, mul, div, pow, remainder, min/max, clamp, where | broadcast ratio, scalar/tensor mode, in-place/materialization |
| Activations/unary math | relu, gelu, silu, sigmoid, tanh, hard activations, leaky relu, elu/selu, softplus/mish, exp/log/sqrt/rsqrt/erf/abs/trig | approximation mode, in-place flag, output numel |
| Reduction | sum, mean, prod, min/max, amax/amin, argmin/argmax, norm, var/std, logsumexp | dimensions, keepdim, reduced ratio, accumulation dtype |
| Loss/probability | softmax/log-softmax, cross entropy/NLL, BCE, MSE, KL, common classification/regression losses | reduction mode, class count, ignore index, label dtype |
| Pool/resample | max/average/adaptive 1-D/2-D/3-D, interpolation modes, unpool, pad, grid sampling | full geometry, adaptive flag, interpolation mode |
| Layout/shape | view/reshape/flatten, transpose/permute, squeeze/unsqueeze, contiguous/clone/copy/to, cat/stack/split/chunk/unbind, expand/repeat/tile | view vs copy, axes, source/destination layout, materialized bytes |
| Index/scatter | slice/select/narrow, index, index_select, gather, scatter, scatter-reduce, index_put, masked select/fill, top-k/sort | index dtype, axis, update/reduction mode, selected/output ratio |
| Embedding/sequence | embedding, embedding bag, RNN/GRU/LSTM decompositions | vocabulary, embedding dim, sequence, layers, direction, packed state |
| Random/regularization | dropout variants, Bernoulli, random sampling used inside models | train/eval mode, probability, generated bytes |
| Training phase | backward ops, gradient accumulation/zeroing, clipping, loss scaling | phase ID, saved activations, grad bytes, accumulation steps |
| Optimizer | SGD, Adam, AdamW; single, foreach, and fused variants | parameter/state bytes, dtype, weight decay, momentum/betas |

### P1: add before claiming broad multimodal coverage

| Family | Coverage |
| --- | --- |
| Sparse/graph | sparse mm/addmm, sparse softmax, segment/scatter reduce, common PyG registered ops |
| Quantized/low precision | quantize/dequantize, fake quant, int8/FP8/FP4 custom/fused operations used by supported backends |
| Vision specialized | ROI align/pool, NMS, deformable convolution, pixel shuffle/unshuffle |
| Spectral/linalg | FFT family, solve, inverse, decompositions used by scientific/audio jobs |
| Audio | STFT/spectrogram-related tensor operations and supported custom kernels |
| Custom/fused | `torch.library` operators, Transformer Engine, Triton-backed custom ops, fused normalization/activation |

### P2: long-tail support

Keep these encodable through the generic path even before exact cost formulas exist:

- unusual distributions and sampling;
- complex-number operations;
- nested tensors;
- uncommon decompositions;
- application-specific C++/CUDA extensions.

The production encoder should accept them with a lower confidence score, not crash or silently treat them as a known operation.

## 4. Step-by-step implementation plan

## PR 1 — Freeze the v2 baseline and add a coverage auditor

### Changes

1. Create a v3 feature branch from the reviewed `v2` head.
2. Record the v2 commit SHA, Python/PyTorch/CUDA versions, dataset fingerprint, configs, checkpoints, and evaluation results.
3. Add `scripts/audit_operation_coverage.py`.
4. Add adapters under `coverage_corpus/` that return:
   - model;
   - args/kwargs;
   - optional dynamic-shape constraints;
   - loss function/targets for training;
   - optimizer factory.
5. Run both:
   - current v2 conversion;
   - experimental strict `torch.export` capture.
6. Write:
   - `reports/v2_operation_coverage.json`;
   - `reports/v2_operation_coverage.md`;
   - `reports/v2_capture_failures.jsonl`.

### Corpus

Start with:

- current PerfSeer generated model families;
- current real-dataset model adapters;
- MLEvolve-generated source samples;
- representative TorchBench models;
- torchvision image models;
- transformer encoder/decoder and vision-transformer models;
- recurrent/audio/time-series models;
- graph and tabular models.

TorchBench provides standardized train/eval model modules and example inputs and is an appropriate external coverage corpus: [official TorchBench repository](https://github.com/pytorch/benchmark).

### Metrics

Report all of:

- model capture success rate;
- complete-graph success rate;
- unique raw operations;
- exact-known, family-known, custom, and unknown counts;
- occurrence-weighted operation coverage;
- FLOP-weighted coverage;
- tensor-byte-weighted coverage;
- profiler-time-weighted coverage on a smaller GPU subset;
- unknown fraction by architecture and modality;
- failure stage and exception taxonomy.

### Gate

Do not set a target vocabulary count yet. The auditor must identify the smallest exact-op set that covers at least 95% of cumulative measured GPU operator time.

## PR 2 — Add the v3 schema, registry, and IR

### New files

- `src/perfseer/graph_ir_v3.py`
- `src/perfseer/op_registry_v3.yaml`
- `src/perfseer/schemas/perfseer_graph_v3.json`
- `scripts/build_v3_schema.py`
- `scripts/validate_v3_graph.py`
- `tests/test_graph_ir_v3.py`
- `tests/test_op_registry_v3.py`

### Registry rules

1. Use exact dispatcher/operator identities when available, including overload.
2. Normalize in-place/out variants through explicit alias tables, not substring matching.
3. Store raw identity even after canonicalization.
4. Map each operation to:
   - family;
   - exact common ID or `UNK`;
   - cost-formula ID;
   - semantic flags;
   - selective-decomposition policy.
5. Generate the runtime JSON and schema hash from one human-maintained registry.
6. Reject duplicate IDs, unstable ordering, missing `UNK`, or conflicting aliases in CI.

### Compatibility

Add a v2-to-v3 adapter only for regression testing. Do not mutate v2 graph files in place.

### Gate

Round-trip serialization must preserve every field. The schema hash must change if any feature order, categorical mapping, normalization rule, or registry entry changes.

## PR 3 — Implement `torch.export` source capture

### Refactor

Split the current monolithic converter into:

- `src/perfseer_source_converter/load.py`
- `src/perfseer_source_converter/inputs.py`
- `src/perfseer_source_converter/capture_export.py`
- `src/perfseer_source_converter/capture_training.py`
- `src/perfseer_source_converter/canonicalize.py`
- `src/perfseer_source_converter/tensor_metadata.py`
- `src/perfseer_source_converter/diagnostics.py`
- legacy `capture_fx.py`

Keep the current public entrypoint but add `capture_backend="export"` and return v3 IR when requested.

### Capture requirements

1. Preserve positional and keyword inputs.
2. Support nested pytrees of tensors.
3. Add optional `dynamic_shapes` to `SourceModelSpec`.
4. Distinguish user inputs, parameters, buffers, constants, and lifted state from `ExportGraphSignature`.
5. Recursively flatten every tensor output from node metadata.
6. Record producer output and consumer input slots.
7. Preserve zero-node/view-only models as valid graphs when appropriate.
8. Capture and normalize mutations rather than silently deleting them.
9. Store guards/range constraints.
10. Store source stack/module metadata when available.

### Custom operators

For registered custom operators:

- preserve namespace and overload;
- require or detect a fake/meta implementation for export;
- document how external operators register fake behavior through `torch.library`;
- use generic cost features if no exact formula exists;
- set a confidence/quality flag.

Do not use `torch.compiler.allow_in_graph` as a generic workaround for opaque Python functions.

### Non-strict fallback

If `strict=False` is used:

1. mark `capture_quality=non_strict`;
2. replay eager and exported programs on at least three valid randomized input samples;
3. compare output structure, shape, dtype, and values within configured tolerances;
4. reject the graph on mismatch.

### Gate

The converter must either:

- encode every tensor operation;
- encode it as unknown/custom;
- or return a structured capture failure.

It must never omit an unsupported tensor-producing node and continue as if the graph were complete.

## PR 4 — Add operation cost, tensor liveness, and phase features

### New files

- `src/perfseer/cost_registry_v3.py`
- `src/perfseer/liveness_v3.py`
- `src/perfseer/training_features_v3.py`
- `tests/test_cost_registry_v3.py`
- `tests/test_liveness_v3.py`

### Correct units

Use explicit fields:

- `input_numel`, `output_numel`, `parameter_numel`;
- `input_bytes`, `output_bytes`, `parameter_bytes`;
- `bytes_read`, `bytes_written`;
- `saved_for_backward_bytes`;
- `optimizer_state_bytes`;
- `estimated_workspace_bytes`;
- `flops`, `macs`;
- `arithmetic_intensity_flops_per_byte`.

Every estimator returns:

- value;
- method, e.g. `exact_formula`, `shape_formula`, `profiled_prior`, or `unknown`;
- confidence.

### Formula tests

For each P0 family:

1. generate random valid shapes and arguments;
2. compare output metadata against eager execution;
3. compare FLOPs against a trusted reference or derivation;
4. compare byte units directly from dtype and numel;
5. cover asymmetric dimensions and broadcasting;
6. cover mixed input/parameter/accumulation dtypes.

### Liveness

Compute:

- tensor birth and last consumer;
- live bytes at each topological position;
- estimated peak live activation bytes;
- fan-out and reuse distance;
- materialized-copy bytes;
- view/alias groups.

Use these as features, not as replacements for measured VRAM labels.

### Training graph

Implement a version-pinned `TrainingGraphCapture`:

1. capture forward/loss with PT2;
2. use AOT/Compiled Autograd integration to obtain backward nodes where supported;
3. capture optimizer work for supported optimizers;
4. tag every node by phase;
5. record saved tensors and gradient/state sizes.

Fallback:

- if backward capture fails, generate analytical backward/optimizer summary nodes;
- set `backward_capture_quality=estimated`;
- lower prediction confidence;
- retain safe scheduler fallback for unsupported custom autograd.

### Gate

On a validation suite, static tensor bytes must be exact for known concrete shapes. FLOP estimators must meet a documented per-family tolerance. All unit conventions must be tested.

## PR 5 — Replace substitute profiling with source-first profiling

### Refactor

Modify:

- `nrp_calibration_pack/build_pack.py`
- `nrp_calibration_pack/profile/run_profile.py`
- `nrp_calibration_pack/profile/make_workload_specs.py`
- `nrp_calibration_pack/workload.py`

Deprecate `GraphModel` as a v3 label source. It may remain for v2 reproduction only.

### Workload identity

Every profile point stores:

- model source/package fingerprint;
- parameter fingerprint or deterministic seed;
- input signature/fingerprint;
- captured graph hash;
- schema and registry hash;
- optimizer/loss/precision;
- batch and gradient accumulation;
- eager/compiled execution mode;
- graph-capture quality;
- exact callable used for profiling.

Before profiling:

1. execute eager output;
2. execute captured/exported output where possible;
3. validate structure/shape/dtype/value;
4. refuse to profile if the two represent different computations.

### Measurement

Keep the current sustained NVML sampling and CUDA allocator peaks, but also save:

- raw timestamped NVML samples;
- warm-up and measured-step boundaries;
- per-repeat and per-epoch durations;
- PyTorch/CUDA/driver/cuDNN/Transformer Engine versions;
- GPU clocks/power limits where available;
- compile mode and compile warm-up exclusion;
- OOM stage and requested configuration;
- other-process memory precheck;
- profiler trace for a sampled subset.

### Gate

No v3 row may claim `MatMul`, `Bmm`, attention, reshape, transpose, graph, or domain-head semantics while executing an unrelated substitute implementation.

## PR 6 — Build a coverage-driven v3 workload corpus

### Three data layers

#### A. Exact operation microbenchmarks

Generate one callable per canonical operation or tightly related family.

Sweep:

- rank and shape;
- small/medium/large and boundary dimensions;
- batch size;
- dtype and accumulation dtype;
- layout/memory format;
- operation-specific arguments;
- forward-only and training modes;
- optimizer when parameters exist;
- OOM boundary where safe.

Microbenchmarks must execute the actual operation named in the graph.

#### B. Composite blocks

Add realistic blocks:

- residual/bottleneck/dense CNN blocks;
- depthwise/mobile inverted bottlenecks;
- encoder/decoder and skip-connection blocks;
- self/cross attention, MLP, MoE, rotary-position, and KV-related blocks;
- recurrent and temporal blocks;
- embedding-heavy recommenders/tabular models;
- graph message-passing and attention blocks;
- audio/spectrogram blocks;
- mixed branches, joins, indexing, and reductions.

#### C. Real and generated models

Use:

- current real dataset adapters;
- representative TorchBench/torchvision/transformer/timm/PyG models;
- actual MLEvolve-generated source models;
- deliberately held-out families for evaluation.

### Active selection

After each profiling wave:

1. recompute exact/family/unknown coverage;
2. find underrepresented operation × shape × dtype × phase cells;
3. sample new microbench/composite workloads for those cells;
4. stop adding data when coverage and error improvements plateau.

This is more efficient than increasing every architecture quota uniformly.

### Minimum coverage matrix

For every P0 exact operation retained in the vocabulary:

- at least five materially different shape regimes;
- at least three composite architecture contexts where semantically possible;
- multiple batch sizes;
- every supported precision;
- forward and backward where differentiable;
- multiple optimizer regimes for parameterized operations;
- repeated stable measurements.

Rare but expensive operations must be oversampled.

### Split policy

Maintain:

- in-distribution validation;
- architecture/source-family held-out;
- operation-combination held-out;
- generated-code robustness;
- dynamic-shape extrapolation;
- precision/optimizer held-out slices;
- custom/OOV suite;
- v2-compatible matched test set.

Group all variants of one source architecture into one split. Never randomly separate batch/precision/width variants of the same source model.

## PR 7 — Implement the v3 feature builder and graph coarsener

### Modify

- `src/perfseer-optimized/data.py`
- `src/perfseer/architecture_schema.py` or replace it with a schema loader
- preprocessing/cache code

### PyG fields

Recommended v3 data fields:

- `x_cont`: continuous node features;
- `op_exact_id`;
- `op_family_id`;
- `op_hash_id`;
- `phase_id`;
- `dtype_id`;
- `layout_id`;
- `node_flags`;
- `edge_index`;
- `edge_cont`;
- `edge_role_id`;
- `u_cont`;
- categorical global fields;
- coverage/quality fields.

Do not concatenate categorical IDs into standardized floats.

### Normalization

- calculate means/stds from the training split only;
- keep categorical embeddings unstandardized;
- use log transforms only for nonnegative magnitude fields;
- store transform type per field in schema metadata;
- add robust clipping based on training quantiles and report clip frequency;
- invalidate caches when schema, registry, coarsening, split, or normalization hashes change.

### Coarsener

Implement `src/perfseer/coarsen_v3.py` with deterministic rules from Section 2.6.

For every coarsened region store:

- member count;
- family and exact-op histograms;
- total and maximum FLOPs/bytes;
- maximum live bytes;
- first/last tensor metadata;
- branch/join boundaries;
- materialization and random/mutation flags.

### Gate

A saved sample must validate against the checkpoint feature layout by name, ID, dimension, and hash. Dimension equality alone is insufficient.

## PR 8 — Add hierarchical categorical encoders to SeerNet

### Modify

- `src/perfseer-optimized/model.py`
- model config dataclasses
- export/deploy wrappers
- unit tests for empty graphs, isolated nodes, batching, and TorchScript/export

### Node encoder

Create:

```text
node_embedding =
  exact_op_embedding
  + projected(family_embedding || hash_embedding || phase_embedding)

node_hidden =
  MLP(node_embedding || dtype/layout/rank embeddings ||
      flags || normalized continuous features)
```

Concatenation followed by projection is also acceptable. Benchmark both on student inference latency.

### Auxiliary outputs

Add optional heads for:

- OOM probability;
- per-target uncertainty/log variance;
- capture/OOD confidence;
- optional graph-level peak-live-byte reconstruction.

Do not allow confidence to hide bad predictions. Evaluate calibration separately.

### Teacher/student

Keep the successful scale separation initially:

- teacher starting point: hidden 1024, 8 blocks;
- student starting point: hidden 192, 2 blocks.

Change capacity only after a controlled ablation. Data/schema correctness comes first.

The student can use the same deterministic coarsened graph view. If later latency requires more coarsening for the student, introduce dual graph views as a separate experiment, not in the first v3 training run.

### Gate

The model must accept a graph containing only unknown/custom operations and produce a finite prediction plus a low-confidence/OOD indication.

## PR 9 — Add v3 configs and staged training

### New configs

- `src/perfseer-optimized/configs/train_hardware_teacher/v3_teacher.yaml`
- `src/perfseer-optimized/configs/train_deploy_model/v3_student.yaml`

Suggested fields:

```yaml
features:
  feature_schema_version: perfseer_graph_v3
  graph_ir_version: perfseer_ir_v3
  op_registry_version: perfseer_aten_ops_v3
  capture_backend: torch_export
  include_training_graph: true
  include_tensor_liveness: true
  categorical_encoder: hierarchical_embedding
  unknown_policy: generic_hash
  graph_view: coarsened_v3

model:
  op_exact_embedding_dim: 32
  op_family_embedding_dim: 16
  op_hash_embedding_dim: 12
  phase_embedding_dim: 4
  predict_uncertainty: true
  predict_oom: true
```

Resolve final dimensions from ablation rather than treating these starting values as fixed.

### Training stages

1. **Encoder pretraining**
   - microbenchmark operation latency/resource targets;
   - analytical FLOP/byte reconstruction;
   - operation family/exact identity as auxiliary supervision.
2. **v3 teacher training**
   - real composite graph-level scheduler targets;
   - mix microbench and composite samples with explicit domain weights;
   - architecture-grouped splits.
3. **v3 student distillation**
   - distill from the matching v3 teacher;
   - retain hard-label weight for real measured rows;
   - never substitute v2 teacher predictions for missing v3 labels.
4. **Calibration**
   - per hardware and supported precision;
   - uncertainty/OOM calibration on validation only.

### Checkpoint initialization

Do not strictly load the v2 checkpoint because:

- categorical semantics changed;
- dimensions changed;
- graph topology changed;
- normalization changed;
- training-phase information was added.

An optional experiment may copy compatible message-passing block weights while reinitializing all encoders and heads, but compare it against full random initialization.

## PR 10 — Evaluation, ablation, and safety policy

### Coverage evaluation

Report:

- strict-export model success;
- validated non-strict success;
- complete v3 encoding success;
- exact/family/hash/unknown rates;
- occurrence/FLOP/byte/GPU-time weighted coverage;
- coverage by modality, architecture, precision, and phase.

### Accuracy evaluation

For each of the six scheduler targets, report:

- MAE;
- MAPE with a documented near-zero policy;
- RMSE in raw and log space;
- R²;
- p50/p90/p95 absolute percentage error;
- calibration/interval coverage if uncertainty is enabled.

Slice by:

- architecture family;
- operation family;
- graph size;
- batch size;
- precision;
- optimizer;
- resource regime;
- capture quality;
- unknown fraction;
- v2-compatible vs newly supported models.

### Required ablations

1. v2 one-hot vs v3 hierarchical encoding.
2. forward-only vs forward + backward/optimizer.
3. occurrence features vs cost/liveness features.
4. raw vs coarsened graph.
5. exact-op embedding only vs family + exact + hash.
6. hand-generated substitute workloads vs source-first workloads.
7. random split vs architecture-grouped split.
8. teacher hard-label only vs distillation.

### Provisional acceptance gates

Set final numeric thresholds after PR 1 records the baseline. At minimum:

- no silent operation drops;
- 100% of successfully captured tensor nodes encoded as exact, family/hash, or custom;
- at least 95% strict complete capture on the declared supported corpus, or document a smaller supported boundary;
- at least 99% complete encoding among successfully captured models;
- unknown/custom operations account for no more than 2% of cumulative measured GPU time on the supported validation corpus;
- v3 is not worse than v2 on the matched v2-compatible test set beyond the agreed statistical tolerance;
- v3 materially improves the complex/new-operation held-out set;
- student CPU p95 latency and artifact size remain within agreed relative budgets, initially 1.25× latency and 1.5× size of v2;
- no operation/shape/source-family leakage between training and held-out splits;
- schema mismatch fails closed.

If a target cannot be met, report the failing families and keep scheduler fallback enabled instead of weakening the gate silently.

## PR 11 — Export, registry, and scheduler migration

### Artifact metadata

Store:

- model release;
- graph IR ID/hash;
- feature schema ID/hash;
- operation registry ID/hash;
- full ordered feature layout;
- normalization hash;
- coarsening config/hash;
- target names/order;
- hardware and precision allowlist;
- capture-quality allowlist;
- supported optimizer/training modes;
- dataset/split fingerprints;
- PyTorch/CUDA build metadata.

At runtime, verify all hashes before inference.

### Scheduler result contract

Return one of:

- `ok`;
- `ok_with_unknowns`;
- `ood_low_confidence`;
- `unsupported_capture`;
- `unsupported_training_mode`;
- `schema_mismatch`;
- `encoder_error`.

Include:

- prediction;
- uncertainty/confidence;
- unknown GPU-cost proxy fraction;
- capture mode/quality;
- schema IDs;
- recommended fallback.

Use the existing branch-profile fallback for every non-`ok` state unless a scheduler policy explicitly accepts `ok_with_unknowns`.

### Migration

1. Keep v2 artifact and encoder intact.
2. Deploy v3 in shadow mode.
3. Compare v2/v3/fallback decisions and real observed resources.
4. Canary v3 only on high-confidence supported jobs.
5. Expand supported traffic after error and OOM guardrails pass.
6. Retain explicit rollback to v2.

## 5. Testing matrix

### Unit tests

- registry ID stability and alias rules;
- schema/hash generation;
- tensor metadata recursion;
- multi-output and repeated-edge topology;
- parameter/buffer/input roles;
- dynamic shape constraints;
- all P0 family cost formulas;
- view/copy/alias/liveness;
- unknown/custom operation encoding;
- coarsening invariants;
- normalization/cache invalidation;
- checkpoint/schema mismatch.

### Golden capture tests

Add small source modules for:

- Conv1d/2d/3d and transposed/grouped/depthwise;
- norm variants;
- matmul/bmm/einsum;
- SDPA and MHA;
- function/method/module spelling equivalence;
- reductions and losses;
- indexing/gather/scatter/top-k;
- layout/copy/view cases;
- embedding/RNN/GRU/LSTM;
- sparse/graph;
- custom `torch.library` operation;
- full train step with SGD, Adam, and AdamW.

Golden files should assert canonical graph semantics, not unstable FX node names.

### Property tests

Use generated legal shapes/arguments to verify:

- eager/export output equivalence;
- tensor numel/byte formulas;
- broadcasting;
- rank-general convolution;
- reduction output shapes;
- no negative/nonfinite encoded features;
- serialization round-trip.

### Integration tests

- capture → IR → PyG → teacher/student forward;
- profile → label → dataset row;
- train a tiny v3 teacher/student smoke pair;
- export → reload → identical prediction;
- scheduler status/fallback behavior;
- v2 and v3 coexistence.

### Corpus CI

Do not run the full corpus on every commit. Use:

- small representative PR smoke set;
- larger nightly coverage set;
- scheduled GPU profile validation;
- checked-in coverage baseline with regression thresholds.

## 6. Recommended pull-request sequence

| PR | Scope | Must merge before |
| --- | --- | --- |
| 1 | Baseline and coverage auditor | vocabulary decisions |
| 2 | v3 schema, registry, IR | capture/features |
| 3 | `torch.export` capture | v3 dataset generation |
| 4 | costs, liveness, training phases | profiling/training |
| 5 | source-first profiler | new labels |
| 6 | coverage-driven corpus | full data collection |
| 7 | v3 PyG features/coarsening | model training |
| 8 | hierarchical SeerNet encoder | configs/training |
| 9 | v3 teacher/student training | final evaluation |
| 10 | evaluation and safety gates | deployment |
| 11 | artifact registry and scheduler canary | production rollout |

Do not combine PRs 1–6 into a single dataset rewrite. Each stage should emit a report that can block the next stage.

## 7. Coding-agent completion checklist

The coding agent should not claim v3 completion until all items are true:

- [ ] Current v2 baseline and coverage report are reproducible.
- [ ] The attached 23-operation artifact is not confused with current v2’s 35-entry schema.
- [ ] `torch.export` is the default capture backend.
- [ ] Every captured tensor operation is retained or explicitly represented as unknown/custom.
- [ ] Multi-output and tensor-slot edges are correct.
- [ ] v3 units distinguish elements from bytes.
- [ ] Forward, backward, and optimizer information is represented or explicitly marked estimated.
- [ ] Profiling executes the same callable that graph capture describes.
- [ ] `MatMul`, `Bmm`, attention, reshape, and transpose are no longer profiled through substitutes.
- [ ] Operation vocabulary comes from coverage measurements.
- [ ] Long-tail/custom operations have a nonzero hierarchical identity.
- [ ] Dataset splits are source-family grouped.
- [ ] Raw measurement time series and environment metadata are retained.
- [ ] v3 teacher is trained on the rebuilt schema/dataset.
- [ ] v3 student is distilled from the matching v3 teacher.
- [ ] v2-compatible and newly supported test sets both pass their gates.
- [ ] Artifact and runtime schema hashes are enforced.
- [ ] Scheduler fallback remains active for capture, schema, OOD, and low-confidence failures.

## 8. Main risks and mitigations

| Risk | Mitigation |
| --- | --- |
| ATen graphs are too large for student latency | deterministic safe coarsening; exact-op embeddings; benchmark before changing capacity |
| Full decomposition loses fused semantics | selective decomposition plus pre-decomposition semantic summary |
| Compiled Autograd APIs change | isolate behind a pinned adapter; preserve analytical fallback and quality flag |
| Unknown custom op has no fake implementation | document/register `torch.library` fake kernel; otherwise structured capture failure |
| Microbenchmarks dominate and distort composite learning | domain weights; auxiliary pretraining; final fine-tuning on real composite labels |
| Same architecture leaks across splits | source-family/group fingerprint splitting |
| Schema evolves without retraining | schema/registry/layout/normalization hashes and fail-closed runtime |
| More coverage increases model size | embeddings rather than one-hot growth; coverage-derived exact vocabulary |
| Static cost formula disagrees with fused runtime | measured labels remain authoritative; formula method/confidence stored; profiler-time coverage reports |
| Non-strict export is unsound for a workload | multi-input replay validation; mark quality; scheduler fallback |

## 9. Source references

- PerfSeer v2 schema: [`src/perfseer/architecture_schema.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer/architecture_schema.py)
- PerfSeer source converter: [`src/perfseer_source_converter/converter.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer_source_converter/converter.py)
- PerfSeer optimized features: [`src/perfseer-optimized/data.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/src/perfseer-optimized/data.py)
- PerfSeer generated workload runtime: [`nrp_calibration_pack/profile/generated_model_runtime.py`](https://github.com/JustinLinKK/PerfSeer-predictor/blob/a24e13979906f8ea8242ce45408ee7b4d1202f4d/nrp_calibration_pack/profile/generated_model_runtime.py)
- PyTorch `torch.export` overview: [PyTorch 2.13 documentation](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export.html)
- PyTorch export API and normalized ATen contract: [`torch.export` API reference](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export/api_reference.html)
- PyTorch compiler IRs and Core ATen: [PyTorch IR documentation](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_ir.html)
- Symbolic FX behavior: [`torch.fx` documentation](https://docs.pytorch.org/docs/2.13/fx.html)
- Backward-graph capture: [Compiled Autograd tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
- Real PyTorch model corpus: [TorchBench](https://github.com/pytorch/benchmark)
