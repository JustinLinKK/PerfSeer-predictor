# PerfSeer v3 Encoder and Teacher–Student Model Goal Prompt

Copy the prompt below into Codex Goal Mode while working in the `JustinLinKK/PerfSeer-predictor` repository.

---

## Goal

Work in the `JustinLinKK/PerfSeer-predictor` repository, starting from the current `v2` branch, and research, design, implement, test, and document a production-quality **PerfSeer v3 graph encoder and teacher–student predictor suite**.

The primary goal is to improve training-resource prediction accuracy by replacing v2's narrow, flat, forward-only operation encoding with a versioned, hierarchical, phase-aware graph representation that can structurally encode nearly all ordinary PyTorch neural-network computations, preserve unknown/custom operators without crashing, and accurately represent the work performed by the full training step.

The secondary goal is to increase model capacity where it produces measurable accuracy gains:

- The v3 teacher may be substantially larger than the v2 teacher because it is not deployed in the scheduler.
- The v3 student may be moderately larger than the v2 student, but it must remain suitable for low-latency CPU deployment.
- Do not assume that larger automatically means better. Select the final capacities through controlled ablations and Pareto analysis.

Do not only write a proposal. Implement the v3 suite in reviewable stages, run all tests and feasible benchmarks, and leave reproducible commands and evidence. If full training cannot be completed because the required dataset or GPU time is unavailable, complete and validate the implementation with smoke datasets, produce the exact full-training commands, and clearly distinguish measured results from proposed targets.

## Repository facts to verify before changing code

Reinspect the branch instead of assuming these facts are still current:

- `src/perfseer/architecture_schema.py` currently defines a fixed 35-label operation vocabulary under `perfseer_graph_v1`.
- `src/perfseer_source_converter/converter.py` uses hand-written FX classification and does not faithfully represent the full training step.
- `src/perfseer-optimized/model.py` feeds flat node, edge, and global vectors through one `Linear` layer each before the existing SeerNet message-passing trunk.
- The v2 teacher config uses approximately `hidden: 1024`, `num_blocks: 8`, and `head_hidden: 1024`.
- The v2 student config uses approximately `hidden: 192`, `num_blocks: 2`, and `head_hidden: 192`.
- `src/perfseer-optimized/data.py`, `train.py`, evaluation code, checkpoint metadata, `src/perfseer_student/export.py`, and `scripts/run_hardware_distill_flow.py` assume the current flat feature contract.
- The generated workload runtime may execute substitute implementations for some declared operations. v3 measurements must profile the same callable and semantics that were captured.

Before editing:

1. Read repository instructions and inspect `git status`, the current branch, relevant tests, dataset formats, profiler/runtime paths, checkpoint/export code, and scheduler integration.
2. Freeze a reproducible v2 baseline: commit SHA, dependency versions, schemas, model parameter counts, prediction metrics, coverage, artifact size, and CPU inference latency.
3. Preserve unrelated user changes and preserve v2 compatibility. Add versioned v3 paths/configs instead of silently changing v2 artifacts.

## Non-negotiable architectural decisions

### 1. Replace fixed one-hot operation growth with hierarchical encoding

Do not solve v3 by continually appending columns to `NODE_TYPES`.

Each operation node must be encodable through:

- a stable broad operation-family ID;
- an exact common-operation ID selected from measured corpus coverage;
- the operator overload/backend identity where relevant;
- a stable hash bucket for long-tail or custom operator identity;
- an explicit execution phase: `forward`, `loss`, `backward`, or `optimizer`;
- dtype, accumulation dtype, layout, rank, device/backend, and semantic flags;
- normalized continuous cost, tensor, topology, and liveness features;
- explicit `unknown`, `custom`, and feature-quality indicators.

An unfamiliar ATen, CUDA, Triton, Transformer Engine, or PyG operation must never crash the encoder merely because it is absent from a fixed vocabulary. It must use family + hash/custom features, reduce confidence appropriately, and remain visible in coverage reports.

### 2. Build a versioned canonical graph IR

Introduce separately versioned identifiers for at least:

- model release: `perfseer_v3_teacher` / `perfseer_v3_student`;
- graph IR: `perfseer_ir_v3`;
- feature schema: `perfseer_graph_v3`;
- operation registry: `perfseer_aten_ops_v3`;
- label schema.

Use `torch.export` as the primary forward/loss capture frontend because its graph is normalized to functional ATen/custom operators. Support explicit example args/kwargs and dynamic-shape constraints. Use strict export first; permit non-strict export only when its eager-equivalence checks pass on multiple legal inputs.

Keep legacy FX only as a diagnostic or compatibility fallback. Never silently omit an unsupported node.

The IR must preserve:

- every operation and every tensor-producing output;
- multiple edges between the same producer and consumer;
- source output slot and destination input slot;
- input, output, parameter, buffer, constant, gradient, and optimizer-state roles;
- tensor shape, symbolic dimensions, dtype, strides/layout, contiguity, alias/view/materialization status, bytes, and lifetime;
- lifted parameters and buffers;
- source module path/stack when available;
- dynamic-shape constraints and capture quality;
- in-place, random, reduction, broadcast, sparse, quantized, fused, grouped, transposed, and custom flags;
- phase boundaries and graph-level statistics.

Use a typed IR or multigraph representation before converting to PyG. Do not use a plain `networkx.DiGraph` in a way that loses tensor slots or parallel edges.

### 3. Represent the full training step

Do not continue predicting epoch time, utilization, and peak memory from an evaluation-mode forward graph alone.

Implement a version-pinned training capture adapter with:

1. forward and loss representation from the exported program;
2. backward representation through a supported AOT Autograd or Compiled Autograd path where feasible;
3. a validated analytical fallback for backward nodes when capture is unavailable;
4. explicit optimizer representation derived from the real optimizer/configuration, with exact handling for SGD, Adam, AdamW, foreach, and fused variants;
5. gradient accumulation, gradient clipping, AMP/loss scaling, activation checkpointing, saved-for-backward tensors, gradient tensors, optimizer states, and temporary workspace/liveness features.

Compiled Autograd is evolving, so isolate it behind a small adapter with pinned-version tests and a structured fallback. Never claim exact backward coverage when the system used an estimate.

### 4. Profile exact semantics

The same source model/callable, inputs, precision, optimizer, loss, and training configuration must be used for:

- capture;
- eager-versus-captured correctness replay;
- warm-up;
- profiling;
- label creation.

Do not label a graph `MatMul`, `Bmm`, graph attention, or another operation while executing `Linear` or another substitute. Store a workload fingerprint and fail data generation when capture and profiling fingerprints disagree.

Analytical FLOPs and byte estimates are auxiliary features, not substitutes for hardware measurements. Correct all unit ambiguities: use explicit names such as `_numel` and `_bytes`, support mixed dtypes, and test formulas against independent references.

## Required operation coverage

Implement structural encoding for every family below. Common operations must have exact canonical IDs and tested feature extractors. Composite or fused architectures such as MLA and Kimi Delta Attention may be represented by their normalized primitives plus a semantic/fused marker; they must not depend on a fragile one-off label. Opaque custom kernels must use the custom-operation path.

| Family | Required representative coverage | Important distinctions |
|---|---|---|
| Convolution | Conv1d/2d/3d, transposed 1d/2d/3d, grouped/depthwise, functional/native convolution | Per-axis asymmetric kernel, stride, padding, dilation, groups, bias, layout |
| Matrix/tensor contraction | Linear, `mm`, `addmm`, `matmul`, `bmm`, `einsum`, batched/broadcast variants | Batch dimensions, transpose state, tensor-core alignment, accumulation dtype |
| Attention | Ordinary attention, SDPA math/flash/memory-efficient backends, causal/padding masks, self/cross-attention, MHA, MQA/GQA, MLA, attention residual paths, Kimi Delta Attention | Heads, KV heads, sequence lengths, mask type, dropout, backend, fused/decomposed state |
| Normalization | BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm, SyncBatchNorm, native/functional forms | Reduced axes, epsilon, affine state, training statistics, dtype |
| Activations | ReLU/ReLU6, Sigmoid, Tanh, LeakyReLU, PReLU, ELU, SELU, SiLU, Mish, Softplus, HardSwish, HardSigmoid, GELU, Softmax/log-softmax | In-place, approximation, axis, fused state |
| Unary mathematics | exp, log, sqrt, rsqrt, erf, abs, negation, reciprocal, trigonometric operations | Dtype, vectorization, domain and broadcast shape |
| Arithmetic/comparison | Mul, Add, Sub, Div, power, remainder, clamp, minimum/maximum, comparisons, `where` | Broadcasting, scalar/tensor operands, in-place state |
| Reductions | sum, mean, product, max/min, argmax/argmin, norm, variance, standard deviation, log-sum-exp | Axes, keepdim, reduction size, accumulation dtype |
| Losses | Cross-entropy, NLL, BCE/BCE-with-logits, MSE, KL divergence and common reductions | Target type/shape, class count, weighting, ignore index, reduction |
| Tensor layout/copy | reshape/view, flatten, transpose/permute, stack, concat, split, chunk, unbind, repeat, expand, tile, squeeze/unsqueeze, contiguous, clone, casts/conversion | View versus materialization, aliasing, axes, bytes moved |
| Indexing/scatter/sort | slice, select, gather, scatter, index-select, index-put, masked operations, top-k, sort | Index dtype/shape, sparsity, reduction mode, materialization |
| Pooling/resampling | 1-D/2-D/3-D average/max pooling, adaptive pooling, interpolation modes, unpool, padding, grid sampling | Kernel/stride/padding per axis, mode, scale, align-corners |
| Sequence/embedding | Embedding, EmbeddingBag, decomposed and native RNN/GRU/LSTM computation | Vocabulary/table size, sequence length, bidirectionality, layers, packed sequences |
| Sparse/graph | Sparse matrix multiplication, segment/scatter reductions, GraphConvolution, GraphAttention, GraphMessage and registered PyG operations | Node/edge counts, density, aggregation, heads, custom/fused state |
| Quantization/precision/custom | Quantize/dequantize, casts, FP16, BF16, FP8, FP4/NVFP4, Transformer Engine and registered CUDA/Triton ops | Treat precision as dtype/configuration unless an actual quantize/cast kernel occurs; encode scale/granularity/backend |
| Training | Backward ops, accumulation, clipping, scaling, SGD, Adam, AdamW, foreach/fused variants | Phase, saved tensors, gradient/state bytes, optimizer hyperparameters |
| Random/regularization | Dropout variants and other common random tensor operations | Training/eval mode, probability, RNG/materialization cost |

Treat “decoded,” “exactly covered,” and “accurately predicted” as separate claims:

1. **Structurally encodable:** every captured node receives family/hash/custom features.
2. **Exactly covered:** the operator has a canonical exact ID and verified feature/cost extraction.
3. **Accurately predicted:** representative measured examples pass model-level prediction gates.

Report all three levels rather than using one optimistic coverage percentage.

## v3 feature encoders

Refactor `SeerTrunk` so categorical identities are not normalized as continuous numbers and continuous values are not treated as arbitrary one-hot columns.

### Node encoder

Combine learned embeddings for:

- exact operation;
- operation family;
- stable custom/hash bucket;
- phase;
- input/output/accumulation dtype;
- rank, layout, backend, and feature-quality categories;

with an MLP over normalized continuous features:

- per-dimension shape summaries and dynamic-range bounds;
- input/output/parameter/buffer/saved-tensor/workspace numel and bytes;
- FLOPs/MACs and arithmetic intensity;
- tensor-core alignment;
- fan-in/fan-out, graph depth, critical-path position;
- live bytes, reuse distance, and lifetime;
- operation-specific normalized arguments and flags.

Project the combined representation into the common hidden dimension. Evaluate additive versus concatenation-plus-projection fusion.

### Edge encoder

Use embeddings for tensor role, source/output slot, destination/input slot, dtype, layout, alias/view/materialization, and phase transition. Combine them with an MLP over bytes, shape/rank summaries, stride/contiguity, lifetime, fan-out, and reuse.

### Global encoder

Represent hardware ID/features, precision policy, optimizer, batch size, gradient accumulation, dataset/input statistics, capture mode/quality, dynamic-shape policy, graph size, per-phase totals, unknown/custom fractions, and critical/live-memory summaries.

### Trunk, pooling, and heads

Keep the existing v2 SeerBlock/GNPB/gated-residual trunk as the first controlled baseline. Do not replace the whole GNN before measuring the benefit of the new representation.

Add a phase-aware pooling experiment that separately summarizes forward, loss, backward, and optimizer nodes before forming the graph representation. Compare it with the existing SynMM/SynMMPlus/attention pooling.

Preserve the six scheduler metric outputs:

- training epoch/step time;
- average SM utilization;
- p95 SM utilization;
- peak VRAM used;
- peak PyTorch reserved memory;
- peak memory-controller utilization.

Add versioned optional heads for:

- OOM probability and failure stage;
- per-target uncertainty/log variance;
- OOD/capture confidence;
- peak-live-byte auxiliary reconstruction.

Do not silently change the scheduler's six-value contract. Expose new outputs through versioned checkpoint/export metadata and a compatible deployment wrapper.

## Teacher and student capacity study

The current v2 teacher/student configurations are controls, not final v3 choices.

Run an explicit capacity sweep and report actual trainable parameters, peak training memory, training time, validation accuracy, calibration, artifact size, and CPU p50/p95 inference latency.

### Teacher candidates

Use these as starting candidates, adjusting batch size or using predictor-training AMP/gradient checkpointing when necessary:

| Candidate | Hidden | Blocks | Purpose |
|---|---:|---:|---|
| T0 | 1024 | 8 | v2-capacity control with the v3 encoder |
| T1 | 1280 | 10 | recommended first larger v3 teacher; roughly a 2× trunk-capacity region |
| T2 | 1536 | 10 | high-capacity ceiling; train only if T1 still underfits and resources permit |

Use larger categorical embeddings in the teacher, with an initial search around:

- exact operation: 64–128;
- family: 32–64;
- hash/custom: 16–32;
- phase: 16–32;
- dtype/layout/rank/backend: 8–24 each.

Select the teacher on family-held-out validation error and uncertainty calibration, not training loss alone. Prefer T1 unless T2 yields a statistically meaningful improvement. Do not increase beyond approximately 3× v2 teacher parameters without evidence of capacity-limited underfitting.

### Student candidates

| Candidate | Hidden | Blocks | Purpose |
|---|---:|---:|---|
| S0 | 192 | 2 | v2-capacity control with the v3 encoder |
| S1 | 224 | 2 | recommended modest increase |
| S2 | 256 | 2 | accuracy-oriented ceiling under deployment budget |
| S3 | 256 | 3 | optional only if S2 underfits and latency/artifact gates still pass |

Start student embeddings near 25–50% of teacher embedding widths. Distill both graph-level predictions and, where useful, phase-pooled/encoder representations from the matching v3 teacher. Retain hard-label loss on real measured samples.

The default deployment gates are:

- CPU p95 inference latency no more than 1.25× the v2 student on the same graph-size buckets;
- artifact size no more than 1.5× v2;
- no unacceptable memory increase in scheduler integration.

If S2/S3 violates the budget, select S1 or use measured compression/quantization rather than hiding the regression.

## Dataset and training redesign

Build a coverage-driven corpus rather than relying only on fixed architecture quotas.

Include:

- exact-operation microbenchmarks spanning shape, dtype, layout, batch, and implementation regimes;
- small compositional graphs that exercise branching, joins, residuals, saved activations, and liveness;
- real source models across image, text, audio, video/3-D, temporal, graph, tabular/recommendation, generative, and MoE workloads;
- precision/backend variants, including supported Transformer Engine/custom kernels;
- full training configurations with multiple losses, optimizers, accumulation settings, and batch sizes;
- valid OOM and near-OOM examples.

At minimum, add golden or corpus fixtures for:

- MobileNetV3, ConvNeXt/EfficientNet, TCN/audio CNN, U-Net/DeepLab, and a 3-D CNN/U-Net;
- BERT, GPT-style decoder, T5, ViT, Swin, DETR, a diffusion U-Net/DiT, and representative MLA/Kimi Delta Attention components;
- RNN, GRU, and LSTM;
- EmbeddingBag/recommender;
- GCN, GAT, GraphSAGE/message passing;
- top-k gather/scatter MoE routing;
- registered custom CUDA/Triton/Transformer Engine examples where the environment supports them.

Use architecture/source-family grouped splits and shape-regime holdouts. Prevent sibling variants or identical graph signatures from leaking across train, validation, and test. Include a matched v2-compatible set and a newly supported/complex-operation set.

Train in stages:

1. optional encoder pretraining on operation identity, cost, liveness, and microbenchmark targets;
2. v3 teacher supervised training on real hardware measurements;
3. v3 student hard-label training plus distillation from the matching v3 teacher;
4. per-hardware/precision calibration and OOM/uncertainty calibration using validation data only.

Do not use a v2 teacher to fill missing v3 labels. Do not load v2 checkpoints strictly into changed encoders. A controlled experiment may reuse shape-compatible trunk weights while reinitializing encoders and heads, but compare it with random initialization.

## Implementation sequence and blocking gates

Implement the work as reviewable stages. Later stages must not hide failures in earlier stages.

1. **Baseline and coverage auditor**
   - Reproduce v2.
   - Produce operation-, model-, FLOP-, byte-, and measured-GPU-time-weighted coverage reports.
   - Record capture failures and effective source coverage separately from schema labels.
2. **v3 schemas, operator registry, and graph IR**
   - Add deterministic serialization, hashes, compatibility checks, and migration boundaries.
3. **`torch.export` capture**
   - Add strict/non-strict policy, dynamic constraints, multi-output tensor edges, custom-op registration checks, and eager replay.
4. **Cost, liveness, and training-phase representation**
   - Correct units/formulas and add backward/optimizer adapters with confidence/source metadata.
5. **Source-first profiling and workload fingerprints**
   - Eliminate substitute execution and reject semantic mismatches.
6. **Coverage-driven corpus generation**
   - Add model fixtures, active gap selection, grouped splits, and leakage checks.
7. **v3 PyG feature builder and safe coarsening**
   - Preserve high-cost, phase-boundary, branch/join, custom, and liveness-critical nodes.
8. **Hierarchical v3 encoder and optional heads**
   - Keep the old trunk as control, add phase-aware pooling experiment, and support all-unknown graphs.
9. **v3 teacher/student configurations and training**
   - Add `v3_teacher.yaml`, `v3_student.yaml`, capacity sweep configs, and a v3 distillation runner.
10. **Evaluation, export, and scheduler migration**
   - Add ablations, TorchScript/other supported CPU export, schema checks, fallback policy, and versioned result contract.

Do not begin expensive full teacher training until capture correctness, schema compatibility, source/profile identity, grouped splits, and dataset-validity gates pass.

## Required tests

Add unit, property, golden, integration, and corpus tests for:

- canonicalization aliases and overloads;
- every operation family in the finalized July 26 coverage list;
- multi-output/tuple operations and multiple tensor edges;
- views, aliases, materializing copies, in-place ops, broadcasting, and dynamic shapes;
- dtype/byte/FLOP formulas under mixed precision;
- liveness and saved-for-backward accounting;
- forward/loss/backward/optimizer phase attribution;
- custom operations using fake/meta implementations and `torch.library.opcheck` where appropriate;
- eager versus captured execution equivalence;
- unknown/custom-only graphs producing finite predictions and low confidence;
- capture → IR → PyG → teacher/student → export integration;
- checkpoint/schema/hash mismatch failing closed;
- v2 compatibility and scheduler fallback.

Pin relevant PyTorch behavior in tests because export and compiled-autograd APIs evolve.

## Acceptance criteria

Treat these as provisional until the v2 baseline report exists. If a gate must change, justify it with data and record the change; never weaken it silently.

### Coverage and correctness

- No silently dropped tensor operation.
- 100% of nodes in every successfully captured graph are structurally encoded by exact ID, family/hash, or custom path.
- At least 95% strict complete capture on the declared supported model corpus, with validated non-strict capture reported separately.
- At least 99% complete encoding among successfully captured models.
- Unknown/custom operations contribute no more than 2% of cumulative measured GPU time on the declared supported validation corpus; list the remaining operations.
- Every required family has golden tests across important dimensionality/dtype variants.
- Captured execution and eager execution pass numerical equivalence within documented dtype-appropriate tolerances.
- Capture and profiling workload fingerprints match.
- Schema/registry/hash mismatch fails closed.

### Prediction

- Report MAE, MAPE with a documented near-zero policy, RMSE in raw/log space, R², and p50/p90/p95 percentage error for each scheduler target.
- Report slices by architecture, operation family, graph size, phase, batch size, optimizer, precision, unknown fraction, and capture quality.
- On the matched v2-compatible test set, no target degrades by more than 5% relative to v2 unless statistically justified.
- On the newly supported/complex held-out set, target at least a 15% relative reduction in the teacher's weighted prediction error and a 10% reduction for the student versus the strongest applicable v2/fallback baseline.
- Demonstrate that improvements are not caused by source-family or graph-signature leakage.
- Calibrate uncertainty and OOM outputs; report interval coverage, OOM recall, precision, false-positive rate, and failure-stage accuracy.

### Deployment

- Student CPU latency and artifact size pass the budgets above on small, median, and large graph buckets.
- The exported v3 student matches eager predictions within tolerance.
- Existing six scheduler metrics remain available through a versioned compatible contract.
- Unsupported capture, high unknown fraction, OOD input, or schema mismatch produces a structured status and activates the existing branch-profile fallback instead of returning an unqualified prediction.

## Required ablations

At minimum compare:

1. v2 one-hot/flat input versus v3 hierarchical input;
2. forward-only versus forward + backward + optimizer;
3. exact-op only versus family + exact + hash/custom;
4. topology-only versus cost + liveness features;
5. raw versus safely coarsened graph;
6. existing pooling versus phase-aware pooling;
7. v2-capacity teacher versus T1/T2;
8. S0 versus S1/S2 student;
9. random initialization versus compatible trunk-weight reuse;
10. student hard labels only versus hard labels + v3-teacher distillation.

Do not refactor the SeerBlock trunk to a graph transformer or heterogeneous GNN unless the hierarchical encoder plus phase-aware pooling remains a measured bottleneck. If you test a new trunk, keep it as an ablation against the existing trunk.

## Deliverables

Produce:

- implemented v3 source, configs, tests, and migration/export code;
- `docs/perfseer_v3_encoder_model_design.md` explaining the final IR, encoders, training graph, model pair, and tradeoffs;
- reproducible v2 and v3 coverage reports in JSON and Markdown;
- a model-corpus compatibility matrix distinguishing structurally encodable, exactly covered, and accuracy-validated support;
- actual parameter counts and teacher/student capacity-sweep results;
- prediction, calibration, OOM, latency, artifact-size, and ablation reports;
- exact commands to generate data, audit coverage, train the teacher, distill the student, evaluate, export, and run scheduler integration;
- a concise list of remaining unsupported or low-confidence operations and the correct fallback behavior;
- a final implementation summary listing files changed, tests/benchmarks run, measured results, unresolved risks, and recommended next work.

Use official PyTorch documentation and source as primary references for `torch.export`, dynamic shapes, custom-operator fake/meta registration, AOT/Compiled Autograd, and optimizer capture. Record the exact PyTorch/CUDA/Transformer Engine/PyG versions used.

The task is complete only when the v3 schema and encoder can safely represent the finalized operation families, the model pair can train and export under the versioned feature contract, coverage and accuracy are evaluated without leakage, and unsupported cases fail safely rather than being silently misdecoded.

