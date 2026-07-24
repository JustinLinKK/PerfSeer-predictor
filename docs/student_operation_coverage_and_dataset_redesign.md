# PerfSeer Student Operation Coverage and Dataset Redesign Report

## Executive summary

The deployed A10 student does not consume arbitrary PyTorch source directly.
There are four separate gates:

1. `torch.fx.symbolic_trace` must be able to trace the model.
2. Shape propagation must run successfully on CPU with the supplied example
   inputs.
3. `perfseer_source_converter` must classify every tensor-producing FX node.
4. Every emitted operation label must have a feature slot in the student's
   fixed operation vocabulary.

The current student contract is `53 / 3 / 40`:

- 53 node values = 23 operation one-hot values + 30 continuous values;
- 3 edge values;
- 40 global values = 13 graph values + 23 operation-histogram values +
  4 precision values.

The converter emits 35 distinct operation labels. Only 17 of those labels are
present in the deployed student's 23-label vocabulary. The other 18 labels
were previously converted but received an all-zero operation identity. The
encoder now rejects them so the scheduler can use its per-job branch-profile
fallback instead of trusting a structurally ambiguous ML prediction.

The current artifact must not be described as a general PyTorch predictor. It
is suitable for graphs composed from its represented operation set and similar
to its training distribution. Supporting more operations properly requires a
versioned feature schema, representative A10 measurements, new normalization
statistics, retraining or distillation, and a new exported artifact.

## Authoritative implementation

| Concern | Source |
| --- | --- |
| PyTorch/FX classification | `src/perfseer_source_converter/converter.py::_classify_node` |
| Ignored pass-through operations | `src/perfseer_source_converter/converter.py::_is_passthrough` |
| Student operation vocabulary | `src/perfseer_student/features.py::OP_VOCAB` |
| Node/global construction | `src/perfseer_student/features.py::featurize_graph` |
| Scheduler-facing source encoder | `src/perfseer_student/encoder.py::encode_source` |
| Deployed dimensions and artifact | `models/registry.json` |

This report describes the checked-in `nvidia_a10_student_v1` artifact. A future
artifact must carry its own versioned vocabulary and feature-layout metadata;
the operation ordering cannot safely remain an undocumented Python constant.

## Current student vocabulary

The following 23 identities have explicit one-hot and global-histogram slots:

| Group | Student labels |
| --- | --- |
| Arithmetic and structure | `Add`, `Concat`, `Flatten` |
| Convolution | `Conv`, `DepthwiseConv` |
| Activations | `Gelu`, `Relu`, `Silu`, `Softmax` |
| Dense and embedding | `Embedding`, `Gemm` |
| Pooling and resize | `GlobalAveragePool`, `MaxPool`, `Upsample` |
| Sequence | `Attention`, `GRU`, `LSTM` |
| Normalization | `LayerNormalization` |
| Domain-specific generated graphs | `DetectorHead`, `GraphAttention`, `GraphMessage`, `SegmentationHead`, `TabularFeature` |

Six vocabulary labels are not emitted by the normal PyTorch source classifier:
`Attention`, `DetectorHead`, `GraphAttention`, `GraphMessage`,
`SegmentationHead`, and `TabularFeature`. They can occur in generated graph
records, but a normal `nn.Module` conversion does not currently produce them.

## Converter labels missing from the student

These 18 labels pass FX classification but have no operation slot in the
deployed student:

| Priority | Converter label | Typical PyTorch source | Why it matters |
| --- | --- | --- | --- |
| P0 | `BatchNormalization` | `nn.BatchNorm2d`, `F.batch_norm` | Extremely common in CNN training graphs |
| P0 | `AveragePool` | `nn.AvgPool2d`, non-global adaptive average pool | Common downsampling operation |
| P0 | `Mul` | `x * y`, `torch.mul` | Used by gates, attention, residual scaling, SE blocks |
| P0 | `MatMul` | `torch.matmul`, `x.matmul(y)` | Core attention and transformer operation |
| P0 | `Bmm` | `torch.bmm`, `x.bmm(y)` | Batched attention and sequence computation |
| P0 | `Reshape` | non-flattening `view` or `reshape` method | Common in attention and tensor-layout changes |
| P0 | `Transpose` | `transpose` or `permute` method | Common in attention, mixers, and image-to-sequence models |
| P0 | `MultiHeadAttention` | `nn.MultiheadAttention` | Naming disagrees with the existing `Attention` slot |
| P1 | `ConvTranspose` | `nn.ConvTranspose2d` | Common in decoders, segmentation, and generative models |
| P1 | `GroupNormalization` | `nn.GroupNorm` | Common when batches are small |
| P1 | `Sigmoid` | `nn.Sigmoid`, `torch.sigmoid` | Gates and binary/multilabel heads |
| P1 | `Tanh` | `nn.Tanh`, `torch.tanh` | Recurrent and bounded-output computations |
| P1 | `HardSwish` | `nn.Hardswish`, `F.hardswish` | MobileNet-family models |
| P1 | `HardSigmoid` | `nn.Hardsigmoid`, `F.hardsigmoid` | Mobile gates |
| P1 | `Sub` | `x - y`, `torch.sub` | Elementwise arithmetic |
| P1 | `Div` | `x / y`, `torch.div` | Scaling and normalization |
| P1 | `Reduce` | non-spatial `mean` | Token pooling and aggregate features |
| P2 | `RNN` | `nn.RNN` | Less common than GRU/LSTM, but already classified |

`nn.MultiheadAttention` should probably emit `Attention`, not introduce a
second synonymous label. That change is safe only after verifying that the
training graphs used `Attention` with the same semantic and continuous-feature
construction.

`torch.erf` is currently classified as `Tanh`. That is semantically incorrect
and should be fixed in the converter and represented independently or through
an explicit elementwise-activation family.

## Common operations rejected during conversion

The following list is intentionally focused on model structures likely to
appear in scheduler jobs. PyTorch has thousands of callable functions; the
complete rejection rule is any tensor-producing FX node not handled by
`_classify_node`.

### Convolution and spatial transforms

- `nn.Conv1d`, `nn.Conv3d`
- `nn.ConvTranspose1d`, `nn.ConvTranspose3d`
- lazy convolution variants
- `nn.Fold`, `nn.Unfold`
- grid sampling and affine-grid operations
- deformable or custom convolution implementations

### Normalization

- `nn.BatchNorm1d`, `nn.BatchNorm3d`, `nn.SyncBatchNorm`
- `nn.InstanceNorm1d`, `nn.InstanceNorm2d`, `nn.InstanceNorm3d`
- `nn.RMSNorm`
- `nn.LocalResponseNorm`

### Pooling

- 1-D and 3-D max/average/adaptive pooling
- adaptive max pooling
- fractional max pooling
- LP pooling
- max unpooling

### Activations and nonlinear mathematics

- `LeakyReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `RReLU`
- `Mish`, `Softplus`, `Softsign`, `LogSigmoid`, `LogSoftmax`, `GLU`
- threshold and shrink activations
- trigonometric functions such as `sin`, `cos`, and `tan`
- exponential, logarithmic, square-root, reciprocal, absolute, and power
  functions
- clamp, minimum, maximum, remainder, and modulo operations

### Reductions

Only `mean` is classified. Common missing reductions include:

- `sum`, `prod`, `max`, `min`, `amax`, `amin`
- `argmax`, `argmin`
- `norm`, `var`, `std`
- `logsumexp`

### Tensor manipulation and indexing

- function-form `torch.reshape`, `torch.transpose`, and similar layout calls
- `stack`, `unbind`, `repeat`, `expand`, `tile`, `roll`, and `flip`
- `gather`, `scatter`, `index_select`, masked selection, and `where`
- `einsum`, dot products, outer products, sorting, and top-k selection

### Transformer, sparse, and specialized computation

- `nn.Transformer*` wrappers
- `scaled_dot_product_attention`
- rotary-position embedding operations
- `EmbeddingBag` and `Bilinear`
- sparse, FFT, linear-algebra, and quantized operators
- custom C++/CUDA operations and custom `autograd.Function` nodes
- data-dependent Python control flow that `torch.fx` cannot symbolize

## Operations currently removed as pass-through

The converter deliberately omits:

- dropout modules/functions and `Identity`;
- `getitem` and slicing;
- squeeze and unsqueeze;
- padding;
- contiguous, clone, detach, type/device conversion, and float conversion;
- chunk and split.

Ignoring dropout in evaluation mode, identity, detach, and view-only shape
changes is usually reasonable. Padding, clone, dtype/device conversion, chunk,
and split can have real allocation or kernel costs. They should be measured
before deciding whether they remain pass-through nodes in a future resource
model.

## Why adding names to the encoder is insufficient

Adding `N` independent one-hot operation slots to the present layout changes
the contract to:

- node dimension: `53 + N`;
- edge dimension: `3`;
- global dimension: `40 + N`.

Adding all 18 converter-only labels independently would therefore produce
`71 / 3 / 58`. The current TorchScript artifact has weights for `53 / 3 / 40`
and cannot consume those tensors. Reusing or reordering an existing slot would
also change its learned meaning.

Each new operation needs:

1. stable classification for module, function, and method spellings;
2. correct tensor dependencies in the converted graph;
3. correct parameter, input, output, byte, and FLOP estimates;
4. operation-specific arguments where they affect cost;
5. training examples spanning relevant shapes, dtypes, and batch sizes;
6. normalization statistics computed only from the training split;
7. a retrained or redistilled checkpoint;
8. a versioned TorchScript export and registry record.

## Recommended v2 feature design

A v2 model should avoid an indefinitely growing one-hot vector while retaining
enough detail to distinguish costly kernels.

Recommended categorical fields:

- a stable broad operation family: convolution, dense/matmul, elementwise,
  activation, normalization, pooling/reduction, layout/indexing,
  embedding/sequence, attention, domain head, and unknown;
- a versioned fine operation identity for common operations;
- dimension/rank class: 1-D, 2-D, 3-D, scalar, sequence, or unknown;
- implementation flags such as depthwise, transposed, grouped, adaptive,
  in-place, bidirectional, fused, and view-only;
- explicit `Unknown` and `Custom` categories.

Recommended continuous additions:

- full kernel/stride/padding/dilation shape summaries instead of only the first
  scalar;
- dtype byte width and accumulation dtype;
- input/output/parameter element counts and byte counts;
- broadcasting ratio and reduction dimensions;
- sequence length, head count, head dimension, and attention matrix size;
- sparsity and group count;
- whether an operation materializes storage or only changes a view;
- optimizer state multiplier and training/inference mode.

The vocabulary and layout should live in a schema JSON file identified by a
schema ID and hash. Checkpoints and registry entries should record that hash,
and runtime loading should fail if the encoder schema differs.

## Dataset redesign

### Labels to collect

For each model, batch size, precision, optimizer, and hardware combination:

- average used GPU VRAM in MiB over the measured training window;
- peak allocated, peak reserved, and peak device-used VRAM;
- step time and examples per second;
- GPU compute utilization and memory-controller utilization;
- training and inference measurements kept as separate targets;
- warm-up count, measured-step count, repeat ID, and failure/OOM status.

Average VRAM must come from time-series device-used memory samples over a
defined steady-state window. It must not be computed from NVML memory
utilization percentage or from a single peak allocator value.

### Two complementary data sources

1. **Operation microbenchmarks**
   - isolate each new operation;
   - sweep tensor rank, shape, dtype, batch size, and important arguments;
   - include forward-only and forward/backward/optimizer measurements;
   - establish whether aliases and broad families have equivalent resource
     behavior.
2. **Composite model graphs**
   - CNN encoders and decoders;
   - residual, dense, depthwise, and mobile networks;
   - MLP and mixer-style graphs;
   - recurrent and embedding models;
   - transformers and attention variants;
   - segmentation, detection, graph, and tabular models;
   - realistic generated models from MLEvolve.

Microbenchmarks teach the cost of individual operations. Composite graphs teach
interactions, activation lifetimes, branching, joins, and allocator behavior.

### Coverage matrix

Every retained operation should have:

- at least three distinct model families;
- at least five materially different tensor shapes;
- multiple batch sizes including an OOM boundary where safe;
- all supported precision modes;
- multiple optimizer-state regimes for training labels;
- repeated measurements after warm-up;
- both isolated and composite examples.

Rare operations should be deliberately oversampled. Dataset reports should
publish operation counts, co-occurrence counts, shape/batch/precision coverage,
OOM counts, and hardware counts.

### Split policy

Do not randomly split individual measurements from the same source model.
Group splits by architecture/source family so width, batch, and precision
variants of one model cannot leak into both training and evaluation.

Maintain:

- an in-distribution validation split;
- an architecture-held-out split;
- an operation-combination-held-out split;
- a generated-code robustness split;
- a separately reported out-of-vocabulary rejection suite.

### Measurement quality gates

- pin the PyTorch, CUDA, driver, and library versions;
- record exact GPU identity and clock/power state;
- warm up before sampling;
- repeat measurements and reject unstable runs using a documented threshold;
- verify no other GPU process contributes memory;
- save raw time series, not only summaries;
- mark OOM and conversion failures explicitly rather than dropping rows;
- validate every graph against the declared schema before training.

## Prioritized implementation order

1. Make unknown operation identity a hard conversion failure. This is now
   implemented for the v1 encoder.
2. Resolve `Attention` versus `MultiHeadAttention` and the incorrect `erf`
   mapping.
3. Add the highest-frequency missing families: batch normalization, average
   pooling, elementwise arithmetic, matmul/bmm, reshape/transpose, and
   attention.
4. Add common normalization, activation, reduction, and decoder operations.
5. Add 1-D/3-D, sparse, graph, detection, segmentation, and custom-operation
   coverage based on actual scheduler workload frequency.
6. Train v2, validate held-out architectures, export a new artifact, and retain
   v1 as a separately versioned model until migration is complete.

## Stress Test Data v1.0

The parent repository contains Stress Test Data v1.0 at
`scheduler_benchmark_test/fixtures/stress_test_data_v1.0/`.
It is a deterministic compatibility stress-test fixture, not a claim of broad
PyTorch coverage.

The historical `standard_histopath_v1` 100-model list from parent commit
`9df6343` could not serve as this compatibility fixture:

- all five of its variants selected BatchNorm, GroupNorm, or InstanceNorm;
- two variants selected unsupported LeakyReLU or ELU activations;
- its mixer models used transpose, reduce, and sometimes multiplication;
- its recurrent image adapters used permute/transpose;
- its transformer models used the mismatched `MultiHeadAttention` label,
  transpose, and reduce.

Consequently, successful training execution of those models did not prove that
the `53/3/40` student could represent them. The new fixture keeps the useful
100-job stress-test shape while separating current-v1 compatibility testing from
future-v2 operation coverage.

The fixture contains:

- 100 one-epoch model specifications;
- 20 architecture structures and five variants per structure;
- five balanced families and four balanced precision modes;
- input sizes from 32 through 96;
- only operations explicitly represented by the current student;
- scheduler-compatible `metadata.perfseer_model` fields.

Its verifier runs all 100 entries through:

1. the scheduler's model-metadata parser;
2. CPU FX tracing and shape propagation;
3. the strict student-vocabulary gate;
4. `53 / 3 / 40` feature construction;
5. the retained CPU TorchScript artifact.

Acceptance requires 100 finite positive `train_mem` predictions, CPU-only
predictor tensors, and no change in CUDA allocation. This fixture protects the
current deployment path while the broader v2 dataset is being designed.
