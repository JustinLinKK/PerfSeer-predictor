# PerfSeer v3 target-GPU dataset design report

Date: 2026-07-27

Implementation root: `src/perfseer_v3`

Status: design specification for production data collection. This document does
not claim that the proposed production dataset has been collected or that a
production teacher/student pair has passed its accuracy gates.

## 1. Executive decision

PerfSeer v3 uses one teacher/student pair for one concrete **target GPU model**.
For example, an `nvidia_h100_sxm_80gb` pair predicts workloads whose labels were
measured on NVIDIA H100 SXM 80 GB GPUs. A separate pair and separate label
dataset are built for `nvidia_rtx_5090_32gb`.

This target-hardware rule applies to the workload being predicted, not to the
device used to optimize or execute the PerfSeer neural network:

| Concept | Must match the pair's `target_hardware_id`? | Explanation |
| --- | --- | --- |
| GPU that produces workload labels | Yes | Its measurements are the learning target. |
| GPU named in every graph and manifest row | Yes | It identifies what the row predicts. |
| Teacher artifact target | Yes | The teacher models that target GPU. |
| Student artifact target | Yes | The student models the same target GPU. |
| GPU/CPU used to train the teacher | No | It only executes predictor training. |
| GPU/CPU used for student distillation | No | It may differ from both the label GPU and teacher-training GPU. |
| Device used to run the deployed predictor | No | CPU deployment is valid; the prediction request must still name the artifact's target GPU. |

During one distillation process, the teacher and student tensors must be on the
same **execution device**, but that execution device does not have to be the
target/label GPU. Teacher training and student distillation may occur on
different execution GPUs at different times.

The implementation already exposes these as separate values:

- `TrainingManifestV3.target_hardware_id` binds the dataset and artifact to the
  predicted GPU.
- `run_training(..., device_name=...)` independently chooses the predictor
  training or distillation execution device.
- The training report contains both `target_hardware_id` and `device`.
- Runtime `hardware_mismatch` compares the graph's requested workload target
  with the artifact target; it does not inspect the CPU/GPU executing the
  predictor.

## 2. Dataset objective and scope

The dataset should teach a hardware-specific model to map:

```text
captured training graph
+ model and tensor shapes
+ per-operation dtype behavior
+ optimizer and parameter groups
+ LR scheduler and training progress
+ workload controls
+ target-GPU characteristics
        -> training time, utilization, memory, OOM, and uncertainty
```

The first production revision should predict **single-GPU training workloads**
on one target GPU model. Keep `world_size=1`, no MIG partitioning, and no MPS
sharing. Multi-GPU jobs introduce communication topology, rank-local memory,
collective overlap, sharding, and network features that are not in the current
schema. They should become a separately versioned extension rather than being
silently mixed into the single-GPU dataset.

The production dataset must use real or faithfully replayed training workloads.
Synthetic operator and composite workloads remain useful for encoder
pretraining, cost calibration, boundary search, and coverage, but they are not
a substitute for scheduler-grade labels from representative end-to-end
training steps.

## 3. Normative terminology

The words **MUST**, **SHOULD**, and **MAY** describe required, recommended, and
optional dataset behavior.

- **Target hardware**: the concrete GPU model whose workload behavior is being
  predicted.
- **Label device**: the physical target-GPU card that executes a profiling run.
- **Target hardware ID**: canonical product identity shared by every row in one
  pair, such as `nvidia_h100_sxm_80gb`.
- **Device UUID**: identity of one physical label device. Several UUIDs may
  contribute to the same target-GPU dataset if they are the same qualified SKU.
- **Predictor execution device**: CPU/GPU that trains the teacher, performs
  distillation, or executes the deployed student.
- **Workload configuration**: immutable model, input, precision, optimizer,
  scheduler, and training-control configuration.
- **Raw run**: one fresh-process measurement repetition of one workload
  configuration.
- **Aggregated sample**: one training row produced from accepted repeated raw
  runs.
- **Source group**: all variants derived from the same architecture/source
  family that must remain in one dataset partition.
- **Graph signature**: immutable v3 graph SHA-256 used for integrity and
  leakage checks.

## 4. Pair identity and hardware qualification

### 4.1 What belongs in the target hardware ID

The ID SHOULD distinguish performance- or capacity-relevant product variants:

```text
vendor + product + form factor + memory capacity + partition profile
```

Examples:

```text
nvidia_h100_sxm_80gb
nvidia_h100_pcie_80gb
nvidia_a100_sxm_80gb
nvidia_a100_pcie_40gb
nvidia_rtx_5090_32gb
nvidia_h100_sxm_80gb_mig_3g_40gb   # future separate pair, not v1
```

Do not use marketing-family-only values such as `h100`, `gpu`, `cuda`,
`unknown`, `mixed`, or `any` for production rows. H100 PCIe and H100 SXM should
not be pooled merely because both contain `H100` in their name.

### 4.2 Hardware provenance recorded on every raw run

Record the following even when some fields are constant within a dataset:

- canonical target hardware ID;
- physical device UUID and PCI bus ID;
- exact product name and total VRAM;
- compute capability and SM count;
- ECC state and MIG mode/profile;
- driver, CUDA runtime/build, cuDNN, PyTorch, and compiler versions;
- VBIOS when available;
- configured and observed power limits;
- application/default SM and memory clocks;
- temperature, power, and clocks throughout the measured window;
- host CPU model, RAM, NUMA relationship, OS/kernel, and container digest;
- whether the GPU was exclusive and which other GPU processes were observed.

The pair identity may stay stable across driver revisions, but a driver,
framework, kernel, or compiler change creates a new **measurement environment
revision**. Do not pool revisions until a golden overlap set demonstrates that
their label distributions are compatible, because the current predictor does
not encode driver/library versions.

### 4.3 Multiple physical cards of one model

It is valid—and preferable—to collect on multiple physical cards with the same
qualified target ID. Record UUIDs, randomize work across cards, and reserve a
device-UUID-held-out audit slice. This measures card-to-card variability while
retaining one model pair for the hardware model.

## 5. Unit of observation and data layers

Do not write profiler output directly into the training manifest. Preserve
three independently auditable layers.

### 5.1 Raw measurement layer

One row per fresh-process repetition. It contains raw step durations, raw NVML
samples, PyTorch allocator peaks, environment, status, failure stage, and all
identity hashes. Raw data is append-only and is never normalized in place.

### 5.2 Aggregated sample layer

One row per unique workload configuration after quality checks. It contains the
six robust labels, uncertainty/dispersion summaries, OOM information, replicate
IDs, aggregation method, and quality flags. Model training consumes this layer.

### 5.3 Training manifest layer

The versioned `perfseer_v3_training_manifest_v2` selects graph/sample rows,
assigns train/validation/test partitions, freezes fingerprints, and declares
the deployment allowlists for one target GPU pair.

Repeated runs MUST be aggregated before splitting. Never put repetitions of the
same configuration in different partitions.

## 6. Canonical model outputs and label definitions

Successful samples have six ordered targets:

| Index | Target | Unit | Canonical production definition |
| ---: | --- | --- | --- |
| 0 | `train_epoch_ms` | ms/epoch | Steady-state optimizer-step wall time multiplied by the exact optimizer steps per epoch, with golden full-epoch validation. |
| 1 | `train_avg_sm_util_percent` | % | Time-weighted mean NVML GPU utilization during accepted measured windows. |
| 2 | `train_p95_sm_util_percent` | % | Time-weighted 95th percentile of the same valid utilization samples. |
| 3 | `train_peak_vram_used_mib` | MiB | Peak isolated-device framebuffer use attributable to the workload, with idle baseline and contamination recorded. |
| 4 | `train_peak_torch_reserved_mib` | MiB | `torch.cuda.max_memory_reserved()` after resetting peak statistics at the measurement boundary. |
| 5 | `train_peak_memory_controller_util_percent` | % | Maximum valid NVML memory-utilization sample during the accepted measured windows. |

The optimizer step measured by the primary timing label includes:

```text
zero_grad
+ gradient-accumulation forward/loss/backward passes
+ gradient scaling/unscaling when configured
+ gradient clipping when configured
+ optimizer.step
+ required synchronization at the timing boundary
```

It excludes one-time initialization, source loading, graph capture, compilation,
autotuning, checkpointing, validation, and dataset download. Compilation and
autotuning receive separate auxiliary labels. Input loading should either be
preloaded/prefetched so it cannot starve the GPU, or measured as a separate
`data_wait_ms` auxiliary target. Do not mix data-loader-bound and GPU-compute
labels unless host/storage inputs are added to the model.

Calculate:

```text
microbatches_per_epoch = ceil_or_drop_last(dataset_examples / micro_batch_size)
optimizer_steps_per_epoch = ceil(microbatches_per_epoch / gradient_accumulation_steps)
train_epoch_ms = robust_optimizer_step_wall_ms * optimizer_steps_per_epoch
```

Store the exact `drop_last` behavior and the number of shorter final steps.
Measure a full steady-state epoch on a golden subset to quantify extrapolation
error. Keep both `train_epoch_ms_measured` and
`train_epoch_ms_step_extrapolated` in raw/auxiliary data even though the current
six-target contract exposes one canonical `train_epoch_ms`.

### 6.1 Required auxiliary labels

Retain these for audits, uncertainty modeling, future outputs, and debugging:

- median, mean, standard deviation, p90, p95, and maximum optimizer-step wall
  time;
- CUDA-event GPU elapsed time and host wall time;
- measured and extrapolated epoch time plus extrapolation error;
- peak PyTorch allocated, active, inactive-split, and reserved bytes;
- average and peak device framebuffer use;
- raw SM/memory utilization samples and timestamps;
- average/peak power, temperature, SM clock, and memory clock;
- compilation/autotuning time and steady-state flag;
- OOM status and phase;
- peak analytical live bytes from the graph;
- repeat count, accepted count, rejection reasons, coefficient of variation,
  confidence interval, and aggregation rule.

### 6.2 OOM rows

OOM is a valid label, not a failed data-pipeline row. Collect deliberate boundary
cases and classify the stage using the model's current vocabulary:

```text
capture, forward, loss, backward, optimizer, allocator
```

For an OOM row, the six regression targets are undefined unless the full
measurement completed. They MUST be stored as null with a target-validity mask;
never insert zero or the last partial measurement as if it were a successful
label.

Current implementation blocker: `TrainingManifestRowV3` requires six finite
targets and `_supervised_loss` does not mask regression loss for OOM rows.
Before production OOM training, add a per-target validity mask and regress only
valid targets. Until then, keep collected OOM rows in a versioned auxiliary
corpus rather than fabricating six values.

## 7. Inputs that must accompany every label

### 7.1 Graph and workload identity

- immutable source fingerprint and model parameter fingerprint;
- strict-first exported `GraphIRV3` file and graph SHA-256;
- architecture family, modality, task, source group, and generator version;
- input pytree schema, shapes, strides, dtypes, dynamic constraints, and actual
  sampled values or deterministic value fingerprint;
- parameter count, trainable parameter count, buffer bytes, graph nodes/edges,
  phase counts, and analytical cost/liveness summaries;
- loss function and its configuration;
- dataset ID/revision, subset ID, examples per epoch, and steps per epoch.

The executable profiled callable, input signature, optimizer, and captured
graph MUST match. The current source-first profiler already checks callable
identity, parameter fingerprint, input signature, precision, train/eval mode,
and optimizer class before measurement.

### 7.2 Batch and execution controls

- microbatch size;
- gradient accumulation steps;
- effective batch size;
- total epochs, current epoch, total/current optimizer steps, and steps/epoch;
- `drop_last` and variable-length batching/bucketing policy;
- activation checkpointing policy and checkpoint segments;
- gradient clip norm and loss scale;
- eager versus compiled execution, compiler/backend/version, and compile mode;
- deterministic/benchmark flags, TF32 policy, and matmul precision;
- optimizer `foreach`, `fused`, `capturable`, and differentiable settings.

### 7.3 Optimizer identity and hyperparameters

V3 currently has first-class exact identities for:

```text
SGD, ASGD, Adadelta, Adafactor, Adagrad, Adam, Adamax, AdamW,
LBFGS, Muon, NAdam, RAdam, RMSprop, Rprop, SparseAdam,
LAMB, LARS, Lion
```

Store the raw implementation name in addition to exact/family/hash identity.
For composite optimizers, store every component and parameter assignment. A
Muon workload should normally distinguish the Muon-controlled 2-D hidden
weights from AdamW-controlled embeddings, biases, normalization parameters, and
other tensors.

Required optimizer configuration includes:

- initial and current learning rate;
- all parameter-group learning rates and parameter fractions;
- weight decay per group and coupled/decoupled mode;
- momentum, dampening, Nesterov, betas, epsilon, rho, and alpha;
- AMSGrad, maximize, relative-step, scale-parameter, and warmup-init flags;
- LAMB/LARS trust coefficient;
- clipping threshold and decay rate;
- Muon Newton-Schulz coefficients/steps and LR-adjustment policy;
- optimizer state dtype/bit width and master-weight dtype;
- implementation/backend, fused/foreach status, paging, sharding, and offload.

The last four implementation dimensions are not yet fully represented by the
current feature schema. Preserve them in raw records now and do not mark those
variants deployment-approved until dedicated features and cost/state formulas
exist.

### 7.4 Scheduler identity and progress

Current first-class scheduler identities are:

```text
none, constant, constant_with_warmup, linear, linear_with_warmup,
step, multi_step, exponential, cosine, cosine_warm_restarts,
cosine_with_warmup, polynomial, one_cycle, cyclic,
reduce_on_plateau, inverse_sqrt, warmup_stable_decay
```

Record:

- scheduler raw/exact/family/hash identity and chained components;
- warmup steps, epochs, and ratio;
- decay rate, step size, milestones, minimum/maximum LR;
- cycles/restarts, polynomial power, patience, threshold, and cooldown;
- current epoch/step and current LR;
- metric history/state for `reduce_on_plateau`;
- whether `scheduler.step()` occurs per microbatch, optimizer step, or epoch.

Sample schedule progress at meaningful points rather than generating duplicate
rows for every step: initialization, warmup midpoint/end, 25%, 50%, 75%, 90%,
and final 5% of training. For performance labels, several points may have
similar cost; retaining progress lets evidence determine whether the field is
predictive instead of assuming it is.

### 7.5 Per-operation dtype policy

Do not rely on a single declared graph precision. Preserve actual node and edge
dtypes and explicit casts. Required policies include:

- homogeneous FP32;
- BF16 autocast with FP32 reductions/state;
- FP16 autocast with gradient scaling and FP32 reductions/state;
- FP32 first/last layers with BF16 or FP16 internal layers;
- FP32 normalization/reduction islands;
- parameter, gradient, accumulation, master-weight, and optimizer-state dtype;
- alternating/cast-heavy diagnostic graphs;
- TF32 permission as an execution flag even though tensor storage is FP32.

The current encoder detects FP32-to-BF16 tensor transitions and emits a `mixed`
precision category. The current bootstrap `WorkloadDescriptor`, however, only
enumerates homogeneous `float32`, `float16`, and `bfloat16`; production workload
generation must add structured mixed-dtype policies.

## 8. Workload coverage matrix

A full Cartesian product is infeasible and would overrepresent artificial
combinations. Use a coverage-driven, constrained design with logarithmic grids,
pairwise interaction coverage, Latin-hypercube sampling for continuous values,
and adaptive sampling near OOM/performance boundaries.

### 8.1 Data layers

| Layer | Purpose | Use in training |
| --- | --- | --- |
| Exact-operation microbenchmarks | Registry cost/state calibration and kernel coverage | Encoder pretraining; not sufficient alone for six-target training |
| Composite kernels/blocks | Interactions, fusion, liveness, dtype transitions | Encoder pretraining and auxiliary teacher rows |
| Representative real models | Scheduler-grade distribution | Main teacher/student supervised dataset |
| Generated architectures | Broader topology/shape coverage | Main data after correctness and source-quality gates |
| Boundary/OOM search | Memory frontier and failure stages | OOM/uncertainty heads; successful near-boundary rows also train regression |
| Custom/OOV suite | Future/custom operation behavior | Held-out confidence and fallback evaluation |

### 8.2 Architecture and modality families

The target dataset SHOULD include, subject to intended deployment:

- CNNs: residual, depthwise, dense-connectivity, detection/segmentation blocks;
- encoder transformers and vision transformers;
- decoder-only and encoder-decoder transformers, including KV/sequence regimes;
- recurrent/sequence models: RNN, LSTM, GRU, temporal convolution;
- graph neural networks and sparse/scatter-heavy models;
- diffusion/UNet and attention-heavy image models;
- audio/speech convolutional, recurrent, and transformer models;
- recommendation/embedding-heavy and tabular models;
- mixture-of-experts/routing diagnostics;
- custom/fused and generated-code workloads.

For every important family, span parameter count, activation footprint,
arithmetic intensity, graph depth, branching, residual lifetime, and dynamic
shape behavior rather than selecting models only by parameter count.

### 8.3 Shape and size regimes

Use at least five regimes per applicable axis:

```text
tiny / small / medium / large / boundary
```

Axes include batch, sequence length, hidden size, attention heads, vocabulary,
image resolution, channels, audio duration, graph nodes/edges, embedding table
size, and expert count. Include non-power-of-two and tensor-core-misaligned
shapes. Boundary values should be discovered adaptively for each configuration,
not hard-coded globally.

### 8.4 Batch and memory frontier design

For each representative base workload:

1. Probe a logarithmic microbatch ladder, normally powers of two plus one or two
   irregular values.
2. Bracket the largest successful batch/shape.
3. Binary-search or staircase-search the boundary.
4. Collect successful rows around approximately 50%, 75%, 90%, 95%, and 99%
   of the observed memory frontier.
5. Preserve at least one true OOM beyond the boundary when safe.
6. Cross selected microbatches with accumulation values such as 1, 2, 4, and 8
   while controlling effective batch size.

Near-OOM observations must be intentionally oversampled; random workload grids
usually contain too few positive OOM examples to calibrate the OOM head.

## 9. Optimizer sampling plan

The bootstrap workload generator currently instantiates only SGD, Adam, and
AdamW even though the encoder recognizes 18 optimizers. Production collection
must close this gap.

Use deployment telemetry when available. Until then, a reasonable starting
allocation of successful real-model configurations is:

| Stratum | Approximate share | Contents |
| --- | ---: | --- |
| Core | 55–65% | AdamW, Adam, SGD, Muon+AdamW |
| Other first-class | 25–35% | ASGD, Adadelta, Adafactor, Adagrad, Adamax, LBFGS, NAdam, RAdam, RMSprop, Rprop, SparseAdam, LAMB, LARS, Lion |
| Custom/implementation diagnostics | 5–10% | OOV names and variants below; held out from deployment approval until modeled |

Every first-class optimizer needs coverage across more than one architecture
family, parameter scale, batch regime, and precision. Rare/specialized
optimizers should use compatible workloads—for example sparse gradients for
SparseAdam and suitable 2-D parameter groups for Muon—rather than meaningless
Cartesian combinations.

### 9.1 First-class gaps to retain in raw records

Prioritize future schema/collector work for:

1. 8-bit/paged states: Adam8bit, AdamW8bit, PagedAdamW, PagedLion, and related
   bitsandbytes variants;
2. fused/vendor kernels: Apex FusedAdam, FusedLAMB,
   FusedMixedPrecisionLAMB, FusedNovoGrad, and FusedSGD;
3. state sharding/offload: ZeRO, CPU/NVMe offload, and rank-local state;
4. preconditioners: Distributed Shampoo, SOAP, KL-Shampoo, K-FAC, and PSGD;
5. low-memory projection/update: GaLore, Q-GaLore, LOMO, and AdaLOMO;
6. schedule-free AdamW/SGD/RAdam;
7. AdEMAMix, Adan, SophiaG, NovoGrad, Prodigy/D-Adaptation,
   Lookahead/Ranger, and SAM/ASAM.

Unknown names are structurally encoded as `other` plus family/hash, but that is
not equivalent to having correct state-memory and optimizer-step cost models.

## 10. Scheduler and hyperparameter experimental design

### 10.1 Core scheduler coverage

Prioritize no schedule/constant, linear warmup-decay, cosine with warmup,
one-cycle, step/multi-step, inverse-square-root, and reduce-on-plateau. Cover all
other first-class schedulers with smaller diagnostic strata.

### 10.2 Learning-rate grid

For each optimizer/model family, define a safe reference LR and sample a
log-scale relative grid, for example:

```text
0.1x, 0.3x, 1x, 3x, and—only after a stability probe—10x reference LR
```

Do not use one absolute LR grid for all optimizers. Record stability failures
separately from CUDA OOM. Include parameter groups with different LRs and weight
decays, especially bias/norm exclusions and Muon+AdamW components.

### 10.3 Other common controls

Cover meaningful ranges for:

- weight decay, momentum, betas, epsilon, trust coefficient, and clipping;
- warmup ratio and decay horizon;
- gradient accumulation and effective batch size;
- activation checkpointing;
- fused and foreach implementations;
- gradient scaler enabled/disabled and representative scale states;
- current epoch/step and schedule progress.

Use pairwise/three-way interaction coverage for the most consequential
interactions:

```text
optimizer × state precision × model scale
precision × dtype policy × tensor shape
batch × accumulation × checkpointing
optimizer × scheduler × LR range
compile mode × dynamic shape × model family
```

## 11. Production measurement protocol on the target GPU

### 11.1 Isolation and setup

Each raw run MUST:

1. verify the exact target hardware ID and physical UUID;
2. reject active unrelated GPU processes;
3. record idle framebuffer use and free memory before allocation;
4. use a pinned container digest and immutable source/workload fingerprints;
5. record clocks, power, temperature, throttle reasons, driver, and libraries;
6. seed Python, NumPy, PyTorch CPU, and all CUDA RNGs;
7. restore the model/optimizer starting state for each repetition;
8. validate eager/exported output equivalence before profiling;
9. execute warmup outside the measured window;
10. synchronize CUDA or use CUDA events at timing boundaries.

CUDA work is asynchronous; unsynchronized host timers are not valid GPU timing.
The official PyTorch CUDA guidance recommends synchronization or CUDA events for
precise elapsed time. PyTorch also warns that exact reproducibility is not
guaranteed across releases/platforms, which is why environment revisions and
raw repetitions are mandatory.

### 11.2 Warmup and sustained window

- Use at least 10–20 optimizer warmup steps for normal eager workloads.
- Compiled workloads must complete compilation and reach stable generated code
  before measurement.
- Continue warmup until recent step times stabilize, subject to a maximum and a
  recorded `warmup_not_stable` failure.
- Measure at least 50 optimizer steps **and** a sustained window long enough to
  contain many real NVML samples; 30–60 seconds is a suitable production
  default for utilization labels.
- For very slow workloads, require at least 10 measured optimizer steps and
  report the shorter statistical window.

Polling NVML every 20 ms does not create 20 ms sensor resolution. NVIDIA states
that ordinary device utilization samples may cover roughly 1/6 second to one
second depending on the product. Short microbenchmarks can measure CUDA time but
cannot produce reliable average/p95 utilization labels by repeating cached
NVML readings.

### 11.3 Memory measurement

At the post-warmup boundary:

```text
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
capture idle/device baseline
start raw NVML sampling
```

After the sustained window, record both allocator peaks and device memory.
PyTorch defines `max_memory_allocated` as peak bytes occupied by tensors and
`max_memory_reserved` as peak bytes managed by the caching allocator. Preserve
both; the canonical output uses reserved memory because scheduler capacity must
account for the allocator's retained pool.

### 11.4 Repetitions and run order

- Pilot collection: at least 3 fresh-process repetitions per configuration.
- Production training rows: normally 5 accepted repetitions.
- Golden, unstable, near-OOM, or high-variance rows: 7–10 repetitions.
- Randomize workload order within safe size bands.
- Interleave stable sentinel workloads to detect temporal drift.
- Use cooldown or temperature-aware scheduling rather than running every large
  workload consecutively.

Do not silently delete outliers. Mark contamination, throttling, correctness
failure, clock drift, or allocator residue and rerun. Retain rejected raw runs
with their reasons.

## 12. Aggregation and quality policy

For each workload configuration:

- aggregate successful fresh-process repetitions using the median for timing
  and memory peaks;
- derive utilization from the union of valid time-stamped samples, weighting
  by sample duration where available;
- retain median absolute deviation, coefficient of variation, min/max, and
  confidence intervals;
- require identical graph, source, input, optimizer, scheduler, target GPU,
  environment revision, and workload hashes;
- never average successful and OOM runs into one regression target;
- if identical configurations alternate between success and OOM, mark the
  aggregate as a boundary/unstable sample and retain the success probability.

Suggested initial quality thresholds, to be calibrated with pilot data:

| Check | Initial rule |
| --- | --- |
| Correctness | All accepted runs pass eager/export equivalence and finite-loss checks |
| Timing variability | Fresh-process step-time CV ≤ 3%; otherwise add repeats or quarantine |
| Peak reserved variability | Range ≤ 2% of median; otherwise investigate allocator/process state |
| Target GPU | Exact canonical ID in graph, run, aggregate, and manifest |
| Environment | One immutable environment revision per aggregate |
| Throttling | No thermal/power/reliability throttle unless explicitly modeled |
| Utilization samples | Sustained window with enough distinct sensor timestamps |
| Capture | Strict preferred; validated non-strict tracked separately |
| Encoding | No tensor-producing operation silently dropped |

The repository's current training gates remain mandatory:

- strict capture rate at least 95%;
- complete encoding rate at least 99%;
- measured unknown-operation GPU-time fraction at most 2%;
- source-group isolation;
- frozen dataset, split, feature-schema, registry, normalization, and
  coarsening hashes.

## 13. Split and leakage policy

Use the current deterministic 80%/10%/10% train/validation/test allocation by
whole source group, stratified by data layer.

Rules:

1. Aggregate repetitions before assigning partitions.
2. Keep every variant from one source/model family in one partition.
3. Keep identical or equivalent graph signatures in one partition.
4. Keep generated mutations of the same source template together unless the
   generator can prove independent architecture provenance.
5. Fit continuous normalization on training only.
6. Fit linear, uncertainty, and OOM calibration on validation only.
7. Freeze test before capacity/model selection.
8. Never move failed test rows into training after seeing errors.

Maintain the eight explicit evaluation suites already defined by v3:

```text
in_distribution_validation
architecture_source_family_held_out
operation_combination_held_out
generated_code_robustness
dynamic_shape_extrapolation
precision_optimizer_held_out
custom_oov_suite
v2_compatible_matched_test
```

Add dataset-specific audits for:

- physical-device-UUID held out within the target SKU;
- memory-frontier/near-OOM rows;
- LR/scheduler progress;
- mixed-layer dtype policies;
- fused versus foreach implementation;
- rare optimizer families;
- driver/framework overlap revisions.

## 14. Balancing and weighting

The natural collection distribution will be dominated by small successful
AdamW/FP32 or AdamW/BF16 workloads. Avoid allowing this to erase rare but
important regimes.

- Select configurations with stratified quotas before measurement.
- Use log buckets for parameter count, FLOPs, activation bytes, graph size,
  batch, sequence/resolution, and label magnitude.
- Oversample near-OOM, mixed dtype, dynamic shape, custom/OOV, and long-tail
  optimizer strata.
- Use `domain_weight` only after reporting unweighted counts and metrics.
- Cap weights so a small stratum cannot dominate gradients.
- Do not duplicate aggregated rows to balance; sample or weight them in the
  loader while preserving one canonical row.
- Report macro metrics across families and regimes in addition to global
  micro-averages.

## 15. Proposed raw measurement record

The following is a design record, not yet a checked-in parser contract:

```json
{
  "record_version": "perfseer_v3_target_gpu_measurement_v1",
  "run_id": "h100sxm80-workload123-rep03",
  "workload_config_id": "sha256-of-complete-workload-config",
  "repetition": 3,
  "target_hardware": {
    "hardware_id": "nvidia_h100_sxm_80gb",
    "device_uuid": "GPU-...",
    "product_name": "NVIDIA H100 80GB HBM3",
    "total_memory_bytes": 85899345920,
    "compute_capability": "9.0",
    "sm_count": 120,
    "mig_mode": "disabled"
  },
  "environment": {
    "container_digest": "sha256:...",
    "source_revision": "...",
    "driver_version": "...",
    "cuda_version": "...",
    "pytorch_version": "...",
    "compiler": {"mode": "eager", "backend": null}
  },
  "identity": {
    "source_group": "decoder_transformer_family_17",
    "source_fingerprint": "...",
    "model_fingerprint": "...",
    "graph_path": "graphs/....json",
    "graph_sha256": "...",
    "input_value_fingerprint": "..."
  },
  "workload": {
    "dataset_id": "...",
    "dataset_revision": "...",
    "examples_per_epoch": 100000,
    "micro_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "optimizer_steps_per_epoch": 3125,
    "activation_checkpointing": true,
    "precision_policy": "mixed_bf16_fp32_norm",
    "optimizer": {
      "name": "muon",
      "components": ["muon", "adamw"],
      "parameter_groups": [
        {"component": "muon", "lr": 0.02, "parameter_fraction": 0.8},
        {"component": "adamw", "lr": 0.0003, "parameter_fraction": 0.2}
      ]
    },
    "scheduler": {
      "name": "cosine_with_warmup",
      "warmup_steps": 1000,
      "total_steps": 100000,
      "current_step": 25000,
      "current_lr": 0.015
    }
  },
  "measurement": {
    "status": "ok",
    "warmup_steps": 20,
    "measured_steps": 100,
    "measured_window_seconds": 42.7,
    "raw_step_ms": [13.5, 13.6],
    "raw_nvml_samples": [],
    "peak_torch_allocated_bytes": 0,
    "peak_torch_reserved_bytes": 0
  },
  "quality": {
    "correctness_validated": true,
    "exclusive_gpu": true,
    "steady_state": true,
    "throttling_detected": false,
    "accepted": true,
    "rejection_reasons": []
  }
}
```

The actual record must preserve the complete raw arrays; the shortened example
does not prescribe truncation.

## 16. Proposed aggregated sample record

```json
{
  "sample_version": "perfseer_v3_target_gpu_sample_v1",
  "sample_id": "h100sxm80-workload123",
  "workload_config_id": "...",
  "target_hardware_id": "nvidia_h100_sxm_80gb",
  "graph_path": "graphs/....json",
  "graph_sha256": "...",
  "source_group": "decoder_transformer_family_17",
  "accepted_run_ids": ["...rep01", "...rep02", "...rep03"],
  "rejected_run_ids": [],
  "targets": {
    "train_epoch_ms": 42500.0,
    "train_avg_sm_util_percent": 91.0,
    "train_p95_sm_util_percent": 98.0,
    "train_peak_vram_used_mib": 62300.0,
    "train_peak_torch_reserved_mib": 61800.0,
    "train_peak_memory_controller_util_percent": 87.0
  },
  "target_validity": [true, true, true, true, true, true],
  "oom": false,
  "oom_stage": "none",
  "dispersion": {
    "step_time_cv": 0.012,
    "peak_reserved_range_fraction": 0.006
  },
  "aggregation": "median_across_fresh_process_runs",
  "quality_status": "accepted"
}
```

Values above are illustrative placeholders, not measurements.

## 17. Repository storage layout

Recommended per-target dataset layout:

```text
datasets/perfseer_v3/<target_hardware_id>/<dataset_revision>/
  metadata/
    dataset.json
    target_hardware.json
    environment_revisions.jsonl
    collection_protocol.md
  sources/
    source_manifest.jsonl
  graphs/
    <graph_sha256>.json
  raw_runs/
    shard-00000.jsonl.zst
    shard-00001.jsonl.zst
  aggregates/
    samples.jsonl
    oom_samples.jsonl
    rejected_samples.jsonl
  splits/
    grouped_split.json
    evaluation_slices.json
  manifests/
    training_manifest_v2.json
  audits/
    coverage.json
    repeatability.json
    leakage.json
    label_distribution.json
    golden_epoch_validation.json
```

Raw telemetry can be compressed and sharded, but graph files, aggregate rows,
split declarations, and manifests must remain content-addressed and easy to
audit. Do not overwrite a dataset revision after its fingerprint is used in a
checkpoint.

## 18. Mapping into the current training manifest

For successful aggregate rows, create one manifest sample:

```json
{
  "sample_id": "h100sxm80-workload123",
  "graph_path": "../graphs/<graph_sha256>.json",
  "split": "train",
  "source_group": "decoder_transformer_family_17",
  "graph_signature": "<graph_sha256>",
  "hardware_id": "nvidia_h100_sxm_80gb",
  "target": [42500.0, 91.0, 98.0, 62300.0, 61800.0, 87.0],
  "oom": 0.0,
  "oom_stage": "none",
  "peak_live_bytes": 64172851200,
  "domain_weight": 1.0
}
```

The manifest's deployment section MUST contain exactly the same target GPU in
`target_hardware_id` and its one-element `hardware_allowlist`. Optimizer,
scheduler, precision, capture-quality, and training-mode allowlists should be
generated from accepted training coverage, not manually broadened after model
training.

Before manifest creation:

1. validate every graph and aggregate hash;
2. compute grouped splits;
3. calculate measured capture/encoding/unknown-time gates;
4. freeze the operation registry selected from target-GPU time coverage;
5. regenerate the feature schema;
6. calculate dataset and split fingerprints;
7. fit normalization on train only during materialization.

## 19. Collection scale and capacity planning

These are planning ranges, not accuracy guarantees.

| Stage | Unique configurations per target GPU | Fresh-process repeats | Purpose |
| --- | ---: | ---: | --- |
| Protocol pilot | 200–500 | 3–5 | Debug correctness, stability, telemetry, and aggregation |
| Coverage pilot | 1,500–3,000 | ≥3 | Exercise all major families/axes and estimate learning curves |
| Initial production | 10,000–20,000 | normally 5 | Train/validate first serious teacher and students |
| Adaptive expansion | +2,000–10,000 per round | 3–7 | Fill high-error, OOD, rare, and boundary regions |

Estimate label-GPU time before launching:

```text
GPU-hours = configurations × attempted_repetitions
            × mean(warmup + measured + setup seconds) / 3600
```

For example, 10,000 configurations × 5 repetitions × 45 seconds is about 625
label-GPU hours before reruns. Parallel collection may use several physical
cards only when they share the exact qualified target ID and pass overlap
repeatability checks.

Use learning curves after each collection stage. Stop adding random rows when
held-out errors plateau; redirect collection toward the worst family,
optimizer, dtype, graph-size, and OOM slices.

## 20. Staged execution plan

### Stage A: freeze the protocol

- qualify target GPU(s) and environment;
- implement immutable raw/aggregate schemas;
- add sustained target-GPU training measurement;
- validate timing, allocator, NVML, correctness, and contamination checks;
- compare derived epoch time with full measured epochs.

Exit: sentinel workloads meet repeatability thresholds and independent
aggregation reproduces identical sample rows.

### Stage B: close collector/schema blockers

- add mixed-dtype workload policies;
- instantiate all first-class optimizers and composite Muon+AdamW;
- capture full scheduler/training progress;
- distinguish fused/foreach/state precision/paging/offload;
- add OOM target masks and phase instrumentation;
- add environment and physical-device provenance.

Exit: every requested input is either encoded or explicitly retained as
provenance with a documented constant scope.

### Stage C: coverage pilot

- collect microbenchmark, composite, real, generated, boundary, and OOV layers;
- run operation-time coverage and unknown-time analysis;
- produce grouped splits and all available challenge suites;
- train a small pilot model and generate slice errors/uncertainty.

Exit: capture ≥95%, complete encoding ≥99%, measured unknown GPU time ≤2%, no
source leakage, and actionable learning curves.

### Stage D: production collection

- expand representative real workloads and target deployment distributions;
- collect five fresh-process repeats by default;
- fill optimizer, scheduler, mixed-dtype, dynamic-shape, and near-OOM quotas;
- freeze immutable dataset/registry/schema/split revisions.

Exit: dataset audit passes and no required evaluation slice is unintentionally
empty.

### Stage E: teacher and student training

- train T0/T1/T2 on any suitable predictor-training device(s);
- select teacher capacity using family-held-out accuracy/calibration;
- distill S0–S3 on any suitable execution GPU(s), possibly different from the
  teacher-training GPU and target label GPU;
- preserve target hardware, dataset, split, normalization, schema, registry,
  and coarsening identities across teacher/student artifacts.

Exit: prediction, calibration, OOM, artifact, latency, and all required
ablation/acceptance gates pass.

### Stage F: deployment

- run the student on CPU or a chosen deployment accelerator;
- select the artifact by the workload's requested target GPU;
- return fallback for target, schema, precision, optimizer, scheduler,
  confidence, or coverage mismatches;
- shadow and canary before scheduler authority.

## 21. Dataset readiness checklist

### Target and provenance

- [ ] Exactly one canonical target hardware ID per dataset/manifest.
- [ ] Every label was measured on a qualified physical device of that model.
- [ ] Device UUID, driver, libraries, container, clocks, power, and temperature recorded.
- [ ] Predictor training/distillation execution device kept separate from target identity.

### Workload inputs

- [ ] Source, model, graph, input, optimizer, scheduler, and configuration hashes frozen.
- [ ] Representative families, scales, shapes, batches, and dynamic regimes covered.
- [ ] Per-operation mixed dtype policies present.
- [ ] All first-class optimizers have executable, semantically valid workloads.
- [ ] Core schedulers and meaningful progress points covered.
- [ ] LR, decay, warmup, parameter groups, accumulation, clipping, and checkpointing recorded.

### Measurements

- [ ] CUDA timing synchronized or event-based.
- [ ] Warmup/compile/autotune excluded and steady state demonstrated.
- [ ] Utilization window long enough for real NVML sampling resolution.
- [ ] PyTorch allocated/reserved and device framebuffer memory retained separately.
- [ ] Fresh-process repetitions, dispersion, rejection reasons, and raw samples retained.
- [ ] Full-epoch golden rows validate step extrapolation.
- [ ] OOM phases and target-validity masks implemented before OOM training.

### Splits and quality

- [ ] Repetitions aggregated before split.
- [ ] Source groups and equivalent graph signatures isolated.
- [ ] Normalization fit on train only; calibration fit on validation only.
- [ ] Frozen test and challenge suites remain untouched by model selection.
- [ ] Capture, encoding, unknown-time, repeatability, and contamination gates pass.
- [ ] Label and feature distributions audited for imbalance and missing regimes.

### Training and release

- [ ] Teacher/student target hardware IDs match the label dataset.
- [ ] Teacher/student dataset, split, normalization, schema, registry, and coarsening hashes match.
- [ ] Training reports identify predictor execution devices separately.
- [ ] Accuracy, uncertainty, OOM, slice, ablation, latency, and artifact gates pass.
- [ ] Runtime allowlists are derived from validated coverage and fail closed.

## 22. Current implementation gaps before production collection

The v3 encoder/model supports the intended semantics, but the existing local
diagnostic corpus is not yet the dataset specified here. Close these concrete
gaps first:

1. `WorkloadDescriptor` only allows homogeneous FP32/FP16/BF16 and needs
   structured mixed-dtype policies.
2. The default training microbenchmark factory only constructs SGD, Adam, and
   AdamW; it needs all first-class and composite optimizer factories.
3. OOM manifest/training needs nullable target masks so regression is not
   trained on fabricated OOM values.
4. Profiling failure stages currently collapse most OOMs to warmup/measurement;
   instrument capture/forward/loss/backward/optimizer/allocator explicitly.
5. Optimizer state precision, paging, fused backend, sharding, and offload need
   dedicated features and state/cost formulas.
6. TF32 and detailed AMP/master-weight policies need explicit execution fields.
7. Production NVML windows need sustained duration; 10 fast microbenchmark
   steps are insufficient for utilization labels.
8. Real dataset adapters should replay actual decoded values/batches rather
   than only deterministic shape-compatible tensors.
9. The approved operation registry must be rebuilt from measured target-GPU
   production workload time; the checked-in registry remains intentionally
   unapproved.
10. Generated evaluation slices currently lack some production-only suites and
    must remain fail-closed until populated.

## 23. Primary references

- Current v3 target order and training contract:
  `src/perfseer_v3/training.py`
- Manifest, split fingerprint, materialization, and execution-device selection:
  `src/perfseer_v3/training_runner.py`
- Training graph capture and optimizer/scheduler inputs:
  `src/perfseer_v3/capture_training.py`
- Exact feature layout and hyperparameters:
  `src/perfseer_v3/schema.py` and `src/perfseer_v3/training_semantics.py`
- Source-first profiling and raw record contract:
  `src/perfseer_v3/profiling.py`
- Workload descriptors and current bootstrap limits:
  `src/perfseer_v3/workloads.py`
- Leakage-safe splitting and required evaluation suites:
  `src/perfseer_v3/splits.py`
- Acceptance gates and slice metrics:
  `src/perfseer_v3/evaluation.py`
- Example current manifest:
  `src/perfseer_v3/training_manifest.example.json`
- [PyTorch optimizer catalog](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch CUDA timing and memory semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch peak allocated-memory API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)
- [NVIDIA NVML utilization definitions and sampling period](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)
- [bitsandbytes 8-bit optimizer catalog](https://huggingface.co/docs/bitsandbytes/v0.43.0/optimizers)
- [NVIDIA Apex fused optimizers](https://github.com/NVIDIA/apex/tree/master/apex/optimizers)
- [Meta Distributed Shampoo](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)
- [GaLore official implementation](https://github.com/jiaweizzhao/GaLore)
- [Schedule-Free official implementation](https://github.com/facebookresearch/schedule_free)
- [Apple AdEMAMix implementation](https://github.com/apple/ml-ademamix)

## 24. Final rule

For each target GPU model:

```text
collect labels on that GPU model
-> build one immutable target-GPU dataset
-> train one target-specific teacher on any suitable execution device
-> distill one target-specific student on any suitable execution device
-> deploy the student on CPU or another suitable device
-> use it only to predict workloads requested for its target GPU model
```

The target GPU determines the meaning of the labels and predictions. It does
not determine where PerfSeer itself must be trained or executed.
