# PerfSeer v3 final local implementation status

Date: 2026-07-27

Implementation root: `src/perfseer_v3`

## Outcome

The encoder/model goal is implemented and locally verified as an isolated v3
path. V2 source, model, artifact, and scheduler contracts were not migrated or
silently modified. The final v3 feature hash is
`7d7966b124db6d473db2391bffa692cad2c74858985ef2cd2f478b8fbe210e78`;
the operation-registry semantic hash is
`e0cca4384b3a151cc0074e853de430a6493af02b3e669ccba2f8fe76a1f04d77`.

The locally executable implementation is complete. Production model training
and acceptance are deliberately blocked: this checkout has no grouped
production scheduler-label corpus, approved measured-GPU-time registry,
family-held-out prediction records, production OOM cases, completed
ablations, matched-v2 latency/size runner, or authenticated Nautilus access.
The registry remains `training_approved: false`; the production runner fails
before allocating a teacher. No production accuracy claim is made.

The proposed production data contract and collection plan are documented in
`docs/perfseer_v3_dataset_design_report.md`; that report is a design
specification, not evidence that the dataset has already been collected.

## Implemented changes

### GPU-specific and training-configuration correction

- Each production teacher/student pair now predicts exactly one canonical
  target GPU. Multi-GPU label manifests, generic target IDs, distillation
  between artifacts for different target GPUs, and prediction requests for a
  different target GPU fail closed. Predictor training/distillation execution
  is independently allowed on any suitable CPU/GPU.
- The versioned training manifest v2 requires the GPU on every row and requires
  a one-item matching hardware allowlist. Artifact/state-dict v2 metadata and
  artifact-registry v2 records preserve the same GPU identity and scheduler
  policy.
- Optimizers and schedulers now use exact/family/hash encodings. The optimizer
  set includes all standard PyTorch optimizers, Muon, LAMB, LARS, and Lion;
  composite Muon + AdamW configurations retain both components, and custom
  future names remain distinguishable through stable hashes.
- Global inputs now include epoch/step progress; initial, current, minimum,
  maximum, and parameter-group learning rates; warmup and decay; weight decay;
  momentum/betas/epsilon; and common optimizer/scheduler-specific controls.
- Mixed layer dtypes are detected from tensor edges. FP32-to-BF16 captures
  retain both dtypes, a `mixed` graph policy, and explicit conversion nodes.
  Runtime precision allowlists validate the actual captured policy.

### Graph and feature semantics

- Expanded the typed node contract to independent exact, family, canonical
  hash, raw-overload hash, phase, input/output/accumulation dtype, backend,
  feature quality, layout, rank, flags, cost, topology, shape, liveness,
  confidence, and operation-argument inputs.
- Expanded typed tensor edges to retain source/destination slots, role, dtype,
  layout, rank, alias/view/materialization, dynamic quality, phase transition,
  shape/stride/lifetime/fanout summaries, and explicit flags.
- Preserved external model input/output, parameter, buffer, and gradient
  semantics as typed self-loops on attached operations instead of dropping
  them.
- Expanded global inputs to hardware, precision, optimizer, capture
  mode/backend, dynamic policy, training mode, batch, accumulation, clipping,
  loss scaling, activation checkpointing, foreach/fused optimizer modes,
  per-phase totals, model-input statistics, and quality fractions.
- The generated ordered widths are 40 node continuous, 21 node flags, 14 edge
  continuous, 5 edge flags, 105 global continuous, and 8 quality values.

### Encoder and model

- Replaced flat categorical treatment with separate hierarchical node, edge,
  and global embeddings plus continuous MLPs.
- Added additive and concatenation-plus-projection identity-fusion controls.
- Retained the existing message-passing trunk as the baseline and added a
  phase-aware pooling experiment over forward/loss/backward/optimizer nodes.
- Preserved the six scheduler predictions and added a versioned optional
  contract for log variance, OOM probability, OOM failure stage, confidence,
  and peak-live bytes. Graph and phase embeddings support representation
  distillation.
- Unknown/custom-only graphs remain finite, hashed, visible to coverage, and
  lower confidence instead of crashing or producing an unqualified result.

### Training, export, and evaluation

- Added a real CUDA/AMP training runner with manifest validation, source-group
  and graph-signature leakage checks, train-only normalization, encoder
  pretraining, supervised teacher training, hard-label plus representation
  distillation, validation calibration, and integrity-checked artifacts.
- Added explicit AMSGrad state handling and foreach workspace/fused semantics.
- Added verified student CPU export with eager/reloaded TorchScript equality
  and a hash/schema/output-contract sidecar.
- Added an executable scheduler-wrapper check and a JSON/JSONL prediction
  evaluator for six-target metrics, twelve slice dimensions, uncertainty, OOM,
  failure stage, and the exact ten required ablations.
- Evaluation has separate teacher/student matched-v2 5% non-regression gates
  and 15%/10% teacher/student new-operation improvement gates.

### Main files

- Core: `schema.py`, `features.py`, `model.py`, `capture_training.py`,
  `training.py`, `training_runner.py`, `artifact.py`, `deployment_export.py`,
  `runtime.py`, `evaluation.py`, and `capacity.py`.
- Configs: teacher/student YAML and
  `configs/capacity_sweep/capacity_candidates.yaml`.
- Commands: capacity benchmark, training runner, prediction evaluator,
  student exporter, scheduler integration, encoder/model auditor, CUDA
  verifier, schema/coverage/profiling/workload tools.
- Tests: 12 v3 test modules covering registry/IR, capture, cost/liveness,
  features/coarsening, model, training, capacity, profiling, workloads,
  evaluation, artifacts/runtime, and baseline coverage.
- Documentation/evidence:
  `docs/perfseer_v3_encoder_model_design.md`, this summary, compatibility and
  requirement-audit JSON/Markdown, coverage, profiling, capacity, workload,
  cost, evaluation-slice, and CUDA reports.

## Measured local evidence

### Coverage and compatibility

- Supported corpus: 18/18 strict captures, 638/638 tensor operation nodes
  structurally encoded, and 100% complete graphs.
- Frontier corpus: 18 strict captures plus one validated non-strict audio
  capture, 650/650 nodes encoded, 94.74% strict capture, and 100% complete
  encoding.
- Model-corpus matrix: 18/18 structurally encodable, 0/18 exact-only, and 0/18
  accuracy-validated. Exact-only is intentionally selective; all nodes still
  have family/hash/custom structure.
- Current local supported profiles contain 126 dispatcher identities and a
  57-operation diagnostic vocabulary covering at least 95% of measured time.
  All 126 resolve to a known registry family, so local unknown measured time is
  0%. This does not approve the production vocabulary.
- Cost/decomposition audit: four captures, no capture failure, no unsupported
  formula rule, no known audited node missing a FLOP formula, and no semantic
  preservation violation.

### Capacity

| Candidate | Hidden × blocks | Trainable parameters | TorchScript bytes | CPU p95 small / median / large (ms) |
|---|---:|---:|---:|---:|
| T0 | 1024 × 8 | 136,980,818 | not measured | not measured |
| T1 | 1280 × 10 | 259,825,562 | not measured | not measured |
| T2 | 1536 × 10 | 374,031,306 | not measured | not measured |
| S0 | 192 × 2 | 1,805,014 | 7,346,687 | 1.289 / 2.446 / 4.348 |
| S1 | 224 × 2 | 2,439,890 | 9,895,125 | 1.578 / 2.729 / 5.805 |
| S2 | 256 × 2 | 3,169,594 | 12,815,795 | 1.853 / 3.402 / 6.676 |
| S3 | 256 × 3 | 4,091,706 | 16,511,205 | 2.435 / 4.244 / 9.278 |

These are exact parameter counts and actual one-thread TorchScript probe
measurements. Teacher training memory/time/accuracy/calibration and all
matched-v2 ratios remain unavailable. T2 is 2.73× T0 and passes the
approximately 3× guardrail.

### CUDA integration

The final local verifier passed on NVIDIA GeForce RTX 5090 with PyTorch
2.11.0+cu130/CUDA 13.0. It produced strict capture with 18 nodes and 31 tensor
edges, all four training phases through `aot_autograd_joint`, finite predictor
backward gradients, `[1, 6]` eager and scripted prediction shapes, `[1, 7]`
OOM-stage logits, `[1, 4, 32]` phase embeddings, eager/TorchScript equality,
and two successful source-first profile samples. Peak allocated CUDA memory was
19,066,880 bytes.

### Verification totals

- Focused v3 suite: 95 passed.
- Complete repository `tests/` suite: 103 passed.
- Legacy source-converter plus calibration-pack suites: 77 passed.
- Full declared workload matrix retained prior final evidence of 1,828/1,828
  successful local CUDA executions; its deterministic manifest and split are
  unchanged by the encoder refactor.
- Current profiling rerun: 75/75 microbenchmarks and 42/42 supported-corpus
  source-first profiles completed.
- Bytecode compilation and `git diff --check` passed.
- Wheel build, package-data inspection, isolated no-dependency install, and
  installed schema/registry/training/export imports passed.

## Current evidence hashes

- Supported coverage report:
  `bcc207f6c9a7cb4ac4c88631e914f59ae02245b57b31caf5e33b1b09d17b49e9`
- Frontier coverage report:
  `ddff39c1bd4474a28b6ebb75d846a3fd995e6a0af8670e287bf5f899f3792d3b`
- Microbenchmark measurement:
  `7d5364f1814434a4dcf6e7972d96460c7461040ed42d90f69386df2baac370ad`
- Supported-corpus measurement:
  `d1d972b5451e2b815ffd6bbc5f9bbbebe365cfc5eaa2edc1b514002fa579badd`
- Workload/split semantic hashes:
  `2f94e28fc5497742f9641568d0fddaeac0c4f35184f18bd6556074fd6294ae27` /
  `7328f63b57ae068b5822508259c2065c215c2dd1da7ba117a8721732fb6dece9`
- Full workload execution:
  `469dc6a142f36bd6eab795c7b8c980944bedd902992f2584152289876b8cd1c4`
- Evaluation slices:
  `d00552b7302b282a02fac40c7e63beca94c5a121e2b0c7d45698ad7d94017cd4`
- Cost/decomposition audit:
  `6d4900c808080e81ced79e897a975b85aea87bdb05f35fd232aa40facdc305c6`
- Capacity study:
  `9c78680521540634f8832fe5d7b4bd0f8d32e1c28f48c417acc74c6294b5f7a4`
- Compatibility matrix:
  `720e7ec3e51efbcea730f3f34570397fe50dcc65775b3cff25cdb43d7cbd9d99`
- Encoder/model requirement audit:
  `c4474c62677b1b5a4475d2855ebcb4c513aacb01117ed60f68d4c4c19fd9f439`
- Local CUDA result file:
  `66e7912ef7fcaf354ec67bbf6b8d0ea91d36ba1b2a96f14a507917fe16e91d4b`
- Non-production training-smoke result file:
  `e36ba864299d899a3d523c741e5664a92f9f7b0af0deb0d62ce8cbe2e62ad7f4`

## Nautilus state

The final deterministic job manifest is
`record/perfseer_v3_cuda_verifier_job.yaml`, SHA-256
`640c2e8ba1e16e585b2452c3e6ae3e612c4c3f374672c814dbb98a5b31727b57`.
Its embedded source-bundle SHA-256 is
`710cff66248e71f29678d6fea6351276e0e2ff87817489d0301d8b5ed19a9fdd`.
It is a finite Kubernetes Job using
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, one NVIDIA GPU, explicit
CPU/memory/ephemeral requests and limits, and a 900-second deadline.

No Nautilus object was submitted and no background monitor was started:
`kubectl` is not installed in this execution environment, so an authenticated
context cannot be checked or used. The tracked monitor is ready at
`record/monitor_perfseer_v3_cuda_verifier.sh`; its required future log path is
`record/perfseer_v3_cuda_verifier_monitor.log`.

## Remaining production work

1. Install `kubectl` and `kubectl-oidc_login`, configure the Nautilus context,
   submit the prepared job, immediately collect job/pod/describe/log/events,
   and run the tracked monitor at the mandated cadence.
2. Collect immutable grouped scheduler labels, target-hardware operation time,
   valid OOM/near-OOM rows, custom/OOV and generated-code rows, and a matched
   v2 set without source-family or graph-signature leakage.
3. Approve a new exact-ID registry revision only from that production
   measurement, then freeze dataset/split/schema/registry hashes.
4. Run T0/T1/T2 training and select on family-held-out error and calibration;
   distill S0–S3 and select the smallest candidate passing prediction, OOM,
   CPU latency, artifact size, memory, and scheduler integration gates.
5. Run all ten ablations and every acceptance slice, then proceed through
   shadow, canary, and rollback-controlled rollout only if all gates pass.
