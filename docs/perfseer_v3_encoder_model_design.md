# PerfSeer v3 encoder and teacher–student design

## Outcome and boundary

PerfSeer v3 is an isolated, versioned implementation under `src/perfseer_v3`.
It replaces v2's flat operation vector with a typed training graph and
hierarchical encoders while leaving every v2 package and artifact contract
unchanged. The locally testable capture, feature, model, training,
distillation, checkpoint, CPU export, evaluation, and scheduler-fallback paths
are implemented.

This is not a claim that a production teacher or student has been trained.
The checked-in operator registry remains `training_approved: false` because
the local source-first corpus has no production scheduler labels and is not an
authenticated target-hardware Nautilus measurement. The full runner enforces
that stop gate. All prediction, calibration, OOM, ablation, matched-v2, peak
training-memory, and training-duration gates remain unmeasured production
requirements.

The source-of-truth goal document is
`doc/PerfSeer_v3_encoder_and_model_goal_prompt.md`, SHA-256
`c422d0d8c1ec15ac187df7713c1fffaeee1fb666ac12fe04291485f2dc66c442`.
The production dataset protocol, coverage matrix, label definitions, split
policy, and collection stages are specified in
`docs/perfseer_v3_dataset_design_report.md`.

## GPU-specific model-pair contract

V3 defines one teacher and one distilled student for each concrete target GPU
type.
The training manifest, every captured graph, the teacher checkpoint, the
student checkpoint, and the artifact-registry record all carry the same
canonical `target_hardware_id`. A manifest cannot combine GPU types, a student
cannot distill from a teacher that predicts another GPU, and a prediction
request returns `hardware_mismatch` when its requested workload GPU differs
from the artifact target. `unknown`,
`generic`, `mixed`, and wildcard hardware targets are invalid production pair
identities.

`target_hardware_id` describes where the profiled workload labels were
measured and what hardware the pair predicts. It does not constrain the device
used to optimize the PerfSeer networks. The teacher may be trained on one GPU,
the student may be distilled later on another GPU, and the deployed student may
execute on CPU or another accelerator. During one distillation process the
teacher and student tensors must share that process's execution device, but
that device need not be the target/label GPU.

This is a model-pair boundary, not a requirement that training configurations
be homogeneous. One GPU pair can cover multiple precisions, optimizers,
schedulers, batch sizes, and model families when its measured dataset and
deployment allowlists contain them.

## Canonical graph and capture

`GraphIRV3` is an operation-node, tensor-multiedge graph. It records raw and
canonical operator identities, exact/family/hash IDs, source path, phase,
normalized arguments, independent input/output tensor metadata, cost source
and confidence, liveness, and saved-for-backward/optimizer state. Tensor edges
retain producer and consumer slots, role, shape, stride, dtype, layout,
alias/view/materialization state, dynamic-shape quality, and use distance.
Global state contains capture quality, replay evidence, workload fingerprints,
execution configuration, hardware, precision, optimizer, accumulation, and
per-phase totals.

Capture is strict-first `torch.export`. A non-strict graph is usable only after
three independent eager/export replay comparisons and is reported separately.
Exported tensor operations are never omitted: exact matches use the registry;
known long-tail operations use family/hash identity; unknown ATen and custom
names use a deterministic nonzero generic hash. Multi-output and repeated
tensor edges retain their individual slots. Custom operations require a
fake/meta implementation and are exercised with `torch.library.opcheck`.

The training adapter combines the exact same forward callable, inputs, loss,
optimizer, precision, and configuration with loss, AOT Autograd backward, and
optimizer phase nodes. Analytical backward remains explicitly marked as an
estimated fallback. Captured and profiled workload fingerprints must match.

The implementation follows the normalized ATen graph and shape-constraint
contract in the official [torch.export API reference](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html),
the fake-kernel and operator verification guidance in the official
[torch.library documentation](https://docs.pytorch.org/docs/stable/library.html),
and treats compiled-autograd behavior as version-pinned because the official
[Compiled Autograd tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
describes the API as under active development.

## Feature contract

The generated schema is `perfseer_graph_v3`. Its semantic hash is
`7d7966b124db6d473db2391bffa692cad2c74858985ef2cd2f478b8fbe210e78`;
the operation-registry semantic hash is
`e0cca4384b3a151cc0074e853de430a6493af02b3e669ccba2f8fe76a1f04d77`.
The ordered dense widths are 40 node continuous, 21 node flags, 14 edge
continuous, 5 edge flags, 105 global continuous, and 8 quality values.

Node categorical inputs are independent rather than conflated:

- exact operation ID, family ID, canonical hash, and raw-overload hash;
- forward/loss/backward/optimizer phase;
- input, output, and accumulation dtype;
- operator backend/namespace and feature-quality class;
- output layout and rank.

The edge encoder receives tensor role, producer/destination slot, dtype,
layout, rank, alias class, dynamic-shape quality, phase transition, edge
flags, and continuous size/stride/lifetime/fanout summaries. Model inputs,
outputs, parameters, buffers, and gradients have no external operation node;
they are represented as typed self-loops on the attached operation so their
semantics enter message passing without inventing an operation identity.
Unattached pass-through outputs are retained in graph metadata.

The global encoder receives the pair GPU and its capabilities; actual graph
precision; exact/family/hash optimizer and learning-rate-scheduler identities;
capture mode/backend; dynamic policy; training mode; graph and phase totals;
and common training hyperparameters. These include epoch and step progress,
initial/current/min/max and per-parameter-group learning rates, warmup and
decay settings, weight decay, momentum, betas, epsilon, clipping, scheduler
patience/cycles/power, and optimizer-specific values such as Muon's
Newton-Schulz step count.

The optimizer vocabulary covers PyTorch's standard optimizer set plus common
LAMB, LARS, and Lion configurations. Unknown future or custom names do not
collapse together: v3 retains an exact-or-`other` category, a semantic family,
and a stable hash. Composite optimizers retain their component signature. This
supports the normal Muon pattern in which Muon handles suitable 2-D hidden
weights while AdamW handles embeddings, biases, and other parameters.
Scheduler identity has the same exact/family/hash fallback and includes
constant, linear, step, exponential, cosine, warmup, cyclic/one-cycle,
reduce-on-plateau, inverse-square-root, polynomial, and chained/custom forms.

Precision is derived from captured tensor edges, not trusted from a single
graph label. A graph whose first layer is FP32 and next layer is BF16 is
encoded as `mixed`, retains both edge dtypes, and marks the conversion node.
Artifacts must explicitly allow `mixed` for such graphs.

These semantics follow the official
[PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html),
including the parameter split described by the
[Muon reference](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html),
and the schedule inputs documented for
[OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
and
[CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).

Continuous normalization is fit only on the training split. Schema, registry,
ordered layout, coarsening policy, split, and normalization hashes are embedded
in checkpoints. Any mismatch fails before inference or distillation.

## Model pair

`SeerNetV3` contains three hierarchical encoders and the original message
passing trunk as the control. The node identity fusion experiment supports
additive projections and concatenation followed by projection. The pooling
experiment supports the existing graph pooling and a phase-aware path that
summarizes all four phases before graph fusion. Both paths work on all-unknown,
isolated, empty-edge, and batched graphs.

The stable scheduler output is still one six-value prediction vector in this
order:

1. `train_epoch_ms`
2. `train_avg_sm_util_percent`
3. `train_p95_sm_util_percent`
4. `train_peak_vram_used_mib`
5. `train_peak_torch_reserved_mib`
6. `train_peak_memory_controller_util_percent`

The versioned `perfseer_v3_outputs_v2` wrapper adds log variance, OOM
probability, OOM failure stage, confidence, and peak-live-byte prediction.
Graph and phase embeddings are exposed for representation distillation but
are not scheduler metrics. Existing callers continue to receive the six
prediction values. Unsupported capture, high unknown fraction, OOD metadata,
low confidence, or a schema/hash mismatch returns a structured status and the
`branch_profile` recommendation.

## Capacity study

The capacity config defines exactly T0/T1/T2 and S0/S1/S2/S3. Counts below are
exact trainable tensor counts from meta-device construction. Student artifact
sizes and CPU latencies are actual TorchScript measurements on synthetic v3
small/median/large graph-depth probes with one CPU thread, three warmups, and
ten measured iterations. They are not comparisons to v2 and do not satisfy
the production deployment gate.

| Candidate | Hidden × blocks | Parameters | TorchScript bytes | CPU p95 small / median / large (ms) |
|---|---:|---:|---:|---:|
| T0 | 1024 × 8 | 136,980,818 | not exported | not measured |
| T1 | 1280 × 10 | 259,825,562 | not exported | not measured |
| T2 | 1536 × 10 | 374,031,306 | not exported | not measured |
| S0 | 192 × 2 | 1,805,014 | 7,346,687 | 1.289 / 2.446 / 4.348 |
| S1 | 224 × 2 | 2,439,890 | 9,895,125 | 1.578 / 2.729 / 5.805 |
| S2 | 256 × 2 | 3,169,594 | 12,815,795 | 1.853 / 3.402 / 6.676 |
| S3 | 256 × 3 | 4,091,706 | 16,511,205 | 2.435 / 4.244 / 9.278 |

T2 is 2.73× T0 and therefore stays below the approximately 3× guardrail. No
teacher has been selected: production selection prefers T1 unless T2 has a
statistically meaningful family-held-out gain. Student selection must choose
the smallest Pareto candidate that passes accuracy, CPU latency, artifact
size, calibration, and memory gates.

The measured report is `reports/perfseer_v3_capacity_study.json`, report hash
`9c78680521540634f8832fe5d7b4bd0f8d32e1c28f48c417acc74c6294b5f7a4`.

## Training and distillation

The production runner accepts `perfseer_v3_training_manifest_v2`. Each row
must name an immutable graph, split, source group, graph signature, six
targets, one concrete hardware ID, OOM label/stage, optional peak-live bytes,
and domain weight. The manifest names exactly one pair GPU and contains
measured dataset gates plus precision, capture, optimizer, scheduler, and
training-mode allowlists. A packaged placeholder is provided at
`src/perfseer_v3/training_manifest.example.json`. It
rejects duplicate samples, empty splits, source-family leakage, graph-signature
leakage, graph hash drift, split-fingerprint drift, schema drift, unapproved
registry data, missing measured GPU-time coverage, and unknown GPU time above
2%.

Encoder pretraining predicts family, exact identity, and cost targets. Teacher
training uses heteroscedastic six-target regression plus OOM, failure-stage,
peak-live, and confidence losses. Student training combines hard labels,
teacher predictions and uncertainty, plus graph- and phase-representation
relational distillation. Teacher and student must use the same target GPU, v3
dataset, split, normalization, schema, and registry hashes; no cross-GPU
teacher, v2 teacher, or strict v2 checkpoint load is allowed.

The runner supports CUDA, bfloat16/float16 autocast, float16 gradient scaling,
train-only normalization, deterministic epoch shuffling, validation
calibration, and a self-describing state-dict artifact. The local smoke command
is marked `smoke_only_not_production` and cannot bypass production gates.

## Reproducible commands

Run from the repository root in the pinned environment.

### Schema, workload, capture, and coverage

```bash
python scripts/build_v3_schema.py
python scripts/build_perfseer_v3_workload_manifest.py \
  --seed 42 \
  --output reports/perfseer_v3_workload_manifest.json
python scripts/verify_perfseer_v3_workload_matrix.py \
  --device cuda \
  --output reports/perfseer_v3_workload_execution.json
python scripts/profile_perfseer_v3_microbenchmarks.py \
  --output reports/perfseer_v3_microbenchmark_gpu_time.json \
  --raw-output reports/perfseer_v3_microbenchmark_profiles.jsonl \
  --dtype float32 --batch-size 4 --warmup-steps 3 --measured-steps 10
python scripts/profile_perfseer_v3_supported_corpus.py \
  --output reports/perfseer_v3_supported_corpus_gpu_time.json \
  --raw-output reports/perfseer_v3_supported_corpus_profiles.jsonl \
  --warmup-steps 3 --measured-steps 10 --operator-repeats 20
python scripts/audit_operation_coverage.py \
  --corpus supported \
  --gpu-time-json reports/perfseer_v3_supported_corpus_gpu_time.json \
  --output-dir reports
python scripts/audit_perfseer_v3_costs.py \
  --output reports/perfseer_v3_cost_decomposition_audit.json
```

### Encoder/model evidence and capacity

```bash
python scripts/benchmark_perfseer_v3_capacity.py \
  --benchmark-students --warmup 3 --iterations 10 --cpu-threads 1 \
  --output reports/perfseer_v3_capacity_study.json
python scripts/audit_perfseer_v3_encoder_model.py
python -m unittest discover -s tests -p 'test_perfseer_v3*.py'
python scripts/run_perfseer_v3_training.py smoke \
  --output reports/perfseer_v3_training_smoke.json
python scripts/run_perfseer_v3_cuda_verifier.py \
  --output reports/perfseer_v3_cuda_verifier.json
```

### Full teacher and student

These commands are exact, but they correctly stop before allocating the large
model until `artifacts/perfseer_v3_training_manifest.json` contains immutable,
measured, leakage-free production evidence and the registry is approved from
that evidence.

```bash
python scripts/run_perfseer_v3_training.py teacher \
  --config src/perfseer_v3/configs/train_hardware_teacher/v3_teacher.yaml \
  --manifest artifacts/perfseer_v3_training_manifest.json \
  --output artifacts/perfseer_v3_teacher.pt \
  --device cuda --amp bfloat16 --pretrain-epochs 20

python scripts/run_perfseer_v3_training.py student \
  --config src/perfseer_v3/configs/train_deploy_model/v3_student.yaml \
  --manifest artifacts/perfseer_v3_training_manifest.json \
  --teacher-artifact artifacts/perfseer_v3_teacher.pt \
  --output artifacts/perfseer_v3_student.pt \
  --device cuda --amp bfloat16
```

### Evaluation, export, and scheduler integration

```bash
python scripts/generate_perfseer_v3_evaluation_slices.py \
  --seed 42 --output reports/perfseer_v3_evaluation_slices.json
python scripts/evaluate_perfseer_v3_predictions.py \
  --predictions artifacts/perfseer_v3_test_predictions.jsonl \
  --ablations artifacts/perfseer_v3_ablations.json \
  --output reports/perfseer_v3_prediction_evaluation.json \
  --near-zero-epsilon 1e-6 --oom-threshold 0.5
python scripts/export_perfseer_v3_student.py \
  --artifact artifacts/perfseer_v3_student.pt \
  --graph artifacts/export_verification_graph.json \
  --output artifacts/perfseer_v3_student.torchscript.pt
python scripts/run_perfseer_v3_scheduler_integration.py \
  --artifact artifacts/perfseer_v3_student.pt \
  --graph artifacts/scheduler_integration_graph.json
```

### Nautilus CUDA verifier

Generate the PyTorch CUDA job first. Submit it only with an authenticated
context. Immediately after submission, run every diagnostic below before any
other work, then start the tracked monitor.

```bash
python scripts/prepare_perfseer_v3_nautilus_verifier.py \
  --namespace ecepxie \
  --output record/perfseer_v3_cuda_verifier_job.yaml
kubectl apply -f record/perfseer_v3_cuda_verifier_job.yaml
kubectl get job --namespace ecepxie perfseer-v3-cuda-verifier -o wide
kubectl get pod --namespace ecepxie -l job-name=perfseer-v3-cuda-verifier -o wide
kubectl describe pod --namespace ecepxie \
  "$(kubectl get pod --namespace ecepxie -l job-name=perfseer-v3-cuda-verifier -o jsonpath='{.items[0].metadata.name}')"
kubectl logs --namespace ecepxie \
  "$(kubectl get pod --namespace ecepxie -l job-name=perfseer-v3-cuda-verifier -o jsonpath='{.items[0].metadata.name}')" \
  --all-containers --tail=200
kubectl get events --namespace ecepxie --sort-by='.lastTimestamp' | tail -n 100
nohup env PERFSEER_NAUTILUS_NAMESPACE=ecepxie \
  bash record/monitor_perfseer_v3_cuda_verifier.sh \
  record/perfseer_v3_cuda_verifier_monitor.log \
  >record/perfseer_v3_cuda_verifier_monitor.nohup.log 2>&1 &
```

The monitor records job/pod state, pod description, events, process state,
persistent log tail, and output/checkpoint files every 45 seconds for the first
five minutes and every 20 minutes afterward.

## Evaluation and acceptance

Evaluation reports MAE, MAPE with an explicit near-zero exclusion, raw/log
RMSE, R², p50/p90/p95 percentage error, interval coverage, OOM precision,
recall, false-positive rate, Brier score, and failure-stage accuracy. It slices
by architecture, operation family, modality, phase, batch, precision,
optimizer, capture quality, graph size, resource regime, unknown fraction, and
held-out evaluation slice.

The exact ten ablations are enforced: v2 one-hot versus hierarchical;
forward-only versus training graph; exact-only versus family/exact/hash;
topology-only versus cost/liveness; raw versus coarsened; existing versus
phase-aware pooling; T0 versus T1/T2; S0 versus S1/S2; random versus compatible
trunk reuse; and hard labels versus hard labels plus distillation.

Teacher and student matched-v2 non-regression are separate 5% gates. The new
complex-operation improvement targets are 15% for the teacher and 10% for the
student. Deployment additionally requires matched-v2 CPU latency and artifact
ratios, eager/export equivalence, no leakage, and fail-closed behavior.

## Measured local evidence and remaining risk

The current supported coverage report has 18/18 structurally encodable models
and 638/638 tensor nodes retained. No model is marked accuracy-validated. The
compatibility report deliberately shows 0/18 exact-only models because exact
identity is a selective hot-operation vocabulary, not a structural-support
requirement. Every measured dispatcher identity in the local supported corpus
resolves to a known family, giving 0% unknown measured diagnostic time, but
this local evidence cannot approve production training.

Exact environment used for the final local work:

- Python 3.13.12 (Anaconda, GCC 14.3.0)
- PyTorch 2.11.0+cu130; CUDA build 13.0; cuDNN 9.19.0
- PyG 2.7.0
- NumPy 2.4.4
- PyYAML 6.0.3
- Transformer Engine unavailable in the local environment

The final production blockers are authenticated Nautilus verification,
production scheduler labels, target-hardware operation timing, registry
approval from immutable measurement, real T/S training, held-out prediction
and calibration records, OOM cases, all ten ablations, predictor peak training
memory/duration, and matched-v2 deployment ratios. Until they exist, the v3
runtime remains shadow-only with branch-profile fallback.
