# Plan

## Completed PerfSeer v3 dataset-design report

Source of truth: the current v3 graph, profiling, manifest, split, training,
artifact, runtime, and evaluation contracts plus the user's corrected hardware
semantics. A hardware-specific pair predicts labels measured on one target GPU
model; the teacher and student training computations may run on any suitable
CPU/GPU and need not run on the target GPU.

Implementation status: complete. The comprehensive design specification is
`docs/perfseer_v3_dataset_design_report.md`. It covers hardware qualification,
raw and aggregate schemas, six label definitions, inputs, workload/optimizer/
scheduler/dtype coverage, OOM boundaries, target-GPU measurement protocol,
aggregation, leakage-safe splits, balancing, storage, capacity planning,
staged execution, and readiness checks. All 18 optimizer identities, 17
scheduler identities, six target labels, six OOM stages, and eight evaluation
slices were checked directly against source constants. All three JSON examples
parse; 35 focused tests, compileall, stale-wording search, and
`git diff --check` pass.

1. Audit the current data contract and separate target/measurement hardware
   from predictor-training execution hardware.
   - Verifier: map every label, feature, provenance field, gate, and split rule
     to current source and identify any ambiguous wording.
2. Specify a production dataset matrix spanning workload families, graph
   regimes, dtypes, optimizers, schedulers, hyperparameters, execution modes,
   OOM boundaries, and repetitions.
   - Verifier: every model input and output has an explicit collection source,
     unit, sampling rule, and missing-value policy.
3. Define leakage-safe splitting, quality control, label aggregation,
   balancing, staged collection, acceptance gates, storage layout, and capacity
   estimation for one target GPU dataset.
   - Verifier: include executable/current manifest mappings, validation rules,
     failure handling, and readiness checklists without inventing measured
     production results.
4. Correct hardware wording in current v3 documentation and verify the report
   against source constants and focused tests.
   - Verifier: stale searches find no claim that predictor training must run on
     the label GPU; relevant manifest/distillation/runtime tests and
     `git diff --check` pass.

## Completed GPU-specific and training-hyperparameter correction

Source of truth: the current user goal and the isolated `src/perfseer_v3`
implementation. Preserve the stable six-output scheduler contract and v2
packages while making the v3 model explicitly per-GPU and broadening its
training-configuration inputs.

Implementation status: complete. Each v3 teacher/student pair is bound to one
concrete GPU; optimizer and scheduler exact/family/hash inputs, mixed per-layer
dtypes, and common learning-rate/training hyperparameters are implemented.
The final feature/registry hashes are
`7d7966b124db6d473db2391bffa692cad2c74858985ef2cd2f478b8fbe210e78` /
`e0cca4384b3a151cc0074e853de430a6493af02b3e669ccba2f8fe76a1f04d77`.
All 95 focused v3, 103 repository, 26 legacy converter, and 51 calibration
tests pass; local CUDA, smoke training, deterministic generation, and isolated
wheel verification also pass. The prepared Nautilus verifier was not submitted
because `kubectl` is unavailable in this environment.

1. Audit the current graph, feature, training-manifest, artifact, registry, and
   runtime contracts for GPU specificity, optimizer coverage, layer-level
   dtype fidelity, and learning-rate schedule inputs.
   - Verifier: map every requested behavior to concrete source and test
     evidence, and use current primary documentation for optimizer semantics.
2. Enforce exactly one canonical GPU target for each teacher/student pair and
   prevent cross-GPU datasets or teacher/student pair mismatches.
   - Verifier: same-GPU teacher/student fixtures pass; multi-GPU manifests and
     cross-GPU distillation fail before training.
3. Add extensible exact/family/hash optimizer and scheduler encodings, including
   Muon, plus common optimizer, epoch, learning-rate, warmup, and decay fields.
   - Verifier: known and previously unseen optimizers/schedulers remain
     distinguishable, finite, serializable, and batchable.
4. Preserve per-operation and per-edge dtype transitions so heterogeneous
   layer dtypes such as FP32 to BF16 are represented without relying on one
   graph-wide precision label.
   - Verifier: a mixed-layer capture retains both dtypes and the transition in
     encoded node/edge tensors and completes model forward/backward.
5. Regenerate deterministic schema/artifact evidence, update documentation,
   and run focused, repository, packaging, and local CUDA verification.
   - Verifier: generated hashes are stable; all focused and full tests,
     compile/import, wheel, CUDA, and `git diff --check` pass.

## Completed PerfSeer v3 encoder/model goal audit and refactor

Source of truth: `doc/PerfSeer_v3_encoder_and_model_goal_prompt.md`.
Preserve the isolated `src/perfseer_v3` implementation and all existing v2
contracts while closing any encoder/model gaps found by direct inspection.

Implementation status: all locally executable encoder/model work is complete.
Its schema/count evidence has been superseded by the GPU-specific correction
above. Production training remains blocked by the intentionally unapproved
registry and missing grouped scheduler labels.

1. Convert the goal prompt into a requirement-to-evidence audit covering the
   IR, feature encoders, trunk/pooling, optional heads, capacity candidates,
   training/distillation, export/runtime, coverage, evaluation, and docs.
   - Verifier: every explicit encoder/model requirement has a current source,
     test/report, missing-evidence, or blocked-data classification.
2. Refactor the graph feature contract and node/edge/global encoders for any
   missing categorical identities, phase/slot semantics, quality indicators,
   and continuous-feature separation.
   - Verifier: schema hashes regenerate deterministically and focused feature,
     model, unknown/custom, batching, backward, and TorchScript tests pass.
3. Add the phase-aware pooling control and versioned optional-output behavior
   without changing the scheduler's six-value prediction contract.
   - Verifier: existing and phase-aware pooling modes both train, serialize,
     export, and preserve finite six-target outputs.
4. Add explicit T0/T1/T2 and S0/S1/S2/S3 capacity-study definitions plus a
   reproducible parameter/latency/artifact benchmark path.
   - Verifier: configs validate, actual trainable parameter counts are emitted,
     and smoke-size candidates complete controlled forward/backward checks.
5. Strengthen staged teacher/student training, representation distillation,
   checkpoint metadata, and fail-closed compatibility where the audit finds
   incomplete contracts.
   - Verifier: tiny teacher/student training and export/reload integration pass;
     schema, registry, layout, normalization, and model-contract mismatches are
     rejected.
6. Produce the requested encoder/model design, compatibility matrix, exact
   commands, audit report, and final implementation evidence. Clearly separate
   measured local results from production-data-dependent accuracy/ablation
   gates.
   - Verifier: documentation values are derived from current artifacts and the
     full focused, repository, legacy, packaging, deterministic-generation,
     and local CUDA regression checks pass.

## Active PerfSeer v3 implementation

The implementation target is the isolated `src/perfseer_v3` package. Existing
v2 packages remain intact and serve as regression references.

Implementation status: all locally executable work in steps 1–12 has an
implemented and verified software path. The complete 1,828-entry workload
matrix passed on local CUDA, and local measurement evidence covers 75 exact
microbenchmarks plus 42 P0, representative-real, and composite profiles. The
supported-corpus diagnostic proposes a 67-operation 95%-time vocabulary but
intentionally does not approve production training. The eight required
evaluation slices are explicit and fail closed because three production-data
slices are absent. Step 13 is deterministic and prepared with a PyTorch CUDA
image, but it was not submitted because this checkout has no authenticated
Kubernetes context. Production teacher/student training remains blocked by the
missing rebuilt measured scheduler-label corpus; exact evidence and hashes are
recorded in `reports/perfseer_v3_implementation_status.md`.

1. Inventory and freeze the v2 baseline, then add the coverage corpus and
   operation-coverage auditor.
   - Verifier: baseline metadata is deterministic; focused auditor tests pass;
     generated report data validates against its schema.
2. Add the versioned operation registry, generated JSON schema, and `GraphIRV3`
   dataclasses with deterministic serialization and hashes.
   - Verifier: registry validation, hash-sensitivity, and complete round-trip
     tests pass.
3. Implement strict-first `torch.export` capture, structured failures,
   tensor-slot edges, dynamic constraints, custom/unknown operation handling,
   and validated non-strict replay.
   - Verifier: golden capture and eager/export equivalence tests pass; no
     captured tensor-producing operation is silently dropped.
4. Implement explicit-unit cost estimation, alias-aware tensor liveness, and
   forward/loss/backward/optimizer phase summaries.
   - Verifier: analytical shape/FLOP/byte tests and liveness invariants pass.
5. Implement source-first profiling contracts so capture and measurement use
   the same callable and retain raw samples plus environment/identity metadata.
   - Verifier: identity mismatch is rejected and profile records round-trip.
6. Build deterministic microbenchmark, composite, and source-model workload
   descriptors with coverage-driven selection and source-family-grouped splits.
   - Verifier: manifest hashes, split isolation, and coverage-cell selection
     tests pass.
7. Implement the v3 feature builder and safe deterministic graph coarsener.
   - Verifier: tensor layout/hash checks, coarsening conservation, and cache
     invalidation tests pass.
8. Implement hierarchical operation/family/hash/phase encoders and
   uncertainty, OOM, and confidence heads for the v3 teacher/student.
   - Verifier: unknown-only, empty, isolated, batched, backward, and
     serialization smoke tests produce finite outputs.
9. Add v3 teacher/student configs and the staged-training orchestration.
   - Verifier: configuration/schema validation and a tiny CPU training and
     distillation smoke run pass before any cluster training is considered.
10. Implement coverage/accuracy/ablation reports and fail-closed acceptance
    gates.
    - Verifier: deliberately failing fixtures are blocked and valid fixtures
      produce deterministic reports.
11. Implement artifact metadata/registry checks, scheduler result states,
    fallback recommendations, shadow comparison, and canary policy.
    - Verifier: artifact corruption/schema mismatch fail closed; every non-OK
      state selects the configured fallback.
12. Run focused tests after every step, followed by the full repository test
    suite, import/compile checks, stale-reference searches, and
    `git diff --check`.
    - Verifier: 95 focused v3, 103 complete `tests/`, 26 legacy converter, and
      51 calibration-pack tests pass; the wheel installs with its v3
      registry/schema/config package data.
13. Only after all prerequisite data gates pass, run the required PyTorch CUDA
    verifier on Nautilus using a PyTorch CUDA image. Immediately collect job,
    pod, describe, logs, and events; start the required tracked monitor under
    `record/`; report live progress at least once per minute until stable.
    - Current stop gate: the manifest and monitor are ready, but `kubectl` has
      no current/configured context and no kubeconfig, so no job was submitted
      and no monitor log was fabricated.

## Preserved prior v2 training work

1. Inventory teacher/student configs, docs, scripts, and tests that still point at legacy six-target training.
2. Add a canonical v2 training target that uses measured epoch time from `scheduler_label_v3.jsonl` plus resource labels from `scheduler_resource_label.jsonl`.
3. Keep exactly one non-legacy teacher architecture and one non-legacy student architecture for per-hardware model pairs.
4. Move older architecture configs under `configs/legacy/` with explicit warnings so teammates do not pick them accidentally.
5. Update `run_hardware_distill_flow.py` and README commands to default to the canonical v2 pair.
6. Verify with dry-run commands, targeted unit tests, stale-reference searches, and `git diff --check`.

## Historical A10 Student Predictor Integration Plan

## RTX PRO 6000 Blackwell integration

- Inspect `models/student_RTX_6000_Blackwell.pt` without modifying it and
  identify its schema, targets, configuration, normalization statistics, and
  checkpoint integrity.
  - Verifier: load on CPU, enumerate checkpoint keys and tensor shapes, and
    compare the declared contract with the production encoder/runtime.
- Export the trusted checkpoint as a self-contained CPU TorchScript artifact
  with embedded input normalization and output de-normalization.
  - Verifier: compare eager and reloaded TorchScript outputs on more than one
    encoded graph, require finite positive `train_mem`, CPU-only parameters,
    buffers, inputs, and outputs, and unchanged CUDA allocation.
- Register normalized RTX PRO 6000 Blackwell aliases, compute capability,
  VRAM bounds, schema, output index, artifact path, and SHA-256.
  - Verifier: test exact and real-world GPU-name aliases, reject mismatched
    capability/VRAM, and detect artifact corruption.
- Integrate without regressing the A10 artifact or per-job branch fallback.
  - Verifier: run focused PerfSeer tests, scheduler ML tests with automatic
    Blackwell selection and explicit override, Stress Test Data v1.0
    prediction verification, and the full scheduler suite.

## Completed deployment work

- Consolidated the student model, graph featurizer, source encoder, CPU runtime,
  and export tooling into the installable `perfseer_student` package.
- Reused `perfseer_source_converter` for FX tracing instead of carrying a
  second converter implementation.
- Exported the self-contained A10 CPU TorchScript artifact with embedded input
  normalization and output de-normalization.
- Retained only `models/nvidia_a10/student_a10_cpu.torchscript.pt` and recorded
  its hardware/schema/hash metadata.
- Verified source conversion and TorchScript inference entirely on CPU.

## Current operation-coverage and pressure-fixture work

- Document the exact boundary between operations recognized by the source
  converter and operation identities represented by the deployed `53/3/40`
  student.
  - Verifier: derive both sets from the checked-in converter and featurizer and
    keep the report's exact mismatch table aligned with those sources.
- Reject converter labels that are absent from the deployed student's
  operation vocabulary instead of silently encoding an all-zero identity.
  - Verifier: add a source fixture using a converter-recognized but
    student-unknown operation and require `encode_source` to reject it.
- Define a deterministic 100-model pressure list that uses only operation
  identities represented by the current student.
  - Verifier: check the fixture count, IDs, source metadata, architecture and
    precision distributions, and deterministic SHA-256 manifest.
- Run every pressure-list entry through source conversion and the retained CPU
  TorchScript artifact.
  - Verifier: require 100/100 finite positive `train_mem` predictions, exact
    `53/3/40` tensors, CPU-only tensors, no CUDA allocation change, and no
    unknown operation identities.
- Run the complete PerfSeer and scheduler test suites after the focused
  verifiers pass.

## Preserved prior integration-guide work

1. Confirm the active v2 teacher/student configs and target source from the repository.
2. Inspect the optimized data/model code to document the real tensor input and six-target output contract.
3. Write the integration guide under `doc/` with exact file inputs, tensor fields, target order, and runtime metadata.
4. Verify the guide against the current configs and source with import checks, text searches, and `git diff --check`.

## GraphCode multi-model workspace implementation

1. Extend GraphCode project initialization, scanning contracts, persistence, canvas APIs, and layout storage to support an ordered list of explicit top modules while preserving existing single-root workspaces.
2. Enable top-level ML model nodes and teach the scanner to resolve the v2 student and teacher YAML configurations through their concrete `SeerNetMulti`, `SeerTrunk`, and `SeerBlock` workflows with source-backed nodes and edges.
3. Add a local `.graphcode/memory` store with progressive disclosure, typed semantic/procedural/episodic entries, hash-based staleness, safe server-owned writes, and retrieval for planning, coding, review, and scanning agents.
4. Generate a local ignored workspace containing the student and teacher roots, their nested layer workflows, tensor contracts, six output heads, and the teacher-to-student distillation relationship.
5. Verify schema migrations, APIs, scanner behavior, memory lifecycle, UI canvas behavior, model construction, graph invariants, and unchanged rescans; run the required PyTorch CUDA verifier through Nautilus and retain its monitoring evidence under `record/`.
