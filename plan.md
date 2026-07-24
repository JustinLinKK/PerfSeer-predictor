# Plan

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
