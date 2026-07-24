# A10 Student Predictor Integration Plan

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
