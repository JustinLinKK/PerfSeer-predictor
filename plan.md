# A10 Student Predictor Integration Plan

- Consolidate the student model, graph featurizer, source encoder, CPU runtime,
  and export tooling into an installable `perfseer_student` package.
- Reuse `perfseer_source_converter` for FX tracing instead of carrying a second
  converter implementation.
- Export a self-contained CPU TorchScript artifact from the A10 checkpoint,
  embedding input normalization and output de-normalization.
- Keep only `models/nvidia_a10/student_a10_cpu.torchscript.pt` as a neural model
  artifact and register its hardware/schema/hash metadata.
- Verify source-to-graph conversion and TorchScript inference entirely on CPU.
- Add focused package tests and run the existing converter test suite.
- Commit the submodule change locally so the parent repository can record the
  new gitlink. Do not push.
