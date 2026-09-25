# Student Distillation Retry: Self-Guard Repair

## Failure

The first student retry completed its 2,960-row cached-teacher output file,
then stopped before epoch one.  The post-cache A100 guard called
`_transfer_a100_guard()` with no permitted process identifier, so it classified
the live student process as foreign GPU contention.

## Root-cause evidence

`validate_a100_state` already supports an `own_pid` exemption.  Transfer
training uses that exemption after CUDA context creation, while the student
post-cache call omitted it.  A new regression test failed with the old helper
signature and passed after the one-argument propagation change.

## Verification and retry

- Focused regression test passed.
- The transfer, dataset, student, launch-visibility, and monitor suite passed
  19 tests.
- Both changed scripts were copied to ABA and verified byte-identical by
  SHA-256.
- The retry reused the exact completed 2,960-row, 145 MiB teacher-output cache
  and did not rerun teacher transfer.
- GPU 0 was empty before launch; GPU 1 VLLM was unchanged.
- Student retry PID 3750143 began epoch one successfully and wrote both best
  and latest checkpoints.
