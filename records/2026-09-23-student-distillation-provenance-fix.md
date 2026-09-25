# Student Distillation Provenance Fix

## Setting

The A100 transfer teacher completed on the mixed original-A100 plus Justin
batch-size-one corpus.  The student launcher rejected the checkpoint before
training because the inherited parent normalizer does not carry a mixed-split
fingerprint.

## Evidence

- The teacher checkpoint, teacher report, and mixed dataset manifest all have
  dataset fingerprint `5cfb147f74af382ee1b4ac44379b185910ff40da7a8a9627581d1fbe77477d6f`.
- The checkpoint normalizer has no split-fingerprint field because it is
  intentionally inherited from the V3.2 parent checkpoint.
- The teacher-transfer code trains the completed teacher with that exact
  inherited normalizer on the mixed corpus.

## Change and verification

Student startup now verifies the checkpoint role, twelve-output contract,
mixed-dataset fingerprint, and normalization payload.  It no longer treats an
absent normalizer split fingerprint as a provenance mismatch.  The focused
test suite passed 4 tests; the related transfer, student, launch-visibility,
and monitor suite passed 18 tests.

## Conclusion

Restart only cached-teacher student distillation; do not rerun the completed
teacher transfer.
