# ABA transfer-trained V3.2 model copied to the local package

## Destination

The completed Justin RTX 5090 batch-size-one mixed-corpus transfer run was copied from ABA into `predictor_v3.2/` without replacing the older V3.2 package pair.

| Role | Local public path | Canonical cached file | SHA-256 |
| --- | --- | --- | --- |
| Student | `predictor_v3.2/weights/predictor_v3.2_transfer_student_justin_batch1_20260922.pt` | `predictor_v3.2/.cache/predictor_v3.2_transfer_student_justin_batch1_20260922.pt` | `c89eedb6291f11eab264ac6110defec34a0ce3d26bfc16d778040fc30e803b54` |
| Teacher | `predictor_v3.2/weights/teacher_v3.2_transfer_justin_batch1_20260922.pt` | `predictor_v3.2/.cache/teacher_v3.2_transfer_justin_batch1_20260922.pt` | `3c488fccd103e16c6247f174a04bdb58511774a648f8e89398469dcf3a71e284` |

The public paths are symbolic links to `.cache`, avoiding duplicate multi-gigabyte checkpoints.  Final reports were copied to `predictor_v3.2/weights/transfer-student-final-report-20260922.json` and `predictor_v3.2/weights/transfer-teacher-final-report-20260922.json`.

## Verification

- Both local hashes match the source ABA artifacts.
- Both checkpoints loaded on the local Central Processing Unit (CPU) using `torch.load(..., map_location="cpu")`.
- `PYTHONPATH=predictor_v3.2/source python -c 'import perfseer_v32.inference'` completed successfully.
- The student final report is `complete`, has 2,888,744 parameters, six heads, and 12 targets.
- Its teacher checksum equals the copied teacher checksum, and both reports use dataset fingerprint `5cfb147f74af382ee1b4ac44379b185910ff40da7a8a9627581d1fbe77477d6f`.
