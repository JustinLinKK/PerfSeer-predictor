# Verification

- Date: 2026-06-11

- Verified action: GPUUtilNet out-of-memory row removal.

- Result: all six CSV copies have zero `OOM_CRASH` rows.

- Verified action: model transfer from `Justin-Linux`.

- Result: local and remote Python model file counts match at 10001.

- Verified action: sampled model source import.

- Result: `calib_0000.py`, `calib_5000.py`, and `calib_9999.py` import and instantiate `GeneratedModel` with `python3`.

- Verified action: label-sampling smoke test from model source code.

- Result: `nrp_calibration_pack/profile/run_profile.py` loaded `calib_0000.py` model source code directly, ran one training repeat and one inference repeat on CPU, generated a result row and label file, then temporary files were deleted.

- Limitation: CPU smoke test only verifies source-code loading and label-file generation. It cannot verify GPU streaming multiprocessor or memory utilization collection.
