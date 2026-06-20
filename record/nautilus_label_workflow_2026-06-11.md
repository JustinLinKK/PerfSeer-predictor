# Nautilus Label Workflow

- Date: 2026-06-11

- Goal: create a Nautilus workflow for sampling labels directly from PyTorch model source code.

- Input model contract: each source file exposes `make_model()` and optionally `MODEL_ID` and `INPUT_SHAPE`.

- Manifest script: `scripts/build_source_manifest.py`.

- Workflow script: `scripts/create_nautilus_label_workflow.py`.

- Default GPUs: `a100`, `a40`, `l40s`, and `v100`.

- Allowed GPU presets are limited to the current README list: `a10`, `a40`, `a100`, `l4`, `l40`, `l40s`, `rtx-a4000`, `rtx-a5000`, `rtx-a6000`, `rtx-4000-ada`, `rtx-5000-ada`, `rtx-pro-6000-blackwell`, `quadro-rtx-6000`, `quadro-rtx-8000`, `t4`, and `v100`.

- Generated Kubernetes Jobs call `nrp_calibration_pack/profile/run_profile.py` with one inference repeat and one training repeat.

- Output labels follow the existing profiler format and result JSON rows include hardware metadata.

- Verification: Python compilation passed for both scripts.

- Verification: manifest generation over `nrp_calibration_pack/models` produced 10000 rows.

- Verification: workflow YAML generation produced four jobs for `a100`, `a40`, `l40s`, and `v100`.

- Verification: invalid GPU preset `h100` was rejected because it is not listed in the current README.
