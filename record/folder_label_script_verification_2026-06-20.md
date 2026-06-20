# Folder Label Script Verification 2026-06-20

## Experiment Setting

- Script: `scripts/run_nautilus_folder_label_sampling.py`
- Test: `python3 scripts/test_run_nautilus_folder_label_sampling.py`
- Compile check: `python3 -m py_compile scripts/run_nautilus_folder_label_sampling.py scripts/test_run_nautilus_folder_label_sampling.py`
- Manual dry run:
  - Input: temporary folder containing `manual_mlp.py`
  - Command: `python3 scripts/run_nautilus_folder_label_sampling.py --models-dir <tmp>/models --local-labels-dir <tmp>/labels --namespace ecepxie --pvc test-pvc --gpus a100,a40,l4,rtx_a4000 --run-id manual-dry --dry-run`
  - Cleanup: removed temporary model folder and generated `record/nautilus_folder_labels_manual-dry.*`

## Result

- Unit tests: `Ran 5 tests in 0.224s`, `OK`
- Compile check: passed with no output
- Manual dry run produced:
  - `record/nautilus_folder_labels_manual-dry.md`
  - `record/nautilus_folder_labels_manual-dry.yaml`
  - `record/nautilus_folder_labels_manual-dry.manifest.jsonl`
- Manual dry-run manifest row:
  - `model_id`: `manual_mlp`
  - `model_file`: `manual_mlp.py`
  - `input_shape`: `[8, 16]`
  - `precision_config`: `fp32_ieee`
  - `label_file`: `label/label/manual_mlp_fp32_ieee.txt`
- Manual dry-run YAML contained:
  - `run_profile.py`
  - `verify_sampled_labels.py`
  - output directories under `/workspace/perfseer-folder-runs/manual-dry/labels/<gpu>`
  - GPU resources for `a100`, `a40`, `l4`, and `rtx_a4000`
- Temporary dry-run artifacts remaining: none

## GPU Switching Update

## Experiment Setting

- Script updated: `scripts/run_nautilus_folder_label_sampling.py`
- Test updated: `scripts/test_run_nautilus_folder_label_sampling.py`
- Switching fields added:
  - `--active-gpus`
  - `--pending-timeout-seconds`
  - `--blacklist-ttl-seconds`
  - `--max-retries-per-gpu`
  - `--min-successful-gpus`
- Manual dry run:
  - Input: temporary folder containing `switch_mlp.py`
  - Command: `python3 scripts/run_nautilus_folder_label_sampling.py --models-dir <tmp>/models --local-labels-dir <tmp>/labels --namespace ecepxie --pvc test-pvc --gpus a100,a40,l4,rtx_a4000 --active-gpus 2 --pending-timeout-seconds 30 --run-id switch-folder-dry --dry-run`
  - Cleanup: removed temporary model folder and generated `record/nautilus_folder_labels_switch-folder-dry.*`

## Result

- Unit tests after switching update: `Ran 7 tests`, `OK`
- Compile check after switching update: passed with no output
- `--help` includes all switching arguments listed above.
- Manual dry-run YAML contained one Job document for each candidate GPU:
  - `a100`
  - `a40`
  - `l4`
  - `rtx_a4000`
- Manual dry-run YAML contained per-GPU output directories under
  `/workspace/perfseer-folder-runs/switch-folder-dry/labels/<gpu>`.
- Manual dry-run YAML contained `run_profile.py` and
  `verify_sampled_labels.py` commands.
- Manual dry-run manifest row:
  - `model_id`: `switch_mlp`
  - `model_file`: `switch_mlp.py`
  - `input_shape`: `[8, 16]`
  - `precision_config`: `fp32_ieee`
- Temporary dry-run artifacts remaining: none
