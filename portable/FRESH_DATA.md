# Release paths and fresh-data pipeline

Run these examples from the clean release branch root, which contains `data/`
for generation/labeling source and `portable/` for predictors, feature packing,
training and scoring. The archived data README retains its original workspace
paths as provenance; this guide describes the clean release layout.

The verified full-dataset archives extract into:

```text
PerfSeer-release-v1/A10/data/A10/dataset.pt
PerfSeer-release-v1/A10G/data/A10G/dataset.pt
PerfSeer-release-v1/A100/data/A100/dataset.pt
```

Each hardware directory also contains its normalizer and validation key.
Validation/test input bundles are inside `data/<HW>/`. The transfer archive
extracts compact packs to `PerfSeer-release-v1/transfer/<HW>/dataset.pt`, with
`selection.json`, exact historical labels, explicit missing outcomes and receipts
alongside. Its archived generator copy is under `PerfSeer-release-v1/generator/`.
The clean branch's maintained source copy is `data/`. Large weight archive paths
are supplied explicitly to the CLI after extracting the model assets.

Load Torch bundles only after importing the portable bootstrap:

```python
from portable.teachers import bootstrap
bootstrap()  # Adds portable/runtime/src and portable/runtime/aliases.
import torch
bundle = torch.load("PerfSeer-release-v1/A10/data/A10/dataset.pt",
                    weights_only=False, map_location="cpu", mmap=True)
```

## Generate, measure, pack and train

Install `portable/requirements-tested.txt` in a Python environment with a CUDA
build of PyTorch compatible with the local NVIDIA driver. The NVML Python
package is needed only for fresh GPU labeling. The workload runtime under
`data/runtime/` implements generated programs; `portable/runtime/` implements
the pinned static feature extractor and prediction models.

```bash
# Directly create the full reproducible design, including frozen held-out groups.
python data/generate.py --mode pretrain --output generated/pretrain

# Or create the smaller independent TRAIN-only transfer design.
python data/generate.py --mode transfer --train-budget 8192 --output generated/transfer

# Detect the actual GPU. The labeler never accepts a substitute hardware name.
python data/label.py detect
python data/label.py run --generated generated/pretrain --output fresh-labels \
  --gpu-lock local-gpu.lock

# Replace the directory below with the hardware_id printed by detect.
python data/verify_local.py --generated generated/pretrain \
  --output fresh-labels/nvidia_geforce_rtx_5090_local --receipt local-labels-check.json

# Static CPU feature extraction, frozen normalization and honest measured labels.
python -I portable/pack_fresh.py --generated generated/pretrain \
  --labels fresh-labels/nvidia_geforce_rtx_5090_local \
  --architecture portable/architecture.pt --output fresh-pack

# Full target continuation once the pack has successful TRAIN and validation rows.
python -I portable/train_pretrain.py \
  --hardware nvidia_geforce_rtx_5090_local \
  --source checkpoints/P_A100.pt --architecture portable/architecture.pt \
  --dataset fresh-pack/dataset.pt --output runs/P_local --device cuda
```

The examples show how to create new target data and explicitly warm-start a
target model. They do not assert that a full RTX5090 dataset or model was trained
for this release. The five-row local smoke measures implementation behavior only.
For historical A10/A100/A10G continuation, use the actual full archive paths and
their corresponding hardware names. The selected transfer trainer remains
restricted to A100-to-A10; its default hyperparameters and reproducible state
are documented in `README.md`.

`pack_fresh.py` reads the directly generated standalone variant source and uses
the pinned static AST graph extractor. It checks source, input, configuration
group, run and GPU identities and recomputes timing, busy and memory targets
from raw labeler observations. It applies the original frozen A10 normalizer
once. Capture remains estimated and forward-only: backward kernels, lifetimes
and actual operator AMP dtypes are not observed by this extractor.

The packer derives the hardware identifier exclusively from the actual measured
GPU. A fresh-pack training command must use that exact identifier; an RTX5090
pack cannot be declared A10 or A100. Failed, OOM, contaminated and unmeasured rows
remain in `missing.jsonl`; they are never imputed or replaced. Group splits remain
those of the generator. A TRAIN-only transfer design therefore has no new
validation set: use its existing identity-matched target evaluation pack or
generate/measure the full design for a new target. The packer marks
`ready_for_training=false` until both TRAIN and validation measurements exist.

Outputs are `dataset.pt` (TRAIN+validation), `all-inputs.pt`,
`validation-inputs.pt`, `validation-key.npz`, `test-inputs.pt`, `test-key.npz`,
`manifest.json`, `missing.jsonl` and a file-hashed `verification.json`. Sealed test
labels are kept out of the training tensor bundle. V5 memory targets are decimal
100 MB ceilings stored in MiB; original measured values remain `raw_targets_v4`.
For multi-step local epochs, mean step time is measured epoch time divided by
the explicit count; the predictor's `derived_step_times_count` contract preserves
that arithmetic. This modern measurement protocol is not claimed equivalent to
historical cluster labels.

Input-only bundles can be passed directly to `portable/infer.py`. Existing
target-specific teachers estimate their own target's costs; recording a new
GPU identity does not turn an old target model into a calibrated new-GPU model.

## Source allowlist and finalizer

`SOURCE_ALLOWLIST.json` enumerates the source files to copy from the staging
tree into the clean branch. It includes generator/labeler programs, the compact
source runtime and every parameter template/split specification. It excludes
generated per-workload programs, historical/fresh label records, cached feature
packs, teacher weights, temporary checkpoints, `__pycache__` and `.pyc` files.

`validate_release.py` checks every file in `SOURCE_MANIFEST.json`, required
passed CPU/inference/pipeline receipts, selected architecture/parameter counts,
and the absence of hidden checkout imports or absolute developer paths. It
performs no training and no GPU access, making it suitable for the unattended
finalizer. Data/model archive member and LFS verification remain the packaging
tools' separate responsibility.

```bash
# Standard-library-only integrity gate; no Torch import or GPU access.
python -I portable/validate_release.py --source-root . --receipt source-validation.json
```

For assembly, copy each `source` to `destination` from `data_files` and
`portable_files` in `SOURCE_ALLOWLIST.json`, then copy its two `control_files`.
All paths are relative to the staging release root and clean branch root.
The two control files describe the copy without a self-referential hash cycle.
The manifest hashes the allowlist and every other shipped portable file.
