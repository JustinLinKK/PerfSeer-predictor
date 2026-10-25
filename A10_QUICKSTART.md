# Test the A10 teacher and student

Both predictors estimate A10 training costs. Inference runs on CPU; an A10 GPU
is needed only to collect new measured workloads for comparison.

## Download the A10 assets

Install Git LFS and clone this branch without downloading other hardware assets:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch paper-release https://github.com/JustinLinKK/PerfSeer-predictor.git
cd PerfSeer-predictor
git lfs pull --include="assets/perfseer-v1-a10-full.tar.gz,assets/perfseer-v1-p_a10.tar.gz,assets/perfseer-v1-s_a10.tar.gz"
python tools/extract_assets.py --manifest assets/manifest.json --extract extracted \
  --asset perfseer-v1-a10-full.tar.gz \
  --asset perfseer-v1-p_a10.tar.gz \
  --asset perfseer-v1-s_a10.tar.gz
```

The extractor verifies both archive and member checksums. The extracted root is
`extracted/PerfSeer-release-v1`. The A10 dataset expands to several GB; leave space
for its extracted tensors and teacher weights.

## Run CPU predictions

Use a Python environment with PyTorch, NumPy, SciPy and PyYAML. The tested versions
are listed in `portable/requirements-tested.txt`; the student export is fp32
TorchScript. No model hub, training service or API key is needed.

Student:

```bash
python tools/predict_student.py \
  --model extracted/PerfSeer-release-v1/checkpoints/S_A10/student.ts \
  --input extracted/PerfSeer-release-v1/A10/data/A10/validation-inputs.pt \
  --output a10-student-predictions.npz
```

Teacher:

```bash
python -I portable/infer.py --kind Adapted \
  --checkpoint extracted/PerfSeer-release-v1/checkpoints/P_A10/best_model.pt \
  --architecture portable/architecture.pt \
  --input extracted/PerfSeer-release-v1/A10/data/A10/validation-inputs.pt \
  --output a10-teacher-predictions.npz
```

These commands consume input-only features. They do not train, select a new
checkpoint or evaluate sealed tests. The NPZ output retains sample IDs, target
names and units. Its six prediction columns are wall-step ms, GPU-step ms,
wall-epoch ms, NVML busy percent, average VRAM MiB and peak VRAM MiB.
NVML busy is a device busy-time proxy, not SM occupancy. Read the model's
validation report in `reports/comparison.json` before relying on its accuracy.
A10 has no independent sealed test and passing export checks does not mean all
quality thresholds passed.

For your own workloads, follow `portable/FRESH_DATA.md`: generate executable
workloads, measure them on the actual A10 with `data/label.py`, and pack the static
features using the frozen normalizer. Never substitute another GPU name for the
detected hardware. Existing A10 predictions are not automatically calibrated to
another GPU.

## Receive the remaining models later

The same branch receives a subsequent commit after the remaining training and
transfer-student selection finish:

```bash
GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only
```

Download additional archives explicitly with `git lfs pull --include=...`.
This A10 quick start and the selected A10 checkpoint hashes remain available.
