# Use the A100 student on CPU

Install Git LFS, then download only S_A100 and the labeled A100 inputs:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch paper-release https://github.com/JustinLinKK/PerfSeer-predictor.git
cd PerfSeer-predictor
git lfs pull --include="assets/perfseer-v1-s_a100.tar.gz,assets/perfseer-v1-a100-full.tar.gz"
python tools/extract_assets.py --manifest assets/manifest.json --extract extracted \
  --asset perfseer-v1-s_a100.tar.gz --asset perfseer-v1-a100-full.tar.gz
```

Use a Python environment with the packages in `portable/requirements-tested.txt`.
The extractor checks archive and member hashes. The code stays in the clone;
weights and data are under `extracted/PerfSeer-release-v1`.

```bash
python tools/predict_student.py \
  --model extracted/PerfSeer-release-v1/checkpoints/S_A100/student.ts \
  --input extracted/PerfSeer-release-v1/A100/data/A100/validation-inputs.pt \
  --output a100-student-predictions.npz
```

The six physical outputs are wall-step ms, GPU-step ms, wall-epoch ms, NVML busy
percent, average VRAM MiB and peak VRAM MiB. Busy percent is not SM occupancy.
Read `reports/comparison.json` for measured time-ordering, busy-ordering and VRAM accuracy, with evaluation splits labeled separately.

The scheduler proposals `oct_proposed` and `oct_proposed_v2` in
[localml_scheduler](https://github.com/JustinLinKK/Impact-Aware-Single-GPU-Scheduling) support this
student through their release adapter. Follow the scheduler branch's integration
guide and point its `student_release_path` to this clone. The default model
directory is `extracted/PerfSeer-release-v1/checkpoints/S_A100`. Source conversion
and inference run on CPU. Keep the model's frozen normalizer and feature runtime
together; the TorchScript graph expects 17 inputs, not the legacy student format.

Fresh workloads require the feature preparation described in
`portable/FRESH_DATA.md`. A different target GPU requires the corresponding model.
To obtain later releases, run `GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only` and
selectively download their additional assets with Git LFS.
