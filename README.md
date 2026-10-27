# PerfSeer paper release

This snapshot contains one consistent P1/SM0 teacher architecture, T7 transfer,
D3 pretrained students and the validation-selected transfer student **TS2**.
The model transfer route is **A100 → A10**. All three hardware datasets and
transfer data resources are included through verified Git LFS archives.

## Model overview

Eight selected checkpoints are available. Click a model name for its Git LFS
archive; extraction instructions follow below. `P` means full-data teacher,
`S` its distilled student, `T` transfer teacher, and `TS` transfer student.
Transfer names use **target_source**, so `A10_A100` means **A100 → A10**.

All accuracy values below are percentages (higher is better). Each row uses its
final sealed test or transfer holdout when available; full-data A10 has validation
results only. Compare teacher/student pairs on the same hardware and split.

| Model / checkpoint | Short description | Parameters | Evaluation split | Time ordering: wall / GPU | Busy ordering | VRAM within 100 MB: avg / peak |
|---|---|---:|---|---:|---:|---:|
| [P_A100](assets/perfseer-v1-p_a100.tar.gz) | P1/SM0 teacher; unchanged A1001 baseline. | 253.64M | A100 sealed test | 94.77 / 94.80 | 75.36 | 92.20 / 91.83 |
| [S_A100](assets/perfseer-v1-s_a100.tar.gz) | D3 student distilled from P_A100; CPU TorchScript. | 1.64M | A100 sealed test | 94.88 / 94.90 | 75.45 | 82.27 / 82.39 |
| [P_A10G](assets/perfseer-v1-p_a10g.tar.gz) | P1/SM0 teacher continued on full A10G training data (A10G1). | 253.64M | A10G sealed test | 99.73 / 99.73 | 90.21 | 89.72 / 90.82 |
| [S_A10G](assets/perfseer-v1-s_a10g.tar.gz) | D3 student distilled from P_A10G; CPU TorchScript. | 1.64M | A10G sealed test | 99.60 / 99.60 | 90.53 | 82.36 / 84.22 |
| [P_A10](assets/perfseer-v1-p_a10.tar.gz) | P1/SM0 teacher continued on full A10 training data. | 253.64M | A10 validation² | 98.48 / 98.47 | 92.17 | 76.44 / 77.42 |
| [S_A10](assets/perfseer-v1-s_a10.tar.gz) | D3 student distilled from P_A10; CPU TorchScript. | 1.64M | A10 validation² | 98.39 / 98.36 | 92.54 | 78.11 / 78.67 |
| [T_A10_A100](assets/perfseer-v1-t_a10_a100.tar.gz) | T7 transfer: P_A100 → A10, with LoRA and calibration. | 255.69M | A10 transfer holdout¹ | 98.07 / 98.03 | 91.49 | 86.77 / 88.54 |
| [TS_A10_A100](assets/perfseer-v1-ts_a10_a100.tar.gz) | TS2 student: S_A100 warm-start, distilled from T_A10_A100. | 1.64M | A10 transfer holdout¹ | 97.47 / 97.47 | 91.86 | 75.55 / 75.12 |

¹ Transfer holdout is a reserved portion of historical A10 validation, not a new
independent test. ² Full-data A10 validation was also used for checkpoint selection.
The full-data and transfer A10 rows therefore use different evaluation sets.

**Metric definitions.** Time ordering is the fraction of correctly ranked workload
pairs whose measured durations differ by at least 10%. Busy ordering uses pairs
whose measured NVML busy values differ by at least 5 percentage points. Prediction
ties receive half credit. These are ranking scores, not duration-error percentages.
VRAM accuracy is the fraction of workloads within **100 decimal MB** of the raw
measurement, after the released 100 MB rounding of predictions; average and peak
are reported separately. Busy is device active-time percent, not SM occupancy.

**Deployment and designs.** The 1.64M-parameter students include `student.ts` for
CPU inference. Teachers are 253.64M parameters (255.69M for T7) and load through
the portable PyTorch API. Checkpoint archives are about 12.4 MB for students and
940–948 MB for teachers; these are compressed download sizes, not inference RAM.
See the [P1/SM0 implementation](portable/teacher_components.py),
[T7 implementation and loading API](portable/teachers.py),
[teacher training guide](portable/README.md), and
[D3/TS2 recipes and student training guide](student_training/README.md).

Values come from [the complete evaluation report](reports/comparison.json);
parameter counts and CPU loading checks are recorded in
[portable verification](reports/portable-selected-verification.json).
Archive sizes, contents and SHA-256 hashes are in [the asset manifest](assets/manifest.json).

P_A100 is the unchanged A1001 checkpoint; P_A10G is A10G1; S_A100 is D3.
P_A10 fits the full A10 training partition. S_A10G and S_A10 distill their matching
full-data teachers. T_A10_A100 retrains the T7 methodology from P_A100 on the
independently generated A10 transfer design. TS_A10_A100 distills that exact
transfer checkpoint. Four student candidates were compared on validation only;
this snapshot contains the winner's implementation and weights. Their numerical
comparison remains in `reports/transfer-student-selection.json`.

## Data and checkpoints

There are twelve archives: three full labeled datasets, one combined transfer
dataset archive, and eight selected model archives. `assets/manifest.json`
records every archive and member size/SHA-256. `reports/lfs/` records successful
GitHub LFS uploads and remote byte-for-byte verification. The archives are LFS
objects, not ordinary Git blobs.

```bash
git lfs pull
python tools/extract_assets.py --manifest assets/manifest.json --extract extracted
```

If the objects are already cached locally, `git lfs checkout` materializes them.
Extraction verifies both archive and member checksums and refuses to overwrite
different files. Use `--asset ARCHIVE_BASENAME` to extract selected archives.
Output is under `extracted/PerfSeer-release-v1/`: full data at `<GPU>/`, compact
transfer data at `transfer/<GPU>/`, and models at `checkpoints/<MODEL>/`.

All student archives contain `student.ts` for CPU TorchScript inference.
The [A10 quick start](A10_QUICKSTART.md) remains valid after this full update.
Teachers load through the self-contained `portable` package with explicit paths.
See `portable/README.md` for the six-output input contract and training commands.
See `student_training/README.md` for TRAIN-only teacher caches, the selected
student recipes, CPU TorchScript export, and configuration generation from the
extracted assets. These commands do not require the historical research checkout.

## Generate and label fresh workloads

`data/generate.py` constructs workloads from architecture specifications and
source templates. Transfer generation never selects rows or labels from the
pretraining tensor bundle. The frozen group map prevents evaluation overlap.

```bash
python data/generate.py --mode pretrain --output generated_pretrain
python data/generate.py --mode transfer --train-budget 8192 --output generated_transfer
python data/label.py detect
python data/label.py run --generated generated_transfer --output fresh_labels --gpu-lock /tmp/perfseer-gpu.lock
```

The labeler records the detected GPU, executable source hashes, raw timing/NVML
observations, failures and resumable per-workload receipts. Share the GPU lock
with other training jobs. Follow `portable/FRESH_DATA.md` to convert fresh
measurements and static graph features into training bundles. Dependencies and
the tested environment are included with the portable code and data receipts.

The transfer manifest requests 8,192 TRAIN workloads spanning all 32 families.
Historical identity-matched caches contain 7,494 A10, 8,080 A100 and 8,192 A10G
TRAIN labels. Missing or ambiguous labels are retained as missing outcomes;
they are never replaced based on model performance. Evaluation rows are separate.
See `data/README.md` and the archived join receipts for exact coverage.

## Results and limits

`reports/comparison.json` reports measured time-ordering, NVML busy-ordering
and VRAM accuracy. Validation results and final holdout/test results are labeled
separately. Use the per-metric measurements and their units to assess each model.
Scores from different hardware or validation partitions should not be ranked
against each other as if they used the same evaluation set.

The A100 source already has A10 ancestry. Target continuation and transfer are
warm-start adaptation, not scratch-only pretraining or unseen-hardware transfer.
A10 has no independent sealed test; its transfer holdout comes from historical
validation. A10G historical executable sources differ from the A10/A100 revision,
and its inherited static features were captured through the A10 source variant.
These datasets therefore do not establish an isolated hardware-only shift.

Fresh local RTX 5090 labeler checks validate the implementation, not replacement
A10/A10G/A100 measurements. Fresh labels explicitly retain their newer capture
protocol and are not claimed equivalent to historical measurements. NVML busy
is a device busy-time proxy, not SM occupancy.

`RELEASE.json` records exact checkpoint lineage, source snapshot and verification
receipts. This full release is a new commit after the A10-first publication on
the same GitHub `paper-release` branch. Original research runs and reports remain
available in the working research checkout.
