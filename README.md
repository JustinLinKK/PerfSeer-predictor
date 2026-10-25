# PerfSeer paper release

This snapshot contains one consistent P1/SM0 teacher architecture, T7 transfer,
D3 pretrained students and the validation-selected transfer student **TS2**.
The model transfer route is **A100 → A10**. All three hardware datasets and
transfer data resources are included through verified Git LFS archives.

| Family | Released names | Design and training code |
|---|---|---|
| Full-data teacher | P_A10G, P_A10, P_A100 | `portable/teachers.py`, `portable/train_pretrain.py` |
| Pretrained student | S_A10G, S_A10, S_A100 | `student_training/`, PRETRAIN recipe |
| Transfer teacher | T_A10_A100 | `portable/teacher_components.py`, `portable/train_transfer.py` |
| Transfer student | TS_A10_A100 | `student_training/`, selected transfer recipe |

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
