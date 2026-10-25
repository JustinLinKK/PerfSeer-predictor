# PerfSeer: pretrained student release

Available checkpoints: **P_A10, S_A10, S_A100, S_A10G**. Each published model has completed training, lineage and CPU export checks.

Start with [A10_QUICKSTART.md](A10_QUICKSTART.md). For A100 or A10G, use the same `tools/predict_student.py` command with `checkpoints/S_A100/student.ts` or `checkpoints/S_A10G/student.ts` and the matching hardware input-only bundle. Only checkpoints listed above are available in this commit.

Download the matching `perfseer-v1-a100-full.tar.gz` / `perfseer-v1-a10g-full.tar.gz` dataset and `perfseer-v1-s_a100.tar.gz` / `perfseer-v1-s_a10g.tar.gz` model archive with Git LFS; verify and extract with `tools/extract_assets.py`.

Validation and final evaluation are recorded separately in `reports/comparison.json`. Checkpoint selection does not imply every quality threshold passes. A10 has no independent sealed test. The A100 source has A10 ancestry; A10G historical executable/static-feature revisions differ from A10/A100. See the portable/data guides for provenance.

The remaining teacher and transfer models will arrive as later commits on this same branch. Only the selected transfer-student design will be added after its four-way comparison.
