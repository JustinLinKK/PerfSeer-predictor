# PerfSeer: A10 testing release

This release adds **P_A10**, the full-data A10 teacher, and **S_A10**,
its D3 student with a verified CPU TorchScript export. Both finished their full
training budgets (30 and 40 epochs respectively). The A10 dataset is included
through Git LFS. The remaining hardware and transfer models will arrive in a
later commit on this same `paper-release` branch. The previously published
**S_A100** student and full A100 dataset remain available; see
[A100_QUICKSTART.md](A100_QUICKSTART.md).

Start with [A10_QUICKSTART.md](A10_QUICKSTART.md). It explains selective downloads,
checksum-verified extraction, and CPU inference with the teacher or student.
Predictions target A10 workloads even though inference runs on CPU.

`reports/comparison.json` contains the validation results. A10 has no independent
sealed test. Selection used validation only; the best checkpoint is not a claim
that every timing, busy-ordering and VRAM quality threshold passed.

P_A10 uses the SM0/P1 architecture, initialized from the immutable A1001 source.
That source already has A10 ancestry. This is full-data target adaptation, not
scratch-only pretraining or unseen-hardware transfer. S_A10 distills that exact
P_A10 checkpoint with the established D3 recipe.

`portable/` contains the teacher implementation, training recipes and frozen
feature runtime. `student_training/` contains only the pretrained D3 student
design at this milestone. No unselected transfer-student design is published.
`data/` contains the direct workload generator and new GPU labeler; follow
`portable/FRESH_DATA.md` for fresh data capture and feature packing. Historical
datasets and new local measurements have different protocols and retain their
own provenance.

`assets/manifest.json` lists archive and member hashes. `reports/lfs/` records
successful upload and remote hash checks. `RELEASE.json` identifies this as the
A10 addition milestone and lists pending release components. The later complete
release will retain this commit in its history.
