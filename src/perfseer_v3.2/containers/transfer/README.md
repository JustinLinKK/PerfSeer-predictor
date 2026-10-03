# RTX 5090 teacher transfer and student distillation on Nautilus A100

This package fine-tunes the local epoch-26 T1 teacher on 4,128 completed RTX 5090
configurations, then distills a fresh S1 student after the teacher passes both
existing quality gates. The A100 runs predictor training; all prediction labels
and graph hardware profiles remain `nvidia_geforce_rtx_5090_32gb_local`.

The package includes the exact pretrained teacher checkpoint, source code,
native labels and training graphs, 4,256 raw measurement attempts, deterministic
input provenance, prepared paired graphs, checksums, and submission scripts.
It does not require the original public source archives or remeasure workloads.
New inference graphs are CPU captures from the saved model definitions and
deterministic input material, with eager/replay checks. The original measured
training graphs and all twelve native labels are unchanged. CPU capture does not
establish CUDA numerical parity or model accuracy.

## Verify and submit

Run from the extracted package directory. Verification requires Python 3.11+;
submission additionally requires Bash, an authenticated Nautilus `kubectl`, and
Docker with registry credentials when building/pushing an image.

```bash
python3 verify_package.py
sha256sum -c SHA256SUMS

# Render manifests locally, without Docker or cluster changes.
bash submit.sh --dry-run --job perfseer-rtx5090-review

# Build this package into the pinned PyTorch CUDA image, push, resolve its digest,
# and submit one A100 80 GB Job in ecepxie.
bash submit.sh --image-tag \
  gitlab-registry.nrp-nautilus.io/justinlinkk/prefseer-predictor-labeling:rtx5090-transfer-20260916
```

To use an already published package image, pass `--image registry/repository@sha256:DIGEST`
instead. `--namespace NAME` selects another authorized namespace. The repository
above is an example based on this project's existing registry; change it if needed.
`--server-dry-run` checks API admission without creating resources or publishing an
image. With no `--image`, dry runs use a deliberately nonexistent example digest;
they do not establish image pullability or GPU availability.

The script requests `nvidia.com/a100: 1`, at least 80 GB GPU memory, 8 CPU cores,
24 GiB RAM, and a dedicated 40 GiB CephFS PVC. A100 access/quota must be enabled
in your namespace. It follows the Nautilus
[GPU resource guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
and [batch Job guidance](https://nrp.ai/documentation/userdocs/running/jobs/).
There is a 48-hour active deadline and two retries; training checkpoints preserve
optimizer, scheduler, RNG, and batch cursor. Completed Jobs release their GPUs.

## Training protocol and results

- Preserve the original architecture-grouped split: 2,988 train / 576 validation /
  564 test rows across 119 groups. No architecture group crosses splits.
- Start T1 from epoch 26 of the supplied A10 checkpoint. Its source label policy
  is recorded in transfer lineage; RTX 5090 targets remain native measurements.
- Retain pretrained normalization numbers and output scales; only dataset-binding
  metadata changes. No target validation/test statistics are fitted. This can clip
  target workloads outside the base training ranges; it is a fixed transfer baseline.
- Fine-tune all teacher parameters with Muon/AdamW, teacher LR `1e-4`, 100 epochs,
  effective predictor batch 64, and memory-probed microbatches up to 4. Teacher
  activation checkpointing is enabled. Early stopping uses patience 10 from epoch 20.
- Select by existing worst/mean 5% and 10% hit rates. Require at least 95% of rows
  within 5% relative error on **every** target on validation and then test.
  A failed teacher gate stops before student training.
- Freeze the accepted teacher and distill a freshly initialized S1 for 100 epochs
  with the existing LR `1e-3` and loss `0.6 hard + 0.4 soft + 0.05 relation`.
  Preserve the same student validation/test gates. Test evaluation is bound to the
  selected checkpoint and cached; it is not used for epoch selection.

Use `--teacher-epochs`, `--student-epochs`, `--learning-rate` (teacher),
`--effective-batch`, and `--microbatch` to set a campaign before its first launch.
Resume rejects changed dataset/checkpoint/code/settings. Adaptation success is not
assumed: `transfer-report.json` reports `teacher_gate_failed`, `student_gate_failed`,
or `accepted`. Candidate exports are separate from `student-rtx5090-accepted.pt`.

## Monitoring, outputs, and resume

Immediately after submission the script prints Job/Pod status, descriptions, logs,
and events. Its detached `nohup` monitor writes to `record/JOB/monitor.log` under the
repository root, or under the package root outside a Git checkout, every minute
for the first five minutes and every twenty minutes afterward.
It records process state, persistent training-log tails, checkpoint files, and
failure exit codes. Run `tail -f record/JOB/monitor.log` for feedback.

All training outputs live on the printed PVC under `/run`: `training.log`, base
validation, teacher/student checkpoints, gates, prediction exports, and final report.
The script never deletes Jobs or PVCs. To resume after a failed/preempted Job, use a
new Job name and the **same image digest, settings, and printed PVC** via `--pvc NAME`.
It refuses a PVC mounted by an active Pod. Review failure logs before retrying;
automatic retries cannot fix deterministic data or resource errors.

## Rebuild from the repository

```bash
python scripts/run_perfseer_v32_transfer_training.py prepare \
  --source record/perfseer-v32/transfer-rtx5090 \
  --dataset record/perfseer-v32/transfer-training-rtx5090/dataset --workers 6
python scripts/package_perfseer_v32_transfer.py
```

Preparation and packaging do not publish an image or submit training. To verify
the runtime locally, use the repository's prepared Python interpreter to run
`verify_package.py --runtime`; this checks a real graph pair with the full teacher
on CPU. Docker building performs the same check inside the exact CUDA image.
