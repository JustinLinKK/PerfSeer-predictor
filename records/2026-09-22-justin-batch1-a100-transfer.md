# Justin-Linux batch-1 A100 transfer campaign

## Source discovery and compatibility

The imported Justin-Linux archive is
`/home/justin/PerfSeer-predictor/record/perfseer-v32/transfer-rtx5090`.
Its RTX 5090 labels contain 4,128 completed records with the same twelve
physical target names, 248 executable anchors, nine source materials, and six
modalities as the current V3.2 transfer stack.  The relevant new family is its
992 batch-1 configurations: 740 train, 128 validation, and 124 test, over all
248 anchors.  The existing A100 label campaign has batch sizes 8, 16, 32, 64,
and 128, but no batch-1 configurations.

The four source files were copied with `scp` and checksum-verified into
`.cache/justin-batch1-source-20260922/`.  RTX values are provenance-only and
are never emitted as A100 Ground Truth targets.

## Staged A100 campaign

`scripts/prepare_perfseer_v32_justin_batch1_transfer.py` selects only verified
batch-1 records, preserves split/group/anchor lineage, and rekeys each
configuration with the A100 hardware identity.  It produced
`.cache/justin-batch1-a100-campaign-20260922/` with 992 configurations and a
checksummed A100 label manifest.  The matching files are staged remotely at
`/data1/yufan/perfseer_v32_a100_transfer_20260920/.cache/justin-batch1-transfer-20260922/`.

GPU 0 was owned by an unrelated Chemprop process at staging time, so no
labeling or training process was started and no foreign workload was altered.

## Mixed-corpus implementation and verification

The campaign now has a non-destructive queued one-workload pilot.  It remains
live but does not start until GPU 0 is exclusively idle; a later foreign
process still owns that device, so the pilot has retained zero attempts.

`scripts/prepare_perfseer_v32_justin_batch1_transfer.py` now additionally
materializes paired V3.2 graphs only from newly captured A100 traces, updates
the inference graph hardware metadata to A100, verifies the resulting
twelve-target dataset, and makes an immutable mixed corpus.  The mixture takes
all batch-1 rows and a deterministic stratified 3:1 subset of the original
A100 corpus per split: 740 + 2,220 train, 128 + 384 validation, and 124 + 372
test.  Thus 992 of 3,840 rows (25.8%) are new batch-1 data while the original
batch, modality, model, and operation diversity is retained.  Original paired
graphs are hard-linked rather than copied.

The local deterministic selection/controller tests pass (11 tests), and the
synchronized script has verified the current original A100 dataset
fingerprint `a8cfd0ce16788478649530b5ceca30b4bc464eba1d5a5dca8797ca7f5968f1d5`
over 11,205 rows before it will be used as a mixture source.

## Queued end-to-end execution

The remote `nohup` controller
`scripts/continue_justin_batch1_a100_pipeline.sh` is live alongside the pilot.
It checks that the pilot has a successful A100 attempt before it performs the
full campaign.  Each GPU-dependent stage retries only on an explicit A100
contention error and resumes from its verified checkpoint; any other failure
ends the controller without silently changing the experiment.  The prepared
student launcher caches each frozen teacher output exactly once, then trains a
six-head, twelve-output student for 20 epochs.  The 30-epoch teacher transfer
and student both retain their validation predictions and evaluate the selected
checkpoint on the held-out Ground Truth test split.

## Pilot correction and full-label launch

The first pilot reached an otherwise idle A100 but failed before capture because
the launcher exposed both physical GPUs, while the profiler correctly requires
exactly one visible device.  A minimal visibility test showed that
`CUDA_VISIBLE_DEVICES=0` exposes one ABA A100, and the launch scripts were
updated accordingly.  The corrected pilot completed configuration
`3e154676b65d02a6f3891ff2aea156722975ba1aea8f95557723b7a1a0061259`
with an A100 graph and all twelve physical targets.  The full resumable
992-workload labeling process then started on GPU 0 from that completed pilot;
the first remaining workload was active at launch.  The initial pipeline
startup race against the stale failed attempt was also corrected by checking
the live pilot process before reading attempt state.

## Next required actions

After GPU 0 is verified free: label the 992 batch-1 workloads on A100, verify
all 12 Ground Truth labels, create a deterministic mixed corpus containing the
new family plus a stratified existing-A100 subset, train the teacher, then
distill and verify the CPU-oriented student.
