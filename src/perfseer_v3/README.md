# PerfSeer V3 Disaster V2 labeler

This package contains the active 11,200-candidate non-vision labeling runtime for
four NVIDIA A10 GPUs. The production controller starts four independent child
workers and pins one visible GPU to each worker; it does not use DDP or NCCL.

The active dataset contract includes two replacements:

- TensorFlow Speech Recognition (`yes`/`no`) replaces the unavailable ICML Whale
  task.
- NLP with Disaster Tweets replaces Detecting Insults in Social Commentary.

Historical IDs, registries, and crosswalk code remain only where they are required
to reproduce the active candidate IDs and prevent measurements with different
dataset semantics from being silently combined.

Use the repository root's `NAUTILUS_SUBMISSION.md` for image publication, local
verification, Nautilus submission, monitoring, chunking, export, and download.
The executable entry point is `scripts/run_perfseer_v3_a10_labeling.py`.
