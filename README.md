# PerfSeer V3 Non-Vision Four-A10 Labeler

This repository contains one active workflow: the 11,200-label, non-vision
PerfSeer V3 campaign for four NVIDIA A10 GPUs on NRP Nautilus.

## Start here

Follow [`NAUTILUS_SUBMISSION.md`](NAUTILUS_SUBMISSION.md) to:

1. accept and verify the 12 Kaggle sources;
2. build and test the immutable `linux/amd64` image;
3. push it to the NRP GitLab Container Registry and resolve its digest;
4. create the Kaggle Secret and 700 GiB CephFS PVC;
5. render, inspect, and submit the four-A10 pilot;
6. monitor the pilot and 44 sequential production chunks; and
7. export and download labels, configurations, and source bundles.

The guide explicitly covers both active substitutions:

- ICML Whale was replaced by TensorFlow Speech Recognition (`yes`/`no`).
- Detecting Insults was replaced by NLP with Disaster Tweets.

## Active repository map

- `containers/a10-nonvision-disaster-v2-labeler/`: pinned image and dependencies.
- `src/perfseer_v3/`: active labeler/runtime plus immutable lineage metadata.
- `scripts/run_perfseer_v3_a10_labeling.py`: unified labeler CLI.
- `scripts/render_a10_disaster_v2_nautilus_job.py`: local digest-only Job renderer and verifier.
- `scripts/monitor_a10_nonvision_job.sh`: required durable Nautilus monitor.
- `k8s/`: active Disaster V2 Job/PVC templates.
- `tests/test_perfseer_v3_a10_disaster_v2.py`: focused contract and substitution tests.
- `record/`: current verification evidence and future monitor logs.

Production uses one Pod with four independent workers, one worker per A10. Candidate
failures are isolated, OOM repair preserves the requested effective batch, and
global integrity failures stop the Job. RTX 5090 results are validation-only and can
never enter the production A10 corpus.

No image publication or Nautilus mutation is performed automatically by this
repository. Submission commands in the operator guide require explicit user action.
