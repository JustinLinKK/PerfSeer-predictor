# Harden the Downloadable Source-and-Label Release

## Objective

Keep the continuous 11,200-label workflow and its reconstructable release format,
while making archive verification strict enough to prove that every accepted label,
candidate configuration, and content-addressed Python source bundle is complete,
safe, and mutually consistent. Publish the verified change as a new immutable
PyTorch CUDA image, but do not submit a Nautilus Job.

## Implementation

1. Harden `a10_export.py` without changing the CLI or release schema/version:
   require complete counts, unique candidate identities, contiguous label joins,
   safe relative paths, exact configuration joins, recomputed bundle/file hashes,
   exhaustive `SHA256SUMS`, and rejection of unsafe or forbidden members.
2. Preserve deterministic atomic `.tar.zst` creation and its JSON checksum sidecar
   under the PVC-backed campaign `releases/` directory.
3. Preserve continuous orchestration ordering: export only after 11,200-label
   verification, verify before writing the terminal receipt, and reverify on resume.
4. Add focused integration and corruption tests for the real exporter/verifier and
   retain the continuous orchestration tests.
5. Update the operator handoff to document the terminal receipt, archive and sidecar,
   NRP S3/rclone transfer, checksum comparison, and local reconstruction verification.

## Verification and publication

1. Run a verifier after each edit: focused tests, syntax/static checks, archive
   round trips, broader Disaster V2 tests, and `git diff --check`.
2. Commit the executable change, regenerate its build manifest, and build the pinned
   `linux/amd64` PyTorch CUDA image with provenance disabled.
3. Verify dependency health, campaign analysis, CLI startup, credential absence,
   source/build hashes, CUDA architectures, and the non-labeling RTX 5090 preflight.
4. Publish only a full-revision tag, anonymously resolve the immutable Docker V2
   digest, record publication evidence, and render/verify the continuous Job for
   namespace `ecepxie`, PVC `perfseer-panns-jingbin-260808-a0af09`, and Secret
   `perfseer-kaggle-disaster-v2`.
5. Do not run real labels and do not create, apply, patch, or delete any Kubernetes
   resource.

## Fixed artifact contract

- Source form: deduplicated reconstructable Python factory/runtime bundles plus one
  canonical JSON configuration and joined accepted label per model.
- Destination: the campaign PVC under `<workspace>/releases/`.
- Compression: deterministic content-hashed `.tar.zst` plus matching JSON sidecar.
- Exclusions: no trained weights, checkpoints, automatic S3 upload, or credentials.
