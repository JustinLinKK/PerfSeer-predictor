# Correct the Nautilus Kaggle Credential Mount

## Objective

Correct the failed Disaster V2 Job without weakening the credential validator or
changing the labeling/release design. Preserve pod-level `fsGroup: 1000` for the
RWX campaign PVC while ensuring the main labeler sees a private `kaggle.json` with
actual mode `0400`.

## Implementation

1. Preserve the original Kubernetes Secret as a read-only source volume visible
   only to an init container.
2. Add a private memory-backed `emptyDir` for the main container's
   `/run/secrets/kaggle` mount.
3. Have the init container copy only `kaggle.json` into that private volume with
   mode `0400`, then prove its mode before the labeler can start.
4. Apply the same credential-staging contract to continuous, pilot, and chunk Job
   templates; export remains unchanged because it does not use Kaggle credentials.
5. Harden the renderer verifier and tests so a direct group-readable Secret mount,
   missing init copy, writable main credential mount, or unsafe security context is
   rejected.
6. Update the operator runbook with the diagnosed failure and corrected mount
   behavior, then render a replacement continuous manifest pinned to the existing
   verified image digest.

## Verification

1. Run YAML/renderer unit tests, focused continuous/Disaster V2 tests, Python syntax
   checks, renderer negative tests, and `git diff --check`.
2. Verify the rendered Job still requests four A10 GPUs, the approved PVC and
   Secret, equal requests/limits, read-only root filesystems, and the immutable
   published image digest.
3. Locally execute the init-container copy command against a group-readable source
   fixture and prove the staged file is exactly mode `0400` and byte-identical.
4. Commit the correction intentionally and provide copy/paste commands for safe
   Secret metadata verification, rendering, submission, immediate diagnostics, and
   monitor startup.

## Safety

- Do not print or copy credential contents into the repository, image, or logs.
- Do not delete or recreate the existing PVC.
- Do not submit or otherwise mutate Kubernetes resources in this implementation
  turn; the operator will run the final commands.
- Do not rebuild the CUDA image because the failure is entirely in the Kubernetes
  mount design and the existing immutable image remains valid.
