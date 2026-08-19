# Repair Four-GPU Telemetry Binding and Publish a Corrected Image

## Objective

Correct the Disaster V2 pilot failure caused by every child opening NVML physical
GPU zero even when CUDA assigned that child to another physical A10. Preserve the
public labeling and release interfaces, improve safe controller diagnostics, and
publish a verified immutable replacement image without mutating Nautilus.

## Implementation

1. Carry the parent-assigned physical GPU index and UUID into each label child.
   Require exactly one numeric `CUDA_VISIBLE_DEVICES` token, open the matching NVML
   index, and reject any CUDA-count or NVML-UUID mismatch before training.
2. Include the worker's bounded, credential-redacted reason message in global
   controller failures so a terminated Pod remains diagnosable without mounting
   the campaign PVC.
3. Add mocked four-GPU regression coverage for exact index/UUID/process binding,
   malformed assignments, mismatches, and sanitized failure propagation.
4. Move recovery manifests to the fresh PVC directory
   `/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1` and use
   the new `perfseer-v3-a10-nonvision-disaster-v2-nvml-v1-*` Job names. Leave the
   failed workspace and Job untouched.
5. Update the renderer, templates, monitoring instructions, and operator runbook
   while keeping the release archive format and content-hashed filename unchanged.

## Verification and Publication

1. Run focused orchestration/export tests, the full local test suite, syntax
   checks, and `git diff --check`.
2. Run the complete 11,200-candidate construction audit and the 177-case RTX 5090
   fixture matrix with the corrected source.
3. Commit the source intentionally, regenerate the ignored build manifest, and
   build `linux/amd64` from the pinned PyTorch CUDA base with provenance disabled.
4. Verify dependency health, CLI entrypoints, source/build hashes, credential and
   weight absence, CUDA architecture coverage, and the non-production RTX 5090
   preflight.
5. Publish only the full source-revision tag, anonymously inspect its immutable
   digest, record publication evidence, and render/verify the final `ecepxie` Job
   YAML for PVC `perfseer-panns-jingbin-260808-a0af09` and Secret
   `perfseer-kaggle-disaster-v2`.

## Safety

- Perform no `kubectl apply`, delete, exec, temporary Pod creation, or other
  Kubernetes mutation.
- Do not migrate or remove the failed workspace; the fresh workspace reruns the
  small pilot under the corrected image identity.
- Do not put credentials, weights, checkpoints, or automatic S3 upload behavior
  into source, images, manifests, or logs.
