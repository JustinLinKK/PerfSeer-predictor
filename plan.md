# PerfSeer V3 V100 Labeling Image and Repository Cleanup

## Objective

Replace the A10G labeling target with the NRP Tesla V100 SXM2 32 GB target
`nvidia_tesla_v100_sxm2_32gb_nrp`, preserve the 18,000-row V3 workload design,
and collect the explicit audio, tabular, and graph subset with four independent
workers (one process and one V100 per worker). Build and test the immutable
container locally. Do not publish it or call Nautilus from this implementation.

## Contract and runtime

- Preserve all 18,000 family, task, and architecture allocations.
- Translate `fp32_tf32` to `fp32_ieee` and `bf16` to
  `fp16_grad_scaler`; preserve existing FP16 AMP rows.
- Select exactly 1,300 audio, 950 tabular, and 1,300 graph rows for the
  production campaign. Exclude the separate generated modality.
- Regenerate identities and hashes for the V100 hardware/precision contract.
- Require four unique Tesla V100 SXM2 32 GB devices, compute capability 7.0,
  with one candidate child process bound to each GPU through
  `CUDA_VISIBLE_DEVICES`.
- Preserve atomic result writes, deterministic OOM repair/substitution,
  resumability, and fail-closed verification.
- Keep RTX 5090 image-validation results explicitly non-production and
  non-mergeable.

## Container and Nautilus handoff

- Pin the base image to
  `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`.
- Embed pinned PerfSeer source, dependency locks, and MLE-bench revision
  `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`; perform no package install or
  Git clone when a Job starts.
- Render a digest-only Job requesting four generic GPUs with required product
  affinity `Tesla-V100-SXM2-32GB`, CPU request/limit 16/32, memory
  request/limit 64/128 GiB, 16 GiB shared memory, a read-only Kaggle Secret,
  and a 700 GiB RWX PVC.
- Provide a four-label pilot followed by the resumable 3,550-label campaign.
- Document NRP GitLab publication, Secret/PVC preparation, rendering,
  submission, immediate feedback, durable monitoring, and failure response.
  These operations are performed by the user, not by this implementation.

## Verification

- Verify exact 18K and 3,550-row counts, unique identities, precision mapping,
  V100 qualification, four-worker isolation, retries, and atomic state.
- Build the image and verify dependency imports, embedded source hashes,
  absence of credentials, and PyTorch architecture support for `sm_70` and
  `sm_120`.
- Run two five-epoch, non-production labels on the RTX 5090: PANNs CNN14 on
  MLSP Birds with IEEE FP32 and CGCNN on NOMAD with FP16 AMP.
- Run one-batch smoke checks for all ten audio/tabular/graph families.
- Validate the Kubernetes manifest offline. The user's bounded V100 pilot is
  the only real-cluster concurrency proof.

## Cleanup

After the candidate image passes, retain only the V3 18K design and its V100
labeler, container, tests, verifier, focused documentation, and current
evidence. Remove v1/v2 packages, A10 artifacts and scripts, legacy datasets,
labels, runs, reports, generated calibration material, caches, builds, and old
virtual environments. Tracked material is recoverable from
`backup/pre-v100-labeler-20260811`; ignored bulky data is intentionally not
archived. Rebuild and repeat final verification after cleanup.

## External prerequisite

MLSP Birds is download-authorized. The operator must accept or re-accept the
Whale, NYC Taxi, Tabular Playground December 2021, Tabular Playground May 2022,
and NOMAD competition rules with the account that owns the Kaggle API token.
Every competition must pass an actual smallest-file download before labeling.
