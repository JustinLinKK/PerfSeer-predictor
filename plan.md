# PerfSeer V3 Native A10 Speech-Substitution V2

## Objective and scope

Create `feature/perfseer-v3-nautilus-a10-18k-labeler-speech-v2` from the clean
native-A10 branch. Preserve the existing V1 and original AWS A10G contracts as
historical, reproducible references. Replace only the unavailable 650-row ICML
whale task with a deterministic binary `yes`/`no` view of the TensorFlow Speech
Recognition Challenge. Build and test locally only: do not publish an image or
create, modify, or submit Nautilus resources.

## Versioned data and design contract

- Add a `native_a10_speech_v2` profile and separate V2 task, model, manifest,
  campaign, image, and Kubernetes contracts.
- Keep the V1 native manifest `361f31ed...`, V1 crosswalk `d66c4f09...`, and
  original AWS A10G manifest `bf805655...` reproducible and unchanged.
- Preserve exactly 18,000 candidates, 22 tasks, 35 families, 54,000 retained
  measured epochs, and every family, precision, batch, execution, optimizer,
  scheduler, checkpoint, regime, seed, and coverage quota.
- Replace `icml-2013-whale` at the same task ordinal with
  `tensorflow-speech-yes-no`, using Kaggle competition
  `tensorflow-speech-recognition-challenge` and binary targets `no -> 0` and
  `yes -> 1`.
- Build the prepared view from exactly 2,048 valid 16 kHz WAVs per class,
  ordered by `(SHA-256(relative path), relative path)`. Fail closed on corrupt
  audio, archive drift, or fewer than 2,048 valid examples in either class.
- Use the pinned MLE-bench preparer only to extract the public training corpus.
  Ignore its test split and independently hash and verify the PerfSeer view.
- Preserve the affected 650-row allocation: 275 PANNs, 200 TCN, 175 M5; and
  130 TF32, 195 BF16, 195 FP16 AMP, 130 mixed-structured candidates.

## Lineage and provenance

- Regenerate all 1,300 audio candidate IDs because the three audio-family
  registry fingerprints change; require all 16,700 non-audio IDs to remain
  byte-identical.
- Produce a three-way V2 lineage crosswalk with original AWS A10G, native V1,
  and native V2 IDs; old/new semantic signatures; row classification; and a
  task-independent compute signature.
- Require crosswalk classes of 16,700 `unchanged`, 650
  `audio_registry_rebound` for MLSP Birds, and 650 `dataset_substitution` for
  Speech Commands. Prove all 18,000 task-independent compute signatures match.
- Record both historical IDs, both historical manifest hashes, substitution
  contract hash, source archive hash, remote inventory hash, and
  `dataset_substitution` in every V2 label record. Never silently merge the
  substituted speech measurements with historical whale measurements.
- Store V2 state only in
  `/workspace/perfseer-v3-native-a10-18k-speech-v2`; reject V1 manifests,
  receipts, locks, and workspaces instead of migrating them.

## Runtime, image, and Nautilus handoff

- Keep the unified CLI commands: `analyze`, `image-preflight`, `smoke-local`,
  `run-campaign --pilot`, `run-campaign --chunk-index N
  --max-new-accepted 256`, and `verify --partial|--complete`.
- Preserve the native one-A10 execution, 96-label canonical pilot, ordered
  256-new-label chunks, exclusive CephFS-compatible lock, atomic state/results,
  deterministic OOM repair, GPU cleanup verification, and fail-closed cache
  cleanup.
- Build a separate immutable `linux/amd64` image with the V2 profile baked in,
  the pinned CUDA/PyTorch and MLE-bench revisions, fully pinned dependencies,
  no startup installs, and no credentials or datasets in layers.
- Add V2-specific pilot/chunk Job names while keeping one `NVIDIA-A10`, equal
  requests and limits of 8 CPU and 32 GiB RAM, 16 GiB `/dev/shm`,
  `backoffLimit: 0`, a read-only Kaggle Secret, and a 700 GiB RWX PVC.
- Update the active 22-source gate and runbook to replace ICML Whale with the
  TensorFlow competition. Keep the whale URL only as historical context.
- Do not push an image, access Nautilus, or create any Kubernetes resource.

## Verification and acceptance

- Before real-data testing, require acceptance of the TensorFlow competition
  rules and prove access by downloading the advertised 50-byte
  `link_to_gcp_credits_form.txt`; file listing alone is not sufficient.
- Download the complete competition once, validate its archive safely, and
  persist source archive and remote inventory hashes. Require all later chunks
  to match that immutable source lock.
- Verify historical V1 hashes remain unchanged; V2 counts, distributions,
  affected allocations, ID lineage, and compute signatures are exact.
- Test deterministic filtering, exact 2,048/2,048 balance, corrupt and
  insufficient WAV rejection, source drift rejection, resume behavior, and
  V1/V2 workspace isolation.
- On the RTX 5090, run real one-batch PANNs, TCN, and M5 labels over all four
  precision paths, then a five-real-epoch PANNs/Speech Commands TF32 label with
  epochs 3--5 retained and `production_eligible: false`.
- Re-run the 35-family fixture matrix, focused V1/V2 regressions, secret scan,
  dependency checks, image preflight, `sm_86`/`sm_120` verification, and fully
  offline pilot/chunk YAML validation.
- Rebuild after all source changes so the tested image exactly matches the final
  executable source.
- If the smallest-file probe remains HTTP 403 after the operator accepts the
  rules, record an external blocker. Do not use a mirror or synthetic evidence
  as a substitute for the required real-data acceptance tests.

## Assumptions

- Binary `yes`/`no` is intentional to preserve the historical binary output
  head and compute comparability, despite the standard Speech Commands
  benchmark having more classes.
- Existing V1 evidence and non-production smoke records remain historical.
- No legacy cleanup, registry publication, or Nautilus submission is in scope.
