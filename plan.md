# PerfSeer V3 Continuous A10 Campaign Image

## Objective

Modify the current Disaster V2 branch and publish a new immutable image that needs
one campaign Job submission. The image must run the canonical 32-label pilot,
verify it, run production chunks 0 through 43 sequentially, verify all 11,200
labels, create the final release archive, and verify that archive.

## Continuous workflow

- Add a `run-continuous` CLI command that validates the embedded image identity and
  runs the Kaggle gates once per Pod.
- Require the verified pilot receipt before production begins.
- Verify every chunk receipt before advancing and stop on the first failed phase.
- Resume from atomic pilot/chunk/export receipts after the Job's one replacement
  Pod without repeating completed work.
- Emit structured progress to stdout and an atomic PVC progress record.
- Write an immutable terminal receipt only after complete corpus verification and
  successful release reconstruction.

## Nautilus contract

- Render one continuous Job in namespace `ecepxie` using PVC
  `perfseer-panns-jingbin-260808-a0af09` and Secret
  `perfseer-kaggle-disaster-v2`.
- Request four `NVIDIA-A10` GPUs for four independent workers, 32 CPU, 128 GiB RAM,
  32 GiB ephemeral storage, and 32 GiB `/dev/shm`, with equal requests and limits.
- Use a seven-day whole-Job deadline, `backoffLimit: 1`, `restartPolicy: Never`, a
  120-second termination grace period, read-only credentials, and a digest-only
  image.
- Update the durable monitor for replacement Pods and terminal-state exit.
- Run no Kubernetes mutation during implementation; Nautilus access is query-only.

## Verification and publication

- Test phase ordering, pilot gating, failure isolation, resume, terminal fast-exit,
  export ordering, corrupt-state rejection, and the offline Job contract with fake
  phase runners.
- Run only short checks: focused unit tests, `pip check`, `analyze`, RTX 5090 CUDA
  preflight without labeling, CLI startup, and credential/vision/source checks.
- Do not run real five-epoch labels, the real 32-label pilot, production chunks, or
  the 11,200-label campaign.
- Commit the executable source on the current branch, regenerate its build manifest,
  build `linux/amd64` with `--provenance=false`, and push a new full-revision tag to
  `gitlab-registry.nrp-nautilus.io/justinlinkk/prefseer-predictor-labeling`.
- Preserve the previous tag and digest, never publish `latest`, anonymously verify
  the new Docker V2 digest, and render the final continuous YAML locally.

## Acceptance boundary

- One operator Job submission includes pilot, production, complete verification,
  and export.
- The seven-day deadline includes the initial Pod and its single retry. If both Pods
  fail or the deadline expires, durable PVC progress remains but the same manifest
  must be submitted again.
- Actual four-A10 behavior remains the operator-run production acceptance test.
