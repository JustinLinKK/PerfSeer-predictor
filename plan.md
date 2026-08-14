# PerfSeer V3 Disaster V2 Single-Workflow Cleanup

## Objective

Make the repository operationally single-purpose: build, publish, submit, monitor,
verify, and export the 11,200-label non-vision four-A10 Disaster V2 campaign. Update
the operator instructions for both required dataset substitutions and remove old
datasets, labels, runs, environments, images, Kubernetes templates, documentation,
and executable workflows. Run no cluster or registry command.

## Safety and retained lineage

- Preserve the current tracked state with local branch
  `backup/pre-disaster-v2-cleanup-20260814` before deletion.
- Keep immutable predecessor hashes and the minimum code/registry metadata required
  to reproduce the active candidate IDs and prove the two substitutions. These are
  active lineage dependencies, not runnable legacy campaigns.
- Keep the current Disaster V2 image, 11,200-row contract, focused verification,
  completed 32-label RTX 5090 verification, active PVC template, renderer, monitor,
  unified CLI, and focused tests.
- Remove ignored bulky data rather than archiving it. It will require regeneration
  or re-download. Tracked material remains recoverable from Git and the backup
  branch.

## Documentation and image organization

- Create one root operator entry point, `NAUTILUS_SUBMISSION.md`, containing the
  complete image-build, GitLab push, digest resolution, Secret/PVC, pilot, immediate
  diagnostics, monitoring, sequential chunk, export, and download workflow.
- Explicitly document both substitutions:
  - ICML Whale -> TensorFlow Speech Recognition (`yes`/`no`).
  - Detecting Insults -> Natural Language Processing with Disaster Tweets.
- Replace the root README with a concise repository map that points to the single
  operator guide and distinguishes future operator commands from actions performed
  during implementation.
- Move the pinned dependency inputs and lock into the active Disaster image
  directory so no active build depends on an old generic A10 container directory.
- Regenerate the build manifest and image after the cleanup because the immutable
  source-tree hash changes.

## Legacy removal

- Remove old V100, AWS A10G, native-A10 V1, Speech-only, and pre-Disaster container,
  Kubernetes, renderer, build-manifest, monitor, runbook, and evidence files.
- Remove old top-level PerfSeer v1/v2/student/source-converter packages, configs,
  evaluation/training scripts, old tests, reports, calibration packs, submission
  helpers, and historical planning documents not needed by the active labeler.
- Remove inactive vision model entrypoint files and the vision adapter entrypoint;
  retain only shared internals required by active non-vision factories and lineage
  verification.
- Remove build outputs, caches, the obsolete CUDA 13 virtual environment, local
  datasets, labels, checkpoints/models, runs, logs, reports, and calibration output.
- Keep `record/` only for current Disaster V2 verification and future Nautilus
  monitor logs.

## Verification

- Verify the active manifest remains exactly 11,200 candidates, 12 tasks, 22
  families, and 33,600 measured epochs with the same manifest hash.
- Verify the substitution crosswalk remains 9,050 unchanged, 1,075 registry rebound,
  and 1,075 dataset substitution rows.
- Run the focused Disaster test suite and offline renderer tests; execute no
  `kubectl`, registry, or Nautilus command.
- Build the cleaned immutable image locally, run `analyze` and RTX 5090 image
  preflight, and verify `sm_86`, `sm_120`, dependency/source hashes, and absence of
  credentials and inactive vision families.
- Validate the canonical guide's referenced local files and commands, scan for
  deprecated competitions outside the explicit lineage section, and leave a clean
  Git worktree.
