# Plan

## Active goal: PerfSeer V3 A10G 18K AWS dataset pack

Source of truth: `docs/perfseer_v3_dataset_design_report.md`.

The target is exactly 18,000 successful end-to-end labels measured on AWS
A10G 24 GiB. Each configuration runs once in a fresh process for five complete
epochs. Epochs 1–2 are warmup; the six V3 labels aggregate epochs 3–5. The
finished pack therefore contains 18,000 accepted run records and 54,000
measured epoch records. Local RTX 5090 results verify source trainability and
compiler paths only and never become A10G labels.

No production operation/composite corpus, successful-run repetition,
distributed AWS service, storage ledger, or AWS API integration is in scope.

1. Freeze the deterministic exact 18K manifest and validate every row.
   - Status: complete. The manifest has 18,000 unique configurations, exact
     stable quotas, 35 families, 22 task adapters, and 50 generated lineages.
2. Apply the Light/Standard/Heavy power-of-two batch ladders and deterministic
   OOM repair/replacement rules.
   - Status: complete. OOM descent creates a new configuration ID; replacements
     preserve the frozen quota/source cell; batch/tier changes are audited.
3. Map all 18K rows to factor-complete local execution evidence and run only
   the representative signatures on the RTX 5090.
   - Status: complete. The final locked environment passed 522/522
     representatives, all 18,000 mappings, and read-only reconstruction with
     zero unresolved failures. Every result explicitly rejects A10G
     measurement status.
4. Materialize one Kaggle task at a time with one shared 4,096-example view,
   resumable atomic state, a simple 600 GiB/40 GiB free-space guard, and
   receipt-first cleanup.
   - Status: complete in source and local fixtures. No real Kaggle data was
     downloaded locally.
5. Run one isolated A10G worker per physical GPU using the five-epoch protocol,
   mixed eager/compiled execution, six-target telemetry, stability checks,
   failed-attempt retention, and GPU cleanup proof.
   - Status: complete in source and CPU/local protocol fixtures. No separate
     A10G smoke campaign is required: the parent qualifies each physical A10G
     before the first production attempt and every row uses the same
     fail-closed execution and repair path.
6. Finalize exact accepted records into grouped splits, train-only target
   transforms, concise audits, immutable artifacts, and a receipt-last pack.
   - Status: complete in source and exact 18K/54K fixtures. The finalizer
     verifies all 22 task receipts, complete repair lineage, provenance
     sidecars, exact stable quotas, legal batch deltas, no source/graph split
     leakage, and byte-deterministic output. `--verify-only` is read-only.
7. Collect the production dataset.
   - Status: pending. Process the 22 Kaggle tasks sequentially until exactly
     18,000 records are accepted, run finalization, and independently verify
     the completed pack. No real A10G label is claimed by the current local
     evidence.

## Verification before handoff

- Exact finalization and full dataset-pack suites passed after the final source
  edit: 13/13 and 99/99 tests. The independent verifier reproduced all 112
  tests and 294 subtests.
- The final RTX 5090 gate passed 522/522 representatives, all 18,000 mappings,
  zero unresolved failures, and independent `--verify-only` reconstruction.
- Parse report JSON examples, check counts and epoch roles, search for stale
  active requirements, run Markdown/link checks, compile/import checks, and
  `git diff --check`.
- Obtain a final independent verifier PASS for the commit-ready snapshot.

## Commit and AWS handoff task

1. Add a comprehensive Markdown runbook for the exact instance-local AWS A10G
   labeling and resume/finalization procedure.
   - Status: complete. All 22 Kaggle slugs, 14 shell examples, version pins,
     workflow behavior, resume semantics, output paths, and links were checked
     against the implementation and independently reviewed.
2. Stage only reviewed dataset-pack source, frozen configs/registries, tests,
   documentation, `uv.lock`, and the two concise canonical RTX evidence files.
   - Status: complete. The explicit allowlist stages 132 files; the only staged
     report artifacts are the canonical 522-row results and gate summary.
3. Verify the staged index excludes credentials, datasets, workspaces, caches,
   build output, large validation plan/queue files, stale evidence, and failure
   artifacts.
   - Status: complete. The staged allowlist and credential-signature scan have
     no prohibited hits.
4. Run the documentation, source, lock, test, canonical-evidence, and staged
   diff checks with an independent verifier.
   - Status: complete. All local gates and the independent final staged-index
     review pass.
5. Create one local commit on branch `v2`; do not push.
   - Status: complete. The reviewed allowlist is committed locally on `v2`;
     no push is part of this task.
