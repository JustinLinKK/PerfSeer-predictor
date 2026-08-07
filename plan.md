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

## Three-person A10G sharding and budget implementation

1. Freeze one deterministic shard contract for `nlp`, `vision`, or `rest`
   beside the unchanged full 18K manifest.
   - Status: complete. Each shard binds its task order, root IDs, exact row
     count, task-registry hash, and full-manifest hash, and resume rejects a
     different group. The verified partition is NLP 6/5,700, vision 10/6,800,
     and rest 6/5,500, with an exact 22-task/18,000-root union.
2. Run only the selected shard's tasks and rows in its persistent workspace.
   - Status: complete. The existing unsharded command remains the default; a
     separate versioned shard loop is bound to the contract/order hash, and a
     completed shard writes a separate receipt without changing task-receipt
     bytes. The receipt binds clean reviewed source, the exact environment,
     flat semantic artifact closure, repair lineages, and A10G provenance.
3. Merge three completed shard workspaces into one canonical campaign
   workspace and invoke the unchanged exact-18K finalizer.
   - Status: complete. Merge rejects missing/duplicate groups, overlapping
     roots or artifacts, environment/manifest/source drift, incomplete task
     receipts, and any non-exact union. It copies only receipt-bound evidence
     into a hash-bound sibling staging workspace, resumes only an identical
     journal, atomically publishes, then re-verifies the unchanged final pack.
4. Document the Ohio `g5.12xlarge` three-person budget, persistent gp3 setup,
   pilot gate, Spot refresh/fallback policy, per-person caps, and merge commands.
   - Status: complete. The USD 5,500 expected authorization, USD 11,000
     conservative authorization, exact owner caps, current planning inputs,
     commands, pilot gate, and strict merge procedure are documented without
     AWS provisioning code or credential material.
5. Verify shard selection, resume safety, interruption idempotence, strict
   merge behavior, exact finalization, CLI compatibility, documentation, and
   the complete existing test suite with an independent verifier.
   - Status: complete. The original 112 tests pass; 16 focused shard tests
     pass; and the real 18K integration test passed three shard finalizations,
     tamper rejection, interrupted-merge resume, atomic publication, exact
     18,000/54,000 finalization, and verify-only replay. The independent
     verifier issued PASS after reproducing focused checks.

## Single-family Nautilus labeling: PANNs CNN14

1. Inspect the repository instructions, the PANNs CNN14 entry in the A10G 18K
   dataset pack, and the existing Nautilus labeling tools.
   - Status: complete.
2. Check the current Nautilus user documentation for namespace, GPU, storage,
   image, and Job requirements relevant to this workload.
   - Status: complete.
3. Prepare a single-model PyTorch CUDA Job manifest plus submit, monitor, and
   verify scripts. The submit path must collect immediate Kubernetes feedback
   and start the required persistent monitor.
   - Status: complete.
4. Verify that the selected family contains exactly 550 configurations,
   validate the rendered manifest, exercise submission with a fake `kubectl`
   where possible, and run focused tests.
   - Status: complete. The 116-test staged snapshot passes, followed by the
     seven-test PANNs suite with two exact-completion cases.
5. Give the user exact prerequisites, commands, expected files, and
   troubleshooting steps. Do not contact the cluster until the prepared
   artifacts have passed their local checks.
   - Status: complete. The operator runbook contains the pilot, production,
     monitoring, completion, resume, and cleanup procedure.

## Commit the PANNs Nautilus workflow

1. Trace the exact runtime and test dependency closure from the PANNs wrapper.
2. Exclude transfer-learning documentation, reports, and unrelated encoder
   wording from the commit.
3. Stage only an explicit reviewed allowlist; do not use broad Git add rules.
4. Verify the staged source, manifests, shell syntax, tests, secrets scan, and
   staged diff before committing.
5. Create one local commit without pushing, then verify its tree and report all
   remaining worktree changes.
