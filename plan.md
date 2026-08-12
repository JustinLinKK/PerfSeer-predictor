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

## Mount Kaggle credentials securely on Nautilus

1. Replace Secret-to-environment injection with a read-only Secret volume that
   exposes only the `kaggle.json` key.
   - Status: complete.
2. Set `KAGGLE_CONFIG_DIR` to the mounted Secret directory and enforce a
   group/other-inaccessible file mode compatible with the fail-closed runner.
   - Status: complete.
3. Update both Nautilus runbooks with the secret creation command and the exact
   two Kaggle competition agreement links.
   - Status: complete.
4. Verify rendered Pod/Job manifests contain only the Secret name, never the
   credential payload, and rerun the focused Nautilus suites.
   - Status: complete. All 19 focused Nautilus tests pass.

## Temporary operator guide for the PANNs CNN14 Nautilus Job

1. Document the namespace trust boundary and make a dedicated namespace the
   required path when Kaggle credentials must be isolated from collaborators.
   - Status: complete.
2. Document collision-resistant names, a create-only immutable Kaggle Secret,
   a unique PVC/workspace, and ownership checks before every cleanup action.
   - Status: complete.
3. Give the operator copy/paste steps for Kaggle agreement, local rendering,
   the bounded pilot, the 550-label production Job, monitoring, verification,
   resume, credential rotation, and exact-resource cleanup.
   - Status: complete.
4. Verify the guide's commands against the existing wrapper, render a sample
   manifest without contacting Kubernetes, scan for credential leakage, and
   run the focused PANNs tests.
   - Status: complete. Bash syntax, guide assertions, dummy immutable-Secret
     transformation, both sample Job renders, the finite audit Pod, and all 19
     focused Nautilus/PANNs tests pass without contacting the cluster.

## Correct the temporary guide for a mandatory shared namespace

1. Replace the dedicated-namespace prerequisite with an explicit accepted
   shared-namespace threat model and truthful limits.
   - Status: complete.
2. Minimize credential exposure with a fresh disposable Kaggle token, a
   randomized create-only immutable Secret, UID/integrity checks, the shortest
   practical lifetime, exact cleanup, and immediate token revocation.
   - Status: complete.
3. Preserve collision avoidance and quota coordination for Jobs, Pods, and the
   per-run PVC without claiming that names or immutability provide privacy.
   - Status: complete. The strict path uses `kubectl create` for the rendered
     pilot and production manifests rather than an apply-based submit action.
4. Re-run guide syntax, manifest, Secret-pipeline, leakage, and focused PANNs
   verifiers without contacting the cluster.
   - Status: complete. Both shared-namespace Job renders, the client-only
     immutable Secret transformation, UID/resource-version guard, guide
     structure/leakage checks, and all 19 focused tests pass.

## Fix the Step 7 manifest verifier environment dependency

1. Replace implicit exported-environment reads with explicit command-line
   arguments for the run ID, namespace, Secret, PVC, and image.
   - Status: complete. The later Secret transformer and audit-Pod verifier were
     made explicit-argument based as well.
2. Run the corrected verifier against the operator's already-rendered pilot
   and production manifests without contacting Kubernetes.
   - Status: complete. Both manifests for `jingbin-260808-005cb8` pass.
3. Re-run shell/Markdown and credential-leak checks for the temporary guide.
   - Status: complete.

## Fail closed instead of falling back to the default namespace

1. Explain that the rejected PVC request created nothing and that an empty or
   lost `NAMESPACE` caused Kubernetes to target `default`.
   - Status: complete from the server error; no Nautilus command will be run.
2. Add a local-only target guard that rejects missing variables, placeholders,
   `default`, and drift between the run ID and resource names.
   - Status: complete.
3. Gate every create/delete operation on that guard and pass the namespace
   explicitly on create-only commands.
   - Status: complete.
4. Verify the guide locally without running any Nautilus or Kubernetes command.
   - Status: complete. The guard rejects unset, `default`, placeholder, and
     mismatched targets and accepts `ecepxie` with the current run identity;
     Markdown shell syntax, diff, and leakage checks pass locally.

## Explain the teammate's Nautilus SSH workflow

1. Read `NAUTILUS-SSH.md` completely and identify the local, Kubernetes, SSH,
   GPU, container, and cleanup stages.
   - Status: complete.
2. Verify the document's commands and assumptions against repository context
   and current official Nautilus user guidance.
   - Status: complete. Read-only live checks matched the documented context,
     namespace, Deployment, image, resource requests, affinity, PVC, bootstrap,
     and readiness sentinel. Current policy does not generally permit a
     long-idle Deployment to request GPUs, so reproducing this exact design
     requires explicit approval.
3. Translate the workflow into an easy, ordered procedure, clearly separating
   one-time setup from the commands repeated for each session.
   - Status: complete.
4. Explain safety checks, expected results, and common failure modes so the
   same workflow can be reproduced without modifying cluster resources during
   this analysis.
   - Status: complete. No Nautilus resource was created, updated, scaled, or
     deleted; all cluster checks were read-only.

## Refactor the guide to request only one A10

1. Remove PVC creation and require an existing namespace-owner-approved RWX
   PVC with enough free space for the workflow.
   - Status: in progress.
2. Make the one-GPU contract explicit: one container, one generic NVIDIA GPU,
   required A10 product affinity, `parallelism: 1`, and sequential pilot then
   production execution.
   - Status: pending.
3. Preserve the unique run workspace without claiming ownership of or deleting
   the shared PVC.
   - Status: pending.
4. Re-run only local guide and manifest verifiers; do not run any Nautilus or
   Kubernetes command.
   - Status: pending.

## Read-only Nautilus A10 availability check

1. Confirm the active Kubernetes context, API reachability, and authorized
   namespaces without changing cluster state.
   - Status: pending.
2. Consult the current NRP Nautilus user documentation for the supported GPU
   labels and recommended read-only availability checks.
   - Status: pending.
3. Inspect visible nodes and current allocatable/requested NVIDIA GPU capacity,
   filtering specifically for A10 or A10G devices.
   - Status: pending.
4. Independently verify the interpretation from raw node labels/capacity and
   current pod requests, then report the time, context, namespace scope, and
   limitations of this point-in-time result.
   - Status: pending.

No Pod or Job will be submitted as part of this check.

## Diagnose the Step 7 PVC DNS-label render failure

1. Inspect the failed local render artifacts and trace the PVC validator.
   - Status: complete. The failed `Jingbin-260808-a0af09` renders are empty,
     and the renderer rejects the derived PVC before emitting YAML because it
     contains an uppercase `J`.
2. Reproduce the failure and verify the corrected lowercase identity by
   rendering both Job manifests locally.
   - Status: complete. The uppercase identity reproduces the exact error with
     empty output; `jingbin-260808-a0af09` renders parseable pilot and
     production Jobs with the expected lowercase PVC claim name.
3. Obtain an independent verifier result and give the operator exact commands
   to rebuild all dependent shell variables and `COMMON_ARGS`.
   - Status: complete. The independent verifier reproduced the exact failure,
     confirmed both lowercase renders and PVC claim names, and verified that
     every name derived from the run ID must be rebuilt together.

No Nautilus or Kubernetes command will be run for this diagnosis.

## Diagnose the Step 9 Secret creation failure

1. Trace the three pipeline errors without exposing Kaggle credentials.
   - Status: complete. `KAGGLE_JSON` expanded to an empty string, so the
     client-side dry run emitted no JSON; the Python and final kubectl errors
     are downstream empty-input failures.
2. Query only Secret metadata/name to confirm whether the failed pipeline
   created an object; do not create, update, or delete anything in Nautilus.
   - Status: complete. A name-only query found no target Secret in `ecepxie`.
3. Reproduce the client-side argument failure locally, verify the safe
   preflight, and obtain an independent verifier result.
   - Status: complete. Empty and nonexistent paths produce distinct errors;
     the independent verifier reproduced both and issued PASS.
4. Give the operator corrected next steps without executing the Secret
   creation command.
   - Status: complete. The handoff restores the absolute credential path and
     validates existence, readability, permissions, and JSON shape locally
     before any user-executed retry.

No Nautilus mutation is authorized for this diagnosis.

## Assess the live PANNs CNN14 pilot status

1. Read the captured status and classify the Job, Pod, scheduling condition,
   monitor process, and immediate event evidence.
   - Status: complete. The capture was taken four seconds after submission;
     the Job was active while its Pod was unscheduled with a transient
     `FailedScheduling` event.
2. Refresh the exact Job, Pod, logs, events, local monitor process/log, and
   persistent output state using read-only checks only.
   - Status: complete. The Job remains active, the Pod is pending without a
     node, the PVC is bound, the Secret name resolves, and the monitor PID and
     minute-spaced log samples are present. No container log exists because
     the Pod has not started.
3. Obtain an independent verifier assessment and identify the next safe
   operator action based on the refreshed state.
   - Status: complete. The verifier agrees this is a capacity wait rather than
     a failed workload: leave the pilot and monitor in place, keep checking,
     and do not submit production until the pilot is `Complete`.

No Nautilus object will be created, updated, or deleted during this status
assessment.

## Re-examine the previous Nautilus PANNs pilot

1. Query the exact pilot Job and Pod conditions, termination state, logs, and
   recent events without mutating the cluster.
   - Status: complete. The Job and Pod no longer exist; no matching object for
     the run ID remains in the namespace.
2. Inspect the tracked local monitor log/process and any persisted completion
   evidence without executing inside the Pod.
   - Status: complete. The durable monitor captured Job `Failed`, Pod `Error`,
     exit code 127, and `/bin/bash: line 9: git: command not found`; there is no
     accepted-record, checkpoint, or completion evidence.
3. Obtain an independent read-only verifier verdict and report whether the
   pilot completed, failed, or remains active, plus the next safe action.
   - Status: complete. The independent verifier confirms the pilot failed
     before labeling. Keep the bound PVC/run ID, do not submit production, and
     verify a digest-pinned PyTorch CUDA image containing `git` before a new
     bounded pilot.

No create, apply, patch, delete, or Pod exec command is authorized.

## Prepare guarded cleanup commands for the failed pilot

1. Query the exact PVC, backing PV/reclaim policy, Secret identity, and live
   consumers without reading Secret data or mutating Nautilus.
   - Status: complete. Exact UID/resource-version identities are recorded,
     the PVC is bound through `rook-cephfs`, the StorageClass currently says
     `Delete`, and no standard Pod/controller/ServiceAccount reference remains.
     Direct PV reads are forbidden by the user's RBAC.
2. Build separate fail-closed operator-executed cleanup blocks with hard-bound
   names, UID preconditions, context/namespace checks, explicit confirmation,
   and post-delete queries.
   - Status: complete. Both shell blocks pass syntax checking and use raw
     Kubernetes `DeleteOptions` with UID and resource-version preconditions.
     The PVC block requires live authorized backing-PV verification and waits
     for both the PVC and PV to disappear.
3. Independently verify the command syntax, deletion scope, irreversible-data
   warning, and evidence that no active workload depends on the targets.
   - Status: complete. The independent verifier refreshed both identities,
     reproduced the no-consumer result, rejected an unsafe combined draft,
     then passed the separated Secret/PVC blocks and raw DELETE form.

Only the user may execute the provided deletion commands; this assessment will
run query-type commands only.

## Evaluate reusing the failed pilot PVC

1. Query whether the exact PVC remains bound and whether any live workload
   currently references it.
   - Status: complete. The claim is `Bound`, 700 Gi, RWX, has no deletion
     timestamp, and no current standard Pod/controller reference.
2. Determine the safe reuse contract for another Pod: namespace, mount mode,
   unique workspace path, single-writer ownership, and provenance/run identity.
   - Status: complete. Reuse is limited to Pods in `ecepxie`; use one writer,
     inspect the old partial prefix before relying on it, and use a distinct
     path/run identity for a different workflow.
3. Obtain an independent read-only verifier verdict and advise whether to keep
   or delete the claim.
   - Status: complete. The verifier recommends conditional reuse and confirms
     the PVC must not be deleted if it will back the replacement Pod.

No PVC content or Nautilus object will be changed during this assessment.

## Recommend a fail-fast Nautilus Job workflow

1. Inspect the retained evidence from the previous failed pilot and identify
   which checks must run before expensive training begins.
   - Status: complete. The A10 queued for roughly three hours, pulled an
     11.7 GB image, and then exited with code 127 because `git` was absent.
2. Define a bounded PyTorch CUDA pilot Job that uses the production image,
   GPU, mounts, credentials, entry point, and workspace but processes only the
   smallest representative unit of work.
   - Status: complete. The recommended gate separates non-GPU image/mount/data
     checks from an exact one-A10, factor-covering end-to-end pilot.
3. Define an objective promotion gate from pilot to production, plus immediate
   feedback and persistent monitoring requirements for both Jobs.
   - Status: complete. Promotion requires Job completion, zero exit, explicit
     receipt verification, expected artifacts, correct A10 identity, and no
     failed preflight; monitoring becomes fast again when the Pod starts.
4. Verify the recommendation against the repository's existing wrappers,
   manifests, tests, and current Nautilus policy without changing any cluster
   resource.
   - Status: complete. Static assertions confirmed the recorded root cause,
     one-attempt pilot, eight-label bound, missing runtime deadline, and current
     60-second/20-minute monitor cadence; all 19 focused Nautilus/PANNs tests
     pass.

## Design a cluster-parity PyTorch CUDA image workflow

1. Inspect the local RTX 5090, Docker/NVIDIA container runtime, repository
   dependency locks, and the exact Nautilus labeling entry point.
   - Status: complete. Docker 28.4 and the NVIDIA runtime execute CUDA on the
     local RTX 5090; the pinned NGC PyTorch container passed a real GPU matrix
     multiplication. The repository already contains a narrower NRP image.
2. Define one immutable PyTorch CUDA image that contains all code, system
   tools, and Python dependencies required by both local validation and the
   Nautilus Job.
   - Status: complete. The image must use one digest-pinned PyTorch/CUDA base,
     install the frozen lock at build time, embed clean PerfSeer and MLE-bench
     revisions, and perform no clone or package installation at Job runtime.
3. Define a local container test ladder that exercises startup, mounts,
   dependency imports, CUDA, and the end-to-end workflow without misclassifying
   RTX 5090 measurements as A10 labels.
   - Status: complete. Local execution uses a distinct non-promotable workspace
     and the existing fail-closed RTX 5090 validation path.
4. Identify the scheduler, Secret/PVC, network, driver, and target-GPU checks
   that cannot be simulated faithfully and must remain in a minimal Nautilus
   pilot.
   - Status: complete.
5. Verify the design against the existing renderer, runner, tests, and current
   official NVIDIA, Docker, and Nautilus guidance; do not build, publish, or
   submit anything during this analysis.
   - Status: complete. The immutable amd64 base identity, local NVIDIA runtime,
     runtime-install drift, 5090/non-A10 provenance gates, and official image
     workflow were checked; all 31 focused local-validation and Nautilus tests
     pass.
