# Repair Replacement Optimizer Contracts and Publish a Corrected Image

## Objective

Repair the deterministic Disaster V2 `chunk-04` failure in which quota-replacement
generation produced a candidate rejected by the optimizer-parameter compatibility
policy. Preserve the 1,056 accepted labels and all public campaign/export interfaces,
prove that every reachable replacement remains valid, and publish a new immutable
PyTorch CUDA image without mutating or resubmitting a Nautilus Job.

## Diagnosis and Implementation

1. Reproduce the failing replacement locally from persisted planning identities and
   inspect the optimizer, model-family, retry, and mutation inputs that created it.
2. Correct replacement generation so architecture mutations normalize the preserved
   optimizer-specific parameter contract before candidate validation.
3. Keep deterministic candidate IDs, quota accounting, retry ancestry, campaign
   identity, label schema, export schema, and CLI arguments unchanged.
4. Make invalid repair proposals recoverable within the deterministic replacement
   search instead of allowing one proposal to abort the global four-GPU campaign.
5. Add regression tests for the exact production failure and every optimizer/family
   combination, including all production replacement indices.
6. Add a fail-closed, append-only recovery identity transition for the exact failed
   image so the corrected image can revalidate and retain its 1,056 accepted records.
   Require an explicit environment authorization containing the frozen prior identity
   hash, an unchanged campaign/crosswalk contract, and exact old/new identity chains.
7. Preserve each accepted record's original build/environment fingerprints and store
   every recovery environment as a content-addressed provenance sidecar; do not rewrite
   the frozen origin identity or any accepted label.
8. Isolate candidate-scoped planning and execution failures: durably mark the failed
   attempt, continue unrelated candidate rows, and route the unresolved quota slot
   through deterministic replacement. Reserve controller termination for corrupted
   shared state, invalid global provenance, or a slot that exhausts its bounded repair
   budget; never count a failed row as one of the 11,200 accepted labels.

## Verification

1. Run focused sampler, repair, workflow, campaign, continuous, and exporter tests.
2. Run the complete local test suite, Python and shell syntax checks, YAML parsing,
   `git diff --check`, and tracked credential/weight/checkpoint scans.
3. Re-run the 11,200 root-candidate construction audit and add an exhaustive audit
   that generates and validates all reachable quota replacements across the full
   Disaster V2 plan, optimizer/family combinations, and supported replacement indices.
4. Re-run the 177-case RTX 5090 fixture matrix and the available non-labeling GPU
   preflight to ensure the repair does not regress hardware binding or telemetry.
5. Test recovery against a realistic frozen workspace: correct authorization resumes,
   missing/wrong authorization fails before training, altered contracts and malformed
   histories fail closed, and repeated resumes cannot append or reorder identities.
6. Inject candidate-local planning and execution failures into multi-row batches and
   prove later rows execute, failure artifacts remain durable, replacement fills the
   original quota slot, and no terminal receipt/archive is emitted while any slot is
   unresolved.

## Publication

1. Commit the verified source intentionally and regenerate the ignored build manifest.
2. Build `linux/amd64` from the pinned PyTorch CUDA base with provenance disabled.
3. Verify `pip check`, `analyze`, CLI help, embedded source/build hashes, credential
   and weight absence, required CUDA architectures, and the non-labeling RTX 5090
   preflight inside the built image.
4. Publish only the full source-revision tag, resolve and anonymously inspect the
   immutable registry digest, record publication evidence, and render/verify a new
   recovery Job manifest that uses the existing PVC and Kaggle Secret.

## Safety and Recovery

- Perform no `kubectl apply`, delete, exec, temporary Pod creation, or other cluster
  mutation. Cluster access during this work is read-only.
- Preserve the failed Job and PVC workspace. The corrected continuous runner must
  revalidate and reuse the existing 1,056 accepted labels before continuing.
- Do not include credentials, weights, checkpoints, or automatic upload behavior.
- Do not claim the release exists until all 11,200 labels verify and export succeeds.
