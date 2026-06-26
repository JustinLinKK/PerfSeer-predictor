# Plan: Ignore Generated Dataset And Label Artifacts

## Goal

Update `.gitignore` so generated datasets, labels, workload indexes, and local
profile/validation outputs do not get accidentally tracked.

## Steps And Verifiers

1. Inspect current ignore coverage.
   - Check existing `.gitignore`, current dirty/untracked paths, and generated
     dataset/label directories.
   - Verifier: identify concrete paths that are currently untracked and should
     be ignored.

2. Patch `.gitignore`.
   - Keep source metadata such as `dataset_sources/` trackable.
   - Ignore generated roots such as `label/`, `by_profile_point/`,
     workload/index JSONL outputs, generated workload specs, and local profiling
     result folders.
   - Verifier: `git check-ignore -v` reports the new rules for representative
     generated paths.

3. Run final checks.
   - `git diff --check`.
   - `git status --short --ignored` for representative paths.

## Completed Verifiers

- Pending.
