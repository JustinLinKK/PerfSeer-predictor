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

- Existing `.gitignore` already covered `dataset/`, `datasets/raw/`, and
  `datasets/prepared/`, but did not cover top-level generated label/materialized
  outputs such as `label/`, `by_profile_point/`, `index.jsonl`, `summary.json`,
  and `workloads.jsonl`.
- Added root-anchored ignore rules for generated dataset-backed label and
  materialization outputs, plus generated workload specs and local
  profile/validation record folders.
- `git check-ignore -v` confirms these generated paths are ignored:
  `/label/`, `/by_profile_point/`, `/index.jsonl`, `/summary.json`,
  `/workloads.jsonl`, `/scheduler_resource_label.jsonl`, `/label_v3.jsonl`,
  `/dataset_precision*/`, `/precision_dataset*/`,
  `nrp_calibration_pack/workload_specs*/`,
  `record/resource_label_validation_*/`, and `record/smoke_*/`.
- `git check-ignore -v dataset_sources/registry.json` returns no match, so
  dataset source metadata remains trackable.
- `git ls-files -ci --exclude-standard` returns no tracked files hidden by the
  updated ignore rules.
