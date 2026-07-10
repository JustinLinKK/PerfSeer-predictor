# Plan

1. Inventory teacher/student configs, docs, scripts, and tests that still point at legacy six-target training.
2. Add a canonical v2 training target that uses measured epoch time from `scheduler_label_v3.jsonl` plus resource labels from `scheduler_resource_label.jsonl`.
3. Keep exactly one non-legacy teacher architecture and one non-legacy student architecture for per-hardware model pairs.
4. Move older architecture configs under `configs/legacy/` with explicit warnings so teammates do not pick them accidentally.
5. Update `run_hardware_distill_flow.py` and README commands to default to the canonical v2 pair.
6. Verify with dry-run commands, targeted unit tests, stale-reference searches, and `git diff --check`.
