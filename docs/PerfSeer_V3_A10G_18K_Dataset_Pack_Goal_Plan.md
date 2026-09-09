# PerfSeer V3 A10G 18K dataset-pack goal

This short goal file replaces the obsolete repeated-run/distributed-cloud
design. The decision-complete implementation contract and current evidence are
maintained in [perfseer_v3_dataset_design_report.md](perfseer_v3_dataset_design_report.md).

## Required outcome

- Produce exactly 18,000 successful end-to-end labels on
  `nvidia_a10g_24gb_aws_g5`.
- Run each accepted configuration once in a fresh process for five epochs.
- Use epochs 1–2 only as warmup and aggregate the six V3 outputs over epochs
  3–5.
- Retain exactly 18,000 accepted records and 54,000 measured epoch subrecords.
- Never use RTX 5090 measurements as A10G labels.

## Local gate

Statically validate all 18,000 rows, group them by executable behavior, and run
only the factor-complete representative signatures on the local RTX 5090. Each
representative must build real-format fixture data, run eager reference work,
compile, complete two compiled training updates, and pass finite-state, shape,
parameter-update, optimizer, scheduler, and eager/compiled-equivalence checks.
Every manifest row must map to passing evidence.

This gate proves source trainability and compiler-path compatibility. It does
not prove A10G memory fit or provide target labels.

## AWS workflow

Use one resumable instance-local command. Before collection it validates the
locked environment, pinned MLE-bench checkout, Kaggle credentials, rules access,
and paginated inventory for all 22 tasks. It then processes one task at a time:
download, verify, prepare one shared view, label its rows with one isolated
worker per A10G, save results atomically, clean reconstructible task/compiler
caches, and continue.

OOM repair creates a new configuration with the next smaller power-of-two
batch. A batch-one OOM is quarantined and replaced in the same quota cell.
Measured epoch instability is quarantined and replaced. Dependency, compiler,
data-integrity, unexpected child, and cleanup failures stop the campaign for
operator correction instead of silently consuming replacements.

No S3, SQS, DynamoDB, Spot controller, distributed lease service, storage
ledger, successful-run repetition, or production operation/composite corpus is
part of this revision. Operation and composite code remains local QA only.

## Completion

After all 22 tasks finish, require exactly 18,000 unique accepted configuration
IDs, preserve exact quotas after linked repairs, build grouped leakage-safe
splits, fit target transforms on training rows only, keep failures outside the
successful manifest, and independently verify the deterministic final pack.
