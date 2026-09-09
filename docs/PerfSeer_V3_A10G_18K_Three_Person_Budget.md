# PerfSeer V3 A10G 18K three-person budget and execution plan

This report is the cost and ownership companion to the
[AWS labeling runbook](PerfSeer_V3_A10G_18K_AWS_Labeling_Runbook.md). Prices
are planning inputs as of 2026-07-31, not AWS quotes. Refresh Spot history and
the AWS Pricing Calculator immediately before launch.

## Decision summary

- Region baseline: `us-east-2` (Ohio).
- Instance: one Linux `g5.12xlarge` per person.
- Parallelism: four independent single-A10G workers per instance. Never split
  one configuration across GPUs.
- Persistent workspace: one 700 GiB gp3 volume per shard, with approximately
  6,000 IOPS and 500 MiB/s, and `DeleteOnTermination` disabled.
- Expected mixed Spot/On-Demand authorization: **USD 5,500**.
- Conservative 30-minute authorization: **USD 11,000**.
- On-Demand-only planning budget: **USD 8,500**.
- On-Demand-only conservative ceiling: **USD 16,500**.
- Planning calendar: 22–26 days; vision is the expected critical path.

AWS lists `g5.12xlarge` with four 24 GB A10G GPUs, 48 vCPUs, 192 GiB host
memory, and one 3,800 GB NVMe device. The campaign uses gp3 for authoritative
state because instance-store data does not survive termination. See the
[AWS G5 specification](https://aws.amazon.com/ec2/instance-types/g5/) and
[EC2 instance-store lifetime](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-lifetime.html).

## Frozen allocation

The split is by the Kaggle task's source modality, not by the quota label.
Generated architectures therefore follow the real task adapter they execute.

| Owner | Task group | Tasks | Accepted roots | Five-epoch batch iterations | Compiled roots | Compressed downloads |
|---|---|---:|---:|---:|---:|---:|
| Person 1 | `nlp` | 6 | 5,700 | 22,994,960 | 1,684 | 0.087 GB |
| Person 2 | `vision` | 10 | 6,800 | 26,539,720 | 2,016 | 149.79 GB |
| Person 3 | `rest` (audio, graph, tabular) | 6 | 5,500 | 17,800,400 | 1,667 | 7.85 GB |

The vision total includes the 116.16 GB SIIM-ISIC archive. Archives are still
downloaded, prepared, receipted, and deleted one task at a time, so cumulative
download volume is not simultaneous disk usage.

The three source-generated shard contracts must report exactly:

```text
nlp:     6 tasks,  5,700 roots
vision: 10 tasks,  6,800 roots
rest:    6 tasks,  5,500 roots
union:  22 tasks, 18,000 roots
```

## Runtime model

For shard `i`, with `N_i` roots and measured average fresh-process duration
`m_i` minutes:

```text
instance_hours_i = N_i * m_i / 60 / 4 * 1.25
```

The 25% multiplier covers setup, downloads and preparation, fresh-process
transitions, OOM repair, debugging, idle imbalance, and Spot interruption.
The local RTX 5090 gate supplies no A10G timing evidence.

The fixed On-Demand planning rate is USD 5.672 per `g5.12xlarge` hour, matching
the rate in this [AWS worked example](https://aws.amazon.com/blogs/containers/fully-sharded-data-parallel-with-ray-on-amazon-ecs/).
The USD 3.40 Spot figure is only an estimate; AWS Spot prices vary by
Availability Zone and supply/demand and can exceed it. See
[AWS Spot pricing](https://aws.amazon.com/ec2/spot/pricing/).

| Mean accepted-root time | NLP | Vision | Rest | Aggregate instance-hours | Spot at assumed USD 3.40/h | On-Demand |
|---:|---:|---:|---:|---:|---:|---:|
| 5 minutes | 6.2 days | 7.4 days | 6.0 days | 468.75 | USD 1,593.75 | USD 2,658.75 |
| 15 minutes | 18.6 days | 22.1 days | 17.9 days | 1,406.25 | USD 4,781.25 | USD 7,976.25 |
| 30 minutes | 37.1 days | 44.3 days | 35.8 days | 2,812.50 | USD 9,562.50 | USD 15,952.50 |

The USD 5,500 working authorization assumes 90% of hours at the USD 3.40 Spot
planning rate, 10% at USD 5.672 On-Demand, a 15-minute mean, and USD 250 for
storage, root disks, public IPv4, and other small charges. That formula yields
USD 5,350.75; the remaining USD 149.25 is campaign contingency.

Working caps are:

- Person 1 / NLP: USD 1,750.
- Person 2 / vision: USD 2,100.
- Person 3 / rest: USD 1,650.

Set account budgets at 50%, 75%, and 90% of each cap. These figures exclude
taxes and engineer labor.

## Before renting GPUs

1. Use the same reviewed clean commit from branch `v2` on all three instances.
   The shard workflow records the commit and a hash closure of production
   source plus `pyproject.toml` and `uv.lock`; dirty or different source fails.
2. Accept only the Kaggle competitions assigned to that task group. A shard
   preflight probes its own tasks, while the legacy unsharded command still
   probes all 22.
3. Request at least 48 G/VT vCPUs per account in Ohio, or 144 if all instances
   share one account. AWS documents that new accelerator quotas may default to
   zero. See [EC2 accelerator quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html).
4. Mount each persistent gp3 volume as the workspace. gp3 includes a baseline
   3,000 IOPS and 125 MiB/s and permits independent provisioning above it. See
   [AWS gp3 behavior](https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html).
5. Complete an 8–12 hour On-Demand A10G calibration using about 96 real frozen
   production rows: approximately 32 accepted rows from each shard, sampled
   across eager/compiled execution, light/heavy shapes, and specialized steps.
   Run the three exact shard commands sequentially on the pilot instance with
   `--max-new-accepted 32`. The limit returns only after a complete worker
   batch, so four GPUs may overshoot slightly. The separate workspaces retain
   every canonical five-epoch success; omit the limit during production and
   those rows resume without repetition. If the first slice misses a required
   execution class, run another bounded increment and include it in the
   reforecast.
6. Do not authorize all three instances if first-pass acceptance is below 90%,
   a systematic A10G incompatibility appears, or the weighted mean exceeds 20
   minutes. Recalculate each shard with the formula above first.

## Production commands

Use a separate persistent workspace for each owner. The repository checkout
and MLE-bench checkout may use the same paths on each instance, but workspace
paths and EBS volumes must not be shared.

Person 1:

```bash
uv run --frozen --extra a10g-dataset-pack python scripts/run_a10g_18k_pack.py \
  --workspace /mnt/perfseer-a10g-18k-nlp \
  --mlebench-checkout "$PERFSEER_MLEBENCH_ROOT" \
  --task-group nlp
```

Person 2:

```bash
uv run --frozen --extra a10g-dataset-pack python scripts/run_a10g_18k_pack.py \
  --workspace /mnt/perfseer-a10g-18k-vision \
  --mlebench-checkout "$PERFSEER_MLEBENCH_ROOT" \
  --task-group vision
```

Person 3:

```bash
uv run --frozen --extra a10g-dataset-pack python scripts/run_a10g_18k_pack.py \
  --workspace /mnt/perfseer-a10g-18k-rest \
  --mlebench-checkout "$PERFSEER_MLEBENCH_ROOT" \
  --task-group rest
```

Each command is inherently resumable only with the same `--task-group`. A
different group, an omitted group, a changed task order, a changed manifest,
dirty source, or environment drift fails before more labels are accepted.

After the final assigned task, the workspace writes
`state/shard_completion.json`. A shard does not write `final/` and cannot be
passed directly to the full finalizer.

If Spot capacity is unavailable for six hours or the same shard is interrupted
twice in 24 hours, move only that shard to On-Demand. Keep its gp3 volume and
rerun the identical command. AWS normally supplies a two-minute interruption
notice, but durability must not depend on receiving it. See
[AWS Spot interruption guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html).

## Strict merge and verification

Stop all shard runners and retain the three source workspaces unchanged. Run
the merge from the same clean reviewed commit into a new path on the same
filesystem as its hidden staging directory:

```bash
uv run --frozen --extra a10g-dataset-pack python \
  scripts/merge_a10g_18k_shards.py \
  --nlp-workspace /mnt/perfseer-a10g-18k-nlp \
  --vision-workspace /mnt/perfseer-a10g-18k-vision \
  --rest-workspace /mnt/perfseer-a10g-18k-rest \
  --output-workspace /mnt/perfseer-a10g-18k-merged
```

The merge revalidates every shard before and after copying. It copies only the
full manifest, task receipts, accepted/failed records, repair artifacts, slot
states, required provenance sidecars, and exact source/environment locks. It
does not copy datasets, caches, logs, dispatch files, credentials, admissions,
or shard-local final output. The hidden staging directory is bound to the
three shard receipt hashes, so an identical interrupted merge resumes; a
different shard set is rejected. Publication to the requested output path is
an atomic same-filesystem rename.

The merger synthesizes the canonical 22-task loop, runs the unchanged exact
18K finalizer, and immediately runs its read-only verification. Independently
repeat both merge and final-pack verification:

```bash
uv run --frozen --extra a10g-dataset-pack python \
  scripts/merge_a10g_18k_shards.py \
  --nlp-workspace /mnt/perfseer-a10g-18k-nlp \
  --vision-workspace /mnt/perfseer-a10g-18k-vision \
  --rest-workspace /mnt/perfseer-a10g-18k-rest \
  --output-workspace /mnt/perfseer-a10g-18k-merged \
  --verify-only
```

```bash
uv run --frozen --extra a10g-dataset-pack python \
  scripts/finalize_a10g_18k_pack.py \
  --workspace /mnt/perfseer-a10g-18k-merged \
  --verify-only
```

Success requires exactly 18,000 unique accepted configuration and run IDs,
54,000 measured epoch records, all 22 task receipts, exact repaired quotas,
matching source/environment locks, A10G-only hardware provenance, and the
existing leakage-safe global splits and train-only transforms.
