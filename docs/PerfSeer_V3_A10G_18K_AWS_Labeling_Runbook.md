# PerfSeer V3 A10G 18K AWS labeling runbook

Date: 2026-07-30

This runbook is the operator procedure for collecting the final PerfSeer V3
label pack on AWS A10G 24 GiB GPUs. The coding contract and validation evidence
are defined in
[perfseer_v3_dataset_design_report.md](perfseer_v3_dataset_design_report.md).

## 1. What this campaign produces

The campaign produces exactly 18,000 accepted end-to-end label records for:

```text
nvidia_a10g_24gb_aws_g5
```

Each accepted configuration runs once in a fresh process for five complete
training epochs:

- epochs 1–2 are warmup and are excluded from the labels;
- epochs 3–5 are measured and aggregated;
- a successful configuration is not repeated;
- an explicit CUDA OOM creates a new configuration with a smaller batch;
- an unstable completed measurement is quarantined and replaced; and
- failed, OOM, and quarantined attempts do not count toward the 18,000.

The final pack contains 18,000 accepted records and 54,000 measured epoch
subrecords. Each accepted record has the six V3 targets:

1. mean complete-epoch time over epochs 3–5;
2. time-weighted mean SM utilization;
3. P95 SM utilization;
4. peak used device VRAM;
5. peak PyTorch reserved memory; and
6. peak memory-controller utilization.

RTX 5090 evidence in this repository verifies source trainability and compiler
paths only. It is not part of the A10G label dataset.

## 2. What is and is not automated

One repository command performs the complete instance-local campaign:

1. verify the locked Python/CUDA environment;
2. verify the pinned MLE-bench checkout;
3. authenticate Kaggle and check access to all 22 competitions;
4. freeze the exact 18,000-row manifest;
5. download, verify, prepare, label, and delete one Kaggle task at a time;
6. run one isolated worker per visible physical A10G;
7. resume from durable atomic state after interruption;
8. repair explicit OOMs and replace unstable configurations; and
9. finalize the pack automatically after the 22nd task.

This revision does not provision EC2 instances, attach disks, manage Spot
instances, call S3/SQS/DynamoDB, or upload the completed pack. Provisioning and
any later transfer of the verified workspace remain operator actions.

## 3. AWS instance requirements

Use an AWS `g5` instance exposing one or more NVIDIA A10G GPUs. Before starting
the campaign, require:

- every visible GPU name contains `A10G`;
- compute capability is 8.6;
- each visible GPU exposes approximately 24 GiB;
- the NVIDIA driver supports the locked CUDA 13.0 PyTorch build;
- no unrelated compute process uses a campaign GPU;
- Python is 3.11, 3.12, or 3.13;
- `git`, `uv`, and the Kaggle CLI installed by the locked environment are
  available;
- outbound access to GitHub, Python package indexes, and Kaggle is available;
- the labeling workspace is outside the repository checkout; and
- the workspace filesystem has at least 600 GiB available at campaign start.

The code retains a 40 GiB safety margin and refuses a task when its conservative
archive/extraction/preparation peak would exceed the 600 GiB campaign limit.
Use a dedicated mounted data volume such as `/mnt/perfseer-a10g-18k`.

The runner uses every visible GPU and requires every one of them to qualify as
an A10G. Do not expose a mixed set of GPU models.

## 4. Freeze the repository and MLE-bench revisions

Clone this repository and check out the reviewed `v2` commit that contains this
runbook. Record the commit before collection:

```bash
export PERFSEER_REPOSITORY_URL=https://github.com/OWNER/PerfSeer-predictor.git
export PERFSEER_REPO_ROOT=/opt/PerfSeer-predictor
git clone "$PERFSEER_REPOSITORY_URL" "$PERFSEER_REPO_ROOT"
git -C "$PERFSEER_REPO_ROOT" checkout v2
git -C "$PERFSEER_REPO_ROOT" rev-parse HEAD
git -C "$PERFSEER_REPO_ROOT" status --short
```

The final status command must print nothing. Do not modify source, registries,
configuration, or the dependency lock after collection begins.

Clone MLE-bench separately and check out the exact audited revision:

```bash
export PERFSEER_MLEBENCH_ROOT=/opt/mle-bench
git clone https://github.com/openai/mle-bench.git "$PERFSEER_MLEBENCH_ROOT"
git -C "$PERFSEER_MLEBENCH_ROOT" checkout --detach 507f92e1138bb6e40dac5c6ee7a6758e6424bf97
git -C "$PERFSEER_MLEBENCH_ROOT" rev-parse HEAD
git -C "$PERFSEER_MLEBENCH_ROOT" status --porcelain --untracked-files=all
```

The final command must also print nothing. The production entrypoint rejects a
missing, modified, dirty, or differently pinned MLE-bench checkout.

## 5. Accept all Kaggle competition rules

The Kaggle account must be able to list files for all 22 frozen competitions.
Accept each competition's rules in the Kaggle web interface before running the
campaign:

| # | Task | Kaggle competition slug |
| ---: | --- | --- |
| 1 | Histopathologic cancer | `histopathologic-cancer-detection` |
| 2 | Dogs vs. cats | `dogs-vs-cats-redux-kernels-edition` |
| 3 | Dog breed | `dog-breed-identification` |
| 4 | SIIM-ISIC melanoma | `siim-isic-melanoma-classification` |
| 5 | APTOS 2019 | `aptos2019-blindness-detection` |
| 6 | Aerial cactus | `aerial-cactus-identification` |
| 7 | Plant pathology | `plant-pathology-2020-fgvc7` |
| 8 | RANZCR CLiP | `ranzcr-clip-catheter-line-classification` |
| 9 | Leaf classification | `leaf-classification` |
| 10 | Denoising dirty documents | `denoising-dirty-documents` |
| 11 | Jigsaw toxic comments | `jigsaw-toxic-comment-classification-challenge` |
| 12 | Detecting insults | `detecting-insults-in-social-commentary` |
| 13 | Spooky author | `spooky-author-identification` |
| 14 | Random acts of pizza | `random-acts-of-pizza` |
| 15 | English text normalization | `text-normalization-challenge-english-language` |
| 16 | Russian text normalization | `text-normalization-challenge-russian-language` |
| 17 | MLSP 2013 birds | `mlsp-2013-birds` |
| 18 | ICML 2013 whale | `the-icml-2013-whale-challenge-right-whale-redux` |
| 19 | NYC taxi fare | `new-york-city-taxi-fare-prediction` |
| 20 | NOMAD 2018 | `nomad2018-predict-transparent-conductors` |
| 21 | Tabular playground December 2021 | `tabular-playground-series-dec-2021` |
| 22 | Tabular playground May 2022 | `tabular-playground-series-may-2022` |

The entrypoint checks the complete paginated file inventory for every
competition before it creates or changes campaign state. Missing rules access
therefore fails before any task is downloaded or labeled.

## 6. Supply Kaggle credentials outside the repository

Never place Kaggle credentials in the repository checkout or labeling
workspace. One supported approach is an external credential directory:

```bash
export PERFSEER_KAGGLE_CONFIG_ROOT=/mnt/perfseer-secrets/kaggle
install -d -m 700 "$PERFSEER_KAGGLE_CONFIG_ROOT"
install -m 600 /secure/source/kaggle.json "$PERFSEER_KAGGLE_CONFIG_ROOT/kaggle.json"
export KAGGLE_CONFIG_DIR="$PERFSEER_KAGGLE_CONFIG_ROOT"
```

Alternatively, use `KAGGLE_API_TOKEN`, or the legacy `KAGGLE_USERNAME` and
`KAGGLE_KEY` pair, through the instance's secret-injection mechanism. Do not
write tokens into shell scripts, logs, Git files, or workspace state.

The parent uses credentials only for Kaggle authentication/download. Runtime
dependency probes, MLE-bench preparation children, and model workers receive
credential-isolated environments.

## 7. Install the exact locked environment

From the repository root, synchronize the checked-in dependency lock:

```bash
cd "$PERFSEER_REPO_ROOT"
uv sync --frozen --extra a10g-dataset-pack
```

If the instance does not already provide Python 3.11–3.13, install a supported
interpreter with `uv python install 3.13`, then add `--python 3.13` to the
`uv sync` command. All later `uv run` commands reuse the resulting project
environment.

The production preflight enforces the following compiler-critical versions:

| Package | Version |
| --- | --- |
| PyTorch | 2.11.0 |
| PyTorch Geometric | 2.7.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| Transformers | 5.7.0 |
| Triton | 3.6.0 |
| Kaggle | 2.2.2 |

Check the visible hardware and CUDA runtime:

```bash
nvidia-smi
uv run --frozen --extra a10g-dataset-pack python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

The CUDA availability value must be `True`, and every listed device must be an
NVIDIA A10G. The production runner performs stricter NVML checks itself.

If a container is used, record its immutable digest before the first campaign
run:

```bash
export PERFSEER_CONTAINER_DIGEST=sha256:REPLACE_WITH_IMMUTABLE_CONTAINER_DIGEST
```

The first labeling invocation freezes Python, package, PyTorch, CUDA, cuDNN,
driver, and optional container identities in
`state/campaign_environment.json`. Every resume must use the same environment.

## 8. Start the resumable campaign

Choose the external workspace once and do not change it:

```bash
export PERFSEER_LABEL_WORKSPACE=/mnt/perfseer-a10g-18k
mkdir -p "$PERFSEER_LABEL_WORKSPACE/operator-logs"
```

Run the campaign in a persistent terminal such as `tmux`. Preserve the Python
exit code when also writing an operator log:

```bash
cd "$PERFSEER_REPO_ROOT"
set -o pipefail
uv run --frozen --extra a10g-dataset-pack python scripts/run_a10g_18k_pack.py \
  --workspace "$PERFSEER_LABEL_WORKSPACE" \
  --mlebench-checkout "$PERFSEER_MLEBENCH_ROOT" \
  2>&1 | tee -a "$PERFSEER_LABEL_WORKSPACE/operator-logs/campaign.log"
```

Do not use `--materialize-only` for the production campaign. That option
prepares only the next task and returns without producing labels.

The command first performs all global preflights. It then processes the frozen
task order sequentially. For each task it:

1. downloads and verifies only that task;
2. safely extracts and prepares a shared 4,096-example training view;
3. runs all configurations assigned to the task;
4. writes accepted, failure, repair, provenance, and resume records atomically;
5. writes and revalidates the task completion receipt; and
6. deletes the task archive, extracted data, prepared view, and reconstructible
   compiler cache before continuing.

On a multi-GPU `g5` instance, the parent launches one isolated worker per
physical A10G. Each attempt runs in a fresh child process.

## 9. Monitor without modifying state

The following commands are read-only:

```bash
tail -f "$PERFSEER_LABEL_WORKSPACE/operator-logs/campaign.log"
```

```bash
nvidia-smi
```

```bash
jq '{active_task_id, completed_task_ids}' \
  "$PERFSEER_LABEL_WORKSPACE/state/task_loop.json"
```

```bash
find "$PERFSEER_LABEL_WORKSPACE/attempts/accepted" \
  -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
```

```bash
find "$PERFSEER_LABEL_WORKSPACE/attempts/failed" \
  -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
```

Do not edit manifest, slot, attempt, receipt, provenance, or task-loop JSON.
Their hashes and one-to-one relationships are finalization inputs.

## 10. Resume after interruption

After a reboot, terminal loss, or corrected hard failure:

1. mount the same workspace at the same path;
2. use the same reviewed repository commit;
3. use the same clean pinned MLE-bench checkout;
4. restore external Kaggle credentials;
5. restore the exact locked Python/CUDA/driver/container environment; and
6. rerun the exact command from section 8.

Do not add a `--resume` flag; the production entrypoint is inherently
resumable. It revalidates the frozen manifest, environment lock, completed task
receipts, accepted records, and active task state before continuing.

If the process was interrupted after a task receipt but before cleanup, resume
finishes the receipt-bound cleanup before advancing.

## 11. Failure handling

The workflow handles only expected measurement outcomes automatically:

- explicit CUDA OOM: retry the next smaller power-of-two batch under a new
  configuration ID;
- batch-one OOM: quarantine and generate a same-cell replacement;
- completed but unstable epochs 3–5: quarantine and generate a same-cell
  replacement.

The following conditions stop the campaign for operator correction:

- dependency or environment drift;
- missing Kaggle access or rules acceptance;
- corrupt, unsafe, or unexpected dataset content;
- MLE-bench checkout/preparer mismatch;
- compiler or unexpected child-process failure;
- timeout without an explicit OOM envelope;
- missing/corrupt result envelope;
- GPU contamination; or
- failure to return GPU memory to the pre-run baseline.

For a hard failure:

1. keep the workspace unchanged;
2. inspect the final operator-log lines;
3. inspect the newest private `0600` log in `attempts/logs`;
4. correct the external dependency, access, disk, driver, or hardware problem;
5. ensure no unrelated GPU process remains; and
6. rerun the exact campaign command.

Never convert a hard failure into an OOM record, manually advance a slot, delete
an accepted record, or modify a completion receipt.

## 12. Finalization and verification

After the 22nd task receipt, the production command automatically finalizes:

```text
$PERFSEER_LABEL_WORKSPACE/final/accepted_labels.jsonl
$PERFSEER_LABEL_WORKSPACE/final/failed_attempts.jsonl
$PERFSEER_LABEL_WORKSPACE/final/target_transform.json
$PERFSEER_LABEL_WORKSPACE/final/audit_report.json
$PERFSEER_LABEL_WORKSPACE/final/dataset_manifest.json
$PERFSEER_LABEL_WORKSPACE/final/completion_receipt.json
```

`completion_receipt.json` is written last. Its presence alone is not enough;
run the independent read-only verifier:

```bash
cd "$PERFSEER_REPO_ROOT"
uv run --frozen --extra a10g-dataset-pack python \
  scripts/finalize_a10g_18k_pack.py \
  --workspace "$PERFSEER_LABEL_WORKSPACE" \
  --verify-only
```

Verification must prove:

- exactly 18,000 unique accepted configuration and run IDs;
- exactly 54,000 measured epoch records with epochs `[3, 4, 5]`;
- exact frozen quotas after linked repairs;
- complete receipts for all 22 tasks;
- no failed attempt in the successful manifest;
- leakage-safe grouped train/validation/test splits;
- target transforms fitted from training rows only; and
- byte-identical final artifacts and receipt hashes.

Preserve the complete verified workspace until the final pack and audit
evidence have been transferred and checked independently. The repository does
not perform that transfer.

## 13. Files that must never be committed

Do not add any of the following to Git:

- Kaggle credential files or token-bearing environment files;
- downloaded Kaggle archives or extracted/prepared datasets;
- the external labeling workspace;
- attempt logs, compiler caches, and task-loop runtime state;
- local Python environments and build outputs;
- the large deterministic RTX validation plan/worker queue;
- superseded RTX evidence or failure artifacts.

The repository `.gitignore` excludes the standard local forms of these files.
The commit should contain source, frozen configs/registries, tests,
documentation, `uv.lock`, and only the concise canonical RTX execution results
and gate summary.
