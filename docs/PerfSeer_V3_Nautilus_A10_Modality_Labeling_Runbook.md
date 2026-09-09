# PerfSeer V3 Nautilus A10 Modality Labeling Runbook

## 1. Scope and safety boundary

This runbook covers the four independently controlled Nautilus workflows that
produce the canonical PerfSeer V3 `rest` shard:

| Wrapper | Native/generated families | Accepted labels | Retained measured epochs |
| --- | --- | ---: | ---: |
| `submit_nautilus_a10_audio.sh` | PANNs CNN14, TCN, M5 | 1,300 | 3,900 |
| `submit_nautilus_a10_tabular.sh` | SELU MLP, TabTransformer, MDN | 950 | 2,850 |
| `submit_nautilus_a10_graph.sh` | CGCNN, GCN, GAT, GraphSAGE | 1,300 | 3,900 |
| `submit_nautilus_a10_generated.sh` | independent generated | 1,950 | 5,850 |
| Final `merged-rest` | exact disjoint union | **5,500** | **16,500** |

Every accepted label runs five epochs. Epochs 1–2 are warm-up; epochs 3–5 are
the three retained measurements. The final measured-epoch count is therefore
`5,500 × 3 = 16,500`.

The instructions have two explicit execution classes:

- **Local/offline** commands inspect source, analyze the dataset contract, or
  render YAML. They do not invoke a Kubernetes executable.
- **Operator-only/Nautilus** commands must be run later by an authenticated
  operator. They invoke the configured Kubernetes client and may query or
  change cluster resources.

No Kubernetes command in this document was executed during development. No
client/server dry-run was used either.

## 2. Official NRP guidance used by this runbook

- [NRP GPU Pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/):
  generic GPU requests use `nvidia.com/gpu`; GPU product affinity can select an
  A10; the current page lists NVIDIA A10 as a 24 GB GPU.
- [NRP Jobs](https://nrp.ai/documentation/userdocs/running/jobs/): Jobs are the
  recommended finite computation mechanism. A Job must have a meaningful
  terminating command and must not use `sleep infinity`.
- [NRP CephFS/RBD](https://nrp.ai/documentation/userdocs/storage/ceph/): CephFS
  is `ReadWriteMany`; unique paths avoid write conflicts; package installations
  must not be placed on CephFS.
- [NRP cluster policy](https://nrp.ai/documentation/userdocs/start/policies/):
  interactive Pods are limited to six hours, 2 GPUs, 32 GB RAM, and 16 CPUs;
  Jobs are recommended for larger work; request/limit sizing and active GPU use
  are mandatory.

Always re-read these pages before a real campaign because cluster labels,
available nodes, and policy can change.

## 3. Dataset analysis and exact allocation

The source of truth is the existing frozen 18,000-row target manifest. The
modality controller projects the six canonical `rest` tasks, then partitions by
the manifest's `quota_modality`. It does not resample rows or invent a second
quota table.

| Modality | Family quota | Kaggle tasks | Compressed download estimate |
| --- | --- | --- | ---: |
| Audio | PANNs 550; TCN 400; M5 350 | `mlsp-2013-birds`, `icml-2013-whale` | 878,240,000 B (0.818 GiB) |
| Tabular | SELU 300; TabTransformer 350; MDN 300 | `nyc-taxi-fare`, `tabular-playground-dec-2021`, `tabular-playground-may-2022` | 6,970,000,000 B (6.491 GiB) |
| Graph | CGCNN 500; GCN 250; GAT 300; GraphSAGE 250 | `nomad2018` | 6,240,000 B (0.0058 GiB) |
| Generated | independent generated 1,950 | `mlsp-2013-birds`, `nyc-taxi-fare`, `nomad2018` | 6,291,340,000 B (5.859 GiB) |

The six unique source archives total about 7.854 GB. Fully isolated modality
workspaces may download about 14.146 GB cumulatively because the generated job
reuses three source tasks without sharing their writable task caches. The 700
GiB PVC is sized for downloads, nested extraction, prepared views, accepted
records, repair evidence, and the existing 600 GiB workspace admission gate
with reserve.

The generated quota contains 30 deterministic lineages: ten sourced from each
of audio, tabular, and graph, with 65 rows per lineage. Six lineages (390 rows)
are held out by the frozen manifest.

**Local/offline — safe now:** inspect the exact projection for each wrapper.

```bash
./submit_nautilus_a10_audio.sh analyze
./submit_nautilus_a10_tabular.sh analyze
./submit_nautilus_a10_graph.sh analyze
./submit_nautilus_a10_generated.sh analyze
```

Each result includes its candidate count, measured-epoch count, task list,
family counts, download estimate, contract hash, and the shared target-manifest
hash. All four contract hashes must remain stable for a single campaign.

## 4. A10/A10G family identity and provenance

The new physical family identifier is:

```text
nvidia_a10_24gb_family_v1
```

The existing candidate IDs commit to the historical logical target ID
`nvidia_a10g_24gb_aws_g5`. That field remains unchanged so existing AWS A10G
hashes and behavior continue to validate. The Nautilus workflow explicitly
opts into the A10 family and separately verifies physical provenance.

For every accepted record, all of these conditions are mandatory:

1. NVML-reported name contains `A10` (therefore accepting physical A10 or
   A10G).
2. Compute capability is exactly 8.6.
3. Total memory is between 22 and 26 GiB.
4. GPU UUID is non-empty.
5. The full physical provenance object is stored under
   `provenance/hardware/<sha256>.json`.
6. The accepted record's `fingerprints.hardware_sha256` matches that object's
   canonical hash.
7. The merged pack rechecks the provenance object; a missing, altered, or
   non-A10 object fails publication.

Do not rewrite `target_hardware_id` in existing candidates. The hardware-family
layer and the physical provenance hash are what permit NRP A10 and AWS A10G
measurements to coexist without pretending that their physical environment is
identical.

## 5. One-time prerequisites

### 5.1 Account and namespace

Before using the cluster:

1. Obtain an NRP account, accept the current AUP, and join the intended
   namespace.
2. Install the operator kubeconfig on the operator machine, outside this
   repository.
3. Confirm with the namespace owner that one generic GPU, 8 CPUs, 32 GiB RAM,
   and a 700 GiB RWX PVC are allowed.
4. Use only a public/read-only Git URL that the container can clone without an
   embedded credential. The renderer rejects HTTP URLs containing user info or
   passwords. For a private repository, arrange an approved read-only clone
   mechanism before using these scripts; never put a token in the repository
   URL.

### 5.2 Kaggle rules and Secret

The operator's Kaggle account must manually accept the rules for all six
competitions before the pilot. The label runner inventories every required
competition before downloading data and fails closed if access is missing.

Download the legacy `kaggle.json` credential from the Kaggle account's
[API settings](https://www.kaggle.com/settings/api), keep it outside the
repository with mode `0600`, and never print it or place it on the PVC.

**Operator-only/Nautilus — run later:** create or update the Secret from that
external file. The Secret must contain an exact key named `kaggle.json`.
The rendered workload contains only the Secret name; Kubernetes mounts the key
read-only with mode `0400` and sets `KAGGLE_CONFIG_DIR` to its directory.

```bash
export NAMESPACE='replace-with-your-namespace'
export KAGGLE_SECRET_NAME='perfseer-kaggle'
export KAGGLE_JSON='/absolute/path/outside/repo/kaggle.json'

test -f "$KAGGLE_JSON"
chmod 600 "$KAGGLE_JSON"
kubectl create secret generic "$KAGGLE_SECRET_NAME" \
  --namespace "$NAMESPACE" \
  --from-file=kaggle.json="$KAGGLE_JSON" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl get secret "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE"
kubectl describe secret "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE"
```

The command above is operator-only. The development process did not run it or
any other dry-run. Do not use `kubectl get secret ... -o yaml`; the workflows
only need the Secret's name. Do not use `envFrom` for a `kaggle.json` Secret;
the filename is not an environment-variable interface.

### 5.3 A 700 GiB RWX PVC

Choose a current CephFS storage class available to the namespace. The example
uses `rook-cephfs`; change it if the namespace owner specifies another RWX
class.

**Operator-only/Nautilus — run later:** save this as `/tmp/perfseer-rwx.yaml` on
the operator machine.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perfseer-v3-rwx
  namespace: replace-with-your-namespace
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 700Gi
  storageClassName: rook-cephfs
```

**Operator-only/Nautilus — run later:** create and inspect the claim.

```bash
kubectl apply -f /tmp/perfseer-rwx.yaml
kubectl get pvc perfseer-v3-rwx --namespace "$NAMESPACE"
kubectl describe pvc perfseer-v3-rwx --namespace "$NAMESPACE"
```

### 5.4 Shared PVC layout

The manifests mount the claim at `/pvc`. Production, debug, and merge paths are
deliberately isolated:

```text
/pvc/perfseer-v3/
├── audio/<run-id>/{source,mle-bench,workspace}/
├── tabular/<run-id>/{source,mle-bench,workspace}/
├── graph/<run-id>/{source,mle-bench,workspace}/
├── generated/<run-id>/{source,mle-bench,workspace}/
├── debug/<modality>/<run-id>/{source,mle-bench,workspace}/
└── merged-rest/<campaign-id>/
```

Never run two writers against the same modality and run ID. The four modality
jobs may run independently because their paths are disjoint. Debug labels are
disposable and are never copied into a production directory or merge input.

The source and MLE-bench checkouts are retained on PVC for pin verification and
resumption. Before package installation the Job copies source to `/tmp`; pip's
target, cache, and build files stay on ephemeral storage, not CephFS.

## 6. Image, GPU, and resource contract

All rendered manifests use the official image and immutable digest:

```text
pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89
```

The default request equals the limit: 1 generic GPU, 8 CPU cores, and 32 GiB
RAM. Node affinity defaults to `nvidia.com/gpu.product In [NVIDIA-A10]`.
Because the official NRP page says the live node list is more authoritative
than its static table, the operator must confirm the exact product label and a
driver compatible with CUDA 13 before submitting.

**Operator-only/Nautilus — run later:** inspect current A10 product and CUDA
labels. Do not run this during local development.

```bash
kubectl get nodes \
  -l nvidia.com/gpu.product \
  -L nvidia.com/gpu.product,nvidia.com/cuda.driver.major,nvidia.com/cuda.runtime.major,nvidia.com/cuda.runtime.minor
```

If the current product value differs, pass it explicitly with
`--gpu-product CURRENT-LABEL`. Do not remove product affinity and hope for the
right GPU. If no compatible A10 node exists, stop and coordinate with NRP; do
not silently substitute another GPU family.

## 7. Common script arguments and safety behavior

There is no submit-all command. Select one wrapper at a time:

```text
submit_nautilus_a10_audio.sh
submit_nautilus_a10_tabular.sh
submit_nautilus_a10_graph.sh
submit_nautilus_a10_generated.sh
```

Every wrapper supports:

```text
analyze
render-pod
submit-pod
render-pilot-job
submit-pilot-job
render-production-job
submit-production-job
status
monitor
verify
```

No arguments print help and do not submit anything. `render-*` writes one YAML
document to standard output and never calls a Kubernetes executable. Only
`submit-*` invokes `apply`; `status` and `monitor` make read/exec calls. Set
`KUBECTL_BIN` only to select the operator's approved executable. Never set it
to a shell fragment.

Use one immutable, clean 40-character Git commit. A branch or tag is rejected.
The Job checks out that exact commit detached and fails if the checkout is
dirty.

**Local/offline — safe now:** define reusable shell arguments and render YAML.

```bash
export NAMESPACE='replace-with-your-namespace'
export PVC_NAME='perfseer-v3-rwx'
export REPOSITORY_URL='https://github.com/your-org/PerfSeer-predictor.git'
export REVISION='replace-with-40-lowercase-hex-commit'
export KAGGLE_SECRET_NAME='perfseer-kaggle'
export RUN_ID='a10-campaign-001'
export IMAGE='pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89'

COMMON_ARGS=(
  --namespace "$NAMESPACE"
  --pvc "$PVC_NAME"
  --repository-url "$REPOSITORY_URL"
  --revision "$REVISION"
  --kaggle-secret "$KAGGLE_SECRET_NAME"
  --image "$IMAGE"
  --run-id "$RUN_ID"
)

./submit_nautilus_a10_audio.sh render-pod "${COMMON_ARGS[@]}" > /tmp/audio-pod.yaml
./submit_nautilus_a10_audio.sh render-pilot-job "${COMMON_ARGS[@]}" > /tmp/audio-pilot.yaml
./submit_nautilus_a10_audio.sh render-production-job "${COMMON_ARGS[@]}" > /tmp/audio-production.yaml
```

Rendering is a local structural check, not cluster validation. Do not run a
client/server dry-run during development.

## 8. Interactive Pod path: short debugging only

SSH is unnecessary. Kubernetes already provides an authenticated exec channel
to the container. The debug Pod uses `sleep infinity`, which NRP permits for an
interactive Pod but not a Job. It is limited to short diagnosis and is expected
to be destroyed within six hours.

### 8.1 Submit and enter a debug Pod

**Operator-only/Nautilus — run later:** submit the chosen debug Pod. The wrapper
immediately runs get/describe/log/event feedback after apply.

```bash
./submit_nautilus_a10_audio.sh submit-pod "${COMMON_ARGS[@]}"
```

**Operator-only/Nautilus — run later:** enter it with `kubectl exec`, not SSH.

```bash
kubectl exec -it \
  "perfseer-a10-audio-debug-${RUN_ID}" \
  --namespace "$NAMESPACE" \
  -- /bin/bash
```

### 8.2 Bootstrap and run a tiny debug sample

The following is executed **inside the operator-only/Nautilus debug Pod**. It
uses the debug workspace selected by the manifest and accepts only two new
records before returning.

```bash
set -euo pipefail
umask 077

source_root="$PERFSEER_RUN_ROOT/source"
mlebench_root="$PERFSEER_RUN_ROOT/mle-bench"
workspace_root="$PERFSEER_RUN_ROOT/workspace"
ephemeral_source="/tmp/perfseer-source-$PERFSEER_RUN_ID"

git clone --filter=blob:none "$PERFSEER_REPOSITORY_URL" "$source_root"
git -C "$source_root" fetch --no-tags origin "$PERFSEER_REPOSITORY_REVISION"
git -C "$source_root" checkout --detach "$PERFSEER_REPOSITORY_REVISION"
git clone --filter=blob:none https://github.com/openai/mle-bench.git "$mlebench_root"
git -C "$mlebench_root" checkout --detach 507f92e1138bb6e40dac5c6ee7a6758e6424bf97
cp -a "$source_root" "$ephemeral_source"
python -m pip install --no-cache-dir "$ephemeral_source[a10g-dataset-pack]"
cd "$ephemeral_source"

python scripts/run_nautilus_a10_modality.py label \
  --modality "$PERFSEER_MODALITY" \
  --workspace "$workspace_root" \
  --mlebench-checkout "$mlebench_root" \
  --repository-revision "$PERFSEER_REPOSITORY_REVISION" \
  --image-digest "$PERFSEER_CONTAINER_DIGEST" \
  --max-new-accepted 2
```

If a path already exists, inspect it and use a new debug run ID; do not delete
or overwrite an unknown path. Debug outputs under `/pvc/perfseer-v3/debug/`
must never be used as production merge inputs.

**Operator-only/Nautilus — run later:** after collecting debug evidence, delete
only the named debug Pod so its GPU is released.

```bash
kubectl delete pod \
  "perfseer-a10-audio-debug-${RUN_ID}" \
  --namespace "$NAMESPACE"
```

## 9. Recommended path: pilot, then production Job

Jobs are finite, retain stdout/stderr, release the GPU on exit, and resume from
the PVC. The manifest uses `backoffLimit: 0` deliberately: a failed attempt is
diagnosed before the operator decides to relaunch, avoiding an unexamined
automatic retry against the same mutable task cache.

Use the same run ID for pilot and production of one modality. The pilot accepts
eight new rows by default and exits at a safe boundary. Production resumes the
same workspace and completes the quota.

### 9.1 Audio

**Operator-only/Nautilus — run later:** pilot and, only after it passes,
production.

```bash
./submit_nautilus_a10_audio.sh submit-pilot-job "${COMMON_ARGS[@]}"
./submit_nautilus_a10_audio.sh status "${COMMON_ARGS[@]}" --job-mode pilot

./submit_nautilus_a10_audio.sh submit-production-job "${COMMON_ARGS[@]}"
```

Expected production completion: 1,300 accepted labels and 3,900 retained
measurements.

### 9.2 Tabular

**Operator-only/Nautilus — run later:** pilot and production.

```bash
./submit_nautilus_a10_tabular.sh submit-pilot-job "${COMMON_ARGS[@]}"
./submit_nautilus_a10_tabular.sh status "${COMMON_ARGS[@]}" --job-mode pilot

./submit_nautilus_a10_tabular.sh submit-production-job "${COMMON_ARGS[@]}"
```

Expected production completion: 950 accepted labels and 2,850 retained
measurements.

### 9.3 Graph

**Operator-only/Nautilus — run later:** pilot and production.

```bash
./submit_nautilus_a10_graph.sh submit-pilot-job "${COMMON_ARGS[@]}"
./submit_nautilus_a10_graph.sh status "${COMMON_ARGS[@]}" --job-mode pilot

./submit_nautilus_a10_graph.sh submit-production-job "${COMMON_ARGS[@]}"
```

Expected production completion: 1,300 accepted labels and 3,900 retained
measurements.

### 9.4 Generated

**Operator-only/Nautilus — run later:** pilot and production.

```bash
./submit_nautilus_a10_generated.sh submit-pilot-job "${COMMON_ARGS[@]}"
./submit_nautilus_a10_generated.sh status "${COMMON_ARGS[@]}" --job-mode pilot

./submit_nautilus_a10_generated.sh submit-production-job "${COMMON_ARGS[@]}"
```

Expected production completion: 1,950 accepted labels and 5,850 retained
measurements.

Choose when to submit each modality. Do not submit all four merely because the
scripts are available. Observe policy, GPU availability, and the prior pilot's
utilization first.

## 10. Mandatory immediate feedback and monitor

After a real Job submission, do not wait for the background monitor before
looking at the Job. Each `submit-*-job` action already performs the following
sequence with the configured Kubernetes executable:

1. `get job`;
2. `get pods` for the Job;
3. resolve the Pod name;
4. `describe pod`;
5. `logs`;
6. recent Pod events;
7. start a local `nohup` monitor.

**Operator-only/Nautilus — run later:** if YAML was applied manually, collect
the equivalent feedback immediately. Replace the example names for the chosen
modality.

```bash
JOB_NAME="perfseer-a10-audio-production-${RUN_ID}"

kubectl get job "$JOB_NAME" --namespace "$NAMESPACE" -o wide
kubectl get pods --namespace "$NAMESPACE" --selector "job-name=$JOB_NAME" -o wide
POD_NAME=$(kubectl get pods --namespace "$NAMESPACE" \
  --selector "job-name=$JOB_NAME" \
  -o 'jsonpath={.items[0].metadata.name}')
kubectl describe pod "$POD_NAME" --namespace "$NAMESPACE"
kubectl logs "$POD_NAME" --namespace "$NAMESPACE" --all-containers=true --tail=200
kubectl get events --namespace "$NAMESPACE" \
  --field-selector "involvedObject.name=$POD_NAME" \
  --sort-by=.lastTimestamp
```

The automatic monitor log is stored in this tracked project path:

```text
record/nautilus_a10_<modality>_<run-id>_monitor.log
```

For the first five minutes it samples at least once every 60 seconds; after
that it samples every 20 minutes. It records Job and Pod status, events,
container logs, remote process status, persistent attempt-log tails, and recent
state/accepted/checkpoint files.

**Operator-only/Nautilus — run later:** manually start or restart the monitor
if necessary.

```bash
nohup ./submit_nautilus_a10_audio.sh monitor \
  "${COMMON_ARGS[@]}" \
  --job-mode production \
  >>"record/nautilus_a10_audio_${RUN_ID}_monitor.log" 2>&1 </dev/null &
```

A monitor file is not by itself proof of success. While actively supervising a
launch, inspect and report progress at least once a minute until the Job is
running stably, completes, or fails.

## 11. Failure diagnosis and resumption

### 11.1 Pending

Check Pod events first. Common causes are an unavailable `NVIDIA-A10` product
label, CUDA/driver incompatibility, an unbound PVC, quota, or insufficient CPU
or RAM. Do not change several constraints at once. Preserve the rendered YAML
and event evidence, correct one confirmed cause, and render again.

### 11.2 Image or bootstrap failure

For `ImagePullBackOff`, verify the immutable image spelling and registry reach.
For Git failures, verify the repository is readable and the exact commit is
advertised. For MLE-bench failure, retain the pinned commit; do not advance it
without changing and reviewing the dataset contract.

The runtime rejects a dirty or wrong source revision, missing frozen package
versions, inaccessible Kaggle rules, and an unqualified GPU before labeling.
Secret values are not included in those errors.

### 11.3 OOM and label-level repair

- A Kubernetes `OOMKilled` status means host RAM exceeded the container limit.
  Inspect actual use and adjust requests/limits within NRP policy.
- A CUDA OOM inside a label attempt is retained as failure evidence. The
  workflow deterministically lowers microbatch size and, if necessary,
  substitutes inside the same quota cell.
- A physical GPU cleanup failure aborts the campaign; the worker is not reused.
- Stability spread, non-finite loss/gradient, a foreign GPU process, or changed
  provenance fails closed.

### 11.4 Required failure report

If a Job enters `Failed`, immediately report the Pod name, container exit code,
last relevant logs, recent events, and the proposed corrective action.

**Operator-only/Nautilus — run later:** collect that report.

```bash
JOB_NAME="perfseer-a10-audio-production-${RUN_ID}"
POD_NAME=$(kubectl get pods --namespace "$NAMESPACE" \
  --selector "job-name=$JOB_NAME" \
  -o 'jsonpath={.items[0].metadata.name}')

kubectl get pod "$POD_NAME" --namespace "$NAMESPACE" \
  -o 'jsonpath={.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
kubectl logs "$POD_NAME" --namespace "$NAMESPACE" --all-containers=true --tail=200
kubectl describe pod "$POD_NAME" --namespace "$NAMESPACE"
kubectl get events --namespace "$NAMESPACE" \
  --field-selector "involvedObject.name=$POD_NAME" \
  --sort-by=.lastTimestamp
```

### 11.5 Resume the same workspace

The source manifest, run identity, task loop, slot repairs, accepted records,
task receipts, provenance, and task materialization state are durable on PVC.
Resume with the same modality, run ID, repository revision, and image digest.

A failed Kubernetes Job object will not create a fresh Pod merely because the
same immutable spec is applied again. After saving all failure evidence and
confirming the exact target, remove only that failed Job object; never remove
the PVC or modality directory.

**Operator-only/Nautilus — run later and only after diagnosis:** delete the
named failed Job, then resubmit the same run ID so the workflow resumes.

```bash
kubectl delete job \
  "perfseer-a10-audio-production-${RUN_ID}" \
  --namespace "$NAMESPACE"

./submit_nautilus_a10_audio.sh submit-production-job "${COMMON_ARGS[@]}"
```

## 12. Per-modality verification

At exact production completion the label command itself runs the verifier and
publishes:

```text
<workspace>/state/modality_contract.json
<workspace>/state/run_identity.json
<workspace>/state/modality_task_receipts/*.json
<workspace>/state/modality_completion.json
```

The verifier requires:

- exactly the modality's frozen root slots;
- one unique resolved accepted record per root slot;
- no extra accepted records;
- three retained epochs `[3,4,5]` per record;
- correct logical target ID;
- valid A10/A10G physical provenance for every record;
- matching repository, image, manifest, and contract pins;
- canonical hashes for every completion artifact.

The wrapper's `verify` action is intentionally local to the filesystem. Run it
only where the PVC is mounted, such as a short operator debug Pod after source
has been copied and installed under `/tmp`.

**Inside an operator-only/Nautilus Pod with the PVC mounted:** verify each
production workspace.

```bash
./submit_nautilus_a10_audio.sh verify "${COMMON_ARGS[@]}" \
  --workspace "/pvc/perfseer-v3/audio/${RUN_ID}/workspace"
./submit_nautilus_a10_tabular.sh verify "${COMMON_ARGS[@]}" \
  --workspace "/pvc/perfseer-v3/tabular/${RUN_ID}/workspace"
./submit_nautilus_a10_graph.sh verify "${COMMON_ARGS[@]}" \
  --workspace "/pvc/perfseer-v3/graph/${RUN_ID}/workspace"
./submit_nautilus_a10_generated.sh verify "${COMMON_ARGS[@]}" \
  --workspace "/pvc/perfseer-v3/generated/${RUN_ID}/workspace"
```

Do not treat a completed Kubernetes Job alone as a verified modality. The
`modality_completion.json` receipt is required.

## 13. Final `merged-rest` publication

Only merge after all four production completion receipts verify. The merger
loads each current workspace again, compares it with its published completion,
requires identical source/image/manifest pins, proves the four canonical root
sets are disjoint, proves their union is the exact 5,500-row `rest` shard, and
copies verified resolved records into a new staging directory. It atomically
renames that directory to `merged-rest` only after all checks pass.

It rejects:

- a missing modality or missing candidate;
- a duplicated resolution;
- cross-modality overlap;
- a record or provenance file altered after completion;
- a wrong contract or target manifest;
- different repository revisions or image digests;
- a pre-existing output directory.

**Inside an operator-only/Nautilus Pod with the PVC mounted and source installed
under `/tmp`:** publish the final rest shard. Choose a new `CAMPAIGN_ID`; the
output must not already exist.

```bash
export CAMPAIGN_ID='a10-rest-001'

python scripts/run_nautilus_a10_modality.py merge \
  --audio-workspace "/pvc/perfseer-v3/audio/${RUN_ID}/workspace" \
  --tabular-workspace "/pvc/perfseer-v3/tabular/${RUN_ID}/workspace" \
  --graph-workspace "/pvc/perfseer-v3/graph/${RUN_ID}/workspace" \
  --generated-workspace "/pvc/perfseer-v3/generated/${RUN_ID}/workspace" \
  --output "/pvc/perfseer-v3/merged-rest/${CAMPAIGN_ID}"
```

Expected final receipt:

```text
/pvc/perfseer-v3/merged-rest/<campaign-id>/state/modality_merge.json
candidate_count: 5500
measured_epoch_count: 16500
```

Do not merge debug workspaces, partial pilots, differently pinned revisions, or
an old modality receipt with a newly changed workspace.

## 14. Evidence checklist

### Offline development evidence

- [ ] `plan.md` contains the appended English implementation plan.
- [ ] All shell scripts pass `bash -n`.
- [ ] ShellCheck passes when installed, or its absence is recorded.
- [ ] All four Pod and Job renders are deterministic and match golden fixtures.
- [ ] YAML parses with `yaml.safe_load_all`.
- [ ] Pod/Job kind, one-GPU resources, A10 affinity, PVC, Secret reference,
      command, and restart policy pass structural assertions.
- [ ] Render actions make zero fake Kubernetes calls.
- [ ] Submit/status/monitor actions use only the injected fake executable and
      use the required command order.
- [ ] Secret fixture values appear in neither YAML, logs, nor errors.
- [ ] Dataset tests prove 1,300/950/1,300/1,950 and the exact 5,500 union.
- [ ] Merge tests reject missing, duplicate, overlap, tampering, and different
      pins.
- [ ] Existing AWS A10G hash and behavior regressions pass.
- [ ] Repository unit tests and `git diff --check` pass.

### Real Nautilus evidence, to be collected later by the operator

- [ ] Current A10 product label and CUDA-compatible driver are confirmed.
- [ ] Namespace, Kaggle rules, Secret, and bound 700 GiB RWX PVC are confirmed.
- [ ] Rendered immutable source/image pins are reviewed before apply.
- [ ] Pilot get/describe/log/event evidence is captured for each modality.
- [ ] GPU utilization and CPU/RAM sizing comply with current NRP policy.
- [ ] A tracked monitor log exists under `record/` for every real launch.
- [ ] Every failed Job report includes Pod, exit code, logs, events, and action.
- [ ] Four production completion receipts pass current-workspace verification.
- [ ] Final merge receipt states 5,500 labels and 16,500 measured epochs.
- [ ] Debug Pods are deleted and disposable debug data is excluded.

## 15. Development handoff statement

**YAML and orchestration logic were verified offline; no Nautilus resources were queried, created, modified, or deleted.**
