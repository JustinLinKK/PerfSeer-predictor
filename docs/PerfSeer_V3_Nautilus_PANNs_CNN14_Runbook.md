# Submit the 550-row PANNs CNN14 label Job to Nautilus

## Scope

This runbook submits exactly this frozen quota cell from
`src/perfseer_v3/configs/a10g_18k_dataset_pack.yaml`:

```yaml
{modality: audio, family_id: panns_cnn14, display_name: PANNs CNN14, accepted_configurations: 550}
```

The family contract selects exactly 550 unique frozen candidate IDs across
`mlsp-2013-birds` and `icml-2013-whale`. Each accepted configuration runs five
epochs in a fresh process; epochs 3–5 provide 1,650 retained epoch
measurements. TCN, M5, and every non-audio family are excluded.

The Job follows the current NRP guidance for [finite batch
Jobs](https://nrp.ai/documentation/userdocs/running/jobs/), [generic GPU
requests and product affinity](https://nrp.ai/documentation/userdocs/running/gpu-pods/),
[cluster resource policy](https://nrp.ai/documentation/userdocs/start/policies/),
and [CephFS storage](https://nrp.ai/documentation/userdocs/storage/ceph/).
The container installs packages under `/tmp`, never on CephFS.

## Prepared entry point

Use only this wrapper for this family:

```text
./submit_nautilus_a10_panns_cnn14.sh
```

It supports `analyze`, `render-pilot-job`, `submit-pilot-job`,
`render-production-job`, `submit-production-job`, `status`, `monitor`, and
`verify`. Render and analyze actions are local-only. Submit actions call
`kubectl apply`, immediately collect Job, Pod, describe, log, and event
feedback, and start the required local `nohup` monitor.

## Step 1: prepare the immutable source revision

The Job clones an immutable commit and refuses a dirty checkout. Commit the
new PANNs family workflow and push that commit to a repository reachable from
Nautilus. Do not use a branch name for `REVISION`.

```bash
git status --short
git rev-parse HEAD
git remote -v
```

If the repository is private, arrange an approved read-only clone mechanism
before submission. Never embed a Git credential in `REPOSITORY_URL`.

## Step 2: prepare Kaggle access

The Kaggle account must have accepted the rules for both competitions:

- [MLSP 2013 Bird Classification Challenge — rules and agreement](https://www.kaggle.com/c/mlsp-2013-birds/rules)
- [The ICML 2013 Whale Challenge — rules and agreement](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/rules)

Sign in to the same Kaggle account represented by `kaggle.json`, open each
link, and accept/join where Kaggle prompts. Download a legacy `kaggle.json`
from [Kaggle API settings](https://www.kaggle.com/settings/api), keep it outside
this repository with mode `0600`, then create a Secret whose key is exactly
`kaggle.json`:

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

Do not inspect the Secret with `-o yaml` and do not commit `kaggle.json`.
The Job YAML contains only the Secret name. Kubernetes mounts only its
`kaggle.json` key at `/var/run/secrets/perfseer-kaggle/kaggle.json`, read-only
with mode `0400`; the Job sets `KAGGLE_CONFIG_DIR` to that directory.

You do not need to download the archives manually: after the agreements are
accepted, the pilot/production Job inventories and downloads both competitions
into its isolated PVC workspace. To confirm access locally without downloading
the data, run:

```bash
export KAGGLE_CONFIG_DIR="$(dirname "$KAGGLE_JSON")"
kaggle competitions files -c mlsp-2013-birds
kaggle competitions files \
  -c the-icml-2013-whale-challenge-right-whale-redux
```

## Step 3: prepare persistent storage

The existing workflow enforces its frozen 600 GiB cloud working-set gate, so
use a 700 GiB PVC even though the two compressed audio archives are much
smaller. `rook-cephfs` is the current US West RWX storage class; use the class
specified by the namespace owner if different.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perfseer-v3-rwx
  namespace: replace-with-your-namespace
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 700Gi
  storageClassName: rook-cephfs
```

Apply that PVC manifest, then wait for it to be usable:

```bash
kubectl apply -f /absolute/path/to/perfseer-v3-rwx.yaml
kubectl get pvc perfseer-v3-rwx --namespace "$NAMESPACE"
kubectl describe pvc perfseer-v3-rwx --namespace "$NAMESPACE"
```

Never run two writers with the same PANNs run ID. The production workspace is
`/pvc/perfseer-v3/families/panns_cnn14/<run-id>`.

## Step 4: set the submission arguments

The image is the project-pinned PyTorch CUDA image. Keep its immutable digest.

```bash
export PVC_NAME='perfseer-v3-rwx'
export REPOSITORY_URL='https://github.com/your-org/PerfSeer-predictor.git'
export REVISION='replace-with-the-pushed-40-character-lowercase-commit'
export RUN_ID='panns-001'
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
```

Check access and the live A10/CUDA labels before rendering. The static NRP GPU
table can lag the cluster, so use the exact live product label if it differs
from `NVIDIA-A10`.

```bash
kubectl config current-context
kubectl auth can-i create jobs.batch --namespace "$NAMESPACE"
kubectl auth can-i get pods --namespace "$NAMESPACE"
kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE"
kubectl get secret "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE"
kubectl get nodes -l nvidia.com/gpu.product \
  -L nvidia.com/gpu.product,nvidia.com/cuda.driver.major,nvidia.com/cuda.runtime.major,nvidia.com/cuda.runtime.minor
```

If the live A10 product label is different, append
`--gpu-product 'LIVE-LABEL'` to `COMMON_ARGS`.

## Step 5: verify the exact family and render locally

These commands do not contact Kubernetes:

```bash
./submit_nautilus_a10_panns_cnn14.sh analyze
./submit_nautilus_a10_panns_cnn14.sh \
  render-pilot-job "${COMMON_ARGS[@]}" > record/panns_cnn14_pilot.yaml
./submit_nautilus_a10_panns_cnn14.sh \
  render-production-job "${COMMON_ARGS[@]}" > record/panns_cnn14_production.yaml
```

Confirm the analysis says `candidate_count: 550`,
`accepted_configurations: 550`, and `measured_epoch_count: 1650`. Inspect the
rendered YAML:

```bash
python3 - <<'PY'
import yaml

for path in ("record/panns_cnn14_pilot.yaml", "record/panns_cnn14_production.yaml"):
    job = yaml.safe_load(open(path, encoding="utf-8"))
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row["value"] for row in container["env"]}
    assert job["kind"] == "Job"
    assert env["PERFSEER_FAMILY_ID"] == "panns_cnn14"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert container["image"].startswith("pytorch/pytorch:")
print("rendered PANNs manifests: PASS")
PY
```

## Step 6: submit the bounded pilot

Use the same `RUN_ID` for the pilot and production Job so production resumes
the pilot’s PVC workspace. The pilot stops after eight new accepted labels.

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  submit-pilot-job "${COMMON_ARGS[@]}"
```

The wrapper immediately prints the Job and Pod state, Pod description, last
logs, and recent events. It also starts this append-only local monitor:

```text
record/nautilus_a10_panns_cnn14_<run-id>_monitor.log
```

The monitor samples every 60 seconds for the first five iterations and every
20 minutes afterward. It records Job/Pod status, events, container logs,
processes, persistent attempt-log tails, state, accepted outputs, and
checkpoints. Keep the operator shell available during the initial scheduling
period and report progress at least once per minute while actively supervising
the launch.

Check the pilot at any time:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode pilot
tail -n 200 "record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log"
```

Do not launch production until the pilot Job completes successfully and its
logs show accepted label records. A failed Job reports the Pod, exit code,
last logs, and next corrective action; correct the root cause before resuming.

## Step 7: submit all 550 PANNs configurations

Production has no `--max-new-accepted` bound and resumes the exact same family
contract and workspace:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  submit-production-job "${COMMON_ARGS[@]}"
```

Check it with:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode production
tail -n 200 "record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log"
```

On successful completion, the workflow independently verifies that the
workspace contains exactly 550 accepted records and valid A10/A10G physical
provenance, then publishes:

```text
/pvc/perfseer-v3/families/panns_cnn14/<run-id>/workspace/state/family_completion.json
```

The completion must report `candidate_count: 550` and
`measured_epoch_count: 1650`. Keep the Job, monitor log, family contract,
completion receipt, accepted records, failure evidence, and hardware
provenance until the labels have been merged into the larger campaign.

## Resume and cleanup rules

- To resume after eviction or a corrected failure, use a new Kubernetes Job
  resource name but the same immutable revision, image digest, PVC, and run ID.
  If the old Job name still exists, delete only that exact Job after preserving
  its diagnostics, then resubmit the same mode.
- Never change `REVISION`, `IMAGE`, or the PANNs family contract inside an
  existing workspace.
- Do not delete the PVC workspace merely because the Pod exited.
- After the completion artifacts have been copied and independently checked,
  delete the finished Kubernetes Jobs to release cluster bookkeeping objects.
