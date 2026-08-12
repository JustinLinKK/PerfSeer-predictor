# Temporary operator guide: label PANNs CNN14 in a shared Nautilus namespace

This is a copy/paste-oriented guide for submitting exactly this frozen dataset
pack cell:

```yaml
{modality: audio, family_id: panns_cnn14, display_name: PANNs CNN14, accepted_configurations: 550}
```

The production Job accepts exactly 550 configurations. Each accepted
configuration has five training epochs; epochs 3–5 produce 1,650 retained
measured-epoch records. The workflow uses the repository's prepared PyTorch
CUDA Job wrapper:

```text
./submit_nautilus_a10_panns_cnn14.sh
```

This guide does not submit anything by itself. Commands marked **cluster
write** create or delete Kubernetes resources when you run them.

## Shared-namespace security bottom line

This guide assumes you **must use a shared namespace**. If other people can
create Pods, Deployments, or Jobs there, the namespace is not a private
boundary for `kaggle.json`. Kubernetes explicitly warns that a person who can
create a Pod in a namespace can arrange for that Pod to read any Secret in the
namespace. A person with `get`, `list`, or `watch` access to Secrets may also
read Secret data.

The practical shared-namespace policy is therefore:

1. Use a freshly generated legacy Kaggle API key for this run, use it nowhere
   else, keep it in the namespace only while required, and revoke it after the
   Job. This limits the useful lifetime of a credential that a sufficiently
   authorized collaborator could read while it exists.
2. The Secret is randomized, create-only, and `immutable: true`.
   Immutability prevents changes to its `data`, but it does not prevent an
   authorized user from deleting and recreating it or changing metadata.
3. Capture the Secret UID and resource version after creation and verify both
   before each submission and cleanup action. This detects replacement or
   metadata changes when checked; it does not prevent them.
4. Use unique per-run Job, Pod, Secret, PVC, and workspace names. Never update,
   patch, reuse, or delete a resource that fails its ownership checks.
5. The Job still consumes one GPU, 8 CPUs, 32 GiB memory, namespace quota, and
   storage. In a shared namespace, coordinate this allocation with its owner;
   no naming convention can eliminate quota and scheduling impact.

This procedure minimizes accidental interference. It cannot make a Kubernetes
Secret confidential from other users who can create Pods in the shared
namespace. If that remaining exposure is unacceptable, the correct solution
requires a different credential-delivery design or administrator-enforced
isolation; Secret naming and immutability cannot provide it.

Official references:

- [Kubernetes Secrets and their namespace security boundary](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Kubernetes immutable Secrets](https://kubernetes.io/docs/concepts/configuration/secret/#immutable-secrets)
- [NRP Nautilus batch Jobs](https://nrp.ai/documentation/userdocs/running/jobs/)
- [NRP Nautilus GPU Pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
- [NRP Nautilus Ceph storage](https://nrp.ai/documentation/userdocs/storage/ceph/)

### What the safeguards do

| Safeguard | What it prevents | What it cannot prevent |
|---|---|---|
| Fresh run-only Kaggle key, promptly revoked | Long-lived use of a credential copied during the run | Reading or using it before revocation |
| Unique random run, Secret, PVC, Job, and Pod names | Accidental collision with somebody else's named resource | Deliberate access by a sufficiently authorized user |
| `kubectl create`, never `apply`, for the Secret and PVC | Silently overwriting an existing Secret or PVC | Somebody deleting and recreating the object |
| `immutable: true` on the Secret | Accidental or unwanted edits to Secret data | Secret deletion, metadata edits, or reading through another Pod |
| Saved UID and resource-version checks | Continuing after a detected replacement or metadata change | Preventing the change or detecting it between checks |
| Read-only `0400` Secret volume | The label container modifying the mounted file | A different authorized Pod mounting the same Secret |
| Unique per-run PVC and workspace path | Accidental writes into another PerfSeer run directory | Namespace users who are permitted to mount that PVC |

Do not try to solve shared-namespace isolation by creating another Role for
yourself. Kubernetes RBAC permissions are additive; a new narrow Role does not
remove permissions that other users already received from other RoleBindings.

## Step 1: confirm the shared namespace and reserve its quota

Notify the namespace owner and collaborators before creating the 700 GiB PVC
or requesting the GPU. Use a message like this:

> I will run one finite PANNs CNN14 PyTorch CUDA labeling campaign in our shared
> namespace. It requests one NVIDIA A10 24 GiB GPU, 8 CPU, 32 GiB memory, and a
> unique 700 GiB PVC. Its Job, Pod, PVC, workspace, and immutable Secret names
> contain my owner tag and a random run ID. I will not reuse or modify existing
> objects, and I will delete only resources whose run labels and UIDs I have
> verified. Please confirm the quota, storage class, and timing will not disrupt
> another campaign.

Do not create the PVC or submit the Job until the namespace owner confirms the
quota and timing. The Secret itself uses little quota, but the 700 GiB PVC and
one-GPU request can affect other namespace users.

## Step 2: accept both Kaggle competition agreements

Sign in to the Kaggle account represented by your `kaggle.json`, open both
links, and accept/join each competition:

1. [MLSP 2013 Bird Classification Challenge](https://www.kaggle.com/c/mlsp-2013-birds/rules)
2. [The ICML 2013 Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/rules)

Under **Legacy API Credentials** in [Kaggle API
settings](https://www.kaggle.com/settings/api), create a fresh legacy API key
for this run and download its `kaggle.json`. This is the credential format the
prepared Job mounts; it matches the [official Kaggle CLI authentication
instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
Do not reuse this run key in notebooks, other clusters, or local automation:
you will revoke it after the labeling Job.

Keep the downloaded file outside this Git repository. The repository ignores
`kaggle.json`, but an ignored credential is still safer outside the repository
entirely. Creating or later revoking a legacy key may affect another system if
that system uses the same key, which is why this run key must not be shared
with any other workflow.

```bash
export KAGGLE_JSON='/absolute/path/outside/PerfSeer-predictor/kaggle.json'
test -f "$KAGGLE_JSON"
chmod 600 "$KAGGLE_JSON"
```

Turn off shell tracing before handling the credential:

```bash
set +x
```

If the Kaggle CLI is available locally, verify authorization without
downloading either dataset:

```bash
KAGGLE_CONFIG_DIR="$(dirname "$KAGGLE_JSON")" \
  kaggle competitions files -c mlsp-2013-birds
KAGGLE_CONFIG_DIR="$(dirname "$KAGGLE_JSON")" \
  kaggle competitions files \
    -c the-icml-2013-whale-challenge-right-whale-redux
```

Both commands must list competition files. An agreement/authentication error
must be fixed before creating the cluster Secret.

Never run any of the following:

```text
git add -f /path/to/kaggle.json
kubectl get secret SECRET_NAME -o yaml
kubectl get secret SECRET_NAME -o json
```

The first can publish the credential. The latter two print base64-encoded
credential data, which is encoding rather than encryption.

## Step 3: prepare an immutable source revision

The Job clones the repository and checks out one exact 40-character commit.
The commit must contain the PANNs wrapper and its dependencies and must be
reachable from Nautilus. Use a public read-only HTTPS URL unless you have
separately configured an approved private-repository clone mechanism.

```bash
cd /home/justin/PerfSeer-predictor

git status --short --branch
git log -1 --oneline
git remote -v

export REPOSITORY_URL='https://github.com/YOUR_ORG/PerfSeer-predictor.git'
export REVISION="$(git rev-parse HEAD)"
test "${#REVISION}" -eq 40
```

Confirm that this exact commit is pushed. For an `origin` remote:

```bash
git fetch origin
git branch --remotes --contains "$REVISION"
```

The output must show the remote branch you intend Nautilus to clone. Do not
pass a branch name as `REVISION`, and do not put a username, password, or token
inside `REPOSITORY_URL`.

## Step 4: create unique names for only this run

Choose a short lowercase owner tag. The random suffix makes accidental name
reuse extremely unlikely. Keep this shell open: later commands use these
variables.

```bash
export NAMESPACE='replace-with-your-shared-namespace'
export OWNER_TAG='justin'
export RUN_TOKEN="$(date -u +%y%m%d)-$(openssl rand -hex 3)"
export RUN_ID="${OWNER_TAG}-${RUN_TOKEN}"
export KAGGLE_SECRET_NAME="perfseer-kaggle-${RUN_ID}"
export PVC_NAME="perfseer-panns-${RUN_ID}"

export PILOT_JOB="perfseer-a10-panns-cnn14-pilot-${RUN_ID}"
export PRODUCTION_JOB="perfseer-a10-panns-cnn14-production-${RUN_ID}"
export AUDIT_POD="perfseer-panns-audit-${RUN_ID}"

export IMAGE='pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89'
export CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf 'namespace: %s\nrun: %s\nsecret: %s\npvc: %s\n' \
  "$NAMESPACE" "$RUN_ID" "$KAGGLE_SECRET_NAME" "$PVC_NAME"
```

Use no more than ten lowercase letters, digits, or hyphens in `OWNER_TAG`.
The renderer will reject invalid or overlong Kubernetes names.

Shell variables disappear when you open a new terminal. If you resume later,
restore the **same** namespace, run ID, names, repository revision, and image;
do not generate another random run ID for an existing workspace or rendered
manifest.

Define this local-only fail-closed target guard. It rejects missing variables,
the `default` namespace, an unchanged placeholder, and resource names that do
not match the run ID:

```bash
require_safe_target() {
  local variable_name
  for variable_name in \
    NAMESPACE OWNER_TAG RUN_ID KAGGLE_SECRET_NAME PVC_NAME \
    PILOT_JOB PRODUCTION_JOB AUDIT_POD; do
    if ! declare -p "$variable_name" >/dev/null 2>&1; then
      printf 'STOP: %s is not set in this shell\n' "$variable_name" >&2
      return 1
    fi
    if [[ -z ${!variable_name} ]]; then
      printf 'STOP: %s is empty in this shell\n' "$variable_name" >&2
      return 1
    fi
  done

  case "$NAMESPACE" in
    default|replace-*)
      printf 'STOP: refusing unsafe namespace %s\n' "$NAMESPACE" >&2
      return 1
      ;;
  esac

  [[ $KAGGLE_SECRET_NAME == "perfseer-kaggle-${RUN_ID}" ]] || return 1
  [[ $PVC_NAME == "perfseer-panns-${RUN_ID}" ]] || return 1
  [[ $PILOT_JOB == "perfseer-a10-panns-cnn14-pilot-${RUN_ID}" ]] || return 1
  [[ $PRODUCTION_JOB == "perfseer-a10-panns-cnn14-production-${RUN_ID}" ]] || return 1
  [[ $AUDIT_POD == "perfseer-panns-audit-${RUN_ID}" ]] || return 1

  printf 'safe target: namespace=%s run=%s\n' "$NAMESPACE" "$RUN_ID"
}

require_safe_target
```

Do not continue unless this prints `safe target`. Every later cluster-write
command is chained to this guard. If the function is missing because you
opened a new shell, the write will stop with `command not found` rather than
silently falling back to `default`.

Define a guard that fails if a resource name is already occupied:

```bash
must_not_exist() {
  local resource_kind=$1
  local resource_name=$2
  require_safe_target || return 1
  if kubectl get "$resource_kind" "$resource_name" \
      --namespace "$NAMESPACE" >/dev/null 2>&1; then
    printf 'STOP: %s/%s already exists in %s\n' \
      "$resource_kind" "$resource_name" "$NAMESPACE" >&2
    return 1
  fi
}
```

Check every name now:

```bash
must_not_exist secret "$KAGGLE_SECRET_NAME"
must_not_exist pvc "$PVC_NAME"
must_not_exist job "$PILOT_JOB"
must_not_exist job "$PRODUCTION_JOB"
must_not_exist pod "$AUDIT_POD"
```

If any guard says `STOP`, do not modify or delete the existing object. Generate
a new `RUN_TOKEN` and repeat this step.

## Step 5: perform read-only cluster preflight

These commands do not create or change cluster objects:

```bash
require_safe_target && {
kubectl config current-context
kubectl get namespace "$NAMESPACE"

kubectl auth can-i create jobs.batch --namespace "$NAMESPACE"
kubectl auth can-i get jobs.batch --namespace "$NAMESPACE"
kubectl auth can-i get pods --namespace "$NAMESPACE"
kubectl auth can-i get pods/log --namespace "$NAMESPACE"
kubectl auth can-i create secrets --namespace "$NAMESPACE"
kubectl auth can-i delete secrets --namespace "$NAMESPACE"
kubectl auth can-i create persistentvolumeclaims --namespace "$NAMESPACE"

kubectl get resourcequota --namespace "$NAMESPACE"
kubectl get limitrange --namespace "$NAMESPACE"
kubectl get storageclass rook-cephfs
kubectl get nodes -l nvidia.com/gpu.product \
  -L nvidia.com/gpu.product,nvidia.com/cuda.driver.major,nvidia.com/cuda.runtime.major,nvidia.com/cuda.runtime.minor
}
```

Every required `can-i` result must be `yes`. These checks reveal only your own
permissions. Because this is a shared namespace, assume that other authorized
users can create Pods and could therefore read the Secret while it exists.
The commands below reduce accidental interference and credential lifetime;
they do not change that trust boundary.

The current workflow defaults to the live product label `NVIDIA-A10`. If the
node listing shows a different exact label for A10 24 GiB nodes, record it and
later add `--gpu-product 'THE-LIVE-LABEL'` to `COMMON_ARGS`.

## Step 6: verify the exact local family contract

This is local-only and does not contact Kubernetes:

```bash
./submit_nautilus_a10_panns_cnn14.sh analyze \
  | tee "record/panns_cnn14_${RUN_ID}_analysis.json"
```

Verify the result mechanically:

```bash
python3 - "$RUN_ID" <<'PY'
import json
from pathlib import Path
import sys

path = Path("record") / f"panns_cnn14_{sys.argv[1]}_analysis.json"
value = json.loads(path.read_text(encoding="utf-8"))
assert value["family_id"] == "panns_cnn14"
assert value["modality"] == "audio"
assert value["candidate_count"] == 550
assert value["accepted_configurations"] == 550
assert value["measured_epoch_count"] == 1650
print("exact PANNs CNN14 contract: PASS")
PY
```

Do not continue unless it prints `PASS`.

## Step 7: render and inspect both Jobs locally

Prepare the common arguments:

```bash
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

If Step 5 found a different live A10 label, add it now:

```bash
# COMMON_ARGS+=(--gpu-product 'THE-LIVE-A10-LABEL')
```

Render without submitting:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  render-pilot-job "${COMMON_ARGS[@]}" \
  >"record/panns_cnn14_${RUN_ID}_pilot.yaml"

./submit_nautilus_a10_panns_cnn14.sh \
  render-production-job "${COMMON_ARGS[@]}" \
  >"record/panns_cnn14_${RUN_ID}_production.yaml"
```

Run this manifest verifier. It checks the family, resource names, pinned
image, one-GPU request, PVC, read-only Secret file, and pilot bound. It also
checks that neither YAML contains Secret data:

```bash
python3 - "$RUN_ID" "$NAMESPACE" "$KAGGLE_SECRET_NAME" "$PVC_NAME" "$IMAGE" <<'PY'
from pathlib import Path
import sys
import yaml

run_id, namespace, secret_name, pvc_name, image = sys.argv[1:]

cases = {
    "pilot": (f"perfseer-a10-panns-cnn14-pilot-{run_id}", True),
    "production": (f"perfseer-a10-panns-cnn14-production-{run_id}", False),
}
for mode, (expected_name, bounded) in cases.items():
    path = Path("record") / f"panns_cnn14_{run_id}_{mode}.yaml"
    text = path.read_text(encoding="utf-8")
    job = yaml.safe_load(text)
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == expected_name
    assert job["metadata"]["namespace"] == namespace
    assert "data" not in job and "stringData" not in job

    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    env = {row["name"]: row["value"] for row in container["env"]}
    assert env["PERFSEER_FAMILY_ID"] == "panns_cnn14"
    assert env["PERFSEER_MODALITY"] == "audio"
    assert env["KAGGLE_CONFIG_DIR"] == "/var/run/secrets/perfseer-kaggle"
    assert not any(name in env for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"))
    assert container["image"] == image
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == pvc_name

    secret = pod["volumes"][1]["secret"]
    assert secret["secretName"] == secret_name
    assert secret["defaultMode"] == 0o400
    assert secret["items"] == [
        {"key": "kaggle.json", "path": "kaggle.json", "mode": 0o400}
    ]
    mount = container["volumeMounts"][1]
    assert mount["readOnly"] is True
    script = container["args"][0]
    assert ("--max-new-accepted 8" in script) is bounded

print("pilot and production manifests: PASS")
PY
```

Do not continue unless it prints `PASS`. Do not publish rendered manifests if
your namespace or Secret name is sensitive. They contain the Secret name, but
not `kaggle.json` or its contents.

## Step 8: create a unique per-run PVC

The workflow enforces a frozen 600 GiB working-space gate, so this guide uses a
700 GiB PVC with a randomized per-run name. This avoids accidental path/name
reuse but does not make the files private from namespace users who may mount
the PVC. Obtain the namespace owner's approval before consuming that storage
quota. The example uses US West `rook-cephfs`; replace it only with the class
confirmed by the administrator.

Re-run the collision guard immediately before creation:

```bash
must_not_exist pvc "$PVC_NAME"
```

If an error names namespace `default`, stop. It means `NAMESPACE` was empty,
lost when a new shell was opened, or not applied to the object. A `Forbidden`
response to `create` means that request created no PVC. Restore the intended
shared namespace and the original run variables, rerun `require_safe_target`,
and repeat the Step 5 permission check for the intended namespace before any
new create attempt. Permission in `default` says nothing about permission in
the intended shared namespace.

**Cluster write — creates only the unique PVC and fails if it already exists:**

```bash
require_safe_target &&
kubectl create --namespace "$NAMESPACE" --filename - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: perfseer-v3-labeling
    app.kubernetes.io/component: panns-cnn14
    perfseer.ai/run-id: ${RUN_ID}
  annotations:
    perfseer.ai/owner: ${OWNER_TAG}
    perfseer.ai/purpose: unique-panns-cnn14-labeling-workspace
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 700Gi
  storageClassName: rook-cephfs
EOF
```

Verify the exact PVC before continuing:

```bash
kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE" --output wide
kubectl describe pvc "$PVC_NAME" --namespace "$NAMESPACE"
test "$(kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID"
```

Wait until its status is `Bound`. Do not substitute a similarly named shared
PVC, and never run two writers with the same `RUN_ID`.

## Step 9: create the unique immutable Kaggle Secret

Create this as late as practical, after the PVC is ready and the rendered
Jobs passed inspection. The command streams the generated Secret object
directly from one process to another; it does not write a Secret manifest into
the repository or `/tmp`.

Re-run the collision guard immediately before creation:

```bash
must_not_exist secret "$KAGGLE_SECRET_NAME"
```

**Cluster write — creates only the unique Secret and fails if it already
exists:**

```bash
require_safe_target &&
kubectl create secret generic "$KAGGLE_SECRET_NAME" \
  --namespace "$NAMESPACE" \
  --from-file=kaggle.json="$KAGGLE_JSON" \
  --dry-run=client \
  --output json \
| python3 -c '
import json
import sys

run_id, owner_tag, created_at = sys.argv[1:]
value = json.load(sys.stdin)
value["immutable"] = True
metadata = value.setdefault("metadata", {})
metadata.setdefault("labels", {}).update({
    "app.kubernetes.io/name": "perfseer-v3-labeling",
    "app.kubernetes.io/component": "panns-cnn14",
    "perfseer.ai/run-id": run_id,
})
metadata.setdefault("annotations", {}).update({
    "perfseer.ai/owner": owner_tag,
    "perfseer.ai/created-at": created_at,
    "perfseer.ai/purpose": "panns-cnn14-kaggle-access",
    "perfseer.ai/do-not-modify": "immutable; contact the owner",
})
json.dump(value, sys.stdout)
' "$RUN_ID" "$OWNER_TAG" "$CREATED_AT" \
| kubectl create --namespace "$NAMESPACE" --filename -
```

This intentionally uses `kubectl create`, not `kubectl apply`. A race or name
collision fails instead of modifying an object.

Verify metadata, immutability, and key names without printing the credential:

```bash
export KAGGLE_SECRET_UID="$(kubectl get secret "$KAGGLE_SECRET_NAME" \
  --namespace "$NAMESPACE" -o jsonpath='{.metadata.uid}')"
export KAGGLE_SECRET_RESOURCE_VERSION="$(kubectl get secret \
  "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.resourceVersion}')"

test -n "$KAGGLE_SECRET_UID"
test -n "$KAGGLE_SECRET_RESOURCE_VERSION"
test "$(kubectl get secret "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE" \
  -o go-template='{{range $key, $_ := .data}}{{$key}}{{"\n"}}{{end}}')" \
  = kaggle.json

kubectl describe secret "$KAGGLE_SECRET_NAME" --namespace "$NAMESPACE"
```

Define the identity verifier used for every later Secret check:

```bash
verify_kaggle_secret_identity() {
  local actual
  actual="$(kubectl get secret "$KAGGLE_SECRET_NAME" \
    --namespace "$NAMESPACE" \
    -o jsonpath='{.metadata.uid}{"|"}{.metadata.resourceVersion}{"|"}{.immutable}{"|"}{.metadata.labels.perfseer\.ai/run-id}')" \
    || return 1

  if [ "$actual" != "${KAGGLE_SECRET_UID}|${KAGGLE_SECRET_RESOURCE_VERSION}|true|${RUN_ID}" ]; then
    printf 'STOP: Kaggle Secret identity, metadata, or immutability changed\n' >&2
    return 1
  fi
  printf 'Kaggle Secret identity: PASS\n'
}

verify_kaggle_secret_identity
```

`describe` should show exactly one data key named `kaggle.json` and its byte
count, not its value. If the key-name test or identity verifier fails, do not
submit or delete anything under that name. Save the observed metadata, contact
the namespace owner, and create a new randomly named Secret only after the
conflict is understood. Do not patch credential data in place.

UID/resource-version checks catch replacement or metadata changes only when
you run them. Another authorized user can still change the object immediately
after a successful check; this unavoidable race is part of using a shared
namespace.

## Step 10: submit the bounded eight-label pilot

The pilot and production use the same `RUN_ID` and PVC workspace so production
resumes the eight accepted pilot configurations rather than repeating them.

Immediately before submission, prove that the Job name is unused and the two
dependencies are the exact resources you created:

```bash
must_not_exist job "$PILOT_JOB"
verify_kaggle_secret_identity
test "$(kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID"
```

For the strict shared-namespace path, submit the already verified manifest with
`kubectl create`. Do not use the wrapper's `submit-pilot-job` action here: that
action uses `apply`, whereas `create` guarantees that an unexpected name
collision fails instead of updating an existing Job.

**Cluster write — creates only the pilot Job:**

```bash
require_safe_target &&
kubectl create --namespace "$NAMESPACE" \
  --filename "record/panns_cnn14_${RUN_ID}_pilot.yaml"
```

Immediately collect the mandatory Job status, Pod status, Pod description,
Pod logs, and recent events, then start the durable local `nohup` monitor:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode pilot

nohup ./submit_nautilus_a10_panns_cnn14.sh \
  monitor "${COMMON_ARGS[@]}" --job-mode pilot \
  >>"record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log" \
  2>&1 </dev/null &
printf 'pilot monitor PID: %s\n' "$!"
```

If the first status call runs before the Pod exists, rerun it immediately
until the Pod appears and the description, logs, and Pod-specific events have
all been collected.

The monitor log is:

```text
record/nautilus_a10_panns_cnn14_<run-id>_monitor.log
```

For this run, inspect it with:

```bash
tail -n 200 "record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log"
```

While the launch is active, check and report progress at least once every 60
seconds until the Job is stably running, complete, or failed:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode pilot
verify_kaggle_secret_identity
```

The background monitor samples every 60 seconds for its first five iterations
and every 20 minutes afterward. It records Job/Pod status, events, logs,
process state, persistent attempt-log tails, accepted output files, state, and
checkpoints.

Pilot success means:

- `kubectl get job "$PILOT_JOB"` reports `Complete`, not `Failed`;
- logs show no Kaggle authorization, CUDA, OOM, or data-integrity failure; and
- the monitor shows accepted record files being written under this run's
  unique PVC workspace.

Verify the Kubernetes completion condition mechanically:

```bash
test "$(kubectl get job "$PILOT_JOB" --namespace "$NAMESPACE" \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
```

Do not submit production while the pilot is still running.

## Step 11: submit the production Job for all 550 labels

The production Job has no `--max-new-accepted` limit. It resumes the same
workspace at:

```text
/pvc/perfseer-v3/families/panns_cnn14/<run-id>/workspace
```

Immediately before submission:

```bash
must_not_exist job "$PRODUCTION_JOB"
kubectl get job "$PILOT_JOB" --namespace "$NAMESPACE" --output wide
verify_kaggle_secret_identity
```

Continue only if the pilot condition is `Complete`.

Again use create-only submission so any collision fails rather than updating a
Job.

**Cluster write — creates only the production Job:**

```bash
require_safe_target &&
kubectl create --namespace "$NAMESPACE" \
  --filename "record/panns_cnn14_${RUN_ID}_production.yaml"
```

Immediately collect feedback, start the production monitor, and append to the
same tracked monitor log:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode production

nohup ./submit_nautilus_a10_panns_cnn14.sh \
  monitor "${COMMON_ARGS[@]}" --job-mode production \
  >>"record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log" \
  2>&1 </dev/null &
printf 'production monitor PID: %s\n' "$!"
```

If the first status call runs before the Pod exists, rerun it immediately
until the Pod description, logs, and Pod-specific events are included.

During active supervision:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode production
verify_kaggle_secret_identity

tail -n 200 "record/nautilus_a10_panns_cnn14_${RUN_ID}_monitor.log"
```

Do not treat the existence of the monitor file as proof of success. The
production Job must reach `Complete`, and the persistent completion receipt
must pass the next step.

Verify the Kubernetes completion condition before creating the audit Pod:

```bash
test "$(kubectl get job "$PRODUCTION_JOB" --namespace "$NAMESPACE" \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')" = True
```

## Step 12: independently inspect the completion receipt

The production workflow verifies the entire family before it exits
successfully and writes:

```text
/pvc/perfseer-v3/families/panns_cnn14/<run-id>/workspace/state/family_completion.json
```

Use a finite, CPU-only audit Pod to read only that receipt. This Pod does not
mount the Kaggle Secret and requests no GPU.

Render its YAML locally:

```bash
must_not_exist pod "$AUDIT_POD"

cat >"record/panns_cnn14_${RUN_ID}_audit.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${AUDIT_POD}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: perfseer-v3-labeling-audit
    app.kubernetes.io/component: panns-cnn14
    perfseer.ai/run-id: ${RUN_ID}
spec:
  restartPolicy: Never
  containers:
    - name: verifier
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      env:
        - name: PERFSEER_RUN_ID
          value: ${RUN_ID}
      command:
        - python
        - -c
      args:
        - |
          import json, os
          from pathlib import Path
          path = (Path('/pvc/perfseer-v3/families/panns_cnn14') /
                  os.environ['PERFSEER_RUN_ID'] /
                  'workspace/state/family_completion.json')
          value = json.loads(path.read_text(encoding='utf-8'))
          assert value['family_id'] == 'panns_cnn14'
          assert value['modality'] == 'audio'
          assert value['candidate_count'] == 550
          assert value['measured_epoch_count'] == 1650
          assert len(value['accepted_record_sha256s']) == 550
          print(json.dumps({
              'family_id': value['family_id'],
              'candidate_count': value['candidate_count'],
              'measured_epoch_count': value['measured_epoch_count'],
              'completion_sha256': value['completion_sha256'],
          }, sort_keys=True))
      resources:
        requests:
          cpu: 100m
          memory: 256Mi
        limits:
          cpu: 1
          memory: 1Gi
      volumeMounts:
        - name: workspace
          mountPath: /pvc
          readOnly: true
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
EOF
```

Verify locally that the audit Pod has no Secret or GPU reference:

```bash
python3 - "$RUN_ID" "$AUDIT_POD" <<'PY'
from pathlib import Path
import sys
import yaml

run_id, audit_pod = sys.argv[1:]
path = Path("record") / f"panns_cnn14_{run_id}_audit.yaml"
pod = yaml.safe_load(path.read_text(encoding="utf-8"))
text = path.read_text(encoding="utf-8")
assert pod["kind"] == "Pod"
assert pod["metadata"]["name"] == audit_pod
assert "secret" not in text.lower()
assert "nvidia.com/gpu" not in text
assert pod["spec"]["containers"][0]["volumeMounts"][0]["readOnly"] is True
print("finite receipt audit Pod: PASS")
PY
```

**Cluster write — create only this finite audit Pod:**

```bash
must_not_exist pod "$AUDIT_POD"
require_safe_target &&
kubectl create --namespace "$NAMESPACE" \
  --filename "record/panns_cnn14_${RUN_ID}_audit.yaml"
```

Immediately collect feedback, just as for the labeling Jobs:

```bash
kubectl get pod "$AUDIT_POD" --namespace "$NAMESPACE" --output wide
kubectl describe pod "$AUDIT_POD" --namespace "$NAMESPACE"
kubectl logs "$AUDIT_POD" --namespace "$NAMESPACE" \
  | tee "record/panns_cnn14_${RUN_ID}_completion_summary.json"
kubectl get events --namespace "$NAMESPACE" \
  --field-selector "involvedObject.name=$AUDIT_POD" \
  --sort-by=.lastTimestamp
```

If the first log attempt says the container is still creating, wait briefly
and repeat `get`, `describe`, `logs`, and events. Success is a zero-exit Pod and
a one-line JSON summary containing `candidate_count: 550` and
`measured_epoch_count: 1650`.

## Step 13: failure and safe resume

If either labeling Job reports `Failed`, immediately record:

1. the failed Pod name;
2. its terminated container exit code;
3. the last relevant logs;
4. recent Pod events; and
5. the corrective action.

The wrapper's `status` action collects those diagnostics:

```bash
./submit_nautilus_a10_panns_cnn14.sh \
  status "${COMMON_ARGS[@]}" --job-mode production
```

Preserve the PVC. Resume with the same `RUN_ID`, PVC, `REVISION`, image digest,
and family contract. Never start another writer against the same workspace.

Before deleting a failed Job, verify both its exact name and run label:

```bash
FAILED_JOB="$PRODUCTION_JOB"  # or "$PILOT_JOB"
test "$(kubectl get job "$FAILED_JOB" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID"
kubectl get job "$FAILED_JOB" --namespace "$NAMESPACE" --output wide
```

Only after saving diagnostics, delete that exact failed Job and allow its Pod
to be removed:

```bash
require_safe_target &&
test "$(kubectl get job "$FAILED_JOB" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID" &&
kubectl delete job "$FAILED_JOB" --namespace "$NAMESPACE" --wait=true
must_not_exist job "$FAILED_JOB"
```

Then rerun the appropriate local render verifier, use `kubectl create` on that
same mode's verified YAML, collect immediate status feedback, and restart its
monitor. Do not use an `apply`-based submit action. Keep the same
`COMMON_ARGS`. If the Kaggle token itself must be rotated, do not attempt to
update the immutable Secret:

1. delete the stopped Job and its Pod first;
2. generate a new unique Secret name;
3. create a new immutable Secret using Step 9;
4. replace only `KAGGLE_SECRET_NAME` and rebuild `COMMON_ARGS`;
5. render and verify the Job again; and
6. resubmit against the unchanged PVC and `RUN_ID`.

Delete the old Secret only after no Pod references it. Never delete and
recreate a Secret under the same name; a new name makes the credential version
explicit.

## Step 14: exact-resource cleanup

Cleanup must name and verify each resource. Never delete by a broad wildcard,
never delete the namespace, and never delete a pre-existing or multi-user PVC.

### 14.1 Remove the audit Pod

After its successful log has been saved:

```bash
require_safe_target &&
test "$(kubectl get pod "$AUDIT_POD" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID" &&
kubectl delete pod "$AUDIT_POD" --namespace "$NAMESPACE" --wait=true
```

### 14.2 Remove only this run's completed Jobs and Pods

Keep them until diagnostics and completion evidence are saved. Then:

```bash
if require_safe_target; then
  for job_name in "$PILOT_JOB" "$PRODUCTION_JOB"; do
    test "$(kubectl get job "$job_name" --namespace "$NAMESPACE" \
      -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID" &&
    kubectl delete job "$job_name" --namespace "$NAMESPACE" --wait=true
  done

  kubectl get pods --namespace "$NAMESPACE" \
    --selector "perfseer.ai/run-id=$RUN_ID"
fi
```

The final command should show no remaining run Pods.

### 14.3 Delete only the immutable Kaggle Secret

Do this after all Pods that mounted it are gone. First verify that the Secret
is still the exact object you created:

```bash
verify_kaggle_secret_identity
```

Next, immediately return to [Kaggle API
settings](https://www.kaggle.com/settings/api) and revoke/delete the fresh
legacy key used for this run. Revocation is what makes a copy taken from the
shared namespace stop working. Confirm in the Kaggle UI that the run key is no
longer active, and be careful not to revoke a different key used by another
workflow.

Finally, delete the now-revoked credential from Kubernetes:

```bash
require_safe_target &&
verify_kaggle_secret_identity &&

kubectl delete secret "$KAGGLE_SECRET_NAME" \
  --namespace "$NAMESPACE" --wait=true
must_not_exist secret "$KAGGLE_SECRET_NAME"
```

Deleting the Kubernetes Secret by itself would not revoke the Kaggle key.
After both revocation and Secret deletion are confirmed, securely remove the
local run-specific `kaggle.json` when you no longer need it.

### 14.4 Retain the unique per-run PVC until results are archived

PVC deletion is intentionally not part of normal Job cleanup. It deletes the
only persistent workspace when the storage class reclaim policy removes the
volume. Keep it until the 550 accepted records, 1,650 measured epochs,
completion receipt, and provenance have been independently copied and checked.

If you later delete it, first verify that it is this guide's unique per-run PVC
and that the run label matches:

```bash
kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE" --output wide
test "$(kubectl get pvc "$PVC_NAME" --namespace "$NAMESPACE" \
  -o jsonpath='{.metadata.labels.perfseer\.ai/run-id}')" = "$RUN_ID"
```

Have the namespace owner confirm the deletion if the namespace or storage is
shared. Do not substitute a different PVC name into a deletion command.

## Final go/no-go checklist

Do not submit the pilot until every pre-submit item is true:

- [ ] I accept that authorized users who can create Pods in this shared
      namespace could read the Secret while it exists.
- [ ] The namespace owner approved one A10 GPU, 8 CPU, 32 GiB memory, and the
      700 GiB PVC allocation.
- [ ] The Kaggle account accepted both competition agreements.
- [ ] A fresh legacy Kaggle key was created only for this run; its
      `kaggle.json` is outside the repository, mode `0600`, and never printed.
- [ ] The exact 40-character Git revision is pushed and reachable by Nautilus.
- [ ] The family analysis verifier reports 550 candidates and 1,650 measured
      epochs.
- [ ] The run, Secret, PVC, pilot Job, production Job, and audit Pod names are
      unique and unoccupied.
- [ ] The unique per-run PVC is `Bound` and carries this `RUN_ID` label.
- [ ] The Secret was created with `kubectl create`, is immutable, and has only
      the `kaggle.json` key.
- [ ] The saved Secret UID/resource version still passes the identity verifier
      before every submission and cleanup operation.
- [ ] Both locally rendered Job manifests pass the verifier and contain no
      credential value.
- [ ] The pilot completes successfully before production starts.
- [ ] The production Job reaches `Complete` and the independent receipt audit
      reports exactly 550 configurations and 1,650 measured epochs.
- [ ] Jobs and the Secret are deleted only by exact verified names; the PVC is
      retained until results are safely archived.
- [ ] The fresh legacy Kaggle key is revoked immediately after Secret cleanup.

Because the shared namespace is mandatory, the achievable guarantee is:
randomized create-only resources avoid ordinary collisions, Secret
immutability prevents data edits, UID/resource-version checks detect changes
when checked, exact labels protect cleanup, and prompt Kaggle-key revocation
limits exposure afterward. Resource-quota impact and access by sufficiently
authorized namespace users while the Secret exists cannot be eliminated.
