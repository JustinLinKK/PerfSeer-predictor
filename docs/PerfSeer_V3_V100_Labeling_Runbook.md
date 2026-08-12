# PerfSeer V3 four-V100 labeling runbook

This guide publishes and runs the already-built labeler. The implementation
does not run any registry or Nautilus command for you. Use the same checkout
revision for the build manifest, image, rendered Job, and every result.

The Pod is one Kubernetes Job with four independent label workers. It is not
distributed training: each child process receives one of four V100s through
`CUDA_VISIBLE_DEVICES`, and each candidate remains isolated and resumable.

## 0. One-time prerequisites

Use a machine with Docker Buildx, `kubectl`, Python 3.12, PyYAML, and access to
your NRP namespace. Keep `kaggle.json` outside this checkout at
`/home/justin/.kaggle/kaggle.json`, mode `0600`. Rotate any token that was ever
copied into the repository, then replace that external file.

Accept all six competition rule pages with the exact Kaggle account that owns
the API token. A file listing is insufficient; step 3 performs an actual
smallest-file download from each competition.

## 1. Create a public NRP GitLab project and log in

1. Sign in to `https://gitlab.nrp-nautilus.io`.
2. Create a blank project such as `perfseer-v100-labeler` and set visibility to
   **Public** so Nautilus nodes can pull without an image-pull Secret.
3. Create a GitLab access token with registry write permission. Do not save it
   in this repository or paste it into shell history.
4. Log in interactively:

```bash
docker login gitlab-registry.nrp-nautilus.io
```

Set these task-specific variables for the examples below:

```bash
export PERFSEER_NRP_NAMESPACE='replace-with-your-namespace'
export PERFSEER_REGISTRY_PROJECT='gitlab-registry.nrp-nautilus.io/replace-group/perfseer-v100-labeler'
export PERFSEER_SOURCE_REVISION="$(git rev-parse HEAD)"
export PERFSEER_LOCAL_IMAGE='perfseer-v3-v100-labeler:verified'
```

## 2. Build and verify locally

The build stage is allowed to install packages and clone the pinned MLE-bench
revision. Job startup performs neither Git, `apt`, nor `pip` operations.

```bash
python scripts/create_v100_build_manifest.py \
  --image-identity "$PERFSEER_LOCAL_IMAGE"
docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "$PERFSEER_LOCAL_IMAGE" \
  --file containers/v100-labeler/Dockerfile .
docker run --rm --gpus 'device=0' \
  "$PERFSEER_LOCAL_IMAGE" image-preflight --require-cuda
```

The preflight must report the pinned dependencies, `sm_70`, `sm_120`, a finite
CUDA matrix, and zero credential files. For an RTX 5090, it must report compute
capability 12.0.

## 3. Prove all Kaggle agreements with real downloads

This downloads and hashes the smallest advertised file from each of the six
competitions into an automatically deleted temporary directory:

```bash
docker run --rm --gpus 'device=0' \
  --mount type=bind,src=/home/justin/.kaggle,dst=/run/secrets/kaggle,readonly \
  "$PERFSEER_LOCAL_IMAGE" image-preflight \
  --require-cuda --verify-kaggle-access
```

Do not continue unless `kaggle_download_probes` contains six successful
entries. A `rules are not accepted` error identifies an account agreement,
not an image-library failure.

## 4. Push once and resolve the immutable digest

```bash
export PERFSEER_IMAGE_TAG="$PERFSEER_REGISTRY_PROJECT:$PERFSEER_SOURCE_REVISION"
docker tag "$PERFSEER_LOCAL_IMAGE" "$PERFSEER_IMAGE_TAG"
docker push "$PERFSEER_IMAGE_TAG"
docker buildx imagetools inspect "$PERFSEER_IMAGE_TAG"
```

Copy the displayed `sha256:...` manifest digest and set the complete reference:

```bash
export PERFSEER_IMAGE_DIGEST="$PERFSEER_REGISTRY_PROJECT@sha256:replace-with-64-hex-digest"
```

Never render a Job from a mutable tag.

## 5. Create the read-only Kaggle Secret and 700 GiB RWX PVC

The Secret only exists in Kubernetes and is mounted read-only at runtime. The
PVC is CephFS `ReadWriteMany`, so the four workers can share prepared data and
atomic state. Do not install Python packages or conda environments on CephFS.

```bash
kubectl create secret generic perfseer-kaggle \
  --namespace "$PERFSEER_NRP_NAMESPACE" \
  --from-file=kaggle.json=/home/justin/.kaggle/kaggle.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --namespace "$PERFSEER_NRP_NAMESPACE" \
  --filename k8s/v100-labeler-pvc.yaml
kubectl get pvc --namespace "$PERFSEER_NRP_NAMESPACE" perfseer-v3-v100-labels
```

Wait for the PVC to become `Bound` before submitting a Job.

## 6. Render and inspect the bounded four-label pilot

The pilot labels PANNs, M5, TabTransformer, and CGCNN concurrently. It writes
into the same modality workspaces that production later resumes.

```bash
python scripts/render_v100_nautilus_job.py \
  --mode pilot \
  --namespace "$PERFSEER_NRP_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --output record/perfseer-v3-v100-pilot.rendered.yaml
kubectl apply --dry-run=client \
  --filename record/perfseer-v3-v100-pilot.rendered.yaml
kubectl diff --filename record/perfseer-v3-v100-pilot.rendered.yaml || true
```

Inspect that the image uses `@sha256`, GPU request and limit are both four,
required product affinity is `Tesla-V100-SXM2-32GB`, and `--pilot` is present.

## 7. Submit, start monitoring, and collect immediate feedback

Only run this step when you intentionally want to consume four V100s:

```bash
kubectl apply --filename record/perfseer-v3-v100-pilot.rendered.yaml
nohup scripts/monitor_v100_nautilus_job.sh \
  "$PERFSEER_NRP_NAMESPACE" perfseer-v3-v100-pilot \
  record/perfseer-v3-v100-pilot-monitor.log \
  </dev/null >record/perfseer-v3-v100-pilot-monitor.nohup 2>&1 &

kubectl get job --namespace "$PERFSEER_NRP_NAMESPACE" perfseer-v3-v100-pilot -o wide
kubectl get pod --namespace "$PERFSEER_NRP_NAMESPACE" \
  --selector job-name=perfseer-v3-v100-pilot -o wide
export PERFSEER_POD="$(kubectl get pod --namespace "$PERFSEER_NRP_NAMESPACE" \
  --selector job-name=perfseer-v3-v100-pilot \
  --output jsonpath='{.items[0].metadata.name}')"
kubectl describe pod --namespace "$PERFSEER_NRP_NAMESPACE" "$PERFSEER_POD"
kubectl logs --namespace "$PERFSEER_NRP_NAMESPACE" "$PERFSEER_POD" \
  --all-containers=true --tail=250
kubectl get events --namespace "$PERFSEER_NRP_NAMESPACE" \
  --sort-by=.lastTimestamp | tail -100
```

The monitor records status every 60 seconds for the first five minutes and
every 20 minutes afterward. Keep the rendered manifest and monitor output
under `record/`. During an active interactive session, also report progress to
the user at least once per minute until the Job is stable, failed, or stopped.

## 8. Verify the pilot, then render production

The Pod itself verifies exact hardware before preparing data. Four unique GPU
UUIDs, the exact name, 32 GiB capacity, and compute capability 7.0 are required;
mixed or unexpected hardware aborts immediately.

After the pilot succeeds, render production against the same PVC and digest:

```bash
python scripts/render_v100_nautilus_job.py \
  --mode production \
  --namespace "$PERFSEER_NRP_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --output record/perfseer-v3-v100-production.rendered.yaml
kubectl apply --dry-run=client \
  --filename record/perfseer-v3-v100-production.rendered.yaml
```

Delete only the completed pilot Job object; never delete its PVC:

```bash
kubectl delete job --namespace "$PERFSEER_NRP_NAMESPACE" perfseer-v3-v100-pilot
kubectl apply --filename record/perfseer-v3-v100-production.rendered.yaml
```

Immediately repeat step 7 with Job name `perfseer-v3-v100-production` and log
path `record/perfseer-v3-v100-production-monitor.log`. Production resumes the
same workspace and completes exactly 3,550 labels / 10,650 measured epochs.

## 9. Failure report and corrective action

If the Job enters `Failed`, report all four items immediately: pod name, exit
code/reason, last relevant logs, and the next corrective action.

```bash
kubectl get pod --namespace "$PERFSEER_NRP_NAMESPACE" \
  --selector job-name=perfseer-v3-v100-production -o wide
kubectl get pod --namespace "$PERFSEER_NRP_NAMESPACE" "$PERFSEER_POD" \
  -o jsonpath='{range .status.containerStatuses[*]}{.name}{" exit="}{.state.terminated.exitCode}{" reason="}{.state.terminated.reason}{"\n"}{end}'
kubectl logs --namespace "$PERFSEER_NRP_NAMESPACE" "$PERFSEER_POD" \
  --all-containers=true --tail=250
kubectl describe pod --namespace "$PERFSEER_NRP_NAMESPACE" "$PERFSEER_POD"
kubectl get events --namespace "$PERFSEER_NRP_NAMESPACE" \
  --sort-by=.lastTimestamp | tail -100
```

Classify before retrying:

- `Forbidden` from Kaggle: accept that competition’s rules with the token’s
  account, rerun the six-download gate, then create a new Job.
- Hardware identity mismatch: fix affinity or scheduling; never weaken the
  V100 verifier.
- Python/import/CUDA failure: rebuild a new immutable image and digest; never
  `pip install` in the running Job.
- Candidate OOM: let the frozen deterministic repair path resume from the PVC.
  A cleanup/state-integrity failure is terminal and must be investigated.
- Node eviction/infrastructure loss: retain the PVC, delete the old Job object,
  and submit a new digest-identical Job so atomic state resumes.

`backoffLimit: 0` intentionally prevents Kubernetes from masking a fast
failure with automatic retries.

## Official NRP references

- [NRP GitLab and container registry](https://nrp.ai/documentation/userdocs/development/gitlab/)
- [Building and pushing Docker images](https://nrp.ai/documentation/userdocs/tutorial/docker/)
- [Container image guidance](https://nrp.ai/documentation/userdocs/tutorial/images)
- [GPU Pod product affinity](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
- [CephFS ReadWriteMany storage](https://nrp.ai/documentation/userdocs/storage/ceph/)
- [Jobs and initial Nautilus usage](https://nrp.ai/documentation/userdocs/start/using-nautilus/)
