# PerfSeer V3 Non-Vision Four-A10 Labeling Runbook

This runbook starts from a clean local checkout and ends with a verified, downloaded
release. It never merges the teammate's vision measurements. The active corpus has
11,200 labels (33,600 retained epochs), 12 tasks, and 22 families. Production is one
Pod with four independent workers: one candidate process per A10, with no DDP/NCCL.

The implementation work did not push an image or change Nautilus. Every command in
the mutation sections below is for the operator to run later.

## 1. Pass all 12 Kaggle gates

Use the same Kaggle account that issued the API token. Open and accept each rules
page before doing anything on Nautilus:

1. [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules)
2. [Detecting Insults in Social Commentary](https://www.kaggle.com/competitions/detecting-insults-in-social-commentary/rules)
3. [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification/rules)
4. [Random Acts of Pizza](https://www.kaggle.com/competitions/random-acts-of-pizza/rules)
5. [English Text Normalization](https://www.kaggle.com/competitions/text-normalization-challenge-english-language/rules)
6. [Russian Text Normalization](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/rules)
7. [MLSP 2013 Birds](https://www.kaggle.com/competitions/mlsp-2013-birds/rules)
8. [TensorFlow Speech Recognition](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules)
9. [NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/rules)
10. [NOMAD 2018](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors/rules)
11. [Tabular Playground December 2021](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/rules)
12. [Tabular Playground May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/rules)

Keep `kaggle.json` outside the repository and set it to mode `0600`. Prove access by
downloading a real smallest file, not merely listing files:

```bash
docker run --rm \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  PERFSEER_LOCAL_IMAGE \
  image-preflight --verify-kaggle-access
```

The Speech Recognition gate specifically requires the advertised 50-byte
`link_to_gcp_credits_form.txt`. A 403 means the rules are still not accepted; stop
there. No mirror or synthetic data is acceptable.

## 2. Create the public NRP GitLab project

Follow the [NRP GitLab and container-registry guide](https://nrp.ai/documentation/userdocs/development/gitlab/).
Create a public project, note its registry path, and authenticate locally:

```bash
docker login gitlab-registry.nrp-nautilus.io
export PERFSEER_REGISTRY=gitlab-registry.nrp-nautilus.io/GROUP/PROJECT
```

Use a deploy token or personal-access token through standard input. Never put it in
the shell command, repository, image build argument, or Kubernetes YAML.

## 3. Build and verify the immutable image locally

Start on `feature/perfseer-v3-nautilus-a10-nonvision-4gpu-labeler` with a clean tree.
The build manifest must refer to the source revision being built:

```bash
export PERFSEER_SOURCE_REVISION=$(git rev-parse HEAD)
python scripts/create_a10_nonvision_build_manifest.py \
  --image-identity "perfseer-v3-a10-nonvision:${PERFSEER_SOURCE_REVISION}"
docker buildx build --platform linux/amd64 --load \
  -f containers/a10-nonvision-4gpu-labeler/Dockerfile \
  -t "perfseer-v3-a10-nonvision:${PERFSEER_SOURCE_REVISION}" .
export PERFSEER_LOCAL_IMAGE="perfseer-v3-a10-nonvision:${PERFSEER_SOURCE_REVISION}"
docker run --rm "$PERFSEER_LOCAL_IMAGE" analyze
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$PWD/.local/nonvision-construction:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" smoke-local \
  --workspace /workspace --construction-audit
```

Verify imports, hashes, CUDA kernels, the read-only runtime, and the actual local RTX
5090. Mount data/workspaces only outside image layers:

```bash
docker run --rm --gpus '"device=0"' --read-only \
  --shm-size=32g --tmpfs /tmp:rw,size=32g \
  "$PERFSEER_LOCAL_IMAGE" \
  image-preflight --hardware-mode local-rtx5090

docker run --rm --gpus '"device=0"' --read-only \
  --shm-size=32g --tmpfs /tmp:rw,size=32g \
  -v "$PWD/.local/nonvision-fixtures:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" \
  smoke-local --workspace /workspace --fixture-matrix
```

The fixture matrix covers every family/precision route, execution mode, batch size,
and generated structural lineage. The real 32-label gate uses all 12 Kaggle tasks,
five epochs each, and a separate non-production workspace:

```bash
docker run --rm --gpus '"device=0"' --read-only \
  --shm-size=32g --tmpfs /tmp:rw,size=32g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  -v "$PWD/.local/nonvision-rtx5090:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" run-campaign \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-local-rtx5090-v1 \
  --repository-revision "$PERFSEER_SOURCE_REVISION" \
  --image-digest "local@sha256:$(docker image inspect "$PERFSEER_LOCAL_IMAGE" --format '{{.Id}}' | cut -d: -f2)" \
  --local-validation

docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$PWD/.local/nonvision-rtx5090:/workspace:ro" \
  "$PERFSEER_LOCAL_IMAGE" verify \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-local-rtx5090-v1 \
  --local-validation
```

Every local record is `production_eligible: false`. The RTX test validates the
container, real-data adapters, five-epoch loop, recovery, and export paths; it cannot
prove four-A10 concurrency or A10 memory fit.

## 4. Push a revision tag and resolve its digest

```bash
docker tag "$PERFSEER_LOCAL_IMAGE" \
  "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker push "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker buildx imagetools inspect \
  "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
export PERFSEER_IMAGE_DIGEST="$PERFSEER_REGISTRY@sha256:REPLACE_WITH_RESOLVED_DIGEST"
```

Inspect that the resolved manifest is `linux/amd64`, then use only
`$PERFSEER_IMAGE_DIGEST`; never submit a tag-only reference.

## 5. Create the Secret and 700 GiB CephFS PVC

Choose your namespace and read the current [NRP resource policy](https://nrp.ai/documentation/userdocs/start/policies/)
and [CephFS guidance](https://nrp.ai/documentation/userdocs/storage/ceph/) first.

```bash
export PERFSEER_NAMESPACE=REPLACE_NAMESPACE
kubectl create secret generic perfseer-kaggle-nonvision \
  --namespace "$PERFSEER_NAMESPACE" \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json" \
  --dry-run=client -o yaml > .local/kaggle-secret.yaml
kubectl apply -f .local/kaggle-secret.yaml

sed "s/REPLACE_NAMESPACE/$PERFSEER_NAMESPACE/" \
  k8s/a10-nonvision-4gpu-labeler-pvc.yaml > .local/a10-nonvision-pvc.yaml
kubectl apply -f .local/a10-nonvision-pvc.yaml
kubectl get pvc --namespace "$PERFSEER_NAMESPACE" perfseer-v3-a10-nonvision-labels
```

Delete `.local/kaggle-secret.yaml` securely after `apply`; it contains credentials.
The committed Job mounts the Secret read-only and uses a dedicated RWX CephFS PVC.

## 6. Render and inspect the pilot locally

NRP's [GPU guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/) uses
`nvidia.com/gpu.product: NVIDIA-A10`. Render the canonical 32-label pilot:

```bash
python scripts/render_a10_nonvision_nautilus_job.py \
  --output .local/a10-nonvision-pilot.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --mode pilot
kubectl apply --dry-run=client -f .local/a10-nonvision-pilot.yaml
kubectl diff -f .local/a10-nonvision-pilot.yaml || true
```

Before submitting, manually confirm: registry digest, revision, four GPUs, equal
requests/limits, A10 required affinity, 32 CPU, 128 GiB RAM, 32 GiB `/dev/shm`,
32 GiB ephemeral storage, read-only Secret, correct PVC/workspace, `--pilot`,
`backoffLimit: 0`, and the 48-hour deadline.

## 7. Submit only through an explicit operator action

```bash
kubectl apply -f .local/a10-nonvision-pilot.yaml
```

Immediately collect all required feedback before doing anything else:

```bash
export PERFSEER_JOB=perfseer-v3-a10-nonvision-pilot
kubectl get job "$PERFSEER_JOB" --namespace "$PERFSEER_NAMESPACE" -o wide
kubectl get pod --namespace "$PERFSEER_NAMESPACE" -l "job-name=$PERFSEER_JOB" -o wide
export PERFSEER_POD=$(kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE"
kubectl logs "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" --all-containers=true
kubectl get events --namespace "$PERFSEER_NAMESPACE" --sort-by=.lastTimestamp | tail -100
```

This branch's implementation boundary permits only `get`, `describe`, `logs`, and
events queries against Nautilus. It does not use `exec` or `kubectl cp`.

## 8. Start durable monitoring

```bash
mkdir -p record
nohup scripts/monitor_a10_nonvision_job.sh \
  "$PERFSEER_NAMESPACE" "$PERFSEER_JOB" \
  "record/${PERFSEER_JOB}-monitor.log" \
  >"record/${PERFSEER_JOB}-monitor.nohup.log" 2>&1 &
echo $! >"record/${PERFSEER_JOB}-monitor.pid"
```

The monitor uses read-only queries, polls once per minute for five iterations and
every 20 minutes afterward, and records Job/Pod status, container process status,
logs (including controller heartbeat/artifact progress), descriptions, and events.
While an interactive operation is active, report progress to the user at least once
per minute until stable, failed, or asked to stop.

If the Job fails, immediately report the Pod, exit code, last relevant log lines,
and corrective action:

```bash
kubectl get pod "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" \
  -o jsonpath='{range .status.containerStatuses[*]}{.name}{" exit="}{.state.terminated.exitCode}{" reason="}{.state.terminated.reason}{"\n"}{end}'
kubectl logs "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" --tail=300
```

## 9. Verify the pilot and four unique A10s

The pilot receipt is written only after all 32 accepted labels pass provenance
verification. Logs and the receipt must show exactly four unique A10 UUIDs, A10 name,
compute capability 8.6, 22–26 GiB, and no mixed hardware. Do not start chunk 0 until
the pilot Job is complete and its receipt verifies. Local RTX records cannot satisfy
this gate.

## 10. Run 44 production chunks sequentially

The remaining 11,168 candidates are 43 chunks of 256 and one final chunk of 160.
For each index 0 through 43:

```bash
export PERFSEER_CHUNK_INDEX=0
python scripts/render_a10_nonvision_nautilus_job.py \
  --output ".local/a10-nonvision-chunk-${PERFSEER_CHUNK_INDEX}.yaml" \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --mode chunk --chunk-index "$PERFSEER_CHUNK_INDEX"
kubectl apply --dry-run=client \
  -f ".local/a10-nonvision-chunk-${PERFSEER_CHUNK_INDEX}.yaml"
# Explicit operator action only after inspection:
kubectl apply -f ".local/a10-nonvision-chunk-${PERFSEER_CHUNK_INDEX}.yaml"
```

Repeat immediate diagnostics and start a new tracked monitor. Never overlap pilot or
chunk Jobs because the exclusive campaign lock will reject a second writer. Resume a
failed chunk with the same index and workspace after diagnosing/correcting the cause.
Candidate-local OOM/errors continue through repair/quarantine; global integrity
failures exit nonzero.

## 11. Export the result on the PVC

The final release deliberately excludes checkpoints and weights. It contains
`labels.jsonl`, one exact configuration per candidate, content-addressed source,
factory entrypoints, index joins, repair/failure ledgers, receipts, manifests, and
`SHA256SUMS`:

```bash
# These are the container args for a digest-pinned utility Job mounting the same PVC:
docker run --rm --read-only --tmpfs /tmp:rw,size=16g \
  -v "$PWD/.local/nonvision-production:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" export \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-11200-v1 \
  --output-directory /workspace/releases --complete --verify-archive
```

Run `verify-export --archive ARCHIVE.tar.zst` in a clean offline image. This verifies
all checksums and reconstructs every indexed model from bundled source without a
network connection.

## 12. Transfer through NRP S3/rclone and verify again

Follow [NRP's data-movement guide](https://nrp.ai/documentation/userdocs/storage/move-data/).
Do not use `kubectl cp` for this large archive. Upload from the PVC to NRP S3 with a
dedicated transfer Pod/rclone, download it locally, then compare the recorded SHA-256:

```bash
sha256sum perfseer-v3-a10-nonvision-complete-*.tar.zst
docker run --rm --read-only --tmpfs /tmp:rw,size=16g \
  -v "$PWD:/release:ro" "$PERFSEER_LOCAL_IMAGE" \
  verify-export --archive /release/ARCHIVE.tar.zst
```

Only this verified archive is the handoff. The teammate's vision corpus remains a
separate artifact with its own contract and provenance.
