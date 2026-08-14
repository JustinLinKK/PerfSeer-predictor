# PerfSeer V3 Non-Vision Four-A10 Disaster V2 Runbook

This is the active operator guide for the 11,200-label non-vision corpus. It uses
12 Kaggle tasks and 22 model families, retains three measured epochs per label, and
runs four independent workers in one Pod—one process per NVIDIA A10, without DDP or
NCCL. The production workspace is:

`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2`

The implementation did not execute any cluster command, publish an image, or create
Nautilus resources. Commands that change NRP state below are explicitly marked as
future operator actions.

## Implemented and focused verification status

The Disaster replacement is complete in source revision
`eb972cd415bd9d293e5633503cf2e89a35264e81`. The final local image is
`perfseer-v3-a10-nonvision-disaster-v2:eb972cd41` with local image ID
`sha256:4d22df3598ccd2689fb6c5b67d8bc3f0dacc2256d9c7e48f293bae2a7e643e7e`.
It has not been pushed.

Focused acceptance on August 13, 2026 passed the Disaster-specific regression suite
(22 tests), image preflight, and 20 real GPU updates: BERT, MLA Mini Transformer,
BiLSTM-CRF, FastText, and DistilBERT across TF32, BF16, FP16 with gradient scaling,
and mixed-structured precision. The real prepared view contained the contracted
4,096 rows and every update passed. Exact hashes and counts are recorded in
[`record/a10-nonvision-disaster-v2-focused-verification-20260813.json`](../record/a10-nonvision-disaster-v2-focused-verification-20260813.json).

At the operator's request, the broad 32-label test was stopped after its partial
results had been preserved; it is not claimed as complete and is not used as
evidence for the dataset substitution. The commands below retain that workflow as
the later full-release gate before publishing or submitting production work.

## 1. Accept and prove all 12 Kaggle agreements

Use the same account that issued the API token. Accept these rules:

1. [Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/rules)
2. [TensorFlow Speech Recognition](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules)
3. [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules)
4. [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification/rules)
5. [Random Acts of Pizza](https://www.kaggle.com/competitions/random-acts-of-pizza/rules)
6. [English Text Normalization](https://www.kaggle.com/competitions/text-normalization-challenge-english-language/rules)
7. [Russian Text Normalization](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/rules)
8. [MLSP 2013 Birds](https://www.kaggle.com/competitions/mlsp-2013-birds/rules)
9. [NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/rules)
10. [NOMAD 2018](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors/rules)
11. [Tabular Playground December 2021](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/rules)
12. [Tabular Playground May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/rules)

Keep `kaggle.json` outside the checkout with mode `0600`. The image preflight checks
Disaster Tweets first, Speech Recognition second, then the other ten. Every check
downloads a real smallest file; listing files is insufficient. Disaster must return
the 22,746-byte `sample_submission.csv`, and Speech must return the 50-byte
`link_to_gcp_credits_form.txt`. A 403 is a rules-agreement blocker: stop, accept the
rules with the same account, and retry. Do not use a mirror.

## 2. Create the public NRP GitLab project

Follow the [NRP GitLab registry guide](https://nrp.ai/documentation/userdocs/development/gitlab/),
create a public project, and record its registry path:

```bash
docker login gitlab-registry.nrp-nautilus.io
export PERFSEER_REGISTRY=gitlab-registry.nrp-nautilus.io/GROUP/PROJECT
```

Pass the token through standard input. Never place registry or Kaggle credentials in
Git, Docker build arguments, image layers, or Job YAML.

## 3. Build and fully verify the immutable image locally

Use branch `feature/perfseer-v3-nautilus-a10-nonvision-4gpu-disaster-v2` with a clean,
committed tree:

```bash
export PERFSEER_SOURCE_REVISION=$(git rev-parse HEAD)
python scripts/create_a10_disaster_v2_build_manifest.py \
  --image-identity "perfseer-v3-a10-nonvision-disaster-v2:${PERFSEER_SOURCE_REVISION}"
docker buildx build --platform linux/amd64 --load \
  -f containers/a10-nonvision-disaster-v2-labeler/Dockerfile \
  -t "perfseer-v3-a10-nonvision-disaster-v2:${PERFSEER_SOURCE_REVISION}" .
export PERFSEER_LOCAL_IMAGE="perfseer-v3-a10-nonvision-disaster-v2:${PERFSEER_SOURCE_REVISION}"
docker run --rm "$PERFSEER_LOCAL_IMAGE" analyze
```

Run the offline construction, 177-route fixture, import/hash, and source checks.
The fixture matrix performs a real backward/optimizer update on each route,
invokes Inductor for compiled candidates, resets compiler state between routes to
match the production fresh-process contract, and covers all retained generated
lineages in both execution modes:

```bash
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$PWD/.local/disaster-construction:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" smoke-local --workspace /workspace --construction-audit
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  -v "$PWD/.local/disaster-fixtures:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" smoke-local --workspace /workspace --fixture-matrix
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  "$PERFSEER_LOCAL_IMAGE" image-preflight --hardware-mode local-rtx5090
```

Prove the 12 downloads and run the 20-update Disaster matrix:

```bash
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  "$PERFSEER_LOCAL_IMAGE" image-preflight --verify-kaggle-access
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  -v "$PWD/.local/disaster-matrix:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" smoke-local --workspace /workspace \
  --disaster-precision-matrix
```

For the later full-release gate, run the frozen 32-label, five-real-epoch RTX 5090
workflow. It covers all 22
families, all 12 tasks, all four precisions, both execution modes, all regimes,
checkpointing on/off, and every effective batch in `{32,64,128,256,512}`:

```bash
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  -v "$PWD/.local/disaster-rtx5090:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" run-campaign \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-disaster-local-rtx5090-v2 \
  --repository-revision "$PERFSEER_SOURCE_REVISION" \
  --image-digest "local@sha256:$(docker image inspect "$PERFSEER_LOCAL_IMAGE" --format '{{.Id}}' | cut -d: -f2)" \
  --local-validation
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$PWD/.local/disaster-rtx5090:/workspace:ro" \
  "$PERFSEER_LOCAL_IMAGE" verify \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-disaster-local-rtx5090-v2 \
  --local-validation
```

All RTX records must say `production_eligible: false`. Local success validates the
image and complete workflow, but only the future four-A10 pilot validates A10 memory
fit and four-GPU scheduling.

## 4. Push, then resolve an immutable digest

```bash
docker tag "$PERFSEER_LOCAL_IMAGE" "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker push "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker buildx imagetools inspect "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
export PERFSEER_IMAGE_DIGEST="$PERFSEER_REGISTRY@sha256:REPLACE_WITH_RESOLVED_DIGEST"
```

Confirm `linux/amd64`. Never submit a tag-only image reference.

## 5. Create the Secret and dedicated 700 GiB CephFS PVC

Review the current [NRP resource policy](https://nrp.ai/documentation/userdocs/start/policies/)
and [CephFS guidance](https://nrp.ai/documentation/userdocs/storage/ceph/).
These are future operator mutations:

```bash
export PERFSEER_NAMESPACE=REPLACE_NAMESPACE
kubectl create secret generic perfseer-kaggle-disaster-v2 \
  --namespace "$PERFSEER_NAMESPACE" \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json"
sed "s/REPLACE_NAMESPACE/$PERFSEER_NAMESPACE/" \
  k8s/a10-nonvision-disaster-v2-labeler-pvc.yaml > .local/disaster-v2-pvc.yaml
kubectl apply -f .local/disaster-v2-pvc.yaml
kubectl get pvc --namespace "$PERFSEER_NAMESPACE" \
  perfseer-v3-a10-nonvision-disaster-v2
```

## 6. Render and inspect locally—without contacting Kubernetes

The renderer parses and verifies the full object locally. It rejects a tag, zero
digest, wrong workspace/profile, unequal resources, non-A10 affinity, wrong GPU
count, unsafe mounts, and invalid pilot/chunk mode:

```bash
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-pilot.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --secret perfseer-kaggle-disaster-v2 \
  --mode pilot
```

Manually confirm: four `NVIDIA-A10` GPUs, 32 CPU, 128 GiB RAM, 32 GiB `/dev/shm`,
32 GiB temporary storage, equal requests/limits, read-only Secret, V2 workspace,
digest-only image, `backoffLimit: 0`, and 48-hour deadline.

## 7. Submit the pilot only by explicit operator action

```bash
kubectl apply -f .local/disaster-v2-pilot.yaml
```

Immediately collect the required feedback:

```bash
export PERFSEER_JOB=perfseer-v3-a10-nonvision-disaster-v2-pilot
kubectl get job "$PERFSEER_JOB" --namespace "$PERFSEER_NAMESPACE" -o wide
kubectl get pod --namespace "$PERFSEER_NAMESPACE" -l "job-name=$PERFSEER_JOB" -o wide
export PERFSEER_POD=$(kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE"
kubectl logs "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" --all-containers=true
kubectl get events --namespace "$PERFSEER_NAMESPACE" --sort-by=.lastTimestamp | tail -100
```

Start the required durable monitor:

```bash
mkdir -p record
nohup scripts/monitor_a10_nonvision_job.sh \
  "$PERFSEER_NAMESPACE" "$PERFSEER_JOB" \
  "record/${PERFSEER_JOB}-monitor.log" \
  >"record/${PERFSEER_JOB}-monitor.nohup.log" 2>&1 &
echo $! >"record/${PERFSEER_JOB}-monitor.pid"
```

The pilot passes only when the receipt verifies 32 accepted labels and exactly four
unique A10 UUIDs. If it fails, immediately report Pod name, exit code, relevant log
tail, and corrective action.

## 8. Run 44 production chunks sequentially

After the pilot, run 43 chunks of 256 and a final chunk of 160. Never overlap Jobs:

```bash
export PERFSEER_CHUNK_INDEX=0
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output ".local/disaster-v2-chunk-${PERFSEER_CHUNK_INDEX}.yaml" \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --secret perfseer-kaggle-disaster-v2 \
  --mode chunk --chunk-index "$PERFSEER_CHUNK_INDEX"
# Explicit operator action after inspection:
kubectl apply -f ".local/disaster-v2-chunk-${PERFSEER_CHUNK_INDEX}.yaml"
```

Repeat immediate diagnostics and monitoring for each Job. Resume a failed chunk with
the same index and PVC after correction. The campaign lock rejects overlapping
writers. Candidate-local OOM/compile/model/timeout/crash failures are isolated;
archive, hardware, credentials, workspace, and GPU-cleanup failures stop globally.

## 9. Export, transfer, and verify the final release

```bash
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-export.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --secret perfseer-kaggle-disaster-v2 \
  --mode export
# Explicit operator action after inspection:
kubectl apply -f .local/disaster-v2-export.yaml
```

The release contains labels, exact training configurations, repair histories,
content-addressed source bundles, factory entrypoints, lineage/source hashes,
receipts, failure ledgers, candidate index, and `SHA256SUMS`; it contains no weights
or checkpoints. Transfer it through NRP S3/rclone per the
[NRP data-movement guide](https://nrp.ai/documentation/userdocs/storage/move-data/),
not `kubectl cp`. Verify the remote hash, downloaded hash, then reconstruct every
model in the clean image:

```bash
sha256sum perfseer-v3-a10-nonvision-disaster-v2-complete-*.tar.zst
docker run --rm --read-only --tmpfs /tmp:rw,size=16g \
  -v "$PWD/downloads:/release:ro" "$PERFSEER_LOCAL_IMAGE" verify-export \
  --archive /release/REPLACE_ARCHIVE.tar.zst
```

## Historical lineage only

The predecessor task was Detecting Insults in Social Commentary. It is deprecated
and unavailable, so it must not appear in active gates or new measurements. V1 is
preserved by manifest `05e4d98b…` and crosswalk `6556f56f…`. Disaster measurements
carry the V1 predecessor ID and manifest, but cannot be silently merged with insults
measurements because their dataset semantics differ.
