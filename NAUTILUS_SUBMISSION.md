# Submit the PerfSeer Disaster V2 Labeler to NRP Nautilus

This is the only active operator guide in this repository. Follow it from top to
bottom to publish the immutable image and run the 11,200-label non-vision campaign.
The Job uses one Pod, four independent workers, and four NVIDIA A10 GPUs. Each
worker owns one GPU; this is not DDP or NCCL training.

Commands in sections 6 through 10 modify Nautilus and must be run by the operator.
No cluster or registry command was executed while preparing this repository.

## 1. Understand the two dataset replacements

Two unavailable Kaggle competitions are no longer part of the active campaign:

| Unavailable historical task | Active replacement | Active prepared view |
| --- | --- | --- |
| ICML 2013 Whale Challenge | [TensorFlow Speech Recognition](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules) | Binary `no -> 0`, `yes -> 1`; 2,048 valid 16 kHz WAV files per class |
| Detecting Insults in Social Commentary | [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/rules) | Binary target; deterministic 4,096-row view of `keyword`, `location`, and `text` |

The active corpus is 11,200 candidates over 12 tasks and 22 model families. It
retains three measured epochs per label. Historical task IDs and hashes remain only
as immutable lineage metadata, so old and replacement measurements cannot be
silently merged.

## 2. Accept and prove all 12 Kaggle sources

Accept the rules with the same Kaggle account that issued your API token:

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

Keep the token outside this checkout and restrict it:

```bash
chmod 0600 "$HOME/.kaggle/kaggle.json"
```

The image preflight downloads a real smallest file from every source. A listing is
not accepted as proof. Disaster must yield the 22,746-byte
`sample_submission.csv`; Speech must yield the 50-byte
`link_to_gcp_credits_form.txt`. Stop on a 403 and accept the rules; never use an
uncontracted mirror.

## 3. Commit, build, and verify locally

Start from a clean commit on
`feature/perfseer-v3-nautilus-a10-nonvision-4gpu-disaster-v2`:

```bash
git status --short
export PERFSEER_SOURCE_REVISION=$(git rev-parse HEAD)
export PERFSEER_LOCAL_IMAGE="perfseer-v3-a10-nonvision-disaster-v2:${PERFSEER_SOURCE_REVISION}"
python scripts/create_a10_disaster_v2_build_manifest.py \
  --image-identity "$PERFSEER_LOCAL_IMAGE"
docker buildx build --platform linux/amd64 --provenance=false --load \
  -f containers/a10-nonvision-disaster-v2-labeler/Dockerfile \
  -t "$PERFSEER_LOCAL_IMAGE" .
docker run --rm "$PERFSEER_LOCAL_IMAGE" analyze
```

Keep `--provenance=false`. NRP's GitLab registry UI can display Buildx
attestation indexes as `Invalid tag: missing manifest digest` with a false `0 B`
size. A conventional single-platform manifest avoids that UI incompatibility.

Run the immutable-image and Kaggle gates:

```bash
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  "$PERFSEER_LOCAL_IMAGE" image-preflight \
  --hardware-mode local-rtx5090 --require-cuda
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  "$PERFSEER_LOCAL_IMAGE" image-preflight --verify-kaggle-access
```

The checked-in RTX 5090 verification covers all 32 pilot candidates, all 22
families, all 12 tasks, five epochs per label, and 96 retained measured epochs. The
records are explicitly non-production. Repeat it before publishing if source code
has changed beyond documentation/build metadata:

```bash
mkdir -p .local/rtx5090
docker run --rm --gpus '"device=0"' --read-only --shm-size=32g \
  --tmpfs /tmp:rw,size=32g \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  -v "$PWD/.local/rtx5090:/workspace:rw" \
  "$PERFSEER_LOCAL_IMAGE" run-campaign \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-disaster-local-rtx5090-v2 \
  --repository-revision "$PERFSEER_SOURCE_REVISION" \
  --image-digest "local@$(docker image inspect "$PERFSEER_LOCAL_IMAGE" --format '{{.Id}}')" \
  --local-validation
docker run --rm --read-only --tmpfs /tmp:rw,size=8g \
  -v "$PWD/.local/rtx5090:/workspace:ro" \
  "$PERFSEER_LOCAL_IMAGE" verify \
  --workspace /workspace/perfseer-v3-native-a10-nonvision-disaster-local-rtx5090-v2 \
  --local-validation
```

Local RTX results must contain `production_eligible: false`. Only the four-A10 pilot
can validate A10 memory fit and real four-worker scheduling.

## 4. Create an NRP GitLab project and publish the image

Create a project at [NRP GitLab](https://gitlab.nrp-nautilus.io), then find its
registry path under **Deploy -> Container Registry**. NRP documents this workflow in
[Building in GitLab](https://nrp.ai/documentation/userdocs/development/gitlab/).

```bash
export PERFSEER_REGISTRY=gitlab-registry.nrp-nautilus.io/REPLACE_GROUP/REPLACE_PROJECT
docker login gitlab-registry.nrp-nautilus.io
docker tag "$PERFSEER_LOCAL_IMAGE" "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker push "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker buildx imagetools inspect "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
```

Copy the registry's resolved 64-character SHA-256, then set and inspect a digest-only
reference:

```bash
export PERFSEER_IMAGE_DIGEST="$PERFSEER_REGISTRY@sha256:REPLACE_RESOLVED_DIGEST"
case "$PERFSEER_IMAGE_DIGEST" in
  gitlab-registry.nrp-nautilus.io/*@sha256:????????????????????????????????????????????????????????????????) ;;
  *) echo "invalid digest-only image" >&2; exit 1 ;;
esac
```

Never submit a tag-only image. Do not place Kaggle or registry credentials in Git,
build arguments, image layers, or Job YAML.

## 5. Set your namespace and render the PVC

NRP currently requires resource limits to remain within 20% of requests; these Jobs
use equal requests and limits for Guaranteed QoS. Review the current
[cluster policy](https://nrp.ai/documentation/userdocs/start/policies/) before
submission.

```bash
export PERFSEER_NAMESPACE=REPLACE_NAMESPACE
mkdir -p .local
sed "s/REPLACE_NAMESPACE/$PERFSEER_NAMESPACE/" \
  k8s/a10-nonvision-disaster-v2-labeler-pvc.yaml \
  > .local/disaster-v2-pvc.yaml
```

Inspect `.local/disaster-v2-pvc.yaml`. It requests a 700 GiB `ReadWriteMany`
`rook-cephfs` volume. The workflow writes unique atomic candidate files and uses an
exclusive campaign lock; it never installs pip or conda packages on CephFS, matching
[NRP CephFS guidance](https://nrp.ai/documentation/userdocs/storage/ceph/).

## 6. Create the Kaggle Secret and PVC — operator actions

The following commands modify your namespace:

```bash
kubectl create secret generic perfseer-kaggle-disaster-v2 \
  --namespace "$PERFSEER_NAMESPACE" \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json"
kubectl apply -f .local/disaster-v2-pvc.yaml
kubectl get secret perfseer-kaggle-disaster-v2 --namespace "$PERFSEER_NAMESPACE"
kubectl get pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --namespace "$PERFSEER_NAMESPACE"
```

## 7. Render and inspect the pilot locally

Rendering is local and does not contact Kubernetes:

```bash
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-pilot.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --secret perfseer-kaggle-disaster-v2 \
  --mode pilot
sed -n '1,260p' .local/disaster-v2-pilot.yaml
```

Confirm the digest-only image, four `nvidia.com/gpu` requests and limits, required
`NVIDIA-A10` affinity, 32 CPU, 128 GiB memory, 32 GiB ephemeral storage, 32 GiB
`/dev/shm`, read-only Secret, correct PVC/workspace, `backoffLimit: 0`, and 48-hour
deadline. NRP's [GPU guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
states that the controller automatically adds the reserved-node toleration for a
four-GPU Job.

## 8. Submit the pilot and collect feedback immediately — operator actions

```bash
kubectl apply -f .local/disaster-v2-pilot.yaml
export PERFSEER_JOB=perfseer-v3-a10-nonvision-disaster-v2-pilot
kubectl get job "$PERFSEER_JOB" --namespace "$PERFSEER_NAMESPACE" -o wide
kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" -o wide
export PERFSEER_POD=$(kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE"
kubectl logs "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" \
  --all-containers=true
kubectl get events --namespace "$PERFSEER_NAMESPACE" \
  --sort-by=.lastTimestamp | tail -100
```

Start the required durable monitor. It samples every 60 seconds through the first
five minutes and every 20 minutes afterward:

```bash
mkdir -p record
nohup scripts/monitor_a10_nonvision_job.sh \
  "$PERFSEER_NAMESPACE" "$PERFSEER_JOB" \
  "record/${PERFSEER_JOB}-monitor.log" \
  >"record/${PERFSEER_JOB}-monitor.nohup.log" 2>&1 &
echo $! >"record/${PERFSEER_JOB}-monitor.pid"
```

The pilot passes only when its receipt verifies 32 accepted labels and four unique,
homogeneous NVIDIA A10 UUIDs. If it fails, report the failed Pod, exit code, relevant
log tail, and corrective action before submitting anything else.

## 9. Run the 44 production chunks sequentially — operator actions

The pilot labels remain canonical. Chunks 0–42 add 256 accepted labels each; chunk
43 adds the final 160. Never run overlapping chunks.

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
kubectl apply -f ".local/disaster-v2-chunk-${PERFSEER_CHUNK_INDEX}.yaml"
```

Immediately repeat the diagnostics and monitor setup from section 8 using Job name
`perfseer-v3-a10-nonvision-disaster-v2-chunk-NN`. Wait for verified completion before
incrementing the index. Resume a failed chunk with the same index and PVC after the
cause is fixed. Candidate-local OOM, compile, model, timeout, or child-process errors
are isolated; hardware, credentials, archive drift, workspace corruption, or failed
GPU cleanup stop the Job globally.

## 10. Export and download the completed release — operator actions

```bash
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-export.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc perfseer-v3-a10-nonvision-disaster-v2 \
  --secret perfseer-kaggle-disaster-v2 \
  --mode export
kubectl apply -f .local/disaster-v2-export.yaml
```

The verified `.tar.zst` contains `labels.jsonl`, exact training configurations,
requested/final microbatch and accumulation, OOM repair histories, content-addressed
model/runtime source bundles, factory entrypoints, the candidate index, receipts,
failure/quarantine ledgers, manifests, and `SHA256SUMS`. It contains no trained
weights or checkpoints.

Move the large archive through NRP S3/rclone, following
[NRP's data-movement guide](https://nrp.ai/documentation/userdocs/storage/move-data/).
Do not use `kubectl cp` for it. Verify the SHA-256 on the PVC, after transfer, and
again after local download, then reconstruct all models offline:

```bash
sha256sum perfseer-v3-a10-nonvision-disaster-v2-complete-*.tar.zst
docker run --rm --read-only --tmpfs /tmp:rw,size=16g \
  -v "$PWD/downloads:/release:ro" "$PERFSEER_LOCAL_IMAGE" verify-export \
  --archive /release/REPLACE_ARCHIVE.tar.zst
```

## Recovery boundary

The active production workspace is exactly:

`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2`

Do not reuse an older V1, Speech-only, V100, AWS A10G, or pre-Disaster workspace.
The workflow deliberately rejects old manifests, receipts, and locks.
