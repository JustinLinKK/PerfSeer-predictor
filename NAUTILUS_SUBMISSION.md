# Submit the PerfSeer Continuous A10 Campaign to NRP Nautilus

This is the only active operator guide. One Kubernetes Job runs the complete
workflow in order:

`32-label pilot -> pilot verification -> chunks 0-43 -> complete verification -> verified export`

The Pod has four independent workers and four NVIDIA A10 GPUs. Each worker owns
one GPU; this is not DDP or NCCL. The Job resumes verified receipts from the PVC
if Kubernetes starts its one replacement Pod.

Only the operator should run the Kubernetes mutation commands below. Building and
publishing this repository did not create, change, submit, execute into, or copy
from any Nautilus resource.

## Why this resumes the NVML V1 workspace

The NVML-corrected continuous Job passed credential staging, all 12 Kaggle access
probes, the pilot, and chunks 0--3. It durably accepted 1,056 labels before
`chunk-04` exposed a deterministic planner defect: fastText replacement generation
could change `sparse_gradients` without rebinding the preserved optimizer. The
resulting optimizer/parameter mismatch raised a candidate-planning exception that
escaped the worker batch and stopped the controller.

The corrected image binds this parameter contract before validation and exhaustively
validates every deterministic replacement proposal. Candidate-local execution and
planning failures are now recorded and isolated so sibling rows and later four-GPU
batches continue; a failed row never counts toward the required 11,200 accepted
labels. The existing NVML V1 workspace remains immutable at its origin identity.
An explicit, content-hashed recovery transition authorizes only that exact origin,
preserves its 1,056 labels and provenance, and records the corrected build/environment
without rewriting old records. Archive filenames and release contents are unchanged.

## 1. Understand the active datasets

The active non-vision corpus has 11,200 labels, 12 Kaggle tasks, 22 model families,
and 33,600 retained measured epochs. Vision work is excluded because it belongs to
a teammate's separate corpus.

Two deprecated competitions were replaced:

| Historical task | Active replacement | Prepared view |
| --- | --- | --- |
| ICML 2013 Whale Challenge | [TensorFlow Speech Recognition](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules) | Binary `no -> 0`, `yes -> 1`; exactly 2,048 valid 16 kHz WAVs per class |
| Detecting Insults in Social Commentary | [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/rules) | Binary target; deterministic 4,096-row view of `keyword`, `location`, and `text` |

Historical IDs and hashes remain only as lineage. Measurements from old and
replacement datasets must not be silently merged.

## 2. Accept and probe all 12 Kaggle competitions

Accept every rules page using the account that issued your Kaggle token:

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

Keep the credential outside this repository:

```bash
chmod 0600 "$HOME/.kaggle/kaggle.json"
kaggle competitions download -c nlp-getting-started \
  -f sample_submission.csv -p /tmp/perfseer-kaggle-probe
kaggle competitions download -c tensorflow-speech-recognition-challenge \
  -f link_to_gcp_credits_form.txt -p /tmp/perfseer-kaggle-probe
```

The Pod repeats a real smallest-file download for all 12 sources once before its
pilot. Merely listing files is not sufficient. Disaster must yield the advertised
22,746-byte `sample_submission.csv`; Speech must yield the advertised 50-byte
`link_to_gcp_credits_form.txt`. A 403 is a rules/account blocker—do not substitute
an uncontracted mirror.

## 3. Use the immutable continuous image

The current continuous image is published under a full source-revision tag. The
exact source revision and digest are recorded here after publication:

```bash
export PERFSEER_SOURCE_REVISION=6e8dd3f470f733007bb7f44df2261b08d0ea0f0f
export PERFSEER_REGISTRY=gitlab-registry.nrp-nautilus.io/justinlinkk/prefseer-predictor-labeling
export PERFSEER_IMAGE_DIGEST="$PERFSEER_REGISTRY@sha256:7c87aeeb886f174d00ba8f6071714dce77bbd0fd6890527257299346c562c8bc"
docker buildx imagetools inspect "$PERFSEER_IMAGE_DIGEST"
```

The GitLab project must remain public for anonymous Nautilus pulls without an
image-pull Secret. Never replace this with a tag-only reference and never use
`latest`.

The immediately previous image contains the replacement optimizer-contract defect
and must not be used for labeling. It remains an investigation artifact only:

- Source: `d8255c521ed37f47ba09096b1c98fadebeb51864`
- Digest: `sha256:f2a68ea4ab1a3fb65b836fb5ef76d57c7537b87a155726e77fd93c7af049752a`

The still older image contains the four-GPU NVML binding defect and is also an
investigation artifact only:

- Source: `81a24dc920d3bf02762da3de2e2bddcaba0e7dd4`
- Digest: `sha256:7551c71cd10a7f089357797b469db5dae4eccac9bf31f736b573bb20d5424ad9`

### Rebuild and publish a future source revision

Run this only for a clean, reviewed commit:

```bash
export PERFSEER_SOURCE_REVISION=$(git rev-parse HEAD)
export PERFSEER_LOCAL_IMAGE="perfseer-v3-a10-nonvision-disaster-v2:${PERFSEER_SOURCE_REVISION}"
python scripts/create_a10_disaster_v2_build_manifest.py \
  --image-identity "$PERFSEER_LOCAL_IMAGE"
docker buildx build --platform linux/amd64 --provenance=false --load \
  -f containers/a10-nonvision-disaster-v2-labeler/Dockerfile \
  -t "$PERFSEER_LOCAL_IMAGE" .
docker run --rm "$PERFSEER_LOCAL_IMAGE" analyze
docker run --rm "$PERFSEER_LOCAL_IMAGE" run-continuous --help
docker login gitlab-registry.nrp-nautilus.io
docker tag "$PERFSEER_LOCAL_IMAGE" "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker push "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
docker buildx imagetools inspect "$PERFSEER_REGISTRY:$PERFSEER_SOURCE_REVISION"
```

Keep `--provenance=false`; it produces a conventional single-platform Docker V2
manifest that the NRP GitLab registry UI handles correctly. Do not include Kaggle
or registry credentials in Git, build arguments, layers, or YAML.

## 4. Set the verified namespace, PVC, and Secret names

The namespace is `ecepxie`:

```bash
export PERFSEER_NAMESPACE=ecepxie
export PERFSEER_PVC=perfseer-panns-jingbin-260808-a0af09
export PERFSEER_KAGGLE_SECRET=perfseer-kaggle-disaster-v2
```

The existing PVC was read-only queried on August 14, 2026. It was `Bound`, 700 GiB,
`ReadWriteMany`, backed by `rook-cephfs`, and not mounted by a current Pod. Its
metadata owner is Jingbin, which the operator confirmed is them. Recheck it before
submission:

```bash
kubectl get pvc "$PERFSEER_PVC" --namespace "$PERFSEER_NAMESPACE" -o wide
kubectl describe pvc "$PERFSEER_PVC" --namespace "$PERFSEER_NAMESPACE"
```

The corrected campaign resumes only
`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1` and uses
both a continuous-controller lock and atomic phase state. Its explicit recovery
authorization is pinned to origin identity
`5bc98b7c0f8697d30323e39d4e645b4e8a249fdb9ad01120c1c2db3bdd2ad4fb`.
The earlier `.../disaster-v2` workspace is separate evidence and must remain
untouched. Do not run another campaign against either workspace concurrently.

## 5. Create the dedicated Kaggle Secret once — operator action

Query first:

```bash
kubectl get secret "$PERFSEER_KAGGLE_SECRET" \
  --namespace "$PERFSEER_NAMESPACE" -o name
```

If it is absent, create it. This is a namespace mutation and must be run by you:

```bash
kubectl create secret generic "$PERFSEER_KAGGLE_SECRET" \
  --namespace "$PERFSEER_NAMESPACE" \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json"
kubectl get secret "$PERFSEER_KAGGLE_SECRET" \
  --namespace "$PERFSEER_NAMESPACE" -o name
```

Never print the Secret as YAML/JSON or request its `.data` field. The source Secret
is visible only to an init container. Because pod `fsGroup` can widen projected
volume permissions, that init container copies only `kaggle.json` into a 1 MiB
memory-backed private volume, sets and verifies actual mode `0400`, and exits. The
main labeler mounts only the private copy read-only. Do not apply a new PVC; the
renderer below uses the existing approved claim.

## 6. Render and inspect the single continuous Job locally

Rendering only parses and verifies YAML locally; it does not contact the cluster:

```bash
mkdir -p .local
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-repair-v1-continuous-ecepxie.yaml \
  --namespace "$PERFSEER_NAMESPACE" \
  --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" \
  --pvc "$PERFSEER_PVC" \
  --secret "$PERFSEER_KAGGLE_SECRET" \
  --mode continuous
sed -n '1,260p' .local/disaster-v2-repair-v1-continuous-ecepxie.yaml
```

The renderer fails unless the Job has all of these properties:

- digest-only public NRP GitLab image;
- command `run-continuous` and the exact Disaster V2 workspace;
- the exact frozen origin-identity recovery authorization;
- one Pod/controller with four independent workers;
- four `nvidia.com/gpu` requests and limits;
- required `nvidia.com/gpu.product: NVIDIA-A10` affinity;
- equal requests/limits of 32 CPU, 128 GiB RAM, and 32 GiB ephemeral storage;
- 32 GiB `/dev/shm`;
- read-only root filesystems and a bounded init container that stages only
  `kaggle.json` at verified mode `0400`;
- no direct Secret mount in the main labeler and a read-only private credential
  mount backed by a 1 MiB memory `emptyDir`;
- the existing 700 GiB RWX PVC;
- `backoffLimit: 1`, seven-day deadline, and 120-second termination grace.

Equal requests and limits satisfy the current [NRP resource policy](https://nrp.ai/documentation/userdocs/start/policies/).
NRP's [GPU guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
documents product affinity and the reserved-node behavior for multi-GPU requests.

## 7. Submit once and inspect immediately — operator action

This is the only normal campaign submission:

```bash
kubectl apply -f .local/disaster-v2-repair-v1-continuous-ecepxie.yaml
export PERFSEER_JOB=perfseer-v3-a10-nonvision-disaster-v2-repair-v1-continuous
kubectl get job "$PERFSEER_JOB" --namespace "$PERFSEER_NAMESPACE" -o wide
kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" -o wide
export PERFSEER_POD=$(kubectl get pod --namespace "$PERFSEER_NAMESPACE" \
  -l "job-name=$PERFSEER_JOB" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE"
kubectl logs "$PERFSEER_POD" --namespace "$PERFSEER_NAMESPACE" \
  --all-containers=true
kubectl get events --namespace "$PERFSEER_NAMESPACE" \
  --sort-by=.lastTimestamp | tail -100
```

Do not submit separate pilot/chunk/export Jobs while the continuous Job exists.
Candidate-local failures are durably marked failed, quarantined/replaced, and do
not stop sibling rows or later batches. The controller stops only for shared-state,
credential, hardware/provenance, or cleanup integrity failures, or after a quota
slot exhausts all bounded replacements. It never emits a terminal receipt or
archive with fewer than 11,200 accepted rows.

## 8. Start the durable replacement-Pod-aware monitor

```bash
mkdir -p record
nohup scripts/monitor_a10_nonvision_job.sh \
  "$PERFSEER_NAMESPACE" "$PERFSEER_JOB" \
  "record/${PERFSEER_JOB}-monitor.log" \
  >"record/${PERFSEER_JOB}-monitor.nohup.log" 2>&1 &
echo $! >"record/${PERFSEER_JOB}-monitor.pid"
```

The monitor follows every Pod carrying the Job label, including the replacement
Pod, polls every 60 seconds for the first five minutes and every 20 minutes
afterward, and stops only after recording a `Complete` or `Failed` Job condition.
The container also writes structured progress to stdout and atomically updates:

`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1/state/continuous_campaign_progress.json`

Progress records identify the phase, completed chunks, accepted-label count,
failure details, and final archive hash. The terminal receipt is:

`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1/state/continuous_campaign_receipt.json`

If the Job fails, immediately collect the failed Pod name, exit code, relevant log
tail, and recent events before deciding on a correction.

## 9. Verify and download the final release

Successful Job logs end with a `campaign_complete` event containing the archive
path and SHA-256. The archive is under:

`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1/releases/`

The content-hashed archive has a matching JSON checksum sidecar in the same
directory:

```text
perfseer-v3-a10-nonvision-disaster-v2-complete-<release-hash>.tar.zst
perfseer-v3-a10-nonvision-disaster-v2-complete-<release-hash>.tar.zst.json
```

The terminal receipt records the absolute PVC archive path, archive SHA-256,
release SHA-256, 11,200 accepted labels, and all 44 completed chunks. A Job is not
successful until that receipt exists and the container has re-extracted the archive,
verified every checksum/source/configuration/label join, and reconstructed all
11,200 model configurations.

It contains:

- `labels.jsonl` with measurements and provenance;
- one canonical training configuration per candidate, including requested batch,
  final microbatch, gradient accumulation, precision, optimizer, scheduler, seeds,
  execution mode, checkpointing, and repair history;
- content-addressed model/runtime source bundles and factory entrypoints;
- the candidate join index, receipts, manifests, failure/quarantine ledgers, and
  `SHA256SUMS`;
- no trained weights or checkpoints.

Move the large archive using NRP S3/rclone as described in the
[NRP data-movement guide](https://nrp.ai/documentation/userdocs/storage/move-data/).
Transfer both the `.tar.zst` and its `.tar.zst.json` sidecar. Do not use `kubectl cp`
for a large release. Compare the sidecar SHA-256 after each transfer and run the
full local verifier:

```bash
export PERFSEER_ARCHIVE=$(find downloads -maxdepth 1 -type f \
  -name 'perfseer-v3-a10-nonvision-disaster-v2-complete-*.tar.zst' -print -quit)
export PERFSEER_EXPECTED_SHA256=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' \
  "${PERFSEER_ARCHIVE}.json")
test "$(sha256sum "$PERFSEER_ARCHIVE" | cut -d' ' -f1)" = \
  "$PERFSEER_EXPECTED_SHA256"
docker run --rm --read-only --tmpfs /tmp:rw,size=16g \
  -v "$PWD/downloads:/release:ro" "$PERFSEER_IMAGE_DIGEST" verify-export \
  --archive "/release/$(basename "$PERFSEER_ARCHIVE")"
```

## 10. Recovery tools

Kubernetes may create one replacement Pod automatically. It reruns the 12 access
gates, then resumes the pilot/chunk receipts on the PVC. A verified terminal
receipt causes an immediate verification-only exit instead of repeating work.

If both Pods fail or the seven-day Job deadline expires, preserve the PVC. After
fixing the cause, delete only the failed Job object and submit the same verified
manifest again; do not delete the PVC or workspace:

```bash
kubectl delete job "$PERFSEER_JOB" --namespace "$PERFSEER_NAMESPACE"
kubectl apply -f .local/disaster-v2-repair-v1-continuous-ecepxie.yaml
```

The older single-phase renderer modes remain emergency tools. Use them only after
the continuous Job is terminal/deleted and only for the phase named by the failure
receipt:

```bash
# Pilot recovery
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-pilot.yaml \
  --namespace "$PERFSEER_NAMESPACE" --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" --pvc "$PERFSEER_PVC" \
  --secret "$PERFSEER_KAGGLE_SECRET" --mode pilot

# One failed chunk; replace N with 0-43
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-chunk-N.yaml \
  --namespace "$PERFSEER_NAMESPACE" --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" --pvc "$PERFSEER_PVC" \
  --secret "$PERFSEER_KAGGLE_SECRET" --mode chunk --chunk-index N

# Export recovery after complete verification
python scripts/render_a10_disaster_v2_nautilus_job.py \
  --output .local/disaster-v2-export.yaml \
  --namespace "$PERFSEER_NAMESPACE" --image "$PERFSEER_IMAGE_DIGEST" \
  --source-revision "$PERFSEER_SOURCE_REVISION" --pvc "$PERFSEER_PVC" \
  --secret "$PERFSEER_KAGGLE_SECRET" --mode export
```

The active recovery workspace is exactly
`/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2-nvml-v1`. Never
migrate the separate `.../disaster-v2` state, a V1, Speech-only, V100, AWS A10G,
or pre-Disaster receipt into it. The renderer's recovery hash must not be changed
or reused for another workspace.
