# PerfSeer V3 Native Nautilus A10 Speech-V2 Labeling Runbook

This is the active operator guide for `native_a10_speech_v2`. The implementation
is tested locally and does not publish an image or create Nautilus resources.
Perform the registry and Kubernetes steps only when the local acceptance gates
pass and you intentionally start production.

V2 retains the native `nvidia_a10_24gb_nrp` compute distribution but replaces
the unavailable 650-row ICML whale dataset with a binary `yes`/`no` view of the
TensorFlow Speech Recognition Challenge. It writes only to
`/workspace/perfseer-v3-native-a10-18k-speech-v2` and refuses V1 campaign state.
The three-way crosswalk preserves the original AWS A10G and native V1 IDs; the
speech measurements are new dataset semantics and must never be silently merged
with whale measurements.

## 1. Accept Speech rules and prove a real download first

Use the same Kaggle account that issued the mounted credential. First accept the
[TensorFlow Speech Recognition Challenge rules](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules).
Then run the access preflight. Its first gate must download
`link_to_gcp_credits_form.txt`, with an advertised and downloaded size of exactly
50 bytes. Listing the four remote files is not sufficient. HTTP 403 means the
account still has an external rules blocker; do not use a mirror.

```bash
docker run --rm \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  IMAGE image-preflight --verify-kaggle-access
```

The same command then performs actual smallest-file downloads for all remaining
sources with bounded retry/backoff. Accept every rules page before reserving an
A10:

| Task | Rules |
|---|---|
| Histopathologic Cancer | <https://www.kaggle.com/competitions/histopathologic-cancer-detection/rules> |
| Dogs vs Cats Redux | <https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/rules> |
| Dog Breed Identification | <https://www.kaggle.com/competitions/dog-breed-identification/rules> |
| SIIM-ISIC Melanoma | <https://www.kaggle.com/competitions/siim-isic-melanoma-classification/rules> |
| APTOS 2019 | <https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules> |
| Aerial Cactus | <https://www.kaggle.com/competitions/aerial-cactus-identification/rules> |
| Plant Pathology 2020 | <https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules> |
| RANZCR CLiP | <https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/rules> |
| Leaf Classification | <https://www.kaggle.com/competitions/leaf-classification/rules> |
| Denoising Dirty Documents | <https://www.kaggle.com/competitions/denoising-dirty-documents/rules> |
| Jigsaw Toxic Comment | <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules> |
| Detecting Insults | <https://www.kaggle.com/competitions/detecting-insults-in-social-commentary/rules> |
| Spooky Author | <https://www.kaggle.com/competitions/spooky-author-identification/rules> |
| Random Acts of Pizza | <https://www.kaggle.com/competitions/random-acts-of-pizza/rules> |
| Text Normalization English | <https://www.kaggle.com/competitions/text-normalization-challenge-english-language/rules> |
| Text Normalization Russian | <https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/rules> |
| MLSP 2013 Birds | <https://www.kaggle.com/competitions/mlsp-2013-birds/rules> |
| TensorFlow Speech Recognition | <https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/rules> |
| NYC Taxi Fare | <https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/rules> |
| NOMAD 2018 | <https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors/rules> |
| TPS December 2021 | <https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/rules> |
| TPS May 2022 | <https://www.kaggle.com/competitions/tabular-playground-series-may-2022/rules> |

The first full speech materialization safely validates the complete competition
ZIP and stores its archive SHA-256, archive-inventory SHA-256, remote-inventory
SHA-256, and substitution-contract SHA-256 under
`state/source_locks/tensorflow-speech-yes-no.json` on the workspace. All later
chunks must match that lock.

## 2. Build and verify the immutable local image

Create a public NRP GitLab project and log in to its registry only when you are
ready to publish. Generate the build identity and build locally:

```bash
python scripts/create_a10_speech_v2_build_manifest.py
docker build --platform linux/amd64 \
  -f containers/a10-speech-v2-labeler/Dockerfile \
  -t gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-speech-v2-labeler:SOURCE_SHA .
docker run --rm --gpus all \
  gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-speech-v2-labeler:SOURCE_SHA \
  image-preflight --require-cuda
```

On the RTX 5090, run the 12 audio-family/precision updates against the verified
real Kaggle speech view, the full 35-family fixture matrix, then the
five-real-epoch PANNs speech label. Every result is marked
`production_eligible: false`.

```bash
docker run --rm --gpus all \
  -v "$PWD/record/a10-speech-v2-local:/workspace" \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  IMAGE smoke-local --workspace /workspace --speech-precision-matrix
docker run --rm --gpus all \
  -v "$PWD/record/a10-speech-v2-local:/workspace" \
  IMAGE smoke-local --workspace /workspace --all-families
docker run --rm --gpus all \
  -v "$PWD/record/a10-speech-v2-real:/workspace" \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  IMAGE smoke-local --workspace /workspace --model panns_cnn14
```

The real record must show five completed epochs, warmup epochs 1--2, retained
measured epochs 3--5, the RTX 5090 identity, the balanced 4,096-row speech-view
fingerprint, and non-production status.

## 3. Push once and resolve an immutable digest

```bash
docker push gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-speech-v2-labeler:SOURCE_SHA
docker buildx imagetools inspect \
  gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-speech-v2-labeler:SOURCE_SHA
```

Record the registry `sha256:` digest. Do not render a Job from a mutable tag.

## 4. Create the read-only Secret and isolated 700 GiB RWX PVC

```bash
kubectl create secret generic perfseer-kaggle --namespace NAMESPACE \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json"
sed 's/REPLACE_NAMESPACE/NAMESPACE/' k8s/a10-speech-v2-labeler-pvc.yaml \
  > record/a10-speech-v2-pvc.yaml
kubectl apply -f record/a10-speech-v2-pvc.yaml
kubectl get pvc perfseer-v3-a10-speech-v2-labels --namespace NAMESPACE
```

The credential is mounted read-only at `/run/secrets/kaggle`; neither credentials
nor datasets are image layers.

## 5. Render and inspect the 96-label pilot offline

```bash
python scripts/render_a10_speech_v2_nautilus_job.py \
  --output record/a10-speech-v2-pilot-job.yaml \
  --namespace NAMESPACE \
  --image gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-speech-v2-labeler@sha256:DIGEST \
  --source-revision SOURCE_SHA \
  --mode pilot
sed -n '1,260p' record/a10-speech-v2-pilot-job.yaml
```

The renderer makes no Kubernetes API call. Confirm exact `NVIDIA-A10` affinity,
one GPU, equal requests/limits of 8 CPU and 32 GiB RAM, 16 GiB `/dev/shm`,
`backoffLimit: 0`, the read-only Secret, the isolated PVC, and the exact V2
workspace.

## 6. Submit, collect immediate evidence, and monitor

```bash
kubectl apply -f record/a10-speech-v2-pilot-job.yaml
kubectl get job perfseer-v3-a10-speech-v2-pilot --namespace NAMESPACE
kubectl get pod --namespace NAMESPACE -l job-name=perfseer-v3-a10-speech-v2-pilot
POD=$(kubectl get pod --namespace NAMESPACE \
  -l job-name=perfseer-v3-a10-speech-v2-pilot \
  -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$POD" --namespace NAMESPACE
kubectl logs "$POD" --namespace NAMESPACE --all-containers=true
kubectl get events --namespace NAMESPACE --sort-by=.lastTimestamp | tail -100
nohup scripts/monitor_a10_nautilus_job.sh \
  NAMESPACE perfseer-v3-a10-speech-v2-pilot \
  record/perfseer-v3-a10-speech-v2-pilot-monitor.log \
  >record/perfseer-v3-a10-speech-v2-pilot-monitor.nohup.log 2>&1 &
```

The monitor records status, events, processes, training-log tails, and output
files every 60 seconds for the first five minutes and every 20 minutes after.
During an active operator session, report progress at least every 60 seconds.
For a failed Job, immediately report the pod, exit code, final relevant logs,
and the corrective action.

## 7. Run 70 ordered production chunks

Verify the pilot, then render and submit chunk indices 0 through 69 in order:

```bash
docker run --rm -v PVC_EXPORT:/workspace IMAGE \
  verify --workspace /workspace --partial
python scripts/render_a10_speech_v2_nautilus_job.py \
  --output record/a10-speech-v2-chunk-00-job.yaml \
  --namespace NAMESPACE --image IMAGE_DIGEST --source-revision SOURCE_SHA \
  --mode chunk --chunk-index 0
```

Repeat the immediate diagnostics and start a distinct tracked monitor for every
Job. Chunks 0--68 contain 256 labels and chunk 69 contains 240. Resume an
interrupted Job with the same index and PVC. Never run concurrent writers.
After chunk 69:

```bash
docker run --rm -v PVC_EXPORT:/workspace IMAGE \
  verify --workspace /workspace --complete
```

Complete verification requires 18,000 accepted labels and 54,000 retained
measured epochs.

## Historical reference only

V1 used the now-unavailable [ICML 2013 Whale Challenge rules page](https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux/rules).
That link and its 650 rows are retained only for lineage. Do not use it as a V2
download gate or merge its measurements into Speech Commands rows.

## References

- NRP policies: <https://nrp.ai/documentation/userdocs/start/policies/>
- NRP GPU Pods: <https://nrp.ai/documentation/userdocs/running/gpu-pods/>
- NRP Jobs: <https://nrp.ai/documentation/userdocs/running/jobs/>
- NRP CephFS: <https://nrp.ai/documentation/userdocs/storage/ceph/>
- NRP GitLab: <https://nrp.ai/documentation/userdocs/development/gitlab/>
- NRP Docker tutorial: <https://nrp.ai/documentation/userdocs/tutorial/docker/>
- MLE-bench known issues: <https://github.com/openai/mle-bench#known-issues>
- Speech Commands dataset: <https://www.tensorflow.org/datasets/catalog/speech_commands>
