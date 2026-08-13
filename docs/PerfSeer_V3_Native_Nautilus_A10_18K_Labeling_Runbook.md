# PerfSeer V3 Native Nautilus A10 18K Labeling Runbook

> **Historical V1 reference.** ICML Whale is no longer downloadable. Do not use
> this guide for a new campaign. Use
> [the active Speech-V2 runbook](PerfSeer_V3_Native_Nautilus_A10_Speech_V2_18K_Labeling_Runbook.md).

This runbook is the operator handoff for the native `NVIDIA-A10` branch. The
implementation and its automated tests do not push an image or create Nautilus
resources. Run the commands below only after the local acceptance gates pass.

The native IDs describe `nvidia_a10_24gb_nrp`; the frozen AWS A10G reference is
`d7abb69b3c79e65e2f3834065ce284484a020ad4` with manifest
`bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e`.
The generated crosswalk pairs the two corpora but never merges them.

## 1. Accept and download-probe all 22 Kaggle competitions

Use the same Kaggle account that issued the mounted token. Open each rules page,
accept or re-accept the rules, and then run `image-preflight
--verify-kaggle-access`. The command performs a real smallest-file download with
bounded retry/backoff; a file-list request alone does not pass.

Run this credential-only gate locally without `--gpus` before reserving an A10.
Competitions that expose each sample as a separate remote file can require many
paginated inventory requests before the true smallest file is known, so allow
the command to finish and retain its checksummed output as the access receipt.

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
| ICML 2013 Whale | <https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux/rules> |
| NYC Taxi Fare | <https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/rules> |
| NOMAD 2018 | <https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors/rules> |
| TPS December 2021 | <https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/rules> |
| TPS May 2022 | <https://www.kaggle.com/competitions/tabular-playground-series-may-2022/rules> |

## 2. Create the NRP GitLab project and build locally

Create a public project in NRP GitLab, then log in to its registry. Use the NRP
GitLab and Docker documentation linked at the end of this guide.

```bash
python scripts/create_a10_build_manifest.py
docker build --platform linux/amd64 \
  -f containers/a10-labeler/Dockerfile \
  -t gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-labeler:SOURCE_SHA .
docker run --rm --gpus all \
  gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-labeler:SOURCE_SHA \
  image-preflight --require-cuda
```

The local RTX 5090 probe is always non-production. Run the 35-family one-batch
matrix and the two five-epoch real labels after all required Kaggle gates pass:

```bash
docker run --rm --gpus all \
  -v "$PWD/record/a10-local-smoke:/workspace" \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  IMAGE smoke-local --workspace /workspace --all-families
docker run --rm --gpus all \
  -v "$PWD/record/a10-local-smoke:/workspace" \
  -v "$HOME/.kaggle:/run/secrets/kaggle:ro" \
  IMAGE smoke-local --workspace /workspace --model panns_cnn14 --model cgcnn
```

## 3. Push once and resolve the immutable digest

```bash
docker push gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-labeler:SOURCE_SHA
docker buildx imagetools inspect \
  gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-labeler:SOURCE_SHA
```

Record the registry `sha256:` digest. Never render a Job from a mutable tag.

## 4. Create the Secret and 700 GiB RWX PVC

```bash
kubectl create secret generic perfseer-kaggle --namespace NAMESPACE \
  --from-file=kaggle.json="$HOME/.kaggle/kaggle.json"
kubectl apply --namespace NAMESPACE -f k8s/a10-labeler-pvc.yaml
kubectl get pvc perfseer-v3-a10-labels --namespace NAMESPACE
```

The Pod mounts the Secret read-only at `/run/secrets/kaggle`; credentials and
datasets are absent from the image.

## 5. Render and inspect the 96-label pilot

```bash
python scripts/render_a10_nautilus_job.py \
  --output record/a10-pilot-job.yaml \
  --namespace NAMESPACE \
  --image gitlab-registry.nrp-nautilus.io/GROUP/perfseer-a10-labeler@sha256:DIGEST \
  --source-revision SOURCE_SHA \
  --mode pilot
sed -n '1,240p' record/a10-pilot-job.yaml
```

The renderer performs the offline schema/contract verification before it writes
the file; it makes no Kubernetes API call. Inspect the rendered YAML. It must
have one GPU, exact `NVIDIA-A10` affinity,
equal 8-CPU and 32-GiB requests/limits, 16-GiB shared memory, `backoffLimit: 0`,
the read-only Secret, and the RWX PVC.

## 6. Submit, immediately diagnose, and monitor

```bash
kubectl apply -f record/a10-pilot-job.yaml
kubectl get job perfseer-v3-a10-pilot --namespace NAMESPACE
kubectl get pod --namespace NAMESPACE -l job-name=perfseer-v3-a10-pilot
POD=$(kubectl get pod --namespace NAMESPACE -l job-name=perfseer-v3-a10-pilot -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod "$POD" --namespace NAMESPACE
kubectl logs "$POD" --namespace NAMESPACE --all-containers=true
kubectl get events --namespace NAMESPACE --sort-by=.lastTimestamp | tail -100
nohup scripts/monitor_a10_nautilus_job.sh NAMESPACE perfseer-v3-a10-pilot \
  record/perfseer-v3-a10-pilot-monitor.log \
  >record/perfseer-v3-a10-pilot-monitor.nohup.log 2>&1 &
```

The monitor samples at least every 60 seconds for the first five minutes and
every 20 minutes afterward. During an active operator session, also report
progress to the user at least every 60 seconds. A `Failed` Job report must name
the pod, exit code, relevant final logs, and next correction.

## 7. Run 70 ordered production chunks

Verify the pilot before rendering chunk 0:

```bash
docker run --rm -v PVC_EXPORT:/workspace IMAGE verify --workspace /workspace --partial
```

For each `N` from 0 through 69, render `--mode chunk --chunk-index N`, inspect,
submit, immediately run the five diagnostic commands above, and start a distinct
tracked monitor log. The previous receipt is mandatory. Chunks 0--68 accept at
most 256 new labels; chunk 69 accepts 240. An interrupted Job is resubmitted with
the same chunk index and workspace. Never run two writers against the PVC.

After chunk 69:

```bash
docker run --rm -v PVC_EXPORT:/workspace IMAGE verify --workspace /workspace --complete
```

Complete verification requires 18,000 accepted labels and 54,000 retained
measured epochs.

## References

- NRP policies: <https://nrp.ai/documentation/userdocs/start/policies/>
- NRP GPU Pods: <https://nrp.ai/documentation/userdocs/running/gpu-pods/>
- NRP Jobs: <https://nrp.ai/documentation/userdocs/running/jobs/>
- NRP CephFS: <https://nrp.ai/documentation/userdocs/storage/ceph/>
- NRP GitLab: <https://nrp.ai/documentation/userdocs/development/gitlab/>
- NRP Docker tutorial: <https://nrp.ai/documentation/userdocs/tutorial/docker/>
