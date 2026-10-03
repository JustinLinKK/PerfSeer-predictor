#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$bundle_root"
sha256sum --check --quiet SHA256SUMS
bundle_digest="$(sha256sum SHA256SUMS)"
bundle_digest="${bundle_digest%% *}"
image_tag="${PERFSEER_IMAGE:-perfseer-v32:bundle-${bundle_digest:0:16}}"

# BuildKit reuses matching layers; the source and dataset always come from this ZIP.
docker build --platform linux/amd64 --progress=plain \
  --label "org.perfseer.bundle.sha256=$bundle_digest" \
  -f "$bundle_root/src/perfseer_v3.2/containers/training/Dockerfile" \
  -t "$image_tag" "$bundle_root"
if [[ "${PERFSEER_BUILD_ONLY:-0}" == "1" ]]; then
  printf 'Built %s; training was not started.\n' "$image_tag"
  exit 0
fi

mkdir -p "$bundle_root/outputs"
exec docker run --rm --init --name perfseer-v32-twelve-training \
  --gpus "device=${PERFSEER_GPU:-0}" --shm-size=2g --stop-timeout=300 \
  --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache \
  -v "$bundle_root/outputs:/outputs" \
  "$image_tag" \
  --dataset /opt/perfseer/src/perfseer_v3.2/dataset_with_label/ready_for_train_12 \
  --output /outputs/v32-twelve-shorter-time --resume \
  --teacher-epochs 600 --student-epochs 100 \
  --effective-batch 256 --microbatch 4 \
  --teacher-early-stopping-patience 6 \
  --teacher-early-stopping-min-epochs 30 \
  --no-teacher-stop-on-validation-gate "$@"
