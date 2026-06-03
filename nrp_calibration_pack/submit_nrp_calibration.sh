#!/usr/bin/env bash
set -euo pipefail

GPU_PRODUCT=""
GPU_RESOURCE="nvidia.com/gpu"
NAMESPACE=""
IMAGE=""
PVC=""
OUTPUT_DIR="/mnt/output/nrp_calibration"
PARALLELISM="4"
COMPLETIONS="64"
JOB_NAME="perfseer-calibration"
PACK_DIR="/workspace/nrp_calibration_pack"
WARMUP="20"
INFER_REPEATS="50"
TRAIN_REPEATS="50"
SAMPLE_INTERVAL="0.01"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  ./nrp_calibration_pack/submit_nrp_calibration.sh \
    --namespace <k8s-namespace> \
    --image <container-image-with-pack> \
    --pvc <output-pvc> \
    --gpu-product NVIDIA-GeForce-RTX-4090

Options:
  --gpu-product VALUE     Optional node affinity value for nvidia.com/gpu.product.
  --gpu-resource VALUE    GPU resource key. Default: nvidia.com/gpu. Use e.g. nvidia.com/a100 for special NRP GPUs.
  --namespace VALUE       Kubernetes namespace.
  --image VALUE           Container image containing /workspace/nrp_calibration_pack.
  --pvc VALUE             PVC used to persist outputs.
  --output-dir VALUE      Output directory inside the mounted PVC. Default: /mnt/output/nrp_calibration.
  --parallelism N         Concurrent pods. Default: 4.
  --completions N         Indexed job shard count. Default: 64.
  --job-name VALUE        Kubernetes Job name. Default: perfseer-calibration.
  --pack-dir VALUE        Pack path inside the image. Default: /workspace/nrp_calibration_pack.
  --warmup N              Warmup iterations before timing each phase. Default: 20.
  --infer-repeats N       Timed inference iterations per model. Default: 50.
  --train-repeats N       Timed train-step iterations per model. Default: 50.
  --sample-interval SEC   NVML sampling interval in seconds. Default: 0.01.
  --dry-run               Print rendered YAML without submitting.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-product) GPU_PRODUCT="$2"; shift 2 ;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --pvc) PVC="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --completions) COMPLETIONS="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --pack-dir) PACK_DIR="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --infer-repeats) INFER_REPEATS="$2"; shift 2 ;;
    --train-repeats) TRAIN_REPEATS="$2"; shift 2 ;;
    --sample-interval) SAMPLE_INTERVAL="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$NAMESPACE" || -z "$IMAGE" || -z "$PVC" ]]; then
  echo "--namespace, --image, and --pvc are required" >&2
  usage
  exit 2
fi

AFFINITY_BLOCK=""
if [[ -n "$GPU_PRODUCT" ]]; then
  AFFINITY_BLOCK=$(cat <<EOF
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - ${GPU_PRODUCT}
EOF
)
fi

YAML=$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  completionMode: Indexed
  completions: ${COMPLETIONS}
  parallelism: ${PARALLELISM}
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: ${JOB_NAME}
    spec:
      restartPolicy: Never
${AFFINITY_BLOCK}
      containers:
      - name: profiler
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: JOB_COMPLETIONS
          value: "${COMPLETIONS}"
        command: ["/bin/bash", "-lc"]
        args:
        - >
          python ${PACK_DIR}/profile/run_profile.py
          --manifest ${PACK_DIR}/manifest/subset_manifest.jsonl
          --models-dir ${PACK_DIR}/models
          --output-dir ${OUTPUT_DIR}
          --shard-index \${JOB_COMPLETION_INDEX:-0}
          --num-shards ${COMPLETIONS}
          --warmup ${WARMUP}
          --infer-repeats ${INFER_REPEATS}
          --train-repeats ${TRAIN_REPEATS}
          --sample-interval ${SAMPLE_INTERVAL}
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            ${GPU_RESOURCE}: "1"
          limits:
            cpu: "8"
            memory: "32Gi"
            ${GPU_RESOURCE}: "1"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: ${PVC}
EOF
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' "$YAML"
else
  printf '%s\n' "$YAML" | kubectl apply -f -
fi
