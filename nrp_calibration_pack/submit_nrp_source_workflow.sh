#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=""
IMAGE=""
PVC=""
GPU_PRODUCT=""
GPU_RESOURCE="nvidia.com/gpu"
JOB_PREFIX="perfseer-nrp-source"
STAGE="all"
WORKFLOW_DIR="/mnt/output/perfseer_nrp_source_workflow"
REPO_DIR="/workspace/PerfSeer-predictor"
HARDWARE_ID="rtx5090"
SUBSET_SIZE="10000"
SEED="20260617"
GENERATION_WORKERS="0"
LOW_PRECISION_FOCUS="none"
PARALLELISM="4"
COMPLETIONS="64"
WARMUP="20"
INFER_REPEATS="50"
TRAIN_REPEATS="50"
SAMPLE_INTERVAL="0.01"
OPTIMIZER="adam"
SM_OCCUPANCY_SOURCE="nvml_proxy"
BOOTSTRAP_COMMAND=""
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  ./nrp_calibration_pack/submit_nrp_source_workflow.sh \
    --namespace <k8s-namespace> \
    --image <repo-image-based-on-nvcr.io/nvidia/pytorch:26.03-py3> \
    --pvc <output-pvc> \
    --gpu-product NVIDIA-GeForce-RTX-5090 \
    --hardware-id rtx5090

Options:
  --namespace VALUE       Kubernetes namespace.
  --image VALUE           Container image containing this repo under --repo-dir.
  --pvc VALUE             PVC used to persist pack, labels, and source tarball.
  --gpu-product VALUE     Optional node affinity value for nvidia.com/gpu.product.
  --gpu-resource VALUE    GPU resource key. Default: nvidia.com/gpu.
  --job-prefix VALUE      Kubernetes Job name prefix. Default: perfseer-nrp-source.
  --stage VALUE           all, prepare, profile, or package. Default: all for --dry-run only.
  --workflow-dir VALUE    Directory inside the PVC mount. Default: /mnt/output/perfseer_nrp_source_workflow.
  --repo-dir VALUE        Repo path inside the image. Default: /workspace/PerfSeer-predictor.
  --hardware-id VALUE     Stable hardware id stored in labels. Default: rtx5090.
  --subset-size N         Number of unique generated source models. Default: 10000.
  --seed N                Deterministic generation/profile seed. Default: 20260617.
  --generation-workers N  Source generation workers. Default: 0 (all CPUs).
  --low-precision-focus VALUE
                          Source-generation focus: none or te_transformer. Default: none.
  --parallelism N         Concurrent profile pods. Default: 4.
  --completions N         Indexed profile shard count. Default: 64.
  --warmup N              Warmup iterations before timing. Default: 20.
  --infer-repeats N       Timed inference iterations per model. Default: 50.
  --train-repeats N       Timed train-step iterations per model. Default: 50.
  --sample-interval SEC   NVML sampling interval. Default: 0.01.
  --optimizer VALUE       Training optimizer for profiling labels. Default: adam.
  --sm-occupancy-source VALUE
                          SM occupancy source: ncu or nvml_proxy. Default: nvml_proxy.
  --bootstrap-command CMD Optional shell command run before each stage command.
  --dry-run               Print rendered YAML without submitting.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --pvc) PVC="$2"; shift 2 ;;
    --gpu-product) GPU_PRODUCT="$2"; shift 2 ;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --workflow-dir) WORKFLOW_DIR="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --hardware-id) HARDWARE_ID="$2"; shift 2 ;;
    --subset-size) SUBSET_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --generation-workers) GENERATION_WORKERS="$2"; shift 2 ;;
    --low-precision-focus) LOW_PRECISION_FOCUS="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --completions) COMPLETIONS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --infer-repeats) INFER_REPEATS="$2"; shift 2 ;;
    --train-repeats) TRAIN_REPEATS="$2"; shift 2 ;;
    --sample-interval) SAMPLE_INTERVAL="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --sm-occupancy-source) SM_OCCUPANCY_SOURCE="$2"; shift 2 ;;
    --bootstrap-command) BOOTSTRAP_COMMAND="$2"; shift 2 ;;
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
case "$STAGE" in
  all|prepare|prepare-sources|profile|profile-labels|package|package-results) ;;
  *) echo "--stage must be one of all, prepare, profile, package" >&2; exit 2 ;;
esac

PACK_DIR="${WORKFLOW_DIR}/pack"
PROFILE_DATASET_DIR="${PACK_DIR}/profile_datasets"
RESULTS_DIR="${WORKFLOW_DIR}/results/${HARDWARE_ID}"
PACKAGE_PATH="${WORKFLOW_DIR}/perfseer_${HARDWARE_ID}_source_labels.tar.gz"
DATASET_DIR="${WORKFLOW_DIR}/dataset/${HARDWARE_ID}"
DATASET_PACKAGE_PATH="${WORKFLOW_DIR}/perfseer_${HARDWARE_ID}_dataset.tar.gz"

BOOTSTRAP_BLOCK=""
if [[ -n "$BOOTSTRAP_COMMAND" ]]; then
  BOOTSTRAP_BLOCK="${BOOTSTRAP_COMMAND}"
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
  name: ${JOB_PREFIX}-prepare-sources
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: ${JOB_PREFIX}
        stage: prepare-sources
    spec:
      restartPolicy: Never
      containers:
      - name: prepare-sources
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        workingDir: ${REPO_DIR}
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:\${PYTHONPATH:-}"
          ${BOOTSTRAP_BLOCK}
          python nrp_calibration_pack/generate_model_sources.py \
            --catalog-mode template \
            --subset-size ${SUBSET_SIZE} \
            --seed ${SEED} \
            --out-dir ${PACK_DIR} \
            --precision-sweep fp32_ieee \
            --validation-mode compile \
            --generation-workers ${GENERATION_WORKERS} \
            --low-precision-focus ${LOW_PRECISION_FOCUS} \
            --force
          python nrp_calibration_pack/profile/make_profile_datasets.py \
            --manifest ${PACK_DIR}/manifest/subset_manifest.jsonl \
            --output-dir ${PROFILE_DATASET_DIR} \
            --train-repeats ${TRAIN_REPEATS} \
            --infer-repeats ${INFER_REPEATS} \
            --seed ${SEED} \
            --force
          printf '%s\n' \
            "repo_dir=${REPO_DIR}" \
            "subset_size=${SUBSET_SIZE}" \
            "seed=${SEED}" \
            "low_precision_focus=${LOW_PRECISION_FOCUS}" \
            "source_manifest=${PACK_DIR}/manifest/subset_manifest.jsonl" \
            "profile_dataset_dir=${PROFILE_DATASET_DIR}" \
            > ${WORKFLOW_DIR}/prepare_provenance.txt
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
          limits:
            cpu: "16"
            memory: "64Gi"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: ${PVC}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-profile-labels
  namespace: ${NAMESPACE}
spec:
  completionMode: Indexed
  completions: ${COMPLETIONS}
  parallelism: ${PARALLELISM}
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: ${JOB_PREFIX}
        stage: profile-labels
    spec:
      restartPolicy: Never
${AFFINITY_BLOCK}
      containers:
      - name: profile-labels
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        workingDir: ${REPO_DIR}
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: JOB_COMPLETIONS
          value: "${COMPLETIONS}"
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:\${PYTHONPATH:-}"
          ${BOOTSTRAP_BLOCK}
          python nrp_calibration_pack/profile/run_profile.py \
            --manifest ${PACK_DIR}/manifest/subset_manifest.jsonl \
            --models-dir ${PACK_DIR}/models \
            --output-dir ${RESULTS_DIR} \
            --hardware-id ${HARDWARE_ID} \
            --precision-sweep auto \
            --profile-dataset-dir ${PROFILE_DATASET_DIR} \
            --device cuda \
            --warmup ${WARMUP} \
            --infer-repeats ${INFER_REPEATS} \
            --train-repeats ${TRAIN_REPEATS} \
            --sample-interval ${SAMPLE_INTERVAL} \
            --optimizer ${OPTIMIZER} \
            --sm-occupancy-source ${SM_OCCUPANCY_SOURCE} \
            --num-shards ${COMPLETIONS} \
            --shard-index \${JOB_COMPLETION_INDEX:-0}
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
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-package-results
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: ${JOB_PREFIX}
        stage: package-results
    spec:
      restartPolicy: Never
      containers:
      - name: package-results
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        workingDir: ${REPO_DIR}
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:\${PYTHONPATH:-}"
          ${BOOTSTRAP_BLOCK}
          python nrp_calibration_pack/package_source_tar.py \
            --pack-dir ${PACK_DIR} \
            --results-dir ${RESULTS_DIR} \
            --out ${PACKAGE_PATH} \
            --note "source-first NRP package; replay with scripts/rebuild_source_tar_dataset.py"
          python scripts/rebuild_source_tar_dataset.py \
            --source-tar ${PACKAGE_PATH} \
            --out-root ${DATASET_DIR} \
            --force
          tar -C ${DATASET_DIR} -czf ${DATASET_PACKAGE_PATH} .
          tar -tzf ${PACKAGE_PATH} | tee ${WORKFLOW_DIR}/package_contents.txt
          tar -tzf ${DATASET_PACKAGE_PATH} | tee ${WORKFLOW_DIR}/dataset_package_contents.txt
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
          limits:
            cpu: "8"
            memory: "32Gi"
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
  case "$STAGE" in
    all) printf '%s\n' "$YAML" ;;
    prepare|prepare-sources) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 0 {print}' ;;
    profile|profile-labels) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 1 {print}' ;;
    package|package-results) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 2 {print}' ;;
  esac
else
  if [[ "$STAGE" == "all" ]]; then
    echo "Refusing to submit all stages at once because prepare, profile, and package have PVC data dependencies." >&2
    echo "Submit them in order with --stage prepare, wait for completion, then --stage profile, then --stage package." >&2
    exit 2
  fi
  case "$STAGE" in
    prepare|prepare-sources) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 0 {print}' | kubectl apply -f - ;;
    profile|profile-labels) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 1 {print}' | kubectl apply -f - ;;
    package|package-results) printf '%s\n' "$YAML" | awk 'BEGIN {doc = 0} /^---$/ {doc += 1; next} doc == 2 {print}' | kubectl apply -f - ;;
  esac
fi
