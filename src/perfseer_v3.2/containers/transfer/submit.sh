#!/usr/bin/env bash
set -euo pipefail
transfer_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
transfer_namespace=ecepxie
transfer_job="perfseer-rtx5090-a100-$(date -u +%Y%m%d%H%M%S)"
transfer_pvc=""
transfer_image=""
transfer_tag=""
transfer_dry_run=false
transfer_server_dry_run=false
transfer_teacher_epochs=100
transfer_student_epochs=100
transfer_learning_rate=0.0001
transfer_batch=64
transfer_microbatch=4
while (( $# )); do
  case "$1" in
    --namespace) transfer_namespace=$2; shift 2 ;;
    --job) transfer_job=$2; shift 2 ;;
    --pvc) transfer_pvc=$2; shift 2 ;;
    --image) transfer_image=$2; shift 2 ;;
    --image-tag) transfer_tag=$2; shift 2 ;;
    --teacher-epochs) transfer_teacher_epochs=$2; shift 2 ;;
    --student-epochs) transfer_student_epochs=$2; shift 2 ;;
    --learning-rate) transfer_learning_rate=$2; shift 2 ;;
    --effective-batch) transfer_batch=$2; shift 2 ;;
    --microbatch) transfer_microbatch=$2; shift 2 ;;
    --dry-run) transfer_dry_run=true; shift ;;
    --server-dry-run) transfer_dry_run=true; transfer_server_dry_run=true; shift ;;
    --help) echo "Usage: bash submit.sh [--image registry/repository@sha256:DIGEST | --image-tag registry/repository:tag] [--namespace ecepxie] [--job NAME] [--pvc NAME] [--dry-run | --server-dry-run] [--teacher-epochs 100] [--student-epochs 100] [--learning-rate 0.0001] [--effective-batch 64] [--microbatch 4]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
transfer_pvc=${transfer_pvc:-$transfer_job-output}
python3 "$transfer_root/verify_package.py"
if [[ -n "$transfer_image" && -n "$transfer_tag" ]]; then
  echo "Choose either --image or --image-tag" >&2; exit 2
fi
if [[ "$transfer_dry_run" == true && -z "$transfer_image" ]]; then
  transfer_image="example.invalid/perfseer/rtx5090@sha256:$(printf '%064d' 0)"
elif [[ -z "$transfer_image" ]]; then
  [[ -n "$transfer_tag" ]] || { echo "Supply a pushed --image digest or --image-tag for build/push" >&2; exit 2; }
  command -v docker >/dev/null
fi
if [[ "$transfer_dry_run" != true || "$transfer_server_dry_run" == true ]]; then
  command -v kubectl >/dev/null
  kubectl config current-context
  kubectl get resourcequota --namespace "$transfer_namespace" --request-timeout=20s
  [[ $(kubectl auth can-i create jobs --namespace "$transfer_namespace") == yes ]]
fi
if [[ "$transfer_dry_run" != true && -n "$transfer_tag" ]]; then
  docker build --platform linux/amd64 --provenance=false -t "$transfer_tag" "$transfer_root"
  docker push "$transfer_tag"
  transfer_image=$(docker image inspect "$transfer_tag" --format '{{index .RepoDigests 0}}')
fi
transfer_repository=$(git -C "$transfer_root" rev-parse --show-toplevel 2>/dev/null || true)
transfer_record="${PERFSEER_RECORD_ROOT:-${transfer_repository:-$transfer_root}/record}/$transfer_job"
mkdir -p "$transfer_record"
python3 - "$transfer_root" "$transfer_record" "$transfer_namespace" "$transfer_job" "$transfer_pvc" "$transfer_image" "$transfer_teacher_epochs" "$transfer_student_epochs" "$transfer_learning_rate" "$transfer_batch" "$transfer_microbatch" <<'PY'
import json, math, re, sys
from pathlib import Path
root, record = map(Path, sys.argv[1:3])
namespace, job, pvc, image, teacher_epochs, student_epochs, lr, batch, microbatch = sys.argv[3:]
for name in (namespace, job, pvc):
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", name) or len(name) > 63:
        raise SystemExit("Names must be DNS labels of at most 63 characters")
if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
    raise SystemExit("A published immutable image digest is required")
if min(int(teacher_epochs), int(student_epochs), int(batch), int(microbatch)) < 1 or not math.isfinite(float(lr)) or float(lr) <= 0:
    raise SystemExit("Training settings must be positive and finite")
dataset = json.loads((root / "dataset/dataset_manifest.json").read_text())
package = json.loads((root / "PACKAGE-MANIFEST.json").read_text())
resources = {"cpu": "8", "memory": "24Gi", "ephemeral-storage": "16Gi", "nvidia.com/a100": "1"}
security = {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}}
spec = {"restartPolicy": "Never", "terminationGracePeriodSeconds": 300,
        "nodeSelector": {"kubernetes.io/arch": "amd64"},
        "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
            {"matchExpressions": [{"key": "nvidia.com/cuda.driver.major", "operator": "Gt", "values": ["569"]},
                                  {"key": "nvidia.com/gpu.memory", "operator": "Gt", "values": ["75000"]}]}]}}},
        "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch"},
        "initContainers": [{"name": "prepare-output", "image": image, "command": ["sh", "-c", "mkdir -p /outputs/run && chmod 1777 /outputs/run"],
                            "securityContext": {"runAsNonRoot": False, "runAsUser": 0, "allowPrivilegeEscalation": False},
                            "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "100m", "memory": "128Mi"}},
                            "volumeMounts": [{"name": "output", "mountPath": "/outputs"}]}],
        "containers": [{"name": "trainer", "image": image, "imagePullPolicy": "IfNotPresent", "securityContext": security,
                        "command": ["bash", "/opt/perfseer/train.sh"],
                        "args": ["--teacher-epochs", teacher_epochs, "--student-epochs", student_epochs, "--learning-rate", lr, "--effective-batch", batch, "--microbatch", microbatch],
                        "env": [{"name": "CUDA_CACHE_PATH", "value": "/tmp/cuda-cache"},
                                {"name": "PERFSEER_EXPECTED_DATASET", "value": dataset["fingerprint"]},
                                {"name": "PERFSEER_EXPECTED_TEACHER", "value": package["base_teacher_sha256"]}],
                        "resources": {"requests": resources, "limits": resources},
                        "volumeMounts": [{"name": "output", "mountPath": "/outputs"}, {"name": "shm", "mountPath": "/dev/shm"}, {"name": "tmp", "mountPath": "/tmp"}]}],
        "volumes": [{"name": "output", "persistentVolumeClaim": {"claimName": pvc}},
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}, {"name": "tmp", "emptyDir": {"sizeLimit": "8Gi"}}]}
manifest = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": job, "namespace": namespace,
            "annotations": {"perfseer.ai/prediction-hardware": dataset["prediction_hardware"], "perfseer.ai/dataset-fingerprint": dataset["fingerprint"]}},
            "spec": {"backoffLimit": 2, "activeDeadlineSeconds": 172800, "template": {"spec": spec}}}
claim = {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": pvc, "namespace": namespace},
         "spec": {"accessModes": ["ReadWriteMany"], "storageClassName": "rook-cephfs", "resources": {"requests": {"storage": "40Gi"}}}}
for name, value in (("job.json", manifest), ("pvc.json", claim)):
    (record / name).write_text(json.dumps(value, indent=2) + "\n")
print(f"Rendered one A100 Job: {record / 'job.json'}; output PVC: {pvc}")
PY
if [[ "$transfer_dry_run" == true ]]; then
  if [[ "$transfer_server_dry_run" == true ]]; then
    kubectl create --dry-run=server --namespace "$transfer_namespace" -f "$transfer_record/pvc.json"
    kubectl create --dry-run=server --namespace "$transfer_namespace" -f "$transfer_record/job.json"
  fi
  echo "Dry run complete; no image published and no cluster resources created."
  exit 0
fi
kubectl get pods --namespace "$transfer_namespace" --request-timeout=20s -o json > "$transfer_record/pods-before.json"
python3 - "$transfer_record/pods-before.json" "$transfer_pvc" <<'PY'
import json, sys
for pod in json.load(open(sys.argv[1]))["items"]:
    if pod.get("status", {}).get("phase") not in ("Succeeded", "Failed") and any(
        v.get("persistentVolumeClaim", {}).get("claimName") == sys.argv[2] for v in pod["spec"].get("volumes", [])):
        raise SystemExit("Output PVC is in use by " + pod["metadata"]["name"])
PY
if [[ -z $(kubectl get pvc "$transfer_pvc" --namespace "$transfer_namespace" --ignore-not-found -o name) ]]; then
  kubectl create --namespace "$transfer_namespace" -f "$transfer_record/pvc.json"
fi
kubectl create --dry-run=server --namespace "$transfer_namespace" -f "$transfer_record/job.json"
kubectl create --namespace "$transfer_namespace" -f "$transfer_record/job.json"
transfer_feedback_status=0
bash "$transfer_root/monitor.sh" "$transfer_namespace" "$transfer_job" once | tee "$transfer_record/immediate-feedback.log" || transfer_feedback_status=$?
nohup bash "$transfer_root/monitor.sh" "$transfer_namespace" "$transfer_job" > "$transfer_record/monitor.log" 2>&1 < /dev/null &
transfer_monitor_pid=$!
echo "$transfer_monitor_pid" > "$transfer_record/monitor.pid"
kill -0 "$transfer_monitor_pid"
echo "Submitted $transfer_job; monitor: $transfer_record/monitor.log (PID $transfer_monitor_pid)"
echo "Check progress: tail -f '$transfer_record/monitor.log'"
echo "Outputs remain on PVC $transfer_pvc under /run."
exit "$transfer_feedback_status"
