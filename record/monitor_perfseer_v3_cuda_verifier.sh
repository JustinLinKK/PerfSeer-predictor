#!/usr/bin/env bash
set -u

namespace="${PERFSEER_NAUTILUS_NAMESPACE:-ecepxie}"
job="${PERFSEER_NAUTILUS_JOB:-perfseer-v3-cuda-verifier}"
log_path="${1:-record/perfseer_v3_cuda_verifier_monitor.log}"
started_at="$(date +%s)"

mkdir -p "$(dirname "$log_path")"
while true; do
  now="$(date +%s)"
  elapsed="$((now - started_at))"
  {
    echo "===== $(date --iso-8601=seconds) elapsed=${elapsed}s ====="
    kubectl get job "$job" --namespace "$namespace" -o wide 2>&1 || true
    kubectl get pods --namespace "$namespace" -l "job-name=$job" -o wide 2>&1 || true
    pod="$(kubectl get pods --namespace "$namespace" -l "job-name=$job" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    if [[ -n "$pod" ]]; then
      kubectl describe pod "$pod" --namespace "$namespace" 2>&1 || true
      kubectl get pod "$pod" --namespace "$namespace" -o jsonpath='phase={.status.phase} init={.status.initContainerStatuses[*].state} process={.status.containerStatuses[*].state}{"\n"}' 2>&1 || true
      kubectl logs "$pod" --namespace "$namespace" --all-containers --tail=120 2>&1 || true
      kubectl exec "$pod" --namespace "$namespace" -c verifier -- /bin/bash -lc \
        'tail -n 80 /workspace/perfseer_v3_cuda_verifier.log 2>/dev/null || true; find /workspace -maxdepth 2 -type f \( -name "*.log" -o -name "*.json" -o -name "*.pt" \) -printf "%p %s bytes\n" 2>/dev/null | sort' \
        2>&1 || true
    fi
    kubectl get events --namespace "$namespace" --field-selector "involvedObject.name=$job" --sort-by='.lastTimestamp' 2>&1 || true
    if [[ -n "${pod:-}" ]]; then
      kubectl get events --namespace "$namespace" --field-selector "involvedObject.name=$pod" --sort-by='.lastTimestamp' 2>&1 || true
    fi
  } >>"$log_path"

  status="$(kubectl get job "$job" --namespace "$namespace" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
  if [[ "$status" == *Complete* || "$status" == *Failed* ]]; then
    break
  fi
  if (( elapsed < 300 )); then
    # Leave time for the kubectl collection itself so record starts stay
    # within the required sixty-second interval.
    sleep 45
  else
    sleep 1200
  fi
done
