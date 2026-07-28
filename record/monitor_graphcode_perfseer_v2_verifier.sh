#!/usr/bin/env bash
set -u

namespace="ecepxie"
job="perfseer-graphcode-v2-verifier"
log_path="${1:-record/graphcode_perfseer_v2_verifier_monitor.log}"
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
      kubectl get pod "$pod" --namespace "$namespace" -o jsonpath='phase={.status.phase} container={.status.containerStatuses[0].state}{"\n"}' 2>&1 || true
      kubectl logs "$pod" --namespace "$namespace" --all-containers --tail=80 2>&1 || true
    fi
    kubectl get events --namespace "$namespace" --field-selector "involvedObject.name=$job" --sort-by='.lastTimestamp' 2>&1 || true
    echo "persistent_training_log=not-applicable verifier_output=pod-logs checkpoint_files=not-applicable"
  } >>"$log_path"

  status="$(kubectl get job "$job" --namespace "$namespace" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
  if [[ "$status" == *Complete* || "$status" == *Failed* ]]; then
    break
  fi
  if (( elapsed < 300 )); then
    sleep 60
  else
    sleep 1200
  fi
done
