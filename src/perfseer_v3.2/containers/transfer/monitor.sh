#!/usr/bin/env bash
set -uo pipefail
monitor_namespace=${1:?namespace required}
monitor_job=${2:?job required}
monitor_mode=${3:-loop}
monitor_started=$(date +%s)
while true; do
  monitor_iteration_started=$(date +%s)
  date --iso-8601=seconds
  kubectl get job "$monitor_job" --namespace "$monitor_namespace" --request-timeout=15s -o wide || true
  kubectl get pod --namespace "$monitor_namespace" --request-timeout=15s -l "job-name=$monitor_job" -o wide || true
  monitor_pods=$(kubectl get pod --namespace "$monitor_namespace" --request-timeout=15s -l "job-name=$monitor_job" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  for monitor_pod in $monitor_pods; do
    kubectl describe pod "$monitor_pod" --namespace "$monitor_namespace" --request-timeout=15s || true
    kubectl logs "$monitor_pod" --namespace "$monitor_namespace" --request-timeout=15s --all-containers=true --tail=80 --timestamps=true || true
    kubectl get pod "$monitor_pod" --namespace "$monitor_namespace" --request-timeout=15s -o jsonpath='{range .status.containerStatuses[*]}name={.name} reason={.state.terminated.reason} exitCode={.state.terminated.exitCode}{"\n"}{end}' || true
    kubectl get events --namespace "$monitor_namespace" --request-timeout=15s --field-selector "involvedObject.name=$monitor_pod" --sort-by=.lastTimestamp || true
    kubectl exec "$monitor_pod" --namespace "$monitor_namespace" --request-timeout=15s -c trainer -- sh -c '
      echo process_status:
      for process in /proc/[0-9]*/cmdline; do tr "\000" " " < "$process" 2>/dev/null; echo; done
      echo persistent_training_log:
      tail -n 30 /outputs/run/training.log 2>/dev/null
      echo checkpoint_and_output_files:
      find /outputs/run -maxdepth 1 -type f -printf "%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n" 2>/dev/null | sort
    ' || true
  done
  kubectl get events --namespace "$monitor_namespace" --request-timeout=15s --field-selector "involvedObject.name=$monitor_job" --sort-by=.lastTimestamp || true
  monitor_terminal=$(kubectl get job "$monitor_job" --namespace "$monitor_namespace" --request-timeout=15s -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}' 2>/dev/null || true)
  if [[ "$monitor_terminal" == *Failed* ]]; then
    echo "FAILED: see pod name, exit code, and last logs above. Correct the reported cause before submitting a new Job with the retained PVC."
    exit 1
  fi
  [[ "$monitor_mode" != once && "$monitor_terminal" != *Complete* ]] || exit 0
  monitor_interval=1200
  (( $(date +%s) - monitor_started >= 300 )) || monitor_interval=60
  monitor_delay=$((monitor_interval - $(date +%s) + monitor_iteration_started))
  (( monitor_delay <= 0 )) || sleep "$monitor_delay"
done
