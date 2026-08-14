#!/usr/bin/env bash
set -u

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 NAMESPACE JOB_NAME [LOG_PATH]" >&2
  exit 2
fi

monitor_namespace=$1
monitor_job=$2
monitor_repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
monitor_log=${3:-"$monitor_repository/record/${monitor_job}-monitor.log"}

if [[ ! $monitor_namespace =~ ^[a-z0-9][-a-z0-9]*[a-z0-9]$ ]]; then
  echo "invalid namespace" >&2
  exit 2
fi
if [[ ! $monitor_job =~ ^[a-z0-9][-a-z0-9]*[a-z0-9]$ ]]; then
  echo "invalid Job name" >&2
  exit 2
fi

mkdir -p "$(dirname "$monitor_log")"
touch "$monitor_log"
chmod 0600 "$monitor_log"
monitor_iteration=0

while true; do
  monitor_terminal=""
  {
    echo "===== $(date --iso-8601=seconds) iteration=$monitor_iteration ====="
    kubectl get job "$monitor_job" --namespace "$monitor_namespace" -o wide || true
    kubectl get pod --namespace "$monitor_namespace" -l "job-name=$monitor_job" -o wide || true
    monitor_pods=$(kubectl get pod --namespace "$monitor_namespace" -l "job-name=$monitor_job" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
    for monitor_pod in $monitor_pods; do
      kubectl describe pod "$monitor_pod" --namespace "$monitor_namespace" || true
      kubectl logs "$monitor_pod" --namespace "$monitor_namespace" --all-containers=true --tail=500 || true
      kubectl get pod "$monitor_pod" --namespace "$monitor_namespace" -o jsonpath='{range .status.containerStatuses[*]}name={.name} ready={.ready} restarts={.restartCount} running={.state.running.startedAt} reason={.state.terminated.reason} exitCode={.state.terminated.exitCode}{"\n"}{end}' || true
    done
    kubectl get events --namespace "$monitor_namespace" --sort-by=.lastTimestamp | tail -100 || true
    monitor_terminal=$(kubectl get job "$monitor_job" --namespace "$monitor_namespace" \
      -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}' \
      2>/dev/null || true)
    if [[ -n $monitor_terminal ]]; then
      echo "terminal_conditions=$monitor_terminal"
    fi
  } >>"$monitor_log" 2>&1

  monitor_iteration=$((monitor_iteration + 1))
  if grep -Eq '^(Complete|Failed)$' <<<"$monitor_terminal"; then
    break
  fi
  if (( monitor_iteration <= 5 )); then
    sleep 60
  else
    sleep 1200
  fi
done
