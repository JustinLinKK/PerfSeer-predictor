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
  monitor_iteration_started=$(date +%s)
  monitor_terminal=""
  {
    echo "===== $(date --iso-8601=seconds) iteration=$monitor_iteration ====="
    kubectl get job "$monitor_job" --namespace "$monitor_namespace" -o wide || true
    kubectl get pod --namespace "$monitor_namespace" -l "job-name=$monitor_job" -o wide || true
    monitor_pods=$(kubectl get pod --namespace "$monitor_namespace" -l "job-name=$monitor_job" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
    for monitor_pod in $monitor_pods; do
      kubectl describe pod "$monitor_pod" --namespace "$monitor_namespace" || true
      kubectl logs "$monitor_pod" --namespace "$monitor_namespace" --all-containers=true --tail=500 --timestamps=true || true
      kubectl get pod "$monitor_pod" --namespace "$monitor_namespace" -o jsonpath='{range .status.containerStatuses[*]}name={.name} ready={.ready} restarts={.restartCount} running={.state.running.startedAt} reason={.state.terminated.reason} exitCode={.state.terminated.exitCode}{"\n"}{end}' || true
      kubectl exec "$monitor_pod" --namespace "$monitor_namespace" -c trainer -- sh -c '
        echo "process_status:"
        for process in /proc/[0-9]*/cmdline; do
          tr "\000" " " < "$process" 2>/dev/null && echo
        done
        echo "persistent_training_log:"
        for training_log in /outputs/perfseer-v31-*/training.log; do
          test ! -f "$training_log" || tail -n 40 "$training_log"
        done
        echo "output_files:"
        find /outputs -maxdepth 3 -type f \( -name "*.pt" -o -name "*-gate.json" -o -name "training.log" \) -printf "%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n" 2>/dev/null | sort
      ' || true
      kubectl get events --namespace "$monitor_namespace" --field-selector "involvedObject.name=$monitor_pod" --sort-by=.lastTimestamp || true
    done
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
    monitor_interval=60
  else
    monitor_interval=1200
  fi
  monitor_delay=$((monitor_interval - $(date +%s) + monitor_iteration_started))
  if (( monitor_delay > 0 )); then
    sleep "$monitor_delay"
  fi
done
