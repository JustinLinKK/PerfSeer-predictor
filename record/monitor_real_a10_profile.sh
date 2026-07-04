#!/usr/bin/env bash
set -u

namespace="ecepxie"
job="real-a10-0629060430-profile-real"
selector="app=real-a10-0629060430,stage=profile-real"
log="record/real-a10-0629060430_profile_monitor_restarted.log"
start_epoch="$(date +%s)"
kubectl_timeout="30s"
kill_after="5s"

while true; do
  now_epoch="$(date +%s)"
  {
    echo "===== $(date -Is) ====="
    timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl get job -n "$namespace" "$job" -o wide 2>&1 || true
    timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl get pod -n "$namespace" -l "$selector" -o wide 2>&1 || true
    for ref in $(timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl get pod -n "$namespace" -l "$selector" --field-selector=status.phase=Running -o name 2>/dev/null | head -4 || true); do
      pod="${ref#pod/}"
      echo "--- pod ${pod} logs ---"
      timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl logs -n "$namespace" "$pod" --tail=8 2>&1 || true
    done
    timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl get events -n "$namespace" --sort-by=.lastTimestamp 2>&1 | tail -60 || true
    echo "remote_output_counts=skipped_lightweight_monitor"
  } >> "$log"

  status="$(timeout --kill-after="$kill_after" "$kubectl_timeout" kubectl get job -n "$namespace" "$job" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || true)"
  if [ "$status" = "Complete" ]; then
    exit 0
  fi
  if [ "$status" = "Failed" ]; then
    exit 1
  fi
  elapsed=$((now_epoch - start_epoch))
  if [ "$elapsed" -lt 300 ]; then
    echo "===== $(date -Is) next_sample_seconds=60 =====" >> "$log"
    sleep 60
  else
    echo "===== $(date -Is) next_sample_seconds=900 =====" >> "$log"
    sleep 900
  fi
done
