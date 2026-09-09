#!/usr/bin/env bash
set -euo pipefail

controller_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
renderer="$controller_root/scripts/render_nautilus_a10_manifest.py"
workflow_cli="$controller_root/scripts/run_nautilus_a10_modality.py"

usage() {
  cat <<'EOF'
Usage: submit_nautilus_a10_<modality>.sh ACTION [OPTIONS]

Actions:
  analyze                  Print the frozen local dataset allocation.
  render-pod               Render an interactive debug Pod to stdout.
  submit-pod               Submit only the rendered debug Pod.
  render-pilot-job         Render a bounded pilot Job to stdout.
  submit-pilot-job         Submit a bounded pilot Job, collect feedback, monitor.
  render-production-job    Render a resumable production Job to stdout.
  submit-production-job    Submit a production Job, collect feedback, monitor.
  status                   Collect current Job/Pod/log/event feedback.
  monitor                  Run the durable local monitoring loop.
  verify                   Verify a completed modality workspace locally.

Required options for every action except analyze:
  --namespace NAME
  --pvc NAME
  --repository-url URL
  --revision 40_HEX_COMMIT
  --kaggle-secret NAME      Secret containing the exact key kaggle.json.
  --image NAME@sha256:64_HEX
  --run-id DNS_LABEL

Additional options:
  --workspace PATH          Required by verify; the mounted modality workspace.
  --gpu-product LABEL       Default: NVIDIA-A10.
  --cpu VALUE               Default: 8.
  --memory VALUE            Default: 32Gi.
  --pilot-labels COUNT      Default: 8.
  --job-mode MODE           status/monitor target: pilot or production.

Safety: no action is selected by default. Render and analyze never call a
Kubernetes executable. Only submit actions call apply. KUBECTL_BIN and
PYTHON_BIN may be injected for offline testing.
EOF
}

die() {
  printf '%s\n' "error: $*" >&2
  exit 2
}

modality=${PERFSEER_MODALITY_WRAPPER:-}
case "$modality" in
  audio|tabular|graph|generated) ;;
  *) die "wrapper did not select a valid modality" ;;
esac
family_id=${PERFSEER_FAMILY_WRAPPER:-}
case "$family_id" in
  "") ;;
  panns_cnn14)
    [[ $modality == audio ]] || die "PANNs CNN14 must use the audio modality"
    ;;
  *) die "wrapper did not select a supported family" ;;
esac
selection=${family_id:-$modality}
resource_segment=${selection//_/-}

if (($# == 0)); then
  usage
  exit 0
fi
action=$1
shift
option_args=("$@")

namespace=
pvc=
repository_url=
revision=
kaggle_secret=
image=
run_id=
workspace=
gpu_product=NVIDIA-A10
cpu=8
memory=32Gi
pilot_labels=8
job_mode=production

while (($#)); do
  case "$1" in
    --namespace|--pvc|--repository-url|--revision|--kaggle-secret|--image|--run-id|--workspace|--gpu-product|--cpu|--memory|--pilot-labels|--job-mode)
      (($# >= 2)) || die "$1 requires a value"
      option=$1
      value=$2
      shift 2
      case "$option" in
        --namespace) namespace=$value ;;
        --pvc) pvc=$value ;;
        --repository-url) repository_url=$value ;;
        --revision) revision=$value ;;
        --kaggle-secret) kaggle_secret=$value ;;
        --image) image=$value ;;
        --run-id) run_id=$value ;;
        --workspace) workspace=$value ;;
        --gpu-product) gpu_product=$value ;;
        --cpu) cpu=$value ;;
        --memory) memory=$value ;;
        --pilot-labels) pilot_labels=$value ;;
        --job-mode) job_mode=$value ;;
      esac
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) die "unknown option $1" ;;
  esac
done

python_bin=${PYTHON_BIN:-python3}
kubectl_bin=${KUBECTL_BIN:-kubectl}

if [[ $action == analyze ]]; then
  selection_args=(--modality "$modality")
  if [[ -n $family_id ]]; then
    selection_args+=(--family-id "$family_id")
  fi
  (cd "$controller_root" && PYTHONPATH="$controller_root/src" "$python_bin" "$workflow_cli" analyze "${selection_args[@]}")
  exit 0
fi

for required_name in namespace pvc repository_url revision kaggle_secret image run_id; do
  [[ -n ${!required_name} ]] || die "--${required_name//_/-} is required"
done

render_mode() {
  local mode=$1
  local family_args=()
  if [[ -n $family_id ]]; then
    family_args=(--family-id "$family_id")
  fi
  (cd "$controller_root" && "$python_bin" "$renderer" \
    --modality "$modality" \
    "${family_args[@]}" \
    --mode "$mode" \
    --namespace "$namespace" \
    --pvc "$pvc" \
    --repository-url "$repository_url" \
    --revision "$revision" \
    --kaggle-secret "$kaggle_secret" \
    --image "$image" \
    --run-id "$run_id" \
    --gpu-product "$gpu_product" \
    --cpu "$cpu" \
    --memory "$memory" \
    --pilot-labels "$pilot_labels")
}

resource_name() {
  local mode=$1
  printf 'perfseer-a10-%s-%s-%s\n' "$resource_segment" "$mode" "$run_id"
}

require_kubernetes_executable() {
  [[ $kubectl_bin != *[[:space:]]* ]] || die "KUBECTL_BIN must be one executable path"
  [[ -x $kubectl_bin ]] || command -v -- "$kubectl_bin" >/dev/null 2>&1 || die "configured Kubernetes executable is unavailable"
}

pod_for_job() {
  local name=$1
  "$kubectl_bin" get pods --namespace "$namespace" \
    --selector "job-name=$name" \
    --output 'jsonpath={.items[0].metadata.name}'
}

feedback_job() {
  local name=$1
  "$kubectl_bin" get job "$name" --namespace "$namespace" --output wide || true
  "$kubectl_bin" get pods --namespace "$namespace" --selector "job-name=$name" --output wide || true
  local pod_name
  pod_name=$(pod_for_job "$name" 2>/dev/null || true)
  if [[ -n $pod_name ]]; then
    "$kubectl_bin" describe pod "$pod_name" --namespace "$namespace" || true
    "$kubectl_bin" logs "$pod_name" --namespace "$namespace" --all-containers=true --tail=200 || true
    "$kubectl_bin" get events --namespace "$namespace" \
      --field-selector "involvedObject.name=$pod_name" \
      --sort-by=.lastTimestamp || true
  else
    "$kubectl_bin" get events --namespace "$namespace" --sort-by=.lastTimestamp || true
  fi
}

feedback_pod() {
  local name=$1
  "$kubectl_bin" get pod "$name" --namespace "$namespace" --output wide || true
  "$kubectl_bin" describe pod "$name" --namespace "$namespace" || true
  "$kubectl_bin" logs "$name" --namespace "$namespace" --all-containers=true --tail=200 || true
  "$kubectl_bin" get events --namespace "$namespace" \
    --field-selector "involvedObject.name=$name" \
    --sort-by=.lastTimestamp || true
}

launch_monitor() {
  local monitored_mode=$1
  mkdir -p "$controller_root/record"
  local monitor_log="$controller_root/record/nautilus_a10_${selection}_${run_id}_monitor.log"
  nohup env NAUTILUS_MONITOR_JOB_MODE="$monitored_mode" \
    "$0" monitor "${option_args[@]}" >>"$monitor_log" 2>&1 </dev/null &
  printf '%s\n' "monitor started: $monitor_log"
}

submit_manifest() {
  local mode=$1
  local short_mode=$2
  require_kubernetes_executable
  local manifest_file
  manifest_file=$(mktemp "${TMPDIR:-/tmp}/perfseer-nautilus-manifest.XXXXXX.yaml")
  trap 'rm -f -- "$manifest_file"' RETURN
  render_mode "$mode" >"$manifest_file"
  "$kubectl_bin" apply --namespace "$namespace" --filename "$manifest_file"
  local name
  name=$(resource_name "$short_mode")
  if [[ $mode == pod ]]; then
    feedback_pod "$name"
  else
    feedback_job "$name"
    if [[ ${NAUTILUS_DISABLE_AUTO_MONITOR:-0} != 1 ]]; then
      launch_monitor "$short_mode"
    fi
  fi
  rm -f -- "$manifest_file"
  trap - RETURN
}

monitor_job() {
  require_kubernetes_executable
  local name
  local monitored_mode=${NAUTILUS_MONITOR_JOB_MODE:-$job_mode}
  [[ $monitored_mode == pilot || $monitored_mode == production ]] || die "--job-mode must be pilot or production"
  name=$(resource_name "$monitored_mode")
  local iteration=0
  local maximum=${NAUTILUS_MONITOR_ITERATIONS:-0}
  while :; do
    printf '\n[%s] modality=%s family=%s run_id=%s iteration=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$modality" "${family_id:-all}" "$run_id" "$iteration"
    local job_output
    job_output=$("$kubectl_bin" get job "$name" --namespace "$namespace" --output wide 2>&1 || true)
    printf '%s\n' "$job_output"
    "$kubectl_bin" get pods --namespace "$namespace" --selector "job-name=$name" --output wide || true
    local pod_name
    pod_name=$(pod_for_job "$name" 2>/dev/null || true)
    if [[ -n $pod_name ]]; then
      "$kubectl_bin" get events --namespace "$namespace" \
        --field-selector "involvedObject.name=$pod_name" \
        --sort-by=.lastTimestamp || true
      "$kubectl_bin" logs "$pod_name" --namespace "$namespace" --all-containers=true --tail=120 || true
      "$kubectl_bin" exec "$pod_name" --namespace "$namespace" -- /bin/bash -lc \
        'ps -eo pid,ppid,stat,etime,cmd; run_root=${PERFSEER_RUN_ROOT:?}; find "$run_root/workspace/attempts/logs" -type f -maxdepth 1 -print -exec tail -n 20 {} \; 2>/dev/null || true; find "$run_root/workspace" \( -path "*/state/*" -o -path "*/attempts/accepted/*" -o -path "*/checkpoints/*" \) -type f -printf "%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n" 2>/dev/null | tail -n 120' || true
    fi
    iteration=$((iteration + 1))
    if [[ $job_output == *Failed* ]]; then
      local exit_code=unknown
      if [[ -n $pod_name ]]; then
        exit_code=$("$kubectl_bin" get pod "$pod_name" --namespace "$namespace" \
          --output 'jsonpath={.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)
        printf 'failed pod: %s; exit code: %s\n' "$pod_name" "${exit_code:-unknown}"
        "$kubectl_bin" logs "$pod_name" --namespace "$namespace" --all-containers=true --tail=200 || true
      fi
      printf '%s\n' 'next corrective action: inspect the pod events and last labeler error, correct that root cause, then submit a new Job with the same run ID to resume the PVC workspace.'
      break
    fi
    if [[ $job_output == *Complete* || $job_output == *"1/1"* ]]; then
      printf '%s\n' 'job completed; run the verify action against the mounted workspace.'
      break
    fi
    if ((maximum > 0 && iteration >= maximum)); then
      break
    fi
    if ((iteration <= 5)); then
      sleep "${NAUTILUS_MONITOR_INITIAL_SECONDS:-60}"
    else
      sleep "${NAUTILUS_MONITOR_STEADY_SECONDS:-1200}"
    fi
  done
}

case "$action" in
  render-pod) render_mode pod ;;
  render-pilot-job) render_mode pilot-job ;;
  render-production-job) render_mode production-job ;;
  submit-pod) submit_manifest pod debug ;;
  submit-pilot-job) submit_manifest pilot-job pilot ;;
  submit-production-job) submit_manifest production-job production ;;
  status)
    require_kubernetes_executable
    [[ $job_mode == pilot || $job_mode == production ]] || die "--job-mode must be pilot or production"
    feedback_job "$(resource_name "$job_mode")"
    ;;
  monitor) monitor_job ;;
  verify)
    [[ -n $workspace ]] || die "--workspace is required for verify"
    family_args=()
    if [[ -n $family_id ]]; then
      family_args=(--family-id "$family_id")
    fi
    (cd "$controller_root" && PYTHONPATH="$controller_root/src" "$python_bin" "$workflow_cli" verify \
      --modality "$modality" \
      "${family_args[@]}" \
      --workspace "$workspace" \
      --repository-revision "$revision" \
      --image-digest "${image##*@}")
    ;;
  *)
    usage >&2
    die "unknown action $action"
    ;;
esac
