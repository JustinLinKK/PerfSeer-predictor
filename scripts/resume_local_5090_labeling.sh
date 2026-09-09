#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
HARDWARE_ID="${HARDWARE_ID:-rtx5090}"
DEVICE="${DEVICE:-cuda}"
PROFILE_PRECISION_SWEEP="${PROFILE_PRECISION_SWEEP:-auto}"
OPTIMIZER="${OPTIMIZER:-adam}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
WARMUP="${WARMUP:-20}"
INFER_REPEATS="${INFER_REPEATS:-50}"
TRAIN_REPEATS="${TRAIN_REPEATS:-50}"
LABEL_TIME_MODE="${LABEL_TIME_MODE:-measured_epochs}"
TIME_LABEL_WARMUP_EPOCHS="${TIME_LABEL_WARMUP_EPOCHS:-1}"
TIME_LABEL_MEASURED_EPOCHS="${TIME_LABEL_MEASURED_EPOCHS:-2}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.01}"
MIN_PHASE_SECONDS="${MIN_PHASE_SECONDS:-20}"
MIN_SAMPLER_SAMPLES="${MIN_SAMPLER_SAMPLES:-100}"
SM_OCCUPANCY_SOURCE="${SM_OCCUPANCY_SOURCE:-nvml_proxy}"
RESOURCE_PROFILE_MODE="${RESOURCE_PROFILE_MODE:-sustained}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"

PACK_DIR="${PACK_DIR:-${REPO_ROOT}/nrp_calibration_pack}"
PACK_TOOL_DIR="${REPO_ROOT}/nrp_calibration_pack"
WORKLOAD_DIR="${WORKLOAD_DIR:-${PACK_DIR}/workload_specs_balanced_local}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/nrp_results_rtx5090_balanced_local}"
RECORD_DIR="${RECORD_DIR:-${REPO_ROOT}/record}"
RUN_ID="${RUN_ID:-local_5090_balanced_labeling_resume_$(date +%Y%m%d_%H%M%S)}"
PID_FILE="${PID_FILE:-${RECORD_DIR}/local_5090_balanced_labeling.pid}"
LOG_FILE="${LOG_FILE:-${RECORD_DIR}/${RUN_ID}.log}"

mkdir -p "${RECORD_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
cd "${REPO_ROOT}"

find_labeler_pids() {
  local pid cwd cmd
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${cmd}" == *"${REPO_ROOT}/nrp_calibration_pack/profile/run_profile.py"* ]] || {
      [[ "${cmd}" == *"nrp_calibration_pack/profile/run_profile.py"* ]] && [[ "${cwd}" == "${REPO_ROOT}" ]]
    }; then
      printf '%s\n' "${pid}"
    fi
  done < <(pgrep -f "nrp_calibration_pack/profile/run_profile.py" || true)
}

mapfile -t active < <(find_labeler_pids)
if (( ${#active[@]} > 0 )); then
  echo "resume_refused=active_labeler pid=${active[*]}" >&2
  exit 1
fi
echo "active_labeler_verifier=none"

WORKLOADS_FILE="${WORKLOAD_DIR}/workloads.jsonl"
test -s "${WORKLOADS_FILE}"
test -d "${PACK_DIR}/models"
"${PYTHON}" - "${WORKLOADS_FILE}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("resume verifier failed: no workload rows")
print(f"resume_workload_verifier=ok rows={len(rows)}")
PY

profile_cmd=(
  "${PYTHON}" "${PACK_TOOL_DIR}/profile/run_profile.py"
  --workload-specs "${WORKLOADS_FILE}"
  --models-dir "${PACK_DIR}/models"
  --output-dir "${OUTPUT_DIR}"
  --hardware-id "${HARDWARE_ID}"
  --device "${DEVICE}"
  --precision-sweep "${PROFILE_PRECISION_SWEEP}"
  --resume
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
  --warmup "${WARMUP}"
  --infer-repeats "${INFER_REPEATS}"
  --train-repeats "${TRAIN_REPEATS}"
  --label-time-mode "${LABEL_TIME_MODE}"
  --time-label-warmup-epochs "${TIME_LABEL_WARMUP_EPOCHS}"
  --time-label-measured-epochs "${TIME_LABEL_MEASURED_EPOCHS}"
  --sample-interval "${SAMPLE_INTERVAL}"
  --min-phase-seconds "${MIN_PHASE_SECONDS}"
  --min-sampler-samples "${MIN_SAMPLER_SAMPLES}"
  --optimizer "${OPTIMIZER}"
  --sm-occupancy-source "${SM_OCCUPANCY_SOURCE}"
  --resource-profile-mode "${RESOURCE_PROFILE_MODE}"
)

echo "resuming_profile=${profile_cmd[*]}"
nohup "${profile_cmd[@]}" >>"${LOG_FILE}" 2>&1 &
profile_pid="$!"
printf '%s\n' "${profile_pid}" >"${PID_FILE}"
sleep 2
if ! kill -0 "${profile_pid}" 2>/dev/null; then
  echo "resume_launch_verifier=failed pid=${profile_pid}" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
fi
echo "resume_launch_verifier=ok pid=${profile_pid}"
if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
  if wait "${profile_pid}"; then
    echo "wait_verifier=ok pid=${profile_pid}"
  else
    status="$?"
    echo "wait_verifier=failed pid=${profile_pid} exit_status=${status}" >&2
    tail -80 "${LOG_FILE}" >&2 || true
    exit "${status}"
  fi
else
  disown %1 2>/dev/null || disown "${profile_pid}" 2>/dev/null || true
fi

PID_FILE="${PID_FILE}" LOG_FILE="${LOG_FILE}" OUTPUT_DIR="${OUTPUT_DIR}" WORKLOAD_DIR="${WORKLOAD_DIR}" \
  "${SCRIPT_DIR}/pulse_local_5090_labeling.sh"
