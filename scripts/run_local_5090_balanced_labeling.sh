#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
SUBSET_SIZE="${SUBSET_SIZE:-10005}"
SEED="${SEED:-20260617}"
HARDWARE_ID="${HARDWARE_ID:-rtx5090}"
DEVICE="${DEVICE:-cuda}"
WORKLOAD_SUBSET_ID="${WORKLOAD_SUBSET_ID:-tiny}"
WORKLOAD_BATCH_SIZE="${WORKLOAD_BATCH_SIZE:-8}"
WORKLOAD_PRECISION_SWEEP="${WORKLOAD_PRECISION_SWEEP:-fp32_ieee}"
PROFILE_PRECISION_SWEEP="${PROFILE_PRECISION_SWEEP:-auto}"
OPTIMIZER="${OPTIMIZER:-adam}"
GENERATION_WORKERS="${GENERATION_WORKERS:-0}"
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
STOP_EXISTING="${STOP_EXISTING:-1}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"

PACK_DIR="${PACK_DIR:-${REPO_ROOT}/nrp_calibration_pack}"
PACK_TOOL_DIR="${REPO_ROOT}/nrp_calibration_pack"
WORKLOAD_DIR="${WORKLOAD_DIR:-${PACK_DIR}/workload_specs_balanced_local}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/nrp_results_rtx5090_balanced_local}"
RECORD_DIR="${RECORD_DIR:-${REPO_ROOT}/record}"
RUN_ID="${RUN_ID:-local_5090_balanced_labeling_$(date +%Y%m%d_%H%M%S)}"
PID_FILE="${PID_FILE:-${RECORD_DIR}/local_5090_balanced_labeling.pid}"
LOG_FILE="${LOG_FILE:-${RECORD_DIR}/${RUN_ID}.log}"

mkdir -p "${RECORD_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "run_id=${RUN_ID}"
echo "repo_root=${REPO_ROOT}"
echo "log_file=${LOG_FILE}"
echo "pid_file=${PID_FILE}"
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

stop_existing_labelers() {
  mapfile -t pids < <(find_labeler_pids)
  if (( ${#pids[@]} == 0 )); then
    echo "stop_verifier=no_existing_repo_local_labeler"
    return 0
  fi
  echo "stopping_existing_labelers=${pids[*]}"
  kill -INT "${pids[@]}" 2>/dev/null || true
  sleep 5
  mapfile -t remaining < <(find_labeler_pids)
  if (( ${#remaining[@]} > 0 )); then
    echo "sigint_remaining=${remaining[*]}"
    kill -TERM "${remaining[@]}" 2>/dev/null || true
    sleep 3
  fi
  mapfile -t remaining < <(find_labeler_pids)
  if (( ${#remaining[@]} > 0 )); then
    echo "stop_verifier=failed remaining=${remaining[*]}" >&2
    return 1
  fi
  echo "stop_verifier=ok"
}

preflight() {
  command -v "${PYTHON}" >/dev/null
  command -v nvidia-smi >/dev/null
  "${PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the default local RTX 5090 run; set DEVICE=cpu only for a tiny verifier run")
PY
  nvidia-smi --query-gpu=name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
  "${PYTHON}" - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
registry = root / "dataset_sources" / "registry.json"
prepared = root / "datasets" / "prepared"
rows = json.loads(registry.read_text()).get("datasets", [])
profiles = []
for row in rows:
    if row.get("status") != "prepared":
        continue
    profile = Path(row.get("prepared_profile") or prepared / row["id"] / "dataset_profile.json")
    if not profile.is_absolute():
        profile = root / profile
    if profile.is_file():
        profiles.append(str(profile))
if not profiles:
    raise SystemExit("no prepared local dataset profiles found")
print(f"prepared_profile_count={len(profiles)}")
PY
}

if [[ "${STOP_EXISTING}" != "0" ]]; then
  stop_existing_labelers
fi

preflight

echo "generating_balanced_pack=subset_size_${SUBSET_SIZE}"
"${PYTHON}" "${PACK_TOOL_DIR}/generate_model_sources.py" \
  --out-dir "${PACK_DIR}" \
  --catalog-mode template \
  --subset-size "${SUBSET_SIZE}" \
  --seed "${SEED}" \
  --precision-sweep fp32_ieee \
  --generation-workers "${GENERATION_WORKERS}" \
  --force

test -s "${PACK_DIR}/manifest/subset_manifest.jsonl"
echo "manifest_verifier=$(wc -l < "${PACK_DIR}/manifest/subset_manifest.jsonl") rows"

workload_cmd=(
  "${PYTHON}" "${PACK_TOOL_DIR}/profile/make_workload_specs.py"
  --manifest "${PACK_DIR}/manifest/subset_manifest.jsonl"
  --registry "${REPO_ROOT}/dataset_sources/registry.json"
  --dataset-profile-root "${REPO_ROOT}/datasets/prepared"
  --raw-root "${REPO_ROOT}/datasets/raw"
  --output-dir "${WORKLOAD_DIR}"
  --subset-id "${WORKLOAD_SUBSET_ID}"
  --batch-size "${WORKLOAD_BATCH_SIZE}"
  --precision-sweep "${WORKLOAD_PRECISION_SWEEP}"
  --optimizer "${OPTIMIZER}"
  --hardware-id "${HARDWARE_ID}"
  --force
)
if [[ -n "${LIMIT:-}" ]]; then
  workload_cmd+=(--limit "${LIMIT}")
fi
"${workload_cmd[@]}"

WORKLOADS_FILE="${WORKLOAD_DIR}/workloads.jsonl"
"${PYTHON}" - "${WORKLOADS_FILE}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("workload verifier failed: no rows")
not_real = [r["profile_point_id"] for r in rows if not r.get("dataset", {}).get("real_dataloader_backed")]
if not_real:
    raise SystemExit(f"workload verifier failed: {len(not_real)} rows are not real_dataloader_backed")
print(f"workload_verifier=ok rows={len(rows)}")
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

echo "launching_profile=${profile_cmd[*]}"
nohup "${profile_cmd[@]}" >>"${LOG_FILE}" 2>&1 &
profile_pid="$!"
printf '%s\n' "${profile_pid}" >"${PID_FILE}"
sleep 2
if ! kill -0 "${profile_pid}" 2>/dev/null; then
  echo "launch_verifier=failed pid=${profile_pid}" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
fi
echo "launch_verifier=ok pid=${profile_pid}"
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
