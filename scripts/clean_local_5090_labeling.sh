#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/nrp_results_rtx5090_balanced_local}"
RECORD_DIR="${RECORD_DIR:-${REPO_ROOT}/record}"
PID_FILE="${PID_FILE:-${RECORD_DIR}/local_5090_balanced_labeling.pid}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${RECORD_DIR}/labeling_archives}"
ARCHIVE_ID="${ARCHIVE_ID:-clean_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"

cd "${REPO_ROOT}"

abs_path() {
  local raw="$1"
  if [[ "${raw}" = /* ]]; then
    printf '%s\n' "${raw}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${raw}"
  fi
}

OUTPUT_DIR="$(abs_path "${OUTPUT_DIR}")"
RECORD_DIR="$(abs_path "${RECORD_DIR}")"
PID_FILE="$(abs_path "${PID_FILE}")"
ARCHIVE_ROOT="$(abs_path "${ARCHIVE_ROOT}")"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${ARCHIVE_ID}"

die() {
  echo "clean_refused=$*" >&2
  exit 1
}

guard_repo_child() {
  local path="$1"
  local name="$2"
  [[ "${path}" == "${REPO_ROOT}/"* ]] || die "${name}_outside_repo:${path}"
  [[ "${path}" != "${REPO_ROOT}" ]] || die "${name}_is_repo_root"
}

guard_output_dir() {
  guard_repo_child "${OUTPUT_DIR}" "output_dir"
  local rel="${OUTPUT_DIR#${REPO_ROOT}/}"
  [[ "${rel}" == nrp_results* ]] || die "output_dir_must_start_with_nrp_results:${OUTPUT_DIR}"
  [[ "${rel}" != dataset* ]] || die "output_dir_points_at_dataset:${OUTPUT_DIR}"
  [[ "${rel}" != datasets* ]] || die "output_dir_points_at_datasets:${OUTPUT_DIR}"
  [[ "${rel}" != nrp_calibration_pack* ]] || die "output_dir_points_at_pack:${OUTPUT_DIR}"
}

guard_record_paths() {
  guard_repo_child "${RECORD_DIR}" "record_dir"
  guard_repo_child "${PID_FILE}" "pid_file"
  guard_repo_child "${ARCHIVE_ROOT}" "archive_root"
  [[ "${RECORD_DIR}" == "${REPO_ROOT}/record" || "${RECORD_DIR}" == "${REPO_ROOT}/record/"* ]] || die "record_dir_must_be_under_record:${RECORD_DIR}"
  [[ "${PID_FILE}" == "${RECORD_DIR}/"* ]] || die "pid_file_must_be_under_record_dir:${PID_FILE}"
  [[ "${ARCHIVE_ROOT}" == "${RECORD_DIR}/"* ]] || die "archive_root_must_be_under_record_dir:${ARCHIVE_ROOT}"
}

find_labeler_pids() {
  local pid cwd cmd
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${cmd}" == *"python"* && "${cmd}" == *"${REPO_ROOT}/nrp_calibration_pack/profile/run_profile.py"* ]] || {
      [[ "${cmd}" == *"python"* && "${cmd}" == *"nrp_calibration_pack/profile/run_profile.py"* && "${cwd}" == "${REPO_ROOT}" ]]
    }; then
      printf '%s\n' "${pid}"
    fi
  done < <(pgrep -f "nrp_calibration_pack/profile/run_profile.py" || true)
}

stop_labelers() {
  mapfile -t pids < <(find_labeler_pids)
  if (( ${#pids[@]} == 0 )); then
    echo "stop_verifier=no_existing_repo_local_labeler"
    return 0
  fi
  echo "stopping_existing_labelers=${pids[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry_run_signal=SIGINT ${pids[*]}"
    return 0
  fi
  kill -INT "${pids[@]}" 2>/dev/null || true
  sleep 10
  mapfile -t remaining < <(find_labeler_pids)
  if (( ${#remaining[@]} > 0 )); then
    echo "sigint_remaining=${remaining[*]}"
    kill -TERM "${remaining[@]}" 2>/dev/null || true
    sleep 5
  fi
  mapfile -t remaining < <(find_labeler_pids)
  if (( ${#remaining[@]} > 0 )); then
    echo "stop_verifier=failed remaining=${remaining[*]}" >&2
    return 1
  fi
  echo "stop_verifier=ok"
}

labeling_logs() {
  find "${RECORD_DIR}" -maxdepth 1 -type f \( \
    -name 'local_5090_balanced_labeling*.log' -o \
    -name 'local_5090_balanced_labeling.pid' \
  \) -print 2>/dev/null || true
}

archive_or_delete() {
  local actions=()
  if [[ -e "${OUTPUT_DIR}" ]]; then
    actions+=("${OUTPUT_DIR}")
  fi
  if [[ -f "${PID_FILE}" ]]; then
    actions+=("${PID_FILE}")
  fi
  while read -r path; do
    [[ -n "${path}" ]] || continue
    [[ "${path}" != "${PID_FILE}" ]] || continue
    actions+=("${path}")
  done < <(labeling_logs)

  if (( ${#actions[@]} == 0 )); then
    echo "cleanup_targets=none"
    return 0
  fi

  printf 'cleanup_targets=%s\n' "${actions[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${DELETE}" == "1" ]]; then
      echo "dry_run_delete_targets=${actions[*]}"
    else
      echo "dry_run_archive_dir=${ARCHIVE_DIR}"
    fi
    return 0
  fi

  if [[ "${DELETE}" == "1" ]]; then
    for path in "${actions[@]}"; do
      if [[ "${path}" == "${OUTPUT_DIR}" ]]; then
        rm -rf -- "${path}"
      else
        rm -f -- "${path}"
      fi
    done
    echo "cleanup_mode=delete"
    return 0
  fi

  mkdir -p "${ARCHIVE_DIR}"
  for path in "${actions[@]}"; do
    [[ -e "${path}" ]] || continue
    mv -- "${path}" "${ARCHIVE_DIR}/$(basename "${path}")"
  done
  python - "${ARCHIVE_DIR}" "${OUTPUT_DIR}" "${PID_FILE}" <<'PY'
import json
import sys
from pathlib import Path

archive = Path(sys.argv[1])
payload = {
    "archive_dir": str(archive),
    "original_output_dir": sys.argv[2],
    "original_pid_file": sys.argv[3],
    "archived_entries": sorted(path.name for path in archive.iterdir() if path.name != "manifest.json"),
}
(archive / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("archive_manifest=" + str(archive / "manifest.json"))
PY
  echo "cleanup_mode=archive archive_dir=${ARCHIVE_DIR}"
}

verify_clean() {
  mapfile -t remaining < <(find_labeler_pids)
  if (( ${#remaining[@]} > 0 )); then
    echo "final_process_verifier=failed remaining=${remaining[*]}" >&2
    return 1
  fi
  echo "final_process_verifier=ok"
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -e "${OUTPUT_DIR}" ]]; then
      echo "output_dir_verifier=dry_run_kept ${OUTPUT_DIR}"
    else
      echo "output_dir_verifier=dry_run_already_clean ${OUTPUT_DIR}"
    fi
    return 0
  fi
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "output_dir_verifier=exists ${OUTPUT_DIR}" >&2
    return 1
  fi
  echo "output_dir_verifier=clean ${OUTPUT_DIR}"
  if command -v nvidia-smi >/dev/null; then
    echo "gpu_status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    echo "gpu_processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fi
}

guard_output_dir
guard_record_paths

echo "repo_root=${REPO_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "record_dir=${RECORD_DIR}"
echo "pid_file=${PID_FILE}"
echo "archive_dir=${ARCHIVE_DIR}"
echo "dry_run=${DRY_RUN}"
echo "delete=${DELETE}"

stop_labelers
archive_or_delete
verify_clean
