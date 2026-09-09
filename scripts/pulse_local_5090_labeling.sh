#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
PACK_DIR="${PACK_DIR:-${REPO_ROOT}/nrp_calibration_pack}"
WORKLOAD_DIR="${WORKLOAD_DIR:-${PACK_DIR}/workload_specs_balanced_local}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/nrp_results_rtx5090_balanced_local}"
RECORD_DIR="${RECORD_DIR:-${REPO_ROOT}/record}"
PID_FILE="${PID_FILE:-${RECORD_DIR}/local_5090_balanced_labeling.pid}"
LOG_FILE="${LOG_FILE:-}"

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

latest_log() {
  if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
    printf '%s\n' "${LOG_FILE}"
    return
  fi
  find "${RECORD_DIR}" -maxdepth 1 -type f -name 'local_5090_balanced_labeling_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 {print $2}'
}

echo "repo_root=${REPO_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "workload_dir=${WORKLOAD_DIR}"
echo "pid_file=${PID_FILE}"

active_pid=""
if [[ -f "${PID_FILE}" ]]; then
  pid_from_file="$(tr -d '[:space:]' <"${PID_FILE}" || true)"
  if [[ -n "${pid_from_file}" ]] && kill -0 "${pid_from_file}" 2>/dev/null; then
    active_pid="${pid_from_file}"
  fi
fi
if [[ -z "${active_pid}" ]]; then
  mapfile -t pids < <(find_labeler_pids)
  if (( ${#pids[@]} > 0 )); then
    active_pid="${pids[0]}"
  fi
fi

if [[ -n "${active_pid}" ]]; then
  echo "process_status=running pid=${active_pid}"
  ps -p "${active_pid}" -o pid,ppid,stat,etime,cmd || true
else
  echo "process_status=not_running"
fi

if command -v nvidia-smi >/dev/null; then
  echo "gpu_status:"
  nvidia-smi --query-gpu=name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  echo "gpu_processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
else
  echo "gpu_status=nvidia-smi_not_found"
fi

if [[ -f "${WORKLOAD_DIR}/summary.json" ]]; then
  echo "workload_summary:"
  "${PYTHON}" - "${WORKLOAD_DIR}/summary.json" <<'PY'
import json, sys
summary = json.loads(open(sys.argv[1]).read())
print(json.dumps({
    "workloads": summary.get("workloads"),
    "dataset_ids": summary.get("dataset_ids", []),
    "subset_ids": summary.get("subset_ids", []),
    "precisions": summary.get("precisions", []),
    "skipped": summary.get("skipped", {}),
}, sort_keys=True))
PY
else
  echo "workload_summary=missing"
fi

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import collections, json, sys
from pathlib import Path
out = Path(sys.argv[1])
results = sorted(out.glob("results_shard*.jsonl"))
labels = list((out / "label" / "label").glob("*.txt")) if (out / "label" / "label").is_dir() else []
label_v3 = sorted(out.glob("label_v3_shard*.jsonl"))
scheduler_resource = sorted(out.glob("scheduler_resource_shard*.jsonl"))
status = collections.Counter()
rows = 0
for path in results:
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            status[json.loads(line).get("status", "unknown")] += 1
        except json.JSONDecodeError:
            status["malformed"] += 1
v3_rows = sum(1 for path in label_v3 for line in path.read_text(errors="ignore").splitlines() if line.strip())
resource_rows = sum(1 for path in scheduler_resource for line in path.read_text(errors="ignore").splitlines() if line.strip())
print("result_summary=" + json.dumps({
    "results_files": [str(path) for path in results],
    "result_rows": rows,
    "status_counts": dict(status),
    "label_files": len(labels),
    "label_v3_files": [str(path) for path in label_v3],
    "label_v3_rows": v3_rows,
    "scheduler_resource_files": [str(path) for path in scheduler_resource],
    "scheduler_resource_rows": resource_rows,
}, sort_keys=True))
for path in sorted(out.glob("hardware_shard*.json")):
    data = json.loads(path.read_text())
    probe = data.get("precision_auto_probe") or {}
    supported = sorted(name for name, meta in probe.items() if isinstance(meta, dict) and meta.get("supported"))
    print("precision_auto_probe=" + json.dumps({"file": str(path), "supported": supported}, sort_keys=True))
PY

log_path="$(latest_log || true)"
if [[ -n "${log_path}" && -f "${log_path}" ]]; then
  echo "log_file=${log_path}"
  echo "log_tail:"
  tail -80 "${log_path}" || true
else
  echo "log_file=missing"
fi

echo "resume_command=PID_FILE='${PID_FILE}' OUTPUT_DIR='${OUTPUT_DIR}' WORKLOAD_DIR='${WORKLOAD_DIR}' scripts/resume_local_5090_labeling.sh"
