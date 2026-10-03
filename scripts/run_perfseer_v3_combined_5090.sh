#!/usr/bin/env bash
set -euo pipefail

launcher_repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
launcher_python=${PERFSEER_PYTHON:-/home/justin/miniconda3/envs/model-preflight/bin/python}
launcher_output=${1:-"$launcher_repository/record/perfseer-v3-combined-local-5090-gated-$(date -u +%Y%m%dT%H%M%SZ)"}

if [[ ! -x $launcher_python ]]; then
  echo "PerfSeer Python is not executable: $launcher_python" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

"$launcher_python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit("local launcher requires exactly one visible GPU")
gpu = torch.cuda.get_device_name(0)
if "5090" not in gpu:
    raise SystemExit(f"expected an RTX 5090, found {gpu!r}")
print(f"verified_gpu={gpu}")
PY

mkdir -p "$launcher_output"
launcher_command=(
  "$launcher_python"
  "$launcher_repository/scripts/train_perfseer_v3_combined_dev.py"
  --dataset-root "$launcher_repository/src/perfseer_v3/dataset_with_label/ready_for_train"
  --output-dir "$launcher_output"
  --epochs 200
  --distillation-epochs 100
  --train-samples 569
  --validation-samples 65
  --batch-size 8
  --max-capture-microbatch 8
  --capacity T1
  --device cuda
  --amp bfloat16
  --learning-rate 0.0005
  --student-learning-rate 0.001
  --weight-decay 0.00001
  --teacher-quality-gate 0.90
  --target-accuracy 0.98
  --hard-label-weight 0.6
  --representation-distillation-weight 0.05
)
if [[ -f $launcher_output/materialized-development-samples.pt ]]; then
  launcher_command+=(--reuse-materialized)
fi

printf 'output_dir=%s\ncommand=' "$launcher_output"
printf ' %q' "${launcher_command[@]}"
printf '\n'
if [[ ${PERFSEER_DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

set +e
"${launcher_command[@]}" 2>&1 | tee "$launcher_output/training.log"
launcher_status=${PIPESTATUS[0]}
set -e

if [[ -f $launcher_output/training-report.json ]]; then
  "$launcher_python" - "$launcher_output" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
report = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
teacher = output / "teacher-development-checkpoint.pt"
student = output / "student-development-checkpoint.pt"
if not teacher.is_file():
    raise SystemExit("teacher checkpoint is missing")
if bool(report["distillation_started"]) != student.is_file():
    raise SystemExit("student checkpoint does not match the distillation report")
print(
    "verified_report="
    f"status={report['status']} "
    f"teacher_accuracy={report['teacher_quality_gate']['observed_accuracy']:.6f} "
    f"distillation_started={report['distillation_started']}"
)
PY
fi

exit "$launcher_status"
