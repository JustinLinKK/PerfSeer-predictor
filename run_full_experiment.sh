#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the full optimized PerfSeer experiment flow.

Default behavior:
  - trains accuracy-track configs
  - evaluates accuracy on GPU
  - evaluates CPU fp32 for every accuracy candidate
  - selects the best accuracy candidate by GPU mean MAPE
  - trains CPU-deployment model configs
  - trains distilled students using accuracy_topo_features as teacher
  - runs the CPU runtime matrix on deployment candidates and the accuracy champion
  - writes summaries and PNG plots

Usage:
  bash run_full_experiment.sh [options]

Options:
  --name NAME           experiment name under runs/experiments/NAME
  --data-root PATH      dataset root, default: dataset
  --conda-env NAME      conda environment, default: perfseer
  --epochs N           override train epochs for all configs
  --batch-size N       override train batch size for all configs
  --limit N            use a limited dataset for smoke/debug runs
  --bench-graphs N     CPU benchmark graph count, default: 1000
  --cpu-threads N      CPU inference threads, default: profile/runtime default
  --smoke              shortcut for --limit 200 --epochs 2 --bench-graphs 100
  --skip-train         reuse checkpoints in this experiment directory
  --skip-distill       skip distill_student_192 and distill_student_128
  --help               show this help

Good unattended command:
  nohup bash run_full_experiment.sh --name full_$(date +%Y%m%d_%H%M%S) \
    > full_experiment.nohup.log 2>&1 &
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-perfseer}"
DATA_ROOT="${DATA_ROOT:-dataset}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-perfseer_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
LIMIT="${LIMIT:-}"
BENCH_GRAPHS="${BENCH_GRAPHS:-1000}"
CPU_THREADS="${CPU_THREADS:-}"
SKIP_TRAIN=0
SKIP_DISTILL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --bench-graphs)
      BENCH_GRAPHS="$2"
      shift 2
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --smoke)
      LIMIT="${LIMIT:-200}"
      EPOCHS="${EPOCHS:-2}"
      BENCH_GRAPHS="100"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-distill)
      SKIP_DISTILL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RUN_DIR="runs/experiments/${EXPERIMENT_NAME}"
CONFIG_DIR="${RUN_DIR}/configs"
CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
PLOT_DIR="${RUN_DIR}/plots"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_PATH="${RUN_DIR}/results.jsonl"
SUMMARY_PATH="${RUN_DIR}/summary.txt"

mkdir -p "$CONFIG_DIR" "$CHECKPOINT_ROOT" "$PLOT_DIR" "$LOG_DIR"
touch "$RESULTS_PATH"
exec > >(tee -a "${RUN_DIR}/full_flow.log") 2>&1

echo "PerfSeer full experiment"
echo "started: $(date -Is)"
echo "experiment: ${EXPERIMENT_NAME}"
echo "run_dir: ${RUN_DIR}"
echo "data_root: ${DATA_ROOT}"
echo "conda_env: ${CONDA_ENV}"
echo "epochs_override: ${EPOCHS:-<config default>}"
echo "batch_size_override: ${TRAIN_BATCH_SIZE:-<config default>}"
echo "limit_override: ${LIMIT:-<full dataset>}"
echo "bench_graphs: ${BENCH_GRAPHS}"
echo

py() {
  conda run -n "$CONDA_ENV" python "$@"
}

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "+ $*"
  "$@"
}

materialize_config() {
  local src="$1"
  local dst="$2"
  local teacher_dir="${3:-}"
  local cmd=(
    scripts/materialize_experiment_config.py
    --src "$src"
    --dst "$dst"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --results-path "$RESULTS_PATH"
    --data-root "$DATA_ROOT"
  )
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ -n "$EPOCHS" ]]; then
    cmd+=(--epochs "$EPOCHS")
  fi
  if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
    cmd+=(--batch-size "$TRAIN_BATCH_SIZE")
  fi
  if [[ -n "$teacher_dir" ]]; then
    cmd+=(--teacher-dir "$teacher_dir")
  fi
  py "${cmd[@]}"
}

ckpt_ready() {
  local dir="$1"
  if [[ -f "$dir/seernet_multi.pt" ]]; then
    return 0
  fi
  shopt -s nullglob
  local metric_files=("$dir"/seernet_metric*.pt)
  shopt -u nullglob
  ((${#metric_files[@]} == 6))
}

train_config() {
  local config="$1"
  local run_id="$2"
  local ckpt_dir="${CHECKPOINT_ROOT}/${run_id}"
  if [[ "$SKIP_TRAIN" == "1" && -d "$ckpt_dir" ]] && ckpt_ready "$ckpt_dir"; then
    echo "skip training ${run_id}; checkpoints already exist"
    return 0
  fi
  run_step "train ${run_id}" py -m perfseer_optimized.train --config "$config"
}

eval_gpu_accuracy() {
  local run_id="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/${run_id}"
  if ! ckpt_ready "$ckpt_dir"; then
    echo "skip GPU eval ${run_id}; checkpoint not found at ${ckpt_dir}"
    return 0
  fi
  run_step "GPU accuracy eval ${run_id}" \
    py -m perfseer_optimized.eval \
      --eval-profile src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml \
      --ckpt-dir "$ckpt_dir" \
      --data-root "$DATA_ROOT" \
      --results-path "$RESULTS_PATH"
}

eval_cpu_fp32() {
  local run_id="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/${run_id}"
  if ! ckpt_ready "$ckpt_dir"; then
    echo "skip CPU fp32 eval ${run_id}; checkpoint not found at ${ckpt_dir}"
    return 0
  fi
  local cmd=(
    py scripts/run_deploy_matrix.py
    --ckpt-dir "$ckpt_dir"
    --data-root "$DATA_ROOT"
    --results-path "$RESULTS_PATH"
    --batch-size 1
    --num-bench-graphs "$BENCH_GRAPHS"
    --profiles src/perfseer-optimized/configs/eval_profiles/cpu_pytorch_fp32.yaml
  )
  if [[ -n "$CPU_THREADS" ]]; then
    cmd+=(--cpu-threads "$CPU_THREADS")
  fi
  run_step "CPU fp32 eval ${run_id}" "${cmd[@]}"
}

run_deploy_matrix() {
  local run_id="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/${run_id}"
  if ! ckpt_ready "$ckpt_dir"; then
    echo "skip deploy matrix ${run_id}; checkpoint not found at ${ckpt_dir}"
    return 0
  fi
  local cmd=(
    py scripts/run_deploy_matrix.py
    --ckpt-dir "$ckpt_dir"
    --data-root "$DATA_ROOT"
    --results-path "$RESULTS_PATH"
    --batch-size 1
    --num-bench-graphs "$BENCH_GRAPHS"
  )
  if [[ -n "$CPU_THREADS" ]]; then
    cmd+=(--cpu-threads "$CPU_THREADS")
  fi
  run_step "CPU deployment matrix ${run_id}" "${cmd[@]}"
}

best_accuracy_run_id() {
  py scripts/select_best_accuracy.py --results "$RESULTS_PATH"
}

ACCURACY_CONFIGS=(
  "src/perfseer-optimized/configs/train_accuracy/baseline.yaml"
  "src/perfseer-optimized/configs/train_accuracy/reg_layernorm_adamw.yaml"
  "src/perfseer-optimized/configs/train_accuracy/gated_2block_256.yaml"
  "src/perfseer-optimized/configs/train_accuracy/topo_features.yaml"
)

DEPLOY_CONFIGS=(
  "src/perfseer-optimized/configs/train_deploy_model/shared_multitask_256_pcgrad.yaml"
  "src/perfseer-optimized/configs/train_deploy_model/shared_multitask_192_pcgrad.yaml"
)

DISTILL_CONFIGS=(
  "src/perfseer-optimized/configs/train_deploy_model/distill_student_192.yaml"
  "src/perfseer-optimized/configs/train_deploy_model/distill_student_128.yaml"
)

ALL_ACCURACY_RUNS=()
ALL_DEPLOY_RUNS=()

run_step "editable install" py -m pip install -e .

for src in "${ACCURACY_CONFIGS[@]}"; do
  dst="${CONFIG_DIR}/train_accuracy/$(basename "$src")"
  run_id="$(materialize_config "$src" "$dst")"
  ALL_ACCURACY_RUNS+=("$run_id")
  train_config "$dst" "$run_id"
  eval_gpu_accuracy "$run_id"
  eval_cpu_fp32 "$run_id"
done

BEST_ACCURACY="$(best_accuracy_run_id)"
echo
echo "accuracy champion by GPU mean MAPE: ${BEST_ACCURACY}"

for src in "${DEPLOY_CONFIGS[@]}"; do
  dst="${CONFIG_DIR}/train_deploy_model/$(basename "$src")"
  run_id="$(materialize_config "$src" "$dst")"
  ALL_DEPLOY_RUNS+=("$run_id")
  train_config "$dst" "$run_id"
  eval_gpu_accuracy "$run_id"
  run_deploy_matrix "$run_id"
done

if [[ "$SKIP_DISTILL" == "0" ]]; then
  TEACHER_DIR="${CHECKPOINT_ROOT}/accuracy_topo_features"
  if ! ckpt_ready "$TEACHER_DIR"; then
    echo "distillation teacher missing at ${TEACHER_DIR}; skipping distilled students"
  else
    for src in "${DISTILL_CONFIGS[@]}"; do
      dst="${CONFIG_DIR}/train_deploy_model/$(basename "$src")"
      run_id="$(materialize_config "$src" "$dst" "$TEACHER_DIR")"
      ALL_DEPLOY_RUNS+=("$run_id")
      train_config "$dst" "$run_id"
      eval_gpu_accuracy "$run_id"
      run_deploy_matrix "$run_id"
    done
  fi
else
  echo "distilled students skipped by --skip-distill"
fi

run_deploy_matrix "$BEST_ACCURACY"

echo
echo "===== summarize ledger ====="
py scripts/summarize_results.py --results "$RESULTS_PATH" | tee "$SUMMARY_PATH"

echo
echo "===== plot ledger ====="
py scripts/plot_results.py --results "$RESULTS_PATH" --out-dir "$PLOT_DIR" --title "$EXPERIMENT_NAME"

echo
echo "finished: $(date -Is)"
echo "results ledger: ${RESULTS_PATH}"
echo "summary table: ${SUMMARY_PATH}"
echo "plots:"
find "$PLOT_DIR" -maxdepth 1 -type f | sort
