#!/bin/bash
#
# tune_tta.sh — Runner cho TTA hyperparameter tuning
#
# Sử dụng:
#   SETTING=ind bash scripts/tune_tta.sh   # chỉ IND
#   SETTING=ood bash scripts/tune_tta.sh   # chỉ OOD
#   N_TRIALS=50 bash scripts/tune_tta.sh   # 50 trials
#
#SBATCH --job-name=tune_tta
#SBATCH --output=logs/tune_tta_%j.out
#SBATCH --error=logs/tune_tta_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=120:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-$MERGESLIDE_LOCAL_ROOT/logs/tune_tta_runs}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    [ -x "$DEFAULT_PYTHON" ] && PYTHON_BIN="$DEFAULT_PYTHON" || PYTHON_BIN="python"
fi

N_TRIALS="${N_TRIALS:-30}"
SETTING="${SETTING:-ind}"
SEED="${SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-10}"
ENTRYPOINT_WRAPPER="${ENTRYPOINT_WRAPPER:-tools/run_classil_with_pt_features.py}"
GPU_A="${GPU_A:-4}"
GPU_B="${GPU_B:-7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
SINGLE_GPU="${SINGLE_GPU:-0}"

cd "$PROJECT_ROOT"

mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints checkpoints_ood; do
    rp="$PROJECT_ROOT/$name"; lp="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$rp" ]; then
        :
    elif [ -e "$rp" ]; then
        echo "[WARN] $rp is not a symlink; hot writes should use $lp"
    else
        ln -s "$lp" "$rp"
    fi
done

mkdir -p "$LOG_DIR/tune_tta"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] start at $(date)"
echo "[INFO] SETTING=$SETTING  N_TRIALS=$N_TRIALS  SEED=$SEED"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] ENTRYPOINT_WRAPPER=$ENTRYPOINT_WRAPPER"
echo "[INFO] GPU_A=$GPU_A  GPU_B=$GPU_B"
echo "[INFO] WORKERS_PER_GPU=$WORKERS_PER_GPU"
echo "[INFO] SINGLE_GPU=$SINGLE_GPU"

if [ "$SETTING" = "ind" ]; then
    BASE_CONFIG="configs/default_tta_eval_num_workers0.yaml"
    MERGE_DIR="./checkpoints/merged"
    SWAG_DIR="$PROJECT_ROOT/checkpoints/swag_diagonal"
    FINETUNED_DIR="./checkpoints/finetuned"
OUTPUT_DIR="$LOG_DIR/tune_tta"
elif [ "$SETTING" = "ood" ]; then
    BASE_CONFIG="configs/default_tta_ood_eval_num_workers0.yaml"
    MERGE_DIR="./checkpoints_ood/merged"
    SWAG_DIR="$PROJECT_ROOT/checkpoints_ood/swag_diagonal"
    FINETUNED_DIR="./checkpoints_ood/finetuned"
    OUTPUT_DIR="$LOG_DIR/tune_tta_ood"
else
    echo "[ERROR] SETTING='$SETTING' không hợp lệ. Dùng: ind | ood" >&2
    exit 1
fi

echo "[INFO] base_config: $BASE_CONFIG"
echo "[INFO] merge_dir:   $MERGE_DIR"
echo "[INFO] swag_dir:    $SWAG_DIR"
echo "[INFO] output_dir:  $OUTPUT_DIR"

MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/manifest_${SETTING}_${N_TRIALS}_${SEED}.json}"
MANIFEST_CSV="${MANIFEST_CSV:-${MANIFEST_PATH%.json}.csv}"

manifest_contains_default() {
    local manifest_path="$1"
    local base_config_path="$2"
    "$PYTHON_BIN" -c '
import json, sys, yaml
manifest_path = sys.argv[1]
base_config_path = sys.argv[2]
with open(base_config_path, "r") as f:
    cfg = yaml.safe_load(f) or {}
defaults = {k: cfg.get("tta", {}).get(k) for k in ["eta_base", "delta", "tau_c", "tau_ood", "gamma_class"]}
with open(manifest_path, "r") as f:
    manifest = json.load(f)
for item in manifest:
    params = item.get("params", {})
    if all(params.get(k) == defaults.get(k) for k in defaults):
        sys.exit(0)
sys.exit(1)
' "$manifest_path" "$base_config_path"
}

check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    [ -z "$resolved_log" ] && return 0
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue
        pid="${fd#/proc/}"; pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path held by PID $pid" >&2
        return 1
    done
}

mkdir -p "$(dirname "$MANIFEST_PATH")"
check_log_not_held "$MANIFEST_PATH"
check_log_not_held "$MANIFEST_CSV"

if [ -f "$MANIFEST_PATH" ] && ! manifest_contains_default "$MANIFEST_PATH" "$BASE_CONFIG"; then
    echo "[INFO] manifest exists and is clean, reuse → $MANIFEST_PATH"
else
    if [ -f "$MANIFEST_PATH" ]; then
        echo "[WARN] manifest contains default-config sample(s); rebuilding → $MANIFEST_PATH"
    fi
    echo "[INFO] preparing manifest..."
    "$PYTHON_BIN" -u tune_tta.py \
        --n_trials      "$N_TRIALS" \
        --setting       "$SETTING" \
        --base_config   "$BASE_CONFIG" \
        --merge_dir     "$MERGE_DIR" \
        --swag_dir      "$SWAG_DIR" \
        --finetuned_dir "$FINETUNED_DIR" \
        --output_dir    "$OUTPUT_DIR" \
        --seed          "$SEED" \
        --num_folds     "$NUM_FOLDS" \
        --project_root  "$PROJECT_ROOT" \
        --python_bin    "$PYTHON_BIN" \
        --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
        --manifest_path "$MANIFEST_PATH" \
        --prepare_manifest
fi

run_worker() {
    local gpu_id="$1"
    local trial_start="$2"
    local trial_end="$3"
    local worker_tag="$4"
    local worker_output="$OUTPUT_DIR/$worker_tag"
    local worker_log_dir="$LOG_DIR/tune_tta/$worker_tag"
    local worker_log="$worker_log_dir/tune_${SETTING}.log"
    local worker_err="$worker_log_dir/tune_${SETTING}.err"

    mkdir -p "$worker_output" "$worker_log_dir"
    check_log_not_held "$worker_log"
    check_log_not_held "$worker_err"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "[INFO] start at $(date)"
        echo "[INFO] gpu=$gpu_id trial_start=$trial_start trial_end=$trial_end"
        echo "[INFO] manifest=$MANIFEST_PATH"
        echo "[INFO] command=$PYTHON_BIN -u tune_tta.py --n_trials $N_TRIALS --setting $SETTING --base_config $BASE_CONFIG --merge_dir $MERGE_DIR --swag_dir $SWAG_DIR --finetuned_dir $FINETUNED_DIR --output_dir $worker_output --seed $SEED --num_folds $NUM_FOLDS --project_root $PROJECT_ROOT --python_bin $PYTHON_BIN --entrypoint_wrapper $ENTRYPOINT_WRAPPER --manifest_path $MANIFEST_PATH --trial_start $trial_start --trial_end $trial_end"
        "$PYTHON_BIN" -u tune_tta.py \
            --n_trials      "$N_TRIALS" \
            --setting       "$SETTING" \
            --base_config   "$BASE_CONFIG" \
            --merge_dir     "$MERGE_DIR" \
            --swag_dir      "$SWAG_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir    "$worker_output" \
            --seed          "$SEED" \
            --num_folds     "$NUM_FOLDS" \
            --project_root  "$PROJECT_ROOT" \
            --python_bin    "$PYTHON_BIN" \
            --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
            --manifest_path "$MANIFEST_PATH" \
            --trial_start   "$trial_start" \
            --trial_end     "$trial_end"
    ) > "$worker_log" 2> "$worker_err" &
    RUN_WORKER_PID=$!
}

if [ "$SINGLE_GPU" = "1" ] || { [ "$GPU_A" = "$GPU_B" ] && [ "$WORKERS_PER_GPU" -eq 1 ]; }; then
    GPU_LIST=("$GPU_A")
    echo "[INFO] single-GPU mode enabled → GPU=$GPU_A"
else
    GPU_LIST=("$GPU_A" "$GPU_B")
fi
NUM_GPUS="${#GPU_LIST[@]}"
TOTAL_WORKERS=$(( NUM_GPUS * WORKERS_PER_GPU ))
if [ "$TOTAL_WORKERS" -le 0 ]; then
    echo "[ERROR] TOTAL_WORKERS must be > 0" >&2
    exit 1
fi
echo "[INFO] TOTAL_WORKERS=$TOTAL_WORKERS"

BASE_TRIALS=$(( N_TRIALS / TOTAL_WORKERS ))
REMAINDER=$(( N_TRIALS % TOTAL_WORKERS ))

declare -a PIDS=()
for ((worker_idx=0; worker_idx<TOTAL_WORKERS; worker_idx++)); do
    gpu_idx=$(( worker_idx % NUM_GPUS ))
    gpu_id="${GPU_LIST[$gpu_idx]}"
    extra=0
    if [ "$worker_idx" -lt "$REMAINDER" ]; then
        extra=1
    fi
    count=$(( BASE_TRIALS + extra ))
    if [ "$worker_idx" -lt "$REMAINDER" ]; then
        start=$(( worker_idx * (BASE_TRIALS + 1) ))
    else
        start=$(( REMAINDER * (BASE_TRIALS + 1) + (worker_idx - REMAINDER) * BASE_TRIALS ))
    fi
    end=$(( start + count ))
    if [ "$count" -le 0 ]; then
        echo "[INFO] skip worker_idx=$worker_idx gpu=$gpu_id (no trials)"
        continue
    fi
    worker_tag="gpu${gpu_id}_w${worker_idx}"
    echo "[INFO] launch worker_idx=$worker_idx gpu=$gpu_id trial_start=$start trial_end=$end tag=$worker_tag"
    run_worker "$gpu_id" "$start" "$end" "$worker_tag"
    PIDS+=("$RUN_WORKER_PID")
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "[INFO] building combined summary..."
"$PYTHON_BIN" -u tune_tta.py \
    --setting       "$SETTING" \
    --output_dir    "$OUTPUT_DIR" \
    --summarize_only

echo "[INFO] finished at $(date)"
echo "[INFO] worker logs under: $LOG_DIR/tune_tta/"
