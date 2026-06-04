#!/bin/bash
#
# TTA evaluation runner — IND forward order (B→R→N→E→T→C)
# Mirror của scripts/test_classIL.sh
#
#SBATCH --job-name=test_tta
#SBATCH --output=logs/test_tta_%j.out
#SBATCH --error=logs/test_tta_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-logs}"

# Config — IND forward, num_workers=0
CONFIG="${CONFIG:-configs/default_tta_eval_num_workers0.yaml}"

# Checkpoint paths
FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
MERGED_DIR="${MERGED_DIR:-./checkpoints/merged}"
SWAG_DIR="${SWAG_DIR:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA/checkpoints/swag_diagonal}"

# Python binary
if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

# Ensure local hot storage directories
mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints; do
    repo_path="$PROJECT_ROOT/$name"
    local_path="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$repo_path" ]; then
        :
    elif [ -e "$repo_path" ]; then
        echo "[WARN] $repo_path is not a symlink; hot writes should use $local_path"
    else
        ln -s "$local_path" "$repo_path"
    fi
done

mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] start at $(date)"
echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] SWAG_DIR=$SWAG_DIR"

# ── Helper: check log không bị giữ bởi tiến trình khác ──────────────────────
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
        state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path held by PID $pid state=$state" >&2; return 1
    done
}

run_to_logs() {
    local result_log="$1" error_log="$2"; shift 2
    echo "[INFO] running: $*"
    check_log_not_held "$result_log"; check_log_not_held "$error_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$result_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$error_log"
    "$@" >> "$result_log" 2>> "$error_log"
}

# ── CLASS-IL TCP (chính) ──────────────────────────────────────────────────────
run_to_logs \
    "$LOG_DIR/test_new_run/result_tta_classil_tcp.log" \
    "$LOG_DIR/test_new_run/error_tta_classil_tcp.log" \
    "$PYTHON_BIN" -u test_tta.py \
        --config        "$CONFIG" \
        --save_dir      "$FINETUNED_DIR" \
        --merge_model_path "$MERGED_DIR" \
        --swag_dir      "$SWAG_DIR" \
        --mode          classil_tcp \
        --result_csv    "$LOG_DIR/tta_results_classil_tcp.csv" \
        --tta_stats_csv "$LOG_DIR/tta_stats_classil_tcp.csv"

# ── CLASS-IL Naive ────────────────────────────────────────────────────────────
run_to_logs \
    "$LOG_DIR/test_new_run/result_tta_classil_naive.log" \
    "$LOG_DIR/test_new_run/error_tta_classil_naive.log" \
    "$PYTHON_BIN" -u test_tta.py \
        --config        "$CONFIG" \
        --save_dir      "$FINETUNED_DIR" \
        --merge_model_path "$MERGED_DIR" \
        --swag_dir      "$SWAG_DIR" \
        --mode          classil_naive \
        --result_csv    "$LOG_DIR/tta_results_classil_naive.csv"

# ── TASK-IL ───────────────────────────────────────────────────────────────────
run_to_logs \
    "$LOG_DIR/test_taskIL/result_tta_taskil.log" \
    "$LOG_DIR/test_taskIL/error_tta_taskil.log" \
    "$PYTHON_BIN" -u test_tta.py \
        --config        "$CONFIG" \
        --save_dir      "$FINETUNED_DIR" \
        --merge_model_path "$MERGED_DIR" \
        --swag_dir      "$SWAG_DIR" \
        --mode          taskil \
        --result_csv    "$LOG_DIR/tta_results_taskil.csv"

echo "[INFO] finished at $(date)"
