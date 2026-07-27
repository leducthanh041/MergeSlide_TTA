#!/bin/bash
#
# TTA-Core evaluation runner — IND forward order (B->R->N->E->T->C)
# KHÔNG TCP, KHÔNG L_task, CÓ Module F (consensus-based sample selection)
# Mirror của scripts/test_tta_v3.sh nhưng dùng test_tta_core.py
#
#SBATCH --job-name=test_tta_core
#SBATCH --output=logs/test_tta_core_%j.out
#SBATCH --error=logs/test_tta_core_%j.err
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
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"

# Config — IND forward, num_workers=0
CONFIG="${CONFIG:-configs/default_tta_core_eval_num_workers0.yaml}"

# Checkpoint paths
MERGED_DIR="${MERGED_DIR:-./checkpoints/merged}"
SWAG_DIR="${SWAG_DIR:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA/checkpoints/swag_diagonal}"
RESET_PER_SLIDE="${RESET_PER_SLIDE:-0}"
FISHER_OMEGA_DIR="${FISHER_OMEGA_DIR:-}"     # chỉ cần nếu tta.use_fisher_reg=true trong CONFIG
TTA_ABLATION_ARGS=()

append_bool_flag() {
    local value="${1,,}" flag="$2" name="$3"
    case "$value" in
        1|true|yes|y|on) TTA_ABLATION_ARGS+=("$flag") ;;
        0|false|no|n|off) ;;
        *)
            echo "[ERROR] $name must be one of: 1/0, true/false, yes/no, on/off" >&2
            exit 1
            ;;
    esac
}

append_bool_flag "$RESET_PER_SLIDE" --reset_per_slide RESET_PER_SLIDE

if [ -n "$FISHER_OMEGA_DIR" ]; then
    TTA_ABLATION_ARGS+=(--fisher_omega_dir "$FISHER_OMEGA_DIR")
fi

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

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi
if ! [[ "$FOLD_START" =~ ^[0-9]+$ && "$FOLD_END" =~ ^[0-9]+$ ]] \
   || [ "$FOLD_START" -ge "$FOLD_END" ]; then
    echo "[ERROR] Invalid fold range: FOLD_START=$FOLD_START FOLD_END=$FOLD_END" >&2
    exit 1
fi

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

mkdir -p "$LOG_DIR" "$LOG_DIR/test_tta_core"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] start at $(date)"
echo "[INFO] MergeSlide-TTA-Core (no TCP, no L_task) — IND forward"
echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] MERGED_DIR=$MERGED_DIR"
echo "[INFO] SWAG_DIR=$SWAG_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] FOLDS=[$FOLD_START,$FOLD_END)"
echo "[INFO] RESET_PER_SLIDE=$RESET_PER_SLIDE"
echo "[INFO] FISHER_OMEGA_DIR=${FISHER_OMEGA_DIR:-<not set>}"

for ((fold = FOLD_START; fold < FOLD_END; fold++)); do
    merged_path="$MERGED_DIR/fold_${fold}/merged_final.pth"
    swag_path="$SWAG_DIR/fold_${fold}.pt"
    if [ ! -f "$merged_path" ]; then
        echo "[ERROR] Missing merged checkpoint: $merged_path" >&2
        exit 1
    fi
    if [ ! -f "$swag_path" ]; then
        echo "[ERROR] Missing SWAG posterior: $swag_path" >&2
        exit 1
    fi
done

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

run_to_logs \
    "$LOG_DIR/test_tta_core/result_tta_core_ind.log" \
    "$LOG_DIR/test_tta_core/error_tta_core_ind.log" \
    "$PYTHON_BIN" -u tools/run_classil_with_pt_features.py \
        --entrypoint    test_tta_core.py \
        --config        "$CONFIG" \
        --merge_model_path "$MERGED_DIR" \
        --swag_dir      "$SWAG_DIR" \
        --result_csv    "$LOG_DIR/tta_core_results_ind.csv" \
        --tta_stats_csv "$LOG_DIR/tta_core_stats_ind.csv" \
        --efficiency_json "$LOG_DIR/efficiency_tta_core_ind.json" \
        --fold_start "$FOLD_START" \
        --fold_end "$FOLD_END" \
        "${TTA_ABLATION_ARGS[@]}"

echo "[INFO] finished at $(date)"
