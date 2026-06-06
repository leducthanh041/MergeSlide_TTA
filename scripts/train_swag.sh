#!/bin/bash
#
# SWAG posterior estimation runner — chạy cho cả 3 settings:
#   1. IND forward   (B→R→N→E→T→C)
#   2. OOD forward   (B→R→N→E→T→C, cross-site splits)
#   3. IND reverse   (C→T→E→N→R→B)
#
# Sử dụng:
#   bash scripts/train_swag.sh                          # chạy cả 3 settings
#   SETTING=ind bash scripts/train_swag.sh              # chỉ IND forward
#   SETTING=ood bash scripts/train_swag.sh              # chỉ OOD forward
#   SETTING=reverse bash scripts/train_swag.sh          # chỉ IND reverse
#   FOLD_START=0 FOLD_END=5 bash scripts/train_swag.sh  # chỉ fold 0–4
#
#SBATCH --job-name=train_swag
#SBATCH --output=logs/train_swag_%j.out
#SBATCH --error=logs/train_swag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-logs}"

# ── Python binary ─────────────────────────────────────────────────────────────
if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

# ── Fold range ────────────────────────────────────────────────────────────────
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"

# ── Setting selector (ind | ood | reverse | all) ──────────────────────────────
SETTING="${SETTING:-all}"

cd "$PROJECT_ROOT"

# ── Local hot storage setup ───────────────────────────────────────────────────
mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints checkpoints_ood; do
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

mkdir -p "$LOG_DIR/train_swag"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# ── Log helpers ───────────────────────────────────────────────────────────────
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

# ── Runner function ───────────────────────────────────────────────────────────
run_swag() {
    local label="$1"        # tên setting, dùng cho log
    local config="$2"       # path tới yaml config
    local merged_dir="$3"   # path tới merged checkpoints
    local swag_dir="$4"     # output dir cho SWAG stats

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "[INFO] SWAG — setting: $label"
    echo "[INFO] config:      $config"
    echo "[INFO] merged_dir:  $merged_dir"
    echo "[INFO] swag_dir:    $swag_dir"
    echo "[INFO] folds:       $FOLD_START → $FOLD_END"
    echo "════════════════════════════════════════════════════════"

    run_to_logs \
        "$LOG_DIR/train_swag/result_swag_${label}.log" \
        "$LOG_DIR/train_swag/error_swag_${label}.log" \
        "$PYTHON_BIN" -u tools/run_classil_with_pt_features.py \
            --entrypoint         train_swag.py \
            --config             "$config" \
            --merged_checkpoints "$merged_dir" \
            --swag_dir           "$swag_dir" \
            --fold_start         "$FOLD_START" \
            --fold_end           "$FOLD_END"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "[INFO] start at $(date)"
echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] SETTING=$SETTING"
echo "[INFO] FOLD_START=$FOLD_START  FOLD_END=$FOLD_END"

case "$SETTING" in

    ind|all)
        run_swag \
            "ind_forward" \
            "configs/default_tta_eval_num_workers0.yaml" \
            "./checkpoints/merged" \
            "./checkpoints/swag_diagonal"
        ;;& # fallthrough nếu SETTING=all

    ood|all)
        run_swag \
            "ood_forward" \
            "configs/default_tta_ood_eval_num_workers0.yaml" \
            "./checkpoints_ood/merged" \
            "./checkpoints_ood/swag_diagonal"
        ;;&

    reverse|all)
        run_swag \
            "ind_reverse" \
            "configs/default_tta_reverse_eval_num_workers0.yaml" \
            "./checkpoints/merged_reverse" \
            "./checkpoints/swag_diagonal_reverse"
        ;;

    *)
        echo "[ERROR] SETTING='$SETTING' không hợp lệ. Dùng: ind | ood | reverse | all" >&2
        exit 1
        ;;
esac

echo ""
echo "[INFO] SWAG training finished at $(date)"
