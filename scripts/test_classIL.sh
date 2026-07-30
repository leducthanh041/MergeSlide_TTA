#!/bin/bash
#
# Baseline CLASS-IL TCP evaluation runner. This is the no-TTA counterpart of
# scripts/test_classIL_tta.sh and saves per-fold/per-task routing metrics to CSV.

#SBATCH --job-name=test_classIL
#SBATCH --output=logs/test_classIL_%j.out
#SBATCH --error=logs/test_classIL_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"

SETTING="${SETTING:-ind}"
ORDER="${ORDER:-forward}"
LOG_DIR="${LOG_DIR:-}"
MODE="${MODE:-tcp}"
CLASSIL_ENTRYPOINT="${CLASSIL_ENTRYPOINT:-tools/run_classil_with_pt_features.py}"
BASELINE_ENTRYPOINT="${BASELINE_ENTRYPOINT:-test_classIL_task_prompt.py}"

if [ -n "$LOG_DIR" ] && [[ "$LOG_DIR" != /* && "$LOG_DIR" != logs && "$LOG_DIR" != logs/* ]]; then
    LOG_DIR="logs/$LOG_DIR"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/classil_baseline/${SETTING}_${ORDER}_${MODE}"
fi
case "$MODE" in
    tcp|naive|all) ;;
    *) echo "[ERROR] Unsupported MODE=$MODE (expected tcp|naive|all)" >&2; exit 1 ;;
esac

case "${SETTING}_${ORDER}" in
    ood_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints_ood/merged}"
        ;;
    ind_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged}"
        ;;
    ind_reverse)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_reverse_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned_reverse}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged_reverse}"
        ;;
    ood_reverse)
        echo "[ERROR] OOD reverse is not configured. Use SETTING=ind ORDER=reverse." >&2
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported SETTING/ORDER: SETTING=$SETTING ORDER=$ORDER (expected ind|ood with forward, or ind with reverse)" >&2
        exit 1
        ;;
esac

BASELINE_RESULT_CSV="${BASELINE_RESULT_CSV:-}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

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

mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

entrypoint_path="$BASELINE_ENTRYPOINT"
if [[ "$entrypoint_path" != /* ]]; then
    entrypoint_path="$PROJECT_ROOT/$entrypoint_path"
fi

supports_arg() {
    local arg_name="$1"
    grep -q -- "$arg_name" "$entrypoint_path"
}

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] setting=$SETTING"
echo "[INFO] order=$ORDER"
echo "[INFO] mode=$MODE"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] config_forward=$CONFIG_FORWARD"
echo "[INFO] save_dir_forward=$SAVE_DIR_FORWARD"
echo "[INFO] merge_model_path_forward=$MERGE_MODEL_PATH_FORWARD"
echo "[INFO] baseline_result_csv=${BASELINE_RESULT_CSV:-<auto per mode>}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] classil_entrypoint=$CLASSIL_ENTRYPOINT"
echo "[INFO] baseline_entrypoint=$BASELINE_ENTRYPOINT"

check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    if [ -z "$resolved_log" ]; then
        return 0
    fi

    local fd target pid state cmdline
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue

        pid="${fd#/proc/}"
        pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue

        state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in
            torch_shm_manager*) continue ;;
        esac
        echo "[ERROR] $log_path is already held by PID $pid state=$state cmd=$cmdline" >&2
        echo "[ERROR] Refusing to reuse this log. Wait for the process to exit or use a different LOG_DIR." >&2
        return 1
    done
}

run_to_logs() {
    local result_log="$1"
    local error_log="$2"
    shift 2

    echo "[INFO] running: $*"
    echo "[INFO] result_log=$result_log"
    echo "[INFO] error_log=$error_log"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"

    {
        echo "[INFO] start at $(date)"
        echo "[INFO] command=$*"
    } > "$result_log"
    {
        echo "[INFO] start at $(date)"
        echo "[INFO] command=$*"
    } > "$error_log"

    "$@" >> "$result_log" 2>> "$error_log"
}

mode_enabled() {
    local run_mode="$1"
    if [ "$MODE" = "all" ]; then
        return 0
    fi
    [ "$MODE" = "$run_mode" ]
}

run_baseline_mode() {
    local run_mode="$1"
    local result_csv="${BASELINE_RESULT_CSV:-$LOG_DIR/baseline_${run_mode}_routing_results.csv}"
    local baseline_args=(
        --config "$CONFIG_FORWARD"
        --save_dir "$SAVE_DIR_FORWARD"
        --merge_model_path "$MERGE_MODEL_PATH_FORWARD"
        --mode "$run_mode"
        --entrypoint "$BASELINE_ENTRYPOINT"
    )

    if [ -n "$result_csv" ]; then
        if supports_arg "--result_csv"; then
            baseline_args+=(--result_csv "$result_csv")
        else
            echo "[WARN] $BASELINE_ENTRYPOINT does not support --result_csv; CSV will not be saved" >&2
        fi
    fi

    run_to_logs "$LOG_DIR/result_test_class_${run_mode}.log" "$LOG_DIR/error_test_class_${run_mode}.log" \
        "$PYTHON_BIN" -u "$CLASSIL_ENTRYPOINT" "${baseline_args[@]}"
}

if mode_enabled tcp; then
    run_baseline_mode tcp
fi
if mode_enabled naive; then
    run_baseline_mode naive
fi

echo "[INFO] finished at $(date)"
