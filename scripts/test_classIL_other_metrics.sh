#!/bin/bash
#
# CLASS-IL continual metrics runner. Logs/checkpoints are kept on local /docker
# via repo symlinks, while datasets remain read-only inputs on /mmlab_students.

#SBATCH --job-name=om_test_classIL
#SBATCH --output=logs/om_test_classIL_%j.out
#SBATCH --error=logs/om_test_classIL_%j.err
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
LOG_DIR="${LOG_DIR:-logs}"
CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_eval_num_workers0.yaml}"
#CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_ood_eval_num_workers0.yaml}"
CONFIG_REVERSE="${CONFIG_REVERSE:-configs/default_reverse_eval_num_workers0.yaml}"
CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
CLASSIL_OTHER_METRICS_ENTRYPOINT="${CLASSIL_OTHER_METRICS_ENTRYPOINT:-test_classIL_task_prompt_other_metrics.py}"

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
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] config_forward=$CONFIG_FORWARD"
echo "[INFO] config_reverse=$CONFIG_REVERSE"
echo "[INFO] classil_wrapper=$CLASSIL_WRAPPER"
echo "[INFO] other_metrics_entrypoint=$CLASSIL_OTHER_METRICS_ENTRYPOINT"

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

run_with_tee() {
    local result_log="$1"
    local error_log="$2"
    shift 2

    echo "[INFO] running: $*"
    echo "[INFO] result_log=$result_log"
    echo "[INFO] error_log=$error_log"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"
    "$@" 2> >(tee "$error_log" >&2) > >(tee "$result_log")
}


run_with_tee "$LOG_DIR/result_om_naive.log" "$LOG_DIR/error_om_naive.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint "$CLASSIL_OTHER_METRICS_ENTRYPOINT" \
        --config "$CONFIG_FORWARD" \
        --save_dir ./checkpoints/finetuned \
        --merge_model_path ./checkpoints/merged \
        --mode naive

#run_with_tee "$LOG_DIR/result_om_tcp.log" "$LOG_DIR/error_om_tcp.log" \
#    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
#        --entrypoint "$CLASSIL_OTHER_METRICS_ENTRYPOINT" \
#        --config "$CONFIG_FORWARD" \
#        --save_dir ./checkpoints_ood/finetuned \
#        --merge_model_path ./checkpoints_ood/merged \
#        --mode tcp

run_with_tee "$LOG_DIR/result_om_naive_re.log" "$LOG_DIR/error_om_naive_re.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint "$CLASSIL_OTHER_METRICS_ENTRYPOINT" \
        --config "$CONFIG_REVERSE" \
        --save_dir ./checkpoints/finetuned_reverse \
        --merge_model_path ./checkpoints/merged_reverse \
        --mode naive

#run_with_tee "$LOG_DIR/result_om_tcp_re.log" "$LOG_DIR/error_om_tcp_re.log" \
#    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
#        --entrypoint "$CLASSIL_OTHER_METRICS_ENTRYPOINT" \
#        --config "$CONFIG_REVERSE" \
#        --save_dir ./checkpoints/finetuned_reverse \
#        --merge_model_path ./checkpoints/merged_reverse \
#        --mode tcp

echo "[INFO] finished at $(date)"
