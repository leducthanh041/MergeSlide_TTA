#!/bin/bash
#
# TASK-IL TTA evaluation runner for IND/OOD forward and IND reverse protocols.
# Task-IL knows the task identity, so tcp/naive is not used here.

#SBATCH --job-name=test_taskIL_tta
#SBATCH --output=logs/test_taskIL_tta_%j.out
#SBATCH --error=logs/test_taskIL_tta_%j.err
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
MODE="${MODE:-}"
LOG_DIR="${LOG_DIR:-}"
if [ -n "$LOG_DIR" ] && [[ "$LOG_DIR" != /* && "$LOG_DIR" != logs && "$LOG_DIR" != logs/* ]]; then
    LOG_DIR="logs/$LOG_DIR"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/taskil_tta/${SETTING}_${ORDER}"
fi

case "${SETTING}_${ORDER}" in
    ood_forward)
        CONFIG="${CONFIG:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints_ood/merged}"
        ;;
    ind_forward)
        CONFIG="${CONFIG:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged}"
        ;;
    ind_reverse)
        CONFIG="${CONFIG:-configs/default_reverse_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned_reverse}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged_reverse}"
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

CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
TASKIL_TTA_ENTRYPOINT="${TASKIL_TTA_ENTRYPOINT:-test_taskIL_tta.py}"

if [ -n "$MODE" ]; then
    echo "[WARN] MODE=$MODE is ignored for Task-IL TTA because task identity is known." >&2
fi

TTA_M="${TTA_M:-8}"
TTA_K_SUB="${TTA_K_SUB:-300}"
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"
TTA_BETA="${TTA_BETA:-1.0}"
TTA_LR="${TTA_LR:-1e-4}"
TTA_N_STEPS="${TTA_N_STEPS:-5}"
TTA_PARAM_SCOPE="${TTA_PARAM_SCOPE:-ln_only}"
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"
TTA_SELECT_MODE="${TTA_SELECT_MODE:-intersection}"
TTA_VERBOSE_LOSS="${TTA_VERBOSE_LOSS:-1}"
TTA_IND_BEST_CONFIG="${TTA_IND_BEST_CONFIG:-configs/ind/best_config.json}"
TTA_OOD_BEST_CONFIG="${TTA_OOD_BEST_CONFIG:-configs/ood/best_config.json}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Fixed best hyperparameters.
# Task-IL uses shared TTA knobs from the best config, ignoring Class-IL-only
# fields such as alpha/gamma/prompt EMA. Missing best config is an error to
# avoid silently running script defaults after hyperparameter tuning is done.
# ---------------------------------------------------------------------------
ACTIVE_BEST_CONFIG=""
ACTIVE_BEST_LABEL=""
if [ "$SETTING" = "ind" ]; then
    ACTIVE_BEST_CONFIG="$TTA_IND_BEST_CONFIG"
    ACTIVE_BEST_LABEL="IND"
elif [ "$SETTING" = "ood" ]; then
    ACTIVE_BEST_CONFIG="$TTA_OOD_BEST_CONFIG"
    ACTIVE_BEST_LABEL="OOD"
fi
if [ -n "$ACTIVE_BEST_CONFIG" ]; then
    if [ ! -f "$ACTIVE_BEST_CONFIG" ]; then
        echo "[ERROR] ${ACTIVE_BEST_LABEL} best Task-IL TTA config not found: $ACTIVE_BEST_CONFIG" >&2
        exit 1
    else
    eval "$("$PYTHON_BIN" - "$ACTIVE_BEST_CONFIG" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

mapping = {
    "M": "TTA_M",
    "K_sub": "TTA_K_SUB",
    "top_ratio": "TTA_TOP_RATIO",
    "beta": "TTA_BETA",
    "lr": "TTA_LR",
    "n_steps": "TTA_N_STEPS",
    "tta_param_scope": "TTA_PARAM_SCOPE",
    "entropy_threshold": "TTA_ENTROPY_THRESHOLD",
    "select_mode": "TTA_SELECT_MODE",
}
for key, env_name in mapping.items():
    if key in cfg:
        print(f"{env_name}={shlex.quote(str(cfg[key]))}")
PY
)"
    echo "[INFO] loaded $ACTIVE_BEST_LABEL best Task-IL TTA config: $ACTIVE_BEST_CONFIG"
    fi
fi

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

RESET_LABEL="continual"
if [ "${TTA_EPISODIC:-0}" = "1" ]; then
    echo "[WARN] TTA_EPISODIC=1 is ignored. MergeSlide_TTA uses continual adaptation." >&2
fi

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] setting=$SETTING"
echo "[INFO] order=$ORDER"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] config=$CONFIG"
echo "[INFO] save_dir=$SAVE_DIR"
echo "[INFO] merge_model_path=$MERGE_MODEL_PATH"
echo "[INFO] classil_wrapper=$CLASSIL_WRAPPER"
echo "[INFO] taskil_tta_entrypoint=$TASKIL_TTA_ENTRYPOINT"
echo "[INFO] M=$TTA_M | K_sub=$TTA_K_SUB | top_ratio=$TTA_TOP_RATIO | beta=$TTA_BETA | lr=$TTA_LR | n_steps=$TTA_N_STEPS | param_scope=$TTA_PARAM_SCOPE | entropy_threshold=$TTA_ENTROPY_THRESHOLD | select_mode=$TTA_SELECT_MODE | reset=$RESET_LABEL | verbose_loss=$TTA_VERBOSE_LOSS"

check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    if [ -z "$resolved_log" ]; then return 0; fi

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
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path is already held by PID $pid state=$state cmd=$cmdline" >&2
        echo "[ERROR] Refusing to reuse this log." >&2
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
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$result_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$error_log"
    "$@" >> "$result_log" 2>> "$error_log"
}

TTA_ARGS=(
    --M                 "$TTA_M"
    --K_sub             "$TTA_K_SUB"
    --top_ratio         "$TTA_TOP_RATIO"
    --beta              "$TTA_BETA"
    --lr                "$TTA_LR"
    --n_steps           "$TTA_N_STEPS"
    --tta_param_scope   "$TTA_PARAM_SCOPE"
    --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --select_mode       "$TTA_SELECT_MODE"
)
if [ "$TTA_VERBOSE_LOSS" = "1" ]; then
    TTA_ARGS+=(--verbose_loss)
fi

run_to_logs "$LOG_DIR/result_taskil_tta.log" "$LOG_DIR/error_taskil_tta.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint "$TASKIL_TTA_ENTRYPOINT" \
        --config "$CONFIG" \
        --save_dir "$SAVE_DIR" \
        --merge_model_path "$MERGE_MODEL_PATH" \
        "${TTA_ARGS[@]}"

echo "[INFO] finished at $(date)"
