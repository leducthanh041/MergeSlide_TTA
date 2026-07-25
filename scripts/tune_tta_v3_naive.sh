#!/bin/bash
#
# Single-phase TTA v3 hyperparameter tuning for CLASS-IL naive mode.
#
# Naive mode does not use TCP routing, so this runner tunes only class/adaptation
# parameters and forces gamma_task=0.0 inside tune_tta_v3_naive.py.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-$MERGESLIDE_LOCAL_ROOT/logs/tune_tta_v3_naive_runs}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    [ -x "$DEFAULT_PYTHON" ] && PYTHON_BIN="$DEFAULT_PYTHON" || PYTHON_BIN="python"
fi

SETTING="${SETTING:-ind}"
SEED="${SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-10}"
N_TRIALS="${N_TRIALS:-60}"
ENTRYPOINT_WRAPPER="${ENTRYPOINT_WRAPPER:-tools/run_classil_with_pt_features.py}"
TUNE_ENTRYPOINT="${TUNE_ENTRYPOINT:-tta_tuners/tune_tta_v3_naive.py}"
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

export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
mkdir -p "$LOG_DIR/tune_tta_v3_naive"

if [ "$SETTING" = "ind" ]; then
    BASE_CONFIG="${BASE_CONFIG:-configs/default_tta_eval_num_workers0.yaml}"
    MERGE_DIR="${MERGE_DIR:-./checkpoints/merged}"
    SWAG_DIR="${SWAG_DIR:-$PROJECT_ROOT/checkpoints/swag_diagonal}"
    FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$LOG_DIR/tune_tta_v3_naive_ind}"
elif [ "$SETTING" = "ood" ]; then
    BASE_CONFIG="${BASE_CONFIG:-configs/default_tta_ood_eval_num_workers0.yaml}"
    MERGE_DIR="${MERGE_DIR:-./checkpoints_ood/merged}"
    SWAG_DIR="${SWAG_DIR:-$PROJECT_ROOT/checkpoints_ood/swag_diagonal}"
    FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints_ood/finetuned}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$LOG_DIR/tune_tta_v3_naive_ood}"
else
    echo "[ERROR] SETTING='$SETTING' khong hop le. Dung: ind | ood" >&2
    exit 1
fi

echo "[INFO] start at $(date)"
echo "[INFO] TTA v3 naive tuning"
echo "[INFO] SETTING=$SETTING"
echo "[INFO] BASE_CONFIG=$BASE_CONFIG"
echo "[INFO] MERGE_DIR=$MERGE_DIR"
echo "[INFO] SWAG_DIR=$SWAG_DIR"
echo "[INFO] FINETUNED_DIR=$FINETUNED_DIR"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] GPU_A=$GPU_A  GPU_B=$GPU_B  WORKERS_PER_GPU=$WORKERS_PER_GPU  SINGLE_GPU=$SINGLE_GPU"
echo "[INFO] reset policy: CLASS-IL naive=continual"

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

manifest_contains_default() {
    local manifest_path="$1"
    local base_config_path="$2"
    "$PYTHON_BIN" -c '
import json, sys, yaml
manifest_path, base_config_path = sys.argv[1:3]
keys = ["eta_base", "delta", "tau_c", "tau_ood", "gamma_class"]
with open(base_config_path, "r") as f:
    cfg = yaml.safe_load(f) or {}
defaults = {k: cfg.get("tta", {}).get(k) for k in keys}
with open(manifest_path, "r") as f:
    manifest = json.load(f)
for item in manifest:
    params = item.get("params", {})
    if all(params.get(k) == defaults.get(k) for k in defaults):
        sys.exit(0)
sys.exit(1)
' "$manifest_path" "$base_config_path"
}

prepare_manifest() {
    local manifest_path="$OUTPUT_ROOT/manifest_${SETTING}_naive_${N_TRIALS}_${SEED}.json"
    local manifest_csv="${manifest_path%.json}.csv"

    mkdir -p "$(dirname "$manifest_path")"
    check_log_not_held "$manifest_path"
    check_log_not_held "$manifest_csv"

    if [ -f "$manifest_path" ] && ! manifest_contains_default "$manifest_path" "$BASE_CONFIG"; then
        echo "[INFO][naive] manifest exists and is clean, reuse -> $manifest_path"
    else
        if [ -f "$manifest_path" ]; then
            echo "[WARN][naive] manifest contains default-config sample(s); rebuilding -> $manifest_path"
        fi
        echo "[INFO][naive] preparing manifest..."
        "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
            --n_trials "$N_TRIALS" \
            --setting "$SETTING" \
            --base_config "$BASE_CONFIG" \
            --merge_dir "$MERGE_DIR" \
            --swag_dir "$SWAG_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir "$OUTPUT_ROOT" \
            --seed "$SEED" \
            --num_folds "$NUM_FOLDS" \
            --project_root "$PROJECT_ROOT" \
            --python_bin "$PYTHON_BIN" \
            --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
            --manifest_path "$manifest_path" \
            --prepare_manifest
    fi
    MANIFEST_PATH="$manifest_path"
}

run_worker() {
    local manifest_path="$1"
    local gpu_id="$2"
    local trial_start="$3"
    local trial_end="$4"
    local worker_tag="$5"
    local worker_output="$OUTPUT_ROOT/$worker_tag"
    local worker_log_dir="$LOG_DIR/tune_tta_v3_naive/$worker_tag"
    local worker_log="$worker_log_dir/tune_${SETTING}_naive.log"
    local worker_err="$worker_log_dir/tune_${SETTING}_naive.err"

    mkdir -p "$worker_output" "$worker_log_dir"
    check_log_not_held "$worker_log"
    check_log_not_held "$worker_err"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "[INFO] start at $(date)"
        echo "[INFO] mode=classil_naive gpu=$gpu_id trial_start=$trial_start trial_end=$trial_end"
        echo "[INFO] manifest=$manifest_path"
        "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
            --n_trials "$N_TRIALS" \
            --setting "$SETTING" \
            --base_config "$BASE_CONFIG" \
            --merge_dir "$MERGE_DIR" \
            --swag_dir "$SWAG_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir "$worker_output" \
            --seed "$SEED" \
            --num_folds "$NUM_FOLDS" \
            --project_root "$PROJECT_ROOT" \
            --python_bin "$PYTHON_BIN" \
            --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
            --manifest_path "$manifest_path" \
            --trial_start "$trial_start" \
            --trial_end "$trial_end"
    ) > "$worker_log" 2> "$worker_err" &
    RUN_WORKER_PID=$!
}

run_naive_tuning() {
    prepare_manifest
    local manifest_path="$MANIFEST_PATH"

    if [ "$SINGLE_GPU" = "1" ] || { [ "$GPU_A" = "$GPU_B" ] && [ "$WORKERS_PER_GPU" -eq 1 ]; }; then
        GPU_LIST=("$GPU_A")
        echo "[INFO][naive] single-GPU mode enabled -> GPU=$GPU_A"
    else
        GPU_LIST=("$GPU_A" "$GPU_B")
    fi

    local num_gpus="${#GPU_LIST[@]}"
    local total_workers=$(( num_gpus * WORKERS_PER_GPU ))
    if [ "$total_workers" -le 0 ]; then
        echo "[ERROR] total_workers must be > 0" >&2
        exit 1
    fi
    echo "[INFO][naive] total_workers=$total_workers"

    local base_trials=$(( N_TRIALS / total_workers ))
    local remainder=$(( N_TRIALS % total_workers ))
    declare -a pids=()

    for ((worker_idx=0; worker_idx<total_workers; worker_idx++)); do
        local gpu_idx=$(( worker_idx % num_gpus ))
        local gpu_id="${GPU_LIST[$gpu_idx]}"
        local extra=0
        if [ "$worker_idx" -lt "$remainder" ]; then
            extra=1
        fi
        local count=$(( base_trials + extra ))
        local start
        if [ "$worker_idx" -lt "$remainder" ]; then
            start=$(( worker_idx * (base_trials + 1) ))
        else
            start=$(( remainder * (base_trials + 1) + (worker_idx - remainder) * base_trials ))
        fi
        local end=$(( start + count ))
        if [ "$count" -le 0 ]; then
            echo "[INFO][naive] skip worker_idx=$worker_idx gpu=$gpu_id (no trials)"
            continue
        fi
        local worker_tag="gpu${gpu_id}_w${worker_idx}"
        echo "[INFO][naive] launch worker_idx=$worker_idx gpu=$gpu_id trial_start=$start trial_end=$end tag=$worker_tag"
        run_worker "$manifest_path" "$gpu_id" "$start" "$end" "$worker_tag"
        pids+=("$RUN_WORKER_PID")
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "[INFO][naive] building combined summary..."
    "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
        --setting "$SETTING" \
        --base_config "$BASE_CONFIG" \
        --merge_dir "$MERGE_DIR" \
        --swag_dir "$SWAG_DIR" \
        --finetuned_dir "$FINETUNED_DIR" \
        --output_dir "$OUTPUT_ROOT" \
        --summarize_only
}

run_naive_tuning

echo "[INFO] finished at $(date)"
echo "[INFO] best=$OUTPUT_ROOT/$SETTING/best_trial.json"
