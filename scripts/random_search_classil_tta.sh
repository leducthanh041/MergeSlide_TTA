#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
PYTHON_BIN="${PYTHON_BIN:-/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10}"
SETTING="${SETTING:-ind}"
N_TRIALS="${N_TRIALS:-60}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-10}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
SINGLE_GPU="${SINGLE_GPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
RESET_MANIFEST="${RESET_MANIFEST:-0}"
TUNER="${TUNER:-tools/random_search_classil_tta.py}"
WRAPPER="${WRAPPER:-tools/run_classil_with_pt_features.py}"

cd "$PROJECT_ROOT"

case "$SETTING" in
    ind)
        BASE_CONFIG="${BASE_CONFIG:-configs/default_eval_num_workers0.yaml}"
        MERGE_DIR="${MERGE_DIR:-./checkpoints/merged}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
        ;;
    ood)
        BASE_CONFIG="${BASE_CONFIG:-configs/default_ood_eval_num_workers0.yaml}"
        MERGE_DIR="${MERGE_DIR:-./checkpoints_ood/merged}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints_ood/finetuned}"
        ;;
    *)
        echo "[ERROR] SETTING must be ind or ood" >&2
        exit 1
        ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-/docker/data/thanhld/MergeSlide_TTA/logs/tune_naive}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/launcher_logs/$SETTING}"
MANIFEST_DIR="$OUTPUT_ROOT/manifests/$SETTING"
MANIFEST_PATH="$MANIFEST_DIR/manifest_${SETTING}_naive_${N_TRIALS}_${SEED}.json"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$MANIFEST_DIR"

if [ "$RESET_MANIFEST" = "1" ] && [ -f "$MANIFEST_PATH" ]; then
    mv "$MANIFEST_PATH" "$MANIFEST_PATH.bak.$(date +%Y%m%d_%H%M%S)"
fi

if [ ! -f "$MANIFEST_PATH" ]; then
    "$PYTHON_BIN" -u "$TUNER" \
        --setting "$SETTING" \
        --n_trials "$N_TRIALS" \
        --seed "$SEED" \
        --base_config "$BASE_CONFIG" \
        --merge_dir "$MERGE_DIR" \
        --finetuned_dir "$FINETUNED_DIR" \
        --output_dir "$OUTPUT_ROOT" \
        --manifest_path "$MANIFEST_PATH" \
        --prepare_manifest
fi

if [ "$SINGLE_GPU" = "1" ] || [ "$GPU_A" = "$GPU_B" ]; then
    GPU_LIST=("$GPU_A")
else
    GPU_LIST=("$GPU_A" "$GPU_B")
fi

NUM_GPUS="${#GPU_LIST[@]}"
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
BASE_COUNT=$((N_TRIALS / TOTAL_WORKERS))
REMAINDER=$((N_TRIALS % TOTAL_WORKERS))
PIDS=()

echo "[INFO] mode=naive setting=$SETTING trials=$N_TRIALS"
echo "[INFO] fixed: n_steps=5 M=8 K_sub=300"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] manifest=$MANIFEST_PATH"

for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
    extra=0
    if [ "$worker" -lt "$REMAINDER" ]; then
        extra=1
    fi
    count=$((BASE_COUNT + extra))
    if [ "$count" -eq 0 ]; then
        continue
    fi
    if [ "$worker" -lt "$REMAINDER" ]; then
        start=$((worker * (BASE_COUNT + 1)))
    else
        start=$((REMAINDER * (BASE_COUNT + 1) + (worker - REMAINDER) * BASE_COUNT))
    fi
    end=$((start + count))
    gpu="${GPU_LIST[$((worker % NUM_GPUS))]}"
    tag="gpu${gpu}_w${worker}"
    worker_output="$OUTPUT_ROOT/$tag"
    stdout_log="$LOG_DIR/${tag}.log"
    stderr_log="$LOG_DIR/${tag}.err"

    echo "[INFO] launch $tag trials=[$start,$end)"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON_BIN" -u "$TUNER" \
            --setting "$SETTING" \
            --n_trials "$N_TRIALS" \
            --seed "$SEED" \
            --base_config "$BASE_CONFIG" \
            --merge_dir "$MERGE_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir "$worker_output" \
            --project_root "$PROJECT_ROOT" \
            --python_bin "$PYTHON_BIN" \
            --entrypoint_wrapper "$WRAPPER" \
            --manifest_path "$MANIFEST_PATH" \
            --trial_start "$start" \
            --trial_end "$end" \
            --timeout_sec "$TIMEOUT_SEC" \
            --top_k "$TOP_K"
    ) >"$stdout_log" 2>"$stderr_log" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

"$PYTHON_BIN" -u "$TUNER" \
    --setting "$SETTING" \
    --n_trials "$N_TRIALS" \
    --seed "$SEED" \
    --base_config "$BASE_CONFIG" \
    --merge_dir "$MERGE_DIR" \
    --finetuned_dir "$FINETUNED_DIR" \
    --output_dir "$OUTPUT_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --python_bin "$PYTHON_BIN" \
    --entrypoint_wrapper "$WRAPPER" \
    --top_k "$TOP_K" \
    --summarize_only

echo "[INFO] best_config=$OUTPUT_ROOT/$SETTING/naive/best_config.json"
