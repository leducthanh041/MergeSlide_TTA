#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SETTING="${SETTING:-ood}"
TTA_PARAM_FILE="${TTA_PARAM_FILE:-configs/${SETTING}/tta_${SETTING}.env}"

if [[ "$TTA_PARAM_FILE" != /* ]]; then
    TTA_PARAM_FILE="$PROJECT_ROOT/$TTA_PARAM_FILE"
fi
if [ ! -f "$TTA_PARAM_FILE" ]; then
    echo "[ERROR] TTA parameter file not found: $TTA_PARAM_FILE" >&2
    exit 1
fi
source "$TTA_PARAM_FILE"

case "$SETTING" in
    ind)
        CONFIG="${CONFIG:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged}"
        ;;
    ood)
        CONFIG="${CONFIG:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints_ood/merged}"
        ;;
    *)
        echo "[ERROR] SETTING must be ind or ood, got: $SETTING" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-logs/ablation/tcp_evidence/${SETTING}}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"
MAX_SLIDES_PER_TASK="${MAX_SLIDES_PER_TASK:-0}"
RESUME="${RESUME:-1}"
PYTHON_BIN="${PYTHON_BIN:-/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10}"

: "${TTA_M:=8}"
: "${TTA_K_SUB:=300}"
: "${TTA_TOP_RATIO:=0.5}"
: "${TTA_ALPHA:=0.5}"
: "${TTA_L2_ANCHOR_BETA:=1.0}"
: "${TTA_LR:=1e-4}"
: "${TTA_N_STEPS:=5}"
: "${TTA_PARAM_SCOPE:=ln_only}"
: "${TTA_ENTROPY_THRESHOLD:=0.4}"
: "${TTA_GAMMA:=0.5}"
: "${TTA_SELECT_MODE:=intersection}"
: "${TTA_EMA_ALPHA:=0.999}"
: "${TTA_EMA_ALPHA_PROMPT:=0.999}"
: "${TTA_DELTA_MARGIN:=0.10}"
: "${TTA_TP_ANCHOR_BETA:=0.3}"
: "${TTA_GAMMA_MARGIN:=0.0}"
: "${TTA_TAU_TASK:=0.70}"
: "${TTA_DAPC_LOSS_WEIGHT:=1.0}"
: "${TTA_ENTROPY_LOSS_WEIGHT:=1.0}"
: "${TTA_DAPC_TAU_ANCHOR:=0.92}"
: "${TTA_DAPC_BETA:=1.2}"
: "${TTA_USE_DAPC:=1}"
: "${TTA_NO_TEACHER:=0}"
: "${TTA_NO_ADAPT_PROMPTS:=0}"
: "${TTA_NO_TASK_AGREEMENT:=0}"
: "${TTA_USE_TASK_DIVERSITY:=0}"
: "${TTA_NO_RESET_PROMPT_PER_TASK:=0}"

if [ "$TTA_USE_DAPC" != "1" ]; then
    echo "[ERROR] Evidence collection requires TTA_USE_DAPC=1 for the immutable source anchor." >&2
    exit 1
fi
if [ "$TTA_NO_TEACHER" = "1" ]; then
    echo "[ERROR] TCP evidence collection requires the EMA teacher." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

ARGS=(
    --entrypoint tools/plot/collect_tcp_alignment_evidence.py
    --config "$CONFIG"
    --save_dir "$SAVE_DIR"
    --merge_model_path "$MERGE_MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --fold_start "$FOLD_START"
    --fold_end "$FOLD_END"
    --max_slides_per_task "$MAX_SLIDES_PER_TASK"
    --M "$TTA_M"
    --K_sub "$TTA_K_SUB"
    --top_ratio "$TTA_TOP_RATIO"
    --alpha "$TTA_ALPHA"
    --l2_anchor_beta "$TTA_L2_ANCHOR_BETA"
    --lr "$TTA_LR"
    --n_steps "$TTA_N_STEPS"
    --tta_param_scope "$TTA_PARAM_SCOPE"
    --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --gamma "$TTA_GAMMA"
    --select_mode "$TTA_SELECT_MODE"
    --ema_alpha "$TTA_EMA_ALPHA"
    --ema_alpha_prompt "$TTA_EMA_ALPHA_PROMPT"
    --delta_margin "$TTA_DELTA_MARGIN"
    --tp_anchor_beta "$TTA_TP_ANCHOR_BETA"
    --gamma_margin "$TTA_GAMMA_MARGIN"
    --tau_task "$TTA_TAU_TASK"
    --dapc_loss_weight "$TTA_DAPC_LOSS_WEIGHT"
    --entropy_loss_weight "$TTA_ENTROPY_LOSS_WEIGHT"
    --dapc_tau_anchor "$TTA_DAPC_TAU_ANCHOR"
    --dapc_beta "$TTA_DAPC_BETA"
    --use_dapc
)

if [ "$TTA_NO_ADAPT_PROMPTS" = "1" ]; then ARGS+=(--no_adapt_prompts); fi
if [ "$TTA_NO_TASK_AGREEMENT" = "1" ]; then ARGS+=(--no_task_agreement); fi
if [ "$TTA_USE_TASK_DIVERSITY" = "1" ]; then ARGS+=(--use_task_diversity); fi
if [ "$TTA_NO_RESET_PROMPT_PER_TASK" = "1" ]; then ARGS+=(--no_reset_prompt_per_task); fi
if [ "$RESUME" = "1" ]; then ARGS+=(--resume); fi

echo "[INFO] setting=$SETTING folds=[$FOLD_START,$FOLD_END)"
echo "[INFO] resume=$RESUME"
echo "[INFO] params=$TTA_PARAM_FILE"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] command=$PYTHON_BIN tools/run_classil_with_pt_features.py ${ARGS[*]}"

"$PYTHON_BIN" tools/run_classil_with_pt_features.py "${ARGS[@]}" \
    > >(tee "$OUTPUT_DIR/collect_result.log") \
    2> >(tee "$OUTPUT_DIR/collect_error.log" >&2)

"$PYTHON_BIN" tools/plot/plot_tcp_alignment_evidence.py \
    --input_csv "$OUTPUT_DIR/paired_wsi_scores.csv" \
    --output_dir "$OUTPUT_DIR" \
    --tag "$SETTING" \
    > >(tee "$OUTPUT_DIR/plot_result.log") \
    2> >(tee "$OUTPUT_DIR/plot_error.log" >&2)
