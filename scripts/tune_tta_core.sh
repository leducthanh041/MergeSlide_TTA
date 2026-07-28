#!/bin/bash
#
# Random search Module F percentile (conf/agree) cho MergeSlide-TTA-Unified.
# n_steps (Module G) và regularizer_type (Module H) ĐÃ CHỐT qua thực
# nghiệm — không sweep lại ở đây.
#
# Usage:
#   1) Chuẩn bị manifest (1 lần):
#      N_TRIALS=20 SEED=42 MODE=naive PREPARE_MANIFEST=1 bash scripts/tune_tta_core.sh
#   2) Chạy trial:
#      N_TRIALS=20 SEED=42 MODE=naive bash scripts/tune_tta_core.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
MODE="${MODE:-naive}"
N_TRIALS="${N_TRIALS:-20}"
SEED="${SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-10}"
BASE_CONFIG="${BASE_CONFIG:-configs/default_tta_core_eval_num_workers0.yaml}"
MERGE_DIR="${MERGE_DIR:-./checkpoints/merged}"
SWAG_DIR="${SWAG_DIR:-./checkpoints/swag_diagonal}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/tune_tta_core}"
PREPARE_MANIFEST="${PREPARE_MANIFEST:-0}"

case "$MODE" in
    naive|tcp|task_il) ;;
    *)
        echo "[ERROR] MODE must be one of: naive, tcp, task_il" >&2
        exit 1
        ;;
esac

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

ARGS=(
    --n_trials "$N_TRIALS" --seed "$SEED" --mode "$MODE"
    --base_config "$BASE_CONFIG" --merge_dir "$MERGE_DIR" --swag_dir "$SWAG_DIR"
    --output_dir "$OUTPUT_DIR" --num_folds "$NUM_FOLDS"
    --project_root "$PROJECT_ROOT"
    --entrypoint_wrapper tools/run_classil_with_pt_features.py
)

if [ "$PREPARE_MANIFEST" = "1" ]; then
    ARGS+=(--prepare_manifest)
fi

echo "[INFO] MODE=$MODE N_TRIALS=$N_TRIALS SEED=$SEED BASE_CONFIG=$BASE_CONFIG"
"$PYTHON_BIN" -u tta_tuners/tune_tta_core.py "${ARGS[@]}"
