#!/bin/bash
#
# Tính Fisher importance omega(theta) cho Module D' (AC-6, tuỳ chọn).
# Chạy SAU khi đã có checkpoints/merged/fold_{N}/merged_final.pth cho
# mọi fold cần dùng. KHÔNG bắt buộc nếu bạn chỉ dùng Module D mặc định
# (use_fim_restore=true, use_fisher_reg=false trong config).
#
# Usage:
#   NUM_FOLDS=10 CONFIG=configs/default_tta_core_eval_num_workers0.yaml \
#     bash scripts/compute_fisher_importance.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
CONFIG="${CONFIG:-configs/default_tta_core_eval_num_workers0.yaml}"
MERGED_DIR="${MERGED_DIR:-./checkpoints/merged}"
OUT_DIR="${OUT_DIR:-./checkpoints/fisher_omega}"
NUM_FOLDS="${NUM_FOLDS:-10}"
POOL_TASKS="${POOL_TASKS:-all}"
SPLIT="${SPLIT:-val}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-100}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

for ((fold=0; fold<NUM_FOLDS; fold++)); do
    merged_path="$MERGED_DIR/fold_${fold}/merged_final.pth"
    if [ ! -f "$merged_path" ]; then
        echo "[WARN] $merged_path không tồn tại — bỏ qua fold $fold"
        continue
    fi
    out_path="$OUT_DIR/fold_${fold}.pt"
    echo "[INFO] Fold $fold -> $out_path"
    "$PYTHON_BIN" -u tools/compute_fisher_importance.py \
        --config               "$CONFIG" \
        --merge_model_path     "$MERGED_DIR" \
        --fold                 "$fold" \
        --pool_tasks            "$POOL_TASKS" \
        --split                "$SPLIT" \
        --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
        --out_path             "$out_path"
done

echo "[INFO] Xong. Set FISHER_OMEGA_DIR=$OUT_DIR khi chạy test_tta_core.sh với tta.use_fisher_reg=true."
