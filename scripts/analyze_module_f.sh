#!/bin/bash
#
# Chạy phân tích AC-8/AC-9/AC-10/AC-12 trên tta_stats CSV đã sinh ra từ
# scripts/test_tta_core.sh (hoặc test_tta_core_ood.sh).
#
# Usage:
#   STATS_CSV=logs/tta_core_stats_ind.csv OUT_DIR=logs/module_f_analysis_ind \
#     bash scripts/analyze_module_f.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
STATS_CSV="${STATS_CSV:-logs/tta_core_stats_ind.csv}"
OUT_DIR="${OUT_DIR:-logs/module_f_analysis_ind}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

if [ ! -f "$STATS_CSV" ]; then
    echo "[ERROR] Không tìm thấy $STATS_CSV. Chạy scripts/test_tta_core.sh trước (nhớ set --tta_stats_csv)." >&2
    exit 1
fi

echo "[INFO] STATS_CSV=$STATS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"

"$PYTHON_BIN" -u tools/analyze_module_f_reliability.py \
    --tta_stats_csv "$STATS_CSV" \
    --out_dir       "$OUT_DIR"

echo "[INFO] Xong. Xem $OUT_DIR/module_f_reliability_report.md"
