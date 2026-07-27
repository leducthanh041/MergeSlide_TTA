#!/bin/bash
#
# So sánh classil_naive (test_tta_v3.py, engine gốc, KHÔNG Module F) vs
# core_naive (test_tta_core.py, engine mới, CÓ Module F) — trả lời AC-2/AC-9.
#
# Yêu cầu CẢ HAI đã chạy xong trên CÙNG checkpoint/fold trước:
#   1) scripts/test_tta_v3.sh với nhánh classil_naive được bật (bỏ comment
#      trong script gốc, hoặc chạy tay — xem README của test_tta_v3.py)
#   2) scripts/test_tta_core.sh
#
# Usage:
#   NAIVE_CSV=logs/tta_v3_results_classil_naive.csv \
#   CORE_CSV=logs/tta_core_results_ind.csv \
#   OUT_DIR=logs/compare_naive_vs_core_ind \
#     bash scripts/compare_naive_vs_core.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
NAIVE_CSV="${NAIVE_CSV:-logs/tta_v3_results_classil_naive.csv}"
CORE_CSV="${CORE_CSV:-logs/tta_core_results_ind.csv}"
OUT_DIR="${OUT_DIR:-logs/compare_naive_vs_core_ind}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

if [ ! -f "$NAIVE_CSV" ]; then
    echo "[ERROR] Không tìm thấy $NAIVE_CSV." >&2
    echo "        Chạy test_tta_v3.py --mode classil_naive trước (xem scripts/test_tta_v3.sh," >&2
    echo "        bỏ comment khối 'CLASS-IL Naive')." >&2
    exit 1
fi
if [ ! -f "$CORE_CSV" ]; then
    echo "[ERROR] Không tìm thấy $CORE_CSV. Chạy scripts/test_tta_core.sh trước." >&2
    exit 1
fi

echo "[INFO] NAIVE_CSV=$NAIVE_CSV"
echo "[INFO] CORE_CSV=$CORE_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"

"$PYTHON_BIN" -u tools/compare_naive_vs_core.py \
    --naive_csv "$NAIVE_CSV" \
    --core_csv  "$CORE_CSV" \
    --out_dir   "$OUT_DIR"

echo "[INFO] Xong. Xem $OUT_DIR/compare_naive_vs_core_report.md"
