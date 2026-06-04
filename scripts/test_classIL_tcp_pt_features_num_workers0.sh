#!/bin/bash
#
# Run forward CLASS-IL TCP evaluation without changing the original entrypoint.
# Feature tensors are read from pt_files when available; coords still come from H5.

#SBATCH --job-name=test_classIL_tcp_pt_nw0
#SBATCH --output=logs/test_classIL_tcp_pt_nw0_%j.out
#SBATCH --error=logs/test_classIL_tcp_pt_nw0_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
CONFIG="${CONFIG:-configs/default_eval_num_workers0.yaml}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned}"
MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged}"
RESULT_LOG="${RESULT_LOG:-logs/result_test_class_tcp_pt_nw0.log}"
ERROR_LOG="${ERROR_LOG:-logs/error_test_class_tcp_pt_nw0.log}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"
mkdir -p logs

echo "[INFO] start at $(date)"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] config=$CONFIG"
echo "[INFO] save_dir=$SAVE_DIR"
echo "[INFO] merge_model_path=$MERGE_MODEL_PATH"
echo "[INFO] result_log=$RESULT_LOG"
echo "[INFO] error_log=$ERROR_LOG"

"$PYTHON_BIN" -u tools/run_classil_with_pt_features.py \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    --merge_model_path "$MERGE_MODEL_PATH" \
    --mode tcp \
    2> >(tee "$ERROR_LOG" >&2) \
    > >(tee "$RESULT_LOG")
