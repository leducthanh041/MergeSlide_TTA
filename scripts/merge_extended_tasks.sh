#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
SETTING="${SETTING:-ind}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"
DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

case "$SETTING" in
    ind)
        CONFIG="${CONFIG:-configs/extended_ind.yaml}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
        MERGED_DIR="${MERGED_DIR:-./checkpoints/merged_extended}"
        ;;
    ood)
        CONFIG="${CONFIG:-configs/extended_ood.yaml}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints_ood/finetuned}"
        MERGED_DIR="${MERGED_DIR:-./checkpoints_ood/merged_extended}"
        ;;
    *)
        echo "[ERROR] SETTING must be ind or ood, got: $SETTING" >&2
        exit 2
        ;;
esac

LOG_DIR="${LOG_DIR:-logs/external_tasks/${SETTING}/merge}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$MERGED_DIR"

for fold in $(seq "$FOLD_START" $((FOLD_END - 1))); do
    for task in $(seq 0 7); do
        checkpoint="$FINETUNED_DIR/fold_${fold}/task_${task}.pt"
        test -f "$checkpoint" || {
            echo "[ERROR] Missing checkpoint: $checkpoint" >&2
            exit 1
        }
    done
done

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] setting=$SETTING config=$CONFIG"
echo "[INFO] folds=[$FOLD_START,$FOLD_END) tasks=[0,8)"
echo "[INFO] finetuned_dir=$FINETUNED_DIR"
echo "[INFO] merged_dir=$MERGED_DIR log_dir=$LOG_DIR"
echo "[INFO] C.OPCM is recomputed from task 0 to preserve exact merge state."

"$PYTHON_BIN" -u merge.py \
    --config "$CONFIG" \
    --fold_start "$FOLD_START" \
    --fold_end "$FOLD_END" \
    --finetuned_checkpoints "$FINETUNED_DIR" \
    --merged_checkpoints "$MERGED_DIR" \
    > >(tee "$LOG_DIR/result_merge_extended.log") \
    2> >(tee "$LOG_DIR/error_merge_extended.log" >&2)
