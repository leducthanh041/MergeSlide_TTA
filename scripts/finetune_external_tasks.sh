#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
SETTING="${SETTING:-ind}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"
GPU="${GPU:-4}"
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
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned}"
        ;;
    ood)
        CONFIG="${CONFIG:-configs/extended_ood.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints_ood/finetuned}"
        ;;
    *)
        echo "[ERROR] SETTING must be ind or ood, got: $SETTING" >&2
        exit 2
        ;;
esac

LOG_DIR="${LOG_DIR:-logs/external_tasks/${SETTING}/finetune}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

for dataset in bracs herohe; do
    split_dir="$("$PYTHON_BIN" - "$CONFIG" "$dataset" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(OmegaConf.to_container(cfg.dataset[sys.argv[2]], resolve=True)["splits"])
PY
)"
    for fold in $(seq "$FOLD_START" $((FOLD_END - 1))); do
        test -f "$split_dir/splits_${fold}.csv" || {
            echo "[ERROR] Missing split: $split_dir/splits_${fold}.csv" >&2
            exit 1
        }
    done
done

export CUDA_VISIBLE_DEVICES="$GPU"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] setting=$SETTING config=$CONFIG gpu=$GPU"
echo "[INFO] folds=[$FOLD_START,$FOLD_END) tasks=[6,8)"
echo "[INFO] save_dir=$SAVE_DIR log_dir=$LOG_DIR"

"$PYTHON_BIN" -u tools/run_classil_with_pt_features.py \
    --entrypoint train.py \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    --fold_start "$FOLD_START" \
    --fold_end "$FOLD_END" \
    --task_start 6 \
    --task_end 8 \
    > >(tee "$LOG_DIR/result_finetune_external.log") \
    2> >(tee "$LOG_DIR/error_finetune_external.log" >&2)
