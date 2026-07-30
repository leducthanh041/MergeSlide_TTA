#!/bin/bash
#
# Task-IL TTA entropy-threshold ablation.
#
# This runner uses scripts/test_taskIL_tta.sh, which evaluates Task-IL with
# known task identity. It sweeps only entropy_threshold and keeps n_steps fixed.
#
# Examples:
#   SETTING=ind bash scripts/test_taskIL_tta_entropy_threshold_ablation.sh
#   SETTING=ood THRESHOLDS="0.2 0.3 0.4 0.6" bash scripts/test_taskIL_tta_entropy_threshold_ablation.sh
#   SETTING=all CUDA_VISIBLE_DEVICES=0 bash scripts/test_taskIL_tta_entropy_threshold_ablation.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
cd "$PROJECT_ROOT"

SETTING="${SETTING:-all}"
ORDER="${ORDER:-forward}"
THRESHOLDS="${THRESHOLDS:-0.2 0.3 0.4 0.6}"
TTA_N_STEPS="${TTA_N_STEPS:-5}"
LOG_BASE="${LOG_BASE:-logs/ablation_entropy_threshold}"
USE_BEST_CONFIG="${USE_BEST_CONFIG:-1}"

case "$ORDER" in
    forward) ;;
    *) echo "[ERROR] Task-IL entropy-threshold ablation expects ORDER=forward, got ORDER=$ORDER" >&2; exit 1 ;;
esac

case "$SETTING" in
    ind) SETTINGS=("ind") ;;
    ood) SETTINGS=("ood") ;;
    all) SETTINGS=("ind" "ood") ;;
    *) echo "[ERROR] Unsupported SETTING=$SETTING (expected ind|ood|all)" >&2; exit 1 ;;
esac

threshold_tag() {
    echo "$1" | sed 's/\./p/g'
}

echo "[INFO] Task-IL TTA entropy-threshold ablation"
echo "[INFO] settings=${SETTINGS[*]} thresholds=$THRESHOLDS n_steps=$TTA_N_STEPS"
echo "[INFO] log_base=$LOG_BASE use_best_config=$USE_BEST_CONFIG"

for setting_name in "${SETTINGS[@]}"; do
    for threshold in $THRESHOLDS; do
        tag="$(threshold_tag "$threshold")"
        run_dir="$LOG_BASE/${setting_name}_forward/taskil_e${tag}"

        echo
        echo "================================================================"
        echo "[INFO] SETTING=$setting_name Task-IL entropy_threshold=$threshold n_steps=$TTA_N_STEPS"
        echo "[INFO] LOG_DIR=$run_dir"
        echo "================================================================"

        SETTING="$setting_name" \
        ORDER=forward \
        MODE=tcp \
        USE_BEST_CONFIG="$USE_BEST_CONFIG" \
        TTA_N_STEPS="$TTA_N_STEPS" \
        TTA_ENTROPY_THRESHOLD="$threshold" \
        LOG_DIR="$run_dir" \
            bash scripts/test_taskIL_tta.sh
    done
done

echo
echo "[INFO] completed Task-IL entropy-threshold ablation"
echo "[INFO] summarize with:"
echo "  python tools/summarize_entropy_threshold_ablation.py --root $LOG_BASE"
