#!/bin/bash
#
# Run Class-IL prefix-merge TTA mACC/FGT/BWT ablation over adaptation steps.
#
# This uses test_classIL_tta_prefix_other_metrics.py through
# scripts/test_classIL_tta_prefix_other_metrics.sh, so the reported mACC/FGT/BWT
# follow the same prefix protocol as the baseline:
#   task_0.pt -> merged_task_1.pth -> ... -> merged_final.pth
#
# Examples:
#   SETTING=ind MODE=tcp bash scripts/test_classIL_tta_prefix_nsteps_ablation.sh
#   SETTING=ood MODE=all NSTEPS="2 3 8" bash scripts/test_classIL_tta_prefix_nsteps_ablation.sh
#   SETTING=all MODE=all CUDA_VISIBLE_DEVICES=0 bash scripts/test_classIL_tta_prefix_nsteps_ablation.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
cd "$PROJECT_ROOT"

SETTING="${SETTING:-all}"
ORDER="${ORDER:-forward}"
MODE="${MODE:-all}"
NSTEPS="${NSTEPS:-2 3 8}"
LOG_BASE="${LOG_BASE:-logs/ablation_nsteps_prefix}"
USE_BEST_CONFIG="${USE_BEST_CONFIG:-1}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-}"

case "$ORDER" in
    forward) ;;
    *) echo "[ERROR] Prefix n_steps ablation currently expects ORDER=forward, got ORDER=$ORDER" >&2; exit 1 ;;
esac

case "$SETTING" in
    ind) SETTINGS=("ind") ;;
    ood) SETTINGS=("ood") ;;
    all) SETTINGS=("ind" "ood") ;;
    *) echo "[ERROR] Unsupported SETTING=$SETTING (expected ind|ood|all)" >&2; exit 1 ;;
esac

case "$MODE" in
    tcp) MODES=("tcp") ;;
    naive) MODES=("naive") ;;
    all) MODES=("tcp" "naive") ;;
    *) echo "[ERROR] Unsupported MODE=$MODE (expected tcp|naive|all)" >&2; exit 1 ;;
esac

echo "[INFO] prefix n_steps ablation"
echo "[INFO] settings=${SETTINGS[*]} modes=${MODES[*]} nsteps=$NSTEPS"
echo "[INFO] log_base=$LOG_BASE use_best_config=$USE_BEST_CONFIG"

for setting_name in "${SETTINGS[@]}"; do
    for mode_name in "${MODES[@]}"; do
        for n_steps in $NSTEPS; do
            run_dir="$LOG_BASE/${setting_name}_forward/${mode_name}_n${n_steps}"
            echo
            echo "================================================================"
            echo "[INFO] SETTING=$setting_name MODE=$mode_name TTA_N_STEPS=$n_steps"
            echo "[INFO] LOG_DIR=$run_dir"
            echo "================================================================"

            env_args=(
                "SETTING=$setting_name"
                "ORDER=forward"
                "MODE=$mode_name"
                "USE_BEST_CONFIG=$USE_BEST_CONFIG"
                "TTA_N_STEPS=$n_steps"
                "LOG_DIR=$run_dir"
            )
            if [ -n "$FOLD_END" ]; then
                env_args+=("FOLD_END=$FOLD_END")
            fi

            env "${env_args[@]}" \
                FOLD_START="$FOLD_START" \
                bash scripts/test_classIL_tta_prefix_other_metrics.sh
        done
    done
done

echo
echo "[INFO] completed prefix n_steps ablation"
echo "[INFO] summarize with:"
echo "  python tools/summarize_prefix_nsteps_ablation.py --root $LOG_BASE"
