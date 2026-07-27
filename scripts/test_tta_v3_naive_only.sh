#!/bin/bash
#
# Chạy RIÊNG nhánh classil_naive của engine v3 GỐC (tta_engine_v3.py,
# use_tcp_gate=False, KHÔNG Module F) — để có baseline đối chiếu với
# core_naive (Module F) qua scripts/compare_naive_vs_core.sh.
#
# File này KHÔNG sửa scripts/test_tta_v3.sh hay test_tta_v3.py — chỉ gọi
# lại đúng lệnh đã có sẵn (nhưng đang bị comment) trong test_tta_v3.sh,
# dưới dạng một script độc lập mới.
#
#SBATCH --job-name=test_tta_v3_naive_only
#SBATCH --output=logs/test_tta_v3_naive_only_%j.out
#SBATCH --error=logs/test_tta_v3_naive_only_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-logs}"

CONFIG="${CONFIG:-configs/default_tta_eval_num_workers0.yaml}"
FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
MERGED_DIR="${MERGED_DIR:-./checkpoints/merged}"
SWAG_DIR="${SWAG_DIR:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA/checkpoints/swag_diagonal}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints; do
    repo_path="$PROJECT_ROOT/$name"
    local_path="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$repo_path" ]; then
        :
    elif [ -e "$repo_path" ]; then
        echo "[WARN] $repo_path is not a symlink; hot writes should use $local_path"
    else
        ln -s "$local_path" "$repo_path"
    fi
done

mkdir -p "$LOG_DIR" "$LOG_DIR/test_new_run"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "[INFO] start at $(date)"
echo "[INFO] MergeSlide-TTA v3 — classil_naive ONLY (baseline doi chieu Module F)"
echo "[INFO] CONFIG=$CONFIG"

"$PYTHON_BIN" -u tools/run_classil_with_pt_features.py \
    --entrypoint    test_tta_v3.py \
    --config        "$CONFIG" \
    --save_dir      "$FINETUNED_DIR" \
    --merge_model_path "$MERGED_DIR" \
    --swag_dir      "$SWAG_DIR" \
    --mode          classil_naive \
    --result_csv    "$LOG_DIR/tta_v3_results_classil_naive.csv" \
    --efficiency_json "$LOG_DIR/efficiency_tta_v3_classil_naive.json" \
    > "$LOG_DIR/test_new_run/result_tta_v3_classil_naive_only.log" \
    2> "$LOG_DIR/test_new_run/error_tta_v3_classil_naive_only.log"

echo "[INFO] finished at $(date)"
echo "[INFO] Ket qua -> $LOG_DIR/tta_v3_results_classil_naive.csv"
echo "[INFO] Dung file nay lam --naive_csv cho scripts/compare_naive_vs_core.sh"
