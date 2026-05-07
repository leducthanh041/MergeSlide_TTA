#!/bin/bash
#SBATCH --job-name=TGCT
#SBATCH --output=/datastore/uittogether2/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/TGCT_%j.out
#SBATCH --error=/datastore/uittogether2/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/TGCT_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=mps:l40:2
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM=15000

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" 2>/dev/null || true
    fi
    if [ -n "${CUDA_MPS_LOG_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

module clear -f
module load slurm/slurm/24.11
# T?m th?i KHÔNG load cuda toolkit d? tránh xung d?t lib
# module load cuda12.8/toolkit/12.8.1

source /datastore/uittogether2/tools/miniconda3/etc/profile.d/conda.sh

# ---- Fix l?i conda activate + cuda-nvcc hook du?i ch? d? set -u ----
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate /datastore/uittogether2/tools/miniconda3/envs/trident
set -u
# -------------------------------------------------------------------

# ================= GPU CHECK =================
# V?i cluster này, workflow MPS th?c t? ?n d?nh là:
# 1) b? CUDA_VISIBLE_DEVICES hi?n t?i
# 2) g?i gpu_check.sh
# 3) ép l?i CUDA_VISIBLE_DEVICES theo BEST_GPU v?t lý
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

echo "[INFO] gpu_check exit_code=$EXIT_CODE"
echo "[INFO] gpu_check output=$CHECK_OUT"

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh returned unexpected exit code: $EXIT_CODE"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] BEST_GPU=$BEST_GPU"

# ================= MPS SETUP =================
# Bám theo workflow dã test ch?y ?n
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"


# ================= RUN =================
echo "[INFO] launching training at $(date)"

python -u TRIDENT/run_batch_of_slides.py --task all --wsi_dir /datastore/uittogether2/LuuTru/Thanhld/WSI/dataset/TCGA-TGCT/ --job_dir /datastore/uittogether2/LuuTru/Thanhld/WSI/dataset/TCGA-TGCT/preprocessed/ --patch_encoder conch_v15 --mag 10 --patch_size 256