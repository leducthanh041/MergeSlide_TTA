#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/extract_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/extract_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=mps:2
#SBATCH --time=48:00:00

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
# Tạm thời KHÔNG load cuda toolkit để tránh xung đột lib
# module load cuda12.8/toolkit/12.8.1

source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh

# ---- Fix lỗi conda activate + cuda-nvcc hook dưới chế độ set -u ----
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/mergePre
set -u
# -------------------------------------------------------------------

PROJECT_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks"
PREPATH_ROOT="${PROJECT_ROOT}/PrePATH"
PYTHON_BIN="/datastore/uittogether3/tools/miniconda3/envs/mergePre/bin/python"

cd "${PREPATH_ROOT}"

echo "[INFO] pwd=$(pwd)"
echo "[INFO] which python=$(which python)"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import sys, pandas; print('[INFO] sys.executable=' + sys.executable); print('[INFO] pandas=' + pandas.__version__)"

# ================= GPU CHECK =================
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
# Bám theo workflow đã test chạy ổn
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

# ================= RUN =================
echo "[INFO] launching training at $(date)"

${PYTHON_BIN} -u extract_features_fp_fast.py \
        --model "conch15" \
        --csv_path "${PREPATH_ROOT}/csv/CESC_feats_conch15/part_0.csv" \
        --data_coors_dir "${PREPATH_ROOT}/downloaded_data/CESC_patches" \
        --data_slide_dir "/datastore/uittogether3/LuuTru/Thanhld/WSI/dataset/CESC/gdc_manifest.2026-04-23.162946" \
        --feat_dir "${PREPATH_ROOT}/downloaded_data/CESC_feats_conch15" \
        --ignore_partial yes \
        --datatype auto \
        --slide_ext ".svs" \
        --save_storage "yes"

        # --batch_size 32 \

