#!/bin/bash
#SBATCH --job-name=CESC
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/CESC_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/CESC_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=mps:l40:2
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM="${REQUIRED_VRAM:-15000}"
PROJECT_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks"
CONDA_SH="/datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/datastore/uittogether3/tools/miniconda3/envs/trident"

WSI_DIR="${WSI_DIR:-/datastore/uittogether3/LuuTru/Thanhld/WSI/dataset/TCGA-CESC}"
JOB_DIR="${JOB_DIR:-${WSI_DIR}/preprocessed}"
CUSTOM_WSI_LIST="${CUSTOM_WSI_LIST:-${PROJECT_ROOT}/log/TCGA-CESC_trident_wsi_with_mpp.csv}"
SKIPPED_WSI_LIST="${SKIPPED_WSI_LIST:-${PROJECT_ROOT}/log/TCGA-CESC_trident_skipped_missing_mpp.csv}"

PATCH_ENCODER="${PATCH_ENCODER:-conch_v15}"
TARGET_MAG="${TARGET_MAG:-10}"
PATCH_SIZE="${PATCH_SIZE:-256}"
TRIDENT_TASK="${TRIDENT_TASK:-all}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=${rc} at $(date)"
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
echo "[INFO] WSI_DIR=${WSI_DIR}"
echo "[INFO] JOB_DIR=${JOB_DIR}"

module clear -f
module load slurm/slurm/24.11

source "${CONDA_SH}"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/log" "${JOB_DIR}"

unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "${REQUIRED_VRAM}" "${SLURM_JOB_ID}" 2>&1)
EXIT_CODE=$?
set -e

echo "[INFO] gpu_check exit_code=${EXIT_CODE}"
echo "[INFO] gpu_check output=${CHECK_OUT}"

if [ "${EXIT_CODE}" -eq 10 ]; then
    echo "${CHECK_OUT}"
    exit 0
elif [ "${EXIT_CODE}" -eq 11 ]; then
    echo "${CHECK_OUT}"
    exit 1
elif [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh returned unexpected exit code: ${EXIT_CODE}" >&2
    exit "${EXIT_CODE}"
fi

BEST_GPU="${CHECK_OUT}"
echo "[INFO] BEST_GPU=${BEST_GPU}"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

echo "[INFO] building TRIDENT WSI list with valid MPP metadata at $(date)"
python -u create_trident_mpp_csv.py \
    --wsi_dir "${WSI_DIR}" \
    --wsi_ext .svs \
    --output_csv "${CUSTOM_WSI_LIST}" \
    --skipped_csv "${SKIPPED_WSI_LIST}" \
    --progress_every "${PROGRESS_EVERY}"

echo "[INFO] custom_wsi_list=${CUSTOM_WSI_LIST}"
echo "[INFO] skipped_missing_mpp=${SKIPPED_WSI_LIST}"
echo "[INFO] launching TRIDENT at $(date)"

python -u TRIDENT/run_batch_of_slides.py \
    --task "${TRIDENT_TASK}" \
    --wsi_dir "${WSI_DIR}" \
    --wsi_ext .svs \
    --custom_list_of_wsis "${CUSTOM_WSI_LIST}" \
    --job_dir "${JOB_DIR}" \
    --patch_encoder "${PATCH_ENCODER}" \
    --mag "${TARGET_MAG}" \
    --patch_size "${PATCH_SIZE}"

echo "[INFO] finish at $(date)"
