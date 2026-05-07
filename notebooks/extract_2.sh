#!/bin/bash
#SBATCH --job-name=NSCLC
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/NSCLC_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/NSCLC_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=mps:l40:2
#SBATCH --time=48:00:00

set -euo pipefail

REQUIRED_VRAM=15000

PROJECT_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks"
PREPATH_ROOT="${PROJECT_ROOT}/PrePATH"
PYTHON_BIN="/datastore/uittogether3/tools/miniconda3/envs/mergePre/bin/python"
CONDA_SH="/datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/datastore/uittogether3/tools/miniconda3/envs/mergePre"

# Notebook section 1/2/3 translated to batch-friendly variables for TCGA-NSCLC.
DATASET_NAME="TCGA-NSCLC"
TASK_NAME="TCGA-NSCLC_feats_conch15"
WSI_SOURCE_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/dataset/TCGA-NSCLC"
WSI_FORMAT="svs"
PATCH_LEVEL=1
PRESET_NAME="tcga.csv"
EXTRACT_MODEL="conch15"
EXTRACT_BATCH_SIZE=32

DOWNLOADED_ROOT="${PREPATH_ROOT}/downloaded_data/${DATASET_NAME}"
PATCH_ROOT="${DOWNLOADED_ROOT}/TCGA-NSCLC_patches"
PATCH_H5_DIR="${PATCH_ROOT}/patches"
MASK_DIR="${PATCH_ROOT}/masks"
STITCH_DIR="${PATCH_ROOT}/stitches"
FEATURE_ROOT="${DOWNLOADED_ROOT}/${TASK_NAME}"
FEATURE_PT_DIR="${FEATURE_ROOT}/pt_files/${EXTRACT_MODEL}"
FEATURE_H5_DIR="${FEATURE_ROOT}/h5_files/${EXTRACT_MODEL}"
CSV_ROOT="${PREPATH_ROOT}/csv"
CSV_TASK_ROOT="${CSV_ROOT}/${TASK_NAME}"
CSV_PATH="${CSV_TASK_ROOT}/part_0.csv"
PRESET_PATH="${PREPATH_ROOT}/presets/${PRESET_NAME}"
CKPT_PATH="${PREPATH_ROOT}/models/ckpts/conch1.5.bin"

# Set to "no" if you only want to rerun a later stage.
RUN_PATCHING="${RUN_PATCHING:-yes}"
RUN_GENERATE_CSV="${RUN_GENERATE_CSV:-yes}"
RUN_FEATURE_EXTRACT="${RUN_FEATURE_EXTRACT:-yes}"

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

log() {
    echo "[INFO] $*"
}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

run_cmd() {
    log "RUN: $*"
    "$@"
}

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

module clear -f
module load slurm/slurm/24.11

source "${CONDA_SH}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

set +u
conda activate "${CONDA_ENV}"
set -u

[ -d "${PREPATH_ROOT}" ] || fail "Missing PrePATH directory: ${PREPATH_ROOT}"
[ -d "${WSI_SOURCE_ROOT}" ] || fail "Missing TCGA-NSCLC WSI source: ${WSI_SOURCE_ROOT}"
[ -f "${PRESET_PATH}" ] || fail "Missing preset file: ${PRESET_PATH}"
[ -f "${CKPT_PATH}" ] || fail "Missing existing CONCH checkpoint: ${CKPT_PATH}"

cd "${PREPATH_ROOT}"

log "pwd=$(pwd)"
log "which python=$(which python)"
log "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys
import pandas
import torch
import timm
import h5py
import openslide

print(f"[INFO] sys.executable={sys.executable}")
print(f"[INFO] pandas={pandas.__version__}")
print(f"[INFO] torch={torch.__version__}")
print(f"[INFO] timm={timm.__version__}")
print(f"[INFO] h5py={h5py.__version__}")
print(f"[INFO] openslide={openslide.__version__}")
PY

if [ "${RUN_PATCHING}" = "yes" ]; then
    log "Notebook 2.1: create patches for ${DATASET_NAME}"
    run_cmd "${PYTHON_BIN}" create_patches_fp.py \
        --source "${WSI_SOURCE_ROOT}" \
        --save_dir "${PATCH_ROOT}" \
        --preset "${PRESET_NAME}" \
        --patch_level "${PATCH_LEVEL}" \
        --wsi_format "${WSI_FORMAT}" \
        --seg \
        --patch \
        --stitch
else
    log "Skip patching because RUN_PATCHING=${RUN_PATCHING}"
fi

log "Using existing CONCH checkpoint at ${CKPT_PATH}"

if [ "${RUN_GENERATE_CSV}" = "yes" ]; then
    log "Notebook 2.3: generate CSV from patch H5 files"
    run_cmd "${PYTHON_BIN}" scripts/extract_feature/generate_csv.py \
        --h5_dir "${PATCH_H5_DIR}" \
        --num 1 \
        --root "${CSV_TASK_ROOT}"
    [ -f "${CSV_PATH}" ] || fail "CSV was not generated: ${CSV_PATH}"
else
    log "Skip CSV generation because RUN_GENERATE_CSV=${RUN_GENERATE_CSV}"
fi

if [ "${RUN_FEATURE_EXTRACT}" != "yes" ]; then
    log "Skip feature extraction because RUN_FEATURE_EXTRACT=${RUN_FEATURE_EXTRACT}"
    exit 0
fi

unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "${REQUIRED_VRAM}" "${SLURM_JOB_ID}" 2>&1)
EXIT_CODE=$?
set -e

log "gpu_check exit_code=${EXIT_CODE}"
log "gpu_check output=${CHECK_OUT}"

if [ "${EXIT_CODE}" -eq 10 ]; then
    echo "${CHECK_OUT}"
    exit 0
elif [ "${EXIT_CODE}" -eq 11 ]; then
    echo "${CHECK_OUT}"
    exit 1
elif [ "${EXIT_CODE}" -ne 0 ]; then
    fail "gpu_check.sh returned unexpected exit code: ${EXIT_CODE}"
fi

BEST_GPU="${CHECK_OUT}"
log "BEST_GPU=${BEST_GPU}"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

log "Notebook 2.3: extract TITAN/CONCH features at $(date)"
run_cmd "${PYTHON_BIN}" -u extract_features_fp_fast.py \
    --model "${EXTRACT_MODEL}" \
    --csv_path "${CSV_PATH}" \
    --data_coors_dir "${PATCH_ROOT}" \
    --data_slide_dir "${WSI_SOURCE_ROOT}" \
    --feat_dir "${FEATURE_ROOT}" \
    --ignore_partial yes \
    --batch_size "${EXTRACT_BATCH_SIZE}" \
    --datatype auto \
    --slide_ext ".svs" \
    --save_storage "yes"

log "finish at $(date)"
log "feature output root=${FEATURE_ROOT}"
log "feature pt dir=${FEATURE_PT_DIR}"
log "feature h5 dir=${FEATURE_H5_DIR}"
