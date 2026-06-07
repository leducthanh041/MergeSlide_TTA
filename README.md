# MergeSlide-TTA

MergeSlide-TTA is a test-time adaptation extension for MergeSlide-style continual learning on whole-slide images (WSIs). The current pipeline uses TITAN WSI representations, per-task finetuning, sequential model merging, SWAG-diagonal posterior estimation, and TTA inference for CLASS-IL and TASK-IL settings.

This repository is prepared for a MICCAI workshop submission. The method is not yet publicly released as a paper.

## Overview

The full workflow is:

```text
Preprocessed WSIs by TRIDENT
        |
        v
TITAN feature bags + coordinates
        |
        v
Per-task finetuning
        |
        v
Sequential model merging
        |
        v
SWAG-diagonal posterior estimation
        |
        v
Inference with MergeSlide-TTA
```

The base MergeSlide evaluation entrypoints are:

- `test_classIL_task_prompt.py`: CLASS-IL evaluation with TCP routing or naive global classification.
- `test_classIL_task_prompt_other_metrics.py`: CLASS-IL continual metrics such as BWT/FGT-style analysis.
- `test_taskIL.py`: TASK-IL evaluation where the task identity is known at inference time.

The TTA entrypoint is:

- `test_tta.py`: CLASS-IL TCP, CLASS-IL naive, and TASK-IL inference with test-time adaptation.

## Environment

The code has been tested on a Linux server with:

- Python 3.10
- CUDA-enabled PyTorch
- NVIDIA RTX 2080 Ti GPUs with 11 GB VRAM
- TITAN loaded from Hugging Face: `MahmoodLab/TITAN`

Create and activate a Python environment:

```bash
conda create -n mergeslide_tta python=3.10 -y
conda activate mergeslide_tta
```

Install PyTorch for your CUDA version first. For example, follow the official selector:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

TITAN is loaded with:

```python
AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
```

Make sure your environment can access the Hugging Face model before launching full training or evaluation.

## Preprocessed WSI Features

This code expects WSIs to be preprocessed before training. In our workflow, WSIs are processed with TRIDENT to obtain patch-level feature bags and patch coordinates.

At a high level, each slide should provide:

```text
features: [N, 768]
coords:   [N, 2]
```

The repository supports a PT-first feature loading wrapper:

```bash
tools/run_classil_with_pt_features.py
```

The wrapper uses `.pt` feature tensors when the patch count matches the coordinate file, and falls back to H5 features only when necessary. This is the recommended path for training, evaluation, and TTA scripts.

Dataset-specific annotation and feature locations are configured through YAML files under `configs/`. Adjust those config files to match your local preprocessed TRIDENT outputs.

## Checkpoints

The pipeline expects three checkpoint stages:

```text
finetuned/
merged/
swag_diagonal/
```

For convenience, pretrained checkpoints will be provided here:

```text
TODO: add Google Drive checkpoint link
```

Expected structure:

```text
checkpoints/
  finetuned/
  merged/
  swag_diagonal/

checkpoints_ood/
  finetuned/
  merged/
  swag_diagonal/
```

Reverse-order experiments use:

```text
checkpoints/
  finetuned_reverse/
  merged_reverse/
  swag_diagonal_reverse/
```

## Configs

Common configs:

- `configs/default_eval_num_workers0.yaml`: IND forward evaluation/training-safe config.
- `configs/default_reverse_eval_num_workers0.yaml`: IND reverse config.
- `configs/default_ood_eval_num_workers0.yaml`: OOD/cross-site config.
- `configs/default_tta_eval_num_workers0.yaml`: IND forward TTA config.
- `configs/default_tta_reverse_eval_num_workers0.yaml`: IND reverse TTA config.
- `configs/default_tta_ood_eval_num_workers0.yaml`: OOD/cross-site TTA config.

The `*_num_workers0.yaml` configs are recommended for reproducible runs and safer WSI feature loading.

## Base MergeSlide Pipeline

Run commands from the repository root.

### 1. Per-Task Finetuning

```bash
bash scripts/finetune.sh
```

Useful overrides:

```bash
CONFIG=configs/default_eval_num_workers0.yaml \
SAVE_DIR=./checkpoints/finetuned \
FOLD_START=0 FOLD_END=10 \
bash scripts/finetune.sh
```

For OOD/cross-site:

```bash
CONFIG=configs/default_ood_eval_num_workers0.yaml \
SAVE_DIR=./checkpoints_ood/finetuned \
bash scripts/finetune.sh
```

Main Python entrypoint:

```text
train.py
```

### 2. Sequential Model Merging

```bash
bash scripts/mergemodel.sh
```

Useful overrides:

```bash
CONFIG=configs/default_eval_num_workers0.yaml \
FINETUNED_DIR=./checkpoints/finetuned \
MERGED_DIR=./checkpoints/merged \
bash scripts/mergemodel.sh
```

For OOD/cross-site:

```bash
CONFIG=configs/default_ood_eval_num_workers0.yaml \
FINETUNED_DIR=./checkpoints_ood/finetuned \
MERGED_DIR=./checkpoints_ood/merged \
bash scripts/mergemodel.sh
```

Main Python entrypoint:

```text
merge.py
```

### 3. Base CLASS-IL Evaluation

```bash
bash scripts/test_classIL.sh
```

This script calls:

```text
test_classIL_task_prompt.py
```

Modes:

- `tcp`: task-to-class prompt routing.
- `naive`: global class prediction without TCP routing.

Direct command example:

```bash
python -u tools/run_classil_with_pt_features.py \
  --entrypoint test_classIL_task_prompt.py \
  --config configs/default_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --mode tcp
```

### 4. Base CLASS-IL Other Metrics

```bash
bash scripts/test_classIL_other_metrics.sh
```

This script calls:

```text
test_classIL_task_prompt_other_metrics.py
```

Use it after the base CLASS-IL evaluation when reporting continual-learning metrics beyond final-task accuracy.

### 5. Base TASK-IL Evaluation

```bash
bash scripts/test_taskIL.sh
```

This script calls:

```text
test_taskIL.py
```

TASK-IL assumes the task identity is known during inference.

## MergeSlide-TTA Pipeline

MergeSlide-TTA adds a SWAG-diagonal posterior and test-time adaptation stage after finetuning and merging.

### 1. Train SWAG-Diagonal Statistics

Run all supported settings:

```bash
bash scripts/train_swag.sh
```

Run one setting only:

```bash
SETTING=ind bash scripts/train_swag.sh
SETTING=reverse bash scripts/train_swag.sh
SETTING=ood bash scripts/train_swag.sh
```

Run a fold subset:

```bash
SETTING=ind FOLD_START=0 FOLD_END=5 bash scripts/train_swag.sh
```

Main Python entrypoint:

```text
train_swag.py
```

Outputs are expected under:

```text
./checkpoints/swag_diagonal
./checkpoints/swag_diagonal_reverse
./checkpoints_ood/swag_diagonal
```

### 2. TTA Inference: IND Forward

```bash
bash scripts/test_tta.sh
```

This script uses:

```text
test_tta.py
configs/default_tta_eval_num_workers0.yaml
./checkpoints/finetuned
./checkpoints/merged
./checkpoints/swag_diagonal
```

The script can run CLASS-IL TCP, CLASS-IL naive, and TASK-IL blocks. Enable or disable blocks directly in the script for the experiment you want to report.

### 3. TTA Inference: IND Reverse

```bash
bash scripts/test_tta_reverse.sh
```

This script uses:

```text
test_tta.py
configs/default_tta_reverse_eval_num_workers0.yaml
./checkpoints/finetuned_reverse
./checkpoints/merged_reverse
./checkpoints/swag_diagonal_reverse
```

### 4. TTA Inference: OOD/Cross-Site

```bash
bash scripts/test_tta_ood.sh
```

This script uses:

```text
test_tta.py
configs/default_tta_ood_eval_num_workers0.yaml
./checkpoints_ood/finetuned
./checkpoints_ood/merged
./checkpoints_ood/swag_diagonal
```

## Direct TTA Commands

CLASS-IL TCP:

```bash
python -u tools/run_classil_with_pt_features.py \
  --entrypoint test_tta.py \
  --config configs/default_tta_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --swag_dir ./checkpoints/swag_diagonal \
  --mode classil_tcp
```

CLASS-IL naive:

```bash
python -u tools/run_classil_with_pt_features.py \
  --entrypoint test_tta.py \
  --config configs/default_tta_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --swag_dir ./checkpoints/swag_diagonal \
  --mode classil_naive
```

TASK-IL:

```bash
python -u tools/run_classil_with_pt_features.py \
  --entrypoint test_tta.py \
  --config configs/default_tta_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --swag_dir ./checkpoints/swag_diagonal \
  --mode taskil
```

Useful flags:

- `--episodic`: reset the adapted model before each slide.
- `--no_reset_per_task`: do not reset adaptation state between tasks.
- `--fold_start` and `--fold_end`: run a subset of folds.

## Inference Modes

`test_tta.py` supports:

- `classil_tcp`: CLASS-IL inference with TCP routing over task prompts.
- `classil_naive`: CLASS-IL inference with the global class head.
- `taskil`: TASK-IL inference using the known task identity.

Base MergeSlide uses:

- `tcp` in `test_classIL_task_prompt.py`.
- `naive` in `test_classIL_task_prompt.py`.
- TASK-IL evaluation in `test_taskIL.py`.

## Hardware Notes

The current experiments were run on single-GPU jobs. The implementation is compatible with a single 11 GB GPU for evaluation/TTA in the tested setup, but full finetuning and SWAG estimation may require careful fold-wise scheduling.

Recommended runtime settings:

- Use batch size 1 for WSI bags.
- Use `num_workers: 0` for stable feature loading.
- Use `tools/run_classil_with_pt_features.py` for training/evaluation/TTA commands.
- Run fold subsets first before launching full 10-fold experiments.

## Repository Map

```text
configs/                         YAML configs for IND, reverse, OOD, and TTA
scripts/                         Bash launchers for finetune/merge/SWAG/eval/TTA
tools/run_classil_with_pt_features.py
                                  PT-first wrapper for robust feature loading
mergeslide_tta/                   Dataset, prompt, model, SWAG, and TTA utilities
train.py                          Per-task finetuning
merge.py                          Sequential model merging
train_swag.py                     SWAG-diagonal posterior estimation
test_tta.py                       MergeSlide-TTA inference
test_classIL_task_prompt.py       Base CLASS-IL inference
test_classIL_task_prompt_other_metrics.py
                                  Base CLASS-IL continual metrics
test_taskIL.py                    Base TASK-IL inference
task_prompts.pt                   Task prompt embeddings used by TCP routing
```

## Citation

This work is under submission. Citation information will be added after publication.
