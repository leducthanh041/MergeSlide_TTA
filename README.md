# [WACV 2026] MergeSlide: Continual Model Merging and Task-to-Class Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images

<p align="center">
  <a href="https://arxiv.org/abs/2511.13099"><img src="https://img.shields.io/badge/arXiv-2511.13099-b31b1b.svg" alt="Arxiv"></a>
  <a href="https://wacv.thecvf.com/"><img src="https://img.shields.io/badge/WACV-2026-blue.svg" alt="WACV2026"></a>
</p>

> Doanh C. Bui (NAIST)*, Ba Hung Ngo (CNU), Hoai Luan Pham (NAIST), Khang Nguyen (UIT), Mai K. Nguyen (ETIS), Yasuhiko Nakashima (NAIST)

This branch is a cleaned MergeSlide WSI codebase with repo-local scripts for TITAN finetuning, OPCM merging, CLASS-IL/TASK-IL evaluation, and safer execution on the `/mmlab_students` NFS filesystem.

## 1. Runtime Setup

Install the Python stack from `requirements.txt`. The scripts default to the local environment:

```bash
/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10
```

TITAN is loaded with:

```python
AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
```

Make sure the environment has GPU access and permission to load TITAN from Hugging Face before running full training or evaluation.

## 2. Storage Layout

Use NFS only for source code and read-heavy datasets:

```text
/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA
/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/dataset
```

Use local SSD `/docker` for hot writes:

```text
/docker/data/$USER/MergeSlide_TTA/logs
/docker/data/$USER/MergeSlide_TTA/checkpoints
/docker/data/$USER/MergeSlide_TTA/checkpoints_ood
/docker/data/$USER/MergeSlide_TTA/sqlite
/docker/data/$USER/MergeSlide_TTA/tmp
```

The provided scripts create repo symlinks when missing:

```text
logs -> /docker/data/$USER/MergeSlide_TTA/logs
checkpoints -> /docker/data/$USER/MergeSlide_TTA/checkpoints
checkpoints_ood -> /docker/data/$USER/MergeSlide_TTA/checkpoints_ood
```

You can choose a dedicated log directory per run:

```bash
LOG_DIR=/docker/data/$USER/MergeSlide_TTA/logs/my_run bash scripts/test_classIL.sh
```

## 3. Project Structure

```text
MergeSlide_TTA/
├── train.py
├── merge.py
├── test_classIL_task_prompt.py
├── test_classIL_task_prompt_other_metrics.py
├── test_taskIL.py
├── task_prompts.pt
├── configs/
├── scripts/
├── tools/
│   └── run_classil_with_pt_features.py
└── mergeslide_tta/
    ├── constants.py
    ├── datasets.py
    ├── prompts_zeroshot.py
    ├── utils.py
    └── checkpoint_mirror.py
```

Important entrypoints:

- `train.py`: per-task TITAN finetuning. Saves `fold_{k}/task_{t}.pt`.
- `merge.py`: OPCM sequential model merging. Saves intermediate and final merged checkpoints.
- `test_classIL_task_prompt.py`: CLASS-IL final-task evaluation with `tcp` or `naive` mode.
- `test_classIL_task_prompt_other_metrics.py`: CLASS-IL continual metrics, including forgetting/BWT/FWT-style evaluation.
- `test_taskIL.py`: TASK-IL evaluation where the test task is known.
- `tools/run_classil_with_pt_features.py`: wrapper used by scripts to prefer `.pt` feature tensors, avoid problematic H5 feature reads when possible, force stable per-class metric shapes, and keep DataLoader multiprocessing disabled through `*_num_workers0.yaml` configs.

## 4. Dataset Configs

The current configs point to:

```text
/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/dataset
```

IND annotations:

```text
wsi_dataset_annotation/
```

OOD/cross-site annotations:

```text
wsi_dataset_annotation_cross_sites/
```

Main config files:

- `configs/default.yaml`: IND forward order, training-style `num_workers`.
- `configs/default_eval_num_workers0.yaml`: IND forward order, evaluation-safe `num_workers: 0`.
- `configs/default_reverse.yaml`: IND reverse order.
- `configs/default_reverse_eval_num_workers0.yaml`: IND reverse order, evaluation-safe `num_workers: 0`.
- `configs/default_ood.yaml`: OOD forward order.
- `configs/default_ood_eval_num_workers0.yaml`: OOD forward order, evaluation-safe `num_workers: 0`.

Task stream:

```text
BRCA -> RCC -> NSCLC -> ESCA -> TGCT -> CESC
```

Reverse configs invert this order internally.

## 5. Scripts

Run scripts from the repo root.

### Finetuning

```bash
bash scripts/finetune.sh
```

Defaults:

- entrypoint: `train.py`
- config: `configs/default_ood_eval_num_workers0.yaml`
- save dir: `/docker/data/$USER/MergeSlide_TTA/checkpoints_ood/finetuned`
- logs: `$LOG_DIR/result_train.log`, `$LOG_DIR/error_train.log`

Common overrides:

```bash
CONFIG=configs/default_eval_num_workers0.yaml \
SAVE_DIR=/docker/data/$USER/MergeSlide_TTA/checkpoints/finetuned \
FOLD_START=0 FOLD_END=10 \
bash scripts/finetune.sh
```

### Merging

```bash
bash scripts/mergemodel.sh
```

Defaults:

- entrypoint: `merge.py`
- config: `configs/default_ood_eval_num_workers0.yaml`
- finetuned dir: `/docker/data/$USER/MergeSlide_TTA/checkpoints_ood/finetuned`
- merged dir: `/docker/data/$USER/MergeSlide_TTA/checkpoints_ood/merged`
- logs: `$LOG_DIR/result_merge_ood.log`, `$LOG_DIR/error_merge_ood.log`

Common overrides:

```bash
CONFIG=configs/default_eval_num_workers0.yaml \
FINETUNED_DIR=/docker/data/$USER/MergeSlide_TTA/checkpoints/finetuned \
MERGED_DIR=/docker/data/$USER/MergeSlide_TTA/checkpoints/merged \
bash scripts/mergemodel.sh
```

### CLASS-IL Final Evaluation

```bash
LOG_DIR=/docker/data/$USER/MergeSlide_TTA/logs/classil_eval bash scripts/test_classIL.sh
```

Current script runs the enabled CLASS-IL commands in `scripts/test_classIL.sh`. It uses `tools/run_classil_with_pt_features.py`, writes logs into `$LOG_DIR`, and keeps hot writes on `/docker`.

If TCP runs are commented out in the script for a specific experiment, uncomment the corresponding `run_to_logs` block before launching.

### CLASS-IL Other Metrics

```bash
LOG_DIR=/docker/data/$USER/MergeSlide_TTA/logs/classil_other_metrics bash scripts/test_classIL_other_metrics.sh
```

This script wraps `test_classIL_task_prompt_other_metrics.py` through the PT-first wrapper and logs to `$LOG_DIR`.

### TASK-IL Evaluation

```bash
LOG_DIR=/docker/data/$USER/MergeSlide_TTA/logs/taskil_eval bash scripts/test_taskIL.sh
```

The current script defaults to OOD config/checkpoints:

```text
configs/default_ood_eval_num_workers0.yaml
./checkpoints_ood/finetuned
./checkpoints_ood/merged
```

Override these variables if you want IND checkpoints:

```bash
CONFIG_FORWARD=configs/default_eval_num_workers0.yaml \
bash scripts/test_taskIL.sh
```

### Diagnostic CLASS-IL TCP Scripts

Two diagnostic scripts may exist locally:

```bash
bash scripts/test_classIL_tcp_num_workers0.sh
bash scripts/test_classIL_tcp_pt_features_num_workers0.sh
```

Use the PT-first version when H5 feature reads on NFS hang. Both are lightweight launch wrappers, not separate methods.

## 6. Direct Python Commands

The scripts are preferred because they set log paths, local hot-write paths, HDF5 locking behavior, and `num_workers: 0` configs. If you run Python directly, use the wrapper for eval:

```bash
python -u tools/run_classil_with_pt_features.py \
  --config configs/default_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --mode tcp
```

Direct entrypoints are still valid when you explicitly want the raw behavior:

```bash
python train.py --config configs/default.yaml --save_dir ./checkpoints/finetuned
python merge.py --config configs/default.yaml --finetuned_checkpoints ./checkpoints/finetuned --merged_checkpoints ./checkpoints/merged
python test_taskIL.py --config configs/default_eval_num_workers0.yaml --save_dir ./checkpoints/finetuned --merge_model_path ./checkpoints/merged
```

## 7. Route Debugging

`test_classIL_task_prompt.py` supports TCP routing diagnostics:

```bash
python -u tools/run_classil_with_pt_features.py \
  --config configs/default_eval_num_workers0.yaml \
  --save_dir ./checkpoints/finetuned \
  --merge_model_path ./checkpoints/merged \
  --mode tcp \
  --debug_route \
  --debug_route_csv logs/debug_route_tcp.csv
```

The CSV records per-slide routing scores and top-k task predictions. It is useful for inspecting task-prompt routing failures such as CESC being routed to ESCA.

## 8. Validation

Use lightweight checks after code edits:

```bash
python -m py_compile train.py merge.py test_classIL_task_prompt.py test_classIL_task_prompt_other_metrics.py test_taskIL.py tools/run_classil_with_pt_features.py
python -m compileall -q mergeslide_tta
bash -n scripts/finetune.sh scripts/mergemodel.sh scripts/test_classIL.sh scripts/test_classIL_other_metrics.sh scripts/test_taskIL.sh
```

Do not run full training, merging, or evaluation without confirming GPU, TITAN/Hugging Face access, dataset paths, fold range, and checkpoint paths.

## 9. Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{
    bui2026merge,
    title={MergeSlide: Continual Model Merging and Task-to-Class Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images},
    author={Doanh C. Bui, Ba Hung Ngo, Hoai Luan Pham, Khang Nguyen, Mai K. Nguyen, Yasuhiko Nakashima},
    booktitle={The IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2026},
}
```

## 10. Acknowledgement

This project builds on ideas and code from:

- [TITAN](https://github.com/mahmoodlab/TITAN)
- [FusionBench](https://github.com/tanganke/fusion_bench)
- [CATE](https://github.com/HKU-MedAI/CATE)
