# test_classIL_task_prompt_other_metrics.py
"""
Class-IL evaluation với TCP inference — BWT, Forgetting, mACC.

Với mỗi fold, load lần lượt intermediate merged checkpoints theo seq_task
và đánh giá tất cả task đã học.

Usage:
    python test_classIL_task_prompt_other_metrics.py \
        --save_dir /path/to/finetuned_checkpoints \
        --merge_model_path /path/to/merged/checkpoints

Cấu trúc checkpoint kỳ vọng:
    Finetuned    : {save_dir}/fold_{id}/task_{t}.pt
    Intermediate : {merge_model_path}_fold_{id}/merged_task_{seq_task}.pth
    Final        : {merge_model_path}_fold_{id}/merged_final.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_CLASSES, NUM_TASKS, TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import backward_transfer, forgetting, pad_numpy_arrays
from mergeslide_tta.model import CustomSequential
from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT = Path(__file__).resolve().parent


def eval_task(
    test_loader,
    model: CustomSequential,
    num_classes: list,
    task_prompts: torch.Tensor,
    task_model_paths: list,
    device: str,
) -> tuple:
    """TCP inference cho 1 task. Returns (metrics, preds_all, targets_all)."""
    preds_all   = []
    probs_all   = []
    targets_all = []

    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    task_weights = []
    for p in task_model_paths:
        state = torch.load(p, map_location="cpu")
        task_weights.append(
            {k.split("mlp.")[-1]: state[k] for k in list(state.keys())[-2:]}
        )

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            slide_embed  = model.backbone(features, coords, ps)
            pred_task_id = int(torch.argmax(slide_embed @ task_prompts.T))

            mlp = nn.Linear(EMBED_DIM, num_classes[pred_task_id]).to(device)
            mlp.load_state_dict(task_weights[pred_task_id])
            logits = mlp(slide_embed).float()
            pred   = int(logits.argmax(1))

            probs = nn.functional.softmax(logits, dim=1)
            preds_all.append(np.array([pred]))
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

    preds_arr   = np.concatenate(preds_all)
    targets_arr = np.concatenate(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    metrics = get_eval_metrics(
        targets_arr, preds_arr, probs_arr,
        roc_kwargs={"multi_class": "ovo", "average": "macro"},
        prefix="",
    )
    return metrics, preds_arr, targets_arr


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL BWT/FGT evaluation")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Prefix thư mục merged: {prefix}_fold_{id}/merged_task_{t}.pth")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks    = cfg.training.num_tasks
    num_classes  = NUM_CLASSES
    seq_dataset  = Sequential_Generic_MIL_Dataset(cfg)
    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)

    mACCs_all_folds        = []
    fgt_all_folds          = []
    bwt_all_folds          = []
    ACC_all_seqs_all_folds = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]

        # acc_per_task_all_seqs[t] = list BAcc của task 0..t
        acc_per_task_all_seqs = []
        ACC_all_seqs          = []

        for seq_task in tqdm(range(1, num_tasks + 1), desc="Seq tasks", leave=False):
            seed_torch(device, cfg.training.seed)

            # Intermediate checkpoint — đúng theo pattern code gốc:
            # {merge_model_path}_fold_{id}/merged_task_{seq_task-1}.pth
            # seq_task=1 → merged_task_1.pth (sau khi merge task_0+task_1)
            # seq_task=6 → merged_final.pth  (sau khi merge tất cả)
            if seq_task < num_tasks:
                ckpt_name = f"merged_task_{seq_task}.pth"
            else:
                ckpt_name = "merged_final.pth"

            merge_model_path = Path(args.merge_model_path) / fold / ckpt_name
            # print(merge_model_path)

            base_model = AutoModel.from_pretrained(
                "MahmoodLab/TITAN", trust_remote_code=True
            ).to(device)
            base_model.vision_encoder.load_state_dict(
                torch.load(str(merge_model_path), map_location="cpu")
            )
            model = CustomSequential(base_model, nn.Identity()).eval()

            num_correct  = 0.0
            num_total    = 0.0
            acc_per_task = []

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                _, preds_all, targets_all = eval_task(
                    test_loader, model,
                    num_classes[:seq_task],
                    task_prompts[:seq_task],
                    task_model_paths[:seq_task],
                    device,
                )
                num_correct += sum(preds_all == targets_all)
                num_total   += len(test_loader)
                acc_per_task.append(
                    sum(preds_all == targets_all) / len(targets_all)
                )

            ACC_all_seqs.append(float(num_correct / num_total))
            acc_per_task_all_seqs.append(acc_per_task)

            del base_model, model
            torch.cuda.empty_cache()

        mACC = np.mean(ACC_all_seqs)
        fgt  = forgetting(acc_per_task_all_seqs)
        bwt  = backward_transfer(acc_per_task_all_seqs)

        ACC_all_seqs_all_folds.append(ACC_all_seqs)
        mACCs_all_folds.append(mACC)
        fgt_all_folds.append(fgt)
        bwt_all_folds.append(bwt)

        print(f"[Fold {fold_id}] mACC={mACC*100:.4f}% FGT={fgt*100:.4f}% BWT={bwt*100:.4f}%")

        print(f"mACC: {np.mean(mACCs_all_folds)*100:.4f}% ({np.std(mACCs_all_folds)*100:.4f}%)")
        print(f"BWT:  {np.mean(bwt_all_folds)*100:.4f}% ({np.std(bwt_all_folds)*100:.4f}%)")
        print(f"FGT:  {np.mean(fgt_all_folds)*100:.4f}% ({np.std(fgt_all_folds)*100:.4f}%)")

        print("\nACC per seq task (mean across folds):")
        acc_seq_arr = np.array(ACC_all_seqs_all_folds)
        for t in range(num_tasks):
            print(f"  After task {t+1}: {np.mean(acc_seq_arr[:, t])*100:.4f}% "
                f"({np.std(acc_seq_arr[:, t])*100:.4f}%)")
