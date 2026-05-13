# test_classIL_task_prompt.py
"""
Class-IL evaluation với TCP inference (MergeSlide w/ TCP).
Metrics: Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall, AUC.

Usage:
    python test_classIL_task_prompt.py \
        --save_dir /path/to/finetuned_checkpoints \
        --merge_model_path /path/to/merged/checkpoints

Cấu trúc checkpoint kỳ vọng:
    Finetuned : {save_dir}/fold_{id}/task_{t}.pt
    Merged    : {merge_model_path}_fold_{id}/merged_final.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_CLASSES, NUM_TASKS,
    TASK_TO_GLOBAL_CLASS, TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import pad_numpy_arrays
from mergeslide_tta.model import CustomSequential
from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT = Path(__file__).resolve().parent


def eval_task(
    test_loader,
    task_id: int,
    model: CustomSequential,
    num_classes: list,
    task_prompts: torch.Tensor,
    task_model_paths: list,
    device: str,
) -> tuple:
    """
    TCP inference cho 1 task:
      1. Slide embedding từ merged backbone.
      2. Predict task_id qua task_prompts.
      3. Apply MLP head của predicted task.
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []

    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    # Pre-load tất cả MLP weights
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

            t0 = time.time()
            slide_embed  = model.backbone(features, coords, ps)
            pred_task_id = int(torch.argmax(slide_embed @ task_prompts.T))

            mlp = nn.Linear(EMBED_DIM, num_classes[pred_task_id]).to(device)
            mlp.load_state_dict(task_weights[pred_task_id])
            logits = mlp(slide_embed).float()
            pred   = int(logits.argmax(1))
            times.append(time.time() - t0)

            probs = nn.functional.softmax(logits, dim=1)
            preds_all.append(np.array([pred]))
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

            g_label = TASK_TO_GLOBAL_CLASS[task_id].get(int(label), -1)
            g_pred  = TASK_TO_GLOBAL_CLASS[task_id].get(pred, -1)
            convert_targets_all.append(np.array([g_label]))
            convert_preds_all.append(np.array([g_pred]))

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
    return (
        metrics, preds_arr, targets_arr, probs_arr,
        np.concatenate(convert_preds_all),
        np.concatenate(convert_targets_all),
        sum(times),
    )


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL evaluation w/ TCP")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Prefix thư mục merged: {prefix}_fold_{id}/merged_final.pth")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks    = cfg.training.num_tasks
    num_classes  = NUM_CLASSES
    seq_dataset  = Sequential_Generic_MIL_Dataset(cfg)
    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_accs         = []
    overall_baccs        = []
    overall_macro_f1s    = []
    overall_weighted_f1s = []
    overall_recalls      = []
    overall_precisions   = []
    overall_aucs         = []
    overall_times        = []
    all_acc_per_task     = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        # Merged checkpoint — prefix + fold name, đúng theo pattern code gốc
        merge_model_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"Loading: {merge_model_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_model_path), map_location="cpu")
        )
        model = CustomSequential(base_model, nn.Identity()).eval()

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]

        num_correct  = 0.0
        num_total    = 0.0
        all_baccs    = []
        all_accs     = []
        all_aucs     = []
        all_preds_g  = []
        all_targets_g = []
        acc_per_task = {}
        fold_time    = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            (results, preds_all, targets_all, probs_all,
             conv_preds, conv_targets, task_time) = eval_task(
                test_loader, task_id, model, num_classes,
                task_prompts, task_model_paths, device,
            )

            num_correct += sum(preds_all == targets_all)
            num_total   += len(test_loader)
            fold_time   += task_time / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)
            for i in range(num_classes[task_id]):
                all_aucs.append(
                    roc_auc_score((targets_all == i).astype(int), probs_all[:, i])
                )

        all_preds_g   = np.concatenate(all_preds_g)
        all_targets_g = np.concatenate(all_targets_g)

        overall_accs.append(np.mean(all_accs))
        overall_baccs.append(np.mean(all_baccs))
        overall_macro_f1s.append(f1_score(all_targets_g, all_preds_g, average="macro"))
        overall_weighted_f1s.append(f1_score(all_targets_g, all_preds_g, average="weighted"))
        overall_recalls.append(recall_score(all_targets_g, all_preds_g, average=None))
        overall_precisions.append(precision_score(all_targets_g, all_preds_g, average=None))
        overall_aucs.append(np.array(all_aucs))
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] Acc={np.mean(all_accs):.4f} BAcc={np.mean(all_baccs):.4f}")

    print("\n===== Class-IL w/ TCP Results =====")
    print(f"Accuracy:        {np.mean(overall_accs)*100:.4f}% ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:    {np.mean(overall_baccs)*100:.4f}% ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:        {np.mean(overall_macro_f1s)*100:.4f}% ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:     {np.mean(overall_weighted_f1s)*100:.4f}% ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time:  {np.mean(overall_times):.3f}s ({np.std(overall_times):.3f}s)")  # time giữ nguyên

    print("\nRecall per class:")
    for v, s in zip(np.mean(np.stack(overall_recalls), axis=0),
                    np.std(np.stack(overall_recalls), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nPrecision per class:")
    for v, s in zip(np.mean(np.stack(overall_precisions), axis=0),
                    np.std(np.stack(overall_precisions), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nAUC per class:")
    for v, s in zip(np.mean(np.stack(overall_aucs), axis=0),
                    np.std(np.stack(overall_aucs), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs[t])*100:.4f}% ({np.std(accs[t])*100:.4f}%)")
