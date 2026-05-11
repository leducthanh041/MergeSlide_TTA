"""
Unified evaluation script for MergeSlide.

Modes:
  class_il       : Class-IL (Table 2 in paper) — Acc, BAcc, F1, AUC
  class_il_bwt   : Class-IL + Forgetting / BWT / FWT (requires intermediate checkpoints)
  task_il        : Task-IL upper bound

Usage:
  python evaluate.py --mode class_il --config configs/default.yaml
  python evaluate.py --mode class_il_bwt --config configs/default.yaml
  python evaluate.py --mode task_il --config configs/default.yaml
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

from mergeslide_tta.constants import (
    NUM_TASKS, NUM_CLASSES, TASK_CLASS_RANGES,
    TASK_TO_GLOBAL_CLASS, TASK_NAMES, K_PATCHES, TITAN_PS_ARG, EMBED_DIM,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import forgetting, backward_transfer, pad_numpy_arrays
from mergeslide_tta.model import CustomSequential, build_prompt_classifier
from mergeslide_tta.utils import seed_torch, get_eval_metrics
from transformers import AutoModel


def load_merged_model(merged_ckpt_path: str, base_model: nn.Module,
                      device: str) -> CustomSequential:
    """Load merged vision encoder weights into a CustomSequential with Identity head."""
    merged_weights = torch.load(merged_ckpt_path, map_location="cpu")
    base_model.vision_encoder.load_state_dict(merged_weights, strict=True)
    model = CustomSequential(base_model, nn.Identity())
    return model.eval().to(device)


def load_task_mlp_weights(ckpt_dir: str, fold_id: int,
                          num_tasks: int) -> list[dict]:
    """Load MLP head weights from per-task finetuned checkpoints."""
    weights = []
    for t in range(num_tasks):
        path = Path(ckpt_dir) / f"fold_{fold_id}" / f"task_{t}.pt"
        state = torch.load(path, map_location="cpu")
        # Extract only MLP keys (last 2 keys: weight + bias)
        mlp_state = {k.split("mlp.")[-1]: state[k]
                     for k in list(state.keys())[-2:]}
        weights.append(mlp_state)
    return weights


@torch.no_grad()
def run_class_il(model: CustomSequential, test_loader, task_id: int,
                 task_prompts: torch.Tensor, mlp_weights: list[dict],
                 device: str) -> dict:
    """
    Class-IL inference: predict task_id from slide embedding × task_prompts,
    then apply corresponding MLP head.
    """
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)
    preds_all, targets_all, probs_all = [], [], []
    convert_preds, convert_targets = [], []

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in test_loader:
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            slide_embed = model.backbone(features, coords, ps)

            # Task routing via prompt alignment
            pred_task = int(torch.argmax(slide_embed @ task_prompts.T))

            mlp = nn.Linear(EMBED_DIM, NUM_CLASSES[pred_task]).to(device)
            mlp.load_state_dict(mlp_weights[pred_task])
            logits = mlp(slide_embed).float()

            probs = nn.functional.softmax(logits, dim=1)
            pred  = logits.argmax(1)

            preds_all.append(pred.cpu().numpy())
            targets_all.append(label.numpy())
            probs_all.append(probs.cpu().numpy())

            # Convert to global class space for cross-task metrics
            g_label = TASK_TO_GLOBAL_CLASS[task_id].get(int(label), -1)
            g_pred  = TASK_TO_GLOBAL_CLASS[task_id].get(int(pred[0]), -1)
            convert_targets.append(g_label)
            convert_preds.append(g_pred)

    preds_all   = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    try:
        probs_all = np.concatenate(probs_all)
    except ValueError:
        probs_all = pad_numpy_arrays(probs_all)

    roc_kwargs = {"multi_class": "ovo", "average": "macro"} if NUM_CLASSES[task_id] > 2 else {}
    metrics = get_eval_metrics(targets_all, preds_all, probs_all,
                               roc_kwargs=roc_kwargs, prefix="")
    return {
        "metrics": metrics,
        "preds": preds_all,
        "targets": targets_all,
        "global_preds": np.array(convert_preds),
        "global_targets": np.array(convert_targets),
    }


def eval_class_il(cfg, args):
    """Run Class-IL evaluation across all folds and print summary."""
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    task_prompts_path = Path(cfg.paths.task_prompts)
    task_prompts = torch.load(task_prompts_path).to(device)

    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    all_baccs, all_accs, all_macro_f1s, all_weighted_f1s = [], [], [], []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        merged_ckpt = (Path(cfg.paths.merged_checkpoints)
                       / f"fold_{fold_id}"
                       / f"merged_weight_opcm_random_sampling_fold_{fold_id}_task_{NUM_TASKS}.pth")

        model = load_merged_model(str(merged_ckpt), base_model, device)
        mlp_weights = load_task_mlp_weights(cfg.paths.finetuned_checkpoints,
                                            fold_id, NUM_TASKS)

        fold_baccs, all_global_preds, all_global_targets = [], [], []

        for task_id in range(NUM_TASKS):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            result = run_class_il(model, test_loader, task_id, task_prompts,
                                  mlp_weights, device)
            fold_baccs.append(balanced_accuracy_score(result["targets"],
                                                       result["preds"]))
            all_global_preds.append(result["global_preds"])
            all_global_targets.append(result["global_targets"])

        all_gp = np.concatenate(all_global_preds)
        all_gt = np.concatenate(all_global_targets)
        all_baccs.append(np.mean(fold_baccs))
        all_macro_f1s.append(f1_score(all_gt, all_gp, average="macro"))
        all_weighted_f1s.append(f1_score(all_gt, all_gp, average="weighted"))

    print("\n===== Class-IL Results =====")
    print(f"BAcc:        {np.mean(all_baccs):.4f} ± {np.std(all_baccs):.4f}")
    print(f"Macro F1:    {np.mean(all_macro_f1s):.4f} ± {np.std(all_macro_f1s):.4f}")
    print(f"Weighted F1: {np.mean(all_weighted_f1s):.4f} ± {np.std(all_weighted_f1s):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default="class_il",
                        choices=["class_il", "class_il_bwt", "task_il"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    if args.mode == "class_il":
        eval_class_il(cfg, args)
    elif args.mode == "class_il_bwt":
        print("BWT mode: implement tiếp theo task_il, pattern tương tự")
        # TODO: gọi run_class_il với từng intermediate checkpoint
    elif args.mode == "task_il":
        print("Task-IL mode: implement theo test_taskIL.py")
        # TODO: tương tự class_il nhưng dùng ground-truth task_id