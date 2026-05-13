# evaluate.py
"""
Unified evaluation script for MergeSlide.

Modes:
  class_il     : Class-IL (Table 2 in paper) — BAcc, Macro F1, Weighted F1
  class_il_bwt : Class-IL + Forgetting / BWT sau mỗi task
  task_il      : Task-IL upper bound (dùng ground-truth task_id)

Usage:
  python evaluate.py --mode class_il     --config configs/default.yaml
  python evaluate.py --mode class_il_bwt --config configs/default.yaml
  python evaluate.py --mode task_il      --config configs/default.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_CLASSES, NUM_TASKS,
    TASK_NAMES, TASK_TO_GLOBAL_CLASS, TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import backward_transfer, forgetting, pad_numpy_arrays
from mergeslide_tta.model import CustomSequential
from mergeslide_tta.utils import get_eval_metrics, seed_torch


# ---------------------------------------------------------------------------
# Checkpoint path helpers — khớp với merge.py
# ---------------------------------------------------------------------------

def get_final_ckpt_path(merged_dir: str, fold_id: int) -> Path:
    """
    Path của final merged checkpoint (sau khi merge xong tất cả NUM_TASKS task).
    Khớp với: torch.save(..., output_dir / "merged_final.pth") trong merge.py
    """
    return Path(merged_dir) / f"fold_{fold_id}" / "merged_final.pth"


def get_intermediate_ckpt_path(merged_dir: str, fold_id: int,
                                task_idx: int) -> Path:
    """
    Path của intermediate checkpoint sau khi merge xong task_idx.
    task_idx bắt đầu từ 1 (sau khi merge task_1 vào task_0).
    Khớp với: torch.save(..., output_dir / f"merged_task_{model_idx}.pth") trong merge.py
    """
    return Path(merged_dir) / f"fold_{fold_id}" / f"merged_task_{task_idx}.pth"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_merged_model(ckpt_path: Path, base_model: nn.Module,
                      device: str) -> CustomSequential:
    """
    Load merged vision encoder weights vào CustomSequential với Identity head.
    base_model được modify in-place — gọi hàm này trong vòng lặp fold sẽ
    overwrite weights mỗi fold, không cần reload base_model từ HuggingFace.
    """
    merged_weights = torch.load(str(ckpt_path), map_location="cpu")
    base_model.vision_encoder.load_state_dict(merged_weights, strict=True)
    model = CustomSequential(base_model, nn.Identity())
    return model.eval().to(device)


def load_task_mlp_weights(finetuned_dir: str, fold_id: int) -> list[dict]:
    """
    Load MLP head (weight + bias) từ mỗi per-task finetuned checkpoint.
    Trả về list có NUM_TASKS phần tử, mỗi phần tử là state_dict của MLP.
    """
    weights = []
    for t in range(NUM_TASKS):
        path  = Path(finetuned_dir) / f"fold_{fold_id}" / f"task_{t}.pt"
        state = torch.load(str(path), map_location="cpu")
        # 2 key cuối của checkpoint là mlp.weight và mlp.bias
        mlp_state = {
            k.split("mlp.")[-1]: state[k]
            for k in list(state.keys())[-2:]
        }
        weights.append(mlp_state)
    return weights


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_class_il_inference(
    model: CustomSequential,
    test_loader,
    task_id: int,
    task_prompts: torch.Tensor,
    mlp_weights: list[dict],
    device: str,
    use_tcp: bool = True,
) -> dict:
    """
    Class-IL inference cho 1 task:
      1. Lấy slide embedding từ merged vision encoder.
      2. Predict task_id bằng cosine similarity với task_prompts.
      3. Apply MLP head của predicted task.

    Returns dict với keys: preds, targets, global_preds, global_targets, metrics.
    """
    ps          = torch.tensor(TITAN_PS_ARG).int().to(device)
    preds_all   = []
    targets_all = []
    probs_all   = []
    g_preds     = []
    g_targets   = []

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in test_loader:
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            slide_embed = model.backbone(features, coords, ps)

            if use_tcp:
                # Routing qua task_prompts → MLP head tương ứng
                pred_task_id = int(torch.argmax(slide_embed @ task_prompts.T))
                mlp = nn.Linear(EMBED_DIM, NUM_CLASSES[pred_task_id]).to(device)
                mlp.load_state_dict(mlp_weights[pred_task_id])
                logits = mlp(slide_embed).float()
            else:
                # Naive: dot-product với toàn bộ 13 class prompts
                logits = (slide_embed @ all_class_prompts.T).float()  # [1, 13]


            probs  = nn.functional.softmax(logits, dim=1)
            pred   = int(logits.argmax(1))

            preds_all.append(pred)
            targets_all.append(int(label))
            probs_all.append(probs.cpu().numpy())

            # Map local class idx → global class idx (0–12)
            g_preds.append(TASK_TO_GLOBAL_CLASS[task_id].get(pred, -1))
            g_targets.append(TASK_TO_GLOBAL_CLASS[task_id].get(int(label), -1))

    preds_arr   = np.array(preds_all)
    targets_arr = np.array(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    roc_kwargs = ({"multi_class": "ovo", "average": "macro"}
                  if NUM_CLASSES[task_id] > 2 else {})
    metrics = get_eval_metrics(targets_arr, preds_arr, probs_arr,
                               roc_kwargs=roc_kwargs, prefix="")

    return {
        "preds":          preds_arr,
        "targets":        targets_arr,
        "global_preds":   np.array(g_preds),
        "global_targets": np.array(g_targets),
        "metrics":        metrics,
    }


@torch.no_grad()
def run_task_il_inference(
    base_model: nn.Module,
    test_loader,
    task_id: int,
    mlp_weights: list[dict],
    device: str,
) -> dict:
    """
    Task-IL inference: ground-truth task_id được cung cấp sẵn.
    Gắn MLP head đúng task vào backbone rồi forward thẳng —
    không cần task routing qua task_prompts.
    """
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    # Gắn MLP head của đúng task vào backbone — đúng theo code gốc test_taskIL.py
    mlp = nn.Linear(EMBED_DIM, NUM_CLASSES[task_id]).to(device)
    mlp.weight.data.normal_(mean=0.0, std=0.01)
    mlp.bias.data.zero_()
    model = CustomSequential(base_model, mlp)
    model.mlp.load_state_dict(mlp_weights[task_id])
    model.eval()

    preds_all   = []
    targets_all = []
    probs_all   = []

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in test_loader:
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            logits = model(features, coords, ps).float()

            if NUM_CLASSES[task_id] == 2:
                probs      = nn.functional.softmax(logits, dim=1)[:, 1]
                roc_kwargs = {}
            else:
                probs      = nn.functional.softmax(logits, dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(label.numpy())
            probs_all.append(probs.cpu().numpy())

    preds_arr   = np.concatenate(preds_all)
    targets_arr = np.concatenate(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    metrics = get_eval_metrics(targets_arr, preds_arr, probs_arr,
                               roc_kwargs=roc_kwargs, prefix="")
    return {
        "preds":   preds_arr,
        "targets": targets_arr,
        "bacc":    balanced_accuracy_score(targets_arr, preds_arr),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def eval_class_il(cfg) -> None:
    """Class-IL evaluation — reproduce Table 2 trong paper."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    task_prompts = torch.load(cfg.paths.task_prompts).to(device)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained(
        "MahmoodLab/TITAN", trust_remote_code=True
    ).to(device)

    all_baccs, all_macro_f1s, all_weighted_f1s = [], [], []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        # ← dùng get_final_ckpt_path — khớp với merge.py
        ckpt_path = get_final_ckpt_path(cfg.paths.merged_checkpoints, fold_id)
        model       = load_merged_model(ckpt_path, base_model, device)
        mlp_weights = load_task_mlp_weights(cfg.paths.finetuned_checkpoints, fold_id)

        fold_baccs, all_gp, all_gt = [], [], []

        for task_id in range(NUM_TASKS):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            result = run_class_il_inference(
                model, test_loader, task_id, task_prompts, mlp_weights, device
            )
            fold_baccs.append(
                balanced_accuracy_score(result["targets"], result["preds"])
            )
            all_gp.append(result["global_preds"])
            all_gt.append(result["global_targets"])

        all_gp = np.concatenate(all_gp)
        all_gt = np.concatenate(all_gt)

        all_baccs.append(np.mean(fold_baccs))
        all_macro_f1s.append(f1_score(all_gt, all_gp, average="macro"))
        all_weighted_f1s.append(f1_score(all_gt, all_gp, average="weighted"))

    print("\n===== Class-IL Results =====")
    print(f"BAcc:        {np.mean(all_baccs):.4f} ± {np.std(all_baccs):.4f}")
    print(f"Macro F1:    {np.mean(all_macro_f1s):.4f} ± {np.std(all_macro_f1s):.4f}")
    print(f"Weighted F1: {np.mean(all_weighted_f1s):.4f} ± {np.std(all_weighted_f1s):.4f}")


def eval_class_il_bwt(cfg) -> None:
    """
    Class-IL + BWT/Forgetting evaluation.
    Sau mỗi task t, load intermediate checkpoint merged_task_{t}.pth
    và đánh giá lại tất cả task 0..t.
    """
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    task_prompts = torch.load(cfg.paths.task_prompts).to(device)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained(
        "MahmoodLab/TITAN", trust_remote_code=True
    ).to(device)

    # results[t][i] = BAcc của task i sau khi học đến task t
    all_fold_results = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        mlp_weights  = load_task_mlp_weights(cfg.paths.finetuned_checkpoints, fold_id)
        fold_results = []

        # Task 0 chưa có intermediate checkpoint (là starting point)
        # Intermediate task_1 = sau khi merge task_0 + task_1
        for merged_up_to in range(1, NUM_TASKS):
            # ← dùng get_intermediate_ckpt_path — khớp với merge.py
            ckpt_path = get_intermediate_ckpt_path(
                cfg.paths.merged_checkpoints, fold_id, merged_up_to
            )
            model = load_merged_model(ckpt_path, base_model, device)

            # Đánh giá tất cả task từ 0 đến merged_up_to
            row = []
            for task_id in range(merged_up_to + 1):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                result = run_class_il_inference(
                    model, test_loader, task_id, task_prompts, mlp_weights, device
                )
                row.append(balanced_accuracy_score(result["targets"], result["preds"]))
            fold_results.append(row)

        all_fold_results.append(fold_results)

    # Tính BWT và Forgetting trung bình qua các fold
    fold_bwts = [backward_transfer(r) for r in all_fold_results]
    fold_fgts = [forgetting(r)        for r in all_fold_results]

    print("\n===== Class-IL BWT Results =====")
    print(f"BWT:       {np.mean(fold_bwts):.4f} ± {np.std(fold_bwts):.4f}")
    print(f"Forgetting:{np.mean(fold_fgts):.4f} ± {np.std(fold_fgts):.4f}")


def eval_task_il(cfg) -> None:
    """Task-IL upper bound evaluation."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained(
        "MahmoodLab/TITAN", trust_remote_code=True
    ).to(device)

    all_baccs       = []
    all_acc_per_task = {task_id: [] for task_id in range(NUM_TASKS)}

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        ckpt_path = get_final_ckpt_path(cfg.paths.merged_checkpoints, fold_id)

        # Load merged vision encoder weights vào base_model
        merged_weights = torch.load(str(ckpt_path), map_location="cpu")
        base_model.vision_encoder.load_state_dict(merged_weights, strict=True)

        mlp_weights = load_task_mlp_weights(cfg.paths.finetuned_checkpoints, fold_id)

        fold_baccs = []
        for task_id in range(NUM_TASKS):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            # Truyền base_model thay vì CustomSequential — run_task_il_inference
            # tự build CustomSequential với đúng MLP head bên trong
            result = run_task_il_inference(
                base_model  = base_model,
                test_loader = test_loader,
                task_id     = task_id,
                mlp_weights = mlp_weights,
                device      = device,
            )

            fold_baccs.append(result["bacc"])
            all_acc_per_task[task_id].append(result["metrics"].get("/acc", 0.0))
            print(f"  Fold {fold_id} | {TASK_NAMES[task_id]}: "
                  f"BAcc={result['bacc']:.4f}")

        all_baccs.append(np.mean(fold_baccs))

    print("\n===== Task-IL Results =====")
    print(f"BAcc: {np.mean(all_baccs):.4f} ± {np.std(all_baccs):.4f}")

    # Per-task accuracy — giống format output code gốc
    for task_id in range(NUM_TASKS):
        accs = all_acc_per_task[task_id]
        print(f"  {TASK_NAMES[task_id]}: "
              f"Acc={np.mean(accs):.4f} ± {np.std(accs):.4f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MergeSlide Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode",   type=str, default="class_il",
                        choices=["class_il", "class_il_bwt", "task_il"])

    parser.add_argument("--tcp", action="store_true", default=True,
                    help="Dùng TCP inference (default: True). --no-tcp để chạy naive.")
    parser.add_argument("--no-tcp", dest="tcp", action="store_false")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    seed_torch(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg.training.seed,
    )

    if args.mode == "class_il":
        eval_class_il(cfg)
    elif args.mode == "class_il_bwt":
        eval_class_il_bwt(cfg)
    elif args.mode == "task_il":
        eval_task_il(cfg)