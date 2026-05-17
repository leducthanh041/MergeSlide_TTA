# test_taskIL.py
"""
Task-IL evaluation (upper bound) — ground-truth task_id được cung cấp sẵn.
Không cần task routing qua task_prompts.

Usage:
    python test_taskIL.py \
        --save_dir /path/to/finetuned_checkpoints \
        --merge_model_path /path/to/merged/checkpoints

Cấu trúc checkpoint kỳ vọng:
    Finetuned : {save_dir}/fold_{id}/task_{t}.pt
    Merged    : {merge_model_path}_fold_{id}/merged_final.pth
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_TASKS,
    TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import pad_numpy_arrays
from mergeslide_tta.model import CustomSequential
from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}


def get_local_hot_root() -> Path:
    user = os.environ.get("USER") or "thanhld"
    default_root = Path("/docker/data") / user / PROJECT_ROOT.name
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", default_root)).expanduser()


def ensure_local_hot_storage() -> Path:
    local_root = get_local_hot_root()
    local_root.mkdir(parents=True, exist_ok=True)
    for name in HOT_DIR_NAMES:
        (local_root / name).mkdir(parents=True, exist_ok=True)
    (local_root / "tmp").mkdir(parents=True, exist_ok=True)

    for name in ("logs", "checkpoints"):
        repo_path = PROJECT_ROOT / name
        local_path = local_root / name
        if repo_path.is_symlink():
            if repo_path.resolve() != local_path.resolve():
                print(f"[WARN] {repo_path} points to {repo_path.resolve()}, expected {local_path}")
        elif repo_path.exists():
            print(f"[WARN] {repo_path} is not a symlink; use {local_path} for hot-write data.")
        else:
            repo_path.symlink_to(local_path, target_is_directory=True)

    os.environ.setdefault("TMPDIR", str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR", str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return local_root


def resolve_hot_path(path: str, local_root: Path) -> Path:
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        parts = raw_path.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return local_root.joinpath(*parts)
        return raw_path

    try:
        relative = raw_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return raw_path

    parts = relative.parts
    if parts and parts[0] in HOT_DIR_NAMES:
        return local_root.joinpath(*parts)
    return raw_path


def eval_task(
    test_loader,
    model: CustomSequential,
    task_id: int,
    device: str,
    num_classes: list,
) -> tuple:
    """
    Task-IL inference: gọi thẳng model với MLP head đúng task đã gắn sẵn.
    Không có task routing — ground-truth task_id đã biết.
    """
    preds_all   = []
    probs_all   = []
    targets_all = []

    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            try:
                logits = model(features, coords, ps).float()
            except RuntimeError:
                # Fallback CPU nếu OOM — giữ đúng theo code gốc
                model.cpu()
                logits = model(
                    features.cpu(), coords.cpu(),
                    torch.tensor(TITAN_PS_ARG).int().cpu()
                ).float()
                model.to(device)

            pred = int(logits.argmax(1))

            if num_classes[task_id] == 2:
                probs      = nn.functional.softmax(logits, dim=1)[:, 1]
                roc_kwargs = {}
            else:
                probs      = nn.functional.softmax(logits, dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

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
        roc_kwargs=roc_kwargs, prefix="",
    )
    return metrics, preds_arr, targets_arr


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Task-IL evaluation (upper bound)")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Prefix thư mục merged: {prefix}_fold_{id}/merged_final.pth")
    args = parser.parse_args()

    local_hot_root = ensure_local_hot_storage()
    args.save_dir = str(resolve_hot_path(args.save_dir, local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    print(f"[INFO] Local hot storage root: {local_hot_root}")
    print(f"[INFO] Finetuned checkpoints: {args.save_dir}")
    print(f"[INFO] Merged checkpoints: {args.merge_model_path}")

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks    = cfg.training.num_tasks
    seq_dataset  = Sequential_Generic_MIL_Dataset(cfg)
    num_classes  = seq_dataset.num_classes

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_accs     = []
    overall_baccs    = []
    all_acc_per_task = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        # Merged checkpoint — prefix + fold name, đúng theo pattern code gốc
        merge_model_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"Loading: {merge_model_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_model_path), map_location="cpu")
        )

        all_baccs    = []
        all_accs     = []
        acc_per_task = {}
        num_correct  = 0.0
        num_total    = 0.0

        for task_id in range(num_tasks):
            # Gắn đúng MLP head của task này vào backbone
            task_ckpt = Path(args.save_dir) / fold / f"task_{task_id}.pt"
            state     = torch.load(str(task_ckpt), map_location="cpu")
            mlp_state = {
                k.split("mlp.")[-1]: state[k]
                for k in list(state.keys())[-2:]
            }

            mlp = nn.Linear(EMBED_DIM, num_classes[task_id]).to(device)
            mlp.weight.data.normal_(mean=0.0, std=0.01)
            mlp.bias.data.zero_()
            model = CustomSequential(base_model, mlp)
            model.mlp.load_state_dict(mlp_state)
            model.eval()

            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            results, preds_all, targets_all = eval_task(
                test_loader, model, task_id, device, num_classes
            )

            num_correct += sum(preds_all == targets_all)
            num_total   += len(test_loader)

            bacc = balanced_accuracy_score(targets_all, preds_all)
            acc  = sum(preds_all == targets_all) / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            all_baccs.append(bacc)
            all_accs.append(acc)

            print(f"  Fold {fold_id} | {seq_dataset.task_names[task_id]}: "
                  f"BAcc={bacc*100:.4f}% Acc={acc*100:.4f}%")

        overall_baccs.append(np.mean(all_baccs))
        overall_accs.append(np.mean(all_accs))
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] BAcc={np.mean(all_baccs):.4f} "
              f"Acc={np.mean(all_accs)*100:.4f}%")

    print("\n===== Task-IL Results =====")
    print(f"Balanced Acc: {np.mean(overall_baccs)*100:.4f}% ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Accuracy:     {np.mean(overall_accs)*100:.4f}% ({np.std(overall_accs)*100:.4f}%)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  {seq_dataset.task_names[t]}: {np.mean(accs[t])*100:.4f}% ({np.std(accs[t])*100:.4f}%)")
