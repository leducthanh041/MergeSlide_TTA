# test_taskIL_tta.py
"""
Task-IL TTA evaluation (upper bound with adaptation).

Task identity is known at inference time, so:
  - No task routing (task_prompts not used in loss)
  - Loss = class-level entropy + diversity over C_task classes only
  - This is the cleanest TTA setup: pure Information Maximization on correct task

Usage:
    python test_taskIL_tta.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import EMBED_DIM, K_PATCHES, NUM_TASKS, TITAN_PS_ARG
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import pad_numpy_arrays
from mergeslide_tta.utils import get_eval_metrics, seed_torch
from mergeslide_tta.tta_adapter import MergeSlide_TTA, load_task_weights

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
        repo_path  = PROJECT_ROOT / name
        local_path = local_root / name
        if not repo_path.exists() and not repo_path.is_symlink():
            repo_path.symlink_to(local_path, target_is_directory=True)
    os.environ.setdefault("TMPDIR",                str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR",         str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return local_root


def resolve_hot_path(path: str, local_root: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        parts = raw.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return local_root.joinpath(*parts)
        return raw
    try:
        relative = raw.relative_to(PROJECT_ROOT)
    except ValueError:
        return raw
    parts = relative.parts
    if parts and parts[0] in HOT_DIR_NAMES:
        return local_root.joinpath(*parts)
    return raw


def eval_task_taskil_tta(
    test_loader,
    task_id:     int,
    tta_model:   MergeSlide_TTA,
    device,
    verbose_loss: bool = False,
) -> tuple:
    """
    Task-IL TTA inference for 1 task.
    task_id is known -> tta_model uses mode='task_il' with fixed_task_id=task_id.
    pred_class is LOCAL (0..C_task-1), targets are LOCAL.
    """
    preds_all   = []
    probs_all   = []
    targets_all = []
    loss_logs   = []

    for features, coords, label in tqdm(test_loader, leave=False):
        features = features.to(device)
        coords   = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        pred_class, probs, _, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        if verbose_loss:
            loss_logs.append(adapt_log)

        preds_all.append(np.array([pred_class]))
        probs_all.append(probs.numpy())
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

    if verbose_loss and loss_logs:
        adapted = [d for d in loss_logs if d.get("slide/adapted")]
        if adapted:
            mean_loss = np.mean([d.get("loss/total_with_reg", 0) for d in adapted])
            print(f"    [TTA] task={task_id} adapted={len(adapted)}/{len(loss_logs)} "
                  f"mean_loss={mean_loss:.4f}")

    return metrics, preds_arr, targets_arr


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Task-IL TTA evaluation (upper bound)")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)
    # TTA hyperparams
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--beta",              type=float, default=1.0)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--entropy_threshold", type=float, default=0.4)
    parser.add_argument("--episodic",          action="store_true")
    parser.add_argument("--verbose_loss",      action="store_true")
    # Note: --alpha not exposed for task_il (always 0.0 internally)
    args = parser.parse_args()

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    # task_prompts still needed for _quick_inference backbone forward
    # (task_il mode overrides pred_task but backbone still needs ps)
    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    if getattr(cfg.dataset, "order", "forward") == "reverse":
        task_prompts = task_prompts.flip(0)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_baccs    = []
    overall_accs     = []
    all_acc_per_task = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        merge_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"\nLoading: {merge_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_path), map_location="cpu")
        )

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]
        task_weights = load_task_weights(task_model_paths, device)

        all_baccs    = []
        all_accs     = []
        acc_per_task = {}

        for task_id in range(num_tasks):
            # Build TTA model with fixed_task_id for this task
            tta_model = MergeSlide_TTA(
                backbone          = base_model.vision_encoder,
                task_prompts      = task_prompts,
                task_weights      = task_weights,
                num_classes       = seq_dataset.num_classes,
                device            = device,
                mode              = "task_il",
                fixed_task_id     = task_id,
                M                 = args.M,
                K_sub             = args.K_sub,
                top_ratio         = args.top_ratio,
                alpha             = 0.0,   # always 0 for task_il
                beta              = args.beta,
                lr                = args.lr,
                n_steps           = args.n_steps,
                episodic          = args.episodic,
                entropy_threshold = args.entropy_threshold,
            )

            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            results, preds_all, targets_all = eval_task_taskil_tta(
                test_loader  = test_loader,
                task_id      = task_id,
                tta_model    = tta_model,
                device       = device,
                verbose_loss = args.verbose_loss,
            )

            bacc = balanced_accuracy_score(targets_all, preds_all)
            acc  = sum(preds_all == targets_all) / len(test_loader)
            acc_per_task[task_id] = acc
            all_baccs.append(bacc)
            all_accs.append(acc)

            n_adapted = tta_model.n_adapted
            n_total   = n_adapted + tta_model.n_skipped
            print(f"  Fold {fold_id} | {seq_dataset.task_names[task_id]}: "
                  f"BAcc={bacc*100:.4f}% Acc={acc*100:.4f}% "
                  f"adapted={n_adapted}/{n_total}")

            tta_model.hard_reset()

        overall_baccs.append(np.mean(all_baccs))
        overall_accs.append(np.mean(all_accs))
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] BAcc={np.mean(all_baccs)*100:.4f}% "
              f"Acc={np.mean(all_accs)*100:.4f}%")

    print("\n===== Task-IL TTA Results =====")
    print(f"Balanced Acc: {np.mean(overall_baccs)*100:.4f}%"
          f" ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Accuracy:     {np.mean(overall_accs)*100:.4f}%"
          f" ({np.std(overall_accs)*100:.4f}%)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  {seq_dataset.task_names[t]}: {np.mean(accs[t])*100:.4f}%"
              f" ({np.std(accs[t])*100:.4f}%)")
