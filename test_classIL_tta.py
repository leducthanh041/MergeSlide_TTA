# test_classIL_tta.py
"""
Class-IL TTA evaluation -- mirrors test_classIL_task_prompt.py.

Modes (same as original):
  tcp   (default): Task-to-Class Prompt-Aligned inference + TTA
  naive          : Direct class inference + TTA

TTA hyperparams:
  --M                  : sub-bags per slide (TTA batch size), default=8
  --K_sub              : patches per sub-bag, default=300
  --top_ratio          : confident sub-bag keep ratio, default=0.5
  --alpha              : task loss weight, default=0.5
  --beta               : L2 anchor weight, default=1.0
  --lr                 : LN optimizer learning rate, default=1e-4
  --n_steps            : adapt steps per slide, default=1
  --episodic           : flag -- reset LN after each slide (default=False = continual)
  --entropy_threshold  : only TTA when entropy >= threshold, default=0.4
                         Set 0.0 to TTA all slides.

Usage:
    python test_classIL_tta.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged \\
        --mode tcp
"""
import argparse
import os
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

from mergeslide_tta.constants import K_PATCHES, NUM_TASKS
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import pad_numpy_arrays
from mergeslide_tta.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts,
)
from mergeslide_tta.utils import get_eval_metrics, seed_torch
from mergeslide_tta.tta_adapter import MergeSlide_TTA, load_task_weights

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}

_PROMPT_FN_MAP = {
    "BRCA":  brca_prompts,
    "RCC":   rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA":  esca_prompts,
    "TGCT":  tgct_prompts,
    "CESC":  cesc_prompts,
}


# ---------------------------------------------------------------------------
# Path helpers -- identical to test_classIL_task_prompt.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build all_class_embeddings (naive mode only)
# ---------------------------------------------------------------------------

def build_class_embeddings(device, task_names: list) -> torch.Tensor:
    """[768, C_total] -- identical logic to test_classIL_task_prompt.py."""
    print("Building all_class_embeddings for naive mode ...")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)
    _, templates = brca_prompts()
    all_prompts = []
    for name in task_names:
        class_prompts, _ = _PROMPT_FN_MAP[name]()
        all_prompts.extend(class_prompts)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = titan.zero_shot_classifier(
            all_prompts, templates, device=str(device)
        )
    del titan
    torch.cuda.empty_cache()
    return classifier.to(device)


# ---------------------------------------------------------------------------
# TTA inference loop for 1 task
# ---------------------------------------------------------------------------

def eval_task_tta(
    test_loader,
    task_id:              int,
    tta_model:            MergeSlide_TTA,
    task_to_global_class: dict,
    device,
    mode:                 str  = "tcp",
    num_classes_per_task: list = None,
    verbose_loss:         bool = False,
) -> tuple:
    """
    TTA inference for 1 task. Each slide: adapt_and_predict().

    TCP mode : pred_class is LOCAL index (0..C_task-1), same as original eval_task_tcp.
    Naive mode: pred_class is COLUMN index of all_class_embeddings (= global index 0..12),
                same as original eval_task_naive. targets also converted to global space.
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []
    loss_logs           = []

    # Naive mode: build column_to_global mapping (mirrors original eval_task_naive)
    if mode == "naive":
        column_to_global = np.array([
            task_to_global_class[t][local]
            for t in range(len(task_to_global_class))
            for local in sorted(task_to_global_class[t].keys())
        ])
        total_classes = len(column_to_global)
    else:
        column_to_global = None
        total_classes    = None

    for features, coords, label in tqdm(test_loader, leave=False):
        features = features.to(device)
        coords   = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        t0 = time.time()
        pred_class, probs, pred_task, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        times.append(time.time() - t0)

        if verbose_loss:
            loss_logs.append(adapt_log)

        probs_np = probs.numpy()  # [1, C_task] for tcp | [1, C_total] for naive

        if mode == "tcp":
            # --- TCP: identical to original eval_task_tcp ---
            # pred_class is LOCAL (0..C_task-1), label is LOCAL
            preds_all.append(np.array([pred_class]))
            probs_all.append(probs_np)
            targets_all.append(label.numpy())

            g_label = task_to_global_class[task_id].get(int(label), -1)
            g_pred  = task_to_global_class[task_id].get(pred_class, -1)
            convert_targets_all.append(np.array([g_label]))
            convert_preds_all.append(np.array([g_pred]))

        else:
            # --- Naive: identical to original eval_task_naive ---
            # pred_class is COLUMN index (= global class index for forward order)
            pred_global = int(column_to_global[pred_class])
            true_global = task_to_global_class[task_id][int(label)]

            # Remap probs from column order to global class order
            probs_out = np.zeros((1, total_classes), dtype=np.float32)
            for col_idx, g_idx in enumerate(column_to_global):
                probs_out[0, g_idx] = probs_np[0, col_idx]

            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))

            convert_preds_all.append(np.array([pred_global]))
            convert_targets_all.append(np.array([true_global]))

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

    return (
        metrics, preds_arr, targets_arr, probs_arr,
        np.concatenate(convert_preds_all),
        np.concatenate(convert_targets_all),
        sum(times),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL TTA evaluation")

    # Paths (same as original)
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)

    # Mode (same as original)
    parser.add_argument("--mode", type=str, default="tcp",
                        choices=["tcp", "naive"],
                        help="tcp (default): TCP inference | naive: direct class inference")

    # TTA hyperparams
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--alpha",             type=float, default=0.5)
    parser.add_argument("--beta",              type=float, default=1.0)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--entropy_threshold", type=float, default=0.4,
                        help="Only TTA when slide entropy >= threshold. "
                             "Set 0.0 to TTA all slides.")
    parser.add_argument("--episodic",          action="store_true",
                        help="Reset LN params after each slide. "
                             "Default=False (continual).")
    parser.add_argument("--verbose_loss",      action="store_true")

    args = parser.parse_args()

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    # Load embeddings by mode (same logic as original)
    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    if getattr(cfg.dataset, "order", "forward") == "reverse":
        task_prompts = task_prompts.flip(0)

    if args.mode == "naive":
        all_class_embeddings = build_class_embeddings(device, seq_dataset.task_names)
    else:
        all_class_embeddings = None

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

        tta_model = MergeSlide_TTA(
            backbone             = base_model.vision_encoder,
            task_prompts         = task_prompts,
            task_weights         = task_weights,
            num_classes          = seq_dataset.num_classes,
            device               = device,
            mode                 = args.mode,
            all_class_embeddings = all_class_embeddings,
            M                    = args.M,
            K_sub                = args.K_sub,
            top_ratio            = args.top_ratio,
            alpha                = args.alpha,
            beta                 = args.beta,
            lr                   = args.lr,
            n_steps              = args.n_steps,
            episodic             = args.episodic,
            entropy_threshold    = args.entropy_threshold,
        )

        all_baccs     = []
        all_accs      = []
        all_aucs      = []
        all_preds_g   = []
        all_targets_g = []
        acc_per_task  = {}
        fold_time     = 0.0
        num_total     = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            result = eval_task_tta(
                test_loader          = test_loader,
                task_id              = task_id,
                tta_model            = tta_model,
                task_to_global_class = seq_dataset.task_to_global_class,
                device               = device,
                mode                 = args.mode,
                num_classes_per_task = seq_dataset.num_classes,
                verbose_loss         = args.verbose_loss,
            )
            results, preds_all, targets_all, probs_all, \
                conv_preds, conv_targets, task_time = result

            num_total += len(test_loader)
            fold_time += task_time / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)

            if args.mode == "tcp":
                # Local class indices
                for i in range(seq_dataset.num_classes[task_id]):
                    all_aucs.append(
                        roc_auc_score((targets_all == i).astype(int), probs_all[:, i])
                    )
            else:
                # Global class indices (targets_all already in global space)
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for g_idx in global_idxs:
                    all_aucs.append(
                        roc_auc_score(
                            (targets_all == g_idx).astype(int),
                            probs_all[:, g_idx]
                        )
                    )

        n_adapted = tta_model.n_adapted
        n_skipped = tta_model.n_skipped
        n_total   = n_adapted + n_skipped
        print(f"[Fold {fold_id}] adapted={n_adapted}/{n_total} "
              f"({100*n_adapted/max(n_total,1):.1f}%) | "
              f"entropy_threshold={args.entropy_threshold}")
        tta_model.hard_reset()

        all_preds_g   = np.concatenate(all_preds_g)
        all_targets_g = np.concatenate(all_targets_g)

        overall_accs.append(np.mean(all_accs))
        overall_baccs.append(np.mean(all_baccs))
        overall_macro_f1s.append(
            f1_score(all_targets_g, all_preds_g, average="macro"))
        overall_weighted_f1s.append(
            f1_score(all_targets_g, all_preds_g, average="weighted"))
        overall_recalls.append(
            recall_score(all_targets_g, all_preds_g, average=None))
        overall_precisions.append(
            precision_score(all_targets_g, all_preds_g, average=None))
        overall_aucs.append(np.array(all_aucs))
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}% "
              f"BAcc={np.mean(all_baccs)*100:.4f}%")

    reset_label = "episodic" if args.episodic else "continual"
    print(f"\n===== Class-IL TTA ({args.mode.upper()}, {reset_label}, M={args.M}) =====")
    print(f"Accuracy:       {np.mean(overall_accs)*100:.4f}%"
          f" ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:   {np.mean(overall_baccs)*100:.4f}%"
          f" ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:       {np.mean(overall_macro_f1s)*100:.4f}%"
          f" ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:    {np.mean(overall_weighted_f1s)*100:.4f}%"
          f" ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time: {np.mean(overall_times):.3f}s"
          f" ({np.std(overall_times):.3f}s)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs[t])*100:.4f}%"
              f" ({np.std(accs[t])*100:.4f}%)")