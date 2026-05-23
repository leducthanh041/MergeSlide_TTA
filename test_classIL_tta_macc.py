# test_classIL_tta_macc.py
"""
Class-IL TTA evaluation for mACC metric.
Mirrors test_classIL_task_prompt_other_metrics.py but applies TTA at each
sequential checkpoint.

mACC = mean(ACC_1, ..., ACC_T) where ACC_t = accuracy after learning t tasks.
TTA is applied at each seq_task checkpoint with partial task_prompts[:seq_task].

BWT and FGT are also reported as side metrics.

Usage:
    python test_classIL_tta_macc.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged \\
        --mode tcp
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import K_PATCHES, NUM_TASKS, TITAN_PS_ARG
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import backward_transfer, forgetting, pad_numpy_arrays
from mergeslide_tta.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts,
)
from mergeslide_tta.utils import get_eval_metrics, seed_torch
from mergeslide_tta.tta_adapter import MergeSlide_TTA, load_task_weights

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}

_PROMPT_FN_MAP = {
    "BRCA":  brca_prompts, "RCC":   rcc_prompts,
    "NSCLC": nsclc_prompts, "ESCA":  esca_prompts,
    "TGCT":  tgct_prompts,  "CESC":  cesc_prompts,
}


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


def build_class_embeddings(device, task_names: list) -> torch.Tensor:
    """[768, C_total] -- for naive mode."""
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


def eval_task_tta_partial(
    test_loader,
    task_id:              int,
    tta_model:            MergeSlide_TTA,
    task_to_global_class: dict,
    mode:                 str,
    column_to_global:     np.ndarray,
) -> tuple:
    """
    TTA inference for 1 task at a given seq_task checkpoint.
    Mirrors eval_task_tta from test_classIL_tta.py.
    """
    preds_all   = []
    targets_all = []
    total_classes = len(column_to_global)

    for features, coords, label in tqdm(test_loader, leave=False):
        features = features.to(tta_model.device)
        coords   = coords.long().to(tta_model.device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        pred_class, _, _, _ = tta_model.adapt_and_predict(features, coords)

        if mode == "tcp":
            preds_all.append(pred_class)
            targets_all.append(int(label))
        else:
            # Naive: pred_class is column index -> map to global
            pred_global = int(column_to_global[pred_class])
            true_global = task_to_global_class[task_id][int(label)]
            preds_all.append(pred_global)
            targets_all.append(true_global)

    return np.array(preds_all), np.array(targets_all)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL TTA mACC evaluation")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)
    parser.add_argument("--mode",             type=str, default="tcp",
                        choices=["tcp", "naive"])
    # TTA hyperparams
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--alpha",             type=float, default=0.5)
    parser.add_argument("--beta",              type=float, default=1.0)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--entropy_threshold", type=float, default=0.4)
    parser.add_argument("--episodic",          action="store_true")
    args = parser.parse_args()

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    # Full task_prompts (sliced per seq_task below)
    task_prompts_full = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    if getattr(cfg.dataset, "order", "forward") == "reverse":
        task_prompts_full = task_prompts_full.flip(0)

    if args.mode == "naive":
        all_class_embeddings = build_class_embeddings(device, seq_dataset.task_names)
        column_to_global = np.array([
            seq_dataset.task_to_global_class[t][local]
            for t in range(num_tasks)
            for local in sorted(seq_dataset.task_to_global_class[t].keys())
        ])
    else:
        all_class_embeddings = None
        column_to_global     = None

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
        task_weights_full = load_task_weights(task_model_paths, device)

        acc_per_task_all_seqs = []
        ACC_all_seqs          = []

        for seq_task in tqdm(range(1, num_tasks + 1), desc="Seq tasks", leave=False):
            seed_torch(device, cfg.training.seed)

            # --- Load backbone for this seq_task (identical to other_metrics.py) ---
            if seq_task == 1:
                ckpt_path = Path(args.save_dir) / fold / "task_0.pt"
                state = torch.load(str(ckpt_path), map_location="cpu")
                backbone_state = {
                    k.split("backbone.")[-1]: state[k]
                    for k in list(state.keys())[:-2]
                }
                base_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                ).to(device)
                base_model.vision_encoder.load_state_dict(backbone_state, strict=True)
            elif seq_task < num_tasks:
                ckpt_name = f"merged_task_{seq_task - 1}.pth"
                merge_path = Path(args.merge_model_path) / fold / ckpt_name
                base_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                ).to(device)
                base_model.vision_encoder.load_state_dict(
                    torch.load(str(merge_path), map_location="cpu")
                )
            else:
                merge_path = Path(args.merge_model_path) / fold / "merged_final.pth"
                base_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                ).to(device)
                base_model.vision_encoder.load_state_dict(
                    torch.load(str(merge_path), map_location="cpu")
                )

            # --- Partial task_prompts and task_weights for this seq_task ---
            task_prompts_partial = task_prompts_full[:seq_task]
            task_weights_partial = task_weights_full[:seq_task]
            
            if args.mode == "naive":
                max_classes = sum(seq_dataset.num_classes[:seq_task])
                all_class_emb_partial  = all_class_embeddings[:, :max_classes]
                col_to_global_partial  = column_to_global[:max_classes]
            else:
                all_class_emb_partial  = None
                col_to_global_partial  = column_to_global

            # --- Build TTA model with partial task scope ---
            tta_model = MergeSlide_TTA(
                backbone             = base_model.vision_encoder,
                task_prompts         = task_prompts_partial,
                task_weights         = task_weights_partial,
                num_classes          = seq_dataset.num_classes[:seq_task],
                device               = device,
                mode                 = args.mode,
                all_class_embeddings = all_class_emb_partial,
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

            num_correct  = 0.0
            num_total    = 0.0
            acc_per_task = []

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

                preds_all, targets_all = eval_task_tta_partial(
                    test_loader          = test_loader,
                    task_id              = task_id,
                    tta_model            = tta_model,
                    task_to_global_class = seq_dataset.task_to_global_class,
                    mode                 = args.mode,
                    column_to_global     = col_to_global_partial,
                )

                num_correct += sum(preds_all == targets_all)
                num_total   += len(test_loader)
                acc_per_task.append(
                    sum(preds_all == targets_all) / len(targets_all)
                )

            ACC_all_seqs.append(float(num_correct / num_total))
            acc_per_task_all_seqs.append(acc_per_task)

            # Reset after each seq_task
            tta_model.hard_reset()
            del base_model, tta_model
            torch.cuda.empty_cache()

        mACC = np.mean(ACC_all_seqs)
        fgt  = forgetting(acc_per_task_all_seqs)
        bwt  = backward_transfer(acc_per_task_all_seqs)

        ACC_all_seqs_all_folds.append(ACC_all_seqs)
        mACCs_all_folds.append(mACC)
        fgt_all_folds.append(fgt)
        bwt_all_folds.append(bwt)

        print(f"[Fold {fold_id}] mACC={mACC*100:.4f}% "
              f"FGT={fgt*100:.4f}% BWT={bwt*100:.4f}%")

    mode_label = args.mode.upper()
    print(f"\n===== Class-IL TTA ({mode_label}) mACC/BWT/FGT =====")
    print(f"mACC: {np.mean(mACCs_all_folds)*100:.4f}%"
          f" ({np.std(mACCs_all_folds)*100:.4f}%)")
    print(f"BWT:  {np.mean(bwt_all_folds)*100:.4f}%"
          f" ({np.std(bwt_all_folds)*100:.4f}%)")
    print(f"FGT:  {np.mean(fgt_all_folds)*100:.4f}%"
          f" ({np.std(fgt_all_folds)*100:.4f}%)")

    print("\nACC per seq task (mean across folds):")
    acc_seq_arr = np.array(ACC_all_seqs_all_folds)
    for t in range(num_tasks):
        print(f"  After task {t+1}: {np.mean(acc_seq_arr[:, t])*100:.4f}%"
              f" ({np.std(acc_seq_arr[:, t])*100:.4f}%)")
