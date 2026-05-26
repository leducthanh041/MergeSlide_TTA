# test_classIL_task_prompt_other_metrics.py
"""
Class-IL evaluation — BWT, Forgetting, mACC — hỗ trợ TCP và Naive.

Modes:
    tcp   (default): Task-to-Class Prompt-Aligned inference
    naive          : y_pred = argmax(Z @ All_Class_Embeddings.T)

Usage:
    python test_classIL_task_prompt_other_metrics.py \
        --save_dir /path/to/finetuned_checkpoints \
        --merge_model_path /path/to/merged/checkpoints \
        --mode tcp      # hoặc --mode naive

Cấu trúc checkpoint kỳ vọng:
    Finetuned    : {save_dir}/fold_{id}/task_{t}.pt
    Intermediate : {merge_model_path}/fold_{id}/merged_task_{seq_task}.pth
    Final        : {merge_model_path}/fold_{id}/merged_final.pth
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

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_TASKS,
    TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import backward_transfer, forgetting, pad_numpy_arrays
from mergeslide_tta.model import CustomSequential
from mergeslide_tta.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts,
)
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

_PROMPT_FN_MAP = {
    "BRCA":  brca_prompts,
    "RCC":   rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA":  esca_prompts,
    "TGCT":  tgct_prompts,
    "CESC":  cesc_prompts,
}

# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def eval_task_tcp(
    test_loader,
    model: CustomSequential,
    num_classes: list,
    task_prompts: torch.Tensor,
    task_model_paths: list,
    device,
) -> tuple:
    """
    TCP inference:
      1. t_hat = argmax(Z @ Task_Embeddings.T)
      2. y_pred = argmax(Z @ Class_Embeddings[t_hat].T)  via MLP head
    Returns (metrics, preds_all, targets_all).
    """
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

    return _pack_results(preds_all, targets_all, probs_all)


def eval_task_naive(
    test_loader,
    task_id: int,
    model: CustomSequential,
    task_model_paths: list,
    num_classes: list,
    device,
    task_to_global_class: dict,
) -> tuple:
    """
    Naive inference (giống test_classIL_meanACC.py gốc):
      - Build merged MLP head bằng cách cat weights của các per-task MLP đã học.
      - y_pred = argmax(merged_mlp(Z))  →  global class index
      - Label được convert sang global space qua task_to_global_class.
 
    task_model_paths và num_classes đã được slice đến seq_task trước khi gọi hàm này,
    nên merged MLP có shape Linear(768, sum(num_classes[:seq_task])).
    """
    preds_all   = []
    probs_all   = []
    targets_all = []
 
    ps            = torch.tensor(TITAN_PS_ARG).int().to(device)
    total_classes = sum(num_classes)
 
    # Build merged MLP: cat per-task weights theo thứ tự task
    mlp_task_weights = []
    for p in task_model_paths:
        state = torch.load(p, map_location="cpu")
        mlp_task_weights.append(
            {k.split("mlp.")[-1]: state[k] for k in list(state.keys())[-2:]}
        )
    merged_mlp_state = {
        "weight": torch.cat([w["weight"] for w in mlp_task_weights], dim=0),
        "bias":   torch.cat([w["bias"]   for w in mlp_task_weights], dim=0),
    }
    merged_mlp = nn.Linear(EMBED_DIM, total_classes).to(device)
    merged_mlp.load_state_dict(merged_mlp_state)
    merged_mlp.eval()
 
    # column_to_global: column i của merged MLP output → global class index
    column_to_global = np.array([
        task_to_global_class[t][local]
        for t in range(len(task_model_paths))
        for local in sorted(task_to_global_class[t].keys())
    ])
 
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]
 
            slide_embed = model.backbone(features, coords, ps)
            logits      = merged_mlp(slide_embed.float())   # [1, total_classes]
            pred_column = int(logits.argmax(1))
            pred_global = int(column_to_global[pred_column])
            true_global = task_to_global_class[task_id][int(label)]
 
            probs_np  = nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            probs_out = np.zeros((1, sum(num_classes)), dtype=np.float32)
            for col_idx, g_idx in enumerate(column_to_global):
                probs_out[0, g_idx] = probs_np[col_idx]
 
            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))
 
    return _pack_results(preds_all, targets_all, probs_all)

def _pack_results(preds_all, targets_all, probs_all) -> tuple:
    """Gộp list → numpy và tính metrics. Dùng chung cho cả 2 modes."""
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


def build_class_embeddings(device, task_names: list) -> torch.Tensor:
    """
    Build all_class_embeddings [EMBED_DIM, total_classes] từ TITAN text encoder.
    Dùng cho Naive mode.

    Columns được sắp xếp theo đúng thứ tự task_names:
        Forward:  col 0,1=BRCA | 2,3,4=RCC | 5,6=NSCLC | 7,8=ESCA | 9,10=TGCT | 11,12=CESC
        Reversed: col 0,1=CESC | 2,3=TGCT  | 4,5=ESCA  | 6,7=NSCLC | 8,9,10=RCC | 11,12=BRCA
    """
    print("Building all_class_embeddings for Naive mode ...")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)

    _, templates = brca_prompts()
    all_prompts  = []
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL BWT/FGT evaluation")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Root dir chứa merged checkpoints")
    parser.add_argument("--mode",             type=str, default="tcp",
                        choices=["tcp", "naive"],
                        help="tcp (default): TCP inference | naive: direct class inference")
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

    num_tasks   = cfg.training.num_tasks
    seq_dataset  = Sequential_Generic_MIL_Dataset(cfg)
    num_classes  = seq_dataset.num_classes

    # Load embeddings tuỳ theo mode
    if args.mode == "tcp":
        task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
        if getattr(cfg.dataset, 'order', 'forward') == 'reverse':
            task_prompts = task_prompts.flip(0)
        all_class_embeddings = None
    else:
        task_prompts         = None

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

        acc_per_task_all_seqs = []
        ACC_all_seqs          = []

        for seq_task in tqdm(range(1, num_tasks + 1), desc="Seq tasks", leave=False):
            seed_torch(device, cfg.training.seed)

            if seq_task == 1:
                # seq_task=1: chỉ task 0 — load thẳng finetuned checkpoint task 0
                # Đây là ACC_t(t) = ACC ngay sau khi finetuned task 0, chưa merge gì
                # Dùng vision encoder của base TITAN (chưa finetune)
                # vì finetuned checkpoint chỉ lưu backbone+mlp state_dict,
                # không lưu riêng vision encoder — giống cách code gốc dùng task_0.pth
                ckpt_path = Path(args.save_dir) / fold / "task_0.pt"
                state = torch.load(str(ckpt_path), map_location="cpu")
                # Lấy backbone weights (bỏ 2 key cuối là mlp)
                backbone_state = {
                    k.split("backbone.")[-1]: state[k]
                    for k in list(state.keys())[:-2]
                }
                base_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                ).to(device)
                base_model.vision_encoder.load_state_dict(backbone_state, strict=True)
            elif seq_task < num_tasks:
                ckpt_name  = f"merged_task_{seq_task - 1}.pth"
                merge_model_path = Path(args.merge_model_path) / fold / ckpt_name
                base_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                ).to(device)
                base_model.vision_encoder.load_state_dict(
                    torch.load(str(merge_model_path), map_location="cpu")
                )
            else:
                # seq_task=6: final merged checkpoint
                merge_model_path = Path(args.merge_model_path) / fold / "merged_final.pth"
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

                if args.mode == "tcp":
                    _, preds_all, targets_all = eval_task_tcp(
                        test_loader, model,
                        num_classes[:seq_task],
                        task_prompts[:seq_task],
                        task_model_paths[:seq_task],
                        device,
                    )
                else:
                    _, preds_all, targets_all = eval_task_naive(
                        test_loader, task_id, model,
                        task_model_paths[:seq_task],
                        num_classes[:seq_task],
                        device,
                        task_to_global_class=seq_dataset.task_to_global_class,
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

        print(f"[Fold {fold_id}] mACC={mACC*100:.4f}% "
              f"FGT={fgt*100:.4f}% BWT={bwt*100:.4f}%")

    mode_label = "TCP" if args.mode == "tcp" else "Naive"
    print(f"\n===== Class-IL ({mode_label}) BWT/FGT Results =====")
    print(f"mACC: {np.mean(mACCs_all_folds)*100:.4f}% ({np.std(mACCs_all_folds)*100:.4f}%)")
    print(f"BWT:  {np.mean(bwt_all_folds)*100:.4f}% ({np.std(bwt_all_folds)*100:.4f}%)")
    print(f"FGT:  {np.mean(fgt_all_folds)*100:.4f}% ({np.std(fgt_all_folds)*100:.4f}%)")

    print("\nACC per seq task (mean across folds):")
    acc_seq_arr = np.array(ACC_all_seqs_all_folds)
    for t in range(num_tasks):
        print(f"  After task {t+1}: {np.mean(acc_seq_arr[:, t])*100:.4f}% "
              f"({np.std(acc_seq_arr[:, t])*100:.4f}%)")
