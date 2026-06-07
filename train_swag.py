"""
train_swag.py — Tính SWAG-diagonal posterior cho MergeSlide vision encoder.

Chạy SAU khi merge.py đã hoàn thành. Load merged_final.pth vào backbone,
chạy thêm N_SWAG_EPOCHS epochs trên train data để collect posterior statistics.
Lưu swag_diagonal/fold_{id}.pt dùng cho PETAL TTA.

Usage:
    python train_swag.py --config configs/default.yaml [--fold_start 0] [--fold_end 10]
    python train_swag.py --config configs/default_ood.yaml
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

from mergeslide_tta.constants import (
    NUM_TASKS, K_PATCHES, TITAN_PS_ARG,
    TASK_NAMES_FORWARD, TASK_CLASS_RANGES_FORWARD,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.model import build_prompt_classifier, cosine_lr, CustomSequential
from mergeslide_tta.swag_diagonal import SWAGDiagonal
from mergeslide_tta.utils import seed_torch


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_RANGE_BY_TASK_NAME = {
    task_name: TASK_CLASS_RANGES_FORWARD[task_id]
    for task_id, task_name in enumerate(TASK_NAMES_FORWARD)
}


def get_local_hot_root() -> Path:
    user = os.environ.get("USER") or "thanhld"
    default_root = Path("/docker/data") / user / PROJECT_ROOT.name
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", default_root)).expanduser()


def ensure_dirs(local_root: Path) -> None:
    for d in ("logs", "checkpoints", "checkpoints_ood", "sqlite", "tmp"):
        (local_root / d).mkdir(parents=True, exist_ok=True)
    for name in ("logs", "checkpoints", "checkpoints_ood"):
        repo_path = PROJECT_ROOT / name
        local_path = local_root / name
        if not repo_path.is_symlink() and not repo_path.exists():
            repo_path.symlink_to(local_path, target_is_directory=True)
    os.environ.setdefault("TMPDIR",             str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR",      str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def resolve_path(path: str, local_root: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        try:
            rel = raw.relative_to(PROJECT_ROOT)
            if rel.parts and rel.parts[0] in {"checkpoints", "logs", "checkpoints_ood"}:
                return local_root / rel
        except ValueError:
            pass
        return raw
    if raw.parts and raw.parts[0] in {"checkpoints", "logs", "checkpoints_ood"}:
        return local_root / raw
    return raw


def train_swag_one_fold(
    fold_id:      int,
    base_model:   nn.Module,
    seq_dataset:  Sequential_Generic_MIL_Dataset,
    classifier:   torch.Tensor,
    merged_dir:   Path,
    swag_dir:     Path,
    n_swag_epochs: int,
    lr:           float,
    weight_decay: float,
    device:       torch.device,
    cfg,
) -> None:
    """
    Load merged_final.pth cho fold_id, chạy SWAG training, lưu stats.
    """
    fold_name = f"fold_{fold_id}"

    merge_path = merged_dir / fold_name / "merged_final.pth"
    if not merge_path.exists():
        print(f"[WARN] merged_final not found: {merge_path} — skipping fold {fold_id}")
        return

    print(f"\n{'='*50}\nFold {fold_id}  — Loading {merge_path}\n{'='*50}")
    base_model.vision_encoder.load_state_dict(
        torch.load(str(merge_path), map_location="cpu")
    )
    backbone = base_model.vision_encoder

    # SWAGDiagonal instance
    swag = SWAGDiagonal(backbone)

    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    # Chạy qua tất cả tasks để collect statistics (joint training)
    for task_id in range(NUM_TASKS):
        task_names = seq_dataset.task_names
        task_name = task_names[task_id]
        n_cls = seq_dataset.num_classes[task_id]
        print(f"\n  Task {task_id} ({task_name}) — {n_swag_epochs} SWAG epochs")

        # Build per-task MLP (frozen, same as training)
        mlp = nn.Linear(768, n_cls).to(device)
        mlp.bias.data.zero_()
        start, end = CLASSIFIER_RANGE_BY_TASK_NAME[task_name]
        prompt_weight = classifier[:, start:end + 1].T.contiguous()
        if prompt_weight.shape != mlp.weight.shape:
            raise RuntimeError(
                "Prompt classifier slice shape mismatch: "
                f"task_id={task_id} task_name={task_name} "
                f"range=({start},{end}) "
                f"weight_shape={tuple(prompt_weight.shape)} "
                f"mlp_shape={tuple(mlp.weight.shape)}"
            )
        mlp.weight.data.copy_(prompt_weight)
        for p in mlp.parameters():
            p.requires_grad_(False)

        train_loader, _, _ = seq_dataset.get_data_loaders(fold_id, task_id)

        # Optimizer: only backbone LN params (matching TTA setting)
        ln_params = [p for m in backbone.modules()
                     if isinstance(m, nn.LayerNorm)
                     for p in m.parameters()]
        # SWAG phase dùng constant LR — không decay
        optimizer = torch.optim.SGD(ln_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        scaler  = torch.cuda.amp.GradScaler()

        for epoch in range(n_swag_epochs):
            backbone.train()
            for features, coords, labels in tqdm(train_loader, leave=False):
                # Không gọi lr_scheduler — giữ LR cố định
                features = features.to(device)
                coords   = coords.long().to(device)
                idx      = torch.randperm(features.shape[0])[:K_PATCHES]
                features, coords = features[idx], coords[idx]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    z      = backbone(features, coords, ps)
                    logits = mlp(z.float())
                    loss   = loss_fn(logits, labels.to(device))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            swag.update(backbone)
            print(f"    Epoch {epoch+1}/{n_swag_epochs} — SWAG snapshots: {swag.n_collected}")

    # Save
    save_path = swag_dir / f"{fold_name}.pt"
    swag.save(save_path)
    print(f"[Fold {fold_id}] SWAG saved → {save_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Compute SWAG-diagonal posterior for MergeSlide")
    parser.add_argument("--config",       type=str, default="configs/default.yaml")
    parser.add_argument("--fold_start",   type=int, default=0)
    parser.add_argument("--fold_end",     type=int, default=None)
    parser.add_argument("--merged_checkpoints", type=str, default=None,
                        help="Override cfg.paths.merged_checkpoints")
    parser.add_argument("--swag_dir",     type=str, default=None,
                        help="Output directory for SWAG stats (default: merged_dir/../swag_diagonal)")
    parser.add_argument("--n_swag_epochs", type=int, default=None,
                        help="Override cfg.tta.swag_epochs (default: 5)")
    args = parser.parse_args()

    local_root = get_local_hot_root()
    ensure_dirs(local_root)

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    fold_start = args.fold_start
    fold_end   = args.fold_end or cfg.training.num_folds

    # Resolve paths
    merged_base = args.merged_checkpoints or cfg.paths.merged_checkpoints
    merged_dir  = resolve_path(merged_base, local_root)

    if args.swag_dir:
        swag_dir = Path(args.swag_dir)
    else:
        # Default: đặt cạnh merged checkpoints
        swag_dir = merged_dir.parent / "swag_diagonal"
    swag_dir.mkdir(parents=True, exist_ok=True)

    _tta_raw = cfg.get("tta", None)
    if _tta_raw is None:
        tta_cfg = {}
    else:
        from omegaconf import DictConfig
        tta_cfg = OmegaConf.to_container(_tta_raw, resolve=True) if isinstance(_tta_raw, DictConfig) else dict(_tta_raw)
    n_swag_epochs = args.n_swag_epochs or tta_cfg.get("swag_epochs", 5)
    lr            = float(tta_cfg.get("swag_lr", 1e-5))
    weight_decay  = float(cfg.training.weight_decay)

    print(f"[INFO] merged_dir:    {merged_dir}")
    print(f"[INFO] swag_dir:      {swag_dir}")
    print(f"[INFO] folds:         {fold_start} → {fold_end}")
    print(f"[INFO] swag_epochs:   {n_swag_epochs}")

    print("Building prompt classifier ...")
    classifier, _ = build_prompt_classifier(str(device))

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)
    base_model.eval()

    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    for fold_id in range(fold_start, fold_end):
        train_swag_one_fold(
            fold_id       = fold_id,
            base_model    = base_model,
            seq_dataset   = seq_dataset,
            classifier    = classifier,
            merged_dir    = merged_dir,
            swag_dir      = swag_dir,
            n_swag_epochs = n_swag_epochs,
            lr            = lr,
            weight_decay  = weight_decay,
            device        = device,
            cfg           = cfg,
        )

    print("\nSWAG training complete.")
