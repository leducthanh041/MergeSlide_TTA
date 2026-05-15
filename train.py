# train.py
"""
Per-task finetuning of TITAN on sequential TCGA tasks.
Saves one checkpoint per (fold, task): checkpoints/finetuned/fold_{k}/task_{t}.pt
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

from mergeslide_tta.constants import NUM_TASKS, NUM_CLASSES, TASK_CLASS_RANGES, K_PATCHES, TITAN_PS_ARG, TASK_NAMES
from mergeslide_tta.checkpoint_mirror import save_checkpoint_with_mirror
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.model import build_model, build_prompt_classifier, cosine_lr, EarlyStopping
from mergeslide_tta.utils import seed_torch


def format_bytes(num_bytes: int) -> str:
    """Format bytes as GiB for readable GPU memory logs."""
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def require_cuda_device() -> torch.device:
    """Require CUDA and return the active PyTorch CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run training on CPU. "
            "Check Slurm GPU allocation, CUDA_VISIBLE_DEVICES, driver, and PyTorch CUDA build."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def print_gpu_vram(prefix: str, device: torch.device) -> None:
    """Print current GPU identity and VRAM usage."""
    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    used_by_context = total_bytes - free_bytes

    print(
        f"[GPU][{prefix}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
        f"torch_device={device} | "
        f"visible_gpus={torch.cuda.device_count()} | "
        f"current_device={device_idx} | "
        f"name={props.name} | "
        f"total={format_bytes(total_bytes)} | "
        f"free={format_bytes(free_bytes)} | "
        f"used_context={format_bytes(used_by_context)} | "
        f"allocated={format_bytes(allocated)} | "
        f"reserved={format_bytes(reserved)}"
    )


def train_one_task(
    train_loader, val_loader, model: nn.Module,
    num_epochs: int, lr: float, weight_decay: float, device: str,
) -> nn.Module:
    """Finetune model on a single task. Returns the trained model."""
    named_params = list(model.named_parameters())
    exclude = lambda n, p: p.ndim < 2 or any(k in n for k in ("bn","ln","bias","logit_scale"))
    gain_or_bias = [p for n, p in named_params if exclude(n, p) and p.requires_grad]
    rest = [p for n, p in named_params if not exclude(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias, "weight_decay": 0.0},
         {"params": rest,         "weight_decay": weight_decay}],
        lr=lr,
    )
    total_steps = len(train_loader) * num_epochs
    lr_scheduler = cosine_lr(optimizer, lr, int(total_steps * 0.1), total_steps)
    loss_fn = nn.CrossEntropyLoss()
    scaler  = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=2, verbose=True)

    step = 0
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        preds_all, targets_all, total_loss = [], [], 0.0

        for features, coords, label in tqdm(train_loader, leave=False):
            lr_scheduler(step)
            features = features.to(device)
            coords   = coords.long().to(device)

            # Random patch sampling
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(features, coords, ps)
                loss   = loss_fn(logits, label.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(label.numpy())
            total_loss += loss.item()
            step += 1

        bacc = balanced_accuracy_score(
            np.concatenate(targets_all), np.concatenate(preds_all)
        )

        # Validation only after epoch 1 (original logic)
        if epoch > 1:
            model.eval()
            v_preds, v_targets, v_loss = [], [], 0.0
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for features, coords, labels in val_loader:
                    idx = torch.randperm(features.shape[0])[:K_PATCHES]
                    features = features.to(device)[idx]
                    coords   = coords.long().to(device)[idx]
                    logits   = model(features, coords, ps)
                    v_loss  += loss_fn(logits, labels.to(device)).item()
                    v_preds.append(logits.argmax(1).cpu().numpy())
                    v_targets.append(labels.numpy())

            avg_val = v_loss / len(val_loader)
            bacc_val = balanced_accuracy_score(
                np.concatenate(v_targets), np.concatenate(v_preds)
            )
            tqdm.write(f"[Epoch {epoch}] BAcc={bacc:.4f} BAcc_val={bacc_val:.4f} "
                       f"Loss={total_loss/len(train_loader):.4f} Val={avg_val:.4f}")
            early_stopping(avg_val, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        else:
            tqdm.write(f"[Epoch {epoch}] BAcc={bacc:.4f} "
                       f"Loss={total_loss/len(train_loader):.4f}")

    model.eval()
    return model


def save_checkpoint(model: nn.Module, save_dir: str, fold_id: int, task_id: int) -> tuple[str, str | None]:
    """Save model state_dict. Returns primary and mirror paths."""
    fold_dir = Path(save_dir) / f"fold_{fold_id}"
    path = fold_dir / f"task_{task_id}.pt"
    primary_path, mirror_path = save_checkpoint_with_mirror(model.state_dict(), path)
    return str(primary_path), str(mirror_path) if mirror_path is not None else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-task finetuning of TITAN")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override checkpoints path from config")
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end",   type=int, default=10)
    args = parser.parse_args()  # ← parse ONCE, outside fold loop

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_vram("startup", device)
    seed_torch(device, cfg.training.seed)

    save_dir = args.save_dir or cfg.paths.finetuned_checkpoints

    # Build prompt classifier ONCE (expensive — loads TITAN text encoder)
    print("Building prompt classifier ...")
    classifier, _ = build_prompt_classifier(str(device))
    print_gpu_vram("after_prompt_classifier", device)

    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    for fold_id in range(args.fold_start, args.fold_end):
        print(f"\n{'='*50}\nFold {fold_id}\n{'='*50}")

        for task_id in range(NUM_TASKS):   # ← FIX: range(NUM_TASKS) không phải range(3)
            task_names = (
                list(reversed(TASK_NAMES))
                if getattr(cfg.dataset, 'order', 'forward') == 'reverse'
                else TASK_NAMES
            )
            print(f"\n--- Task {task_id} ({seq_dataset.task_names[task_id]}) ---")
            print_gpu_vram(f"before_task_{task_id}", device)
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_id, task_id)

            model = build_model(
                task_id=task_id,
                num_classes=seq_dataset.num_classes[task_id],
                prompt_classifier=classifier,
                classifier_class_ranges=seq_dataset.classifier_class_ranges,
                device=str(device),
            )
            print_gpu_vram(f"after_build_model_task_{task_id}", device)

            t0 = time.time()
            model = train_one_task(
                train_loader, val_loader, model,
                num_epochs=cfg.training.num_epochs,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                device=str(device),
            )
            elapsed = time.time() - t0

            # ← FIX: checkpoint saving (bị thiếu hoàn toàn trong code gốc)
            ckpt_path, mirror_path = save_checkpoint(model, save_dir, fold_id, task_id)
            if mirror_path is not None:
                print(f"Saved: {ckpt_path} | Mirror: {mirror_path} | Time: {elapsed:.1f}s")
            else:
                print(f"Saved: {ckpt_path} | Time: {elapsed:.1f}s")

            del model
            torch.cuda.empty_cache()
            print_gpu_vram(f"after_cleanup_task_{task_id}", device)
