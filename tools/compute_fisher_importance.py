#!/usr/bin/env python3
"""
tools/compute_fisher_importance.py
===================================
Tính Fisher importance omega(theta) MỘT LẦN, TRƯỚC khi TTA bắt đầu, cho
Module D' (EATA-inspired anti-forgetting regularizer, xem TTAConfig_Core.
use_fisher_reg). KHÔNG bắt buộc chạy — Module D (FIM restore động) vẫn là
mặc định và hoạt động độc lập với script này.

Khác biệt so với EATA gốc: EATA dùng pseudo-label (không có nhãn thật) vì
họ không truy cập được training data. Ở đây, mergeslide_tta/datasets.py
CÓ SẴN train/val loader với NHÃN THẬT cho từng task (train_loader,
val_loader = seq_dataset.get_data_loaders(fold, task_id)) tại thời điểm
model vừa merge xong task đó — dữ liệu này KHÔNG vi phạm ràng buộc
no-old-data (research_protocol.md §3.2) vì nó là dữ liệu hiện có của
chính task đang xử lý, không phải replay dữ liệu của task đã lùi xa.
Do đó, omega(theta) ở đây dùng cross-entropy với NHÃN THẬT thay vì
pseudo-label — kỳ vọng ít nhiễu hơn EATA gốc (Assumption, cần ablate AC-6).

Công thức (xem tài liệu nghiên cứu §4.5):
    omega(theta_i) = (1/Q) * sum_{x_q in D_F} ( d/d theta_i^0
                        L_CE( f_{theta_0}(x_q), y_q ) )^2

Usage::
    python tools/compute_fisher_importance.py \\
        --config configs/default_tta_core_eval_num_workers0.yaml \\
        --merge_model_path checkpoints/merged \\
        --fold 0 \\
        --pool_tasks all \\
        --max_samples_per_task 100 \\
        --out_path checkpoints/fisher_omega/fold_0.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import TASK_CLASS_RANGES_FORWARD, TASK_NAMES_FORWARD, TITAN_PS_ARG
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.model import build_prompt_classifier
from mergeslide_tta.utils import seed_torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CLASSIFIER_RANGE_BY_NAME = {
    name: TASK_CLASS_RANGES_FORWARD[i] for i, name in enumerate(TASK_NAMES_FORWARD)
}


def build_global_mlp_weights(classifier, task_names, device):
    parts = [
        classifier[:, _CLASSIFIER_RANGE_BY_NAME[n][0]:_CLASSIFIER_RANGE_BY_NAME[n][1] + 1]
        for n in task_names
    ]
    global_w = torch.cat(parts, dim=1).T.contiguous().to(device)
    return global_w, torch.zeros(global_w.shape[0], device=device)


def make_ln_only_backbone(base: nn.Module, sd: dict, device) -> nn.Module:
    """Giống _make_backbone(train=True) trong tta_engine_core.py — chỉ LN trainable."""
    import copy
    bb = copy.deepcopy(base).to(device)
    bb.load_state_dict(sd, strict=True)
    bb.train()
    for p in bb.parameters():
        p.requires_grad_(False)
    for m in bb.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    return bb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",             type=str, required=True)
    parser.add_argument("--merge_model_path",   type=str, required=True)
    parser.add_argument("--fold",               type=int, required=True)
    parser.add_argument("--pool_tasks",         type=str, default="all",
                         help="'all' hoac danh sach task_id cach nhau boi dau phay, vd '0,1,2'")
    parser.add_argument("--split",              type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max_samples_per_task", type=int, default=100,
                         help="Q_per_task -- tham khao AC-6/AC-11: EATA dung Q~300-500 tong the")
    parser.add_argument("--out_path",           type=str, required=True)
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    num_tasks   = cfg.training.num_tasks

    if args.pool_tasks == "all":
        task_ids = list(range(num_tasks))
    else:
        task_ids = [int(t) for t in args.pool_tasks.split(",")]

    print("Building prompt classifier ...")
    classifier, _ = build_prompt_classifier(str(device))
    global_w, global_b = build_global_mlp_weights(classifier, seq_dataset.task_names, device)

    merged_path = Path(args.merge_model_path) / f"fold_{args.fold}" / "merged_final.pth"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_final not found: {merged_path}")
    backbone_sd = torch.load(str(merged_path), map_location="cpu")

    print("Loading TITAN base model ...")
    base_titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_titan = base_titan.to(device).eval()

    model = make_ln_only_backbone(base_titan.vision_encoder, backbone_sd, device)
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    ln_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fisher_sum = {n: torch.zeros_like(p, device="cpu") for n, p in ln_params.items()}
    total_q = 0

    for task_id in task_ids:
        task_name = seq_dataset.task_names[task_id]
        train_loader, val_loader, _ = seq_dataset.get_data_loaders(args.fold, task_id)
        loader = train_loader if args.split == "train" else val_loader

        n_this_task = 0
        for features, coords, label in tqdm(loader, desc=f"Fisher[{task_name}]", leave=False):
            if n_this_task >= args.max_samples_per_task:
                break
            features = features.to(device)
            coords   = coords.long().to(device)
            g_label  = int(seq_dataset.task_to_global_class[task_id][int(label)])

            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                z = model(features, coords, ps)
            logits = F.linear(z.float(), global_w, global_b)
            loss = F.cross_entropy(logits, torch.tensor([g_label], device=device))
            loss.backward()

            for n, p in ln_params.items():
                if p.grad is not None:
                    fisher_sum[n] += p.grad.detach().float().pow(2).cpu()

            n_this_task += 1
            total_q += 1

        print(f"[INFO] task={task_name} used {n_this_task} samples ({args.split})")

    if total_q == 0:
        raise RuntimeError("Khong co sample nao duoc dung de tinh Fisher -- kiem tra lai --pool_tasks/--split.")

    fisher_omega = {n: (v / total_q) for n, v in fisher_sum.items()}

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher_omega, str(out_path))
    print(f"[INFO] Fisher omega (Q={total_q} samples) saved -> {out_path}")
    print("[INFO] Su dung: set tta.use_fisher_reg=true va tta.fisher_omega_path trong config,")
    print("       hoac truyen fisher_omega_path khi khoi tao TTAConfig_Core.")


if __name__ == "__main__":
    main()
