# merge.py
"""
C.OPCM Continual Model Merging cho MergeSlide.

Với mỗi fold, merge tuần tự N task checkpoints thành 1 merged vision encoder.
Lưu intermediate checkpoint sau mỗi task (dùng cho BWT/FWT evaluation)
và final checkpoint sau task cuối.

Usage:
    python merge.py --config configs/default.yaml
    python merge.py --config configs/default.yaml --fold_start 0 --fold_end 1
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModel
from omegaconf import OmegaConf

from mergeslide_tta.checkpoint_mirror import save_checkpoint_with_mirror
from mergeslide_tta.utils import (
    get_task_vector_norm,
    get_task_vector_state_dict,
    is_leaf_module,
    svd,
)


# ---------------------------------------------------------------------------
# C.OPCM core functions — không sửa công thức
# ---------------------------------------------------------------------------

def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """
    Merge Linear layer weights theo C.OPCM:
    - Tính task vector của merged model và task model so với pretrained.
    - Project task vector lên không gian trực giao (SVD), loại diagonal.
    - Cộng dồn có trọng số theo lambda.
    """
    original_device = merged_W.device
    merged_W    = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W      = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv            = task_W - pretrained_W

    u, s, v = svd(previous_merged_tv)
    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)           # loại diagonal component
    cleaned_task_tv = u @ projected_task_tv @ v.T

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """
    Merge các parameter không phải Linear weight (bias, LayerNorm, v.v.)
    bằng weighted sum đơn giản, không dùng SVD projection.
    """
    original_device = merged_W.device
    merged_W    = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W      = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv            = task_W - pretrained_W

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_backbone_weights(ckpt_path: str) -> dict:
    """
    Load checkpoint và trích xuất phần backbone (vision encoder),
    bỏ qua 2 key cuối là MLP head (weight + bias).

    Args:
        ckpt_path: Đường dẫn file .pt checkpoint.

    Returns:
        Dict state_dict chỉ chứa backbone keys, đã strip prefix 'backbone.'.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    backbone_keys = list(state.keys())[:-2]     # bỏ mlp.weight, mlp.bias
    return {
        k.split("backbone.")[-1]: state[k].detach()
        for k in backbone_keys
    }


def merge_one_task(
    base_model: nn.Module,
    merged_weight: dict,
    task_weight: dict,
    previous_lambda_t: float,
    lambda_t: float,
    task_idx: int,
) -> dict:
    """
    Merge task_weight vào merged_weight cho toàn bộ module của vision encoder.

    Args:
        base_model: TITAN base model (frozen, chỉ dùng để lấy pretrained weights).
        merged_weight: State dict của merged model hiện tại.
        task_weight: State dict của task model cần merge vào.
        previous_lambda_t: Lambda của bước trước.
        lambda_t: Lambda của bước hiện tại (temporary = 1 khi gọi hàm này).
        task_idx: Index task hiện tại (chỉ dùng cho tqdm label).

    Returns:
        merged_weight đã được update in-place.
    """
    vision_encoder = base_model.vision_encoder

    for module_name, module in tqdm(
        list(vision_encoder.named_modules()),
        desc=f"Merging task {task_idx}",
        leave=False,
    ):
        if not is_leaf_module(module):
            continue

        pretrained_module = vision_encoder.get_submodule(module_name)

        if isinstance(module, nn.Linear):
            # Linear weight — dùng SVD projection
            merged_weight[f"{module_name}.weight"] = merge_linear_weights(
                merged_W    = merged_weight[f"{module_name}.weight"],
                pretrained_W = pretrained_module.weight.detach(),
                task_W      = task_weight[f"{module_name}.weight"],
                previous_lambda_t=previous_lambda_t,
                lambda_t=lambda_t,
            )
            # Linear bias — không dùng SVD
            if module.bias is not None:
                merged_weight[f"{module_name}.bias"] = merge_other_parameters(
                    merged_W    = merged_weight[f"{module_name}.bias"],
                    pretrained_W = pretrained_module.bias.detach(),
                    task_W      = task_weight[f"{module_name}.bias"],
                    previous_lambda_t=previous_lambda_t,
                    lambda_t=lambda_t,
                )
        else:
            # Tất cả parameter còn lại (LayerNorm, v.v.)
            for param_name, _ in module.named_parameters():
                key = f"{module_name}.{param_name}"
                merged_weight[key] = merge_other_parameters(
                    merged_W    = merged_weight[key],
                    pretrained_W = pretrained_module.get_parameter(param_name).detach(),
                    task_W      = task_weight[key],
                    previous_lambda_t=previous_lambda_t,
                    lambda_t=lambda_t,
                )

    return merged_weight


def normalize_merged_weight(
    base_model: nn.Module,
    merged_weight: dict,
    avg_task_vector_norm: float,
) -> dict:
    """
    Rescale merged task vector về avg_task_vector_norm để tránh magnitude drift.
    Công thức: merged_W = base_W + task_vector * (avg_norm / current_norm)
    """
    vision_encoder = base_model.vision_encoder
    task_vector_norm = get_task_vector_norm(
        merged_weight,
        {k: v.detach() for k, v in vision_encoder.state_dict().items()},
    )

    for param_name, param in vision_encoder.named_parameters():
        base_W = param.detach()
        task_vector = merged_weight[param_name] - base_W
        merged_weight[param_name] = base_W + task_vector * (
            avg_task_vector_norm / task_vector_norm
        )

    return merged_weight, task_vector_norm


# ---------------------------------------------------------------------------
# Main merging pipeline
# ---------------------------------------------------------------------------

def run_merging_for_fold(
    fold_id: int,
    base_model: nn.Module,
    src_dir: str,
    dst_dir: str,
    num_tasks: int,
) -> None:
    """
    Chạy C.OPCM merging cho 1 fold.

    Args:
        fold_id: Index của fold (int).
        base_model: TITAN base model đã load, weights không thay đổi.
        src_dir: Thư mục chứa per-task finetuned checkpoints.
        dst_dir: Thư mục lưu merged checkpoints.
        num_tasks: Số task cần merge.
    """
    fold_name   = f"fold_{fold_id}"
    output_dir  = Path(dst_dir) / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Danh sách path checkpoint của từng task
    task_ckpt_paths = [
        str(Path(src_dir) / fold_name / f"task_{t}.pt")
        for t in range(num_tasks)
    ]

    base_weight = {
        k: v.detach()
        for k, v in base_model.vision_encoder.state_dict().items()
    }

    # Khởi tạo merged_weight = task_0 (chưa cần merge)
    merged_weight        = extract_backbone_weights(task_ckpt_paths[0])
    previous_lambda_t    = 1.0
    avg_task_vector_norm = get_task_vector_norm(merged_weight, base_weight)
    all_task_vector_norms = [avg_task_vector_norm]

    print(f"\n[Fold {fold_id}] Task 0 norm: {avg_task_vector_norm:.4f}")

    # Merge task 1 → num_tasks-1 vào merged_weight
    for model_idx, task_ckpt in enumerate(task_ckpt_paths[1:], start=1):
        task_weight = extract_backbone_weights(task_ckpt)

        all_task_vector_norms.append(get_task_vector_norm(task_weight, base_weight))
        avg_task_vector_norm = float(np.mean(all_task_vector_norms))

        lambda_t = 1.0      # temporary — sẽ được rescale sau merge

        # Merge toàn bộ parameters
        merged_weight = merge_one_task(
            base_model=base_model,
            merged_weight=merged_weight,
            task_weight=task_weight,
            previous_lambda_t=previous_lambda_t,
            lambda_t=lambda_t,
            task_idx=model_idx,
        )

        # Rescale lambda theo task vector norm
        merged_weight, task_vector_norm = normalize_merged_weight(
            base_model, merged_weight, avg_task_vector_norm
        )
        lambda_t           = lambda_t * (task_vector_norm / avg_task_vector_norm)
        previous_lambda_t  = lambda_t

        # Lưu intermediate checkpoint (dùng cho BWT/FWT evaluation)
        intermediate_path = output_dir / f"merged_task_{model_idx}.pth"
        primary_path, mirror_path = save_checkpoint_with_mirror(merged_weight, intermediate_path)
        if mirror_path is not None:
            print(f"[Fold {fold_id}] Task {model_idx} merged → {primary_path} | Mirror: {mirror_path}")
        else:
            print(f"[Fold {fold_id}] Task {model_idx} merged → {primary_path}")

    # Lưu final checkpoint (dùng cho Class-IL evaluation)
    final_path = output_dir / f"merged_final.pth"
    primary_path, mirror_path = save_checkpoint_with_mirror(merged_weight, final_path)
    if mirror_path is not None:
        print(f"[Fold {fold_id}] Final checkpoint → {primary_path} | Mirror: {mirror_path}")
    else:
        print(f"[Fold {fold_id}] Final checkpoint → {primary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C.OPCM Continual Model Merging")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--fold_start", type=int, default=None,
                        help="Override fold start (default: 0 từ config)")
    parser.add_argument("--fold_end",   type=int, default=None,
                        help="Override fold end (default: num_folds từ config)")
    parser.add_argument("--finetuned_checkpoints", type=str, default=None,
                        help="Override cfg.paths.finetuned_checkpoints")
    parser.add_argument("--merged_checkpoints", type=str, default=None,
                        help="Override cfg.paths.merged_checkpoints")
    args = parser.parse_args()

    cfg        = OmegaConf.load(args.config)
    fold_start = args.fold_start if args.fold_start is not None else 0
    fold_end   = args.fold_end   if args.fold_end   is not None else cfg.training.num_folds
    num_tasks  = cfg.training.num_tasks     # thêm field này vào default.yaml

    src_dir = args.finetuned_checkpoints or cfg.paths.finetuned_checkpoints
    dst_dir = args.merged_checkpoints or cfg.paths.merged_checkpoints

    print(f"[INFO] finetuned checkpoints: {src_dir}")
    print(f"[INFO] merged checkpoints: {dst_dir}")

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model.eval()

    for fold_id in range(fold_start, fold_end):
        run_merging_for_fold(
            fold_id=fold_id,
            base_model=base_model,
            src_dir=src_dir,
            dst_dir=dst_dir,
            num_tasks=num_tasks,
        )

    print("\nMerging complete.")
