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

import csv

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_TASKS,
    TASK_NAMES, TITAN_PS_ARG,
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
    task_prompts: torch.Tensor = None,   # [T, 768] — dùng để tính embedding sim
    debug: bool = False,
    fold_id: int = 0,
) -> tuple:
    """
    Task-IL inference với debug tuỳ chọn.

    Khi debug=True và task_prompts được cung cấp, mỗi slide sẽ được log thêm:
      - slide embedding norm
      - cosine similarity với tất cả T task prompts (→ xem CESC bị kéo về đâu)
      - top-1 task sim (predicted task nếu dùng TCP)
      - per-class confidence (softmax prob của từng class trong task)
    """
    preds_all   = []
    probs_all   = []
    targets_all = []
    debug_rows  = []   # chỉ fill khi debug=True

    ps         = torch.tensor(TITAN_PS_ARG).int().to(device)
    T          = task_prompts.shape[0] if task_prompts is not None else 0
    task_names = TASK_NAMES

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            try:
                # backbone forward — trả về [1, 768] embedding
                embed  = model.backbone(features, coords, ps).float()
                logits = model.mlp(embed).float()
            except RuntimeError:
                model.cpu()
                embed  = model.backbone(
                    features.cpu(), coords.cpu(),
                    torch.tensor(TITAN_PS_ARG).int().cpu()
                ).float()
                logits = model.mlp(embed).float()
                model.to(device)
                embed = embed.to(device)

            pred = int(logits.argmax(1))

            if num_classes[task_id] == 2:
                probs      = nn.functional.softmax(logits.float(), dim=1)[:, 1]
                roc_kwargs = {}
            else:
                probs      = nn.functional.softmax(logits.float(), dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(np.array([pred]))
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

            # --- debug logging ---
            if debug and task_prompts is not None:
                embed_f32 = embed.float()  # [1, 768]

                # Cosine sim với từng task prompt
                tp_f32    = task_prompts.float()                     # [T, 768]
                embed_n   = nn.functional.normalize(embed_f32, dim=1)
                tp_n      = nn.functional.normalize(tp_f32, dim=1)
                task_sims = (embed_n @ tp_n.T).float().squeeze(0).cpu().numpy()  # [T]

                tcp_pred_task = int(task_sims.argmax())

                # Softmax class probs tất cả classes (nếu binary thì 2 giá trị)
                class_probs_full = nn.functional.softmax(logits.float(), dim=1).float().squeeze(0).cpu().numpy()

                row = {
                    "fold":          fold_id,
                    "task":          task_id,
                    "task_name":     task_names[task_id] if task_id < len(task_names) else str(task_id),
                    "true_label":    int(label),
                    "pred_label":    pred,
                    "correct":       int(pred == int(label)),
                    "embed_norm":    float(embed_f32.norm().item()),
                    "tcp_pred_task": tcp_pred_task,
                    "tcp_correct":   int(tcp_pred_task == task_id),
                    "max_task_sim":  float(task_sims.max()),
                    "true_task_sim": float(task_sims[task_id]),
                    "sim_margin":    float(task_sims[task_id] - task_sims.max())
                                     if tcp_pred_task != task_id else 0.0,
                }
                # Task sim per task (để biết distribution đầy đủ)
                for t in range(T):
                    row[f"task_sim_{t}_{task_names[t] if t < len(task_names) else t}"] = float(task_sims[t])
                # Class probs
                for c, cp in enumerate(class_probs_full):
                    row[f"class_prob_{c}"] = float(cp)

                debug_rows.append(row)

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
    return metrics, preds_arr, targets_arr, debug_rows


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Task-IL evaluation (upper bound)")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Prefix thư mục merged: {prefix}/fold_{id}/merged_final.pth")
    # --- debug flags ---
    parser.add_argument("--debug",            action="store_true",
                        help="Log per-slide embedding sim, TCP routing, class probs")
    parser.add_argument("--debug_csv",        type=str, default="",
                        help="Path lưu CSV debug. Yêu cầu --debug.")
    parser.add_argument("--debug_tasks",      type=str, default="",
                        help="Chỉ debug các task cụ thể, vd: '3,4,5'. Mặc định debug tất cả.")
    args = parser.parse_args()

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    num_classes = seq_dataset.num_classes

    # Task prompts — dùng để tính embedding similarity trong debug
    task_prompts = None
    if args.debug:
        tp_path = PROJECT_ROOT / "task_prompts.pt"
        if tp_path.exists():
            task_prompts = torch.load(str(tp_path), map_location="cpu").to(device)
            if getattr(cfg.dataset, "order", "forward") == "reverse":
                task_prompts = task_prompts.flip(0)
            print(f"[DEBUG] task_prompts loaded: {task_prompts.shape}")
        else:
            print(f"[WARN] task_prompts.pt not found at {tp_path} — sim columns will be skipped")

    # Parse debug_tasks filter
    debug_task_set = None
    if args.debug_tasks:
        debug_task_set = set(int(t) for t in args.debug_tasks.split(","))
        print(f"[DEBUG] Debugging tasks: {sorted(debug_task_set)}")

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_accs     = []
    overall_baccs    = []
    all_acc_per_task = []
    all_debug_rows   = []   # acumulados

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        merge_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"\nLoading: {merge_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_path), map_location="cpu")
        )

        all_baccs    = []
        all_accs     = []
        acc_per_task = {}

        for task_id in range(num_tasks):
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

            do_debug = args.debug and (
                debug_task_set is None or task_id in debug_task_set
            )

            results, preds_all, targets_all, debug_rows = eval_task(
                test_loader,
                model,
                task_id,
                device,
                num_classes,
                task_prompts = task_prompts if do_debug else None,
                debug        = do_debug,
                fold_id      = fold_id,
            )

            all_debug_rows.extend(debug_rows)

            bacc = balanced_accuracy_score(targets_all, preds_all)
            acc  = sum(preds_all == targets_all) / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            all_baccs.append(bacc)
            all_accs.append(acc)

            # Per-class accuracy
            n_cls = num_classes[task_id]
            per_class_acc = []
            for c in range(n_cls):
                mask = targets_all == c
                per_class_acc.append(
                    f"cls{c}={sum(preds_all[mask]==c)/max(mask.sum(),1)*100:.1f}%"
                )

            # TCP sim summary (từ debug_rows nếu có)
            tcp_info = ""
            if debug_rows and task_prompts is not None:
                tcp_correct = sum(r["tcp_correct"] for r in debug_rows if r["task"]==task_id)
                tcp_total   = sum(1 for r in debug_rows if r["task"]==task_id)
                tcp_info    = f" | TCP={tcp_correct}/{tcp_total}({100*tcp_correct/max(tcp_total,1):.0f}%)"

            print(f"  Fold {fold_id} | {seq_dataset.task_names[task_id]:6s}: "
                  f"BAcc={bacc*100:.2f}%  Acc={acc*100:.2f}%  "
                  f"[{' | '.join(per_class_acc)}]{tcp_info}")

        overall_baccs.append(np.mean(all_baccs))
        overall_accs.append(np.mean(all_accs))
        all_acc_per_task.append(acc_per_task)
        print(f"[Fold {fold_id}] BAcc={np.mean(all_baccs)*100:.4f}%  "
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
        print(f"  {seq_dataset.task_names[t]:6s}: "
              f"{np.mean(accs[t])*100:.4f}% ({np.std(accs[t])*100:.4f}%)")

    # --- Embedding sim summary per task (nếu debug) ---
    if args.debug and all_debug_rows and task_prompts is not None:
        print("\n===== Embedding Similarity Analysis =====")
        T = task_prompts.shape[0]
        for t in range(num_tasks):
            task_name = TASK_NAMES[t] if t < len(TASK_NAMES) else str(t)
            rows_t = [r for r in all_debug_rows if r["task"] == t]
            if not rows_t:
                continue
            tcp_acc = sum(r["tcp_correct"] for r in rows_t) / len(rows_t)
            true_sim  = [r["true_task_sim"] for r in rows_t]
            max_sim   = [r["max_task_sim"]  for r in rows_t]
            print(f"\n  Task {t} {task_name} ({len(rows_t)} slides):")
            print(f"    TCP routing acc  : {tcp_acc*100:.1f}%")
            print(f"    Mean sim (true)  : {sum(true_sim)/len(true_sim):.4f}")
            print(f"    Mean sim (max)   : {sum(max_sim)/len(max_sim):.4f}")
            # Sim to each other task
            print(f"    Mean sim to each task:")
            for t2 in range(T):
                t2_name = TASK_NAMES[t2] if t2 < len(TASK_NAMES) else str(t2)
                col = f"task_sim_{t2}_{t2_name}"
                if col in rows_t[0]:
                    vals = [r[col] for r in rows_t]
                    marker = " ← true" if t2 == t else (
                        " *** HIGHEST" if abs(sum(vals)/len(vals) - max(sum([r[f'task_sim_{x}_{TASK_NAMES[x]}'] for r in rows_t])/len(rows_t) for x in range(T))) < 0.001 else ""
                    )
                    print(f"      → {t2_name:6s}: {sum(vals)/len(vals):.4f}{marker}")

    # --- Save CSV ---
    if args.debug and args.debug_csv and all_debug_rows:
        fieldnames = list(all_debug_rows[0].keys())
        with open(args.debug_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_debug_rows)
        print(f"\n[DEBUG] Saved {len(all_debug_rows)} rows → {args.debug_csv}")
