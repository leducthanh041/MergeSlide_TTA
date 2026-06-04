# test_classIL_task_prompt.py
"""
Class-IL evaluation — hỗ trợ cả TCP và Naive inference.

Modes:
    tcp   (default): Task-to-Class Prompt-Aligned inference
                     t_hat = argmax(Z @ Task_Embeddings.T)
                     y_pred = argmax(Z @ Class_Embeddings[t_hat].T)

    naive          : Direct class inference
                     y_pred = argmax(Z @ All_Class_Embeddings.T)

Usage:
    python test_classIL_task_prompt.py \
        --save_dir /path/to/finetuned_checkpoints \
        --merge_model_path /path/to/merged/checkpoints \
        --mode tcp      # hoặc --mode naive

Cấu trúc checkpoint kỳ vọng:
    Finetuned : {save_dir}/fold_{id}/task_{t}.pt
    Merged    : {merge_model_path}/fold_{id}/merged_final.pth
"""
import argparse
import csv
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


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def eval_task_tcp(
    test_loader,
    task_id: int,
    model: CustomSequential,
    num_classes: list,
    task_prompts: torch.Tensor,
    task_model_paths: list,
    device,
    seq_dataset: Sequential_Generic_MIL_Dataset,
    fold_id: int,
    debug_route: bool = False,
    debug_route_topk: int = 3,
    debug_rows: list | None = None,
) -> tuple:
    """
    TCP inference:
      1. t_hat = argmax(Z @ Task_Embeddings.T)
      2. y_pred = argmax(Z @ Class_Embeddings[t_hat].T)  via MLP head
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []
    route_correct       = 0
    strict_global_correct = 0
    legacy_global_correct = 0
    route_counts        = np.zeros(task_prompts.shape[0], dtype=np.int64)
    route_margins       = []
    route_score_sums    = np.zeros(task_prompts.shape[0], dtype=np.float64)

    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    # Pre-load tất cả MLP weights
    task_weights = []
    for p in task_model_paths:
        state = torch.load(p, map_location="cpu")
        task_weights.append(
            {k.split("mlp.")[-1]: state[k] for k in list(state.keys())[-2:]}
        )

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for sample_idx, (features, coords, label) in enumerate(tqdm(test_loader, leave=False)):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]

            t0 = time.time()

            # Bước 1: task routing
            slide_embed  = model.backbone(features, coords, ps)
            route_logits = (slide_embed.float() @ task_prompts.float().T).squeeze(0)
            pred_task_id = int(torch.argmax(route_logits).item())
            topk = min(debug_route_topk, route_logits.numel())
            top_vals, top_ids = torch.topk(route_logits, k=topk)
            route_margin = 0.0
            if topk > 1:
                route_margin = float((top_vals[0] - top_vals[1]).detach().cpu())

            label_int = int(label)
            route_correct += int(pred_task_id == task_id)
            route_counts[pred_task_id] += 1
            route_margins.append(route_margin)
            route_score_sums += route_logits.detach().float().cpu().numpy()

            # Bước 2: class prediction via MLP head của task dự đoán
            mlp = nn.Linear(EMBED_DIM, num_classes[pred_task_id]).to(device)
            mlp.load_state_dict(task_weights[pred_task_id])
            logits = mlp(slide_embed).float()
            pred   = int(logits.argmax(1))
            times.append(time.time() - t0)

            probs = nn.functional.softmax(logits, dim=1)
            preds_all.append(np.array([pred]))
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

            g_label = seq_dataset.task_to_global_class[task_id].get(label_int, -1)
            # Legacy mapping keeps the original script behavior for reported metrics.
            # Routed mapping exposes the strict Class-IL effect of task routing.
            g_pred  = seq_dataset.task_to_global_class[task_id].get(pred, -1)
            routed_g_pred = seq_dataset.task_to_global_class[pred_task_id].get(pred, -1)
            legacy_global_correct += int(g_pred == g_label)
            strict_global_correct += int(routed_g_pred == g_label)
            convert_targets_all.append(np.array([g_label]))
            convert_preds_all.append(np.array([g_pred]))

            if debug_route and debug_rows is not None:
                row = {
                    "fold": fold_id,
                    "task_id": task_id,
                    "task_name": seq_dataset.task_names[task_id],
                    "sample_idx": sample_idx,
                    "true_local_label": label_int,
                    "true_global_label": g_label,
                    "pred_task_id": pred_task_id,
                    "pred_task_name": seq_dataset.task_names[pred_task_id],
                    "route_correct": int(pred_task_id == task_id),
                    "pred_local_label": pred,
                    "legacy_global_pred": g_pred,
                    "routed_global_pred": routed_g_pred,
                    "legacy_global_correct": int(g_pred == g_label),
                    "routed_global_correct": int(routed_g_pred == g_label),
                    "route_margin": route_margin,
                }
                for rank, (route_id, route_score) in enumerate(zip(top_ids, top_vals), start=1):
                    route_id_int = int(route_id.item())
                    row[f"top{rank}_task_id"] = route_id_int
                    row[f"top{rank}_task_name"] = seq_dataset.task_names[route_id_int]
                    row[f"top{rank}_score"] = float(route_score.detach().cpu())
                debug_rows.append(row)

    if debug_route:
        n_samples = max(1, len(test_loader))
        avg_scores = route_score_sums / n_samples
        route_dist = ", ".join(
            f"{seq_dataset.task_names[t]}={route_counts[t]}"
            for t in range(len(route_counts))
            if route_counts[t] > 0
        )
        avg_score_text = ", ".join(
            f"{seq_dataset.task_names[t]}={avg_scores[t]:.4f}"
            for t in range(len(avg_scores))
        )
        print(
            f"[ROUTE][Fold {fold_id}][Task {task_id} {seq_dataset.task_names[task_id]}] "
            f"route_acc={route_correct/n_samples*100:.2f}% "
            f"legacy_global_acc={legacy_global_correct/n_samples*100:.2f}% "
            f"strict_routed_global_acc={strict_global_correct/n_samples*100:.2f}% "
            f"avg_margin={np.mean(route_margins) if route_margins else 0.0:.4f} "
            f"route_dist={{ {route_dist} }}"
        )
        print(
            f"[ROUTE][Fold {fold_id}][Task {task_id} {seq_dataset.task_names[task_id]}] "
            f"avg_route_scores={{ {avg_score_text} }}"
        )

    return _pack_results(
        preds_all, targets_all, probs_all,
        convert_preds_all, convert_targets_all, times,
    )


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
    Naive inference (giống test_classIL.py gốc):
      - Build merged MLP head bằng cách cat weights của tất cả per-task MLP.
      - y_pred = argmax(merged_mlp(Z))  →  global class index  [0..12]
      - Label được convert sang global space qua task_to_global_class.
 
    Không dùng text embeddings. Merged MLP có shape Linear(768, sum(num_classes)).
    Column i của output tương ứng trực tiếp với global class i theo thứ tự
    ghép nối của task_to_global_class.
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []
 
    ps             = torch.tensor(TITAN_PS_ARG).int().to(device)
    total_classes  = sum(num_classes)
 
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
    # (built từ task_to_global_class, theo đúng thứ tự ghép nối weights)
    column_to_global = np.array([
        task_to_global_class[t][local]
        for t in range(len(task_to_global_class))
        for local in sorted(task_to_global_class[t].keys())
    ])
 
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            features = features.to(device)
            coords   = coords.long().to(device)
            idx      = torch.randperm(features.shape[0])[:K_PATCHES]
            features, coords = features[idx], coords[idx]
 
            t0 = time.time()
 
            slide_embed   = model.backbone(features, coords, ps)
            logits        = merged_mlp(slide_embed.float())        # [1, total_classes]
            pred_column   = int(logits.argmax(1))
            pred_global   = int(column_to_global[pred_column])
            true_global   = task_to_global_class[task_id][int(label)]
 
            times.append(time.time() - t0)
 
            probs_np  = nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            probs_out = np.zeros((1, total_classes), dtype=np.float32)
            for col_idx, g_idx in enumerate(column_to_global):
                probs_out[0, g_idx] = probs_np[col_idx]
 
            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))
 
            convert_preds_all.append(np.array([pred_global]))
            convert_targets_all.append(np.array([true_global]))
 
    return _pack_results(
        preds_all, targets_all, probs_all,
        convert_preds_all, convert_targets_all, times,
    )


def _pack_results(
    preds_all, targets_all, probs_all,
    convert_preds_all, convert_targets_all, times,
) -> tuple:
    """Gộp list → numpy array và tính metrics. Dùng chung cho cả 2 modes."""
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
    return (
        metrics, preds_arr, targets_arr, probs_arr,
        np.concatenate(convert_preds_all),
        np.concatenate(convert_targets_all),
        sum(times),
    )


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

    parser = argparse.ArgumentParser(description="Class-IL evaluation")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Root dir chứa merged checkpoints: {path}/fold_{id}/merged_final.pth")
    parser.add_argument("--mode",             type=str, default="tcp",
                        choices=["tcp", "naive"],
                        help="tcp (default): TCP inference | naive: direct class inference")
    parser.add_argument(
        "--debug_route",
        action="store_true",
        help="Print per-task TCP routing diagnostics and optionally save per-slide route rows.",
    )
    parser.add_argument(
        "--debug_route_csv",
        type=str,
        default="",
        help="CSV path for per-slide TCP routing diagnostics. Only used with --debug_route.",
    )
    parser.add_argument(
        "--debug_route_topk",
        type=int,
        default=3,
        help="Number of top TCP route scores to save in debug CSV.",
    )
    args = parser.parse_args()

    local_hot_root = ensure_local_hot_storage()
    args.save_dir = str(resolve_hot_path(args.save_dir, local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    if args.debug_route_csv:
        args.debug_route_csv = str(resolve_hot_path(args.debug_route_csv, local_hot_root))
    print(f"[INFO] Local hot storage root: {local_hot_root}")
    print(f"[INFO] Finetuned checkpoints: {args.save_dir}")
    print(f"[INFO] Merged checkpoints: {args.merge_model_path}")
    if args.debug_route:
        print(f"[INFO] TCP route debug enabled; csv={args.debug_route_csv or '<disabled>'}")

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks    = cfg.training.num_tasks
    seq_dataset  = Sequential_Generic_MIL_Dataset(cfg)
    num_classes  = seq_dataset.num_classes

    # Load embeddings tuỳ theo mode
    if args.mode == "tcp":
        task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
        if getattr(cfg.dataset, 'order', 'forward') == 'reverse':
            task_prompts = task_prompts.flip(0)
        all_class_embeddings = None
    else:
        task_prompts        = None

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
    route_debug_rows     = []

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        merge_model_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"Loading: {merge_model_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_model_path), map_location="cpu")
        )
        model = CustomSequential(base_model, nn.Identity()).eval()

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]

        num_correct   = 0.0
        num_total     = 0.0
        all_baccs     = []
        all_accs      = []
        all_aucs      = []
        all_preds_g   = []
        all_targets_g = []
        acc_per_task  = {}
        fold_time     = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            if args.mode == "tcp":
                result = eval_task_tcp(
                    test_loader, task_id, model, num_classes,
                    task_prompts, task_model_paths, device,
                    seq_dataset=seq_dataset,
                    fold_id=fold_id,
                    debug_route=args.debug_route,
                    debug_route_topk=args.debug_route_topk,
                    debug_rows=route_debug_rows,
                )
            else:
                result = eval_task_naive(
                    test_loader, task_id, model,
                    task_model_paths, num_classes, device,
                    task_to_global_class=seq_dataset.task_to_global_class,
                )

            results, preds_all, targets_all, probs_all, \
                conv_preds, conv_targets, task_time = result

            num_correct += sum(preds_all == targets_all)
            num_total   += len(test_loader)
            fold_time   += task_time / len(test_loader)

            task_acc = sum(preds_all == targets_all) / len(test_loader)
            task_bacc = balanced_accuracy_score(targets_all, preds_all)
            acc_per_task[task_id] = results["/acc"]
            all_baccs.append(task_bacc)
            all_accs.append(task_acc)
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            print(
                f"  [Fold {fold_id}] Task {task_id} ({seq_dataset.task_names[task_id]}) "
                f"ACC={task_acc*100:.4f}% BAcc={task_bacc*100:.4f}%"
            )

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)
            if args.mode == "tcp":
                for i in range(num_classes[task_id]):           # local index
                    all_aucs.append(
                        roc_auc_score((targets_all == i).astype(int), probs_all[:, i])
                    )
            else:   # naive: targets_all là global index
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for g_idx in global_idxs:
                    all_aucs.append(
                        roc_auc_score((targets_all == g_idx).astype(int), probs_all[:, g_idx])
                    )

        all_preds_g   = np.concatenate(all_preds_g)
        all_targets_g = np.concatenate(all_targets_g)

        overall_accs.append(np.mean(all_accs))
        overall_baccs.append(np.mean(all_baccs))
        overall_macro_f1s.append(f1_score(all_targets_g, all_preds_g, average="macro"))
        overall_weighted_f1s.append(f1_score(all_targets_g, all_preds_g, average="weighted"))
        overall_recalls.append(recall_score(all_targets_g, all_preds_g, average=None))
        overall_precisions.append(precision_score(all_targets_g, all_preds_g, average=None))
        overall_aucs.append(np.array(all_aucs))
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}% "
              f"BAcc={np.mean(all_baccs)*100:.4f}%")

    mode_label = "TCP" if args.mode == "tcp" else "Naive"
    print(f"\n===== Class-IL ({mode_label}) Results =====")
    print(f"Accuracy:        {np.mean(overall_accs)*100:.4f}% ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:    {np.mean(overall_baccs)*100:.4f}% ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:        {np.mean(overall_macro_f1s)*100:.4f}% ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:     {np.mean(overall_weighted_f1s)*100:.4f}% ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time:  {np.mean(overall_times):.3f}s ({np.std(overall_times):.3f}s)")

    print("\nRecall per class:")
    for v, s in zip(np.mean(np.stack(overall_recalls), axis=0),
                    np.std(np.stack(overall_recalls), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nPrecision per class:")
    for v, s in zip(np.mean(np.stack(overall_precisions), axis=0),
                    np.std(np.stack(overall_precisions), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nAUC per class:")
    for v, s in zip(np.mean(np.stack(overall_aucs), axis=0),
                    np.std(np.stack(overall_aucs), axis=0)):
        print(f"  {v*100:.4f}% ({s*100:.4f}%)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs[t])*100:.4f}% ({np.std(accs[t])*100:.4f}%)")

    if args.debug_route and args.debug_route_csv:
        debug_csv_path = Path(args.debug_route_csv)
        debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in route_debug_rows for key in row.keys()})
        with debug_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(route_debug_rows)
        print(f"\n[INFO] Saved TCP route debug CSV: {debug_csv_path}")
