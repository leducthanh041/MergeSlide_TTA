"""
test_tta_v3.py
==============
Test-Time Adaptation evaluation — MergeSlide-TTA v3.

Changes vs. test_tta.py (v2):
    - Import MergeSlide_TTA_Adapter_v3 và TTAConfig_v3 từ tta_engine_v3
    - Diagnostics bổ sung: loss_task, task_margin, prompt_updated_rate
    - Tất cả phần còn lại giữ nguyên hoàn toàn

Modes:
    classil_tcp    : CLASS-IL + TCP Confidence Gate
    classil_naive  : CLASS-IL + Global MLP (no TCP routing)
    taskil         : TASK-IL  (ground-truth task_id provided)

Usage::
    python test_tta_v3.py \\
        --config   configs/default_tta_eval_num_workers0.yaml \\
        --save_dir checkpoints/finetuned \\
        --merge_model_path checkpoints/merged \\
        --swag_dir checkpoints/swag_diagonal \\
        --mode classil_tcp
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
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from mergeslide_tta.constants import (
    EMBED_DIM, K_PATCHES, NUM_TASKS,
    TASK_NAMES_FORWARD, TASK_CLASS_RANGES_FORWARD,
    TITAN_PS_ARG,
)
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.metrics import pad_numpy_arrays
from mergeslide_tta.model import build_prompt_classifier
from mergeslide_tta.swag_diagonal import SWAGDiagonal

# V3: import engine mới
from mergeslide_tta.tta_engine_v3 import MergeSlide_TTA_Adapter_v3, TTAConfig_v3

from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT  = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "checkpoints_ood", "logs", "sqlite"}


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers (giữ nguyên từ test_tta.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_local_hot_root() -> Path:
    user = os.environ.get("USER") or "thanhld"
    default = Path("/docker/data") / user / PROJECT_ROOT.name
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", default)).expanduser()


def ensure_local_hot_storage() -> Path:
    local_root = get_local_hot_root()
    local_root.mkdir(parents=True, exist_ok=True)
    for name in HOT_DIR_NAMES | {"tmp"}:
        (local_root / name).mkdir(parents=True, exist_ok=True)
    for name in ("logs", "checkpoints"):
        rp = PROJECT_ROOT / name
        lp = local_root / name
        if rp.is_symlink():
            if rp.resolve() != lp.resolve():
                print(f"[WARN] {rp} points to {rp.resolve()}, expected {lp}")
        elif rp.exists():
            print(f"[WARN] {rp} is not a symlink; use {lp} for hot-write data.")
        else:
            rp.symlink_to(lp, target_is_directory=True)
    os.environ.setdefault("TMPDIR",              str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR",       str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return local_root


def resolve_hot_path(path: str, local_root: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        repo_path = PROJECT_ROOT / raw
        if repo_path.exists():
            return repo_path.resolve()
        parts = raw.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return local_root.joinpath(*parts)
        return raw
    if raw.exists():
        return raw.resolve()
    try:
        rel = raw.relative_to(PROJECT_ROOT)
        if rel.parts and rel.parts[0] in HOT_DIR_NAMES:
            return local_root / rel
    except ValueError:
        pass
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Build MLP weight matrices (giữ nguyên từ test_tta.py)
# ──────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_RANGE_BY_NAME: dict[str, list[int]] = {
    name: TASK_CLASS_RANGES_FORWARD[i]
    for i, name in enumerate(TASK_NAMES_FORWARD)
}


def build_per_task_mlp_weights(classifier, task_names, device):
    out = {}
    for t, name in enumerate(task_names):
        start, end = _CLASSIFIER_RANGE_BY_NAME[name]
        w = classifier[:, start:end + 1].T.contiguous().to(device)
        b = torch.zeros(end - start + 1, device=device)
        out[t] = (w, b)
    return out


def build_global_mlp_weights(classifier, task_names, device):
    parts = [
        classifier[:, _CLASSIFIER_RANGE_BY_NAME[n][0]:_CLASSIFIER_RANGE_BY_NAME[n][1] + 1]
        for n in task_names
    ]
    global_w = torch.cat(parts, dim=1).T.contiguous().to(device)
    return global_w, torch.zeros(13, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# One-fold, one-task evaluation loop — V3
# ──────────────────────────────────────────────────────────────────────────────

def eval_task_tta_v3(
    test_loader,
    task_id: int,
    adapter: MergeSlide_TTA_Adapter_v3,
    seq_dataset: Sequential_Generic_MIL_Dataset,
    fold_id: int,
    mode: str,
    inference_model: str,
    reset_per_slide: bool,
    device: torch.device,
) -> tuple:
    """
    Evaluation loop cho một task + fold — v3.

    Giống eval_task_tta từ test_tta.py, bổ sung tracking:
      - loss_task per slide
      - task_margin per slide (gap top1 - top2 task scores)
      - prompt_updated per slide (bool)
    """
    preds_all:           list[np.ndarray] = []
    targets_all:         list[np.ndarray] = []
    probs_all:           list[np.ndarray] = []
    convert_preds_all:   list[np.ndarray] = []
    convert_targets_all: list[np.ndarray] = []
    tta_stats_list:      list[dict]       = []
    times: list[float] = []

    task_name = seq_dataset.task_names[task_id]
    is_naive  = (mode == "classil_naive")

    for sample_idx, (features, coords, label) in enumerate(
        tqdm(test_loader, desc=f"  Task {task_id} {task_name}", leave=False)
    ):
        features  = features.to(device)
        coords    = coords.long().to(device)
        label_int = int(label)

        if reset_per_slide:
            adapter.reset_to_source()

        t0 = time.time()

        if mode == "taskil":
            pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(
                features, coords, task_id=task_id, use_tcp_gate=False,
                inference_model=inference_model,
            )
        elif is_naive:
            pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(
                features, coords, task_id=None, use_tcp_gate=False,
                inference_model=inference_model,
            )
        else:  # classil_tcp
            pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(
                features, coords, task_id=None, use_tcp_gate=True,
                inference_model="teacher",
            )

        times.append(time.time() - t0)

        if is_naive:
            g_pred  = seq_dataset.task_to_global_class[pred_task].get(pred_local, -1)
            g_label = seq_dataset.task_to_global_class[task_id].get(label_int, -1)
            preds_all.append(np.array([g_pred]))
            targets_all.append(np.array([g_label]))
            probs_all.append(prob_np.reshape(1, -1))
        else:
            preds_all.append(np.array([pred_local]))
            targets_all.append(np.array([label_int]))
            probs_all.append(prob_np.reshape(1, -1))

        g_label = seq_dataset.task_to_global_class[task_id].get(label_int, -1)
        strict_g_pred = seq_dataset.task_to_global_class[pred_task].get(pred_local, -1)
        if mode == "classil_tcp":
            g_pred = seq_dataset.task_to_global_class[task_id].get(pred_local, -1)
        else:
            g_pred = strict_g_pred
        convert_preds_all.append(np.array([g_pred]))
        convert_targets_all.append(np.array([g_label]))

        tta_stats_list.append({
            "fold": fold_id, "task_id": task_id, "task_name": task_name,
            "sample_idx": sample_idx, "true_local": label_int,
            "pred_local": pred_local, "pred_task": pred_task,
            "g_label": g_label, "g_pred": g_pred,
            "strict_g_pred": strict_g_pred,
            "global_correct": int(g_pred == g_label),
            "strict_global_correct": int(strict_g_pred == g_label),
            **debug,
        })

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
        sum(times), tta_stats_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="MergeSlide TTA v3 evaluation")
    parser.add_argument("--config",            type=str,
                        default="configs/default_tta_eval_num_workers0.yaml")
    parser.add_argument("--save_dir",          type=str, required=True)
    parser.add_argument("--merge_model_path",  type=str, required=True)
    parser.add_argument("--swag_dir",          type=str, required=True)
    parser.add_argument("--mode",              type=str, default="classil_tcp",
                        choices=["classil_tcp", "classil_naive", "taskil"])
    parser.add_argument(
        "--inference_model",
        type=str,
        choices=["teacher", "student"],
        default=None,
        help=(
            "Final prediction backbone for classil_naive/taskil. "
            "Overrides tta.inference_model; classil_tcp always uses teacher."
        ),
    )
    parser.add_argument("--no_reset_per_task", action="store_true")
    parser.add_argument("--episodic",          action="store_true")
    parser.add_argument("--result_csv",        type=str, default="")
    parser.add_argument("--tta_stats_csv",     type=str, default="")
    parser.add_argument("--fold_start",        type=int, default=0)
    parser.add_argument("--fold_end",          type=int, default=None)
    args = parser.parse_args()

    local_root = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_root))
    args.swag_dir         = str(resolve_hot_path(args.swag_dir,         local_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_folds       = cfg.training.num_folds
    fold_start      = args.fold_start
    fold_end        = args.fold_end or num_folds
    reset_per_task  = not args.no_reset_per_task
    reset_per_slide = args.episodic

    print(f"[INFO] MergeSlide-TTA v3  mode={args.mode}  "
          f"reset_per_task={reset_per_task}  episodic={reset_per_slide}")
    print(f"[INFO] folds: {fold_start} → {fold_end}")
    print(f"[INFO] swag_dir: {args.swag_dir}")

    # ── TTA v3 config ────────────────────────────────────────────────────────
    raw_tta = OmegaConf.to_container(cfg.get("tta", OmegaConf.create({})), resolve=True)
    tta_cfg = TTAConfig_v3(**{k: v for k, v in raw_tta.items() if hasattr(TTAConfig_v3, k)})
    tta_cfg.k_patches_std = K_PATCHES
    configured_inference_model = str(tta_cfg.inference_model).strip().lower()
    if configured_inference_model not in {"teacher", "student"}:
        parser.error(
            "tta.inference_model must be 'teacher' or 'student', "
            f"got {tta_cfg.inference_model!r}"
        )
    inference_model = args.inference_model or configured_inference_model
    if args.mode == "classil_tcp":
        inference_model = "teacher"
    print(f"[INFO] TTAConfig_v3: {tta_cfg}")
    print(f"[INFO] final inference model for mode={args.mode}: {inference_model}")

    # ── Dataset + prompt artifacts ───────────────────────────────────────────
    seq_dataset   = Sequential_Generic_MIL_Dataset(cfg)
    num_tasks     = cfg.training.num_tasks
    num_classes   = seq_dataset.num_classes
    total_classes = sum(num_classes)
    order         = getattr(cfg.dataset, "order", "forward")

    print("Building prompt classifier ...")
    classifier, _ = build_prompt_classifier(str(device))

    task_prompts: torch.Tensor = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    if order == "reverse":
        task_prompts = task_prompts.flip(0)

    per_task_mlp       = build_per_task_mlp_weights(classifier, seq_dataset.task_names, device)
    global_w, global_b = build_global_mlp_weights(classifier, seq_dataset.task_names, device)

    print("Loading TITAN base model ...")
    base_titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_titan = base_titan.to(device).eval()

    # ── Per-fold accumulators ────────────────────────────────────────────────
    overall_accs:         list[float]       = []
    overall_baccs:        list[float]       = []
    overall_macro_f1s:    list[float]       = []
    overall_weighted_f1s: list[float]       = []
    overall_recalls:      list[np.ndarray]  = []
    overall_precisions:   list[np.ndarray]  = []
    overall_aucs:         list[np.ndarray]  = []
    overall_times:        list[float]       = []
    all_acc_per_task:     list[dict]        = []
    overall_tta_diag:     list[dict]        = []
    # V3: routing accuracy accumulators
    overall_routing_acc:           list[float]       = []   # mean across tasks
    overall_routing_acc_per_task:  list[dict]        = []   # {task_id: float}

    all_results:   list[dict] = []
    all_tta_stats: list[dict] = []

    for fold_id in tqdm(range(fold_start, fold_end), desc="Folds"):
        fold_name   = f"fold_{fold_id}"
        merged_path = Path(args.merge_model_path) / fold_name / "merged_final.pth"
        swag_path   = Path(args.swag_dir) / f"{fold_name}.pt"

        if not merged_path.exists():
            print(f"[WARN] merged_final not found: {merged_path} — skipping"); continue
        if not swag_path.exists():
            print(f"[WARN] SWAG not found: {swag_path} — skipping"); continue

        backbone_sd     = torch.load(str(merged_path), map_location="cpu")
        mean_sd, var_sd = SWAGDiagonal.load(str(swag_path), device="cpu")

        print(f"\n[Fold {fold_id}] Building TTA v3 adapter ...")
        adapter = MergeSlide_TTA_Adapter_v3(
            base_vision_encoder  = base_titan.vision_encoder,
            backbone_sd          = backbone_sd,
            mean_sd              = mean_sd,
            var_sd               = var_sd,
            per_task_mlp_weights = per_task_mlp,
            global_mlp_weight    = global_w,
            global_mlp_bias      = global_b,
            task_prompts         = task_prompts,
            task_class_ranges    = seq_dataset.task_class_ranges,
            cfg                  = tta_cfg,
            device               = device,
        )
        trainable = sum(p.numel() for p in adapter.student.parameters() if p.requires_grad)
        print(f"[Fold {fold_id}] Trainable LN params: {trainable:,}")

        all_accs:   list[float] = []
        all_baccs:  list[float] = []
        all_aucs_fold = np.full(total_classes, np.nan, dtype=float)
        all_preds_g:   list[np.ndarray] = []
        all_targets_g: list[np.ndarray] = []
        acc_per_task:  dict[int, float] = {}
        fold_time: float = 0.0
        # V3: routing accuracy per task trong fold này
        routing_acc_per_task: dict[int, float] = {}

        fold_diag = {
            "avg_ood_score":        [],
            "avg_tcp_conf":         [],
            "aug_rate":             [],
            "avg_loss_petal":       [],
            "avg_loss_class":       [],
            "avg_loss_task":        [],    # V3 NEW
            "avg_task_margin":      [],    # V3 NEW
            "prompt_updated_rate":  [],    # V3 NEW
        }

        for task_id in range(num_tasks):
            task_name = seq_dataset.task_names[task_id]
            n_cls     = num_classes[task_id]

            if reset_per_task and task_id > 0:
                adapter.reset_to_source()

            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            (metrics, preds_arr, targets_arr, probs_arr,
             conv_preds, conv_targets, elapsed, tta_stats) = eval_task_tta_v3(
                test_loader     = test_loader,
                task_id         = task_id,
                adapter         = adapter,
                seq_dataset     = seq_dataset,
                fold_id         = fold_id,
                mode            = args.mode,
                inference_model = inference_model,
                reset_per_slide = reset_per_slide,
                device          = device,
            )

            n_samples = len(test_loader)
            task_acc  = float(np.sum(preds_arr == targets_arr)) / max(n_samples, 1)
            task_bacc = balanced_accuracy_score(targets_arr, preds_arr)
            fold_time += elapsed / max(n_samples, 1)

            all_accs.append(task_acc)
            all_baccs.append(task_bacc)
            acc_per_task[task_id] = metrics.get("/acc", task_acc)
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            # ── Per-class AUC ────────────────────────────────────────────────
            if args.mode == "classil_naive":
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for g_idx in global_idxs:
                    if probs_arr.shape[1] > g_idx:
                        try:
                            all_aucs_fold[g_idx] = roc_auc_score(
                                (targets_arr == g_idx).astype(int), probs_arr[:, g_idx]
                            )
                        except ValueError:
                            pass
            else:
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for i in range(n_cls):
                    if i < probs_arr.shape[1]:
                        try:
                            all_aucs_fold[global_idxs[i]] = roc_auc_score(
                                (targets_arr == i).astype(int), probs_arr[:, i]
                            )
                        except ValueError:
                            pass

            # ── TTA diagnostics ──────────────────────────────────────────────
            ood_scores      = [s["ood_score"]       for s in tta_stats]
            tcp_confs       = [s["tcp_conf"]        for s in tta_stats]
            loss_petals     = [s["loss_petal"]      for s in tta_stats]
            loss_classes    = [s["loss_class"]      for s in tta_stats]
            loss_tasks      = [s["loss_task"]       for s in tta_stats]   # V3
            task_margins    = [s["task_margin"]     for s in tta_stats]   # V3
            prompt_updates  = [s["prompt_updated"]  for s in tta_stats]   # V3
            aug_rate        = sum(s["use_aug"] for s in tta_stats) / max(1, len(tta_stats))
            prompt_upd_rate = sum(prompt_updates) / max(1, len(prompt_updates))   # V3

            # V3: routing accuracy cho task này
            # Chỉ có ý nghĩa với classil_tcp (naive và taskil không dùng routing)
            if args.mode == "classil_tcp":
                routing_correct = sum(
                    1 for s in tta_stats if s["pred_task"] == task_id
                )
                task_routing_acc = routing_correct / max(1, len(tta_stats))
            else:
                task_routing_acc = float("nan")
            routing_acc_per_task[task_id] = task_routing_acc

            fold_diag["avg_ood_score"].append(np.mean(ood_scores))
            fold_diag["avg_tcp_conf"].append(np.mean(tcp_confs))
            fold_diag["aug_rate"].append(aug_rate)
            fold_diag["avg_loss_petal"].append(np.mean(loss_petals))
            fold_diag["avg_loss_class"].append(np.mean(loss_classes))
            fold_diag["avg_loss_task"].append(np.mean(loss_tasks))
            fold_diag["avg_task_margin"].append(np.mean(task_margins))
            fold_diag["prompt_updated_rate"].append(prompt_upd_rate)

            print(
                f"  [Fold {fold_id}][Task {task_id} {task_name}] "
                f"ACC={task_acc*100:.4f}%  BAcc={task_bacc*100:.4f}%  "
                f"OOD={np.mean(ood_scores):.3f}  TCP={np.mean(tcp_confs):.3f}  "
                f"L_task={np.mean(loss_tasks):.4f}  "
                f"margin={np.mean(task_margins):.3f}  "
                f"prompt_upd={prompt_upd_rate*100:.1f}%  "
                f"routing_acc={task_routing_acc*100:.1f}%"
            )

            all_results.append({
                "fold": fold_id, "task_id": task_id, "task_name": task_name,
                "mode": args.mode, "bacc": task_bacc, "acc": task_acc,
                "inference_model": inference_model,
                "n_samples": n_samples, "elapsed_s": elapsed,
                "avg_ood_score":        np.mean(ood_scores),
                "avg_tcp_conf":         np.mean(tcp_confs),
                "aug_rate":             aug_rate,
                "avg_loss_petal":       np.mean(loss_petals),
                "avg_loss_class":       np.mean(loss_classes),
                "avg_loss_task":        np.mean(loss_tasks),
                "avg_task_margin":      np.mean(task_margins),
                "prompt_updated_rate":  prompt_upd_rate,
                "routing_acc":          task_routing_acc,   # V3
            })
            all_tta_stats.extend(tta_stats)

        # ── Fold-level global metrics ────────────────────────────────────────
        all_preds_g_cat   = np.concatenate(all_preds_g)
        all_targets_g_cat = np.concatenate(all_targets_g)

        fold_macro_f1    = f1_score(all_targets_g_cat, all_preds_g_cat, average="macro",    zero_division=0)
        fold_weighted_f1 = f1_score(all_targets_g_cat, all_preds_g_cat, average="weighted", zero_division=0)
        fold_recall      = recall_score(all_targets_g_cat, all_preds_g_cat, average=None,   zero_division=0)
        fold_precision   = precision_score(all_targets_g_cat, all_preds_g_cat, average=None,zero_division=0)

        overall_accs.append(np.mean(all_accs))
        overall_baccs.append(np.mean(all_baccs))
        overall_macro_f1s.append(fold_macro_f1)
        overall_weighted_f1s.append(fold_weighted_f1)
        overall_recalls.append(fold_recall)
        overall_precisions.append(fold_precision)
        overall_aucs.append(all_aucs_fold)
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)
        overall_tta_diag.append({k: np.mean(v) for k, v in fold_diag.items()})
        # V3: routing accuracy
        valid_routing = [v for v in routing_acc_per_task.values() if not np.isnan(v)]
        overall_routing_acc.append(np.mean(valid_routing) if valid_routing else float("nan"))
        overall_routing_acc_per_task.append(routing_acc_per_task)

        print(
            f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}%  "
            f"BAcc={np.mean(all_baccs)*100:.4f}%  "
            f"MacroF1={fold_macro_f1*100:.4f}%"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    mode_label = {"classil_tcp": "TCP", "classil_naive": "Naive", "taskil": "TASK-IL"}[args.mode]
    print(f"\n===== TTA v3 Class-IL ({mode_label}) Results =====")
    print(f"Accuracy:        {np.mean(overall_accs)*100:.4f}% ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:    {np.mean(overall_baccs)*100:.4f}% ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:        {np.mean(overall_macro_f1s)*100:.4f}% ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:     {np.mean(overall_weighted_f1s)*100:.4f}% ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time:  {np.mean(overall_times):.3f}s ({np.std(overall_times):.3f}s)")

    def _fmt_per_class(arrays: list[np.ndarray], label: str) -> None:
        stacked = np.stack([a for a in arrays if len(a) > 0])
        if np.isnan(stacked).any():
            means = np.nanmean(stacked, axis=0)
            stds  = np.nanstd(stacked, axis=0)
        else:
            means = np.mean(stacked, axis=0)
            stds  = np.std(stacked, axis=0)
        print(f"\n{label}:")
        for m, s in zip(means, stds):
            if np.isnan(m):
                print("  nan% (nan%)")
            else:
                print(f"  {m*100:.4f}% ({s*100:.4f}%)")

    _fmt_per_class(overall_recalls,    "Recall per class")
    _fmt_per_class(overall_precisions, "Precision per class")
    _fmt_per_class(overall_aucs,       "AUC per class")

    print("\nAcc per task:")
    accs_by_task = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs_by_task[t].append(fold_acc.get(t, 0.0))
    for t in range(num_tasks):
        v = np.mean(accs_by_task[t])
        s = np.std(accs_by_task[t])
        print(f"  Task {t}: {v*100:.4f}% ({s*100:.4f}%)")

    # V3: routing accuracy summary — primary metric cho routing tune
    print("\nRouting Accuracy per task (pred_task == true_task):")
    routing_by_task = {t: [] for t in range(num_tasks)}
    for fold_r in overall_routing_acc_per_task:
        for t in range(num_tasks):
            v = fold_r.get(t, float("nan"))
            if not np.isnan(v):
                routing_by_task[t].append(v)
    for t in range(num_tasks):
        vals = routing_by_task[t]
        if vals:
            print(f"  Routing Task {t}: {np.mean(vals)*100:.4f}% ({np.std(vals)*100:.4f}%)")
        else:
            print(f"  Routing Task {t}: N/A (naive/taskil mode)")
    overall_routing_valid = [v for v in overall_routing_acc if not np.isnan(v)]
    if overall_routing_valid:
        print(f"Routing Accuracy (mean): {np.mean(overall_routing_valid)*100:.4f}% "
              f"({np.std(overall_routing_valid)*100:.4f}%)")

    # V3: diagnostics bổ sung
    print("\n===== TTA v3 Diagnostics =====")
    diag_keys = [
        "avg_ood_score", "avg_tcp_conf", "aug_rate",
        "avg_loss_petal", "avg_loss_class",
        "avg_loss_task", "avg_task_margin", "prompt_updated_rate",  # V3
    ]
    for k in diag_keys:
        vals = [d[k] for d in overall_tta_diag]
        print(f"  {k:25s}: {np.mean(vals):.4f} (±{np.std(vals):.4f})")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if all_results:
        out_csv = args.result_csv or f"logs/tta_v3_results_{args.mode}.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader(); writer.writerows(all_results)
        print(f"\n[INFO] Results saved → {out_csv}")

    if all_tta_stats and args.tta_stats_csv:
        Path(args.tta_stats_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.tta_stats_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_tta_stats[0].keys()))
            writer.writeheader(); writer.writerows(all_tta_stats)
        print(f"[INFO] TTA stats saved → {args.tta_stats_csv}")

    print("\nDone.")
