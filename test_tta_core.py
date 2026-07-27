"""
test_tta_core.py
=================
Test-Time Adaptation evaluation — MergeSlide-TTA-Core (no TCP, no L_task).

Mirror có chủ đích của test_tta_v3.py, nhưng:
    - Chỉ chạy naive/global objective (không có --mode; luôn tương đương
      classil_naive của v3, nhưng dùng engine riêng tta_engine_core.py
      thay vì tta_engine_v3.py với use_tcp_gate=False).
    - KHÔNG cần task_prompts.pt, KHÔNG cần per_task_mlp_weights.
    - Log thêm h_bar, jsd_k, sample_active, conf_thresh, agree_thresh mỗi
      slide (phục vụ AC-8/AC-9/AC-10/AC-12 — xem tools/analyze_module_f_reliability.py).
    - reset_task_boundary() được gọi ở đầu mỗi task (reset cửa sổ trượt
      Module F — KHÔNG reset model, giữ đúng continual-adaptation mặc định).

Usage::
    python test_tta_core.py \\
        --config   configs/default_tta_core_eval_num_workers0.yaml \\
        --merge_model_path checkpoints/merged \\
        --swag_dir checkpoints/swag_diagonal \\
        --result_csv logs/tta_core_results.csv \\
        --tta_stats_csv logs/tta_core_stats.csv
"""

import argparse
import csv
import json
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

# CORE: import engine mới (không TCP, không L_task, có Module F)
from mergeslide_tta.tta_engine_core import MergeSlide_TTA_Adapter_Core, TTAConfig_Core

from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT  = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "checkpoints_ood", "logs", "sqlite"}


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers (giữ nguyên từ test_tta_v3.py)
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
        parts = raw.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return local_root.joinpath(*parts)
        return raw
    try:
        rel = raw.relative_to(PROJECT_ROOT)
        if rel.parts and rel.parts[0] in HOT_DIR_NAMES:
            return local_root / rel
    except ValueError:
        pass
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Build global MLP weight matrix (giống test_tta_v3.py, chỉ giữ phần global)
# ──────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_RANGE_BY_NAME: dict[str, list[int]] = {
    name: TASK_CLASS_RANGES_FORWARD[i]
    for i, name in enumerate(TASK_NAMES_FORWARD)
}


def build_global_mlp_weights(classifier, task_names, device):
    parts = [
        classifier[:, _CLASSIFIER_RANGE_BY_NAME[n][0]:_CLASSIFIER_RANGE_BY_NAME[n][1] + 1]
        for n in task_names
    ]
    global_w = torch.cat(parts, dim=1).T.contiguous().to(device)
    return global_w, torch.zeros(global_w.shape[0], device=device)


# ──────────────────────────────────────────────────────────────────────────────
# One-fold, one-task evaluation loop — Core
# ──────────────────────────────────────────────────────────────────────────────

def eval_task_tta_core(
    test_loader,
    task_id: int,
    adapter: MergeSlide_TTA_Adapter_Core,
    seq_dataset: Sequential_Generic_MIL_Dataset,
    fold_id: int,
    reset_per_slide: bool,
    device: torch.device,
) -> tuple:
    preds_all:           list[np.ndarray] = []
    targets_all:         list[np.ndarray] = []
    probs_all:           list[np.ndarray] = []
    convert_preds_all:   list[np.ndarray] = []
    convert_targets_all: list[np.ndarray] = []
    tta_stats_list:      list[dict]       = []
    times: list[float] = []

    task_name = seq_dataset.task_names[task_id]

    for sample_idx, (features, coords, label) in enumerate(
        tqdm(test_loader, desc=f"  Task {task_id} {task_name}", leave=False)
    ):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        features  = features.to(device)
        coords    = coords.long().to(device)
        label_int = int(label)

        if reset_per_slide:
            adapter.reset_adaptation_state()

        pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(features, coords)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

        # Naive/global evaluation — giống nhánh is_naive của test_tta_v3.py
        g_pred  = seq_dataset.task_to_global_class[pred_task].get(pred_local, -1)
        g_label = seq_dataset.task_to_global_class[task_id].get(label_int, -1)
        preds_all.append(np.array([g_pred]))
        targets_all.append(np.array([g_label]))
        # Core predicts over all 13 classes. Metrics must use the matching
        # global distribution, not the task-local compatibility return value.
        prob_global = np.asarray(debug["prob_global"])
        probs_all.append(prob_global.reshape(1, -1))

        strict_g_pred = seq_dataset.task_to_global_class[pred_task].get(pred_local, -1)
        convert_preds_all.append(np.array([g_pred]))
        convert_targets_all.append(np.array([g_label]))

        debug_log = {k: v for k, v in debug.items() if k != "prob_global"}
        tta_stats_list.append({
            "fold": fold_id, "task_id": task_id, "task_name": task_name,
            "sample_idx": sample_idx, "true_local": label_int,
            "pred_local": pred_local, "pred_task": pred_task,
            "g_label": g_label, "g_pred": g_pred,
            "strict_g_pred": strict_g_pred,
            "global_correct": int(g_pred == g_label),
            "strict_global_correct": int(strict_g_pred == g_label),
            **debug_log,
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

    parser = argparse.ArgumentParser(description="MergeSlide-TTA-Core evaluation (no TCP, no L_task)")
    parser.add_argument("--config",            type=str,
                        default="configs/default_tta_core_eval_num_workers0.yaml")
    parser.add_argument("--save_dir",          type=str, default="")  # giữ để tương thích CLI wrapper, không dùng
    parser.add_argument("--merge_model_path",  type=str, required=True)
    parser.add_argument("--swag_dir",          type=str, required=True)
    parser.add_argument(
        "--reset_per_slide",
        action="store_true",
        help="Ablation: reset student, teacher, optimizer trước mỗi slide.",
    )
    parser.add_argument("--result_csv",        type=str, default="")
    parser.add_argument("--tta_stats_csv",     type=str, default="")
    parser.add_argument("--efficiency_json",   type=str, default="")
    parser.add_argument("--fold_start",        type=int, default=0)
    parser.add_argument("--fold_end",          type=int, default=None)
    parser.add_argument(
        "--fisher_omega_dir", type=str, default="",
        help=(
            "Thư mục chứa fold_{fold_id}.pt sinh bởi tools/compute_fisher_importance.py. "
            "Chỉ bắt buộc nếu tta.use_fisher_reg=true trong config (Module D', AC-6)."
        ),
    )
    args = parser.parse_args()

    local_root = ensure_local_hot_storage()
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_root))
    args.swag_dir         = str(resolve_hot_path(args.swag_dir, local_root))
    if args.result_csv:
        args.result_csv = str(resolve_hot_path(args.result_csv, local_root))
    if args.tta_stats_csv:
        args.tta_stats_csv = str(resolve_hot_path(args.tta_stats_csv, local_root))
    if args.efficiency_json:
        args.efficiency_json = str(resolve_hot_path(args.efficiency_json, local_root))
    if args.fisher_omega_dir:
        args.fisher_omega_dir = str(resolve_hot_path(args.fisher_omega_dir, local_root))

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_folds       = cfg.training.num_folds
    fold_start      = args.fold_start
    fold_end        = num_folds if args.fold_end is None else args.fold_end
    if not (0 <= fold_start < fold_end <= num_folds):
        raise ValueError(
            f"Invalid fold range [{fold_start}, {fold_end}); "
            f"config provides {num_folds} folds."
        )
    reset_per_slide = args.reset_per_slide

    print(f"[INFO] MergeSlide-TTA-Core  (no TCP, no L_task)  "
          f"reset_per_slide={reset_per_slide}")
    print(f"[INFO] folds: {fold_start} -> {fold_end}")
    print(f"[INFO] swag_dir: {args.swag_dir}")

    # ── TTA Core config ──────────────────────────────────────────────────────
    raw_tta = OmegaConf.to_container(cfg.get("tta", OmegaConf.create({})), resolve=True)
    tta_cfg = TTAConfig_Core(**{k: v for k, v in raw_tta.items() if hasattr(TTAConfig_Core, k)})
    tta_cfg.k_patches_std = K_PATCHES
    print(f"[INFO] TTAConfig_Core: {tta_cfg}")

    # ── Dataset + prompt artifacts ───────────────────────────────────────────
    seq_dataset   = Sequential_Generic_MIL_Dataset(cfg)
    num_tasks     = cfg.training.num_tasks
    num_classes   = seq_dataset.num_classes
    total_classes = sum(num_classes)
    order         = getattr(cfg.dataset, "order", "forward")

    print("Building prompt classifier ...")
    classifier, _ = build_prompt_classifier(str(device))

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

    all_results:   list[dict] = []
    all_tta_stats: list[dict] = []
    efficiency_params: dict | None = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_wall_start = time.perf_counter()

    for fold_id in tqdm(range(fold_start, fold_end), desc="Folds"):
        fold_name   = f"fold_{fold_id}"
        merged_path = Path(args.merge_model_path) / fold_name / "merged_final.pth"
        swag_path   = Path(args.swag_dir) / f"{fold_name}.pt"

        if not merged_path.exists():
            print(f"[WARN] merged_final not found: {merged_path} -- skipping"); continue
        if not swag_path.exists():
            print(f"[WARN] SWAG not found: {swag_path} -- skipping"); continue

        backbone_sd     = torch.load(str(merged_path), map_location="cpu")
        mean_sd, var_sd = SWAGDiagonal.load(str(swag_path), device="cpu")

        fisher_omega = None
        if tta_cfg.use_fisher_reg:
            if not args.fisher_omega_dir:
                raise ValueError(
                    "tta.use_fisher_reg=true nhưng --fisher_omega_dir không được cung cấp. "
                    "Chạy tools/compute_fisher_importance.py cho từng fold trước."
                )
            fisher_path = Path(args.fisher_omega_dir) / f"{fold_name}.pt"
            if not fisher_path.exists():
                raise FileNotFoundError(
                    f"Fisher omega not found for {fold_name}: {fisher_path}. "
                    "Chạy: python tools/compute_fisher_importance.py --fold "
                    f"{fold_id} --out_path {fisher_path} ..."
                )
            fisher_omega = torch.load(str(fisher_path), map_location="cpu")
            print(f"[Fold {fold_id}] Loaded Fisher omega <- {fisher_path}")

        print(f"\n[Fold {fold_id}] Building TTA-Core adapter ...")
        adapter = MergeSlide_TTA_Adapter_Core(
            base_vision_encoder = base_titan.vision_encoder,
            backbone_sd         = backbone_sd,
            mean_sd             = mean_sd,
            var_sd              = var_sd,
            global_mlp_weight   = global_w,
            global_mlp_bias     = global_b,
            task_class_ranges   = seq_dataset.task_class_ranges,
            cfg                 = tta_cfg,
            device              = device,
            fisher_omega        = fisher_omega,
        )
        trainable = sum(p.numel() for p in adapter.student.parameters() if p.requires_grad)
        if efficiency_params is None:
            total_params = sum(p.numel() for p in adapter.student.parameters())
            num_ln_layers = sum(
                1
                for module in adapter.student.modules()
                if isinstance(module, nn.LayerNorm)
                and any(p.requires_grad for p in module.parameters(recurse=False))
            )
            efficiency_params = {
                "updated_object": "student LayerNorm parameters",
                "updated_params": int(trainable),
                "total_params": int(total_params),
                "update_ratio": float(trainable / max(total_params, 1)),
                "ln_layers": int(num_ln_layers),
            }
        print(f"[Fold {fold_id}] Trainable LN params: {trainable:,}")

        all_accs:   list[float] = []
        all_baccs:  list[float] = []
        all_aucs_fold = np.full(total_classes, np.nan, dtype=float)
        all_preds_g:   list[np.ndarray] = []
        all_targets_g: list[np.ndarray] = []
        acc_per_task:  dict[int, float] = {}
        fold_time: float = 0.0

        fold_diag = {
            "avg_ood_score":  [],
            "aug_rate":       [],
            "avg_loss_petal": [],
            "avg_loss_class": [],
            "avg_h_bar":      [],
            "avg_jsd_k":      [],
            "sample_active_rate": [],
        }

        for task_id in range(num_tasks):
            task_name = seq_dataset.task_names[task_id]
            n_cls     = num_classes[task_id]

            # Reset Module F sliding windows tại ranh giới task (KHÔNG reset model)
            adapter.reset_task_boundary()

            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            (metrics, preds_arr, targets_arr, probs_arr,
             conv_preds, conv_targets, elapsed, tta_stats) = eval_task_tta_core(
                test_loader     = test_loader,
                task_id         = task_id,
                adapter         = adapter,
                seq_dataset     = seq_dataset,
                fold_id         = fold_id,
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

            global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
            for g_idx in global_idxs:
                if probs_arr.shape[1] > g_idx:
                    try:
                        all_aucs_fold[g_idx] = roc_auc_score(
                            (targets_arr == g_idx).astype(int), probs_arr[:, g_idx]
                        )
                    except ValueError:
                        pass

            ood_scores      = [s["ood_score"]        for s in tta_stats]
            loss_petals     = [s["loss_petal"]        for s in tta_stats if not np.isnan(s["loss_petal"])]
            loss_classes    = [s["loss_class"]        for s in tta_stats if not np.isnan(s["loss_class"])]
            h_bars          = [s["h_bar"]             for s in tta_stats]
            jsd_ks          = [s["jsd_k"]              for s in tta_stats]
            sample_actives  = [s["sample_active"]      for s in tta_stats]
            aug_rate        = sum(s["use_aug"] for s in tta_stats) / max(1, len(tta_stats))
            active_rate     = sum(sample_actives) / max(1, len(sample_actives))

            fold_diag["avg_ood_score"].append(np.mean(ood_scores))
            fold_diag["aug_rate"].append(aug_rate)
            fold_diag["avg_loss_petal"].append(np.mean(loss_petals) if loss_petals else float("nan"))
            fold_diag["avg_loss_class"].append(np.mean(loss_classes) if loss_classes else float("nan"))
            fold_diag["avg_h_bar"].append(np.mean(h_bars))
            fold_diag["avg_jsd_k"].append(np.mean(jsd_ks))
            fold_diag["sample_active_rate"].append(active_rate)

            print(
                f"  [Fold {fold_id}][Task {task_id} {task_name}] "
                f"ACC={task_acc*100:.4f}%  BAcc={task_bacc*100:.4f}%  "
                f"OOD={np.mean(ood_scores):.3f}  "
                f"H_bar={np.mean(h_bars):.3f}  JSD_K={np.mean(jsd_ks):.4f}  "
                f"active={active_rate*100:.1f}%"
            )

            all_results.append({
                "fold": fold_id, "task_id": task_id, "task_name": task_name,
                "mode": "core_naive", "bacc": task_bacc, "acc": task_acc,
                "reset_per_slide": reset_per_slide,
                "n_samples": n_samples, "elapsed_s": elapsed,
                "avg_ood_score":     np.mean(ood_scores),
                "aug_rate":          aug_rate,
                "avg_loss_petal":    fold_diag["avg_loss_petal"][-1],
                "avg_loss_class":    fold_diag["avg_loss_class"][-1],
                "avg_h_bar":         np.mean(h_bars),
                "avg_jsd_k":         np.mean(jsd_ks),
                "sample_active_rate": active_rate,
                "use_module_f":      tta_cfg.use_module_f,
                "use_fisher_reg":    tta_cfg.use_fisher_reg,
                "use_fim_restore":   tta_cfg.use_fim_restore,
            })
            all_tta_stats.extend(tta_stats)

        all_preds_g_cat   = np.concatenate(all_preds_g)
        all_targets_g_cat = np.concatenate(all_targets_g)

        fold_macro_f1    = f1_score(all_targets_g_cat, all_preds_g_cat, average="macro",    zero_division=0)
        fold_weighted_f1 = f1_score(all_targets_g_cat, all_preds_g_cat, average="weighted", zero_division=0)
        fold_recall      = recall_score(all_targets_g_cat, all_preds_g_cat, average=None,   zero_division=0,
                                         labels=list(range(total_classes)))
        fold_precision   = precision_score(all_targets_g_cat, all_preds_g_cat, average=None,zero_division=0,
                                            labels=list(range(total_classes)))

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

        print(
            f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}%  "
            f"BAcc={np.mean(all_baccs)*100:.4f}%  "
            f"MacroF1={fold_macro_f1*100:.4f}%"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n===== TTA-Core (no TCP, no L_task) Results =====")
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

    print("\n===== TTA-Core Diagnostics (Module F) =====")
    diag_keys = [
        "avg_ood_score", "aug_rate", "avg_loss_petal", "avg_loss_class",
        "avg_h_bar", "avg_jsd_k", "sample_active_rate",
    ]
    for k in diag_keys:
        vals = [d[k] for d in overall_tta_diag]
        print(f"  {k:25s}: {np.mean(vals):.4f} (+/-{np.std(vals):.4f})")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if all_results:
        out_csv = args.result_csv or "logs/tta_core_results.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader(); writer.writerows(all_results)
        print(f"\n[INFO] Results saved -> {out_csv}")

    if all_tta_stats and args.tta_stats_csv:
        Path(args.tta_stats_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.tta_stats_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_tta_stats[0].keys()))
            writer.writeheader(); writer.writerows(all_tta_stats)
        print(f"[INFO] TTA stats saved -> {args.tta_stats_csv}")

    wall_elapsed_s = float(time.perf_counter() - eval_wall_start)
    total_slide_count = int(sum(row["n_samples"] for row in all_results))
    timed_elapsed_s = float(sum(row["elapsed_s"] for row in all_results))
    time_per_slide_s = timed_elapsed_s / max(total_slide_count, 1)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    n_active = sum(1 for s in all_tta_stats if s.get("sample_active"))
    efficiency = {
        "method": "MergeSlide-TTA-Core",
        "eval_setting": "class_il_naive",
        "param_scope": "ln_only",
        "use_module_f": tta_cfg.use_module_f,
        "use_fisher_reg": tta_cfg.use_fisher_reg,
        "use_fim_restore": tta_cfg.use_fim_restore,
        "patches_per_wsi": int(K_PATCHES),
        "augmented_views_K": int(tta_cfg.K),
        "num_slides": total_slide_count,
        "num_slides_adapted": n_active,
        "num_slides_gated_out": total_slide_count - n_active,
        "fold_start": int(fold_start),
        "fold_end": int(fold_end),
        "timed_elapsed_s": timed_elapsed_s,
        "wall_elapsed_s": wall_elapsed_s,
        "time_per_slide_s": time_per_slide_s,
        "throughput_slides_per_s": total_slide_count / max(timed_elapsed_s, 1e-12),
        "peak_vram_eval_mb": peak_vram_mb,
        "backprop": True,
        "source_free": True,
        "label_free": True,
        "reset_per_slide": bool(reset_per_slide),
        **(efficiency_params or {}),
    }
    print(f"[EFFICIENCY] {json.dumps(efficiency, sort_keys=True)}")
    if args.efficiency_json:
        efficiency_path = Path(args.efficiency_json)
        efficiency_path.parent.mkdir(parents=True, exist_ok=True)
        efficiency_path.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
        print(f"[INFO] Efficiency JSON saved -> {efficiency_path}")

    print("\nDone.")
