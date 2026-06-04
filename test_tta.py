"""
test_tta.py
===========
Test-Time Adaptation evaluation cho MergeSlide.

Modes:
    classil_tcp    : CLASS-IL + TCP Confidence Gate (default)
    classil_naive  : CLASS-IL + Global MLP (no TCP routing)
    taskil         : TASK-IL  (ground-truth task_id provided)

Usage::
    python test_tta.py \\
        --config   configs/default_tta_eval_num_workers0.yaml \\
        --save_dir checkpoints/finetuned \\
        --merge_model_path checkpoints/merged \\
        --swag_dir checkpoints/swag_diagonal \\
        --mode classil_tcp

Checkpoint layout kỳ vọng:
    Finetuned : {save_dir}/fold_{k}/task_{t}.pt
    Merged    : {merge_model_path}/fold_{k}/merged_final.pth
    SWAG      : {swag_dir}/fold_{k}.pt
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
from sklearn.metrics import balanced_accuracy_score
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
from mergeslide_tta.tta_engine import MergeSlide_TTA_Adapter, TTAConfig
from mergeslide_tta.utils import get_eval_metrics, seed_torch

PROJECT_ROOT  = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "checkpoints_ood", "logs", "sqlite"}


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers (mirros test_classIL_task_prompt.py)
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
        if not rp.is_symlink() and not rp.exists():
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
# Build per-task MLP weights (classifier-based, handles forward/reverse order)
# ──────────────────────────────────────────────────────────────────────────────

# Maps task name → column range in FORWARD classifier  (always fixed)
_CLASSIFIER_RANGE_BY_NAME: dict[str, list[int]] = {
    name: TASK_CLASS_RANGES_FORWARD[i]
    for i, name in enumerate(TASK_NAMES_FORWARD)
}


def build_per_task_mlp_weights(
    classifier: torch.Tensor,   # [768, 13] FORWARD order
    task_names: list[str],      # current order (forward or reverse)
    device: torch.device,
) -> dict:
    """
    {task_id: (weight [n_cls, 768], bias [n_cls])}
    task_id 0..T-1 theo current order (forward or reverse).
    Weight sliced từ FORWARD classifier theo task name.
    """
    per_task = {}
    for t, name in enumerate(task_names):
        start, end = _CLASSIFIER_RANGE_BY_NAME[name]
        w = classifier[:, start:end + 1].T.contiguous().to(device)   # [n_cls, 768]
        b = torch.zeros(end - start + 1, device=device)
        per_task[t] = (w, b)
    return per_task


def build_global_mlp_weights(
    classifier: torch.Tensor,   # [768, 13] FORWARD order
    task_names: list[str],      # current order
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build global [13, 768] weight matrix IN CURRENT ORDER.
    Columns: task_0_classes | task_1_classes | ... | task_{T-1}_classes
    Argmax của Z @ global_w.T → current-order global class index [0..12].
    """
    parts = []
    for name in task_names:
        start, end = _CLASSIFIER_RANGE_BY_NAME[name]
        parts.append(classifier[:, start:end + 1])     # [768, n_cls_t]
    global_w = torch.cat(parts, dim=1).T.contiguous().to(device)   # [13, 768]
    global_b = torch.zeros(13, device=device)
    return global_w, global_b


# ──────────────────────────────────────────────────────────────────────────────
# One-fold, one-task evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def eval_task_tta(
    test_loader,
    task_id: int,
    adapter: MergeSlide_TTA_Adapter,
    seq_dataset: Sequential_Generic_MIL_Dataset,
    fold_id: int,
    mode: str,                # "classil_tcp" | "classil_naive" | "taskil"
    reset_per_slide: bool,    # reset adapter before each slide (episodic)
    device: torch.device,
) -> tuple:
    """
    Adaptive evaluation loop cho một task + fold.

    Returns (metrics_dict, preds_arr, targets_arr, probs_arr,
             convert_preds_arr, convert_targets_arr, elapsed,
             tta_stats_list)
    """
    preds_all:           list[np.ndarray] = []
    targets_all:         list[np.ndarray] = []
    probs_all:           list[np.ndarray] = []
    convert_preds_all:   list[np.ndarray] = []
    convert_targets_all: list[np.ndarray] = []
    tta_stats_list:      list[dict]       = []
    times: list[float] = []

    n_classes = seq_dataset.num_classes[task_id]
    task_name = seq_dataset.task_names[task_id]

    for sample_idx, (features, coords, label) in enumerate(
        tqdm(test_loader, desc=f"  Task {task_id} {task_name}", leave=False)
    ):
        features = features.to(device)
        coords   = coords.long().to(device)

        if reset_per_slide:
            adapter.reset_to_source()

        t0 = time.time()

        # Mode dispatch
        if mode == "taskil":
            pred_local, pred_task, debug = adapter.adapt_and_predict(
                features, coords,
                task_id=task_id,
                use_tcp_gate=False,
            )
        elif mode == "classil_naive":
            pred_local, pred_task, debug = adapter.adapt_and_predict(
                features, coords,
                task_id=None,
                use_tcp_gate=False,
            )
        else:  # classil_tcp (default)
            pred_local, pred_task, debug = adapter.adapt_and_predict(
                features, coords,
                task_id=None,
                use_tcp_gate=True,
            )

        times.append(time.time() - t0)

        label_int = int(label)

        # ── Probability vector (local-class space) ──────────────────────────
        # Construct simple one-hot prob for metrics; full soft prob unavailable
        # without another forward pass. Use pred_local as hard prediction.
        prob_vec = np.zeros((1, n_classes), dtype=np.float32)
        if 0 <= pred_local < n_classes:
            prob_vec[0, pred_local] = 1.0

        preds_all.append(np.array([pred_local]))
        targets_all.append(np.array([label_int]))
        probs_all.append(prob_vec)

        # ── Global class conversion ──────────────────────────────────────────
        g_label = seq_dataset.task_to_global_class[task_id].get(label_int, -1)
        g_pred  = seq_dataset.task_to_global_class[pred_task].get(pred_local, -1)
        convert_preds_all.append(np.array([g_pred]))
        convert_targets_all.append(np.array([g_label]))

        # ── TTA diagnostics ──────────────────────────────────────────────────
        tta_stats_list.append({
            "fold":        fold_id,
            "task_id":     task_id,
            "task_name":   task_name,
            "sample_idx":  sample_idx,
            "true_local":  label_int,
            "pred_local":  pred_local,
            "pred_task":   pred_task,
            "g_label":     g_label,
            "g_pred":      g_pred,
            **debug,
        })

    # ── Pack results ─────────────────────────────────────────────────────────
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
    convert_preds_arr   = np.concatenate(convert_preds_all)
    convert_targets_arr = np.concatenate(convert_targets_all)

    return (
        metrics, preds_arr, targets_arr, probs_arr,
        convert_preds_arr, convert_targets_arr,
        sum(times), tta_stats_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="MergeSlide TTA evaluation")
    parser.add_argument("--config",           type=str,
                        default="configs/default_tta_eval_num_workers0.yaml")
    parser.add_argument("--save_dir",         type=str, required=True,
                        help="Root dir chứa finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, required=True,
                        help="Root dir chứa merged checkpoints")
    parser.add_argument("--swag_dir",         type=str, required=True,
                        help="Dir chứa SWAG stats: {swag_dir}/fold_{k}.pt")
    parser.add_argument("--mode",             type=str, default="classil_tcp",
                        choices=["classil_tcp", "classil_naive", "taskil"])
    parser.add_argument("--reset_per_task",   action="store_true",
                        help="Reset adapter trước mỗi task (default: True, flag để disable)")
    parser.add_argument("--no_reset_per_task", action="store_true",
                        help="Không reset adapter giữa các tasks")
    parser.add_argument("--episodic",         action="store_true",
                        help="Reset adapter trước mỗi slide (episodic mode)")
    parser.add_argument("--result_csv",       type=str, default="",
                        help="Path lưu kết quả metrics CSV")
    parser.add_argument("--tta_stats_csv",    type=str, default="",
                        help="Path lưu per-slide TTA stats CSV")
    parser.add_argument("--fold_start",       type=int, default=0)
    parser.add_argument("--fold_end",         type=int, default=None)
    args = parser.parse_args()

    local_root = ensure_local_hot_storage()
    args.save_dir          = str(resolve_hot_path(args.save_dir,          local_root))
    args.merge_model_path  = str(resolve_hot_path(args.merge_model_path,  local_root))
    args.swag_dir          = str(resolve_hot_path(args.swag_dir,          local_root))

    # ── Config ───────────────────────────────────────────────────────────────
    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    # Resolve per-fold range
    num_folds  = cfg.training.num_folds
    fold_start = args.fold_start
    fold_end   = args.fold_end or num_folds

    # Reset strategy
    reset_per_task  = not args.no_reset_per_task   # default True
    reset_per_slide = args.episodic                 # default False

    print(f"[INFO] mode={args.mode}  reset_per_task={reset_per_task}  "
          f"episodic={reset_per_slide}")
    print(f"[INFO] folds: {fold_start} → {fold_end}")
    print(f"[INFO] merge_model_path: {args.merge_model_path}")
    print(f"[INFO] swag_dir:         {args.swag_dir}")

    # ── TTA config từ cfg.tta (nếu có) ──────────────────────────────────────
    raw_tta = OmegaConf.to_container(cfg.get("tta", OmegaConf.create({})), resolve=True)
    tta_cfg = TTAConfig(**{k: v for k, v in raw_tta.items() if hasattr(TTAConfig, k)})
    tta_cfg.k_patches_std = K_PATCHES
    print(f"[INFO] TTAConfig: {tta_cfg}")

    # ── Dataset + prompt artifacts ───────────────────────────────────────────
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    num_tasks   = cfg.training.num_tasks
    order       = getattr(cfg.dataset, "order", "forward")

    print("Building prompt classifier (TITAN text encoder) ...")
    classifier, _ = build_prompt_classifier(str(device))   # [768, 13] FORWARD

    task_prompts: torch.Tensor = torch.load(
        PROJECT_ROOT / "task_prompts.pt"
    ).to(device)
    if order == "reverse":
        task_prompts = task_prompts.flip(0)

    # Per-task and global MLP weights
    per_task_mlp = build_per_task_mlp_weights(classifier, seq_dataset.task_names, device)
    global_w, global_b = build_global_mlp_weights(classifier, seq_dataset.task_names, device)

    # ── Load TITAN base model (shared across folds; only VE weights change) ──
    print("Loading TITAN base model ...")
    base_titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_titan = base_titan.to(device)
    base_titan.eval()

    # ── Result accumulators ──────────────────────────────────────────────────
    all_results:    list[dict] = []
    all_tta_stats:  list[dict] = []

    fold_baccs = []   # for final summary

    for fold_id in range(fold_start, fold_end):
        fold_name = f"fold_{fold_id}"
        print(f"\n{'='*60}")
        print(f"Fold {fold_id}")
        print(f"{'='*60}")

        # ── Load merged backbone ─────────────────────────────────────────────
        merged_path = Path(args.merge_model_path) / fold_name / "merged_final.pth"
        if not merged_path.exists():
            print(f"[WARN] merged_final not found: {merged_path} — skipping")
            continue
        backbone_sd = torch.load(str(merged_path), map_location="cpu")

        # ── Load SWAG posterior ──────────────────────────────────────────────
        swag_path = Path(args.swag_dir) / f"{fold_name}.pt"
        if not swag_path.exists():
            print(f"[WARN] SWAG stats not found: {swag_path} — skipping")
            continue
        mean_sd, var_sd = SWAGDiagonal.load(str(swag_path), device="cpu")

        # ── Build adapter ────────────────────────────────────────────────────
        print(f"[Fold {fold_id}] Building TTA adapter ...")
        adapter = MergeSlide_TTA_Adapter(
            base_vision_encoder = base_titan.vision_encoder,
            backbone_sd         = backbone_sd,
            mean_sd             = mean_sd,
            var_sd              = var_sd,
            per_task_mlp_weights= per_task_mlp,
            global_mlp_weight   = global_w,
            global_mlp_bias     = global_b,
            task_prompts        = task_prompts,
            task_class_ranges   = seq_dataset.task_class_ranges,
            cfg                 = tta_cfg,
            device              = device,
        )

        print(adapter.ln_param_names[:5])  # phải là tên chứa "norm" hoặc "ln"
        trainable = sum(p.numel() for p in adapter.student.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}")  # ~vài nghìn, không phải triệu

        fold_task_baccs: list[float] = []

        for task_id in range(num_tasks):
            task_name = seq_dataset.task_names[task_id]

            # Reset adapter trước mỗi task (default)
            if reset_per_task and task_id > 0:
                adapter.reset_to_source()

            # Data loader
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            t_start = time.time()
            (metrics, preds_arr, targets_arr, probs_arr,
             cpreds, ctargets, elapsed, tta_stats) = eval_task_tta(
                test_loader    = test_loader,
                task_id        = task_id,
                adapter        = adapter,
                seq_dataset    = seq_dataset,
                fold_id        = fold_id,
                mode           = args.mode,
                reset_per_slide= reset_per_slide,
                device         = device,
            )

            bacc = metrics.get("/bacc", 0.0)
            acc  = metrics.get("/acc",  0.0)
            fold_task_baccs.append(bacc)

            # TTA stats summary
            ood_scores = [s["ood_score"] for s in tta_stats]
            tcp_confs  = [s["tcp_conf"]  for s in tta_stats]
            loss_petals= [s["loss_petal"]for s in tta_stats]
            aug_rate   = sum(s["use_aug"] for s in tta_stats) / max(1, len(tta_stats))

            print(
                f"  [Task {task_id} {task_name:6s}] "
                f"bACC={bacc*100:.2f}%  ACC={acc*100:.2f}%  "
                f"OOD={np.mean(ood_scores):.3f}  "
                f"TCP={np.mean(tcp_confs):.3f}  "
                f"aug_rate={aug_rate*100:.1f}%  "
                f"L_petal={np.mean(loss_petals):.4f}  "
                f"t={elapsed:.1f}s"
            )

            row = {
                "fold":         fold_id,
                "task_id":      task_id,
                "task_name":    task_name,
                "mode":         args.mode,
                "bacc":         bacc,
                "acc":          acc,
                "n_samples":    len(targets_arr),
                "elapsed_s":    elapsed,
                "avg_ood_score":np.mean(ood_scores),
                "avg_tcp_conf": np.mean(tcp_confs),
                "aug_rate":     aug_rate,
                "avg_loss_petal":np.mean(loss_petals),
                "avg_loss_class":np.mean([s["loss_class"] for s in tta_stats]),
            }
            all_results.append(row)
            all_tta_stats.extend(tta_stats)

        fold_mean_bacc = np.mean(fold_task_baccs)
        fold_baccs.append(fold_mean_bacc)
        print(f"\n  [Fold {fold_id}] Mean bACC across {num_tasks} tasks: "
              f"{fold_mean_bacc*100:.2f}%")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Overall Mean bACC: {np.mean(fold_baccs)*100:.2f}%  "
          f"(±{np.std(fold_baccs)*100:.2f}%)")
    print(f"{'='*60}")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    if all_results:
        result_csv_path = args.result_csv or f"logs/tta_results_{args.mode}.csv"
        Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[INFO] Results saved → {result_csv_path}")

    if all_tta_stats and args.tta_stats_csv:
        Path(args.tta_stats_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.tta_stats_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_tta_stats[0].keys()))
            writer.writeheader()
            writer.writerows(all_tta_stats)
        print(f"[INFO] TTA stats saved → {args.tta_stats_csv}")

    print("\nDone.")
