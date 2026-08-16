#!/usr/bin/env python3
"""Class-IL prefix-merge evaluation with MergeSlide-TTA.

This file mirrors ``test_classIL_task_prompt_other_metrics.py`` but replaces
plain inference with online TTA at every prefix model:

  seq_task=1 -> load finetuned task_0.pt
  seq_task=2 -> load merged_task_1.pth
  ...
  seq_task=T -> load merged_final.pth

For each prefix model, this script evaluates all seen tasks with normal
Class-IL prediction plus TTA.  Task boundaries are used only to reproduce the
baseline metric protocol and to choose which test loaders are part of the
current prefix; the model still predicts in Class-IL mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from cast_slide.constants import K_PATCHES
from cast_slide.datasets import Sequential_Generic_MIL_Dataset
from cast_slide.metrics import backward_transfer, forgetting, pad_numpy_arrays
from cast_slide.tta_adapter import CASTSlide, load_task_weights
from cast_slide.utils import get_eval_metrics, seed_torch

from test_classIL_tta import (
    PROJECT_ROOT,
    build_class_embeddings,
    ensure_local_hot_storage,
    load_task_prompts_for_tasks,
    resolve_hot_path,
)

_EXPECTED_TASKS = {
    "forward": [
        "BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC",
    ],
    "reverse": [
        "CESC", "TGCT", "ESCA", "NSCLC", "RCC", "BRCA",
    ],
}
_EXPECTED_NUM_CLASSES = {
    "BRCA": 2,
    "RCC": 3,
    "NSCLC": 2,
    "ESCA": 2,
    "TGCT": 2,
    "CESC": 2,
}


def _sample_patches(
    features: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randperm(features.shape[0], device=features.device)[:K_PATCHES]
    return features[idx], coords[idx]


def _prefix_column_to_global(
    task_to_global_class: dict,
    num_seen_tasks: int,
) -> np.ndarray:
    return np.array(
        [
            task_to_global_class[t][local]
            for t in range(num_seen_tasks)
            for local in sorted(task_to_global_class[t].keys())
        ],
        dtype=np.int64,
    )


def _prefix_class_columns(num_classes: list[int], num_seen_tasks: int) -> slice:
    return slice(0, int(sum(num_classes[:num_seen_tasks])))


def _load_prefix_backbone_state(
    save_dir: Path,
    merge_model_path: Path,
    fold: str,
    seq_task: int,
    num_tasks: int,
) -> tuple[dict, str]:
    """Load the exact prefix checkpoint used by the baseline metric script."""
    if seq_task == 1:
        ckpt_path = save_dir / fold / "task_0.pt"
        state = torch.load(str(ckpt_path), map_location="cpu")
        backbone_state = {
            key.split("backbone.")[-1]: state[key]
            for key in list(state.keys())[:-2]
        }
        return backbone_state, str(ckpt_path)

    if seq_task < num_tasks:
        ckpt_path = merge_model_path / fold / f"merged_task_{seq_task - 1}.pth"
    else:
        ckpt_path = merge_model_path / fold / "merged_final.pth"
    return torch.load(str(ckpt_path), map_location="cpu"), str(ckpt_path)


def _validate_six_task_protocol(
    cfg,
    seq_dataset,
    save_dir: Path,
    merge_model_path: Path,
    fold_start: int,
    fold_end: int,
) -> None:
    order = str(cfg.dataset.order).lower()
    if order not in _EXPECTED_TASKS:
        raise ValueError(
            f"Unsupported dataset.order={order!r}; expected forward or reverse"
        )

    expected_tasks = _EXPECTED_TASKS[order]
    task_names = list(seq_dataset.task_names)
    num_classes = list(seq_dataset.num_classes)
    num_tasks = int(cfg.training.num_tasks)
    if num_tasks != 6 or task_names != expected_tasks:
        raise ValueError(
            "The prefix TTA metrics protocol requires the configured 7-task "
            f"{order} stream {expected_tasks}; got num_tasks={num_tasks}, "
            f"task_names={task_names}"
        )

    expected_classes = [_EXPECTED_NUM_CLASSES[name] for name in expected_tasks]
    if num_classes != expected_classes:
        raise ValueError(
            f"Class-count mismatch for {order}: expected {expected_classes}, "
            f"got {num_classes}"
        )
    if len(seq_dataset.task_to_global_class) != num_tasks:
        raise ValueError(
            "task_to_global_class must contain one mapping per configured task"
        )

    missing = []
    for fold_id in range(fold_start, fold_end):
        fold = f"fold_{fold_id}"
        for task_id in range(num_tasks):
            path = save_dir / fold / f"task_{task_id}.pt"
            if not path.is_file():
                missing.append(str(path))
        for merged_task_id in range(1, num_tasks - 1):
            path = merge_model_path / fold / f"merged_task_{merged_task_id}.pth"
            if not path.is_file():
                missing.append(str(path))
        final_path = merge_model_path / fold / "merged_final.pth"
        if not final_path.is_file():
            missing.append(str(final_path))

    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"Missing required 7-task prefix checkpoints:\n  {preview}{suffix}"
        )


def _run_task_with_tta(
    test_loader,
    task_id: int,
    num_seen_tasks: int,
    tta_model: CASTSlide,
    task_to_global_class: dict,
    device: torch.device,
    mode: str,
    verbose_loss: bool = False,
) -> dict:
    preds_all = []
    probs_all = []
    targets_all = []
    global_preds_all = []
    global_targets_all = []
    loss_logs = []
    routing_correct = 0
    elapsed_s = 0.0
    adapted = 0
    skipped = 0

    if mode == "naive":
        column_to_global = _prefix_column_to_global(
            task_to_global_class, num_seen_tasks
        )
        total_classes = len(column_to_global)
    else:
        column_to_global = None
        total_classes = None

    for features, coords, label in tqdm(test_loader, leave=False):
        features = features.to(device, non_blocking=True)
        coords = coords.long().to(device, non_blocking=True)
        features, coords = _sample_patches(features, coords)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred_class, probs, pred_task, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_s += time.perf_counter() - start

        adapted += int(adapt_log.get("slide/adapted", False))
        skipped += int(not adapt_log.get("slide/adapted", False))
        if verbose_loss:
            loss_logs.append(adapt_log)

        probs_np = probs.numpy()
        if mode == "tcp":
            routing_correct += int(pred_task == task_id)

            # Per-task metrics use local labels.
            preds_all.append(np.array([pred_class]))
            probs_all.append(probs_np)
            targets_all.append(label.numpy())

            true_global = task_to_global_class[task_id].get(int(label), -1)
            pred_global = task_to_global_class[pred_task].get(pred_class, -1)
            global_targets_all.append(np.array([true_global]))
            global_preds_all.append(np.array([pred_global]))
        else:
            pred_global = int(column_to_global[pred_class])
            true_global = task_to_global_class[task_id][int(label)]

            probs_out = np.zeros((1, total_classes), dtype=np.float32)
            for col_idx, global_idx in enumerate(column_to_global):
                probs_out[0, global_idx] = probs_np[0, col_idx]

            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))
            global_targets_all.append(np.array([true_global]))
            global_preds_all.append(np.array([pred_global]))

    preds_arr = np.concatenate(preds_all)
    targets_arr = np.concatenate(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    metrics = get_eval_metrics(
        targets_arr,
        preds_arr,
        probs_arr,
        roc_kwargs={"multi_class": "ovo", "average": "macro"},
        prefix="",
    )
    bacc = balanced_accuracy_score(targets_arr, preds_arr)

    if verbose_loss and loss_logs:
        adapted_logs = [row for row in loss_logs if row.get("slide/adapted")]
        if adapted_logs:
            mean_loss = np.mean(
                [row.get("loss/total_with_reg", 0.0) for row in adapted_logs]
            )
            print(
                f"    [TTA] task={task_id} adapted={len(adapted_logs)}/"
                f"{len(loss_logs)} mean_loss={mean_loss:.4f}"
            )

    return {
        "acc": float(metrics["/acc"]),
        "bacc": float(bacc),
        "n_samples": int(len(targets_arr)),
        "elapsed_s": float(elapsed_s),
        "routing_acc": (
            float(routing_correct / max(1, len(targets_arr)))
            if mode == "tcp" else 0.0
        ),
        "adapted": int(adapted),
        "skipped": int(skipped),
        "global_preds": np.concatenate(global_preds_all),
        "global_targets": np.concatenate(global_targets_all),
    }


def _weighted_prefix_acc(row: np.ndarray, counts: np.ndarray) -> float:
    valid = (~np.isnan(row)) & (~np.isnan(counts)) & (counts > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(row[valid] * counts[valid]) / np.sum(counts[valid]))


def _write_matrix(path: Path, matrix: np.ndarray, task_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state_after_task", *task_names])
        for row_idx, row in enumerate(matrix):
            writer.writerow(
                [
                    f"after_{row_idx}_{task_names[row_idx]}",
                    *["" if np.isnan(val) else f"{val:.8f}" for val in row],
                ]
            )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-IL prefix-merge TTA mACC/FGT/BWT evaluation"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs/prefix_tta_metrics")
    parser.add_argument("--mode", type=str, default="tcp", choices=["tcp", "naive"])
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=-1)

    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--K_sub", type=int, default=300)
    parser.add_argument("--top_ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l2_anchor_beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument(
        "--tta_param_scope",
        type=str,
        default="ln_only",
        choices=["ln_only", "full"],
    )
    parser.add_argument("--entropy_threshold", type=float, default=0.4)
    parser.add_argument("--use_task_diversity", action="store_true")
    parser.add_argument("--no_task_agreement", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--select_mode",
        type=str,
        default="intersection",
        choices=["union", "intersection", "class_only", "task_only"],
    )
    parser.add_argument("--no_teacher", action="store_true")
    parser.add_argument(
        "--tcp_inference_model",
        choices=["teacher", "student"],
        default="teacher",
    )
    parser.add_argument(
        "--naive_inference_model",
        choices=["student", "teacher"],
        default="student",
    )
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--no_adapt_prompts", action="store_true")
    parser.add_argument("--ema_alpha_prompt", type=float, default=0.999)
    parser.add_argument("--delta_margin", type=float, default=0.10)
    parser.add_argument("--tp_anchor_beta", type=float, default=0.3)
    parser.add_argument("--gamma_margin", type=float, default=0.0)
    parser.add_argument("--tau_task", type=float, default=0.70)
    parser.add_argument(
        "--naive_use_task_entropy",
        dest="naive_use_task_entropy",
        action="store_true",
    )
    parser.add_argument(
        "--no_naive_task_entropy",
        dest="naive_use_task_entropy",
        action="store_false",
    )
    parser.set_defaults(naive_use_task_entropy=True)
    parser.add_argument("--use_dapc", action="store_true")
    parser.add_argument("--dapc_loss_weight", type=float, default=1.0)
    parser.add_argument("--class_loss_weight", type=float, default=1.0)
    parser.add_argument("--entropy_loss_weight", type=float, default=1.0)
    parser.add_argument("--dapc_tau_anchor", type=float, default=0.92)
    parser.add_argument("--dapc_beta", type=float, default=1.2)
    parser.add_argument(
        "--no_reset_prompt_per_task",
        action="store_true",
        help="Do not reset task prompts between eval tasks inside a prefix.",
    )
    parser.add_argument("--verbose_loss", action="store_true")
    return parser


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = build_parser().parse_args()
    if (
        args.mode == "naive"
        and args.naive_inference_model == "student"
        and not args.use_dapc
    ):
        args.no_teacher = True

    local_hot_root = ensure_local_hot_storage()
    args.save_dir = str(resolve_hot_path(args.save_dir, local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    output_dir = resolve_hot_path(args.output_dir, local_hot_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    num_tasks = int(cfg.training.num_tasks)
    fold_end = int(cfg.training.num_folds) if args.fold_end < 0 else args.fold_end
    if not (0 <= args.fold_start < fold_end <= int(cfg.training.num_folds)):
        raise ValueError(
            f"Invalid fold range [{args.fold_start}, {fold_end}) for "
            f"{int(cfg.training.num_folds)} folds"
        )

    _validate_six_task_protocol(
        cfg=cfg,
        seq_dataset=seq_dataset,
        save_dir=Path(args.save_dir),
        merge_model_path=Path(args.merge_model_path),
        fold_start=args.fold_start,
        fold_end=fold_end,
    )

    task_prompts_full = load_task_prompts_for_tasks(
        PROJECT_ROOT / "task_prompts.pt",
        seq_dataset.task_names,
        device,
    )

    # Naive predicts globally; TCP needs the same classifier for its
    # low-confidence fallback.
    class_embeddings_full = build_class_embeddings(
        device, seq_dataset.task_names
    )

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    fold_summary_rows = []
    event_rows = []
    wall_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for fold_id in tqdm(range(args.fold_start, fold_end), desc="Folds"):
        fold = f"fold_{fold_id}"
        task_model_paths_full = [
            str(Path(args.save_dir) / fold / f"task_{task_id}.pt")
            for task_id in range(num_tasks)
        ]

        acc_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        bacc_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        count_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        routing_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        fold_events = []

        acc_per_task_all_seqs = []
        bacc_per_task_all_seqs = []
        acc_all_seqs = []

        for seq_task in tqdm(
            range(1, num_tasks + 1), desc=f"Fold {fold_id} prefixes", leave=False
        ):
            seed_torch(device, cfg.training.seed)
            prefix_state, ckpt_path = _load_prefix_backbone_state(
                save_dir=Path(args.save_dir),
                merge_model_path=Path(args.merge_model_path),
                fold=fold,
                seq_task=seq_task,
                num_tasks=num_tasks,
            )
            print(f"\n[INFO] fold={fold_id} seq_task={seq_task} load={ckpt_path}")
            base_model.vision_encoder.load_state_dict(prefix_state, strict=True)

            task_model_paths = task_model_paths_full[:seq_task]
            task_weights = load_task_weights(task_model_paths, device)
            task_prompts = task_prompts_full[:seq_task].detach().clone()
            class_slice = _prefix_class_columns(
                seq_dataset.num_classes, seq_task
            )
            all_class_embeddings = class_embeddings_full[:, class_slice]

            # With only one seen task, task-prompt confidence margin/top2 is not
            # defined. Keep LN-TTA active, but disable prompt/margin add-ons for
            # this first prefix.
            adapt_task_prompts = (not args.no_adapt_prompts) and seq_task >= 2
            gamma_margin = args.gamma_margin if seq_task >= 2 else 0.0

            tta_model = CASTSlide(
                backbone=base_model.vision_encoder,
                task_prompts=task_prompts,
                task_weights=task_weights,
                num_classes=seq_dataset.num_classes[:seq_task],
                device=device,
                mode=args.mode,
                all_class_embeddings=all_class_embeddings,
                param_scope=args.tta_param_scope,
                M=args.M,
                K_sub=args.K_sub,
                top_ratio=args.top_ratio,
                alpha=args.alpha,
                l2_anchor_beta=args.l2_anchor_beta,
                lr=args.lr,
                n_steps=args.n_steps,
                episodic=False,
                entropy_threshold=args.entropy_threshold,
                use_task_diversity=args.use_task_diversity,
                use_task_agreement=(not args.no_task_agreement),
                gamma=args.gamma,
                select_mode=args.select_mode,
                use_teacher=(not args.no_teacher),
                tcp_inference_model=args.tcp_inference_model,
                naive_inference_model=args.naive_inference_model,
                ema_alpha=args.ema_alpha,
                adapt_task_prompts=adapt_task_prompts,
                ema_alpha_prompt=args.ema_alpha_prompt,
                delta_margin=args.delta_margin,
                tp_anchor_beta=args.tp_anchor_beta,
                gamma_margin=gamma_margin,
                tau_task=args.tau_task,
                naive_use_task_entropy=args.naive_use_task_entropy,
                use_dapc=args.use_dapc,
                dapc_loss_weight=args.dapc_loss_weight,
                class_loss_weight=args.class_loss_weight,
                entropy_loss_weight=args.entropy_loss_weight,
                dapc_tau_anchor=args.dapc_tau_anchor,
                dapc_beta=args.dapc_beta,
            )

            prefix_accs = []
            prefix_baccs = []
            prefix_counts = []

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

                if (
                    (not args.no_reset_prompt_per_task)
                    and adapt_task_prompts
                    and args.mode == "tcp"
                ):
                    tta_model.reset_task_prompts()

                result = _run_task_with_tta(
                    test_loader=test_loader,
                    task_id=task_id,
                    num_seen_tasks=seq_task,
                    tta_model=tta_model,
                    task_to_global_class=seq_dataset.task_to_global_class,
                    device=device,
                    mode=args.mode,
                    verbose_loss=args.verbose_loss,
                )

                row_idx = seq_task - 1
                acc_matrix[row_idx, task_id] = result["acc"]
                bacc_matrix[row_idx, task_id] = result["bacc"]
                count_matrix[row_idx, task_id] = result["n_samples"]
                routing_matrix[row_idx, task_id] = result["routing_acc"]

                prefix_accs.append(result["acc"])
                prefix_baccs.append(result["bacc"])
                prefix_counts.append(result["n_samples"])

                event = {
                    "fold": fold_id,
                    "seq_task": seq_task,
                    "state_after_task": seq_task - 1,
                    "eval_task": task_id,
                    "task_name": seq_dataset.task_names[task_id],
                    "checkpoint": ckpt_path,
                    "acc": result["acc"],
                    "bacc": result["bacc"],
                    "routing_acc": result["routing_acc"],
                    "adapted": result["adapted"],
                    "skipped": result["skipped"],
                    "n_samples": result["n_samples"],
                    "elapsed_s": result["elapsed_s"],
                }
                fold_events.append(event)
                print(
                    f"  [Fold {fold_id}] after task {seq_task - 1} "
                    f"eval task {task_id} ({seq_dataset.task_names[task_id]}) "
                    f"ACC={result['acc']*100:.4f}% "
                    f"BAcc={result['bacc']*100:.4f}% "
                    f"routing_acc={result['routing_acc']*100:.4f}%"
                )

            prefix_acc = _weighted_prefix_acc(
                acc_matrix[seq_task - 1], count_matrix[seq_task - 1]
            )
            acc_all_seqs.append(prefix_acc)
            acc_per_task_all_seqs.append(prefix_accs)
            bacc_per_task_all_seqs.append(prefix_baccs)

            del tta_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        macc = float(np.mean(acc_all_seqs))
        fgt = forgetting([row.copy() for row in acc_per_task_all_seqs])
        bwt = backward_transfer([row.copy() for row in acc_per_task_all_seqs])
        bacc_macc_task_mean = float(
            np.mean([np.mean(row) for row in bacc_per_task_all_seqs])
        )
        bacc_fgt = forgetting([row.copy() for row in bacc_per_task_all_seqs])
        bacc_bwt = backward_transfer([row.copy() for row in bacc_per_task_all_seqs])

        fold_summary_rows.append({
            "fold": fold_id,
            "mode": args.mode,
            "macc_acc": macc,
            "fgt_acc": fgt,
            "bwt_acc": bwt,
            "macc_bacc_task_mean": bacc_macc_task_mean,
            "fgt_bacc": bacc_fgt,
            "bwt_bacc": bacc_bwt,
        })
        event_rows.extend(fold_events)

        fold_dir = output_dir / f"fold_{fold_id}"
        _write_matrix(fold_dir / "acc_matrix.csv", acc_matrix, seq_dataset.task_names)
        _write_matrix(fold_dir / "bacc_matrix.csv", bacc_matrix, seq_dataset.task_names)
        _write_matrix(fold_dir / "count_matrix.csv", count_matrix, seq_dataset.task_names)
        _write_matrix(fold_dir / "routing_matrix.csv", routing_matrix, seq_dataset.task_names)
        _write_rows(fold_dir / "events.csv", fold_events)

        print(
            f"[Fold {fold_id}] mACC={macc*100:.4f}% "
            f"FGT={fgt*100:.4f}% BWT={bwt*100:.4f}%"
        )

    _write_rows(output_dir / "fold_summary.csv", fold_summary_rows)
    _write_rows(output_dir / "events_all_folds.csv", event_rows)

    summary = {}
    for key in (
        "macc_acc", "fgt_acc", "bwt_acc",
        "macc_bacc_task_mean", "fgt_bacc", "bwt_bacc",
    ):
        vals = np.array([row[key] for row in fold_summary_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))

    summary.update({
        "mode": args.mode,
        "fold_start": int(args.fold_start),
        "fold_end": int(fold_end),
        "num_tasks": int(num_tasks),
        "protocol": "baseline prefix checkpoints plus online TTA at each prefix",
        "macc_definition": "old baseline sample-weighted ACC over seen tasks per prefix",
        "wall_elapsed_s": float(time.perf_counter() - wall_start),
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else 0.0
        ),
    })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        f"===== Class-IL prefix-merge TTA metrics ({args.mode}) =====",
        f"mACC(ACC): {summary['macc_acc_mean']*100:.4f}% ({summary['macc_acc_std']*100:.4f}%)",
        f"FGT(ACC):  {summary['fgt_acc_mean']*100:.4f}% ({summary['fgt_acc_std']*100:.4f}%)",
        f"BWT(ACC):  {summary['bwt_acc_mean']*100:.4f}% ({summary['bwt_acc_std']*100:.4f}%)",
        f"mACC(bACC task-mean): {summary['macc_bacc_task_mean_mean']*100:.4f}% ({summary['macc_bacc_task_mean_std']*100:.4f}%)",
        f"FGT(bACC):            {summary['fgt_bacc_mean']*100:.4f}% ({summary['fgt_bacc_std']*100:.4f}%)",
        f"BWT(bACC):            {summary['bwt_bacc_mean']*100:.4f}% ({summary['bwt_bacc_std']*100:.4f}%)",
        f"[INFO] Saved outputs to: {output_dir}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (output_dir / "result.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
