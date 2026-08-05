#!/usr/bin/env python3
"""Collect routing-confidence plot inputs for one Class-IL TTA fold.

Run this through tools/run_classil_with_pt_features.py so WSI features are
loaded from PT files first when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mergeslide_tta.constants import K_PATCHES
from mergeslide_tta.datasets import Sequential_Generic_MIL_Dataset
from mergeslide_tta.prompts_zeroshot import (
    brca_prompts,
    cesc_prompts,
    esca_prompts,
    nsclc_prompts,
    rcc_prompts,
    tgct_prompts,
)
from mergeslide_tta.tta_adapter import MergeSlide_TTA, load_task_weights
from mergeslide_tta.utils import seed_torch


PROMPT_FN_BY_TASK = {
    "BRCA": brca_prompts,
    "RCC": rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA": esca_prompts,
    "TGCT": tgct_prompts,
    "CESC": cesc_prompts,
}


def _cast_like(value: str, default):
    if isinstance(default, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


def load_best_config(path: str | None, defaults: dict) -> dict:
    cfg = dict(defaults)
    if not path:
        return cfg
    best_path = Path(path)
    if not best_path.is_file():
        raise FileNotFoundError(f"best_config not found: {best_path}")
    raw = json.loads(best_path.read_text(encoding="utf-8"))
    for key, value in raw.items():
        if key in cfg:
            cfg[key] = _cast_like(value, cfg[key])
    return cfg


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)


def score_row(
    state: str,
    fold: int,
    task_id: int,
    task_name: str,
    slide_index: int,
    label: int,
    vector: np.ndarray,
    prompts: np.ndarray,
    xy: np.ndarray,
    scores: np.ndarray | None = None,
    adapted: bool | None = None,
    prompt_updated: bool | None = None,
) -> dict:
    if scores is None:
        scores = vector.astype(np.float32)[None, :] @ prompts.astype(np.float32).T
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    pred_task = int(np.argmax(scores))
    true_score = float(scores[task_id])
    wrong_scores = np.delete(scores, task_id)
    max_wrong = float(np.max(wrong_scores)) if wrong_scores.size else float("nan")
    top2 = np.sort(scores)[-2:]
    row = {
        "state": state,
        "fold": fold,
        "task_id": task_id,
        "task_name": task_name,
        "slide_index": slide_index,
        "label": label,
        "pred_task": pred_task,
        "true_score": true_score,
        "max_wrong_score": max_wrong,
        "true_vs_wrong_margin": true_score - max_wrong,
        "top1_top2_margin": float(top2[-1] - top2[-2]) if top2.size == 2 else 0.0,
        "x": float(xy[0]),
        "y": float(xy[1]),
    }
    if adapted is not None:
        row["adapted"] = bool(adapted)
    if prompt_updated is not None:
        row["prompt_updated"] = bool(prompt_updated)
    return row


def routing_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "routing_acc": 0.0, "mean_true_vs_wrong_margin": 0.0}
    correct = [int(r["pred_task"]) == int(r["task_id"]) for r in rows]
    margins = [float(r["true_vs_wrong_margin"]) for r in rows]
    by_task = {}
    for task_id in sorted({int(r["task_id"]) for r in rows}):
        task_rows = [r for r in rows if int(r["task_id"]) == task_id]
        task_correct = [int(r["pred_task"]) == task_id for r in task_rows]
        by_task[str(task_id)] = {
            "n": len(task_rows),
            "routing_acc": float(np.mean(task_correct)),
            "mean_true_vs_wrong_margin": float(np.mean([float(r["true_vs_wrong_margin"]) for r in task_rows])),
        }
    return {
        "n": len(rows),
        "routing_acc": float(np.mean(correct)),
        "mean_true_vs_wrong_margin": float(np.mean(margins)),
        "mean_top1_top2_margin": float(np.mean([float(r["top1_top2_margin"]) for r in rows])),
        "per_task": by_task,
    }


def forward_vector(tta_model: MergeSlide_TTA, features: torch.Tensor, coords: torch.Tensor) -> np.ndarray:
    model = tta_model.teacher if tta_model.use_teacher else tta_model.backbone
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        z = model(features, coords, tta_model.ps)
    if was_training and model is tta_model.backbone:
        model.train()
    return z.detach().float().cpu().numpy().reshape(-1)


def build_class_embeddings(model, device: torch.device, task_names: list[str]) -> torch.Tensor:
    missing = [name for name in task_names if name not in PROMPT_FN_BY_TASK]
    if missing:
        raise ValueError(f"Missing zero-shot prompt functions for tasks: {missing}")
    _, templates = brca_prompts()
    prompts = []
    for task_name in task_names:
        class_prompts, _ = PROMPT_FN_BY_TASK[task_name]()
        prompts.extend(class_prompts)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = model.zero_shot_classifier(prompts, templates, device=str(device))
    return classifier.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_ood_eval_num_workers0.yaml")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--merge_model_path", required=True)
    parser.add_argument("--best_config", default="")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--output_dir", default="logs/Ablations/prompt_embedding_space")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max_slides_per_task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose_loss", action="store_true")
    parser.add_argument("--use_dapc", action="store_true")
    parser.add_argument("--no_teacher", action="store_true")
    parser.add_argument("--no_adapt_prompts", action="store_true")
    parser.add_argument("--no_reset_prompt_per_task", action="store_true")

    defaults = {
        "M": 8,
        "K_sub": 300,
        "top_ratio": 0.5,
        "alpha": 0.5,
        "beta": 1.0,
        "lr": 1e-4,
        "n_steps": 5,
        "tta_param_scope": "ln_only",
        "entropy_threshold": 0.4,
        "gamma": 0.5,
        "select_mode": "intersection",
        "ema_alpha": 0.999,
        "ema_alpha_prompt": 0.999,
        "delta_margin": 0.10,
        "tp_anchor_beta": 0.3,
        "gamma_margin": 0.0,
        "tau_task": 0.70,
        "dapc_loss_weight": 1.0,
        "entropy_loss_weight": 1.0,
        "dapc_tau_anchor": 0.92,
        "dapc_beta": 1.2,
    }
    for key, default in defaults.items():
        value_type = type(default) if not isinstance(default, bool) else int
        parser.add_argument(f"--{key}", type=value_type, default=default)

    args = parser.parse_args()
    tta_cfg = load_best_config(args.best_config, {k: getattr(args, k) for k in defaults})

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed + int(args.seed))

    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    order = getattr(cfg.dataset, "order", "forward")
    setting = "ood" if "ood" in Path(args.config).name.lower() else "ind"
    tag = args.tag or f"{setting}_{order}_fold{args.fold}_tsne"

    print(f"[INFO] collect routing inputs tag={tag}")
    print(f"[INFO] config={args.config}")
    print(f"[INFO] fold={args.fold} setting={setting} order={order}")
    print(f"[INFO] tta_cfg={json.dumps(tta_cfg, sort_keys=True)}")

    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    task_prompts = task_prompts[: int(cfg.training.num_tasks)]
    if order in ("reverse", "reverse_extended"):
        task_prompts = task_prompts.flip(0)

    print("[INFO] loading TITAN base model")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)
    print("[INFO] building global class embeddings for TCP fallback")
    all_class_embeddings = build_class_embeddings(
        base_model,
        device,
        list(seq_dataset.task_names),
    )

    merge_path = Path(args.merge_model_path) / f"fold_{args.fold}" / "merged_final.pth"
    print(f"[INFO] loading merged checkpoint: {merge_path}")
    base_model.vision_encoder.load_state_dict(torch.load(str(merge_path), map_location="cpu"))

    task_model_paths = [
        str(Path(args.save_dir) / f"fold_{args.fold}" / f"task_{t}.pt")
        for t in range(cfg.training.num_tasks)
    ]
    task_weights = load_task_weights(task_model_paths, device)

    tta_model = MergeSlide_TTA(
        backbone=base_model.vision_encoder,
        task_prompts=task_prompts,
        task_weights=task_weights,
        num_classes=seq_dataset.num_classes,
        device=device,
        mode="tcp",
        all_class_embeddings=all_class_embeddings,
        param_scope=str(tta_cfg["tta_param_scope"]),
        M=int(tta_cfg["M"]),
        K_sub=int(tta_cfg["K_sub"]),
        top_ratio=float(tta_cfg["top_ratio"]),
        alpha=float(tta_cfg["alpha"]),
        l2_anchor_beta=float(tta_cfg["beta"]),
        lr=float(tta_cfg["lr"]),
        n_steps=int(tta_cfg["n_steps"]),
        episodic=False,
        entropy_threshold=float(tta_cfg["entropy_threshold"]),
        use_task_diversity=False,
        use_task_agreement=True,
        gamma=float(tta_cfg["gamma"]),
        select_mode=str(tta_cfg["select_mode"]),
        ema_alpha=float(tta_cfg["ema_alpha"]),
        ema_alpha_prompt=float(tta_cfg["ema_alpha_prompt"]),
        delta_margin=float(tta_cfg["delta_margin"]),
        tp_anchor_beta=float(tta_cfg["tp_anchor_beta"]),
        gamma_margin=float(tta_cfg["gamma_margin"]),
        tau_task=float(tta_cfg["tau_task"]),
        use_dapc=bool(args.use_dapc),
        dapc_loss_weight=float(tta_cfg["dapc_loss_weight"]),
        entropy_loss_weight=float(tta_cfg["entropy_loss_weight"]),
        dapc_tau_anchor=float(tta_cfg["dapc_tau_anchor"]),
        dapc_beta=float(tta_cfg["dapc_beta"]),
        use_teacher=not args.no_teacher,
        tcp_inference_model="teacher",
        adapt_task_prompts=not args.no_adapt_prompts,
    )

    baseline_vectors: list[np.ndarray] = []
    tta_vectors: list[np.ndarray] = []
    labels: list[int] = []
    task_ids_for_rows: list[int] = []
    task_names_for_rows: list[str] = []
    slide_indices_for_rows: list[int] = []
    adapted_flags: list[bool] = []
    prompt_updated_flags: list[bool] = []
    baseline_probs: list[np.ndarray] = []
    tta_probs: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    tta_scores: list[np.ndarray] = []
    source_prompts_np = tta_model.task_prompts_source.detach().float().cpu().numpy()

    for task_id in range(cfg.training.num_tasks):
        if task_id > 0 and not args.no_reset_prompt_per_task:
            tta_model.reset_task_prompts()
        _, _, test_loader = seq_dataset.get_data_loaders(args.fold, task_id)
        for slide_index, (features, coords, label) in enumerate(tqdm(test_loader, desc=f"Task {task_id}", leave=False)):
            if args.max_slides_per_task > 0 and slide_index >= args.max_slides_per_task:
                break
            features = features.to(device)
            coords = coords.long().to(device)
            idx = torch.randperm(features.shape[0], device=device)[:K_PATCHES]
            features = features[idx]
            coords = coords[idx]

            baseline_vec = forward_vector(tta_model, features, coords)
            baseline_score = baseline_vec[None, :] @ source_prompts_np.T
            baseline_scores.append(baseline_score[0].astype(np.float32))
            baseline_probs.append(softmax_np(baseline_score)[0])

            _, _, _, adapt_log = tta_model.adapt_and_predict(features, coords)
            tta_vec = forward_vector(tta_model, features, coords)
            adapted_prompt_now = tta_model.task_prompts.detach().float().cpu().numpy()
            tta_score = tta_vec[None, :] @ adapted_prompt_now.T
            tta_scores.append(tta_score[0].astype(np.float32))
            tta_probs.append(softmax_np(tta_score)[0])

            baseline_vectors.append(baseline_vec)
            tta_vectors.append(tta_vec)
            labels.append(int(label.item() if torch.is_tensor(label) else label))
            task_ids_for_rows.append(task_id)
            task_names_for_rows.append(seq_dataset.task_names[task_id])
            slide_indices_for_rows.append(slide_index)
            adapted_flags.append(bool(adapt_log.get("slide/adapted", False)))
            prompt_updated_flags.append(bool(adapt_log.get("adapt/prompt_updated", False)))

    baseline_vectors_np = np.asarray(baseline_vectors, dtype=np.float32)
    tta_vectors_np = np.asarray(tta_vectors, dtype=np.float32)
    source_prompts_np = source_prompts_np.astype(np.float32)
    adapted_prompts_np = tta_model.task_prompts.detach().float().cpu().numpy().astype(np.float32)
    baseline_scores_np = np.asarray(baseline_scores, dtype=np.float32)
    tta_scores_np = np.asarray(tta_scores, dtype=np.float32)
    baseline_probs_np = np.asarray(baseline_probs, dtype=np.float32)
    tta_probs_np = np.asarray(tta_probs, dtype=np.float32)

    n = baseline_vectors_np.shape[0]
    paired_lengths = {
        len(tta_vectors_np),
        len(labels),
        len(task_ids_for_rows),
        len(task_names_for_rows),
        len(slide_indices_for_rows),
        len(adapted_flags),
        len(prompt_updated_flags),
        len(baseline_scores_np),
        len(tta_scores_np),
        len(baseline_probs_np),
        len(tta_probs_np),
    }
    if paired_lengths != {n}:
        raise RuntimeError(f"Unpaired routing evidence lengths: baseline={n}, others={sorted(paired_lengths)}")

    pca_input = np.concatenate(
        [baseline_vectors_np, tta_vectors_np, source_prompts_np, adapted_prompts_np],
        axis=0,
    )
    xy_all = PCA(n_components=2, random_state=0).fit_transform(pca_input)
    baseline_xy = xy_all[:n].astype(np.float32)
    tta_xy = xy_all[n:2 * n].astype(np.float32)
    source_prompts_xy = xy_all[2 * n:2 * n + source_prompts_np.shape[0]].astype(np.float32)
    adapted_prompts_xy = xy_all[2 * n + source_prompts_np.shape[0]:].astype(np.float32)

    baseline_rows = []
    tta_rows = []
    for i in range(n):
        baseline_rows.append(score_row(
            state="baseline",
            fold=args.fold,
            task_id=task_ids_for_rows[i],
            task_name=task_names_for_rows[i],
            slide_index=slide_indices_for_rows[i],
            label=labels[i],
            vector=baseline_vectors_np[i],
            prompts=source_prompts_np,
            xy=baseline_xy[i],
            scores=baseline_scores_np[i],
        ))
        tta_rows.append(score_row(
            state="tta",
            fold=args.fold,
            task_id=task_ids_for_rows[i],
            task_name=task_names_for_rows[i],
            slide_index=slide_indices_for_rows[i],
            label=labels[i],
            vector=tta_vectors_np[i],
            prompts=adapted_prompts_np,
            xy=tta_xy[i],
            scores=tta_scores_np[i],
            adapted=adapted_flags[i],
            prompt_updated=prompt_updated_flags[i],
        ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = output_dir / f"{tag}_baseline_points.csv"
    tta_csv = output_dir / f"{tag}_tta_points.csv"
    raw_npz = output_dir / f"{tag}_raw_embeddings.npz"
    summary_json = output_dir / f"{tag}_summary.json"

    with baseline_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_rows)
    with tta_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tta_rows)
    np.savez_compressed(
        raw_npz,
        baseline_vectors=baseline_vectors_np,
        tta_vectors=tta_vectors_np,
        source_prompts=source_prompts_np,
        adapted_prompts=adapted_prompts_np,
        baseline_xy=baseline_xy,
        tta_xy=tta_xy,
        source_prompts_xy=source_prompts_xy,
        adapted_prompts_xy=adapted_prompts_xy,
        baseline_probs=baseline_probs_np,
        tta_probs=tta_probs_np,
        baseline_scores=baseline_scores_np,
        tta_scores=tta_scores_np,
    )

    summary = {
        "setting": setting,
        "order": order,
        "fold": args.fold,
        "tasks": list(range(cfg.training.num_tasks)),
        "max_slides_per_task": int(args.max_slides_per_task),
        "k_patches": int(K_PATCHES),
        "config": args.config,
        "save_dir": args.save_dir,
        "merge_model_path": args.merge_model_path,
        "best_config": args.best_config,
        "tta_config": tta_cfg,
        "baseline": routing_summary(baseline_rows),
        "tta": routing_summary(tta_rows),
        "adapted_slides": int(sum(adapted_flags)),
        "prompt_updates": int(sum(prompt_updated_flags)),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] {baseline_csv}")
    print(f"[DONE] {tta_csv}")
    print(f"[DONE] {raw_npz}")
    print(f"[DONE] {summary_json}")


if __name__ == "__main__":
    main()
