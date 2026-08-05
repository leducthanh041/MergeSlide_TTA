#!/usr/bin/env python3
"""Collect paired TCP-alignment evidence for MergeSlide and CAST-Slide."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
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
from mergeslide_tta.task_prompt_io import load_task_prompts_for_tasks
from mergeslide_tta.tta_adapter import MergeSlide_TTA, load_task_weights
from mergeslide_tta.utils import seed_torch


PROMPT_FN = {
    "BRCA": brca_prompts,
    "RCC": rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA": esca_prompts,
    "TGCT": tgct_prompts,
    "CESC": cesc_prompts,
}

STATE_NAMES = {
    "s00": "source_embedding_source_prompts",
    "s10": "adapted_embedding_source_prompts",
    "s01": "source_embedding_adapted_prompts",
    "s11": "adapted_embedding_adapted_prompts",
}


def build_class_embeddings(device: torch.device, task_names: list[str]) -> torch.Tensor:
    missing = [name for name in task_names if name not in PROMPT_FN]
    if missing:
        raise ValueError(f"Missing zero-shot prompt functions for: {missing}")

    print("[INFO] building global class embeddings for TCP fallback")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)
    _, templates = brca_prompts()
    prompts = []
    for task_name in task_names:
        class_prompts, _ = PROMPT_FN[task_name]()
        prompts.extend(class_prompts)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = titan.zero_shot_classifier(prompts, templates, device=str(device))
    del titan
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return classifier.to(device)


def forward_embedding(
    model: torch.nn.Module,
    features: torch.Tensor,
    coords: torch.Tensor,
    ps: torch.Tensor,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        embedding = model(features, coords, ps).detach().float()
    if was_training:
        model.train()
    return embedding


def state_metrics(
    embedding: torch.Tensor,
    prompts: torch.Tensor,
    true_task: int,
) -> dict[str, object]:
    scores = (embedding.float() @ prompts.float().T).squeeze(0)
    probs = F.softmax(scores, dim=0)
    pred_task = int(scores.argmax().item())
    true_score = float(scores[true_task].item())
    wrong_mask = torch.ones_like(scores, dtype=torch.bool)
    wrong_mask[true_task] = False
    max_wrong = float(scores[wrong_mask].max().item())
    rank = int((scores > scores[true_task]).sum().item()) + 1
    top2 = torch.topk(probs, k=min(2, probs.numel())).values
    top1_top2 = float((top2[0] - top2[1]).item()) if top2.numel() > 1 else 0.0
    return {
        "pred_task": pred_task,
        "correct": int(pred_task == true_task),
        "confidence": float(probs.max().item()),
        "true_probability": float(probs[true_task].item()),
        "true_score": true_score,
        "max_wrong_score": max_wrong,
        "true_vs_wrong_margin": true_score - max_wrong,
        "true_rank": rank,
        "top1_top2_margin": top1_top2,
        "scores": [float(value) for value in scores.detach().cpu().tolist()],
    }


def add_state_columns(row: dict[str, object], state: str, metrics: dict[str, object]) -> None:
    for key, value in metrics.items():
        if key == "scores":
            for task_id, score in enumerate(value):
                row[f"{state}_score_task_{task_id}"] = score
        else:
            row[f"{state}_{key}"] = value


def transition_label(before_correct: int, after_correct: int) -> str:
    labels = {
        (1, 1): "retained_correct",
        (0, 1): "corrected",
        (1, 0): "regressed",
        (0, 0): "retained_wrong",
    }
    return labels[(int(before_correct), int(after_correct))]


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty checkpoint: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--merge_model_path", required=True)
    parser.add_argument("--output_dir", default="logs/ablation/tcp_evidence/ood")
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=-1)
    parser.add_argument("--max_slides_per_task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--K_sub", type=int, default=300)
    parser.add_argument("--top_ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l2_anchor_beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--tta_param_scope", choices=["ln_only", "full"], default="ln_only")
    parser.add_argument("--entropy_threshold", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--select_mode", choices=["union", "intersection"], default="intersection")
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--ema_alpha_prompt", type=float, default=0.999)
    parser.add_argument("--delta_margin", type=float, default=0.10)
    parser.add_argument("--tp_anchor_beta", type=float, default=0.3)
    parser.add_argument("--gamma_margin", type=float, default=0.0)
    parser.add_argument("--tau_task", type=float, default=0.70)
    parser.add_argument("--dapc_loss_weight", type=float, default=1.0)
    parser.add_argument("--entropy_loss_weight", type=float, default=1.0)
    parser.add_argument("--dapc_tau_anchor", type=float, default=0.92)
    parser.add_argument("--dapc_beta", type=float, default=1.2)
    parser.add_argument("--no_teacher", action="store_true")
    parser.add_argument("--no_adapt_prompts", action="store_true")
    parser.add_argument("--no_task_agreement", action="store_true")
    parser.add_argument("--use_task_diversity", action="store_true")
    parser.add_argument("--use_dapc", action="store_true")
    parser.add_argument("--no_reset_prompt_per_task", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-fold CSV checkpoints in output_dir/folds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, int(cfg.training.seed) + args.seed)
    dataset = Sequential_Generic_MIL_Dataset(cfg)
    task_names = list(dataset.task_names)
    num_tasks = len(task_names)
    fold_end = int(cfg.training.num_folds) if args.fold_end < 0 else args.fold_end
    if not 0 <= args.fold_start < fold_end <= int(cfg.training.num_folds):
        raise ValueError(
            f"Invalid fold range [{args.fold_start}, {fold_end}) for "
            f"{cfg.training.num_folds} folds"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "paired_wsi_scores.csv"
    metadata_json = output_dir / "collection_metadata.json"
    signature_json = output_dir / "collection_signature.json"
    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    ignored_signature_keys = {"output_dir", "fold_start", "fold_end", "resume"}
    signature = {
        "task_names": task_names,
        "num_classes": [int(value) for value in dataset.num_classes],
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in ignored_signature_keys
        },
    }
    if args.resume and signature_json.is_file():
        previous_signature = json.loads(signature_json.read_text(encoding="utf-8"))
        if previous_signature != signature:
            raise ValueError(
                "Resume configuration does not match existing fold checkpoints. "
                "Use a different OUTPUT_DIR or rerun with RESUME=0."
            )
    elif args.resume and any(fold_dir.glob("fold_*.csv")):
        raise ValueError(
            f"Cannot safely resume fold CSVs without {signature_json}. "
            "Use a different OUTPUT_DIR or rerun with RESUME=0."
        )
    signature_json.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    pending_folds = []
    for fold in range(args.fold_start, fold_end):
        fold_csv = fold_dir / f"fold_{fold}.csv"
        if args.resume and fold_csv.is_file():
            completed_rows = read_rows(fold_csv)
            if completed_rows and all(int(row["fold"]) == fold for row in completed_rows):
                rows.extend(completed_rows)
                print(f"[RESUME] fold={fold} rows={len(completed_rows)} from {fold_csv}")
                continue
            print(f"[WARN] invalid fold checkpoint will be recomputed: {fold_csv}")
        pending_folds.append(fold)

    if not pending_folds:
        write_rows_atomic(output_csv, rows)
        metadata = {
            "states": STATE_NAMES,
            "task_names": task_names,
            "num_classes": [int(value) for value in dataset.num_classes],
            "fold_start": args.fold_start,
            "fold_end": fold_end,
            "num_records": len(rows),
            "config": str(args.config),
            "save_dir": str(args.save_dir),
            "merge_model_path": str(args.merge_model_path),
            "arguments": vars(args),
            "resumed_without_gpu_work": True,
        }
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[DONE] all requested folds already completed: {output_csv}")
        print(f"[DONE] {metadata_json}")
        return

    task_prompts = load_task_prompts_for_tasks(
        PROJECT_ROOT / "task_prompts.pt", task_names, device
    )
    class_embeddings = build_class_embeddings(device, task_names)

    print("[INFO] loading TITAN vision encoder")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)

    for fold in pending_folds:
        fold_rows: list[dict[str, object]] = []
        merge_path = Path(args.merge_model_path) / f"fold_{fold}" / "merged_final.pth"
        print(f"[INFO] fold={fold} checkpoint={merge_path}")
        titan.vision_encoder.load_state_dict(torch.load(merge_path, map_location="cpu"))
        task_paths = [
            str(Path(args.save_dir) / f"fold_{fold}" / f"task_{task_id}.pt")
            for task_id in range(num_tasks)
        ]
        task_weights = load_task_weights(task_paths, device)
        adapter = MergeSlide_TTA(
            backbone=titan.vision_encoder,
            task_prompts=task_prompts.detach().clone(),
            task_weights=task_weights,
            num_classes=dataset.num_classes,
            device=device,
            mode="tcp",
            all_class_embeddings=class_embeddings,
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
            use_task_agreement=not args.no_task_agreement,
            gamma=args.gamma,
            select_mode=args.select_mode,
            use_teacher=not args.no_teacher,
            tcp_inference_model="teacher",
            ema_alpha=args.ema_alpha,
            adapt_task_prompts=not args.no_adapt_prompts,
            ema_alpha_prompt=args.ema_alpha_prompt,
            delta_margin=args.delta_margin,
            tp_anchor_beta=args.tp_anchor_beta,
            gamma_margin=args.gamma_margin,
            tau_task=args.tau_task,
            use_dapc=args.use_dapc,
            dapc_loss_weight=args.dapc_loss_weight,
            entropy_loss_weight=args.entropy_loss_weight,
            dapc_tau_anchor=args.dapc_tau_anchor,
            dapc_beta=args.dapc_beta,
        )
        source_model = adapter.anchor
        if source_model is None:
            raise ValueError(
                "Evidence collection requires --use_dapc so the adapter owns "
                "an immutable source anchor."
            )

        for task_id, task_name in enumerate(task_names):
            if not args.no_reset_prompt_per_task and adapter.adapt_task_prompts:
                adapter.reset_task_prompts()
            _, _, loader = dataset.get_data_loaders(fold, task_id)
            iterator = tqdm(loader, desc=f"Fold {fold} | {task_name}", leave=False)
            for slide_index, (features, coords, label) in enumerate(iterator):
                if args.max_slides_per_task > 0 and slide_index >= args.max_slides_per_task:
                    break
                features = features.to(device)
                coords = coords.long().to(device)
                keep = min(int(features.shape[0]), int(K_PATCHES))
                patch_ids = torch.randperm(features.shape[0], device=device)[:keep]
                features = features[patch_ids]
                coords = coords[patch_ids]

                source_prompts = adapter.task_prompts_source.detach().clone()
                source_embedding = forward_embedding(
                    source_model, features, coords, adapter.ps
                )
                s00 = state_metrics(source_embedding, source_prompts, task_id)

                _, _, final_task, adapt_log = adapter.adapt_and_predict(features, coords)
                inference_model = adapter.teacher if adapter.use_teacher else adapter.backbone
                adapted_embedding = forward_embedding(
                    inference_model, features, coords, adapter.ps
                )
                adapted_prompts = adapter.task_prompts.detach().clone()
                s10 = state_metrics(adapted_embedding, source_prompts, task_id)
                s01 = state_metrics(source_embedding, adapted_prompts, task_id)
                s11 = state_metrics(adapted_embedding, adapted_prompts, task_id)

                baseline_final_task = int(s00["pred_task"])
                row: dict[str, object] = {
                    "fold": fold,
                    "task_id": task_id,
                    "task_name": task_name,
                    "slide_index": slide_index,
                    "label": int(label.item() if torch.is_tensor(label) else label),
                    "adapted": int(bool(adapt_log.get("slide/adapted", False))),
                    "prompt_updated": int(bool(adapt_log.get("adapt/prompt_updated", False))),
                    "baseline_final_pred_task": baseline_final_task,
                    "baseline_final_correct": int(baseline_final_task == task_id),
                    "baseline_fallback": 0,
                    "cast_final_pred_task": int(final_task),
                    "cast_final_correct": int(final_task == task_id),
                    "cast_fallback": int(bool(adapt_log.get("slide/tcp_fallback", False))),
                }
                add_state_columns(row, "s00", s00)
                add_state_columns(row, "s10", s10)
                add_state_columns(row, "s01", s01)
                add_state_columns(row, "s11", s11)
                row["raw_transition"] = transition_label(s00["correct"], s11["correct"])
                row["final_transition"] = transition_label(
                    row["baseline_final_correct"], row["cast_final_correct"]
                )
                fold_rows.append(row)

        fold_csv = fold_dir / f"fold_{fold}.csv"
        write_rows_atomic(fold_csv, fold_rows)
        rows.extend(fold_rows)
        print(f"[CHECKPOINT] fold={fold} rows={len(fold_rows)} saved={fold_csv}")

    if not rows:
        raise RuntimeError("No WSI records were collected")
    write_rows_atomic(output_csv, rows)

    metadata = {
        "states": STATE_NAMES,
        "task_names": task_names,
        "num_classes": [int(value) for value in dataset.num_classes],
        "fold_start": args.fold_start,
        "fold_end": fold_end,
        "num_records": len(rows),
        "config": str(args.config),
        "save_dir": str(args.save_dir),
        "merge_model_path": str(args.merge_model_path),
        "arguments": vars(args),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] {output_csv}")
    print(f"[DONE] {metadata_json}")


if __name__ == "__main__":
    main()
