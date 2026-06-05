#!/usr/bin/env python3
"""Visualize ESCA/CESC task-prompt routing diagnostics.

Input is the per-slide debug CSV produced by test_classIL_task_prompt.py with
--debug_route. Outputs PNG figures and a compact text summary.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mergeslide")

import matplotlib.pyplot as plt
import numpy as np
import torch


TASKS = ["BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC"]
FOCUS = {"ESCA", "CESC"}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict, key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _to_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _save_heatmap(matrix: np.ndarray, title: str, path: Path, fmt: str = ".0f") -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(TASKS)), TASKS, rotation=35, ha="right")
    ax.set_yticks(range(len(TASKS)), TASKS)
    ax.set_xlabel("Predicted route")
    ax.set_ylabel("True task")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:{fmt}}" if np.isfinite(val) else "nan"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _route_confusions(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(TASKS)}
    counts = np.zeros((len(TASKS), len(TASKS)), dtype=float)
    for row in rows:
        true_name = row["task_name"]
        pred_name = row["pred_task_name"]
        if true_name in idx and pred_name in idx:
            counts[idx[true_name], idx[pred_name]] += 1
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return counts, rates * 100.0


def _plot_route_accuracy(rows: list[dict], out: Path) -> dict[str, float]:
    total = Counter(row["task_name"] for row in rows)
    correct = Counter(row["task_name"] for row in rows if _to_int(row, "route_correct") == 1)
    acc = {task: correct[task] / max(total[task], 1) * 100.0 for task in TASKS}

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    vals = [acc[t] for t in TASKS]
    colors = ["#d95f02" if t in FOCUS else "#4c78a8" for t in TASKS]
    ax.bar(TASKS, vals, color=colors)
    ax.axhline(np.mean(vals), color="#333333", linestyle="--", linewidth=1.2, label="mean")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Route accuracy (%)")
    ax.set_title("Task-prompt routing accuracy by true task")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return acc


def _plot_esca_cesc_counts(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, true_task in zip(axes, ["ESCA", "CESC"]):
        subset = [r for r in rows if r["task_name"] == true_task]
        counts = Counter(r["pred_task_name"] for r in subset)
        vals = [counts[t] for t in TASKS]
        colors = ["#d95f02" if t in FOCUS else "#9ecae9" for t in TASKS]
        ax.bar(TASKS, vals, color=colors)
        ax.set_title(f"True {true_task}: routed task counts")
        ax.set_xlabel("Predicted route")
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + max(vals) * 0.015, str(v), ha="center", fontsize=8)
    axes[0].set_ylabel("Slides")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_margin_hist(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for task, color in [("ESCA", "#1f77b4"), ("CESC", "#d95f02")]:
        vals = [_to_float(r, "route_margin") for r in rows if r["task_name"] == task]
        vals = [v for v in vals if np.isfinite(v)]
        ax.hist(vals, bins=35, alpha=0.58, density=True, label=task, color=color)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Top1 - Top2 route score margin")
    ax.set_ylabel("Density")
    ax.set_title("ESCA/CESC routing margin distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_score_scatter(rows: list[dict], out: Path) -> None:
    xs, ys, colors = [], [], []
    for row in rows:
        if row["task_name"] not in FOCUS:
            continue
        scores = {row.get(f"top{i}_task_name"): _to_float(row, f"top{i}_score") for i in (1, 2, 3)}
        if "ESCA" not in scores or "CESC" not in scores:
            continue
        xs.append(scores["ESCA"])
        ys.append(scores["CESC"])
        colors.append("#d95f02" if row["task_name"] == "CESC" else "#1f77b4")
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(xs, ys, s=13, alpha=0.45, c=colors, edgecolors="none")
    if xs and ys:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("ESCA route score")
    ax.set_ylabel("CESC route score")
    ax.set_title("ESCA vs CESC top-k route score scatter")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="true ESCA", markerfacecolor="#1f77b4", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="true CESC", markerfacecolor="#d95f02", markersize=7),
    ]
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_fold_trends(rows: list[dict], out: Path) -> None:
    fold_task = defaultdict(lambda: [0, 0])
    for row in rows:
        task = row["task_name"]
        if task not in FOCUS:
            continue
        fold = _to_int(row, "fold")
        fold_task[(fold, task)][1] += 1
        fold_task[(fold, task)][0] += int(_to_int(row, "route_correct") == 1)

    folds = sorted({fold for fold, _ in fold_task})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for task, color in [("ESCA", "#1f77b4"), ("CESC", "#d95f02")]:
        vals = []
        for fold in folds:
            corr, total = fold_task[(fold, task)]
            vals.append(corr / max(total, 1) * 100.0)
        ax.plot(folds, vals, marker="o", label=task, color=color)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Route accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ESCA/CESC route accuracy across folds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_prompt_cosine(prompt_path: Path, out: Path) -> np.ndarray | None:
    if not prompt_path.exists():
        return None
    prompts = torch.load(prompt_path, map_location="cpu").float()
    prompts = torch.nn.functional.normalize(prompts, dim=1)
    cosine = (prompts @ prompts.T).numpy()
    _save_heatmap(cosine, "task_prompts.pt cosine similarity", out, fmt=".2f")
    return cosine


def _write_summary(rows: list[dict], route_acc: dict[str, float], cosine: np.ndarray | None, out: Path) -> None:
    counts, rates = _route_confusions(rows)
    idx = {name: i for i, name in enumerate(TASKS)}
    lines = [
        "# ESCA/CESC Routing Visualization Summary",
        "",
        f"rows: {len(rows)}",
        "",
        "## Route accuracy",
    ]
    for task in TASKS:
        lines.append(f"- {task}: {route_acc[task]:.2f}%")
    lines.extend([
        "",
        "## Focus pair confusion",
        f"- True CESC -> ESCA: {counts[idx['CESC'], idx['ESCA']]:.0f} ({rates[idx['CESC'], idx['ESCA']]:.2f}%)",
        f"- True CESC -> CESC: {counts[idx['CESC'], idx['CESC']]:.0f} ({rates[idx['CESC'], idx['CESC']]:.2f}%)",
        f"- True ESCA -> ESCA: {counts[idx['ESCA'], idx['ESCA']]:.0f} ({rates[idx['ESCA'], idx['ESCA']]:.2f}%)",
        f"- True ESCA -> CESC: {counts[idx['ESCA'], idx['CESC']]:.0f} ({rates[idx['ESCA'], idx['CESC']]:.2f}%)",
    ])
    if cosine is not None:
        lines.extend([
            "",
            "## task_prompts.pt cosine",
            f"- cos(CESC, ESCA): {cosine[idx['CESC'], idx['ESCA']]:.4f}",
            f"- cos(CESC, NSCLC): {cosine[idx['CESC'], idx['NSCLC']]:.4f}",
            f"- cos(ESCA, NSCLC): {cosine[idx['ESCA'], idx['NSCLC']]:.4f}",
        ])
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="debug_route_tcp.csv")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--task_prompts", type=Path, default=Path("task_prompts.pt"))
    args = parser.parse_args()

    rows = _read_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts, rates = _route_confusions(rows)
    _save_heatmap(counts, "Routing confusion counts", args.out_dir / "route_confusion_counts.png", fmt=".0f")
    _save_heatmap(rates, "Routing confusion rate (%)", args.out_dir / "route_confusion_rate.png", fmt=".1f")
    route_acc = _plot_route_accuracy(rows, args.out_dir / "route_accuracy_by_task.png")
    _plot_esca_cesc_counts(rows, args.out_dir / "esca_cesc_routed_task_counts.png")
    _plot_margin_hist(rows, args.out_dir / "esca_cesc_route_margin_hist.png")
    _plot_score_scatter(rows, args.out_dir / "esca_cesc_score_scatter.png")
    _plot_fold_trends(rows, args.out_dir / "esca_cesc_route_accuracy_by_fold.png")
    cosine = _plot_prompt_cosine(args.task_prompts, args.out_dir / "task_prompt_cosine_heatmap.png")
    _write_summary(rows, route_acc, cosine, args.out_dir / "summary.md")

    print(f"[INFO] wrote visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
