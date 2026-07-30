#!/usr/bin/env python3
"""Plot TCP routing accuracy by task for MergeSlide vs MergeSlide_TTA.

The plot uses per-fold task-level routing accuracies:
- MergeSlide TCP:
  logs/base_results/{IND,OOD}_results/test_new_run/baseline_tcp_routing_results.csv
- MergeSlide_TTA TCP:
  logs/final_best_tta/{ind,ood}/classil_tcp/results.csv

Each point/bar is mean ± std over folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


REPO_ROOT = Path("/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA")
OUT_DIR = REPO_ROOT / "logs" / "Ablations" / "routing_accuracy"
TASK_ORDER = ["BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC"]


@dataclass(frozen=True)
class SettingSpec:
    setting: str
    title: str
    baseline_csv: Path
    tta_csv: Path


SETTINGS = [
    SettingSpec(
        "ood",
        "Out-of-domain",
        REPO_ROOT / "logs/base_results/OOD_results/test_new_run/baseline_tcp_routing_results.csv",
        REPO_ROOT / "logs/final_best_tta/ood/classil_tcp/results.csv",
    ),
]


METHOD_STYLE = {
    "MergeSlide": {
        "color": "#1976c9",
        "light": "#87c9f5",
        "edge": "#0e4f8d",
        "marker": "o",
    },
    "MergeSlide_TTA": {
        "color": "#008f4f",
        "light": "#7bd99f",
        "edge": "#005f35",
        "marker": "D",
    },
}


def apply_vertical_gradient(ax, bars, bottom_color: str, top_color: str) -> None:
    cmap = LinearSegmentedColormap.from_list("bar_gradient", [bottom_color, top_color])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height) or height <= 0:
            continue
        x0 = bar.get_x()
        x1 = x0 + bar.get_width()
        image = ax.imshow(
            gradient,
            extent=[x0, x1, 0, height],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="bicubic",
            zorder=bar.get_zorder() + 0.05,
        )
        image.set_clip_path(bar)
        bar.set_facecolor((1, 1, 1, 0))


def load_method(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {method} routing CSV: {path}")
    df = pd.read_csv(path)
    required = {"fold", "task_id", "task_name", "routing_acc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["fold", "task_id", "task_name", "routing_acc"]].copy()
    out["method"] = method
    out["fold"] = out["fold"].astype(int)
    out["task_id"] = out["task_id"].astype(int)
    out["task_name"] = out["task_name"].astype(str)
    out["routing_acc"] = out["routing_acc"].astype(float)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method", "task_id", "task_name"], as_index=False)["routing_acc"]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["method", "task_id"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def plot_setting(spec: SettingSpec) -> None:
    df = pd.concat(
        [
            load_method(spec.baseline_csv, "MergeSlide"),
            load_method(spec.tta_csv, "MergeSlide_TTA"),
        ],
        ignore_index=True,
    )
    summary = summarize(df)

    summary_csv = OUT_DIR / f"{spec.setting}_tcp_routing_accuracy_by_task_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    x = list(range(len(TASK_ORDER)))
    offsets = {"MergeSlide": -0.18, "MergeSlide_TTA": 0.18}
    width = 0.32

    for method in ["MergeSlide", "MergeSlide_TTA"]:
        part = summary[summary["method"] == method].copy()
        means = []
        stds = []
        for task_name in TASK_ORDER:
            row = part[part["task_name"] == task_name]
            if row.empty:
                means.append(float("nan"))
                stds.append(0.0)
            else:
                means.append(float(row["mean"].iloc[0]))
                stds.append(float(row["std"].iloc[0]))
        style = METHOD_STYLE[method]
        xpos = [v + offsets[method] for v in x]
        bars = ax.bar(
            xpos,
            means,
            width=width,
            label=method,
            color=style["color"],
            edgecolor=style["edge"],
            linewidth=1.3 if method == "MergeSlide_TTA" else 0.9,
            alpha=0.92,
            zorder=3 if method == "MergeSlide_TTA" else 2,
        )
        apply_vertical_gradient(ax, bars, style["light"], style["color"])
        ax.errorbar(
            xpos,
            means,
            yerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=1.45,
            capsize=3.8,
            capthick=1.45,
            zorder=8,
        )
        ax.plot(
            xpos,
            means,
            color=style["edge"],
            marker=style["marker"],
            markersize=5.5 if method == "MergeSlide_TTA" else 4.8,
            markerfacecolor=style["edge"],
            markeredgecolor=style["edge"],
            markeredgewidth=1.0,
            linewidth=1.4,
            zorder=10,
        )

    ax.set_ylabel("Routing Accuracy", fontsize=17, labelpad=9)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_ORDER, fontsize=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=13, width=1.1, length=4)
    ax.tick_params(axis="x", width=1.1, length=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    legend_handles = [
        Patch(
            facecolor=METHOD_STYLE["MergeSlide"]["color"],
            edgecolor=METHOD_STYLE["MergeSlide"]["edge"],
            linewidth=1.0,
            label="MergeSlide",
        ),
        Patch(
            facecolor=METHOD_STYLE["MergeSlide_TTA"]["color"],
            edgecolor=METHOD_STYLE["MergeSlide_TTA"]["edge"],
            linewidth=1.3,
            label="MergeSlide_TTA",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=15,
        frameon=True,
        framealpha=0.94,
        borderpad=0.35,
        labelspacing=0.32,
        handlelength=1.6,
        handletextpad=0.55,
    )
    for text in legend.get_texts():
        if text.get_text() == "MergeSlide_TTA":
            text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = OUT_DIR / f"{spec.setting}_tcp_routing_accuracy_by_task.{suffix}"
        fig.savefig(out, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    print(f"Saved: {summary_csv}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SETTINGS:
        plot_setting(spec)


if __name__ == "__main__":
    main()
