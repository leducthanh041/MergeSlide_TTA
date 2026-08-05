#!/usr/bin/env python3
"""Plot the four-state task-prompt adaptation ablation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-mergeSlide"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


STATES = ["s00", "s10", "s01", "s11"]
LABELS = [
    "Source embedding\n+\nSource prompt",
    "Adapted embedding\n+\nSource prompt",
    "Source embedding\n+\nAdapted prompt",
    "Adapted embedding\n+\nAdapted prompt",
]
COLORS = ["#B6BBC3", "#547AA5", "#FFD23F", "#F4A261"]
EDGES = ["#5B6169", "#263F5B", "#D35400", "#A63F00"]


def adapted_prompt_legend_entry(index: int, embedding: str) -> HPacker:
    symbol = DrawingArea(36, 32, 0, 0)
    symbol.add_artist(
        Rectangle(
            (2, 3),
            31,
            25,
            facecolor=COLORS[index],
            edgecolor=EDGES[index],
            linewidth=3.0,
            hatch="///",
        )
    )
    prompt_line = HPacker(
        children=[
            TextArea("+ ", textprops={"fontsize": 13.5}),
            TextArea(
                "Adapted prompt",
                textprops={"fontsize": 13.5, "color": "#D35400", "weight": "bold"},
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    label = VPacker(
        children=[TextArea(embedding, textprops={"fontsize": 13.5}), prompt_line],
        align="left",
        pad=0,
        sep=1,
    )
    return HPacker(children=[symbol, label], align="center", pad=0, sep=6)


def summarize_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    columns = [f"{state}_correct" for state in STATES]
    missing = sorted(set(columns + ["fold"]) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    fold_scores = df.groupby("fold", sort=True)[columns].mean().mul(100.0)
    if fold_scores.empty:
        raise ValueError("Input CSV contains no fold data")

    rows = []
    for state, column, label in zip(STATES, columns, LABELS):
        values = fold_scores[column]
        rows.append(
            {
                "state": state,
                "label": label,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_ablation(summary: pd.DataFrame, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.98)

    x = np.arange(len(summary), dtype=float)
    means = summary["mean"].to_numpy(dtype=float)
    stds = summary["std"].to_numpy(dtype=float)
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=COLORS,
        edgecolor=EDGES,
        linewidth=[1.8, 1.8, 3.0, 3.0],
        yerr=stds,
        error_kw={
            "ecolor": "#353A40",
            "elinewidth": 3.0,
            "capsize": 7,
            "capthick": 2.8,
        },
        zorder=3,
    )
    bars[2].set_hatch("///")
    bars[3].set_hatch("///")

    ax.set_xticks([])
    ax.set_xlim(-0.65, len(summary) - 0.35)

    lower = max(0.0, float(np.floor((means - stds).min() / 5.0) * 5.0 - 5.0))
    upper = min(100.0, float((means + stds).max() + 5.0))
    ax.set_ylim(lower, upper)
    ax.set_ylabel("TCP Routing Accuracy (%)", fontsize=20, labelpad=9)
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=4.5)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_box_aspect(1)

    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor=edge,
            linewidth=3.0 if index >= 2 else 1.8,
            hatch="///" if index >= 2 else None,
        )
        for index, (color, edge) in enumerate(zip(COLORS, EDGES))
    ]
    legend_labels = [
        "Source embedding\n+ Source prompt",
        "Adapted embedding\n+ Source prompt",
        "Source embedding\n+ Adapted prompt",
        "Adapted embedding\n+ Adapted prompt",
    ]
    legend_left = ax.legend(
        legend_handles[:2],
        legend_labels[:2],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        fontsize=13.5,
        frameon=True,
        borderpad=0.65,
        labelspacing=0.75,
        handletextpad=0.65,
        handlelength=1.80,
        handleheight=1.35,
    )
    ax.add_artist(legend_left)

    right_panel = VPacker(
        children=[
            adapted_prompt_legend_entry(2, "Source embedding"),
            adapted_prompt_legend_entry(3, "Adapted embedding"),
        ],
        align="left",
        pad=0,
        sep=6,
    )
    legend_right = AnchoredOffsetbox(
        loc="upper right",
        child=right_panel,
        pad=0.35,
        borderpad=0.55,
        frameon=True,
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
    )
    legend_right.patch.set_edgecolor("#D0D0D0")
    legend_right.patch.set_linewidth(1.2)
    ax.add_artist(legend_right)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tag", default="ood")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    summary = summarize_by_fold(df)
    output_stem = args.output_dir / f"cast_slide_task_prompt_ablation_{args.tag}"
    plot_ablation(summary, output_stem)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"[DONE] {output_stem.with_suffix('.png')}")
    print(f"[DONE] {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
