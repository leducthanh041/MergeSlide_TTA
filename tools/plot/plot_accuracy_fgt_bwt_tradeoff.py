#!/usr/bin/env python3
"""Plot Mean ACC vs forgetting/backward-transfer trade-offs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-mergeSlide"

import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse


DEFAULT_ROOT = Path("logs/ablation/trade-off")


STYLE = {
    "CAST-Slide": {
        "color": "#FFD23F",
        "marker": "*",
        "size": 960,
        "edge": "#D35400",
        "lw": 2.4,
        "z": 12,
    },
    "LayerWise AdaMerging++": {
        "color": "#D8C7E8",
        "marker": "o",
        "size": 500,
        "edge": "#5B3A70",
        "lw": 1.8,
        "z": 5,
    },
    "AdaRank": {
        "color": "#D28E2C",
        "marker": "s",
        "size": 260,
        "edge": "#704500",
        "lw": 1.0,
        "z": 4,
    },
    "Hi-Vec": {
        "color": "#3A7D44",
        "marker": "P",
        "size": 280,
        "edge": "#173C1D",
        "lw": 1.0,
        "z": 4,
    },
    "MINGLE": {
        "color": "#B8325A",
        "marker": "X",
        "size": 220,
        "edge": "#5E102A",
        "lw": 1.2,
        "z": 7,
    },
    "WEMOE": {
        "color": "#547AA5",
        "marker": ">",
        "size": 280,
        "edge": "#263F5B",
        "lw": 1.0,
        "z": 4,
    },
    "CONCRETE": {
        "color": "#7A7F87",
        "marker": "h",
        "size": 290,
        "edge": "#363A40",
        "lw": 1.0,
        "z": 4,
    },
    "T3": {
        "color": "#85754E",
        "marker": "<",
        "size": 270,
        "edge": "#453A23",
        "lw": 1.0,
        "z": 4,
    },
}


ORDER = [
    "CAST-Slide",
    "LayerWise AdaMerging++",
    "AdaRank",
    "Hi-Vec",
    "MINGLE",
    "WEMOE",
    "T3",
    "CONCRETE",
]


def canonical_method(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    aliases = {
        "CAST-Slide": "CAST-Slide",
        "CAST-Slide (ours)": "CAST-Slide",
        "MergeSlide + TTA": "CAST-Slide",
        "MergeSlide + TTA (ours)": "CAST-Slide",
        "MergeSlide_TTA": "CAST-Slide",
        "MergeSlide_TTA (ours)": "CAST-Slide",
        "WEMOE (2 Layer)": "WEMOE",
        "WEMOE (2 layers)": "WEMOE",
    }
    return aliases.get(name, name)


def display_method(name: str) -> str:
    if name == "CAST-Slide":
        return "CAST-Slide (ours)"
    if name == "WEMOE":
        return "WEMOE (2 layers)"
    return name


def parse_metric(text: str) -> tuple[float, float]:
    match = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)%\s*\((-?[0-9]+(?:\.[0-9]+)?)%\)\s*$", text)
    if not match:
        raise ValueError(f"Cannot parse metric: {text}")
    return float(match.group(1)), float(match.group(2))


def read_results(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"Cannot parse line: {line}")
        method = canonical_method(parts[0])
        acc_mean, acc_std = parse_metric(parts[1])
        fgt_mean, fgt_std = parse_metric(parts[2])
        bwt_mean, bwt_std = parse_metric(parts[3])
        rows.append(
            {
                "method": method,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "fgt_mean": fgt_mean,
                "fgt_std": fgt_std,
                "bwt_mean": bwt_mean,
                "bwt_std": bwt_std,
            }
        )
    return pd.DataFrame(rows)


def add_uncertainty_ellipse(ax, x: float, y: float, x_std: float, y_std: float, color: str, zorder: int) -> None:
    ellipse = Ellipse(
        (x, y),
        width=max(2.0 * x_std, 0.15),
        height=max(2.0 * y_std, 0.15),
        facecolor=color,
        edgecolor=color,
        linewidth=1.2,
        alpha=0.18,
        zorder=zorder,
    )
    ax.add_patch(ellipse)


def draw_points(ax, df: pd.DataFrame, x_col: str, x_std_col: str, *, include_label: bool = True, scale: float = 1.0) -> None:
    for method in ORDER:
        row_df = df[df["method"] == method]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        st = STYLE[method]
        x = float(row[x_col])
        x_std = float(row[x_std_col])
        y = float(row["acc_mean"])
        y_std = float(row["acc_std"])

        add_uncertainty_ellipse(ax, x, y, x_std, y_std, st["color"], st["z"] - 3)
        ax.errorbar(
            x,
            y,
            xerr=x_std,
            yerr=y_std,
            fmt="none",
            ecolor=st["color"],
            elinewidth=3.2 * scale,
            capsize=7 * scale,
            capthick=3.0 * scale,
            alpha=1.0,
            zorder=st["z"] - 1,
        )
        ax.scatter(
            x,
            y,
            s=st["size"] * scale,
            marker=st["marker"],
            c=st["color"],
            edgecolors=st["edge"],
            linewidths=st["lw"] * max(scale, 0.8),
            label=display_method(method) if include_label else None,
            zorder=st["z"],
        )


def zoom_limits(df: pd.DataFrame, x_col: str, x_std_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    zoom_df = df[df["method"] != "AdaRank"].copy()
    x_min = float((zoom_df[x_col] - zoom_df[x_std_col]).min())
    x_max = float((zoom_df[x_col] + zoom_df[x_std_col]).max())
    y_min = float((zoom_df["acc_mean"] - zoom_df["acc_std"]).min())
    y_max = float((zoom_df["acc_mean"] + zoom_df["acc_std"]).max())

    x_pad = max((x_max - x_min) * 0.12, 0.20)
    y_pad = max((y_max - y_min) * 0.15, 0.25)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def full_limits(df: pd.DataFrame, x_col: str, x_std_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float((df[x_col] - df[x_std_col]).min())
    x_max = float((df[x_col] + df[x_std_col]).max())
    y_min = float((df["acc_mean"] - df["acc_std"]).min())
    y_max = float((df["acc_mean"] + df["acc_std"]).max())

    x_pad = max((x_max - x_min) * 0.06, 0.25)
    y_pad = max((y_max - y_min) * 0.08, 0.22)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_tradeoff(
    df: pd.DataFrame,
    root: Path,
    x_col: str,
    x_std_col: str,
    xlabel: str,
    out_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)

    draw_points(ax, df, x_col, x_std_col)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.20)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=8)
    ax.set_ylabel("Mean CLASS-IL ACC (%) ± STD", fontsize=20, labelpad=9)
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=4.5)
    ax.set_box_aspect(1)

    xlim, ylim = full_limits(df, x_col, x_std_col)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if x_col == "fgt_mean":
        ax.set_xscale("symlog", linthresh=3.0)
        ax.set_xticks([0, 1, 2, 3, 5, 10, 15])
        ax.set_xticklabels(["0", "1", "2", "3", "5", "10", "15"])
    else:
        ax.set_xscale("symlog", linthresh=3.0)
        ax.set_xticks([-15, -10, -5, -3, -2, -1, 0, 1])
        ax.set_xticklabels(["-15", "-10", "-5", "-3", "-2", "-1", "0", "1"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    legend_labels = [display_method(method) for method in ORDER if display_method(method) in seen]
    legend_handles = [seen[label] for label in legend_labels]
    legend_loc = "lower right" if x_col == "bwt_mean" else "lower left"
    legend = ax.legend(
        legend_handles,
        legend_labels,
        title="Method",
        loc=legend_loc,
        frameon=True,
        fontsize=16,
        title_fontsize=17,
        borderpad=0.50,
        labelspacing=0.38,
        handletextpad=0.42,
        handlelength=1.28,
        markerscale=0.88,
        ncol=1,
    )
    for text in legend.get_texts():
        if text.get_text() == "CAST-Slide (ours)":
            text.set_fontweight("bold")

    out_png = root / f"{out_stem}.png"
    out_pdf = root / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OOD ACC/FGT and ACC/BWT trade-offs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    df = read_results(root / "accuracy-fgt-bwt.txt")
    print(df[["method", "acc_mean", "acc_std", "fgt_mean", "fgt_std", "bwt_mean", "bwt_std"]].to_string(index=False))
    plot_tradeoff(
        df,
        root,
        "fgt_mean",
        "fgt_std",
        "Forgetting (± STD)",
        "cast_slide_accuracy_fgt_tradeoff_ood",
    )
    plot_tradeoff(
        df,
        root,
        "bwt_mean",
        "bwt_std",
        "Backward Transfer (± STD)",
        "cast_slide_accuracy_bwt_tradeoff_ood",
    )


if __name__ == "__main__":
    main()
