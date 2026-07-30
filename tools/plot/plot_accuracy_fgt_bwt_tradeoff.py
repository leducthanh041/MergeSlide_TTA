#!/usr/bin/env python3
"""Plot Mean ACC vs forgetting/backward-transfer trade-offs."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mergeSlide")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse


ROOT = Path("/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA/logs/Ablations/trade-off")
INPUT_TXT = ROOT / "accuracy-fgt-bwt.txt"


STYLE = {
    "MergeSlide_TTA": {
        "color": "#ff2d55",
        "marker": "D",
        "size": 115,
        "edge": "black",
        "lw": 2.0,
        "z": 10,
    },
    "LayerWise AdaMerging++": {
        "color": "#f2b701",
        "marker": "h",
        "size": 135,
        "edge": "#7a4a00",
        "lw": 0.9,
        "z": 4,
    },
    "AdaRank": {
        "color": "#9b5de5",
        "marker": "p",
        "size": 140,
        "edge": "#4b237a",
        "lw": 0.9,
        "z": 4,
    },
    "Hi-Vec": {
        "color": "#00a6a6",
        "marker": "8",
        "size": 145,
        "edge": "#005f5f",
        "lw": 0.9,
        "z": 4,
    },
    "MINGLE": {
        "color": "#f97316",
        "marker": "<",
        "size": 135,
        "edge": "#8f3d00",
        "lw": 0.9,
        "z": 4,
    },
    "WEMOE": {
        "color": "#2ca02c",
        "marker": ">",
        "size": 140,
        "edge": "#145214",
        "lw": 0.9,
        "z": 4,
    },
    "CONCRETE": {
        "color": "#8c564b",
        "marker": "H",
        "size": 145,
        "edge": "#4b2a24",
        "lw": 0.9,
        "z": 4,
    },
    "T3": {
        "color": "#1f77b4",
        "marker": "s",
        "size": 128,
        "edge": "#0d3f67",
        "lw": 0.9,
        "z": 4,
    },
}


ORDER = [
    "MergeSlide_TTA",
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
        "MergeSlide + TTA": "MergeSlide_TTA",
        "MergeSlide + TTA (ours)": "MergeSlide_TTA",
        "MergeSlide_TTA (ours)": "MergeSlide_TTA",
        "WEMOE (2 Layer)": "WEMOE",
        "WEMOE (2 layers)": "WEMOE",
    }
    return aliases.get(name, name)


def display_method(name: str) -> str:
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
        edgecolor="none",
        alpha=0.16,
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
            elinewidth=1.55 * scale,
            capsize=3.5 * scale,
            capthick=1.25 * scale,
            alpha=0.82,
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


def plot_tradeoff(df: pd.DataFrame, x_col: str, x_std_col: str, xlabel: str, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(4.25, 3.85), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98)

    draw_points(ax, df, x_col, x_std_col)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.20)
    ax.set_xlabel(xlabel, fontsize=14.0, labelpad=6)
    ax.set_ylabel("Mean CLASS-IL ACC (%) ± STD", fontsize=14.0, labelpad=7)
    ax.tick_params(axis="both", labelsize=11.8, width=1.1, length=4.0)
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
        fontsize=8.8,
        title_fontsize=9.4,
        borderpad=0.50,
        labelspacing=0.38,
        handletextpad=0.42,
        handlelength=1.28,
        markerscale=0.88,
        ncol=1,
    )
    for text in legend.get_texts():
        if text.get_text() == "MergeSlide_TTA":
            text.set_fontweight("bold")

    out_png = ROOT / f"{out_stem}.png"
    out_pdf = ROOT / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    df = read_results(INPUT_TXT)
    print(df[["method", "acc_mean", "acc_std", "fgt_mean", "fgt_std", "bwt_mean", "bwt_std"]].to_string(index=False))
    plot_tradeoff(df, "fgt_mean", "fgt_std", "Forgetting (± STD)", "accuracy_fgt_tradeoff")
    plot_tradeoff(df, "bwt_mean", "bwt_std", "Backward Transfer (± STD)", "accuracy_bwt_tradeoff")


if __name__ == "__main__":
    main()
