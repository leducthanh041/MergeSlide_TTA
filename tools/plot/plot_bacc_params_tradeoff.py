#!/usr/bin/env python3
"""Plot bACC vs updated parameters from ablation txt files."""

from pathlib import Path
import argparse
import os
import re

# Avoid GUI backend/display probing and slow font-cache writes on shared storage.
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-mergeSlide"

import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("logs/ablation/trade-off")


def canonical_method(name: str) -> str:
    name = name.strip().strip('"')
    name = re.sub(r"\s+", " ", name)
    aliases = {
        "CAST-Slide": "CAST-Slide",
        "CAST-Slide (ours)": "CAST-Slide",
        "MergeSlide + TTA (ours)": "CAST-Slide",
        "MergeSlide_TTA": "CAST-Slide",
        "MergeSlide_TTA (ours)": "CAST-Slide",
        "AdaMerging": "LayerWise AdaMerging++",
        "WEMOE (2 Layer)": "WEMOE",
    }
    return aliases.get(name, name)


def display_method(name: str) -> str:
    labels = {
        "CAST-Slide": "CAST-Slide (ours)",
        "LayerWise AdaMerging++": "LayerWise AdaMerging++",
        "AdaRank": "AdaRank",
        "Hi-Vec": "Hi-Vec",
        "MINGLE": "MINGLE",
        "WEMOE": "WEMOE (2 layers)",
        "T3": "T3",
        "CONCRETE": "CONCRETE",
    }
    return labels.get(name, name)


def read_params(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        match = re.match(r"^(.*?)\s+([0-9,]+)\s*$", line.strip())
        if not match:
            raise ValueError(f"Cannot parse params line: {line}")
        method = canonical_method(match.group(1))
        params = int(match.group(2).replace(",", ""))
        rows.append({"method": method, "params": params})
    return pd.DataFrame(rows).drop_duplicates("method")


def read_results(path: Path, setting: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)%\s*\(([0-9]+(?:\.[0-9]+)?)%\)")
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        match = pattern.search(line)
        if not match:
            raise ValueError(f"Cannot parse result line: {line}")
        method = canonical_method(line[: match.start()].replace("\t", " "))
        rows.append(
            {
                "setting": setting,
                "method": method,
                "bacc_mean": float(match.group(1)),
                "bacc_std": float(match.group(2)),
            }
        )
    return pd.DataFrame(rows)


def build_dataframe(root: Path) -> pd.DataFrame:
    params_df = read_params(root / "params.txt")
    results_df = read_results(root / "results_ood.txt", "OOD")

    df = results_df.merge(params_df, on="method", how="left")
    if df["params"].isna().any():
        missing = df.loc[df["params"].isna(), "method"].unique().tolist()
        raise ValueError(f"Missing params for methods: {missing}")

    df = df[df["method"] != "T3"].copy()
    df["label"] = df["method"].map(display_method)
    return df


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
    "T3": {
        "color": "#85754E",
        "marker": "<",
        "size": 270,
        "edge": "#453A23",
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
}


def plot_one(ax, df: pd.DataFrame, setting: str, show_title: bool = True) -> None:
    sub = df[df["setting"] == setting].copy()
    for _, row in sub.iterrows():
        st = STYLE.get(
            row["method"],
            {"color": "gray", "marker": "o", "size": 90, "edge": "black", "lw": 0.8, "z": 3},
        )
        ax.errorbar(
            row["params"],
            row["bacc_mean"],
            yerr=row["bacc_std"],
            fmt="none",
            ecolor=st["color"],
            elinewidth=3.2,
            capsize=7,
            capthick=3.0,
            alpha=1.0,
            zorder=st["z"] - 1,
        )
        ax.scatter(
            row["params"],
            row["bacc_mean"],
            s=st["size"],
            marker=st["marker"],
            c=st["color"],
            edgecolors=st["edge"],
            linewidths=st["lw"],
            zorder=st["z"],
            label=row["label"],
        )

    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlim(100, 2_500_000)
    ax.set_ylim(62.2, 90.5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
    if show_title:
        title = {"IND": "In-domain", "OOD": "Out-of-domain"}.get(setting, setting)
        ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Updated parameters (symlog scale)", fontsize=17, labelpad=8)
    ax.set_ylabel("CLASS-IL bACC (%) ± STD", fontsize=18, labelpad=9)
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=4.5)
    ax.set_box_aspect(1)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    order = [
        "CAST-Slide (ours)",
        "LayerWise AdaMerging++",
        "AdaRank",
        "Hi-Vec",
        "MINGLE",
        "WEMOE (2 layers)",
        "CONCRETE",
    ]
    legend_handles = [seen[label] for label in order if label in seen]
    legend_labels = [label for label in order if label in seen]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        title="Method",
        loc="lower left",
        frameon=True,
        fontsize=16,
        title_fontsize=17,
        borderpad=0.52,
        labelspacing=0.40,
        handletextpad=0.45,
        handlelength=1.32,
        markerscale=0.92,
    )
    for text in legend.get_texts():
        if text.get_text() == "CAST-Slide (ours)":
            text.set_fontweight("bold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the OOD bACC/parameter trade-off.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(root)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    plot_one(ax, df, "OOD", show_title=False)

    out_png = root / "cast_slide_bacc_params_tradeoff_ood.png"
    out_pdf = root / "cast_slide_bacc_params_tradeoff_ood.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
