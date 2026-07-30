#!/usr/bin/env python3
"""Plot bACC vs updated parameters from ablation txt files."""

from pathlib import Path
import os
import re

import pandas as pd

# Avoid GUI backend/display probing and slow font-cache writes on shared storage.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mergeSlide")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


ROOT = Path("/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA/logs/Ablations/trade-off")
PARAMS_TXT = ROOT / "params.txt" 
IND_TXT = ROOT / "results_ind.txt"
OOD_TXT = ROOT / "results_ood.txt"

OUT_PNG = ROOT / "bacc_params_tradeoff_ood.png"
OUT_PDF = ROOT / "bacc_params_tradeoff_ood.pdf"


def canonical_method(name: str) -> str:
    name = name.strip().strip('"')
    name = re.sub(r"\s+", " ", name)
    aliases = {
        "MergeSlide + TTA (ours)": "MergeSlide_TTA",
        "MergeSlide_TTA": "MergeSlide_TTA",
        "MergeSlide_TTA (ours)": "MergeSlide_TTA",
        "AdaMerging": "LayerWise AdaMerging++",
        "WEMOE (2 Layer)": "WEMOE",
    }
    return aliases.get(name, name)


def display_method(name: str) -> str:
    labels = {
        "MergeSlide_TTA": "MergeSlide_TTA",
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


def build_dataframe() -> pd.DataFrame:
    params_df = read_params(PARAMS_TXT)
    results_df = pd.concat(
        [
            read_results(IND_TXT, "IND"),
            read_results(OOD_TXT, "OOD"),
        ],
        ignore_index=True,
    )

    df = results_df.merge(params_df, on="method", how="left")
    if df["params"].isna().any():
        missing = df.loc[df["params"].isna(), "method"].unique().tolist()
        raise ValueError(f"Missing params for methods: {missing}")

    df = df[df["method"] != "T3"].copy()
    df["label"] = df["method"].map(display_method)
    return df


STYLE = {
    "MergeSlide_TTA": {
        "color": "#ff2d55",
        "marker": "D",
        "size": 120,
        "edge": "black",
        "lw": 2.1,
        "z": 10,
    },
    "LayerWise AdaMerging++": {
        "color": "#f2b701",
        "marker": "h",
        "size": 145,
        "edge": "#7a4a00",
        "lw": 0.9,
        "z": 4,
    },
    "AdaRank": {
        "color": "#9b5de5",
        "marker": "p",
        "size": 150,
        "edge": "#4b237a",
        "lw": 0.9,
        "z": 4,
    },
    "Hi-Vec": {
        "color": "#00a6a6",
        "marker": "8",
        "size": 155,
        "edge": "#005f5f",
        "lw": 0.9,
        "z": 4,
    },
    "MINGLE": {
        "color": "#f97316",
        "marker": "<",
        "size": 140,
        "edge": "#8f3d00",
        "lw": 0.9,
        "z": 4,
    },
    "WEMOE": {
        "color": "#2ca02c",
        "marker": ">",
        "size": 145,
        "edge": "#145214",
        "lw": 0.9,
        "z": 4,
    },
    "T3": {
        "color": "#1f4e99",
        "marker": "d",
        "size": 155,
        "edge": "#0b254f",
        "lw": 0.9,
        "z": 4,
    },
    "CONCRETE": {
        "color": "#8c564b",
        "marker": "H",
        "size": 155,
        "edge": "#4b2a24",
        "lw": 0.9,
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
            elinewidth=1.7,
            capsize=4,
            capthick=1.4,
            alpha=0.85,
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
    ax.set_ylim(62.2, 84.5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
    if show_title:
        title = {"IND": "In-domain", "OOD": "Out-of-domain"}.get(setting, setting)
        ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Updated parameters (symlog scale)", fontsize=12.6, labelpad=5)
    ax.set_ylabel("CLASS-IL bACC (%) ± STD", fontsize=14.0, labelpad=7)
    ax.tick_params(axis="both", labelsize=11.8, width=1.1, length=4.0)
    ax.set_box_aspect(1)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    order = [
        "MergeSlide_TTA",
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
        fontsize=8.8,
        title_fontsize=9.4,
        borderpad=0.52,
        labelspacing=0.40,
        handletextpad=0.45,
        handlelength=1.32,
        markerscale=0.92,
    )
    for text in legend.get_texts():
        if text.get_text() == "MergeSlide_TTA":
            text.set_fontweight("bold")


def main() -> None:
    df = build_dataframe()
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    plot_one(ax, df, "OOD", show_title=False)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
