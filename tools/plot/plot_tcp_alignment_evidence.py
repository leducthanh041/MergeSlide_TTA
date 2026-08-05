#!/usr/bin/env python3
"""Render a four-panel TCP-alignment evidence figure for CAST-Slide."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-mergeSlide"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


GOLD = "#FFD23F"
ORANGE = "#D35400"
NAVY = "#263F5B"
BASE = "#9AA0A8"
BASE_EDGE = "#454B52"
GREEN = "#2E8B57"
RED = "#C43D4E"
PURPLE = "#76528B"
PALE_PURPLE = "#E5D8EE"

STATE_ORDER = ["s00", "s10", "s01", "s11"]
STATE_LABELS = {
    "s00": "MergeSlide\nsource/source",
    "s10": "Backbone only\nadapted/source",
    "s01": "Prompt only\nsource/adapted",
    "s11": "CAST-Slide\nadapted/adapted",
}
STATE_COLORS = {
    "s00": BASE,
    "s10": "#547AA5",
    "s01": "#B88A44",
    "s11": GOLD,
}
TRANSITION_ORDER = ["retained_correct", "corrected", "regressed", "retained_wrong"]
TRANSITION_LABELS = {
    "retained_correct": "Correct retained",
    "corrected": "Wrong to correct",
    "regressed": "Correct to wrong",
    "retained_wrong": "Wrong retained",
}
TRANSITION_COLORS = {
    "retained_correct": "#5AAE61",
    "corrected": GOLD,
    "regressed": RED,
    "retained_wrong": "#B8BDC5",
}


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(low), float(high)


def state_fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        for fold, part in df.groupby("fold"):
            rows.append(
                {
                    "state": state,
                    "fold": int(fold),
                    "routing_acc": 100.0 * float(part[f"{state}_correct"].mean()),
                    "mean_margin": float(part[f"{state}_true_vs_wrong_margin"].mean()),
                    "mean_true_rank": float(part[f"{state}_true_rank"].mean()),
                }
            )
    return pd.DataFrame(rows)


def state_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        part = fold_df[fold_df["state"] == state]
        acc_low, acc_high = bootstrap_ci(part["routing_acc"].to_numpy())
        margin_low, margin_high = bootstrap_ci(part["mean_margin"].to_numpy())
        rows.append(
            {
                "state": state,
                "label": STATE_LABELS[state].replace("\n", " "),
                "routing_acc_mean": float(part["routing_acc"].mean()),
                "routing_acc_std": float(part["routing_acc"].std(ddof=1)),
                "routing_acc_ci_low": acc_low,
                "routing_acc_ci_high": acc_high,
                "margin_mean": float(part["mean_margin"].mean()),
                "margin_std": float(part["mean_margin"].std(ddof=1)),
                "margin_ci_low": margin_low,
                "margin_ci_high": margin_high,
                "true_rank_mean": float(part["mean_true_rank"].mean()),
                "n_folds": int(part["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def task_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_fold = (
        df.groupby(["fold", "task_id", "task_name"], as_index=False)
        .agg(
            baseline_acc=("s00_correct", lambda x: 100.0 * float(np.mean(x))),
            cast_acc=("s11_correct", lambda x: 100.0 * float(np.mean(x))),
            baseline_margin=("s00_true_vs_wrong_margin", "mean"),
            cast_margin=("s11_true_vs_wrong_margin", "mean"),
        )
    )
    return (
        per_fold.groupby(["task_id", "task_name"], as_index=False)
        .agg(
            baseline_acc_mean=("baseline_acc", "mean"),
            baseline_acc_std=("baseline_acc", "std"),
            cast_acc_mean=("cast_acc", "mean"),
            cast_acc_std=("cast_acc", "std"),
            baseline_margin_mean=("baseline_margin", "mean"),
            cast_margin_mean=("cast_margin", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values("task_id")
    )


def final_gate_summary(df: pd.DataFrame) -> pd.DataFrame:
    fold_df = (
        df.groupby("fold", as_index=False)
        .agg(
            merge_slide=("baseline_final_correct", lambda x: 100.0 * float(np.mean(x))),
            cast_slide=("cast_final_correct", lambda x: 100.0 * float(np.mean(x))),
        )
    )
    rows = []
    for method, column in [("MergeSlide", "merge_slide"), ("CAST-Slide", "cast_slide")]:
        low, high = bootstrap_ci(fold_df[column].to_numpy())
        rows.append(
            {
                "method": method,
                "routing_acc_mean": float(fold_df[column].mean()),
                "routing_acc_std": float(fold_df[column].std(ddof=1)),
                "routing_acc_ci_low": low,
                "routing_acc_ci_high": high,
                "n_folds": int(fold_df["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def transition_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts().reindex(TRANSITION_ORDER, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "transition": TRANSITION_ORDER,
            "count": [int(counts[name]) for name in TRANSITION_ORDER],
            "percentage": [100.0 * float(counts[name]) / max(total, 1) for name in TRANSITION_ORDER],
        }
    )


def add_box(ax, xy: tuple[float, float], width: float, height: float, text: str, **kwargs) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=kwargs.get("facecolor", "white"),
        edgecolor=kwargs.get("edgecolor", NAVY),
        linewidth=kwargs.get("linewidth", 2.0),
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=kwargs.get("fontsize", 12),
        fontweight=kwargs.get("fontweight", "normal"),
        color=kwargs.get("textcolor", "black"),
        zorder=4,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        zorder=2,
    )
    ax.add_patch(arrow)


def panel_mechanism(ax) -> None:
    ax.set_axis_off()
    ax.set_title("A  Task-aligned routing mechanism", loc="left", fontsize=18, fontweight="bold", pad=10)
    ax.text(0.02, 0.84, "MergeSlide", transform=ax.transAxes, fontsize=14, fontweight="bold", color=BASE_EDGE)
    ax.text(0.02, 0.34, "CAST-Slide", transform=ax.transAxes, fontsize=14, fontweight="bold", color=ORANGE)

    for y, adapted in [(0.60, False), (0.10, True)]:
        add_box(ax, (0.04, y), 0.17, 0.18, "WSI patch\nbag", facecolor="#F3ECF8", edgecolor=PURPLE)
        add_arrow(ax, (0.21, y + 0.09), (0.30, y + 0.09), NAVY)
        add_box(
            ax,
            (0.30, y),
            0.18,
            0.18,
            "adapted slide\nembedding  z'" if adapted else "source slide\nembedding  z0",
            facecolor="#FFF4BF" if adapted else "#ECEFF2",
            edgecolor=ORANGE if adapted else BASE_EDGE,
            fontweight="bold" if adapted else "normal",
        )
        add_arrow(ax, (0.48, y + 0.09), (0.57, y + 0.09), ORANGE if adapted else NAVY)
        add_box(
            ax,
            (0.57, y),
            0.18,
            0.18,
            "adapted task\nprompt bank  E'T" if adapted else "source task\nprompt bank  ET",
            facecolor="#FFF4BF" if adapted else PALE_PURPLE,
            edgecolor=ORANGE if adapted else PURPLE,
            fontweight="bold" if adapted else "normal",
        )
        add_arrow(ax, (0.75, y + 0.09), (0.84, y + 0.09), GREEN if adapted else RED)
        add_box(
            ax,
            (0.84, y),
            0.13,
            0.18,
            "correct\ntask" if adapted else "wrong\ntask",
            facecolor="#DFF2E5" if adapted else "#F7DFE2",
            edgecolor=GREEN if adapted else RED,
            fontweight="bold",
        )

    ax.text(
        0.50,
        0.01,
        "Conceptual path; Panels B-D report all paired WSIs across folds",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        color="#555555",
    )


def panel_attribution(ax, summary: pd.DataFrame) -> None:
    ax.set_title("B  Where does TCP improvement come from?", loc="left", fontsize=18, fontweight="bold", pad=10)
    part = summary.set_index("state").loc[STATE_ORDER]
    x = np.arange(len(STATE_ORDER))
    means = part["routing_acc_mean"].to_numpy()
    low = means - part["routing_acc_ci_low"].to_numpy()
    high = part["routing_acc_ci_high"].to_numpy() - means
    bars = ax.bar(
        x,
        means,
        width=0.66,
        color=[STATE_COLORS[state] for state in STATE_ORDER],
        edgecolor=[BASE_EDGE, NAVY, "#765019", ORANGE],
        linewidth=[1.4, 1.5, 1.5, 2.4],
        zorder=3,
    )
    ax.errorbar(x, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#282828", elinewidth=2.4, capsize=6, capthick=2.4, zorder=5)
    for idx, (bar, value) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + high[idx] + 0.8, f"{value:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[state] for state in STATE_ORDER], fontsize=10.5)
    ax.set_ylabel("Raw TCP routing accuracy (%)", fontsize=14)
    y_min = max(0.0, float(np.nanmin(part["routing_acc_ci_low"])) - 8.0)
    ax.set_ylim(y_min, min(103.0, float(np.nanmax(part["routing_acc_ci_high"])) + 8.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    prompt_gain = float(part.loc["s01", "routing_acc_mean"] - part.loc["s00", "routing_acc_mean"])
    joint_gain = float(part.loc["s11", "routing_acc_mean"] - part.loc["s00", "routing_acc_mean"])
    ax.text(
        0.98,
        0.97,
        f"Prompt-only gain: {prompt_gain:+.2f} pp\nJoint gain: {joint_gain:+.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": ORANGE, "alpha": 0.95},
    )


def panel_tasks(ax, tasks: pd.DataFrame) -> None:
    ax.set_title("C  Correct-task routing improves by cohort", loc="left", fontsize=18, fontweight="bold", pad=10)
    part = tasks.sort_values("task_id", ascending=False).reset_index(drop=True)
    y = np.arange(len(part))
    for idx, row in part.iterrows():
        before = float(row["baseline_acc_mean"])
        after = float(row["cast_acc_mean"])
        color = GREEN if after >= before else RED
        ax.plot([before, after], [idx, idx], color=color, linewidth=3.0, alpha=0.85, zorder=2)
        ax.scatter(before, idx, s=115, marker="o", color=BASE, edgecolor=BASE_EDGE, linewidth=1.4, zorder=4)
        ax.scatter(after, idx, s=250, marker="*", color=GOLD, edgecolor=ORANGE, linewidth=1.5, zorder=5)
        ax.text(max(before, after) + 1.0, idx, f"{after - before:+.1f} pp", va="center", fontsize=11, fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(part["task_name"], fontsize=12)
    ax.set_xlabel("Raw TCP routing accuracy (%)", fontsize=14)
    x_min = max(0.0, float(min(part["baseline_acc_mean"].min(), part["cast_acc_mean"].min())) - 8.0)
    ax.set_xlim(x_min, 108.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.scatter([], [], s=115, marker="o", color=BASE, edgecolor=BASE_EDGE, label="MergeSlide")
    ax.scatter([], [], s=250, marker="*", color=GOLD, edgecolor=ORANGE, label="CAST-Slide")
    ax.legend(loc="lower right", fontsize=11, frameon=True)


def panel_outcome(ax, transitions: pd.DataFrame, final_gate: pd.DataFrame) -> None:
    ax.set_title("D  WSI-level corrections and final gated routing", loc="left", fontsize=18, fontweight="bold", pad=10)
    left = 0.0
    for _, row in transitions.iterrows():
        width = float(row["percentage"])
        name = str(row["transition"])
        ax.barh(1.0, width, left=left, height=0.42, color=TRANSITION_COLORS[name], edgecolor="white", linewidth=1.2, zorder=3)
        if width >= 5.0:
            ax.text(left + width / 2, 1.0, f"{width:.1f}%", ha="center", va="center", fontsize=10.5, fontweight="bold")
        left += width

    gate = final_gate.set_index("method")
    base = float(gate.loc["MergeSlide", "routing_acc_mean"])
    cast = float(gate.loc["CAST-Slide", "routing_acc_mean"])
    ax.plot([base, cast], [0.15, 0.15], color=GREEN if cast >= base else RED, linewidth=4.0, zorder=2)
    ax.scatter(base, 0.15, s=180, marker="o", color=BASE, edgecolor=BASE_EDGE, linewidth=1.5, zorder=4)
    ax.scatter(cast, 0.15, s=360, marker="*", color=GOLD, edgecolor=ORANGE, linewidth=1.8, zorder=5)
    ax.annotate(
        f"MergeSlide\n{base:.1f}%",
        xy=(base, 0.15),
        xytext=(-12, -38),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=11,
    )
    ax.annotate(
        f"CAST-Slide\n{cast:.1f}%",
        xy=(cast, 0.15),
        xytext=(12, -38),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text((base + cast) / 2, 0.31, f"{cast - base:+.2f} pp", ha="center", fontsize=12, fontweight="bold", color=GREEN if cast >= base else RED)

    handles = [Rectangle((0, 0), 1, 1, color=TRANSITION_COLORS[name]) for name in TRANSITION_ORDER]
    labels = [TRANSITION_LABELS[name] for name in TRANSITION_ORDER]
    ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.56),
        ncol=1,
        fontsize=10,
        frameon=False,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([0.15, 1.0])
    ax.set_yticklabels(["Final routing decision", "Raw TCP transitions"], fontsize=11.5)
    ax.set_xlabel("WSIs / routing accuracy (%)", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.30, zorder=0)


def plot_outcome_only(
    df: pd.DataFrame,
    transitions: pd.DataFrame,
    final_gate: pd.DataFrame,
    out_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)

    transition_map = transitions.set_index("transition")
    bar_order = ["retained_correct", "corrected", "regressed", "retained_wrong"]
    x = np.arange(len(bar_order))
    percentages = np.array(
        [float(transition_map.loc[name, "percentage"]) for name in bar_order]
    )
    counts = [int(transition_map.loc[name, "count"]) for name in bar_order]
    bars = ax.bar(
        x,
        percentages,
        width=0.62,
        color=[TRANSITION_COLORS[name] for name in bar_order],
        edgecolor=["#27743F", ORANGE, "#7F1F2D", "#6D737B"],
        linewidth=[1.8, 3.0, 1.8, 1.8],
        zorder=3,
    )
    bars[1].set_hatch("///")
    bars[1].set_linewidth(4.0)
    max_percentage = max(float(percentages.max()), 1.0)
    label_x_offsets = [0.0, -0.10, 0.10, 0.0]
    for bar, percentage, count, x_offset in zip(
        bars, percentages, counts, label_x_offsets
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_offset,
            float(percentage) + 1.2,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color="#242424",
        )

    corrected = float(transition_map.loc["corrected", "percentage"])
    regressed = float(transition_map.loc["regressed", "percentage"])
    net_correction = corrected - regressed
    ax.set_xticks([])
    ax.set_xlim(-0.80, 4.30)
    ax.set_ylim(0, max_percentage + 15.0)
    ax.set_ylabel(
        "Percentage of paired WSIs (%)",
        fontsize=20,
        labelpad=9,
    )
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=4.5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.20, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_box_aspect(1)

    ax.text(
        0.96,
        0.95,
        f"Net correction: {net_correction:+.2f}%",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=GREEN if net_correction >= 0 else RED,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GREEN if net_correction >= 0 else RED,
            "linewidth": 1.5,
        },
    )
    legend_handles = []
    for name in bar_order:
        is_corrected = name == "corrected"
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=TRANSITION_COLORS[name],
                edgecolor=ORANGE if is_corrected else "none",
                linewidth=3.0 if is_corrected else 0.0,
                hatch="///" if is_corrected else None,
            )
        )
    legend_labels = [
        "MergeSlide correct\n" + r"$\rightarrow$ $\mathbf{CAST}$-$\mathbf{Slide}$ correct",
        "MergeSlide wrong\n" + r"$\rightarrow$ $\mathbf{CAST}$-$\mathbf{Slide}$ correct",
        "MergeSlide correct\n" + r"$\rightarrow$ $\mathbf{CAST}$-$\mathbf{Slide}$ wrong",
        "MergeSlide wrong\n" + r"$\rightarrow$ $\mathbf{CAST}$-$\mathbf{Slide}$ wrong",
    ]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.76),
        ncol=1,
        fontsize=15,
        frameon=True,
        borderpad=0.65,
        labelspacing=0.75,
        handletextpad=0.65,
        handlelength=1.80,
        handleheight=1.35,
    )
    legend.get_texts()[1].set_fontweight("bold")
    legend.get_texts()[1].set_color(ORANGE)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tag", default="ood")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {
        "fold",
        "task_id",
        "task_name",
        "s00_correct",
        "s11_correct",
        "s00_true_vs_wrong_margin",
        "s11_true_vs_wrong_margin",
        "baseline_final_correct",
        "cast_final_correct",
        "raw_transition",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary = state_fold_summary(df)
    attribution = state_summary(fold_summary)
    tasks = task_summary(df)
    final_gate = final_gate_summary(df)
    transitions = transition_summary(df, "raw_transition")

    fold_summary.to_csv(args.output_dir / "tcp_state_metrics_by_fold.csv", index=False)
    attribution.to_csv(args.output_dir / "component_attribution.csv", index=False)
    tasks.to_csv(args.output_dir / "routing_by_task.csv", index=False)
    final_gate.to_csv(args.output_dir / "final_gate_summary.csv", index=False)
    transitions.to_csv(args.output_dir / "routing_transitions.csv", index=False)

    summary = {
        "n_wsi": int(len(df)),
        "n_folds": int(df["fold"].nunique()),
        "raw_tcp_baseline": float(100.0 * df["s00_correct"].mean()),
        "raw_tcp_cast": float(100.0 * df["s11_correct"].mean()),
        "final_gate_baseline": float(100.0 * df["baseline_final_correct"].mean()),
        "final_gate_cast": float(100.0 * df["cast_final_correct"].mean()),
        "corrected_wsi": int((df["raw_transition"] == "corrected").sum()),
        "regressed_wsi": int((df["raw_transition"] == "regressed").sum()),
    }
    (args.output_dir / "tcp_evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    out_stem = args.output_dir / f"cast_slide_tcp_routing_outcome_{args.tag}"
    plot_outcome_only(df, transitions, final_gate, out_stem)
    print(f"[DONE] {out_stem.with_suffix('.png')}")
    print(f"[DONE] {out_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
