#!/usr/bin/env python3
"""Plot prefix performance drop for the first five continual tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from plot_tta_guided_performance_drop import METHODS, load_method, plot_panel, summarize, validate_coverage


DEFAULT_OUT_DIR = Path("logs/ablation/performance-drop")
TASK_LABELS = {
    0: "TCGA-BRCA (first task)",
    1: "TCGA-RCC (second task)",
    2: "TCGA-NSCLC (third task)",
    3: "TCGA-ESCA (fourth task)",
    4: "TCGA-TGCT (fifth task)",
}


def fit_y_axis(ax, summary: pd.DataFrame) -> None:
    lower = float((summary["mean"] - summary["std"]).min())
    upper = float((summary["mean"] + summary["std"]).max())
    span = max(upper - lower, 0.04)
    ax.set_ylim(max(0.0, lower - 0.08 * span), min(1.0, upper + 0.08 * span))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def plot(setting: str, mode: str, metric: str, out_dir: Path) -> None:
    frames = [load_method(spec, setting, metric, mode) for spec in METHODS]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError(f"No data loaded for setting={setting}")

    data = pd.concat(frames, ignore_index=True)
    validate_coverage(data, setting)

    summaries = []
    fig, axes = plt.subplots(1, 5, figsize=(25.5, 5.2), sharey=False)
    for ax, eval_task in zip(axes, TASK_LABELS):
        summary = summarize(data, eval_task, metric)
        summary.insert(0, "setting", setting)
        summary.insert(1, "mode", mode)
        summary.insert(2, "metric", metric)
        summary.insert(3, "eval_task", eval_task)
        summary.insert(4, "task_name", TASK_LABELS[eval_task])
        summaries.append(summary)

        plot_panel(ax, summary, setting, eval_task)
        fit_y_axis(ax, summary)
        ax.set_title(TASK_LABELS[eval_task], fontsize=20)
        ax.set_xlabel("Number of Tasks", fontsize=22, labelpad=9)

        if eval_task in {0, 2, 3, 4}:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                handles,
                labels,
                loc="lower left",
                fontsize=14.0,
                frameon=True,
                framealpha=0.92,
                ncol=1,
                title="Method",
                title_fontsize=15.0,
                borderpad=0.42,
                labelspacing=0.30,
                handlelength=1.50,
                handletextpad=0.50,
            )
            for text in legend.get_texts():
                if text.get_text() == "CAST-Slide (ours)":
                    text.set_weight("bold")

    metric_label = "Accuracy" if metric == "acc" else "Balanced Accuracy"
    axes[0].set_ylabel(f"{metric_label} (CLASS-IL)", fontsize=22, labelpad=11)

    fig.tight_layout(w_pad=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cast_slide_performance_drop_first_five_tasks_{setting}_{mode}_{metric}"
    pd.concat(summaries, ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / f'{stem}.csv'}")
    print(f"Saved: {out_dir / f'{stem}.png'}")
    print(f"Saved: {out_dir / f'{stem}.pdf'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot performance drop for the first five continual tasks.")
    parser.add_argument("--setting", choices=("ind", "ood"), default="ood")
    parser.add_argument("--mode", choices=("naive", "tcp"), default="naive")
    parser.add_argument("--metric", choices=("acc", "bacc"), default="acc")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    plot(args.setting, args.mode, args.metric, args.output_dir.resolve())


if __name__ == "__main__":
    main()
