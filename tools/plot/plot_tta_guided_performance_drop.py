#!/usr/bin/env python3
"""Plot prefix performance drop for merging methods and CAST-Slide.

Inputs:
- TTA-guided merging baselines use prefix continual matrix CSV files produced by
  each method's `run_prefix_continual_metrics.sh`.
- CAST-Slide uses TCP prefix matrices produced by the other-metrics runner.

Outputs are saved next to this script for IND and OOD.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


WSI_ROOT = Path("/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI")
REPO_ROOT = WSI_ROOT / "MergeSlide_TTA"
DEFAULT_OUT_DIR = REPO_ROOT / "logs" / "ablation" / "performance-drop"

TASK_LABELS = {
    "ind": {
        0: "TCGA-BRCA (first task)",
        1: "TCGA-RCC (second task)",
    },
    "ood": {
        0: "TCGA-BRCA (first task)",
        1: "TCGA-RCC (second task)",
    },
}

Y_LIMITS = {
    "ind": {
        0: (0.85, 1.01),
        1: (0.90, 1.01),
    },
    "ood": {
        0: (0.80, 0.93),
        1: (0.87, 0.93),
    },
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path_template: str
    color: str
    marker: str
    linewidth: float = 2.0
    markersize: float = 5.0
    alpha: float = 0.16
    zorder: int = 3
    edgecolor: str | None = None
    edgewidth: float = 0.8

    def path(self, setting: str, mode: str = "tcp") -> Path:
        return Path(self.path_template.format(setting=setting, mode=mode))


METHODS: list[MethodSpec] = [
    MethodSpec(
        "CAST-Slide (ours)",
        str(REPO_ROOT / "logs/tta/{setting}/classil_other_metrics/{mode}/outputs"),
        "#FFD23F",
        "*",
        linewidth=3.1,
        markersize=13.0,
        alpha=0.20,
        zorder=10,
        edgecolor="#D35400",
        edgewidth=1.6,
    ),
    MethodSpec(
        "LayerWise AdaMerging++",
        str(WSI_ROOT / "AdaMerging/logs/adamerging_prefix/{setting}/metrics/adamerging_{setting}_class_il_continual_matrix.csv"),
        "#D8C7E8",
        "o",
        edgecolor="#5B3A70",
    ),
    MethodSpec(
        "AdaRank",
        str(WSI_ROOT / "AdaRank/logs/adarank_prefix/{setting}/metrics/adarank_{setting}_class_il_continual_matrix.csv"),
        "#D28E2C",
        "s",
        edgecolor="#704500",
    ),
    MethodSpec(
        "Hi-Vec",
        str(WSI_ROOT / "Hi-Vec/logs/hivec_prefix/{setting}/metrics/hivec_{setting}_class_il_continual_matrix.csv"),
        "#3A7D44",
        "P",
        edgecolor="#173C1D",
    ),
    MethodSpec(
        "MINGLE",
        str(WSI_ROOT / "MINGLE/logs/mingle_prefix/{setting}/metrics/mingle_{setting}_class_il_continual_matrix.csv"),
        "#B8325A",
        "X",
        edgecolor="#5E102A",
    ),
    MethodSpec(
        "WEMOE (2 layers)",
        str(WSI_ROOT / "weight-ensembling_MoE/logs/wemoe_prefix/{setting}/metrics/wemoe_{setting}_class_il_continual_matrix.csv"),
        "#547AA5",
        ">",
        edgecolor="#263F5B",
    ),
    MethodSpec(
        "T3",
        str(WSI_ROOT / "TCube/logs/t3_prefix/{setting}/metrics/tcube_{setting}_class_il_continual_matrix.csv"),
        "#85754E",
        "<",
        edgecolor="#453A23",
    ),
    MethodSpec(
        "CONCRETE",
        str(WSI_ROOT / "subspace_fusion/logs/concrete_prefix/{setting}/metrics/concrete_{setting}_class_il_continual_matrix.csv"),
        "#7A7F87",
        "h",
        edgecolor="#363A40",
    ),
]


def require_columns(path: Path, df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def load_prefix_matrix(spec: MethodSpec, setting: str, metric: str) -> pd.DataFrame:
    path = spec.path(setting)
    if not path.exists():
        print(f"[WARN] missing {spec.name} for {setting}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    require_columns(path, df, {"fold", "seq_task", "eval_task", metric})
    out = df[["fold", "seq_task", "eval_task", metric]].copy()
    out["method"] = spec.name
    out["fold"] = out["fold"].astype(int)
    out["seq_task"] = out["seq_task"].astype(int)
    out["eval_task"] = out["eval_task"].astype(int)
    out[metric] = out[metric].astype(float)
    out["num_tasks"] = out["seq_task"] + 1
    return out


def load_cast_slide(spec: MethodSpec, setting: str, metric: str, mode: str) -> pd.DataFrame:
    root = spec.path(setting, mode)
    if not root.exists():
        print(f"[WARN] missing CAST-Slide TCP matrix root for {setting}: {root}")
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for matrix_path in sorted(root.glob(f"fold_*/{metric}_matrix.csv")):
        fold_name = matrix_path.parent.name
        try:
            fold = int(fold_name.split("_")[-1])
        except ValueError:
            print(f"[WARN] skip unexpected fold directory: {matrix_path.parent}")
            continue

        df = pd.read_csv(matrix_path)
        if df.empty:
            continue
        task_columns = [col for col in df.columns if col != "state_after_task"]
        for seq_task, row in df.iterrows():
            for eval_task, col in enumerate(task_columns[: seq_task + 1]):
                value = row[col]
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "method": spec.name,
                        "fold": fold,
                        "seq_task": int(seq_task),
                        "eval_task": int(eval_task),
                        "num_tasks": int(seq_task) + 1,
                        metric: float(value),
                    }
                )

    if not rows:
        print(f"[WARN] no CAST-Slide rows found for {setting}: {root}")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_method(spec: MethodSpec, setting: str, metric: str, mode: str) -> pd.DataFrame:
    if spec.name == "CAST-Slide (ours)":
        return load_cast_slide(spec, setting, metric, mode)
    return load_prefix_matrix(spec, setting, metric)


def summarize(df: pd.DataFrame, eval_task: int, metric: str) -> pd.DataFrame:
    target = df[(df["eval_task"] == eval_task) & (df["seq_task"] >= eval_task)]
    if target.empty:
        return pd.DataFrame(columns=["method", "num_tasks", "mean", "std", "n"])
    summary = (
        target.groupby(["method", "num_tasks"], as_index=False)[metric]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["method", "num_tasks"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def validate_coverage(df: pd.DataFrame, setting: str) -> None:
    for method in [spec.name for spec in METHODS]:
        part = df[df["method"] == method]
        if part.empty:
            continue
        for eval_task in TASK_LABELS[setting]:
            expected = set(range(eval_task + 1, 7))
            observed = set(part.loc[part["eval_task"] == eval_task, "num_tasks"].unique())
            missing = sorted(expected.difference(observed))
            if missing:
                print(
                    f"[WARN] {setting} {method}: missing eval_task={eval_task} "
                    f"at num_tasks={missing}; plotting available rows only."
                )


def plot_panel(ax, summary: pd.DataFrame, setting: str, eval_task: int) -> None:
    for spec in METHODS:
        part = summary[summary["method"] == spec.name]
        if part.empty:
            continue

        x = part["num_tasks"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        std = part["std"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=spec.marker,
            markersize=spec.markersize,
            linewidth=spec.linewidth,
            color=spec.color,
            label=spec.name,
            zorder=spec.zorder,
            markeredgecolor=spec.edgecolor or spec.color,
            markeredgewidth=spec.edgewidth,
        )
        ax.fill_between(
            x,
            y - std,
            y + std,
            color=spec.color,
            alpha=spec.alpha,
            linewidth=0,
            zorder=max(1, spec.zorder - 1),
        )

    min_x = eval_task + 1
    ax.set_xlim(min_x - 0.15, 6.15)
    ax.set_xticks(range(min_x, 7))
    y_min, y_max = Y_LIMITS.get(setting, {}).get(eval_task, (0.55, 1.02))
    ax.set_ylim(y_min, y_max)
    ticks = [
        tick
        for tick in [0.80, 0.83, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95]
        if y_min <= tick <= y_max
    ]
    ax.set_yticks(ticks)
    ax.set_box_aspect(1)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)
    ax.tick_params(axis="both", labelsize=16.0, width=1.2, length=4.5)


def plot_setting(setting: str, metric: str, mode: str, out_dir: Path) -> None:
    frames = [load_method(spec, setting, metric, mode) for spec in METHODS]
    frames = [df for df in frames if not df.empty]
    if not frames:
        raise RuntimeError(f"No data loaded for setting={setting}")

    df = pd.concat(frames, ignore_index=True)
    validate_coverage(df, setting)

    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.3), sharey=False)
    for ax, eval_task in zip(axes, [0, 1]):
        summary = summarize(df, eval_task, metric)
        summary.insert(0, "setting", setting)
        summary.insert(1, "metric", metric)
        summary.insert(2, "eval_task", eval_task)
        summary.insert(3, "task_name", TASK_LABELS[setting][eval_task])
        summaries.append(summary)
        plot_panel(ax, summary, setting, eval_task)
        ax.set_title(TASK_LABELS[setting][eval_task], fontsize=20)
        ax.set_xlabel("Number of Tasks", fontsize=22, labelpad=9)

    metric_label = "Accuracy" if metric == "acc" else "Balanced Accuracy"
    axes[0].set_ylabel(f"{metric_label} (CLASS-IL)", fontsize=22, labelpad=11)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(
        handles,
        labels,
        loc="lower left",
        fontsize=15.0,
        frameon=True,
        framealpha=0.90,
        ncol=1,
        title="Method",
        title_fontsize=15.5,
        borderpad=0.35,
        labelspacing=0.30,
        handlelength=1.55,
        handletextpad=0.55,
    )
    for text in legend.get_texts():
        if text.get_text() == "CAST-Slide (ours)":
            text.set_weight("bold")

    fig.tight_layout(w_pad=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"cast_slide_performance_drop_{setting}_{mode}_{metric}.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    for suffix in ("png", "pdf"):
        out = out_dir / f"cast_slide_performance_drop_{setting}_{mode}_{metric}.{suffix}"
        fig.savefig(out, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CAST-Slide prefix performance drop.")
    parser.add_argument("--setting", choices=("ind", "ood", "all"), default="all")
    parser.add_argument("--mode", choices=("naive", "tcp"), default="tcp")
    parser.add_argument("--metric", choices=("acc", "bacc"), default="acc")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    settings = ("ind", "ood") if args.setting == "all" else (args.setting,)
    for setting in settings:
        plot_setting(setting, args.metric, args.mode, args.output_dir.resolve())


if __name__ == "__main__":
    main()
