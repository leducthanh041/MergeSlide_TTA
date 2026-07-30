#!/usr/bin/env python3
"""Summarize Class-IL bACC for entropy-threshold ablation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def pct(mean: float, std: float) -> str:
    return f"{mean * 100:.4f}% ({std * 100:.4f}%)"


def parse_run_dir(path: Path) -> tuple[str, str, float] | None:
    # Expected:
    #   <root>/ind_forward/tcp_e0p2/results.csv
    #   <root>/ood_forward/naive_e0p6/results.csv
    try:
        setting_dir = path.parents[1].name
        run_dir = path.parent.name
    except IndexError:
        return None

    setting = setting_dir.replace("_forward", "")
    match = re.fullmatch(r"(tcp|naive)_e([0-9]+p[0-9]+)", run_dir)
    if not match:
        return None
    mode = match.group(1)
    threshold = float(match.group(2).replace("p", "."))
    return setting, mode, threshold


def parse_taskil_log(path: Path) -> tuple[str, str, float, str] | None:
    # Expected:
    #   <root>/ind_forward/taskil_e0p2/result_taskil_tta.log
    try:
        setting_dir = path.parents[1].name
        run_dir = path.parent.name
    except IndexError:
        return None

    match = re.fullmatch(r"taskil_e([0-9]+p[0-9]+)", run_dir)
    if not match:
        return None
    setting = setting_dir.replace("_forward", "")
    threshold = float(match.group(1).replace("p", "."))
    text = path.read_text(encoding="utf-8", errors="replace")
    metric = re.search(r"Balanced Acc:\s+([0-9.]+)%\s+\(([0-9.]+)%\)", text)
    if metric is None:
        return None
    return setting, "task_il", threshold, f"{float(metric.group(1)):.4f}% ({float(metric.group(2)):.4f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Class-IL bACC for entropy-threshold ablation"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/ablation_entropy_threshold"),
    )
    args = parser.parse_args()

    rows = []
    for result_csv in sorted(args.root.glob("*_forward/*_e*/results.csv")):
        parsed = parse_run_dir(result_csv)
        if parsed is None:
            continue
        setting, mode, threshold = parsed
        df = pd.read_csv(result_csv).dropna(subset=["fold", "task_id", "bacc"])
        if df.empty:
            continue
        per_fold = df.groupby("fold")["bacc"].mean().sort_index()
        rows.append({
            "setting": setting.upper(),
            "mode": mode,
            "entropy_threshold": f"{threshold:.1f}",
            "n_folds": len(per_fold),
            "classil_bacc": pct(per_fold.mean(), per_fold.std(ddof=1)),
            "path": str(result_csv),
        })

    for taskil_log in sorted(args.root.glob("*_forward/taskil_e*/result_taskil_tta.log")):
        parsed = parse_taskil_log(taskil_log)
        if parsed is None:
            continue
        setting, mode, threshold, formatted = parsed
        rows.append({
            "setting": setting.upper(),
            "mode": mode,
            "entropy_threshold": f"{threshold:.1f}",
            "n_folds": "10",
            "classil_bacc": formatted,
            "path": str(taskil_log),
        })

    rows.sort(key=lambda row: (row["setting"], row["mode"], float(row["entropy_threshold"])))
    if not rows:
        print(f"[WARN] no results.csv files found under {args.root}")
        return

    headers = ["setting", "mode", "entropy_threshold", "n_folds", "classil_bacc"]
    widths = {
        key: max(len(key), *(len(str(row[key])) for row in rows))
        for key in headers
    }
    print("  ".join(key.ljust(widths[key]) for key in headers))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in headers))


if __name__ == "__main__":
    main()
