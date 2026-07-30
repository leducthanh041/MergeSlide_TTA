#!/usr/bin/env python3
"""Summarize prefix-merge TTA n_steps ablation.

Reads summary.json files produced by scripts/test_classIL_tta_prefix_other_metrics.sh
and prints mACC/FGT/BWT using the same prefix protocol as
test_classIL_tta_prefix_other_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def pct(mean: float, std: float) -> str:
    return f"{mean * 100:.4f}% ({std * 100:.4f}%)"


def parse_run_dir(path: Path) -> tuple[str, str, int] | None:
    # Expected:
    #   <root>/ind_forward/tcp_n2/tcp/outputs/summary.json
    #   <root>/ood_forward/naive_n8/naive/outputs/summary.json
    try:
        setting_dir = path.parents[3].name
        run_dir = path.parents[2].name
    except IndexError:
        return None

    setting = setting_dir.replace("_forward", "")
    match = re.fullmatch(r"(tcp|naive)_n(\d+)", run_dir)
    if not match:
        return None
    mode = match.group(1)
    n_steps = int(match.group(2))
    return setting, mode, n_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Class-IL prefix TTA n_steps ablation"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/ablation_nsteps_prefix"),
        help="Root directory containing <setting>_forward/<mode>_n*/<mode>/outputs/summary.json",
    )
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.glob("*_forward/*_n*/*/outputs/summary.json")):
        parsed = parse_run_dir(summary_path)
        if parsed is None:
            continue
        setting, mode, n_steps = parsed
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "setting": setting.upper(),
            "mode": mode,
            "n_steps": n_steps,
            "macc_acc": pct(data["macc_acc_mean"], data["macc_acc_std"]),
            "fgt_acc": pct(data["fgt_acc_mean"], data["fgt_acc_std"]),
            "bwt_acc": pct(data["bwt_acc_mean"], data["bwt_acc_std"]),
            "macc_bacc": pct(
                data["macc_bacc_task_mean_mean"],
                data["macc_bacc_task_mean_std"],
            ),
            "summary": str(summary_path),
        })

    rows.sort(key=lambda r: (r["setting"], r["mode"], r["n_steps"]))
    if not rows:
        print(f"[WARN] no summary.json files found under {args.root}")
        return

    headers = ["setting", "mode", "n_steps", "macc_acc", "fgt_acc", "bwt_acc", "macc_bacc"]
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
