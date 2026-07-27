#!/usr/bin/env python3
"""
tools/compare_naive_vs_core.py
================================
So sánh trực tiếp hai kết quả:
    - classil_naive  (test_tta_v3.py --mode classil_naive — engine v3 gốc,
                       use_tcp_gate=False, KHÔNG Module F)
    - core_naive      (test_tta_core.py — engine mới, KHÔNG TCP/L_task,
                       CÓ Module F)

chạy trên CÙNG checkpoint đã merge / CÙNG fold, để trả lời AC-2/AC-9:
"Module F (sample selection) có thực sự nâng naive bACC không?"

Input: hai --result_csv sinh bởi hai script tương ứng (mỗi dòng = 1 task x
1 fold, có cột fold, task_id, task_name, bacc, acc).

Phương pháp thống kê: vì cùng checkpoint/fold được đánh giá bởi cả hai
phương pháp, đây là dữ liệu GHÉP ĐÔI (paired) theo (fold, task_name) — dùng
paired t-test và Wilcoxon signed-rank theo đúng quy ước thống kê đã thống
nhất cho project (paired t-test, Wilcoxon, Cohen's d_z qua các fold).

LƯU Ý QUAN TRỌNG VỀ DIỄN GIẢI:
    - classil_naive (v3) và core_naive đều đánh giá trên KHÔNG GIAN FLAT
      (toàn bộ 13 lớp), nên bACC hai bên có thể so sánh trực tiếp.
    - Nếu n_folds < 5, kết quả kiểm định (đặc biệt p-value) chỉ mang tính
      tham khảo — không đủ mạnh để kết luận "có ý nghĩa thống kê".

Usage::
    python tools/compare_naive_vs_core.py \\
        --naive_csv logs/tta_v3_results_classil_naive.csv \\
        --core_csv  logs/tta_core_results_ind.csv \\
        --out_dir   logs/compare_naive_vs_core_ind
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


REQUIRED_COLS = ["fold", "task_id", "task_name", "bacc", "acc"]


def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} ({label}) thieu cot: {missing}")
    df = df[REQUIRED_COLS].copy()
    df.columns = ["fold", "task_id", "task_name", f"bacc_{label}", f"acc_{label}"]
    return df


def _cohens_d_z(diff: np.ndarray) -> float:
    """Cohen's d_z cho paired sample: mean(diff) / std(diff, ddof=1)."""
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd)


def _paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: mang gia tri ghep doi (vd bacc_core, bacc_naive) cung do dai."""
    diff = a - b
    out = {
        "n_pairs":       len(diff),
        "mean_diff":     float(diff.mean()),
        "std_diff":      float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "cohens_d_z":    _cohens_d_z(diff) if len(diff) > 1 else float("nan"),
    }
    if _HAVE_SCIPY and len(diff) > 1:
        try:
            t_stat, p_ttest = scipy_stats.ttest_rel(a, b)
            out["ttest_t"] = float(t_stat)
            out["ttest_p"] = float(p_ttest)
        except Exception:
            out["ttest_t"] = float("nan")
            out["ttest_p"] = float("nan")
        try:
            # Wilcoxon yeu cau it nhat 1 cap khac 0
            if np.any(diff != 0):
                w_stat, p_wil = scipy_stats.wilcoxon(a, b)
                out["wilcoxon_stat"] = float(w_stat)
                out["wilcoxon_p"] = float(p_wil)
            else:
                out["wilcoxon_stat"] = float("nan")
                out["wilcoxon_p"] = float("nan")
        except Exception:
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_p"] = float("nan")
    else:
        out["ttest_t"] = out["ttest_p"] = float("nan")
        out["wilcoxon_stat"] = out["wilcoxon_p"] = float("nan")
    return out


def compare_per_task(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, g in merged.groupby("task_name"):
        g = g.sort_values("fold")
        a = g["bacc_core"].to_numpy(dtype=float)
        b = g["bacc_naive"].to_numpy(dtype=float)
        stat = _paired_tests(a, b)
        rows.append({
            "task_name": task,
            "n_folds": len(g),
            "bacc_naive_mean": b.mean(), "bacc_naive_std": b.std(ddof=1) if len(b) > 1 else 0.0,
            "bacc_core_mean":  a.mean(), "bacc_core_std":  a.std(ddof=1) if len(a) > 1 else 0.0,
            **stat,
        })
    return pd.DataFrame(rows).sort_values("task_name")


def compare_overall(merged: pd.DataFrame) -> dict:
    """
    Gop theo fold (trung binh cac task trong tung fold) truoc khi ghep doi
    -- moi fold la 1 don vi quan sat doc lap, giong quy uoc 'across 10 folds'
    da thong nhat cho project.
    """
    per_fold = merged.groupby("fold")[["bacc_naive", "bacc_core"]].mean().reset_index()
    a = per_fold["bacc_core"].to_numpy(dtype=float)
    b = per_fold["bacc_naive"].to_numpy(dtype=float)
    stat = _paired_tests(a, b)
    return {
        "n_folds": len(per_fold),
        "bacc_naive_mean": float(b.mean()), "bacc_naive_std": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "bacc_core_mean":  float(a.mean()), "bacc_core_std":  float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        **stat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive_csv", type=Path, required=True,
                         help="result_csv tu test_tta_v3.py --mode classil_naive")
    parser.add_argument("--core_csv",  type=Path, required=True,
                         help="result_csv tu test_tta_core.py")
    parser.add_argument("--out_dir",   type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    naive_df = _load(args.naive_csv, "naive")
    core_df  = _load(args.core_csv,  "core")

    merged = pd.merge(naive_df, core_df, on=["fold", "task_id", "task_name"], how="inner")
    if merged.empty:
        raise RuntimeError(
            "Khong co (fold, task_id) nao khop giua hai file. Kiem tra hai CSV co "
            "cung tap fold/task khong (vd cung --fold_start/--fold_end)."
        )
    dropped_naive = len(naive_df) - len(merged)
    dropped_core  = len(core_df)  - len(merged)
    if dropped_naive or dropped_core:
        print(f"[WARN] {dropped_naive} dong tu naive_csv va {dropped_core} dong tu core_csv "
              f"khong co cap tuong ung, da bi bo qua khoi so sanh ghep doi.")

    merged.to_csv(args.out_dir / "merged_raw.csv", index=False)

    by_task = compare_per_task(merged)
    by_task.to_csv(args.out_dir / "compare_by_task.csv", index=False)

    overall = compare_overall(merged)

    print("\n===== So sanh theo task (paired qua cac fold) =====")
    print(by_task.to_string(index=False))
    print("\n===== Tong the (trung binh cac task moi fold, paired qua fold) =====")
    for k, v in overall.items():
        print(f"  {k:16s}: {v}")

    report_path = args.out_dir / "compare_naive_vs_core_report.md"
    with open(report_path, "w") as f:
        f.write("# So sanh classil_naive (v3 goc) vs core_naive (Module F)\n\n")
        f.write(f"- naive_csv: `{args.naive_csv}`\n- core_csv: `{args.core_csv}`\n\n")
        f.write("## Tong the\n\n")
        f.write(f"- So fold ghep doi duoc: {overall['n_folds']}\n")
        f.write(f"- bACC naive (v3, khong Module F): {overall['bacc_naive_mean']*100:.4f}% "
                f"(+/-{overall['bacc_naive_std']*100:.4f}%)\n")
        f.write(f"- bACC core  (co Module F):        {overall['bacc_core_mean']*100:.4f}% "
                f"(+/-{overall['bacc_core_std']*100:.4f}%)\n")
        f.write(f"- Chenh lech trung binh (core - naive): {overall['mean_diff']*100:+.4f} pp\n")
        f.write(f"- Cohen's d_z: {overall['cohens_d_z']:.4f}\n")
        if _HAVE_SCIPY:
            f.write(f"- Paired t-test: t={overall['ttest_t']:.4f}, p={overall['ttest_p']:.4f}\n")
            f.write(f"- Wilcoxon signed-rank: stat={overall['wilcoxon_stat']:.4f}, p={overall['wilcoxon_p']:.4f}\n")
        else:
            f.write("- (scipy khong co san trong moi truong chay script nay -- "
                    "khong tinh duoc p-value; cai `pip install scipy` de bo sung.)\n")
        f.write(
            f"\n**Luu y:** n={overall['n_folds']} fold. "
            "Neu n < 5, khong nen dua ra ket luan 'co y nghia thong ke' du p < 0.05, "
            "chi nen dung nhu tin hieu tham khao (theo dung nguyen tac scientific rigor "
            "da thong nhat cho project).\n\n"
        )
        f.write("## Theo tung task\n\n")
        f.write(by_task.to_markdown(index=False))
        f.write("\n\n## Diem chu y CESC/ESCA\n\n")
        focus = by_task[by_task["task_name"].isin(["CESC", "ESCA"])]
        if focus.empty:
            f.write("Khong tim thay task CESC/ESCA trong du lieu (kiem tra ten task trong CSV).\n")
        else:
            for _, row in focus.iterrows():
                f.write(
                    f"- **{row['task_name']}**: naive={row['bacc_naive_mean']*100:.2f}% -> "
                    f"core={row['bacc_core_mean']*100:.2f}% "
                    f"(chenh lech {row['mean_diff']*100:+.2f} pp, "
                    f"d_z={row['cohens_d_z']:.3f})\n"
                )
            f.write(
                "\nDoi chieu voi ket qua AC-10 (tools/analyze_module_f_reliability.py) "
                "de biet chenh lech nay (neu co) den tu loc duoc bias hay chi tu loc "
                "duoc variance -- xem Sec 10.6 tai lieu nghien cuu truoc khi ket luan.\n"
            )

    print(f"\n[INFO] Report -> {report_path}")
    print(f"[INFO] CSV     -> {args.out_dir / 'compare_by_task.csv'}")


if __name__ == "__main__":
    main()
