#!/usr/bin/env python3
"""
tools/analyze_module_f_reliability.py
======================================
Phân tích phân phối H_bar (bất định nội tại) và JSD_K (bất định do bất
đồng giữa K view) từ file tta_stats_csv do test_tta_core.py sinh ra.

Trực tiếp trả lời các ablation đã lên kế hoạch trong tài liệu nghiên cứu:
    AC-8  : phân phối H_bar/JSD_K của CESC so với các task khác
    AC-9  : hiệu quả active_rate / accuracy khi bật/tắt Module F
    AC-10 : (quan trọng nhất) H_bar/JSD_K cho slide ĐÚNG vs SAI, tách theo
            task — kiểm chứng bảng bias/variance trong tài liệu nghiên cứu
            (§10.6): nếu slide CESC sai có JSD_K THẤP (đồng thuận cao) mà
            vẫn sai -> bias hệ thống, Module F không cứu được.
            Nếu H_bar cũng cao -> có tín hiệu uncertainty, S_conf có cơ hội.
    AC-12 : so sánh ngưỡng percentile động (đã dùng trong engine) — tool
            này in phân phối percentile thực tế theo từng task để đối chiếu.

Input : CSV từ --tta_stats_csv của test_tta_core.py (mỗi dòng = 1 slide,
        có cột: task_name, global_correct, h_bar, jsd_k, sample_active, ...)
Output: các bảng thống kê in ra terminal + 1 CSV tổng hợp theo task và
        1 markdown report ngắn.

Usage::
    python tools/analyze_module_f_reliability.py \\
        --tta_stats_csv logs/tta_core_stats.csv \\
        --out_dir logs/module_f_analysis
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("h_bar", "jsd_k", "global_correct", "sample_active"):
        if col not in df.columns:
            raise ValueError(
                f"Cot '{col}' khong ton tai trong {path}. "
                "File nay phai la tta_stats_csv sinh boi test_tta_core.py."
            )
    df["global_correct"] = df["global_correct"].astype(int)
    df["sample_active"]  = df["sample_active"].astype(bool)
    return df


def summarize_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """AC-8: phân phối H_bar/JSD_K theo task."""
    rows = []
    for task, g in df.groupby("task_name"):
        rows.append({
            "task_name":      task,
            "n_slides":       len(g),
            "h_bar_mean":     g["h_bar"].mean(),
            "h_bar_median":   g["h_bar"].median(),
            "h_bar_p90":      g["h_bar"].quantile(0.90),
            "jsd_k_mean":     g["jsd_k"].mean(),
            "jsd_k_median":   g["jsd_k"].median(),
            "jsd_k_p90":      g["jsd_k"].quantile(0.90),
            "acc_naive":      g["global_correct"].mean(),
            "active_rate":    g["sample_active"].mean(),
        })
    return pd.DataFrame(rows).sort_values("task_name")


def _mannwhitney_effect(wrong: np.ndarray, right: np.ndarray) -> dict:
    """
    Kiem dinh Mann-Whitney U (H1: gia tri o slide SAI cao hon slide DUNG)
    + rank-biserial correlation lam effect size (khong nhay voi outlier/
    phan phoi lech nhu Cohen's d, phu hop cho H_bar/JSD_K von rat lech phai).

    SUA LOI so voi phien ban truoc: ban truoc chi so sanh MEAN(sai) vs
    MEAN(dung) -- da chung minh cho ket luan SAI tren du lieu that (CESC:
    mean(sai)=0.0107 > mean(dung)=0.0073 nhin nhu "variance", nhung
    Mann-Whitney p=0.97 -- khong co y nghia thong ke, hai phan phoi thuc
    chat khong the phan biet duoc). Ham nay thay the hoan toan logic cu.

    QUY UOC DAU (da kiem chung bang du lieu gia lap co kiem soat truoc khi
    sua): voi scipy.stats.mannwhitneyu(wrong, right, alternative='greater'),
    khi 'wrong' THUC SU lon hon 'right' ro ret, U tra ve LON (gan n1*n2),
    nen cong thuc chuan `1 - 2U/(n1*n2)` cho gia tri AM. Do do o day dao dau
    (`2U/(n1*n2) - 1`) de effect DUONG nghia la "sai co xu huong cao hon
    dung" -- dung truc giac, tranh lap lai loi dau da phat hien khi chay
    thu tren du lieu that (ban dau moi task deu bi gan nham "BIAS THUAN").
    """
    from scipy import stats as scipy_stats
    n1, n2 = len(wrong), len(right)
    if n1 < 5 or n2 < 5:
        return {"p_value": float("nan"), "rank_biserial": float("nan"), "n_wrong": n1, "n_right": n2}
    try:
        u_stat, p_value = scipy_stats.mannwhitneyu(wrong, right, alternative="greater")
        rank_biserial = (2 * u_stat) / (n1 * n2) - 1
    except ValueError:
        p_value, rank_biserial = float("nan"), float("nan")
    return {"p_value": float(p_value), "rank_biserial": float(rank_biserial), "n_wrong": n1, "n_right": n2}


def summarize_correct_vs_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    """
    AC-10 (uu tien cao nhat): H_bar/JSD_K cho slide DUNG vs SAI, theo task,
    KEM kiem dinh Mann-Whitney U + effect size cho ca hai bien (khong chi
    mean/std tho).
    """
    rows = []
    for task, g in df.groupby("task_name"):
        wrong = g[g["global_correct"] == 0]
        right = g[g["global_correct"] == 1]
        h_test   = _mannwhitney_effect(wrong["h_bar"].to_numpy(),  right["h_bar"].to_numpy())
        jsd_test = _mannwhitney_effect(wrong["jsd_k"].to_numpy(), right["jsd_k"].to_numpy())
        rows.append({
            "task_name":         task,
            "n_wrong":           len(wrong),
            "n_right":           len(right),
            "h_bar_wrong_mean":  wrong["h_bar"].mean(),
            "h_bar_right_mean":  right["h_bar"].mean(),
            "h_bar_p":           h_test["p_value"],
            "h_bar_effect":      h_test["rank_biserial"],
            "jsd_k_wrong_mean":  wrong["jsd_k"].mean(),
            "jsd_k_right_mean":  right["jsd_k"].mean(),
            "jsd_k_p":           jsd_test["p_value"],
            "jsd_k_effect":      jsd_test["rank_biserial"],
            "active_rate_wrong": wrong["sample_active"].mean(),
            "active_rate_right": right["sample_active"].mean(),
        })
    return pd.DataFrame(rows).sort_values("task_name")


def diagnose_bias_vs_variance(summary: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    """
    Sinh ket luan tu dong theo dung bang bias/variance o Sec 10.6 tai lieu
    nghien cuu, nhung nay dua tren KIEM DINH THONG KE (Mann-Whitney p +
    rank-biserial effect size), khong con dua tren so sanh mean tho.

    Quy tac (rank-biserial > 0 nghia la gia tri o nhom SAI co xu huong cao
    hon nhom DUNG -- do alternative='greater' da dat dung trong _mannwhitney_effect):
        - JSD_K khong co y nghia (p >= alpha)  VA  H_bar khong co y nghia
            -> BIAS THUAN: Module F (ca S_conf lan S_agree) KHONG co co so
               thong ke de loc loi nay.
        - JSD_K khong co y nghia  NHUNG  H_bar co y nghia (p < alpha, effect > 0)
            -> BIAS + UNCERTAINTY: chi S_conf co co so thong ke de loc,
               S_agree khong dong gop gi cho task nay (co the can tat rieng
               qua agree_percentile=1.0 de tranh loc oan slide tot).
        - JSD_K co y nghia (p < alpha, effect > 0)
            -> VARIANCE: ca S_agree lan S_conf co co so thong ke.
    """
    lines = []
    for _, row in summary.iterrows():
        task = row["task_name"]
        jsd_sig = (row["jsd_k_p"] < alpha) and (row["jsd_k_effect"] > 0)
        h_sig   = (row["h_bar_p"] < alpha) and (row["h_bar_effect"] > 0)

        if not jsd_sig and not h_sig:
            verdict = (
                "BIAS THUAN (kiem dinh xac nhan): ca JSD_K lan H_bar deu KHONG phan biet duoc "
                "sai/dung co y nghia thong ke. Module F khong co co so de loc loi nay -- "
                "dung nhu du doan R-C1/Sec 10.6."
            )
        elif not jsd_sig and h_sig:
            verdict = (
                f"BIAS + UNCERTAINTY (kiem dinh xac nhan): H_bar CO y nghia thong ke "
                f"(p={row['h_bar_p']:.4f}, effect={row['h_bar_effect']:+.3f}) nhung JSD_K KHONG "
                f"(p={row['jsd_k_p']:.4f}, effect={row['jsd_k_effect']:+.3f}). Chi S_conf co tac dung; "
                "S_agree co the dang loc ngau nhien (khong co tin hieu that) cho task nay."
            )
        elif jsd_sig:
            verdict = (
                f"VARIANCE (kiem dinh xac nhan): JSD_K co y nghia thong ke "
                f"(p={row['jsd_k_p']:.4f}, effect={row['jsd_k_effect']:+.3f}). Ca S_agree lan S_conf "
                "deu co co so thong ke de loc loi nay."
            )
        else:
            verdict = "Khong ro rang -- can them du lieu."

        lines.append(
            f"- {task} (n_sai={int(row['n_wrong'])}, n_dung={int(row['n_right'])}): "
            f"H_bar p={row['h_bar_p']:.4f} effect={row['h_bar_effect']:+.3f} | "
            f"JSD_K p={row['jsd_k_p']:.4f} effect={row['jsd_k_effect']:+.3f}\n  => {verdict}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta_stats_csv", type=Path, required=True)
    parser.add_argument("--out_dir",       type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_stats(args.tta_stats_csv)

    by_task = summarize_by_task(df)
    by_task_path = args.out_dir / "ac8_by_task.csv"
    by_task.to_csv(by_task_path, index=False)
    print("\n===== AC-8: H_bar / JSD_K theo task =====")
    print(by_task.to_string(index=False))

    corr_vs_incorr = summarize_correct_vs_incorrect(df)
    corr_path = args.out_dir / "ac10_correct_vs_incorrect.csv"
    corr_vs_incorr.to_csv(corr_path, index=False)
    print("\n===== AC-10: DUNG vs SAI theo task =====")
    print(corr_vs_incorr.to_string(index=False))

    verdict_lines = diagnose_bias_vs_variance(corr_vs_incorr, alpha=0.05)

    overall_active_rate = df["sample_active"].mean()
    overall_acc_active   = df.loc[df["sample_active"], "global_correct"].mean() if df["sample_active"].any() else float("nan")
    overall_acc_inactive = df.loc[~df["sample_active"], "global_correct"].mean() if (~df["sample_active"]).any() else float("nan")

    report_path = args.out_dir / "module_f_reliability_report.md"
    with open(report_path, "w") as f:
        f.write("# Module F Reliability Report\n\n")
        f.write(f"Nguon: `{args.tta_stats_csv}`\n\n")
        f.write(f"- Tong so slide: {len(df)}\n")
        f.write(f"- Ty le active (duoc adapt): {overall_active_rate*100:.2f}%\n")
        f.write(f"- Accuracy tren slide active:   {overall_acc_active*100:.2f}%\n")
        f.write(f"- Accuracy tren slide bi gate:  {overall_acc_inactive*100:.2f}%\n\n")
        f.write("## AC-8: H_bar / JSD_K theo task\n\n")
        f.write(by_task.to_markdown(index=False))
        f.write("\n\n## AC-10: Dung vs Sai theo task (kem kiem dinh Mann-Whitney U)\n\n")
        f.write(corr_vs_incorr.to_markdown(index=False))
        f.write(
            "\n\n`*_p` la p-value cua kiem dinh Mann-Whitney U mot phia (H1: gia tri o "
            "nhom SAI cao hon nhom DUNG). `*_effect` la rank-biserial correlation "
            "(khoang [-1,1], duong nghia la nhom SAI co xu huong cao hon).\n"
        )
        f.write("\n## Chan doan tu dong (bias vs variance, xem Sec 10.6 tai lieu nghien cuu)\n\n")
        f.write(
            "Chan doan duoi day dua tren kiem dinh Mann-Whitney U + rank-biserial "
            "effect size (khong con dua tren so sanh trung binh don thuan -- phien ban "
            "truoc da cho ket luan sai cho truong hop JSD_K co phan phoi lech manh).\n\n"
        )
        for line in verdict_lines:
            f.write(f"{line}\n")
    print(f"\n[INFO] Report -> {report_path}")
    print(f"[INFO] CSV     -> {by_task_path}, {corr_path}")

    print("\n===== Chan doan tu dong =====")
    for line in verdict_lines:
        print(line)


if __name__ == "__main__":
    main()
