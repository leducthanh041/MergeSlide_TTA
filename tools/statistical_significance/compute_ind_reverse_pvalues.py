#!/usr/bin/env python3
import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import compute_ind_pvalues as common


METRICS = ("bacc", "macro_f1")
HIGHER_IS_BETTER = {"bacc": True, "macro_f1": True}


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fold-wise paired significance tests for CAST-Slide on IND reverse."
    )
    parser.add_argument("--manifest", type=Path, default=here / "ind_reverse_sources.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def compare_metric(reference, baseline, metric, expected_folds, allow_partial, samples, seed):
    paired = reference[["fold", metric]].merge(
        baseline[["fold", metric]], on="fold", suffixes=("_reference", "_baseline")
    ).dropna()
    if not allow_partial and len(paired) != expected_folds:
        return None, f"requires {expected_folds} paired folds, found {len(paired)}"
    if len(paired) < 2:
        return None, f"requires at least 2 paired folds, found {len(paired)}"

    ref = paired[f"{metric}_reference"].to_numpy(float)
    base = paired[f"{metric}_baseline"].to_numpy(float)
    raw_difference = ref - base
    improvement = raw_difference if HIGHER_IS_BETTER[metric] else -raw_difference
    std_improvement = improvement.std(ddof=1)
    dz = (
        improvement.mean() / std_improvement
        if std_improvement > 0
        else math.copysign(math.inf, improvement.mean())
    )
    ci_low, ci_high = common.bootstrap_ci(raw_difference, samples, seed)
    p_wilcoxon = np.nan
    p_ttest = np.nan
    if common.stats is not None:
        try:
            p_wilcoxon = common.stats.wilcoxon(
                improvement, alternative="greater", method="auto"
            ).pvalue
        except ValueError:
            pass
        p_ttest = common.stats.ttest_1samp(
            improvement, popmean=0.0, alternative="greater"
        ).pvalue

    return {
        "metric": metric,
        "direction": "higher" if HIGHER_IS_BETTER[metric] else "lower",
        "n_folds": len(paired),
        "folds": ",".join(map(str, paired.fold.tolist())),
        "reference_mean": ref.mean(),
        "reference_std": ref.std(ddof=1),
        "baseline_mean": base.mean(),
        "baseline_std": base.std(ddof=1),
        "mean_difference": raw_difference.mean(),
        "mean_improvement": improvement.mean(),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": dz,
        "wins": int((improvement > 0).sum()),
        "ties": int((improvement == 0).sum()),
        "losses": int((improvement < 0).sum()),
        "p_signflip_raw": common.exact_sign_flip_greater(improvement),
        "p_wilcoxon_raw": p_wilcoxon,
        "p_paired_t_raw": p_ttest,
    }, None


def write_report(output_dir, manifest, results, coverage):
    lines = [
        "# IND reverse fold-wise statistical significance report",
        "",
        f"Reference method: **{manifest['reference_method']}**",
        "",
        "Primary test: one-sided exact paired sign-flip test over matched folds.",
        "Holm correction is applied separately within each metric family.",
        "Higher is better for both bACC and macroF1.",
        "",
        "## Coverage",
        "",
        "| Method | bACC | macroF1 | Status |",
        "|---|---:|---:|---|",
    ]
    for row in coverage.to_dict("records"):
        lines.append(
            f"| {row['method']} | {row['bacc_folds']} | "
            f"{row['macro_f1_folds']} | {row['status']} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    if results.empty:
        lines.append("No complete paired comparisons were available.")
    else:
        lines.extend([
            "| Metric | Baseline | CAST-Slide | Baseline | Difference | p raw | p Holm | Sig. |",
            "|---|---|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in results.sort_values(["metric", "baseline"]).to_dict("records"):
            lines.append(
                f"| {row['metric']} | {row['baseline']} | "
                f"{100 * row['reference_mean']:.3f} | {100 * row['baseline_mean']:.3f} | "
                f"{100 * row['mean_difference']:+.3f} | {row['p_signflip_raw']:.6g} | "
                f"{row['p_holm']:.6g} | {row['stars']} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Difference is CAST-Slide minus baseline in the metric's original scale.",
        "- Missing fold-level metrics are not reconstructed from aggregate mean and standard deviation.",
        "- A star is assigned only from the Holm-adjusted exact sign-flip p-value.",
        "- bACC/macroF1 definitions must be checked before combining task-averaged and pooled evaluators.",
    ])
    (output_dir / "ind_reverse_statistical_report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    common.setup_logging(args.output_dir)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    expected_folds = int(manifest.get("expected_folds", 10))
    common.METRICS = METRICS

    frames = []
    source_records = []
    method_errors = {}
    for method in manifest["methods"]:
        frame, paths, errors = common.merge_sources(method, repo_root)
        frames.append(frame)
        method_errors[method["name"]] = errors
        source_records.extend(
            {"method": method["name"], "family": method["family"], "path": path}
            for path in paths
        )
        logging.info(
            "%s: %s",
            method["name"],
            ", ".join(f"{metric}={frame[metric].notna().sum()}" for metric in METRICS),
        )
        for error in errors:
            logging.warning("%s source skipped: %s", method["name"], error)

    normalized = pd.concat(frames, ignore_index=True).sort_values(["method", "fold"])
    normalized.to_csv(args.output_dir / "ind_reverse_fold_metrics.csv", index=False)
    pd.DataFrame(source_records).to_csv(args.output_dir / "source_files.csv", index=False)

    coverage_rows = []
    for method in manifest["methods"]:
        frame = normalized[normalized.method == method["name"]]
        counts = {metric: int(frame[metric].notna().sum()) for metric in METRICS}
        status = [
            f"{metric}: {count}/{expected_folds}"
            for metric, count in counts.items()
            if count != expected_folds
        ]
        if method_errors[method["name"]]:
            status.append("source error")
        coverage_rows.append({
            "method": method["name"],
            "family": method["family"],
            **{f"{metric}_folds": count for metric, count in counts.items()},
            "status": "; ".join(status) or "complete",
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "coverage_report.csv", index=False)

    reference_name = manifest["reference_method"]
    reference = normalized[normalized.method == reference_name]
    result_rows = []
    skipped_rows = []
    for method in manifest["methods"]:
        if method["name"] == reference_name:
            continue
        baseline = normalized[normalized.method == method["name"]]
        for metric_index, metric in enumerate(METRICS):
            result, reason = compare_metric(
                reference, baseline, metric, expected_folds, args.allow_partial,
                args.bootstrap_samples, args.seed + metric_index,
            )
            if result is None:
                skipped_rows.append({"baseline": method["name"], "metric": metric, "reason": reason})
            else:
                result.update({"baseline": method["name"], "family": method["family"]})
                result_rows.append(result)

    results = pd.DataFrame(result_rows)
    if not results.empty:
        parts = []
        for _, part in results.groupby("metric", sort=False):
            part = part.copy()
            part["p_holm"] = common.holm_adjust(part.p_signflip_raw.to_numpy())
            part["stars"] = part.p_holm.map(common.significance_stars)
            parts.append(part)
        results = pd.concat(parts, ignore_index=True)
    results.to_csv(args.output_dir / "ind_reverse_pvalues_all.csv", index=False)
    table_columns = [
        "metric", "direction", "baseline", "family", "n_folds", "reference_mean",
        "baseline_mean", "mean_difference", "mean_improvement", "ci95_low", "ci95_high",
        "cohen_dz", "wins", "ties", "losses", "p_signflip_raw", "p_holm", "stars",
    ]
    results.reindex(columns=table_columns).to_csv(
        args.output_dir / "ind_reverse_significance_table.csv", index=False
    )
    pd.DataFrame(skipped_rows).to_csv(args.output_dir / "skipped_comparisons.csv", index=False)
    write_report(args.output_dir, manifest, results, coverage)
    logging.info("Wrote results to %s", args.output_dir.resolve())
    logging.info("Complete comparisons: %d; skipped: %d", len(results), len(skipped_rows))


if __name__ == "__main__":
    main()
