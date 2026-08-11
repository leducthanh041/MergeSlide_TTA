#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import compute_ind_pvalues as common


METRICS = ("bacc",)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fold-wise paired bACC significance tests for CAST-Slide on OOD."
    )
    parser.add_argument("--manifest", type=Path, default=here / "ood_bacc_sources.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def write_report(output_dir, manifest, results, coverage):
    lines = [
        "# OOD bACC fold-wise statistical significance report",
        "",
        f"Reference method: **{manifest['reference_method']}**",
        "",
        "Primary test: one-sided exact paired sign-flip test over 10 matched folds.",
        "Holm correction is applied across all OOD bACC baseline comparisons.",
        "",
        "## Coverage",
        "",
        "| Method | bACC folds | Status |",
        "|---|---:|---|",
    ]
    for row in coverage.to_dict("records"):
        lines.append(f"| {row['method']} | {row['bacc_folds']} | {row['status']} |")
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "| Baseline | CAST-Slide | Baseline | Delta | 95% CI | p raw | p Holm | Sig. |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in results.sort_values("baseline").to_dict("records"):
        lines.append(
            f"| {row['baseline']} | {100 * row['reference_mean']:.3f} | "
            f"{100 * row['baseline_mean']:.3f} | {100 * row['mean_difference']:+.3f} | "
            f"[{100 * row['ci95_low']:+.3f}, {100 * row['ci95_high']:+.3f}] | "
            f"{row['p_signflip_raw']:.6g} | {row['p_holm']:.6g} | {row['stars']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Positive delta means CAST-Slide has higher bACC.",
        "- Missing fold-level results are not reconstructed from aggregate statistics.",
        "- Significance stars use only Holm-adjusted exact sign-flip p-values.",
        "- Verify task-averaged versus pooled bACC definitions before paper submission.",
    ])
    (output_dir / "ood_bacc_statistical_report.md").write_text("\n".join(lines) + "\n")


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
        logging.info("%s: bacc=%d folds", method["name"], frame.bacc.notna().sum())
        for error in errors:
            logging.warning("%s source skipped: %s", method["name"], error)

    normalized = pd.concat(frames, ignore_index=True).sort_values(["method", "fold"])
    normalized.to_csv(args.output_dir / "ood_bacc_fold_metrics.csv", index=False)
    pd.DataFrame(source_records).to_csv(args.output_dir / "source_files.csv", index=False)

    coverage_rows = []
    for method in manifest["methods"]:
        count = int(normalized.loc[normalized.method.eq(method["name"]), "bacc"].notna().sum())
        status = []
        if count != expected_folds:
            status.append(f"bacc: {count}/{expected_folds}")
        if method_errors[method["name"]]:
            status.append("source error")
        coverage_rows.append({
            "method": method["name"],
            "family": method["family"],
            "bacc_folds": count,
            "status": "; ".join(status) or "complete",
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "coverage_report.csv", index=False)

    reference_name = manifest["reference_method"]
    reference = normalized[normalized.method.eq(reference_name)]
    rows = []
    skipped = []
    for method in manifest["methods"]:
        if method["name"] == reference_name:
            continue
        baseline = normalized[normalized.method.eq(method["name"])]
        result, reason = common.compare_metric(
            reference, baseline, "bacc", expected_folds, args.allow_partial,
            args.bootstrap_samples, args.seed,
        )
        if result is None:
            skipped.append({"baseline": method["name"], "metric": "bacc", "reason": reason})
        else:
            result.update({"baseline": method["name"], "family": method["family"]})
            rows.append(result)

    results = pd.DataFrame(rows)
    if not results.empty:
        results["p_holm"] = common.holm_adjust(results.p_signflip_raw.to_numpy())
        results["stars"] = results.p_holm.map(common.significance_stars)
    results.to_csv(args.output_dir / "ood_bacc_pvalues_all.csv", index=False)
    columns = [
        "baseline", "family", "n_folds", "reference_mean", "baseline_mean",
        "mean_difference", "ci95_low", "ci95_high", "cohen_dz", "wins", "ties",
        "losses", "p_signflip_raw", "p_holm", "stars",
    ]
    results.reindex(columns=columns).to_csv(
        args.output_dir / "ood_bacc_significance_table.csv", index=False
    )
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped_comparisons.csv", index=False)
    write_report(args.output_dir, manifest, results, coverage)
    logging.info("Wrote results to %s", args.output_dir.resolve())
    logging.info("Complete comparisons: %d; skipped: %d", len(results), len(skipped))


if __name__ == "__main__":
    main()
